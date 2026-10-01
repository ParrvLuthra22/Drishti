import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

from drishti.actions.dataset import IMAGENET_MEAN, IMAGENET_STD, RWF2000Dataset

logger = logging.getLogger(__name__)
console = Console()

LOG_EVERY_N_STEPS = 50


def collate_clips(batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack dataset items into a VideoMAE batch.

    RWF2000Dataset already resizes to 224 and applies ImageNet normalisation, so the
    VideoMAEImageProcessor must NOT be applied again (it would rescale by 1/255 and
    normalise a second time). VideoMAE wants (B, T, C, H, W); the dataset yields (C, T, H, W).
    """
    clips, labels = zip(*batch, strict=True)
    pixel_values = torch.stack(clips).permute(0, 2, 1, 3, 4).contiguous()
    return pixel_values, torch.tensor(labels, dtype=torch.long)


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        fallback = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.warning("CUDA requested but not available, falling back to %s", fallback)
        return torch.device(fallback)
    if requested == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but not available, falling back to cpu")
        return torch.device("cpu")
    return torch.device(requested)


class VideoMAETrainer:
    def __init__(
        self,
        data_root: str,
        output_dir: str = "models/videomae_fight",
        model_name: str = "MCG-NJU/videomae-base",
        num_frames: int = 16,
        batch_size: int = 4,
        num_epochs: int = 5,
        lr_head: float = 1e-3,
        lr_backbone: float = 1e-5,
        device: str = "cuda",
        num_workers: int = 4,
    ) -> None:
        self.data_root = data_root
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.lr_head = lr_head
        self.lr_backbone = lr_backbone
        self.device = resolve_device(device)
        self.num_workers = num_workers

        self.label2id = {"NonFight": 0, "Fight": 1}
        self.id2label = {0: "NonFight", 1: "Fight"}
        self.target_names = [self.id2label[i] for i in range(len(self.id2label))]

        # Kept for saving alongside the checkpoint; batches are NOT passed through it (see collate_clips).
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        if not (
            np.allclose(self.processor.image_mean, IMAGENET_MEAN)
            and np.allclose(self.processor.image_std, IMAGENET_STD)
        ):
            logger.warning(
                "Processor mean/std differ from the dataset's ImageNet normalisation: %s / %s",
                self.processor.image_mean,
                self.processor.image_std,
            )

        self.model = VideoMAEForVideoClassification.from_pretrained(
            model_name,
            num_labels=2,
            label2id=self.label2id,
            id2label=self.id2label,
            ignore_mismatched_sizes=True,
        ).to(self.device)

        self.criterion = nn.CrossEntropyLoss()

    # ── data ────────────────────────────────────────────────────────────────

    def _make_loader(self, split: str, shuffle: bool) -> DataLoader:
        dataset = RWF2000Dataset(self.data_root, split=split, num_frames=self.num_frames)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=self._collate_fn,
        )

    def _get_dataloaders(self) -> tuple[DataLoader, DataLoader]:
        return self._make_loader("train", shuffle=True), self._make_loader("val", shuffle=False)

    def _collate_fn(self, batch: list[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        return collate_clips(batch)

    # ── parameter groups ────────────────────────────────────────────────────

    def _head_params(self) -> list[nn.Parameter]:
        # fc_norm is the freshly initialised norm that feeds the classifier, so it trains with the head.
        head = [*self.model.classifier.parameters()]
        if getattr(self.model, "fc_norm", None) is not None:
            head += [*self.model.fc_norm.parameters()]
        return head

    def _set_requires_grad(self, backbone: bool) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        for p in self._head_params():
            p.requires_grad = True
        if backbone:
            for p in self.model.videomae.parameters():
                p.requires_grad = True

    # ── training ────────────────────────────────────────────────────────────

    def train(self) -> dict:
        train_loader, val_loader = self._get_dataloaders()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model.config.save_pretrained(self.output_dir)
        self.processor.save_pretrained(self.output_dir)

        history: list[dict] = []
        best = {"val_f1": -1.0, "epoch": 0}
        started = time.time()

        # Phase A: head only (epoch 1)
        self._set_requires_grad(backbone=False)
        console.print("[bold]Phase A[/bold]: training classifier head only")
        optimizer = AdamW([{"params": self._head_params(), "lr": self.lr_head}])
        self._run_epoch(1, train_loader, val_loader, optimizer, None, history, best, started)

        # Phase B: full fine-tune (epochs 2..N)
        if self.num_epochs > 1:
            self._set_requires_grad(backbone=True)
            console.print("[bold]Phase B[/bold]: fine-tuning backbone + head")
            optimizer = AdamW(
                [
                    {"params": self.model.videomae.parameters(), "lr": self.lr_backbone},
                    {"params": self._head_params(), "lr": self.lr_head},
                ]
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=self.num_epochs - 1)
            for epoch in range(2, self.num_epochs + 1):
                self._run_epoch(epoch, train_loader, val_loader, optimizer, scheduler, history, best, started)

        total = time.time() - started
        console.print(
            f"\n[bold green]Done in {total / 60:.1f} min. "
            f"Best val F1 {best['val_f1']:.4f} at epoch {best['epoch']}.[/bold green]"
        )
        return {"best_val_f1": best["val_f1"], "best_epoch": best["epoch"], "history": history}

    def _run_epoch(
        self,
        epoch: int,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None,
        history: list[dict],
        best: dict,
        started: float,
    ) -> None:
        train = self._train_epoch(train_loader, optimizer, epoch)
        if scheduler is not None:
            scheduler.step()
        val = self._val_epoch(val_loader)

        console.print(
            f"Epoch {epoch}/{self.num_epochs} | "
            f"train loss {train['loss']:.4f} acc {train['acc']:.4f} | "
            f"val loss {val['loss']:.4f} acc {val['acc']:.4f} f1 {val['f1']:.4f}"
        )
        console.print(val["report"])

        improved = val["f1"] > best["val_f1"]
        if improved:
            best.update(val_f1=val["f1"], epoch=epoch)
            torch.save(self.model.state_dict(), self.output_dir / "best_model.pt")
            console.print(f"[green]Saved best_model.pt (val F1 {val['f1']:.4f})[/green]")

        history.append(
            {
                "epoch": epoch,
                "train_loss": train["loss"],
                "train_acc": train["acc"],
                "val_loss": val["loss"],
                "val_acc": val["acc"],
                "val_f1": val["f1"],
                "confusion_matrix": val["confusion_matrix"].tolist(),
                "classification_report": val["report_dict"],
            }
        )
        metrics = {
            "best_val_f1": best["val_f1"],
            "best_epoch": best["epoch"],
            "elapsed_s": time.time() - started,
            "history": history,
        }
        (self.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int) -> dict:
        self.model.train()
        total_loss, correct, seen = 0.0, 0, 0
        for step, (pixel_values, labels) in enumerate(loader, start=1):
            pixel_values = pixel_values.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = self.model(pixel_values=pixel_values).logits
            loss = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()

            n = labels.size(0)
            total_loss += loss.item() * n
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += n
            if step % LOG_EVERY_N_STEPS == 0:
                console.print(
                    f"  epoch {epoch} step {step}/{len(loader)} "
                    f"loss {total_loss / seen:.4f} acc {correct / seen:.4f}"
                )
        return {"loss": total_loss / seen, "acc": correct / seen}

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        preds: list[int] = []
        targets: list[int] = []
        for pixel_values, labels in loader:
            pixel_values = pixel_values.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            logits = self.model(pixel_values=pixel_values).logits
            total_loss += self.criterion(logits, labels).item() * labels.size(0)
            preds += logits.argmax(dim=1).tolist()
            targets += labels.tolist()

        report_kwargs = {"target_names": self.target_names, "labels": [0, 1], "zero_division": 0}
        return {
            "loss": total_loss / len(targets),
            "acc": float(np.mean(np.asarray(preds) == np.asarray(targets))),
            "f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
            "report": classification_report(targets, preds, **report_kwargs),
            "report_dict": classification_report(targets, preds, output_dict=True, **report_kwargs),
            "confusion_matrix": confusion_matrix(targets, preds, labels=[0, 1]),
        }

    def evaluate(self, checkpoint_path: str) -> dict:
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        val = self._val_epoch(self._make_loader("val", shuffle=False))

        console.print(f"[bold]Val loss {val['loss']:.4f}  acc {val['acc']:.4f}  macro-F1 {val['f1']:.4f}[/bold]")
        console.print(val["report"])
        console.print("Confusion matrix (rows = true, cols = predicted; order NonFight, Fight):")
        console.print(val["confusion_matrix"])
        return val
