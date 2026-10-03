import json
import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from rich.console import Console
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)
console = Console()

FRAME_SIZE = 64  # the architecture is fixed to 64x64 inputs
LATENT_DIM = 256
WARMUP_FRAMES = 15  # discarded at capture start while the camera's auto-exposure settles
SUSPICIOUS_FACTOR = 1.5


def _resolve_device(requested: str) -> torch.device:
    if requested == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but not available, falling back to cpu")
        return torch.device("cpu")
    return torch.device(requested)


def preprocess_frame(frame: np.ndarray, size: int = FRAME_SIZE) -> np.ndarray:
    """BGR (or already grayscale) frame -> float32 (size, size) array in [0, 1]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    resized = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


# ── model ───────────────────────────────────────────────────────────────────


class ConvEncoder(nn.Module):
    """(B, 1, 64, 64) -> (B, 256) latent vector."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, LATENT_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvDecoder(nn.Module):
    """(B, 256) latent vector -> (B, 1, 64, 64) reconstruction in [0, 1]."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, 128 * 8 * 8),
            nn.Unflatten(1, (128, 8, 8)),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 2, stride=2),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class SceneAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = ConvEncoder()
        self.decoder = ConvDecoder()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# ── dataset ─────────────────────────────────────────────────────────────────


class NormalSceneDataset(Dataset):
    """Grayscale 64x64 frames of a scene in its normal state, as (1, H, W) tensors in [0, 1]."""

    def __init__(
        self,
        video_source: int | str,  # webcam index, file path, or RTSP URL
        num_frames: int = 2000,
        frame_size: int = FRAME_SIZE,
        skip: int = 3,  # keep every Nth frame
    ) -> None:
        self.frames: list[torch.Tensor] = []

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise RuntimeError(f"could not open video source {video_source!r}")
        try:
            read = 0
            while len(self.frames) < num_frames:
                ok, frame = cap.read()
                if not ok:  # end of file or camera lost
                    break
                read += 1
                if read <= WARMUP_FRAMES or (read - WARMUP_FRAMES) % skip != 0:
                    continue
                self.frames.append(torch.from_numpy(preprocess_frame(frame, frame_size)).unsqueeze(0))
                if len(self.frames) % 250 == 0:
                    console.print(f"  collected {len(self.frames)}/{num_frames} frames")
        finally:
            cap.release()

        console.print(f"Collected [bold]{len(self.frames)}[/bold] frames (requested {num_frames}, skip={skip})")

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.frames[idx]


# ── training ────────────────────────────────────────────────────────────────


class AnomalyTrainer:
    def __init__(
        self,
        device: str = "mps",
        lr: float = 1e-3,
        epochs: int = 20,
        batch_size: int = 32,
        output_dir: str = "models/anomaly",
    ) -> None:
        self.device = _resolve_device(device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.output_dir = Path(output_dir)

        self.model = SceneAutoencoder().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.threshold: float | None = None

    def train(self, dataset: NormalSceneDataset) -> None:
        if len(dataset) == 0:
            raise ValueError("dataset is empty, nothing to train on")

        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total, seen = 0.0, 0
            for batch in loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(batch), batch)
                loss.backward()
                self.optimizer.step()
                total += loss.item() * batch.size(0)
                seen += batch.size(0)
            if epoch % 5 == 0:
                console.print(f"Epoch {epoch}/{self.epochs}  loss {total / seen:.6f}")

        errors = self._reconstruction_errors(dataset)
        mean, std = float(errors.mean()), float(errors.std())
        self.threshold = mean + 2 * std

        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), self.output_dir / "autoencoder.pt")
        (self.output_dir / "threshold.json").write_text(
            json.dumps(
                {
                    "threshold": self.threshold,
                    "mean_recon_error": mean,
                    "std_recon_error": std,
                    "num_frames": len(dataset),
                },
                indent=2,
            )
        )
        console.print(
            f"Saved model + threshold to {self.output_dir} "
            f"(error mean {mean:.6f}, std {std:.6f})"
        )

    @torch.no_grad()
    def _reconstruction_errors(self, dataset: NormalSceneDataset) -> np.ndarray:
        """Per-frame MSE of the trained model over the whole dataset."""
        self.model.eval()
        errors: list[np.ndarray] = []
        for batch in DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0):
            batch = batch.to(self.device)
            per_frame = ((self.model(batch) - batch) ** 2).mean(dim=(1, 2, 3))
            errors.append(per_frame.cpu().numpy())
        return np.concatenate(errors)

    def load(self, output_dir: str) -> None:
        directory = Path(output_dir)
        state = torch.load(directory / "autoencoder.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.threshold = float(json.loads((directory / "threshold.json").read_text())["threshold"])
        self.model.eval()


# ── scoring ─────────────────────────────────────────────────────────────────


class AnomalyScorer:
    def __init__(
        self,
        model: SceneAutoencoder,
        threshold: float,
        device: str = "mps",
        window: int = 8,
    ) -> None:
        self.device = _resolve_device(device)
        self.model = model.to(self.device).eval()  # BatchNorm must use running stats when scoring
        self.threshold = threshold
        self.window = window
        self._buffer: deque[float] = deque(maxlen=window)

    @torch.no_grad()
    def score_frame(self, frame: np.ndarray) -> dict:
        x = torch.from_numpy(preprocess_frame(frame)).unsqueeze(0).unsqueeze(0).to(self.device)
        raw = float(((self.model(x) - x) ** 2).mean().item())

        self._buffer.append(raw)
        smoothed = float(np.mean(self._buffer))

        if smoothed < self.threshold:
            level = "normal"
        elif smoothed < self.threshold * SUSPICIOUS_FACTOR:
            level = "suspicious"
        else:
            level = "anomaly"

        return {
            "raw_score": raw,
            "smoothed_score": smoothed,
            "threshold": self.threshold,
            "is_anomaly": smoothed > self.threshold,
            "anomaly_level": level,
        }
