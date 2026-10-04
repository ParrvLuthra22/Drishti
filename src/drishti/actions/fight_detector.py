import logging
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from rich.console import Console
from torch import nn
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

from drishti.actions.dataset import IMAGENET_MEAN, IMAGENET_STD
from drishti.actions.trainer import resolve_device

logger = logging.getLogger(__name__)
console = Console()

CHECKPOINT_FILE = "best_model.pt"


def preprocess_frame(frame: np.ndarray, size: int = 224) -> torch.Tensor:
    """BGR frame -> (3, size, size) float tensor, resized and ImageNet-normalised exactly as in training."""
    rgb = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
    normalised = (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(normalised).permute(2, 0, 1).contiguous()


def load_model(model_repo: str) -> nn.Module:
    """Load a fine-tuned VideoMAE from a Hub repo id or a local directory.

    Uses from_pretrained when the repo has standard weights; the training run uploads a raw
    state dict (`best_model.pt`) next to `config.json`, so fall back to building the model from
    the config and loading that file.
    """
    try:
        return VideoMAEForVideoClassification.from_pretrained(model_repo)
    except OSError:
        local = Path(model_repo)
        checkpoint = local / CHECKPOINT_FILE if local.is_dir() else Path(hf_hub_download(model_repo, CHECKPOINT_FILE))
        model = VideoMAEForVideoClassification(VideoMAEConfig.from_pretrained(model_repo))
        result = model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True), strict=False
        )
        # Checkpoints saved by older transformers carry a per-layer attention key bias that newer
        # releases dropped. It never affected outputs (a key bias shifts every attention score of a
        # query by the same amount, which softmax cancels), so skip those and nothing else.
        unexpected = [k for k in result.unexpected_keys if not k.endswith("attention.key.bias")]
        if result.missing_keys or unexpected:
            raise RuntimeError(f"checkpoint does not match the model: missing={result.missing_keys}, unexpected={unexpected}")
        return model


class FightDetector:
    def __init__(
        self,
        model_repo: str = "Parrv/drishti-fight-detector",
        device: str = "mps",
        num_frames: int = 16,
        frame_size: int = 224,
        confidence_threshold: float = 0.5,
        sample_interval_s: float = 0.0,
        model: nn.Module | None = None,
    ) -> None:
        """`sample_interval_s` > 0 keeps at most one frame per interval, so the buffer spans
        num_frames * interval seconds. The model was trained on 16 frames spread over ~5 s clips,
        so live use wants about 0.31 s; 0 keeps every frame you add.

        `model` lets callers supply an already-built model instead of loading one from the Hub.
        """
        self.device = resolve_device(device)
        self.num_frames = num_frames
        self.frame_size = frame_size
        self.confidence_threshold = confidence_threshold
        self.sample_interval_s = sample_interval_s

        self.model = (model if model is not None else load_model(model_repo)).to(self.device).eval()
        self._fight_index = int(self.model.config.label2id.get("Fight", 1))

        self._frame_buffer: deque[torch.Tensor] = deque(maxlen=num_frames)
        self._last_added = float("-inf")
        # add_frame runs on the vision loop while predict runs on a worker thread.
        self._lock = threading.Lock()
        console.print("[bold green]FightDetector ready[/bold green]")

    def add_frame(self, frame: np.ndarray, timestamp: float | None = None) -> None:
        now = time.monotonic() if timestamp is None else timestamp
        if self.sample_interval_s > 0 and now - self._last_added < self.sample_interval_s:
            return
        tensor = preprocess_frame(frame, self.frame_size)
        with self._lock:
            self._frame_buffer.append(tensor)
            self._last_added = now

    def reset(self) -> None:
        with self._lock:
            self._frame_buffer.clear()
            self._last_added = float("-inf")

    def predict(self) -> dict:
        with self._lock:
            if len(self._frame_buffer) < self.num_frames:
                return {"fight_probability": 0.0, "fight_detected": False, "ready": False}
            frames = list(self._frame_buffer)

        # (T, 3, H, W) -> (1, T, 3, H, W): VideoMAE takes (batch, frames, channels, height, width)
        pixel_values = torch.stack(frames).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(pixel_values=pixel_values).logits
            fight_prob = torch.softmax(logits, dim=-1)[0][self._fight_index].item()

        return {
            "fight_probability": fight_prob,
            "fight_detected": fight_prob > self.confidence_threshold,
            "ready": True,
        }
