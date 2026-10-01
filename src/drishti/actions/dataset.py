import logging
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch
from rich.console import Console
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
console = Console()

# Expected layout: <root>/{train,val}/{Fight,NonFight}/*.avi
LABELS = {"NonFight": 0, "Fight": 1}
VIDEO_EXTENSIONS = {".avi", ".mp4"}
IMAGE_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class RWF2000Dataset(Dataset):
    """RWF-2000 fight / non-fight clips, returned as (3, num_frames, 224, 224) tensors."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        num_frames: int = 16,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        """`transform`, if given, is applied to the normalised (3, T, H, W) tensor."""
        self.root = Path(root).expanduser()
        self.split = split
        self.num_frames = num_frames
        self.transform = transform

        split_dir = self.root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"RWF-2000 split folder not found: {split_dir}")

        self.samples: list[tuple[str, int]] = []
        for class_name, label in sorted(LABELS.items(), key=lambda kv: -kv[1]):  # Fight first
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"RWF-2000 class folder not found: {class_dir}")
            clips = sorted(
                p
                for p in class_dir.iterdir()
                if p.suffix.lower() in VIDEO_EXTENSIONS and not p.name.startswith(".")
            )
            self.samples.extend((str(p), label) for p in clips)

        counts = self.class_counts()
        console.print(
            f"RWF-2000 [bold]{split}[/bold]: "
            f"[red]{counts['Fight']} Fight[/red], [green]{counts['NonFight']} NonFight[/green]"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        video_path, label = self.samples[idx]
        frames = self._load_frames(video_path)
        if frames is None:
            # Unreadable or too-short clip: return zeros with label 0 rather than crash a training run.
            return torch.zeros(3, self.num_frames, IMAGE_SIZE, IMAGE_SIZE), 0

        video = (frames.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD  # (T, H, W, 3)
        tensor = torch.from_numpy(video).permute(3, 0, 1, 2).contiguous()  # (3, T, H, W)
        if self.transform is not None:
            tensor = self.transform(tensor)
        return tensor, label

    def _load_frames(self, video_path: str) -> np.ndarray | None:
        """Return (num_frames, 224, 224, 3) uint8 RGB frames, evenly spaced, or None on failure."""
        cap = cv2.VideoCapture(video_path)
        try:
            if not cap.isOpened():
                logger.warning("could not open %s", video_path)
                return None
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < self.num_frames:
                logger.warning("%s has %d frames, need %d", video_path, total, self.num_frames)
                return None

            wanted = set(np.linspace(0, total - 1, self.num_frames).astype(int).tolist())
            frames: list[np.ndarray] = []
            # Decode sequentially: seeking inside AVI files is slow and not frame-accurate.
            for frame_idx in range(total):
                if not cap.grab():
                    break
                if frame_idx in wanted:
                    ok, frame = cap.retrieve()
                    if not ok:
                        break
                    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if len(frames) == self.num_frames:
                        break
            if len(frames) != self.num_frames:
                logger.warning("%s: only decoded %d/%d frames", video_path, len(frames), self.num_frames)
                return None
            return np.stack(frames)
        finally:
            cap.release()

    def class_counts(self) -> dict[str, int]:
        fights = sum(label for _, label in self.samples)
        return {"Fight": fights, "NonFight": len(self.samples) - fights}
