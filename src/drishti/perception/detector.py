from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console
from ultralytics import YOLO

console = Console()

ROOT = Path(__file__).resolve().parents[3]
PERSON_CLASS_ID = 0


@dataclass
class Detection:
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    track_id: int = -1  # filled in by the tracker later


class Detector:
    def __init__(
        self,
        model_path: str = "models/yolov8n.pt",
        device: str = "mps",
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        person_only: bool = True,
    ) -> None:
        path = Path(model_path)
        if not path.is_absolute():
            path = ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)

        self.device = device
        self.model_path = str(path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.person_only = person_only
        self.model = YOLO(str(path))  # auto-downloads on first run

        self.detect(np.zeros((640, 640, 3), dtype=np.uint8))  # warm-up
        console.print("[bold green]Detector ready[/bold green]")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        result = self.model(
            frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=[PERSON_CLASS_ID] if self.person_only else None,
            verbose=False,
        )[0]

        boxes = result.boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        detections: list[Detection] = []
        for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, class_ids):
            if self.person_only and class_id != PERSON_CLASS_ID:
                continue
            detections.append(
                Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=float(conf),
                    class_id=int(class_id),
                    class_name=str(self.model.names[int(class_id)]),
                )
            )
        return detections
