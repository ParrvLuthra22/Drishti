import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "yolov8n.pt"
DEVICE = "mps"
PERSON_CLASS_ID = 0
FPS_WINDOW = 30


def parse_source(value: str) -> int | str:
    """Webcam index if numeric, otherwise a file path or stream URL."""
    return int(value) if value.isdigit() else value


def draw_overlay(frame, fps: float, people: int) -> None:
    lines = [f"FPS: {fps:.1f}", f"Device: {DEVICE}", f"People: {people}"]
    line_height = 22
    y = frame.shape[0] - 10 - line_height * (len(lines) - 1)
    for line in lines:
        cv2.putText(
            frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
        )
        y += line_height


def main() -> int:
    parser = argparse.ArgumentParser(description="YOLOv8n live detection demo")
    parser.add_argument("--source", default="1", help="webcam index (default 1: laptop webcam), file path, or RTSP URL")
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    args = parser.parse_args()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))  # auto-downloads on first run

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        print(f"Error: could not open source {args.source!r}", file=sys.stderr)
        return 1

    frame_times: deque[float] = deque(maxlen=FPS_WINDOW)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = model(frame, conf=args.conf, device=DEVICE, verbose=False)[0]
            annotated = result.plot()
            people = int((result.boxes.cls == PERSON_CLASS_ID).sum().item())

            frame_times.append(time.perf_counter())
            fps = 0.0
            if len(frame_times) > 1:
                fps = (len(frame_times) - 1) / (frame_times[-1] - frame_times[0])

            draw_overlay(annotated, fps, people)
            cv2.imshow("Drishti - YOLOv8n", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
