import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from drishti.anomaly.autoencoder import SUSPICIOUS_FACTOR, AnomalyScorer, AnomalyTrainer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "anomaly"
BAR_HEIGHT = 24


def parse_source(value: str) -> int | str:
    """Webcam index if numeric, otherwise a file path or stream URL."""
    return int(value) if value.isdigit() else value


def score_color(score: float, threshold: float) -> tuple[int, int, int]:
    """BGR colour: green when quiet, yellow exactly at the threshold, red at threshold * 1.5."""
    u = score / threshold if threshold > 0 else 0.0
    # OpenCV hue: 60 = green, 30 = yellow, 0 = red
    hue = 60 - 30 * min(u, 1.0) - 30 * float(np.clip((u - 1.0) / (SUSPICIOUS_FACTOR - 1.0), 0.0, 1.0))
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[int(hue), 255, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


def draw_overlay(frame: np.ndarray, result: dict) -> None:
    """Score bar across the top, plus level text. The bar is full at threshold * SUSPICIOUS_FACTOR."""
    height, width = frame.shape[:2]
    full_scale = result["threshold"] * SUSPICIOUS_FACTOR
    ratio = min(result["smoothed_score"] / full_scale, 1.0) if full_scale > 0 else 0.0

    cv2.rectangle(frame, (0, 0), (width, BAR_HEIGHT), (40, 40, 40), -1)
    cv2.rectangle(frame, (0, 0), (int(width * ratio), BAR_HEIGHT), score_color(result["smoothed_score"], result["threshold"]), -1)
    tick_x = int(width / SUSPICIOUS_FACTOR)  # where the threshold sits on the bar
    cv2.line(frame, (tick_x, 0), (tick_x, BAR_HEIGHT), (255, 255, 255), 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    if result["is_anomaly"]:
        cv2.putText(frame, "ANOMALY DETECTED", (10, BAR_HEIGHT + 40), font, 1.1, (0, 0, 255), 3, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"score {result['smoothed_score']:.5f} / threshold {result['threshold']:.5f}",
        (10, height - 40),
        font,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame, f"level: {result['anomaly_level']}", (10, height - 15), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Live anomaly scoring demo")
    parser.add_argument("--source", default="1", help="webcam index (default 1), file path, or RTSP URL")
    args = parser.parse_args()

    if not (MODEL_DIR / "autoencoder.pt").exists():
        print(f"No trained model in {MODEL_DIR}. Run scripts/train_anomaly.py first.", file=sys.stderr)
        return 1

    trainer = AnomalyTrainer(device="mps")
    trainer.load(str(MODEL_DIR))
    scorer = AnomalyScorer(trainer.model, trainer.threshold, device="mps")

    cap = cv2.VideoCapture(parse_source(args.source))
    if not cap.isOpened():
        print(f"Could not open source {args.source!r}", file=sys.stderr)
        return 1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            draw_overlay(frame, scorer.score_frame(frame))
            cv2.imshow("Drishti - anomaly", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
