import argparse
import logging

from drishti.perception.detector import Detector
from drishti.perception.tracker import Tracker
from drishti.perception.worker import PerceptionWorker


def parse_source(value: str) -> int | str:
    """Webcam index if numeric, otherwise a file path or stream URL."""
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8n + ByteTrack live demo")
    parser.add_argument(
        "--source",
        default="1",
        help="webcam index (default 1: laptop webcam), file path, or RTSP URL",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    detector = Detector(conf_threshold=args.conf)
    tracker = Tracker()
    worker = PerceptionWorker("cam01", parse_source(args.source), detector, tracker, visualize=True)
    worker.run()


if __name__ == "__main__":
    main()
