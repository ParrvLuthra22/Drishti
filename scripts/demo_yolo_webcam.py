import argparse
import logging

import numpy as np

from drishti.crowd.analyser import CrowdAnalyser, CrowdMetrics, HeatmapAccumulator
from drishti.perception.detector import Detector
from drishti.perception.tracker import Tracker
from drishti.perception.worker import PerceptionResult, PerceptionWorker

logger = logging.getLogger("demo")

HEATMAP_SIZE = (1280, 720)  # (width, height); resized below if the camera delivers another size


class CrowdWorker(PerceptionWorker):
    """PerceptionWorker that also feeds the crowd analyser and heatmap."""

    def __init__(
        self,
        camera_id: str,
        source: int | str,
        detector: Detector,
        tracker: Tracker,
        analyser: CrowdAnalyser,
        heatmap: HeatmapAccumulator,
        show_heatmap: bool = False,
    ) -> None:
        super().__init__(camera_id, source, detector, tracker, visualize=True)
        self.analyser = analyser
        self.heatmap = heatmap
        self.show_heatmap = show_heatmap
        self.metrics: CrowdMetrics | None = None

    def _on_result(self, result: PerceptionResult) -> None:
        self.metrics = self.analyser.update(result)
        self.heatmap.update(result)
        if result.frame_id % 30 == 0:
            m = self.metrics
            logger.info(
                "density=%s (%.2f) flow=%s (%.2f) speed=%.3f box-heights/frame bottleneck=%s",
                m.density_level,
                m.density_score,
                m.flow_direction,
                m.flow_score,
                m.avg_speed,
                m.bottleneck_detected,
            )
        super()._on_result(result)

    def _draw(self, frame: np.ndarray, result: PerceptionResult) -> None:
        height, width = frame.shape[:2]
        if (width, height) != self.heatmap.size:
            # Track coordinates are in camera pixels, so the map must match the real frame size.
            logger.info("Camera frames are %dx%d; resizing heatmap to match", width, height)
            self.heatmap = HeatmapAccumulator(width, height, self.heatmap.decay)
        if self.show_heatmap:
            frame[:] = self.heatmap.get_overlay(frame)
        super()._draw(frame, result)

    def _overlay_lines(self, result: PerceptionResult) -> list[str]:
        m = self.metrics
        if m is None:
            return [f"FPS: {result.fps:.1f}"]
        return [
            f"FPS: {result.fps:.1f}",
            f"Tracks: {m.total_tracks}",
            f"Density: {m.density_level} ({m.density_score:.2f})",
            f"Flow: {m.flow_direction} ({m.flow_score:.2f})",
            f"Bottleneck: {'YES' if m.bottleneck_detected else 'NO'}",
        ]


def parse_source(value: str) -> int | str:
    """Webcam index if numeric, otherwise a file path or stream URL."""
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8n + ByteTrack + crowd analytics live demo")
    parser.add_argument(
        "--source",
        default="1",
        help="webcam index (default 1: laptop webcam), file path, or RTSP URL",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--heatmap", action="store_true", help="blend the crowd heatmap onto the video")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    detector = Detector(conf_threshold=args.conf)
    tracker = Tracker()
    analyser = CrowdAnalyser()
    heatmap = HeatmapAccumulator(frame_width=HEATMAP_SIZE[0], frame_height=HEATMAP_SIZE[1])
    worker = CrowdWorker(
        "cam01",
        parse_source(args.source),
        detector,
        tracker,
        analyser,
        heatmap,
        show_heatmap=args.heatmap,
    )
    worker.run()


if __name__ == "__main__":
    main()
