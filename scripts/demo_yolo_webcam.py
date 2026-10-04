import argparse
import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import requests

from drishti.actions.fight_detector import FightDetector
from drishti.anomaly.autoencoder import AnomalyScorer, AnomalyTrainer
from drishti.crowd.analyser import CrowdAnalyser, CrowdMetrics, HeatmapAccumulator
from drishti.intelligence.risk_agent import DrishtiState, RiskAgent
from drishti.perception.detector import Detector
from drishti.perception.tracker import Tracker
from drishti.perception.worker import PerceptionResult, PerceptionWorker

logger = logging.getLogger("demo")

ROOT = Path(__file__).resolve().parents[1]
ANOMALY_MODEL_DIR = ROOT / "models" / "anomaly"
HEATMAP_SIZE = (1280, 720)  # (width, height); resized below if the camera delivers another size
PUSH_INTERVAL_S = 1.0  # risk assessment and dashboard update cadence
DASHBOARD_TIMEOUT_S = 0.1
# VideoMAE was trained on 16 frames spread over ~5 s clips, so live input samples a frame every 0.3 s
# (16 consecutive frames at 15 FPS cover only ~1 s, and the model then calls everything a non-fight).
FIGHT_SAMPLE_INTERVAL_S = 0.3


class CrowdWorker(PerceptionWorker):
    """PerceptionWorker that also feeds the crowd analyser, heatmap, anomaly scorer and risk agent."""

    def __init__(
        self,
        camera_id: str,
        source: int | str,
        detector: Detector,
        tracker: Tracker,
        analyser: CrowdAnalyser,
        heatmap: HeatmapAccumulator,
        show_heatmap: bool = False,
        visualize: bool = True,
        anomaly_scorer: AnomalyScorer | None = None,
        risk_agent: RiskAgent | None = None,
        dashboard_url: str = "",
        fight_detector: FightDetector | None = None,
    ) -> None:
        super().__init__(camera_id, source, detector, tracker, visualize=visualize)
        self.analyser = analyser
        self.heatmap = heatmap
        self.show_heatmap = show_heatmap
        self.anomaly_scorer = anomaly_scorer
        self.risk_agent = risk_agent
        self.fight_detector = fight_detector
        self.dashboard_url = dashboard_url.rstrip("/")

        self.metrics: CrowdMetrics | None = None
        self.anomaly: dict | None = None
        self.risk_state: DrishtiState | None = None
        self.fight: dict | None = None

        # Risk assessment can call an LLM and the dashboard POST is network I/O, so both run on a
        # background thread; the vision loop never waits on them.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="risk")
        self._pending: Future | None = None
        self._last_push = 0.0

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def _on_result(self, result: PerceptionResult) -> None:
        self.metrics = self.analyser.update(result)
        self.heatmap.update(result)
        if self.anomaly_scorer is not None and self.last_frame is not None:
            self.anomaly = self.anomaly_scorer.score_frame(self.last_frame)
        if self.fight_detector is not None and self.last_frame is not None:
            self.fight_detector.add_frame(self.last_frame)  # cheap; thins itself to one frame per 0.3 s

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
        self._maybe_assess(result)
        super()._on_result(result)

    def _maybe_assess(self, result: PerceptionResult) -> None:
        if self.risk_agent is None or self.metrics is None:
            return
        now = time.monotonic()
        busy = self._pending is not None and not self._pending.done()
        if busy or now - self._last_push < PUSH_INTERVAL_S:
            return
        self._last_push = now
        self._pending = self._pool.submit(self._assess_and_push, result, self.metrics, self.anomaly)

    def _assess_and_push(
        self, result: PerceptionResult, metrics: CrowdMetrics, anomaly: dict | None
    ) -> None:
        try:
            # Inference runs here, off the vision loop, once per push; 0.0 until the buffer has 16 frames.
            fight_probability = 0.0
            if self.fight_detector is not None:
                self.fight = self.fight_detector.predict()
                if self.fight["ready"]:
                    fight_probability = self.fight["fight_probability"]
                logger.info("fight probability %.2f (ready=%s)", fight_probability, self.fight["ready"])
            state = self.risk_agent.assess(
                result.camera_id,
                result.frame_id,
                result.timestamp,
                metrics,
                fight_probability=fight_probability,
                anomaly_result=anomaly,
            )
            self.risk_state = state
            if self.dashboard_url:
                requests.post(
                    f"{self.dashboard_url}/api/update",
                    json={"risk_state": state, "metrics": asdict(metrics), "fps": result.fps},
                    timeout=DASHBOARD_TIMEOUT_S,
                )
        except Exception as exc:  # noqa: BLE001 - the dashboard must never affect the vision loop
            logger.debug("risk/dashboard update failed: %s", exc)

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
        lines = [
            f"FPS: {result.fps:.1f}",
            f"Tracks: {m.total_tracks}",
            f"Density: {m.density_level} ({m.density_score:.2f})",
            f"Flow: {m.flow_direction} ({m.flow_score:.2f})",
            f"Bottleneck: {'YES' if m.bottleneck_detected else 'NO'}",
        ]
        if self.fight is not None:
            lines.append(
                f"Fight: {self.fight['fight_probability']:.0%}" if self.fight["ready"] else "Fight: warming up"
            )
        if self.risk_state is not None:
            lines.append(f"Risk: {self.risk_state['risk_score']:.1f} ({self.risk_state['risk_level']})")
        return lines


def load_anomaly_scorer() -> AnomalyScorer | None:
    if not (ANOMALY_MODEL_DIR / "autoencoder.pt").exists():
        logger.info("No trained anomaly model in %s; anomaly scoring disabled", ANOMALY_MODEL_DIR)
        return None
    trainer = AnomalyTrainer(device="mps")
    trainer.load(str(ANOMALY_MODEL_DIR))
    return AnomalyScorer(trainer.model, trainer.threshold, device="mps")


def parse_source(value: str) -> int | str:
    """Webcam index if numeric, otherwise a file path or stream URL."""
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLOv8n + ByteTrack + crowd analytics + risk live demo")
    parser.add_argument(
        "--source",
        default="1",
        help="webcam index (default 1: laptop webcam), file path, or RTSP URL",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--heatmap", action="store_true", help="blend the crowd heatmap onto the video")
    parser.add_argument(
        "--fight",
        action="store_true",
        help="run the VideoMAE fight detector (extra startup time and memory)",
    )
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show the annotated video window (--no-visualize for headless)",
    )
    parser.add_argument(
        "--dashboard-url",
        default="http://localhost:8002",
        help="dashboard API to push risk state to; pass an empty string to disable",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    worker = CrowdWorker(
        "cam01",
        parse_source(args.source),
        Detector(conf_threshold=args.conf),
        Tracker(),
        CrowdAnalyser(),
        HeatmapAccumulator(frame_width=HEATMAP_SIZE[0], frame_height=HEATMAP_SIZE[1]),
        show_heatmap=args.heatmap,
        visualize=args.visualize,
        anomaly_scorer=load_anomaly_scorer(),
        risk_agent=RiskAgent(),
        dashboard_url=args.dashboard_url,
        fight_detector=FightDetector(sample_interval_s=FIGHT_SAMPLE_INTERVAL_S) if args.fight else None,
    )
    try:
        worker.run()
    finally:
        worker.close()


if __name__ == "__main__":
    main()
