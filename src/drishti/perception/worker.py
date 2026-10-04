import logging
import signal
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from drishti.perception.detector import Detector
from drishti.perception.publisher import DetectionEvent, RedisEventPublisher
from drishti.perception.tracker import Track, Tracker

logger = logging.getLogger(__name__)

LOG_EVERY_N_FRAMES = 30
FPS_WINDOW = 30
WINDOW_NAME = "Drishti"


@dataclass
class PerceptionResult:
    camera_id: str
    frame_id: int
    tracks: list[Track]
    fps: float
    timestamp: float  # time.time()


class PerceptionWorker:
    def __init__(
        self,
        camera_id: str,
        source: int | str,  # webcam index, file path, or RTSP URL
        detector: Detector,
        tracker: Tracker,
        visualize: bool = False,
        publisher: RedisEventPublisher | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.visualize = visualize
        self.publisher = publisher
        self._running = False
        self.frame_id = 0
        self.last_frame: np.ndarray | None = None  # the frame behind the result in _on_result
        self._fps_window: deque[float] = deque(maxlen=FPS_WINDOW)

    def run(self) -> None:
        self._running = True
        previous_handlers = self._install_signal_handlers()

        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            logger.error("cam=%s could not open source %r", self.camera_id, self.source)
            self._restore_signal_handlers(previous_handlers)
            return

        try:
            while self._running:
                ok, frame = cap.read()
                if not ok:
                    break

                self.last_frame = frame
                self._fps_window.append(time.perf_counter())
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame)

                result = PerceptionResult(
                    camera_id=self.camera_id,
                    frame_id=self.frame_id,
                    tracks=tracks,
                    fps=self._current_fps(),
                    timestamp=time.time(),
                )
                self._on_result(result)

                if self.visualize:
                    self._draw(frame, result)
                    cv2.imshow(WINDOW_NAME, frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                self.frame_id += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self._restore_signal_handlers(previous_handlers)

    def _current_fps(self) -> float:
        """FPS over the last FPS_WINDOW frames (wall-clock, includes capture and drawing)."""
        if len(self._fps_window) < 2:
            return 0.0
        elapsed = self._fps_window[-1] - self._fps_window[0]
        return (len(self._fps_window) - 1) / elapsed if elapsed > 0 else 0.0

    def _on_result(self, result: PerceptionResult) -> None:
        if self.publisher is not None:
            self.publisher.publish(
                DetectionEvent.from_tracks(
                    camera_id=result.camera_id,
                    frame_id=result.frame_id,
                    timestamp=result.timestamp,
                    tracks=result.tracks,
                    fps=result.fps,
                )
            )

        if result.frame_id % LOG_EVERY_N_FRAMES == 0:
            logger.info(
                "cam=%s frame=%d tracks=%d fps=%.1f",
                result.camera_id,
                result.frame_id,
                len(result.tracks),
                result.fps,
            )

    def _draw(self, frame: np.ndarray, result: PerceptionResult) -> None:
        for track in result.tracks:
            x1, y1, x2, y2 = (int(v) for v in track.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"ID {track.track_id}",
                (x1, max(y1 - 8, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        lines = self._overlay_lines(result)
        line_height = 22
        y = frame.shape[0] - 10 - line_height * (len(lines) - 1)
        for line in lines:
            cv2.putText(
                frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA
            )
            y += line_height

    def _overlay_lines(self, result: PerceptionResult) -> list[str]:
        """Text lines drawn in the bottom-left corner. Subclasses override to add metrics."""
        return [
            f"FPS: {result.fps:.1f}",
            f"Camera: {result.camera_id}",
            f"Tracks: {len(result.tracks)}",
        ]

    def _handle_stop(self, signum: int, _frame: object) -> None:
        self._running = False

    def _install_signal_handlers(self) -> dict[int, object]:
        previous: dict[int, object] = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous[sig] = signal.signal(sig, self._handle_stop)
            except ValueError:  # not in the main thread; caller must stop us another way
                logger.warning("cam=%s cannot install signal handlers off the main thread", self.camera_id)
                break
        return previous

    def _restore_signal_handlers(self, previous: dict[int, object]) -> None:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
