import warnings
from dataclasses import dataclass

import numpy as np
import supervision as sv

from drishti.perception.detector import Detection


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    age: int  # frames this track has been alive


class Tracker:
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 15,
    ) -> None:
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.frame_rate = frame_rate
        self.reset()

    def reset(self) -> None:
        """Clear all tracking state."""
        with warnings.catch_warnings():
            # sv.ByteTrack is deprecated (removal in 0.31); supervision is capped below that.
            warnings.simplefilter("ignore", FutureWarning)
            self.tracker = sv.ByteTrack(
                track_activation_threshold=self.track_thresh,
                lost_track_buffer=self.track_buffer,
                minimum_matching_threshold=self.match_thresh,
                frame_rate=self.frame_rate,
            )
        self._frame_count = 0
        self._first_seen: dict[int, int] = {}
        self._class_names: dict[int, str] = {}

    def update(self, detections: list[Detection], frame: np.ndarray) -> list[Track]:
        """Associate detections with existing tracks.

        `frame` is accepted for API symmetry with appearance-based trackers;
        ByteTrack only uses box geometry and confidence.
        """
        self._frame_count += 1

        if detections:
            for det in detections:
                self._class_names[det.class_id] = det.class_name
            sv_detections = sv.Detections(
                xyxy=np.array([d.bbox for d in detections], dtype=np.float32),
                confidence=np.array([d.confidence for d in detections], dtype=np.float32),
                class_id=np.array([d.class_id for d in detections], dtype=int),
            )
        else:
            sv_detections = sv.Detections.empty()

        tracked = self.tracker.update_with_detections(sv_detections)

        tracks: list[Track] = []
        for i in range(len(tracked)):
            track_id = int(tracked.tracker_id[i])
            class_id = int(tracked.class_id[i])
            first_seen = self._first_seen.setdefault(track_id, self._frame_count)
            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
            tracks.append(
                Track(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    confidence=float(tracked.confidence[i]),
                    class_id=class_id,
                    class_name=self._class_names.get(class_id, str(class_id)),
                    age=self._frame_count - first_seen + 1,
                )
            )
        return tracks
