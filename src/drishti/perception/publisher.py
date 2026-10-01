import logging
import uuid

import redis
from pydantic import BaseModel

from drishti.perception.tracker import Track

logger = logging.getLogger(__name__)


class TrackEvent(BaseModel):
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    age: int


class DetectionEvent(BaseModel):
    event_id: str  # str(uuid.uuid4())
    camera_id: str
    frame_id: int
    timestamp: float  # time.time()
    tracks: list[TrackEvent]
    fps: float

    @classmethod
    def from_tracks(
        cls, camera_id: str, frame_id: int, timestamp: float, tracks: list[Track], fps: float
    ) -> "DetectionEvent":
        return cls(
            event_id=str(uuid.uuid4()),
            camera_id=camera_id,
            frame_id=frame_id,
            timestamp=timestamp,
            tracks=[
                TrackEvent(
                    track_id=t.track_id,
                    bbox=t.bbox,
                    confidence=t.confidence,
                    class_id=t.class_id,
                    class_name=t.class_name,
                    age=t.age,
                )
                for t in tracks
            ],
            fps=fps,
        )


class RedisEventPublisher:
    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        stream_name: str = "drishti:detections",
        max_len: int = 10_000,
    ) -> None:
        # Short timeouts so a dead Redis cannot stall the perception loop.
        self.client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        self.stream_name = stream_name
        self.max_len = max_len
        self._healthy = True
        self._ping()

    def _ping(self) -> None:
        try:
            self.client.ping()
        except Exception as exc:  # noqa: BLE001 - Redis problems must never stop perception
            self._healthy = False
            logger.warning("Redis unavailable, events will be dropped: %s", exc)

    def publish(self, event: DetectionEvent) -> None:
        try:
            self.client.xadd(
                self.stream_name,
                {"data": event.model_dump_json()},
                maxlen=self.max_len,
                approximate=True,
            )
        except Exception as exc:  # noqa: BLE001 - Redis problems must never stop perception
            # Warn on the first failure only, not on every frame.
            if self._healthy:
                logger.warning("Redis publish failed, dropping events until it recovers: %s", exc)
            self._healthy = False
        else:
            if not self._healthy:
                logger.info("Redis publishing recovered")
            self._healthy = True
