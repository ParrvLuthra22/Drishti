from collections import deque
from dataclasses import dataclass, field

import cv2
import numpy as np

from drishti.perception.worker import PerceptionResult

MAX_TRACKS_FOR_FULL_DENSITY = 20  # frame-level density saturates at this many tracks
ZONE_CRITICAL_PEOPLE_PER_SQM = 4.0  # zone density saturates at this many people per m²
CALM_SPEED_PX_PER_FRAME = 2.0
DIRECTIONAL_FLOW_SCORE = 0.3
POSITION_HISTORY = 15
BOTTLENECK_MIN_COUNT = 5  # a bottleneck needs strictly more tracks than this


@dataclass
class ZoneConfig:
    zone_id: str
    polygon: list[tuple[float, float]]  # (x, y) vertices
    area_sqm: float  # real-world area in m²
    name: str  # e.g. "Gate A"


@dataclass
class CrowdMetrics:
    camera_id: str
    timestamp: float
    frame_id: int

    # Counts
    total_tracks: int

    # Density
    density_level: str  # "low" | "moderate" | "high" | "critical"
    density_score: float  # 0.0 to 1.0

    # Flow
    avg_speed: float  # pixels per frame, avg across tracks
    flow_direction: str  # "calm" | "directional" | "chaotic"
    flow_score: float  # 0.0 to 1.0 (higher = more chaotic)

    # Zone metrics (keyed by zone_id)
    zone_counts: dict[str, int] = field(default_factory=dict)
    zone_density: dict[str, str] = field(default_factory=dict)

    # Bottleneck
    bottleneck_detected: bool = False
    bottleneck_zones: list[str] = field(default_factory=list)


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2, (y1 + y2) / 2


class CrowdAnalyser:
    def __init__(
        self,
        zones: list[ZoneConfig] | None = None,
        history_len: int = 90,  # 6 seconds at 15fps
        density_thresholds: tuple[float, float, float] = (0.3, 0.6, 0.85),
    ) -> None:
        self.zones = zones or []
        self.density_thresholds = density_thresholds
        self._history: deque[PerceptionResult] = deque(maxlen=history_len)
        self._track_positions: dict[int, deque[tuple[float, float]]] = {}
        self._last_seen: dict[int, int] = {}  # track_id -> update counter
        self._updates = 0
        self._stale_after = history_len

    def update(self, result: PerceptionResult) -> CrowdMetrics:
        self._history.append(result)
        self._updates += 1
        self._update_positions(result)

        density_score = self._density_score(len(result.tracks))
        avg_speed, flow_direction, flow_score = self._compute_flow(result)
        zone_counts, zone_density = self._zone_counts(result)
        bottleneck, bottleneck_zones = self._detect_bottleneck(zone_counts, zone_density)

        return CrowdMetrics(
            camera_id=result.camera_id,
            timestamp=result.timestamp,
            frame_id=result.frame_id,
            total_tracks=len(result.tracks),
            density_level=self._density_level(density_score),
            density_score=density_score,
            avg_speed=avg_speed,
            flow_direction=flow_direction,
            flow_score=flow_score,
            zone_counts=zone_counts,
            zone_density=zone_density,
            bottleneck_detected=bottleneck,
            bottleneck_zones=bottleneck_zones,
        )

    def _update_positions(self, result: PerceptionResult) -> None:
        for track in result.tracks:
            positions = self._track_positions.setdefault(
                track.track_id, deque(maxlen=POSITION_HISTORY)
            )
            positions.append(_bbox_center(track.bbox))
            self._last_seen[track.track_id] = self._updates

        # Drop tracks that have not been seen for a while so the dict cannot grow forever.
        stale = [t for t, seen in self._last_seen.items() if self._updates - seen > self._stale_after]
        for track_id in stale:
            del self._track_positions[track_id]
            del self._last_seen[track_id]

    def _density_score(self, track_count: int) -> float:
        return float(np.clip(track_count / MAX_TRACKS_FOR_FULL_DENSITY, 0.0, 1.0))

    def _density_level(self, score: float) -> str:
        low, moderate, high = self.density_thresholds
        if score < low:
            return "low"
        if score < moderate:
            return "moderate"
        if score < high:
            return "high"
        return "critical"

    def _compute_flow(self, result: PerceptionResult) -> tuple[float, str, float]:
        velocities: list[tuple[float, float]] = []
        for track in result.tracks:
            positions = self._track_positions.get(track.track_id)
            if positions is not None and len(positions) >= 2:
                (x0, y0), (x1, y1) = positions[-2], positions[-1]
                velocities.append((x1 - x0, y1 - y0))

        if not velocities:
            return 0.0, "calm", 0.0

        vel = np.asarray(velocities, dtype=np.float64)
        speeds = np.hypot(vel[:, 0], vel[:, 1])
        avg_speed = float(speeds.mean())

        flow_score = self._flow_score(vel, speeds)
        if avg_speed < CALM_SPEED_PX_PER_FRAME:
            direction = "calm"
        elif flow_score < DIRECTIONAL_FLOW_SCORE:
            direction = "directional"
        else:
            direction = "chaotic"
        return avg_speed, direction, flow_score

    @staticmethod
    def _flow_score(vel: np.ndarray, speeds: np.ndarray) -> float:
        """Circular std of movement angles over pi: 0 = everyone aligned, 1 = no common direction.

        A plain std of angles would treat headings of +179° and -179° as opposite; the
        circular form handles the wrap-around.
        """
        moving = speeds > 1e-6  # angle is undefined for a track that did not move
        if moving.sum() < 2:
            return 0.0
        angles = np.arctan2(vel[moving, 1], vel[moving, 0])
        resultant = np.hypot(np.cos(angles).mean(), np.sin(angles).mean())  # 1 = aligned
        if resultant < 1e-9:
            return 1.0
        circular_std = np.sqrt(-2.0 * np.log(min(resultant, 1.0)))
        return float(np.clip(circular_std / np.pi, 0.0, 1.0))

    def _zone_counts(self, result: PerceptionResult) -> tuple[dict[str, int], dict[str, str]]:
        counts: dict[str, int] = {}
        density: dict[str, str] = {}
        centers = [_bbox_center(t.bbox) for t in result.tracks]

        for zone in self.zones:
            polygon = np.array(zone.polygon, dtype=np.float32)
            count = sum(
                cv2.pointPolygonTest(polygon, (float(cx), float(cy)), False) >= 0
                for cx, cy in centers
            )
            counts[zone.zone_id] = count

            # Zone density is people per m², scaled so ZONE_CRITICAL_PEOPLE_PER_SQM maps to 1.0.
            people_per_sqm = count / zone.area_sqm if zone.area_sqm > 0 else 0.0
            score = float(np.clip(people_per_sqm / ZONE_CRITICAL_PEOPLE_PER_SQM, 0.0, 1.0))
            density[zone.zone_id] = self._density_level(score)
        return counts, density

    def _detect_bottleneck(
        self, zone_counts: dict[str, int], zone_density: dict[str, str]
    ) -> tuple[bool, list[str]]:
        zones = [
            zone_id
            for zone_id, count in zone_counts.items()
            if count > BOTTLENECK_MIN_COUNT and zone_density[zone_id] in ("high", "critical")
        ]
        return bool(zones), zones


class HeatmapAccumulator:
    def __init__(
        self,
        frame_width: int = 640,
        frame_height: int = 480,
        decay: float = 0.995,
    ) -> None:
        self._map = np.zeros((frame_height, frame_width), dtype=np.float32)
        self.decay = decay

    def update(self, result: PerceptionResult) -> None:
        self._map *= self.decay  # fade old heat
        for track in result.tracks:
            cx, cy = _bbox_center(track.bbox)
            blob = np.zeros_like(self._map)
            # Filled circle approximates a gaussian blob.
            cv2.circle(blob, (int(cx), int(cy)), 20, 1.0, thickness=-1)
            self._map += blob

    def get_overlay(self, frame: np.ndarray) -> np.ndarray:
        peak = float(self._map.max())
        normalised = (
            (self._map / peak * 255).astype(np.uint8) if peak > 0 else np.zeros_like(self._map, np.uint8)
        )
        heatmap = cv2.applyColorMap(normalised, cv2.COLORMAP_JET)
        if heatmap.shape[:2] != frame.shape[:2]:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        return cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)

    def reset(self) -> None:
        self._map[:] = 0
