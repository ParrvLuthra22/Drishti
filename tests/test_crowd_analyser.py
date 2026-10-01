import numpy as np

from drishti.crowd.analyser import CrowdAnalyser, HeatmapAccumulator, ZoneConfig
from drishti.perception.tracker import Track
from drishti.perception.worker import PerceptionResult

FRAME_SHAPE = (480, 640, 3)


def make_result(positions: list[tuple[float, float]], frame_id: int = 0) -> PerceptionResult:
    """Fake result with one 20x20 person track centred on each position; ids follow list order."""
    tracks = [
        Track(
            track_id=i,
            bbox=(x - 10, y - 10, x + 10, y + 10),
            confidence=0.9,
            class_id=0,
            class_name="person",
            age=frame_id + 1,
        )
        for i, (x, y) in enumerate(positions)
    ]
    return PerceptionResult(
        camera_id="cam01", frame_id=frame_id, tracks=tracks, fps=15.0, timestamp=1000.0 + frame_id
    )


def grid(n: int) -> list[tuple[float, float]]:
    return [(30.0 + 40 * (i % 10), 30.0 + 40 * (i // 10)) for i in range(n)]


def square_zone(zone_id: str = "gate_a", area_sqm: float = 2.0) -> ZoneConfig:
    return ZoneConfig(
        zone_id=zone_id,
        polygon=[(100, 100), (300, 100), (300, 300), (100, 300)],
        area_sqm=area_sqm,
        name="Gate A",
    )


def test_zero_tracks_is_low_density() -> None:
    metrics = CrowdAnalyser().update(make_result([]))
    assert metrics.density_level == "low"
    assert metrics.density_score == 0.0
    assert metrics.total_tracks == 0


def test_25_tracks_is_critical_density() -> None:
    metrics = CrowdAnalyser().update(make_result(grid(25)))
    assert metrics.density_level == "critical"
    assert metrics.density_score == 1.0
    assert metrics.total_tracks == 25


def test_stationary_tracks_are_calm() -> None:
    analyser = CrowdAnalyser()
    positions = grid(5)
    for frame_id in range(5):
        metrics = analyser.update(make_result(positions, frame_id))
    assert metrics.flow_direction == "calm"
    assert metrics.avg_speed == 0.0


def test_zone_counts_tracks_inside_polygon() -> None:
    analyser = CrowdAnalyser(zones=[square_zone()])
    inside = [(150.0, 150.0), (200.0, 200.0), (250.0, 250.0)]
    outside = [(10.0, 10.0), (500.0, 400.0)]
    metrics = analyser.update(make_result(inside + outside))
    assert metrics.zone_counts == {"gate_a": 3}
    assert not metrics.bottleneck_detected


def test_bottleneck_when_zone_is_crowded() -> None:
    analyser = CrowdAnalyser(zones=[square_zone(area_sqm=2.0)])  # 6 people / 2 m² = 3 per m²
    positions = [(120.0 + 20 * i, 200.0) for i in range(6)]
    metrics = analyser.update(make_result(positions))
    assert metrics.zone_counts["gate_a"] == 6
    assert metrics.zone_density["gate_a"] in ("high", "critical")
    assert metrics.bottleneck_detected is True
    assert metrics.bottleneck_zones == ["gate_a"]


def test_directional_flow() -> None:
    analyser = CrowdAnalyser()
    for frame_id in range(3):
        positions = [(x + 6.0 * frame_id, y) for x, y in grid(6)]
        metrics = analyser.update(make_result(positions, frame_id))
    assert metrics.flow_direction == "directional"
    assert metrics.flow_score < 0.3
    assert metrics.avg_speed > 2.0


def test_chaotic_flow() -> None:
    analyser = CrowdAnalyser()
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)] * 2
    base = grid(len(directions))
    for frame_id in range(3):
        positions = [
            (x + 6.0 * frame_id * dx, y + 6.0 * frame_id * dy)
            for (x, y), (dx, dy) in zip(base, directions)
        ]
        metrics = analyser.update(make_result(positions, frame_id))
    assert metrics.flow_direction == "chaotic"
    assert metrics.flow_score >= 0.3


def test_heatmap_overlay_shape_and_reset() -> None:
    heatmap = HeatmapAccumulator()
    frame = np.zeros(FRAME_SHAPE, dtype=np.uint8)
    heatmap.update(make_result([(320.0, 240.0)]))
    overlay = heatmap.get_overlay(frame)
    assert overlay.shape == frame.shape and overlay.dtype == np.uint8
    assert overlay.any()

    heatmap.reset()
    assert not heatmap._map.any()
