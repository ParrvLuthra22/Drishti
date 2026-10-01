import numpy as np

from drishti.perception.detector import Detection
from drishti.perception.tracker import Track, Tracker

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def make_detection(x_offset: float = 0.0) -> Detection:
    return Detection(
        bbox=(100.0 + x_offset, 100.0, 200.0 + x_offset, 300.0),
        confidence=0.9,
        class_id=0,
        class_name="person",
    )


def test_tracker_initialises() -> None:
    assert Tracker() is not None


def test_update_with_no_detections_returns_empty_list() -> None:
    assert Tracker().update([], FRAME) == []


def test_track_id_persists_across_frames() -> None:
    tracker = Tracker()
    track_ids = []
    for frame_idx in range(5):
        tracks = tracker.update([make_detection(10.0 * frame_idx)], FRAME)
        assert len(tracks) == 1
        assert isinstance(tracks[0], Track)
        track_ids.append(tracks[0].track_id)

    assert len(set(track_ids)) == 1
    assert tracks[0].age == 5
    assert tracks[0].class_name == "person"


def test_reset_clears_state() -> None:
    tracker = Tracker()
    for frame_idx in range(3):
        tracker.update([make_detection(10.0 * frame_idx)], FRAME)
    tracker.reset()
    tracks = tracker.update([make_detection()], FRAME)
    assert tracks[0].age == 1
