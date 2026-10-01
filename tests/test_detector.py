import dataclasses
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from drishti.perception.detector import Detection, Detector

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_URL = "https://ultralytics.com/images/bus.jpg"  # contains several people
SAMPLE_PATH = ROOT / "data" / "test" / "bus.jpg"


@pytest.fixture(scope="module")
def detector() -> Detector:
    return Detector()


@pytest.fixture(scope="module")
def sample_frame() -> np.ndarray:
    if not SAMPLE_PATH.exists():
        SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            urllib.request.urlretrieve(SAMPLE_URL, SAMPLE_PATH)
        except OSError as exc:
            pytest.skip(f"could not download sample image: {exc}")
    frame = cv2.imread(str(SAMPLE_PATH))
    assert frame is not None
    return frame


def test_detector_loads(detector: Detector) -> None:
    assert detector.model is not None


def test_detect_black_frame_returns_list(detector: Detector) -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    assert isinstance(detector.detect(frame), list)


def test_detections_have_correct_fields(detector: Detector, sample_frame: np.ndarray) -> None:
    detections = detector.detect(sample_frame)
    assert len(detections) > 0, "expected at least one person in the sample image"

    expected = {"track_id", "bbox", "confidence", "class_id", "class_name"}
    for det in detections:
        assert isinstance(det, Detection)
        assert {f.name for f in dataclasses.fields(det)} == expected
        assert det.track_id == -1
        assert len(det.bbox) == 4 and all(isinstance(v, float) for v in det.bbox)
        x1, y1, x2, y2 = det.bbox
        assert x1 < x2 and y1 < y2
        assert 0.5 <= det.confidence <= 1.0
        assert det.class_id == 0
        assert det.class_name == "person"
