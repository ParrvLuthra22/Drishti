import numpy as np
import pytest
import torch
from transformers import VideoMAEConfig, VideoMAEForVideoClassification

BGR_FRAME = np.random.default_rng(0).integers(0, 256, size=(480, 640, 3), dtype=np.uint8)


def tiny_model() -> VideoMAEForVideoClassification:
    """Randomly initialised VideoMAE small enough to run instantly, so logic tests need no download."""
    torch.manual_seed(0)
    config = VideoMAEConfig(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        num_labels=2,
        id2label={0: "NonFight", 1: "Fight"},
        label2id={"NonFight": 0, "Fight": 1},
    )
    return VideoMAEForVideoClassification(config)


def make_detector(**kwargs):
    from drishti.actions.fight_detector import FightDetector

    return FightDetector(device="cpu", model=tiny_model(), **kwargs)


def test_fight_detector_imports() -> None:
    from drishti.actions.fight_detector import FightDetector

    assert FightDetector is not None


def test_add_frame_accepts_480x640_bgr_frame() -> None:
    detector = make_detector()
    detector.add_frame(BGR_FRAME)
    assert len(detector._frame_buffer) == 1
    assert detector._frame_buffer[0].shape == (3, 224, 224)
    assert detector._frame_buffer[0].dtype == torch.float32


def test_preprocessing_is_imagenet_normalised_rgb() -> None:
    from drishti.actions.fight_detector import preprocess_frame

    blue_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    blue_bgr[..., 0] = 255  # pure blue in BGR order
    tensor = preprocess_frame(blue_bgr)
    # Channels are RGB after conversion: blue channel (2) is max, red and green are min.
    assert tensor[2].mean().item() == pytest.approx((1.0 - 0.406) / 0.225, abs=1e-4)
    assert tensor[0].mean().item() == pytest.approx((0.0 - 0.485) / 0.229, abs=1e-4)
    assert tensor[1].mean().item() == pytest.approx((0.0 - 0.456) / 0.224, abs=1e-4)


def test_predict_with_empty_buffer_is_not_ready() -> None:
    result = make_detector().predict()
    assert result == {"fight_probability": 0.0, "fight_detected": False, "ready": False}


def test_predict_with_a_partial_buffer_is_not_ready() -> None:
    detector = make_detector()
    for _ in range(15):
        detector.add_frame(BGR_FRAME)
    assert detector.predict()["ready"] is False


def test_predict_with_16_frames_returns_a_probability() -> None:
    detector = make_detector()
    for _ in range(16):
        detector.add_frame(BGR_FRAME)
    result = detector.predict()  # also proves the (batch, frames, channels, H, W) layout is accepted
    assert result["ready"] is True
    assert 0.0 <= result["fight_probability"] <= 1.0
    assert result["fight_detected"] == (result["fight_probability"] > 0.5)


def test_buffer_keeps_only_the_latest_16_frames() -> None:
    detector = make_detector()
    for _ in range(20):
        detector.add_frame(BGR_FRAME)
    assert len(detector._frame_buffer) == 16


def test_sample_interval_thins_the_frames() -> None:
    detector = make_detector(sample_interval_s=0.3)
    for t in (0.0, 0.1, 0.2, 0.31, 0.4, 0.62):
        detector.add_frame(BGR_FRAME, timestamp=t)
    assert len(detector._frame_buffer) == 3  # kept t = 0.0, 0.31, 0.62


def test_reset_clears_the_buffer() -> None:
    detector = make_detector()
    for _ in range(16):
        detector.add_frame(BGR_FRAME)
    detector.reset()
    assert detector.predict()["ready"] is False


@pytest.fixture(scope="module")
def hub_detector():
    from drishti.actions.fight_detector import FightDetector

    try:
        return FightDetector(device="cpu")  # downloads ~345 MB from the Hub on the first run
    except OSError as exc:
        pytest.skip(f"cannot reach the Hugging Face Hub: {exc}")


def test_real_model_predicts_with_16_frames(hub_detector) -> None:
    for _ in range(16):
        hub_detector.add_frame(BGR_FRAME)
    result = hub_detector.predict()
    assert result["ready"] is True
    assert 0.0 <= result["fight_probability"] <= 1.0
    assert hub_detector.model.classifier.out_features == 2
