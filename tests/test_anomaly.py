import numpy as np
import pytest
import torch

from drishti.anomaly.autoencoder import (
    AnomalyScorer,
    AnomalyTrainer,
    NormalSceneDataset,
    SceneAutoencoder,
)

REQUIRED_KEYS = {"raw_score", "smoothed_score", "threshold", "is_anomaly", "anomaly_level"}


class WhiteFrames(NormalSceneDataset):
    """50 all-white (1, 64, 64) frames, without touching a camera."""

    def __init__(self, count: int = 50) -> None:
        self.frames = [torch.ones(1, 64, 64) for _ in range(count)]


@pytest.fixture(scope="module")
def trained_on_white(tmp_path_factory) -> AnomalyTrainer:
    torch.manual_seed(0)
    trainer = AnomalyTrainer(
        device="cpu", epochs=30, batch_size=16, output_dir=str(tmp_path_factory.mktemp("anomaly"))
    )
    trainer.train(WhiteFrames())
    return trainer


def white_frame() -> np.ndarray:
    return np.full((480, 640, 3), 255, dtype=np.uint8)


def black_frame() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def test_autoencoder_forward_shape_and_range() -> None:
    model = SceneAutoencoder().eval()
    with torch.no_grad():
        out = model(torch.rand(2, 1, 64, 64))
    assert out.shape == (2, 1, 64, 64)
    assert out.min() >= 0.0 and out.max() <= 1.0  # Sigmoid output


def test_scorer_returns_expected_fields_on_black_frame() -> None:
    scorer = AnomalyScorer(SceneAutoencoder(), threshold=0.01, device="cpu")
    result = scorer.score_frame(black_frame())
    assert set(result) == REQUIRED_KEYS
    assert isinstance(result["is_anomaly"], bool)
    assert result["smoothed_score"] >= 0
    assert result["anomaly_level"] in {"normal", "suspicious", "anomaly"}


def test_low_error_on_seen_data(trained_on_white: AnomalyTrainer) -> None:
    scorer = AnomalyScorer(trained_on_white.model, trained_on_white.threshold, device="cpu")
    assert scorer.score_frame(white_frame())["raw_score"] < 0.1


def test_high_error_on_unseen_data(trained_on_white: AnomalyTrainer) -> None:
    scorer = AnomalyScorer(trained_on_white.model, trained_on_white.threshold, device="cpu")
    white_score = scorer.score_frame(white_frame())["raw_score"]
    black_score = scorer.score_frame(black_frame())["raw_score"]
    assert black_score > white_score


def test_anomaly_levels_follow_threshold() -> None:
    scorer = AnomalyScorer(SceneAutoencoder(), threshold=1.0, device="cpu", window=1)
    scorer.model.forward = lambda x: x + 0.0  # perfect reconstruction -> error 0

    assert scorer.score_frame(black_frame())["anomaly_level"] == "normal"

    for error, level, flagged in [(1.2, "suspicious", True), (1.5, "anomaly", True), (0.5, "normal", False)]:
        scorer.model.forward = lambda x, e=error: x + e**0.5  # constant offset -> MSE == error
        result = scorer.score_frame(black_frame())
        assert result["anomaly_level"] == level
        assert result["is_anomaly"] is flagged


def test_trainer_saves_and_reloads(trained_on_white: AnomalyTrainer) -> None:
    reloaded = AnomalyTrainer(device="cpu", output_dir=str(trained_on_white.output_dir))
    reloaded.load(str(trained_on_white.output_dir))
    assert reloaded.threshold == pytest.approx(trained_on_white.threshold)
    assert not reloaded.model.training
