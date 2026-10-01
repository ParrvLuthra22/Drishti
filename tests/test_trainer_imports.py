import pytest
import torch

MODEL_NAME = "MCG-NJU/videomae-base"


def test_trainer_imports() -> None:
    from drishti.actions.trainer import VideoMAETrainer

    assert VideoMAETrainer is not None


def test_collate_produces_videomae_layout() -> None:
    from drishti.actions.trainer import collate_clips

    batch = [(torch.randn(3, 16, 224, 224), 1), (torch.randn(3, 16, 224, 224), 0)]
    pixel_values, labels = collate_clips(batch)
    assert pixel_values.shape == (2, 16, 3, 224, 224)  # (B, T, C, H, W)
    assert labels.tolist() == [1, 0] and labels.dtype == torch.long
    # Channels must have moved, not just been reshaped: clip 0, frame 5, channel 2 is the source channel 2.
    assert torch.equal(pixel_values[0, 5, 2], batch[0][0][2, 5])


def test_image_processor_loads() -> None:
    from transformers import VideoMAEImageProcessor

    try:
        processor = VideoMAEImageProcessor.from_pretrained(MODEL_NAME)
    except OSError as exc:
        pytest.skip(f"no network access to the Hugging Face Hub: {exc}")
    assert processor is not None


def test_model_has_two_class_head() -> None:
    from drishti.actions.trainer import VideoMAETrainer

    try:
        trainer = VideoMAETrainer(data_root="unused", device="cpu", model_name=MODEL_NAME)
    except OSError as exc:
        pytest.skip(f"no network access to the Hugging Face Hub: {exc}")
    assert hasattr(trainer.model, "classifier")
    assert trainer.model.classifier.out_features == 2
    assert trainer.model.config.id2label == {0: "NonFight", 1: "Fight"}
