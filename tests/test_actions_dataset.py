import os
from pathlib import Path

import pytest

ROOT = Path(os.environ.get("RWF2000_ROOT", "~/Downloads/RWF-2000")).expanduser()

requires_dataset = pytest.mark.skipif(not ROOT.is_dir(), reason=f"RWF-2000 not found at {ROOT}")


def test_dataset_imports_cleanly() -> None:
    from drishti.actions.dataset import RWF2000Dataset

    assert RWF2000Dataset is not None


@pytest.fixture(scope="module")
def train_set():
    from drishti.actions.dataset import RWF2000Dataset

    return RWF2000Dataset(str(ROOT), split="train")


@pytest.fixture(scope="module")
def val_set():
    from drishti.actions.dataset import RWF2000Dataset

    return RWF2000Dataset(str(ROOT), split="val")


@requires_dataset
def test_train_split_has_samples(train_set) -> None:
    assert len(train_set) > 0


@requires_dataset
def test_val_split_has_samples(val_set) -> None:
    assert len(val_set) > 0


@requires_dataset
def test_getitem_returns_correct_shape_and_label(train_set) -> None:
    video, label = train_set[0]
    assert tuple(video.shape) == (3, 16, 224, 224)
    assert label in (0, 1)
    assert video.abs().sum() > 0, "got the all-zero fallback tensor: the clip failed to decode"


@requires_dataset
def test_class_counts_has_both_classes(train_set) -> None:
    counts = train_set.class_counts()
    assert set(counts) == {"Fight", "NonFight"}
    assert counts["Fight"] > 0 and counts["NonFight"] > 0
