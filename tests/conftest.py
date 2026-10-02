from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mri_pipeline.synthetic import make_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """6 images per class + 1 exact duplicate in yes/, mixed sizes and
    extensions (.jpg .JPG .jpeg .png), plus a non-image file in no/."""
    out = tmp_path_factory.mktemp("synthetic") / "brain_tumor_dataset"
    make_dataset(out, n_per_class=6, seed=0)
    return out


@pytest.fixture(scope="session")
def larger_synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("synthetic_large") / "brain_tumor_dataset"
    make_dataset(out, n_per_class=10, seed=1)
    return out


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
