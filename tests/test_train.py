from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from mri_pipeline import train


@st.composite
def duplicate_patterns(draw):
    """Random labels with random duplicate groups (a group never spans classes,
    like the real data)."""
    n_groups = draw(st.integers(min_value=10, max_value=40))
    sizes = draw(st.lists(st.integers(1, 4), min_size=n_groups, max_size=n_groups))
    labels = draw(st.lists(st.integers(0, 1), min_size=n_groups, max_size=n_groups))
    if len(set(labels)) < 2:
        labels[0], labels[1] = 0, 1
    for cls in (0, 1):
        if sum(1 for lab in labels if lab == cls) < 5:
            labels = [i % 2 for i in range(n_groups)]
    groups = np.repeat(np.arange(n_groups), sizes)
    y = np.repeat(np.array(labels), sizes)
    perm = np.random.default_rng(draw(st.integers(0, 10_000))).permutation(len(y))
    return y[perm], groups[perm]


@settings(max_examples=60, deadline=None)
@given(duplicate_patterns(), st.integers(0, 2**31 - 1))
def test_property_grouped_cv_never_leaks(pattern, seed):
    y, groups = pattern
    splits = train.make_splits(y, groups, seed, n_splits=5, scheme="grouped")
    assert len(splits) == 5
    assert train.leaked_groups(splits, groups) == 0
    covered = np.sort(np.concatenate([test for _, test in splits]))
    np.testing.assert_array_equal(covered, np.arange(len(y)))  # every row tested once


def _labels_and_groups_with_duplicates(n_unique=40, copies=2):
    """Every file appears ``copies`` times; a group is one file hash."""
    groups = np.repeat(np.arange(n_unique), copies)
    y = groups % 2
    return y, groups


def test_grouped_cv_keeps_duplicate_files_on_one_side():
    y, groups = _labels_and_groups_with_duplicates()
    splits = train.make_splits(y, groups, seed=0, scheme="grouped")
    assert train.leaked_groups(splits, groups) == 0


def test_leaky_scheme_does_leak():
    y, groups = _labels_and_groups_with_duplicates()
    splits = train.make_splits(y, groups, seed=0, scheme="leaky")
    assert train.leaked_groups(splits, groups) > 0


def test_unknown_scheme_and_model():
    with pytest.raises(ValueError, match="scheme"):
        train.make_splits(np.array([0, 1]), np.array([0, 1]), 0, scheme="random")
    with pytest.raises(ValueError, match="unknown model"):
        train.make_model("xgboost", 0)


def test_models_build_and_scaler_inside_pipeline():
    for name in train.MODEL_NAMES:
        model = train.make_model(name, seed=3)
        assert hasattr(model, "fit")
    assert train.make_model("svm", 0).steps[0][0] == "standardscaler"


def test_score_metrics():
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 0])
    s = train.score(y_true, y_pred)
    assert s["accuracy"] == 0.75
    assert s["tumour_recall"] == 0.5
    assert s["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_cross_validate_rows_and_seed_reproducible():
    rng = np.random.default_rng(0)
    y = np.array([0, 1] * 20)
    X = rng.normal(size=(40, 5)) + y[:, None]
    groups = np.arange(40) // 2
    a = train.cross_validate(X, y, groups, "rf", seed=7)
    b = train.cross_validate(X, y, groups, "rf", seed=7)
    assert a == b
    assert len(a) == 5 and {r["cv"] for r in a} == {"grouped"}
    assert sum(r["n_test"] for r in a) == 40
