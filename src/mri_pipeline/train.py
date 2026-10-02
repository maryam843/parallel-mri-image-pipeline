"""Models, duplicate-grouped cross-validation and baselines.

The dataset contains exact-duplicate files, so a plain random split can put a
copy on each side. Every split here is a ``StratifiedGroupKFold`` where the
group is the file's SHA-256, so identical files always stay on the same side,
and every source of randomness takes an explicit seed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

TUMOUR_LABEL = 1
CV_SCHEMES = ("grouped", "leaky")


def make_model(name: str, seed: int) -> ClassifierMixin | Pipeline:
    """Build an unfitted model. Scaling sits inside the pipeline, so it is fit
    on the training fold only."""
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
    if name == "svm":
        return make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=seed))
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    raise ValueError(f"unknown model {name!r}")


MODEL_NAMES = ("majority", "logreg", "svm", "rf")


def make_splits(
    y: np.ndarray, groups: np.ndarray, seed: int, n_splits: int = 5, scheme: str = "grouped"
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold splits. ``grouped`` keeps each duplicate group on one
    side; ``leaky`` ignores groups and exists only to measure the leakage."""
    x_dummy = np.zeros((len(y), 1))
    if scheme == "grouped":
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(x_dummy, y, groups))
    if scheme == "leaky":
        leaky = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(leaky.split(x_dummy, y))
    raise ValueError(f"scheme must be one of {CV_SCHEMES}, got {scheme!r}")


def leaked_groups(splits: list[tuple[np.ndarray, np.ndarray]], groups: np.ndarray) -> int:
    """Number of (fold, group) pairs where a group appears in train and test."""
    total = 0
    for train_idx, test_idx in splits:
        total += len(set(groups[train_idx]) & set(groups[test_idx]))
    return total


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "tumour_recall": float(
            recall_score(y_true, y_pred, pos_label=TUMOUR_LABEL, zero_division=0)
        ),
    }


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    model_name: str,
    seed: int,
    n_splits: int = 5,
    scheme: str = "grouped",
    model_factory: Callable[[str, int], ClassifierMixin | Pipeline] = make_model,
) -> list[dict[str, float | int | str]]:
    """One row per fold with accuracy, macro F1 and tumour recall."""
    rows: list[dict[str, float | int | str]] = []
    splits = make_splits(y, groups, seed, n_splits, scheme)
    for fold, (train_idx, test_idx) in enumerate(splits):
        model = model_factory(model_name, seed)
        model.fit(X[train_idx], y[train_idx])
        metrics = score(y[test_idx], model.predict(X[test_idx]))
        rows.append(
            {
                "model": model_name,
                "seed": seed,
                "fold": fold,
                "cv": scheme,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                **metrics,
            }
        )
    return rows
