"""Tests for the classical baseline classifiers."""

from __future__ import annotations

import numpy as np
import pytest

from encoding_atlas.benchmark.baselines import (
    CLASSICAL_BASELINE_NAMES,
    get_classical_baseline,
    run_baseline_single_fold,
)


@pytest.fixture
def split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 1, (20, 2)), rng.normal(4, 1, (20, 2))])
    y = np.array([0] * 20 + [1] * 20, dtype=np.intp)
    idx = rng.permutation(40)
    X, y = X[idx], y[idx]
    return X[:30], X[30:], y[:30], y[30:]


@pytest.mark.parametrize("name", CLASSICAL_BASELINE_NAMES)
def test_baseline_runs(
    name: str, split: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> None:
    X_train, X_test, y_train, y_test = split
    result = run_baseline_single_fold(name, X_train, X_test, y_train, y_test, seed=0)
    assert result["status"] == "success"
    assert result["test_accuracy"] >= 0.8  # clearly separable clusters


def test_unknown_baseline_raises() -> None:
    with pytest.raises(ValueError, match="Unknown baseline"):
        get_classical_baseline("not_a_model", seed=0)


def test_get_baseline_is_sklearn_like() -> None:
    clf = get_classical_baseline("random_forest", seed=0)
    assert hasattr(clf, "fit") and hasattr(clf, "predict")
