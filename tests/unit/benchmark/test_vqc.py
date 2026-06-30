"""Tests for the VQC evaluator (fast; tiny circuits / few epochs)."""

from __future__ import annotations

import numpy as np
import pytest

from encoding_atlas import AngleEncoding
from encoding_atlas.benchmark.vqc import VQCClassifier, run_vqc_single_fold


@pytest.fixture
def encoding() -> AngleEncoding:
    return AngleEncoding(n_features=2, rotation="Y")


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.5, 0.2, (8, 2)), rng.normal(2.5, 0.2, (8, 2))])
    y = np.array([0] * 8 + [1] * 8, dtype=np.intp)
    return X, y


class TestVQCConstruction:
    @pytest.mark.parametrize(
        "kwargs,msg",
        [
            ({"n_var_layers": 0}, "n_var_layers"),
            ({"lr": 0}, "lr"),
            ({"epochs": 0}, "epochs"),
        ],
    )
    def test_invalid_args(
        self, encoding: AngleEncoding, kwargs: dict, msg: str
    ) -> None:
        with pytest.raises(ValueError, match=msg):
            VQCClassifier(encoding, **kwargs)


class TestVQCTraining:
    def test_fit_sets_params_and_status(
        self, encoding: AngleEncoding, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = data
        vqc = VQCClassifier(encoding, n_var_layers=2, epochs=4, seed=0).fit(X, y)
        assert vqc.params_ is not None
        assert vqc.params_.shape == (2, encoding.n_qubits)
        assert vqc.status_ in {"success", "diverged"}
        assert len(vqc.loss_history_) >= 1

    def test_predict_shapes_and_labels(
        self, encoding: AngleEncoding, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = data
        vqc = VQCClassifier(encoding, epochs=4, seed=0).fit(X, y)
        proba = vqc.predict_proba(X)
        assert proba.shape == (len(y), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        preds = vqc.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})
        assert 0.0 <= vqc.score(X, y) <= 1.0

    def test_predict_before_fit_raises(self, encoding: AngleEncoding) -> None:
        with pytest.raises(ValueError, match="not fitted"):
            VQCClassifier(encoding).predict(np.array([[0.1, 0.2]]))

    def test_determinism_same_seed(
        self, encoding: AngleEncoding, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = data
        a = VQCClassifier(encoding, epochs=4, seed=7).fit(X, y).predict(X)
        b = VQCClassifier(encoding, epochs=4, seed=7).fit(X, y).predict(X)
        assert np.array_equal(a, b)


class TestRunVQCSingleFold:
    def test_returns_metrics(
        self, encoding: AngleEncoding, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = data
        result = run_vqc_single_fold(
            encoding, X[:12], X[12:], y[:12], y[12:], epochs=4, seed=0
        )
        assert result["status"] in {"success", "diverged"}
        for key in ("train_accuracy", "test_accuracy", "precision", "recall", "f1"):
            assert 0.0 <= result[key] <= 1.0

    @pytest.mark.slow
    def test_learns_separable_data(
        self, encoding: AngleEncoding, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """With enough epochs the VQC separates well-separated clusters."""
        X, y = data
        vqc = VQCClassifier(encoding, n_var_layers=2, epochs=40, lr=0.1, seed=0)
        vqc.fit(X, y)
        assert vqc.score(X, y) >= 0.75
