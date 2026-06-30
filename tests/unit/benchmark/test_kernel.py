"""Tests for the quantum-kernel evaluator (fast; small statevector sims)."""

from __future__ import annotations

import numpy as np
import pytest

from encoding_atlas import AngleEncoding
from encoding_atlas.benchmark.kernel import (
    QuantumKernelClassifier,
    centered_kernel_target_alignment,
    compute_kernel_entry,
    compute_kernel_matrix,
    compute_kernel_matrix_cross,
    ensure_psd,
    kernel_target_alignment,
    run_kernel_single_fold,
)


@pytest.fixture
def encoding() -> AngleEncoding:
    return AngleEncoding(n_features=2, rotation="Y")


@pytest.fixture
def xor_like() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    a = rng.normal(0.4, 0.15, (10, 2))
    b = rng.normal(2.6, 0.15, (10, 2))
    X = np.vstack([a, b])
    y = np.array([0] * 10 + [1] * 10, dtype=np.intp)
    return X, y


class TestKernelMatrix:
    def test_properties(self, encoding: AngleEncoding) -> None:
        X = np.array([[0.1, 0.2], [1.0, 1.1], [2.0, 2.1], [3.0, 3.0]])
        K = compute_kernel_matrix(encoding, X)
        assert K.shape == (4, 4)
        assert np.allclose(K, K.T)  # symmetric
        assert np.allclose(np.diag(K), 1.0)  # self-fidelity
        assert K.min() >= 0.0 and K.max() <= 1.0 + 1e-9

    def test_return_states(self, encoding: AngleEncoding) -> None:
        X = np.array([[0.1, 0.2], [1.0, 1.1]])
        K, states = compute_kernel_matrix(encoding, X, return_states=True)
        assert len(states) == 2
        assert K.shape == (2, 2)

    def test_cross_kernel_shape(self, encoding: AngleEncoding) -> None:
        X_train = np.array([[0.1, 0.2], [1.0, 1.1], [2.0, 2.0]])
        X_test = np.array([[0.5, 0.5], [1.5, 1.5]])
        K = compute_kernel_matrix_cross(encoding, X_train, X_test)
        assert K.shape == (2, 3)

    def test_kernel_entry_identical_states(self, encoding: AngleEncoding) -> None:
        from encoding_atlas.benchmark.kernel import simulate_encoding_state

        s = simulate_encoding_state(encoding, np.array([0.3, 0.7]))
        assert compute_kernel_entry(s, s) == pytest.approx(1.0)


class TestEnsurePsd:
    def test_already_psd_unchanged(self) -> None:
        K = np.array([[1.0, 0.5], [0.5, 1.0]])
        K_psd, modified = ensure_psd(K)
        assert not modified
        assert np.allclose(K_psd, K)

    def test_indefinite_is_regularized(self) -> None:
        K = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues 3, -1
        K_psd, modified = ensure_psd(K)
        assert modified
        assert np.linalg.eigvalsh(K_psd).min() > -1e-7


class TestKernelTargetAlignment:
    def test_perfectly_aligned_kernel(self) -> None:
        y = np.array([0, 0, 1, 1])
        y_signed = 2.0 * y - 1.0
        K = np.outer(y_signed, y_signed).astype(float)
        # Uncentered alignment of the ideal kernel is 1.
        assert kernel_target_alignment(K, y) == pytest.approx(1.0)

    def test_alignment_in_range(
        self, encoding: AngleEncoding, xor_like: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = xor_like
        K = compute_kernel_matrix(encoding, X)
        for fn in (kernel_target_alignment, centered_kernel_target_alignment):
            assert -1.0 <= fn(K, y) <= 1.0

    def test_centered_handles_tiny(self) -> None:
        assert centered_kernel_target_alignment(np.array([[1.0]]), np.array([1])) == 0.0


class TestQuantumKernelClassifier:
    def test_fit_predict_separable(
        self, encoding: AngleEncoding, xor_like: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = xor_like
        clf = QuantumKernelClassifier(encoding, seed=0).fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (len(y),)
        assert set(np.unique(preds)).issubset({0, 1})
        assert clf.score(X, y) >= 0.75  # clearly separable clusters

    def test_predict_before_fit_raises(self, encoding: AngleEncoding) -> None:
        with pytest.raises(ValueError, match="not fitted"):
            QuantumKernelClassifier(encoding).predict(np.array([[0.1, 0.2]]))

    def test_invalid_C(self, encoding: AngleEncoding) -> None:
        with pytest.raises(ValueError, match="C must be positive"):
            QuantumKernelClassifier(encoding, C=0)


class TestRunKernelSingleFold:
    def test_returns_metrics(
        self, encoding: AngleEncoding, xor_like: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = xor_like
        result = run_kernel_single_fold(encoding, X[:14], X[14:], y[:14], y[14:])
        assert result["status"] == "success"
        assert 0.0 <= result["test_accuracy"] <= 1.0
        assert -1.0 <= result["kernel_target_alignment"] <= 1.0
        assert isinstance(result["kernel_regularized"], bool)
