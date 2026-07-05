"""Tests for kernel-geometry generalization diagnostics.

Matrix-level tests assert exact, verifiable invariants (``g(K, K) = 1``,
``d_eff(I_n, 1) = n/2``, ``KTA(yy^T, y) = 1``); encoding-level tests check the
metrics on real fidelity kernels and confirm they track the accuracy ranking
that expressibility does not.
"""

from __future__ import annotations

import numpy as np
import pytest

from encoding_atlas import AngleEncoding, IQPEncoding
from encoding_atlas.analysis.generalization import (
    centered_kernel_target_alignment,
    compute_effective_dimension,
    compute_fidelity_kernel,
    compute_geometric_difference,
    compute_kernel_target_alignment,
    geometric_difference,
    kernel_effective_dimension,
    kernel_target_alignment,
)


def _psd_matrix(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    return A @ A.T


@pytest.fixture
def separable() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.5, 0.2, (8, 2)), rng.normal(2.7, 0.2, (8, 2))])
    y = np.array([0] * 8 + [1] * 8, dtype=np.intp)
    return X, y


# =====================================================================
# Fidelity kernel
# =====================================================================


class TestFidelityKernel:
    def test_properties(self) -> None:
        enc = AngleEncoding(n_features=2, rotation="Y")
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 2 * np.pi, (10, 2))
        K = compute_fidelity_kernel(enc, X)
        assert K.shape == (10, 10)
        assert np.allclose(K, K.T)
        assert np.allclose(np.diag(K), 1.0)
        assert np.linalg.eigvalsh(K).min() > -1e-9  # PSD
        assert K.min() >= 0.0 and K.max() <= 1.0 + 1e-9


# =====================================================================
# Kernel-target alignment
# =====================================================================


class TestKernelTargetAlignment:
    def test_ideal_kernel_alignment_is_one(self) -> None:
        y = np.array([0, 0, 1, 1])
        ys = 2 * y - 1
        K_ideal = np.outer(ys, ys).astype(float)
        assert kernel_target_alignment(K_ideal, y) == pytest.approx(1.0)

    def test_anticorrelated_is_minus_one(self) -> None:
        y = np.array([0, 0, 1, 1])
        ys = 2 * y - 1
        assert kernel_target_alignment(
            -np.outer(ys, ys).astype(float), y
        ) == pytest.approx(-1.0)

    def test_centered_ideal_is_one(self) -> None:
        y = np.array([0, 0, 1, 1, 0, 1])
        ys = 2 * y - 1
        assert centered_kernel_target_alignment(
            np.outer(ys, ys).astype(float), y
        ) == pytest.approx(1.0)

    def test_single_class_centered_is_zero(self) -> None:
        # Centering the all-same label matrix yields the zero matrix, so the
        # centered alignment is 0 (no class structure to align to). The
        # uncentered variant lacks this property but must stay finite/in range.
        y = np.array([1, 1, 1, 1])
        K = _psd_matrix(4)
        assert centered_kernel_target_alignment(K, y) == 0.0
        assert -1.0 <= kernel_target_alignment(K, y) <= 1.0

    def test_centered_tiny_returns_zero(self) -> None:
        assert centered_kernel_target_alignment(np.array([[1.0]]), np.array([1])) == 0.0

    def test_range(self, separable: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = separable
        K = compute_fidelity_kernel(AngleEncoding(n_features=2), X)
        assert -1.0 <= kernel_target_alignment(K, y) <= 1.0
        assert -1.0 <= centered_kernel_target_alignment(K, y) <= 1.0


# =====================================================================
# Geometric difference
# =====================================================================


class TestGeometricDifference:
    def test_identical_kernels_is_one(self) -> None:
        K = _psd_matrix(8)
        assert geometric_difference(K, K, regularization=1e-10) == pytest.approx(
            1.0, abs=1e-4
        )

    def test_identity_kernels_is_one(self) -> None:
        identity = np.eye(6)
        assert geometric_difference(
            identity, identity, regularization=1e-10
        ) == pytest.approx(1.0, abs=1e-5)

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="shapes differ"):
            geometric_difference(np.eye(4), np.eye(5))

    def test_positive_and_finite(self) -> None:
        g = geometric_difference(_psd_matrix(8, 1), _psd_matrix(8, 2))
        assert np.isfinite(g) and g > 0.0

    def test_encoding_level_positive(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = separable
        g = compute_geometric_difference(AngleEncoding(n_features=2), X)
        assert np.isfinite(g) and g > 0.0

    def test_invalid_classical_kernel_raises(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = separable
        with pytest.raises(ValueError, match="rbf.*linear"):
            compute_geometric_difference(AngleEncoding(n_features=2), X, classical_kernel="poly")  # type: ignore[arg-type]


# =====================================================================
# Effective dimension
# =====================================================================


class TestEffectiveDimension:
    @pytest.mark.parametrize("n", [4, 10, 16])
    def test_identity_gives_half_n(self, n: int) -> None:
        # d_eff(I_n, lambda=1) = sum 1/(1+1) = n/2.
        assert kernel_effective_dimension(
            np.eye(n), regularization=1.0
        ) == pytest.approx(n / 2)

    def test_rank_one_is_small(self) -> None:
        # All-ones kernel has one eigenvalue n, rest 0 -> d_eff = n/(n+1) < 1.5.
        assert kernel_effective_dimension(np.ones((10, 10)), regularization=1.0) < 1.5

    def test_in_zero_to_n(self) -> None:
        K = _psd_matrix(12)
        d = kernel_effective_dimension(K)
        assert 0.0 <= d <= 12.0

    def test_larger_regularization_lowers_dimension(self) -> None:
        K = _psd_matrix(8)
        assert kernel_effective_dimension(
            K, regularization=5.0
        ) < kernel_effective_dimension(K, regularization=0.1)

    def test_invalid_regularization_raises(self) -> None:
        with pytest.raises(ValueError, match="regularization must be positive"):
            kernel_effective_dimension(np.eye(4), regularization=0.0)

    def test_encoding_level_range(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = separable
        d = compute_effective_dimension(AngleEncoding(n_features=2), X)
        assert 0.0 < d <= len(X)


# =====================================================================
# The payoff: metrics track accuracy where expressibility does not
# =====================================================================


class TestExplainsAccuracyRanking:
    def test_alignment_orders_angle_above_iqp(
        self, separable: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Centered KTA reproduces the benchmark ranking (angle #1 > IQP #16),
        which expressibility inverts. Deterministic: seeded data, exact kernels."""
        X, y = separable
        angle_kta = compute_kernel_target_alignment(AngleEncoding(n_features=2), X, y)
        iqp_kta = compute_kernel_target_alignment(
            IQPEncoding(n_features=2, reps=2), X, y
        )
        assert angle_kta > iqp_kta
