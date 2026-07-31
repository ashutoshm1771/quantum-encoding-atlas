"""Tests for fidelity-kernel concentration analysis.

The load-bearing tests are the closed-form ones. Angle encoding with inputs
drawn uniformly from ``[0, 2*pi)`` has an exactly solvable kernel: each qubit
contributes ``cos^2((x - x') / 2)`` with ``E = 1/2`` and ``E[.^2] = 3/8``, so
for ``n`` independent qubits

    E[K] = 2^-n ,    Var[K] = (3/8)^n - (1/4)^n .

That pins both the measurement and the Haar normalisation to arithmetic rather
than to a previously recorded value, and it exhibits the degeneracy that
motivates the whole design: the *mean* of this product ensemble equals the Haar
mean ``2^-n`` exactly, so only the variance can separate a structured encoding
from a scrambling one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from encoding_atlas import (
    AmplitudeEncoding,
    AngleEncoding,
    IQPEncoding,
    SO2EquivariantFeatureMap,
    ZZFeatureMap,
)
from encoding_atlas.analysis.concentration import (
    CONCENTRATION_THRESHOLD,
    ConcentrationResult,
    ScalingResult,
    _shots_to_resolve,
    compute_kernel_concentration,
    estimate_concentration_scaling,
    haar_kernel_moments,
)

# Sample count for statistical assertions. 150 inputs give 11,175 off-diagonal
# pairs, enough for the closed-form checks to hold at the tolerances below
# while keeping the suite fast.
N_STAT = 150


def _angle_theory(n_qubits: int) -> tuple[float, float]:
    """Exact ``(mean, variance)`` of the angle-encoding kernel on U[0, 2*pi)."""
    return 2.0**-n_qubits, (3.0 / 8.0) ** n_qubits - (1.0 / 4.0) ** n_qubits


# =====================================================================
# Haar reference moments
# =====================================================================


class TestHaarMoments:
    @pytest.mark.parametrize("n", [1, 2, 4, 8])
    def test_mean_is_inverse_dimension(self, n: int) -> None:
        mean, _ = haar_kernel_moments(n)
        assert mean == pytest.approx(1.0 / 2**n)

    @pytest.mark.parametrize("n", [1, 2, 4, 8])
    def test_variance_matches_beta_distribution(self, n: int) -> None:
        # |<psi|phi>|^2 ~ Beta(1, d-1) has variance (d-1)/(d^2 (d+1)).
        d = 2**n
        _, variance = haar_kernel_moments(n)
        assert variance == pytest.approx((d - 1) / (d**2 * (d + 1)))

    def test_single_qubit_is_exact(self) -> None:
        # d = 2: Beta(1, 1) is uniform on [0, 1] -> mean 1/2, variance 1/12.
        mean, variance = haar_kernel_moments(1)
        assert mean == pytest.approx(0.5)
        assert variance == pytest.approx(1.0 / 12.0)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True, "4"])
    def test_invalid_qubit_count_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            haar_kernel_moments(bad)  # type: ignore[arg-type]


# =====================================================================
# Closed-form validation of the measurement itself
# =====================================================================


class TestClosedForm:
    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_mean_matches_theory(self, n: int) -> None:
        result = compute_kernel_concentration(
            AngleEncoding(n_features=n, rotation="Y"), n_samples=N_STAT, seed=0
        )
        expected_mean, _ = _angle_theory(n)
        assert result.offdiagonal_mean == pytest.approx(expected_mean, rel=0.15)

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_variance_matches_theory(self, n: int) -> None:
        result = compute_kernel_concentration(
            AngleEncoding(n_features=n, rotation="Y"), n_samples=N_STAT, seed=0
        )
        _, expected_variance = _angle_theory(n)
        assert result.offdiagonal_variance == pytest.approx(expected_variance, rel=0.25)

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_mean_ratio_is_degenerate_at_one(self, n: int) -> None:
        """The reason the mean cannot be the order parameter.

        A product of independent uniformly-random single-qubit states has mean
        overlap exactly ``2^-n`` — identical to Haar — even though its variance
        is exponentially larger. Any mean-based measure would call angle
        encoding maximally concentrated, which is wrong.
        """
        result = compute_kernel_concentration(
            AngleEncoding(n_features=n, rotation="Y"), n_samples=N_STAT, seed=0
        )
        assert result.mean_ratio == pytest.approx(1.0, rel=0.15)

    @pytest.mark.parametrize("n", [2, 4, 6])
    def test_concentration_ratio_matches_theory(self, n: int) -> None:
        """...while the variance ratio correctly reports structure."""
        result = compute_kernel_concentration(
            AngleEncoding(n_features=n, rotation="Y"), n_samples=N_STAT, seed=0
        )
        _, theory_variance = _angle_theory(n)
        _, haar_variance = haar_kernel_moments(n)
        assert result.concentration_ratio == pytest.approx(
            theory_variance / haar_variance, rel=0.25
        )
        # Grows away from the floor: (3/8)/(1/4) = 1.5 per qubit.
        assert result.concentration_ratio > 1.5


# =====================================================================
# Single-width measurement
# =====================================================================


class TestComputeKernelConcentration:
    def test_result_fields_are_consistent(self) -> None:
        result = compute_kernel_concentration(
            AngleEncoding(n_features=3), n_samples=20, seed=0
        )
        assert isinstance(result, ConcentrationResult)
        assert result.n_qubits == 3
        assert result.n_samples == 20
        assert 0.0 <= result.offdiagonal_mean <= 1.0
        assert result.offdiagonal_min <= result.offdiagonal_mean
        assert result.offdiagonal_mean <= result.offdiagonal_max
        assert result.offdiagonal_std == pytest.approx(
            math.sqrt(result.offdiagonal_variance)
        )
        assert result.threshold == CONCENTRATION_THRESHOLD
        assert result.is_concentrated == (
            result.concentration_ratio < CONCENTRATION_THRESHOLD
        )

    def test_haar_fields_match_helper(self) -> None:
        result = compute_kernel_concentration(
            AngleEncoding(n_features=4), n_samples=20, seed=0
        )
        assert (result.haar_mean, result.haar_variance) == haar_kernel_moments(4)

    def test_ratio_is_variance_over_haar_variance(self) -> None:
        result = compute_kernel_concentration(
            AngleEncoding(n_features=4), n_samples=20, seed=0
        )
        assert result.concentration_ratio == pytest.approx(
            result.offdiagonal_variance / result.haar_variance
        )

    def test_entangling_is_more_concentrated_than_non_entangling(self) -> None:
        """The headline discrimination, at a width where it is unambiguous."""
        angle = compute_kernel_concentration(
            AngleEncoding(n_features=6), n_samples=N_STAT, seed=0
        )
        iqp = compute_kernel_concentration(
            IQPEncoding(n_features=6, reps=2), n_samples=N_STAT, seed=0
        )
        assert iqp.concentration_ratio < angle.concentration_ratio
        assert iqp.is_concentrated
        assert not angle.is_concentrated

    def test_orthogonal_states_give_identity_kernel(self) -> None:
        """Orthogonal basis states give K = I: no spread, unbounded shot cost.

        The entries are zero only up to floating point, so the budget is
        astronomically large rather than literally infinite; the exact-zero
        branch is covered in :class:`TestShotFormulaEdgeCases`.
        """
        # Basis states |00>, |01>, |10>, |11> via distinct one-hot rotations.
        X = np.array([[0.0, 0.0], [0.0, np.pi], [np.pi, 0.0], [np.pi, np.pi]])
        result = compute_kernel_concentration(
            AngleEncoding(n_features=2, rotation="Y"), X
        )
        assert result.offdiagonal_mean == pytest.approx(0.0, abs=1e-12)
        assert result.identity_distance == pytest.approx(0.0, abs=1e-12)
        assert result.shots_per_entry > 1e20

    def test_identity_distance_is_rms_offdiagonal(self) -> None:
        from encoding_atlas.analysis.generalization import compute_fidelity_kernel

        enc = AngleEncoding(n_features=3)
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 2 * np.pi, (12, 3))
        result = compute_kernel_concentration(enc, X)
        K = compute_fidelity_kernel(enc, X)
        n = len(X)
        expected = np.linalg.norm(K - np.eye(n), "fro") / math.sqrt(n * (n - 1))
        assert result.identity_distance == pytest.approx(expected)

    def test_supplied_data_is_used(self) -> None:
        enc = AngleEncoding(n_features=2)
        rng = np.random.default_rng(3)
        X = rng.uniform(0, 2 * np.pi, (17, 2))
        result = compute_kernel_concentration(enc, X, n_samples=999)
        assert result.n_samples == 17  # n_samples ignored when X is given

    def test_determinism_same_seed(self) -> None:
        enc = IQPEncoding(n_features=3, reps=2)
        a = compute_kernel_concentration(enc, n_samples=20, seed=11)
        b = compute_kernel_concentration(enc, n_samples=20, seed=11)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        enc = AngleEncoding(n_features=3)
        a = compute_kernel_concentration(enc, n_samples=20, seed=1)
        b = compute_kernel_concentration(enc, n_samples=20, seed=2)
        assert a.offdiagonal_mean != b.offdiagonal_mean

    def test_sobol_sampling_supported(self) -> None:
        result = compute_kernel_concentration(
            AngleEncoding(n_features=2), n_samples=16, sampling="sobol", seed=0
        )
        assert math.isfinite(result.concentration_ratio)

    def test_narrower_input_range_reduces_concentration(self) -> None:
        """Concentration is a joint property of circuit and input distribution.

        Restricting angle encoding's inputs to [0, pi/2] keeps the encoded
        states in a narrow cone, so the kernel stays far from the Haar floor.
        """
        enc = AngleEncoding(n_features=4, rotation="Y")
        wide = compute_kernel_concentration(
            enc, n_samples=N_STAT, input_range=(0.0, 2 * math.pi), seed=0
        )
        narrow = compute_kernel_concentration(
            enc, n_samples=N_STAT, input_range=(0.0, math.pi / 2), seed=0
        )
        assert narrow.concentration_ratio > wide.concentration_ratio

    def test_custom_threshold_changes_flag(self) -> None:
        enc = AngleEncoding(n_features=4)
        strict = compute_kernel_concentration(enc, n_samples=40, seed=0, threshold=1e6)
        lax = compute_kernel_concentration(enc, n_samples=40, seed=0, threshold=1e-6)
        assert strict.is_concentrated
        assert not lax.is_concentrated


class TestShotFormulaEdgeCases:
    """Direct cover of the two degenerate branches of the shot formula."""

    def test_zero_variance_is_unbounded(self) -> None:
        # Identical entries can never be separated by any shot budget.
        assert math.isinf(_shots_to_resolve(mean=0.3, variance=0.0))

    def test_negative_variance_is_unbounded(self) -> None:
        assert math.isinf(_shots_to_resolve(mean=0.3, variance=-1e-18))

    @pytest.mark.parametrize("mean", [0.0, 1.0])
    def test_deterministic_estimator_needs_one_shot(self, mean: float) -> None:
        # K exactly 0 or exactly 1: the binomial draw is deterministic.
        assert _shots_to_resolve(mean=mean, variance=0.25) == 1.0

    def test_matches_closed_form(self) -> None:
        assert _shots_to_resolve(mean=0.25, variance=0.01) == pytest.approx(
            4.0 * 0.25 * 0.75 / 0.01
        )


class TestShotBudget:
    def test_shots_formula(self) -> None:
        result = compute_kernel_concentration(
            AngleEncoding(n_features=3), n_samples=40, seed=0
        )
        mean, variance = result.offdiagonal_mean, result.offdiagonal_variance
        assert result.shots_per_entry == pytest.approx(
            4.0 * mean * (1.0 - mean) / variance
        )

    def test_concentrated_encoding_costs_more_shots(self) -> None:
        angle = compute_kernel_concentration(
            AngleEncoding(n_features=6), n_samples=N_STAT, seed=0
        )
        iqp = compute_kernel_concentration(
            IQPEncoding(n_features=6, reps=2), n_samples=N_STAT, seed=0
        )
        assert iqp.shots_per_entry > angle.shots_per_entry

    def test_shots_for_dataset_counts_pairs_only(self) -> None:
        result = compute_kernel_concentration(
            AngleEncoding(n_features=2), n_samples=20, seed=0
        )
        assert result.shots_for_dataset(100) == pytest.approx(
            result.shots_per_entry * 100 * 99 / 2
        )

    def test_shots_for_dataset_of_one_is_zero(self) -> None:
        result = compute_kernel_concentration(
            AngleEncoding(n_features=2), n_samples=20, seed=0
        )
        assert result.shots_for_dataset(1) == 0.0

    @pytest.mark.parametrize("bad", [0, -5, 2.5, True])
    def test_shots_for_dataset_validates(self, bad: object) -> None:
        result = compute_kernel_concentration(
            AngleEncoding(n_features=2), n_samples=20, seed=0
        )
        with pytest.raises(ValueError, match="positive integer"):
            result.shots_for_dataset(bad)  # type: ignore[arg-type]


class TestValidation:
    @pytest.mark.parametrize("bad", [0, 1, -3, True])
    def test_bad_n_samples_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            compute_kernel_concentration(
                AngleEncoding(n_features=2), n_samples=bad  # type: ignore[arg-type]
            )

    def test_bad_input_range_order_raises(self) -> None:
        with pytest.raises(ValueError, match="min < max"):
            compute_kernel_concentration(
                AngleEncoding(n_features=2), input_range=(1.0, 0.0)
            )

    def test_bad_input_range_length_raises(self) -> None:
        with pytest.raises(ValueError, match="min, max"):
            compute_kernel_concentration(
                AngleEncoding(n_features=2),
                input_range=(0.0, 1.0, 2.0),  # type: ignore[arg-type]
            )

    def test_non_finite_input_range_raises(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            compute_kernel_concentration(
                AngleEncoding(n_features=2), input_range=(0.0, math.inf)
            )

    def test_bad_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be positive"):
            compute_kernel_concentration(AngleEncoding(n_features=2), threshold=0.0)

    def test_bad_sampling_raises(self) -> None:
        with pytest.raises(ValueError, match="sampling must be"):
            compute_kernel_concentration(
                AngleEncoding(n_features=2), sampling="halton"  # type: ignore[arg-type]
            )

    def test_one_dimensional_X_raises(self) -> None:
        with pytest.raises(ValueError, match="2D array"):
            compute_kernel_concentration(
                AngleEncoding(n_features=2), np.array([0.1, 0.2])
            )

    def test_single_sample_X_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 samples"):
            compute_kernel_concentration(
                AngleEncoding(n_features=2), np.array([[0.1, 0.2]])
            )


# =====================================================================
# Scaling sweep
# =====================================================================


class TestScaling:
    def test_measures_every_requested_width(self) -> None:
        scaling = estimate_concentration_scaling(
            lambda d: AngleEncoding(n_features=d),
            feature_counts=(2, 4, 6),
            n_samples=30,
            seed=0,
        )
        assert isinstance(scaling, ScalingResult)
        assert scaling.n_qubits == (2, 4, 6)
        assert scaling.feature_counts == (2, 4, 6)
        assert len(scaling.results) == 3
        assert scaling.skipped == {}
        assert scaling.encoding_name == "AngleEncoding"

    def test_angle_decay_rate_matches_theory(self) -> None:
        """Var = (3/8)^n - (1/4)^n decays at ~8/3 per qubit asymptotically."""
        scaling = estimate_concentration_scaling(
            lambda d: AngleEncoding(n_features=d, rotation="Y"),
            feature_counts=(2, 4, 6, 8),
            n_samples=N_STAT,
            seed=0,
        )
        assert scaling.decay_rate == pytest.approx(8.0 / 3.0, rel=0.20)
        assert scaling.r_squared > 0.95

    def test_angle_mean_decay_rate_is_exactly_two(self) -> None:
        """E[K] = 2^-n exactly, so the mean halves with every added qubit."""
        scaling = estimate_concentration_scaling(
            lambda d: AngleEncoding(n_features=d, rotation="Y"),
            feature_counts=(2, 4, 6, 8),
            n_samples=N_STAT,
            seed=0,
        )
        assert scaling.mean_decay_rate == pytest.approx(2.0, rel=0.10)

    def test_angle_pulls_away_from_haar_floor(self) -> None:
        """log-ratio slope is ln(3/2) = 0.405: structure grows with width."""
        scaling = estimate_concentration_scaling(
            lambda d: AngleEncoding(n_features=d, rotation="Y"),
            feature_counts=(2, 4, 6, 8),
            n_samples=N_STAT,
            seed=0,
        )
        assert scaling.haar_normalized_slope == pytest.approx(math.log(1.5), rel=0.25)
        assert scaling.concentration_horizon() is None

    def test_entangling_map_sits_at_the_floor(self) -> None:
        scaling = estimate_concentration_scaling(
            lambda d: IQPEncoding(n_features=d, reps=2),
            feature_counts=(2, 4, 6, 8),
            n_samples=N_STAT,
            seed=0,
        )
        # Variance decays at nearly the maximal (Haar) rate of 4 per qubit.
        assert scaling.decay_rate > 3.0
        assert scaling.concentration_horizon() == 2
        assert all(r < 2.0 for r in scaling.concentration_ratios)

    def test_horizon_requires_a_sustained_crossing(self) -> None:
        """A dip below threshold at the narrowest width is not a horizon.

        ZZ feature map is at the floor throughout, so it has one; angle dips
        near 2.0 at two qubits but grows away, so it must not.
        """
        zz = estimate_concentration_scaling(
            lambda d: ZZFeatureMap(n_features=d, reps=2),
            feature_counts=(2, 4, 6),
            n_samples=N_STAT,
            seed=0,
        )
        assert zz.concentration_horizon() is not None

        # Synthesise the transient case directly: below threshold only at the
        # narrowest width, then growing away from the floor.
        transient = estimate_concentration_scaling(
            lambda d: AngleEncoding(n_features=d),
            feature_counts=(2, 4, 6, 8),
            n_samples=N_STAT,
            seed=0,
        )
        ratios = transient.concentration_ratios
        cut = (ratios[0] + ratios[1]) / 2.0  # below width 2, above width 4
        assert ratios[0] < cut < ratios[1]
        assert transient.concentration_horizon(threshold=cut) is None

    def test_horizon_validates_threshold(self) -> None:
        scaling = estimate_concentration_scaling(
            lambda d: AngleEncoding(n_features=d),
            feature_counts=(2, 4),
            n_samples=20,
            seed=0,
        )
        with pytest.raises(ValueError, match="threshold must be positive"):
            scaling.concentration_horizon(threshold=0.0)

    def test_skips_unsupported_widths_without_aborting(self) -> None:
        """SO(2) equivariance requires exactly two features."""
        scaling = estimate_concentration_scaling(
            lambda d: SO2EquivariantFeatureMap(n_features=d),
            feature_counts=(2, 4, 6),
            n_samples=20,
            seed=0,
        )
        assert len(scaling.results) == 1
        assert scaling.feature_counts == (2,)
        assert set(scaling.skipped) == {4, 6}
        assert all("ValueError" in reason for reason in scaling.skipped.values())

    def test_single_point_yields_unfittable_nans(self) -> None:
        """One width cannot support a trend: every fitted quantity is nan."""
        scaling = estimate_concentration_scaling(
            lambda d: SO2EquivariantFeatureMap(n_features=d),
            feature_counts=(2, 4),
            n_samples=20,
            seed=0,
        )
        assert len(scaling.results) == 1
        assert math.isnan(scaling.decay_rate)
        assert math.isnan(scaling.mean_decay_rate)
        assert math.isnan(scaling.haar_normalized_slope)
        assert math.isnan(scaling.r_squared)
        # No fit means no extrapolation beyond the single measured width.
        assert math.isnan(scaling.shots_per_entry_at(20))
        # The horizon still reports what the one measurement shows, and only
        # that: it can never be a width that was not measured.
        horizon = scaling.concentration_horizon()
        ratio = scaling.concentration_ratios[0]
        assert horizon == (scaling.n_qubits[0] if ratio < 2.0 else None)

    def test_all_widths_failing_raises(self) -> None:
        def always_fails(n_features: int) -> AngleEncoding:
            raise RuntimeError("no encoding here")

        with pytest.raises(RuntimeError, match="no measurable widths"):
            estimate_concentration_scaling(
                always_fails, feature_counts=(2, 4), n_samples=20, seed=0
            )

    def test_qubit_count_not_feature_count_is_the_abscissa(self) -> None:
        """Amplitude encoding uses ceil(log2(n_features)) qubits."""
        scaling = estimate_concentration_scaling(
            lambda d: AmplitudeEncoding(n_features=d),
            feature_counts=(2, 4, 8),
            n_samples=20,
            seed=0,
        )
        assert scaling.feature_counts == (2, 4, 8)
        assert scaling.n_qubits == (1, 2, 3)

    def test_derived_sequences_align_with_results(self) -> None:
        scaling = estimate_concentration_scaling(
            lambda d: AngleEncoding(n_features=d),
            feature_counts=(2, 4),
            n_samples=20,
            seed=0,
        )
        assert scaling.concentration_ratios == tuple(
            r.concentration_ratio for r in scaling.results
        )
        assert scaling.offdiagonal_means == tuple(
            r.offdiagonal_mean for r in scaling.results
        )
        assert scaling.offdiagonal_variances == tuple(
            r.offdiagonal_variance for r in scaling.results
        )

    def test_determinism_same_seed(self) -> None:
        def build() -> ScalingResult:
            return estimate_concentration_scaling(
                lambda d: AngleEncoding(n_features=d),
                feature_counts=(2, 4),
                n_samples=20,
                seed=5,
            )

        assert build().concentration_ratios == build().concentration_ratios

    def test_duplicate_feature_counts_are_deduplicated(self) -> None:
        scaling = estimate_concentration_scaling(
            lambda d: AngleEncoding(n_features=d),
            feature_counts=(2, 2, 4),
            n_samples=20,
            seed=0,
        )
        assert scaling.feature_counts == (2, 4)

    @pytest.mark.parametrize("bad", [(), (0,), (2, -1)])
    def test_bad_feature_counts_raise(self, bad: tuple) -> None:
        with pytest.raises(ValueError, match="feature_counts"):
            estimate_concentration_scaling(
                lambda d: AngleEncoding(n_features=d),
                feature_counts=bad,
                n_samples=20,
            )


class TestShotExtrapolation:
    def test_within_measured_range_returns_measurement(self) -> None:
        scaling = estimate_concentration_scaling(
            lambda d: AngleEncoding(n_features=d),
            feature_counts=(2, 4),
            n_samples=30,
            seed=0,
        )
        assert scaling.shots_per_entry_at(4) == scaling.results[-1].shots_per_entry
        assert scaling.shots_per_entry_at(2) == scaling.results[0].shots_per_entry

    def test_concentrated_encoding_extrapolates_to_a_larger_budget(self) -> None:
        angle = estimate_concentration_scaling(
            lambda d: AngleEncoding(n_features=d),
            feature_counts=(2, 4, 6, 8),
            n_samples=N_STAT,
            seed=0,
        )
        iqp = estimate_concentration_scaling(
            lambda d: IQPEncoding(n_features=d, reps=2),
            feature_counts=(2, 4, 6, 8),
            n_samples=N_STAT,
            seed=0,
        )
        assert iqp.shots_per_entry_at(20) > 100 * angle.shots_per_entry_at(20)

    def test_extrapolation_is_monotonic_in_width(self) -> None:
        scaling = estimate_concentration_scaling(
            lambda d: IQPEncoding(n_features=d, reps=2),
            feature_counts=(2, 4, 6, 8),
            n_samples=N_STAT,
            seed=0,
        )
        budgets = [scaling.shots_per_entry_at(n) for n in (10, 12, 14, 16)]
        assert budgets == sorted(budgets)

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True])
    def test_validates_qubit_count(self, bad: object) -> None:
        scaling = estimate_concentration_scaling(
            lambda d: AngleEncoding(n_features=d),
            feature_counts=(2, 4),
            n_samples=20,
            seed=0,
        )
        with pytest.raises(ValueError, match="positive integer"):
            scaling.shots_per_entry_at(bad)  # type: ignore[arg-type]


# =====================================================================
# Reduction from an already-computed kernel
# =====================================================================


class TestSummarizeKernelConcentration:
    """The seam that lets a caller holding a kernel skip a second simulation.

    Screening computes one kernel per encoding and derives both alignment and
    concentration from it, so this must agree exactly with the encoding-level
    entry point rather than approximately.
    """

    def test_agrees_with_the_encoding_level_entry_point(self) -> None:
        from encoding_atlas.analysis.concentration import summarize_kernel_concentration
        from encoding_atlas.analysis.generalization import compute_fidelity_kernel

        enc = AngleEncoding(n_features=4)
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 2 * np.pi, (30, 4))
        assert summarize_kernel_concentration(
            compute_fidelity_kernel(enc, X), enc.n_qubits
        ) == compute_kernel_concentration(enc, X)

    def test_identity_kernel_is_maximally_concentrated(self) -> None:
        from encoding_atlas.analysis.concentration import summarize_kernel_concentration

        result = summarize_kernel_concentration(np.eye(10), n_qubits=4)
        assert result.offdiagonal_mean == 0.0
        assert result.offdiagonal_variance == 0.0
        assert result.concentration_ratio == 0.0
        assert math.isinf(result.shots_per_entry)
        assert result.is_concentrated

    def test_threshold_is_honoured(self) -> None:
        from encoding_atlas.analysis.concentration import summarize_kernel_concentration

        # Needs genuine off-diagonal spread: a constant kernel has variance 0
        # and so sits below every positive threshold by construction.
        rng = np.random.default_rng(0)
        K = rng.uniform(0.0, 1.0, (8, 8))
        K = (K + K.T) / 2.0
        np.fill_diagonal(K, 1.0)
        assert summarize_kernel_concentration(K, 3, threshold=1e9).is_concentrated
        assert not summarize_kernel_concentration(K, 3, threshold=1e-9).is_concentrated

    def test_constant_kernel_has_no_spread(self) -> None:
        """Equal off-diagonals carry no geometry, whatever their common value."""
        from encoding_atlas.analysis.concentration import summarize_kernel_concentration

        K = np.full((8, 8), 0.5)
        np.fill_diagonal(K, 1.0)
        result = summarize_kernel_concentration(K, 3)
        assert result.offdiagonal_mean == pytest.approx(0.5)
        assert result.offdiagonal_variance == 0.0
        assert result.concentration_ratio == 0.0
        assert result.is_concentrated
        assert math.isinf(result.shots_per_entry)

    @pytest.mark.parametrize(
        ("K", "match"),
        [
            (np.zeros((2, 3)), "square 2D matrix"),
            (np.zeros(4), "square 2D matrix"),
            (np.ones((1, 1)), "at least 2x2"),
        ],
    )
    def test_invalid_kernel_raises(self, K: np.ndarray, match: str) -> None:
        from encoding_atlas.analysis.concentration import summarize_kernel_concentration

        with pytest.raises(ValueError, match=match):
            summarize_kernel_concentration(K, n_qubits=2)

    def test_invalid_threshold_raises(self) -> None:
        from encoding_atlas.analysis.concentration import summarize_kernel_concentration

        with pytest.raises(ValueError, match="threshold must be positive"):
            summarize_kernel_concentration(np.eye(4), n_qubits=2, threshold=0.0)
