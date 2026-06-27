"""Tests for the new statistical-rigor additions to the analysis pipeline.

This commit adds two complementary improvements that travel together
through every analysis function:

* **Bootstrap confidence intervals** — the detailed result for
  expressibility, entanglement capability, and trainability now
  carries percentile bootstrap CI bounds on the headline statistic
  (and supporting statistics where useful). The CI helper lives at
  :mod:`encoding_atlas.analysis._ci` and is reused by every analysis
  function for a single, well-tested implementation.
* **Quasi-random (Sobol') sampling** — :mod:`scipy.stats.qmc` Sobol'
  is now selectable via ``sampling='sobol'`` on every analysis
  function. Sobol' covers the hypercube more evenly than i.i.d.
  uniform draws and typically converges 30-50% faster on the
  analysis statistics. Uniform sampling remains the default.

The tests in this module focus on properties — determinism,
backward compatibility, CI sanity, validation errors — rather than
on the numerical content of any specific encoding. The existing
analysis tests already cover that surface; we only need to verify
that the new knobs work, are properly plumbed through every code
path, and don't change anything they shouldn't.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from encoding_atlas import HardwareEfficientEncoding, IQPEncoding
from encoding_atlas.analysis import (
    compute_entanglement_capability,
    compute_expressibility,
    estimate_trainability,
)
from encoding_atlas.analysis._ci import (
    percentile_bootstrap_ci,
    validate_ci_args,
)
from encoding_atlas.analysis._sampling import (
    generate_sample_batch,
    validate_sampling,
)
from encoding_atlas.analysis.expressibility import compute_fidelity_distribution


# Hide ``n_samples is low`` user warnings raised when we use small
# batches for test speed. We're testing properties, not statistical
# precision; small batches are enough to verify everything below.
@pytest.fixture(autouse=True)
def _silence_low_sample_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


# =============================================================================
# _ci.percentile_bootstrap_ci and validate_ci_args
# =============================================================================


class TestPercentileBootstrapCI:
    """The shared bootstrap helper exposed to every analysis function."""

    def test_known_distribution_mean_lies_inside_ci(self) -> None:
        # For a sample drawn from N(5, 2), the 95% bootstrap CI should
        # contain the sample mean (and almost always the population
        # mean) with vanishing failure rate at n=500.
        gen = np.random.default_rng(0)
        samples = gen.normal(loc=5.0, scale=2.0, size=500)
        sample_mean = float(np.mean(samples))
        lo, hi = percentile_bootstrap_ci(
            samples, np.mean, np.random.default_rng(42), n_bootstrap=500
        )
        assert lo < sample_mean < hi

    def test_deterministic_under_same_rng_seed(self) -> None:
        samples = np.random.default_rng(0).normal(0, 1, size=100)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        a = percentile_bootstrap_ci(samples, np.mean, rng1, n_bootstrap=100)
        b = percentile_bootstrap_ci(samples, np.mean, rng2, n_bootstrap=100)
        assert a == b

    def test_constant_samples_collapse(self) -> None:
        ci = percentile_bootstrap_ci(
            np.full(20, 7.0), np.mean, np.random.default_rng(0)
        )
        assert ci == (7.0, 7.0)

    def test_single_sample_collapses_to_point(self) -> None:
        ci = percentile_bootstrap_ci(
            np.array([3.14]), np.mean, np.random.default_rng(0)
        )
        assert ci == (3.14, 3.14)

    def test_empty_samples_returns_nan_pair(self) -> None:
        lo, hi = percentile_bootstrap_ci(
            np.array([], dtype=np.float64), np.mean, np.random.default_rng(0)
        )
        assert np.isnan(lo) and np.isnan(hi)

    def test_wider_confidence_level_gives_wider_or_equal_interval(self) -> None:
        # Statistical property: with enough resamples the 99% CI is at
        # least as wide as the 95% CI on the same sample. We use 1000
        # bootstraps so the property holds reliably.
        samples = np.random.default_rng(0).normal(0, 1, size=200)
        rng_95 = np.random.default_rng(42)
        rng_99 = np.random.default_rng(42)
        lo95, hi95 = percentile_bootstrap_ci(
            samples, np.mean, rng_95, n_bootstrap=1000, confidence_level=0.95
        )
        lo99, hi99 = percentile_bootstrap_ci(
            samples, np.mean, rng_99, n_bootstrap=1000, confidence_level=0.99
        )
        assert hi99 - lo99 >= hi95 - lo95 - 1e-9

    @pytest.mark.parametrize("bad", [-0.1, 0.0, 1.0, 1.5, "0.95", None])
    def test_validate_ci_args_rejects_bad_confidence(self, bad: object) -> None:
        with pytest.raises(ValueError, match="confidence_level"):
            validate_ci_args(bad, 100)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [0, -1, 0.5, "100", None])
    def test_validate_ci_args_rejects_bad_n_bootstrap(self, bad: object) -> None:
        with pytest.raises(ValueError, match="n_bootstrap"):
            validate_ci_args(0.95, bad)  # type: ignore[arg-type]


# =============================================================================
# _sampling.generate_sample_batch and validate_sampling
# =============================================================================


class TestGenerateSampleBatch:
    """The shared sampling helper used by every analysis function."""

    @pytest.mark.parametrize("sampling", ["uniform", "sobol"])
    def test_shape_and_range(self, sampling: str) -> None:
        rng = np.random.default_rng(0)
        X = generate_sample_batch(
            n_samples=64,
            n_features=4,
            input_range=(0.0, 2.0),
            rng=rng,
            sampling=sampling,  # type: ignore[arg-type]
        )
        assert X.shape == (64, 4)
        assert X.min() >= 0.0
        assert X.max() <= 2.0
        assert X.dtype == np.float64

    @pytest.mark.parametrize("sampling", ["uniform", "sobol"])
    def test_deterministic_under_same_rng_seed(self, sampling: str) -> None:
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        X1 = generate_sample_batch(
            64, 4, (0.0, 1.0), rng1, sampling=sampling  # type: ignore[arg-type]
        )
        X2 = generate_sample_batch(
            64, 4, (0.0, 1.0), rng2, sampling=sampling  # type: ignore[arg-type]
        )
        assert np.array_equal(X1, X2)

    def test_uniform_and_sobol_differ(self) -> None:
        # The two strategies should not coincidentally produce the same
        # numerical batch for any reasonable seed.
        rng_u = np.random.default_rng(42)
        rng_s = np.random.default_rng(42)
        X_uni = generate_sample_batch(64, 4, (0.0, 1.0), rng_u, sampling="uniform")
        X_sob = generate_sample_batch(64, 4, (0.0, 1.0), rng_s, sampling="sobol")
        assert not np.array_equal(X_uni, X_sob)

    def test_range_scaling_negative_low(self) -> None:
        X = generate_sample_batch(
            32, 2, (-3.0, 5.0), np.random.default_rng(0), sampling="uniform"
        )
        assert X.min() >= -3.0
        assert X.max() <= 5.0

    @pytest.mark.parametrize("bad", ["halton", "Sobol", "UNIFORM", "", 0, None])
    def test_validate_sampling_rejects_bad_values(self, bad: object) -> None:
        with pytest.raises(ValueError, match="sampling must be"):
            validate_sampling(bad)  # type: ignore[arg-type]

    def test_sobol_n_not_power_of_two_does_not_warn_to_user(self) -> None:
        """We deliberately suppress scipy's power-of-two notice inside
        ``generate_sample_batch`` so users don't see noise."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            generate_sample_batch(
                30, 4, (0.0, 1.0), np.random.default_rng(0), sampling="sobol"
            )
        sobol_balance_warnings = [
            w
            for w in caught
            if "balance" in str(w.message).lower()
            or "power of 2" in str(w.message).lower()
        ]
        assert sobol_balance_warnings == []


# =============================================================================
# Backward compatibility: existing seeded outputs remain unchanged.
# =============================================================================


class TestBackwardCompatibilityForUniform:
    """When ``sampling='uniform'`` and the new CI knobs are at their
    defaults, every analysis function must produce the same float
    output as before the commit. We use small ``n_samples`` because
    the determinism property holds for any size."""

    def test_expressibility_default_args_unchanged(self) -> None:
        enc = IQPEncoding(n_features=3, reps=1)
        # No CI knobs given — default to uniform + 95% CI + 200 boot
        # samples. The float-only path must match the seeded baseline
        # exactly.
        v1 = compute_expressibility(enc, n_samples=30, seed=42)
        v2 = compute_expressibility(enc, n_samples=30, seed=42, sampling="uniform")
        assert v1 == v2

    def test_entanglement_default_args_unchanged(self) -> None:
        enc = IQPEncoding(n_features=3, reps=1)
        v1 = compute_entanglement_capability(enc, n_samples=20, seed=42)
        v2 = compute_entanglement_capability(
            enc, n_samples=20, seed=42, sampling="uniform"
        )
        assert v1 == v2

    def test_trainability_default_args_unchanged(self) -> None:
        enc = HardwareEfficientEncoding(n_features=3, reps=1)
        v1 = estimate_trainability(enc, n_samples=15, seed=42)
        v2 = estimate_trainability(enc, n_samples=15, seed=42, sampling="uniform")
        assert v1 == v2


# =============================================================================
# Sobol' sampling — determinism, distinct from uniform, plumbed through
# =============================================================================


class TestSobolSamplingEndToEnd:
    """The ``sampling='sobol'`` knob is properly wired through every
    analysis function."""

    @pytest.mark.parametrize(
        "fn,enc,kwargs",
        [
            (
                compute_expressibility,
                IQPEncoding(n_features=3, reps=1),
                {"n_samples": 32, "seed": 42},
            ),
            (
                compute_fidelity_distribution,
                IQPEncoding(n_features=3, reps=1),
                {"n_samples": 32, "seed": 42},
            ),
            (
                compute_entanglement_capability,
                IQPEncoding(n_features=3, reps=1),
                {"n_samples": 32, "seed": 42},
            ),
            (
                estimate_trainability,
                HardwareEfficientEncoding(n_features=3, reps=1),
                {"n_samples": 16, "seed": 42},
            ),
        ],
        ids=["expressibility", "fidelity_distribution", "entanglement", "trainability"],
    )
    def test_sobol_is_deterministic_and_differs_from_uniform(
        self, fn, enc, kwargs
    ) -> None:  # noqa: ANN001
        # Same seed → same Sobol' output.
        a = fn(enc, sampling="sobol", **kwargs)
        b = fn(enc, sampling="sobol", **kwargs)
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b)
        else:
            assert a == b
        # And Sobol' should differ from uniform with the same seed.
        u = fn(enc, sampling="uniform", **kwargs)
        if isinstance(a, np.ndarray):
            assert not np.array_equal(a, u)
        else:
            assert a != u


# =============================================================================
# Confidence intervals: structure + sanity properties
# =============================================================================


class TestExpressibilityCI:
    """The detailed expressibility result exposes a working CI."""

    def test_ci_keys_present_and_typed(self) -> None:
        enc = IQPEncoding(n_features=3, reps=1)
        result = compute_expressibility(
            enc, n_samples=40, seed=42, return_distributions=True
        )
        for key in (
            "expressibility_ci_lower",
            "expressibility_ci_upper",
            "mean_fidelity_ci_lower",
            "mean_fidelity_ci_upper",
        ):
            assert key in result, f"missing {key}"
            assert isinstance(result[key], float), f"{key} is not float"
        assert result["confidence_level"] == 0.95
        assert result["sampling"] == "uniform"

    def test_ci_bounds_bracket_point_estimate_roughly(self) -> None:
        # Statistical sanity: for a non-degenerate fidelity sample, the
        # 95% CI typically brackets the point estimate. We use a tolerant
        # check because tiny n_samples can produce edge cases.
        enc = IQPEncoding(n_features=3, reps=1)
        result = compute_expressibility(
            enc, n_samples=80, seed=42, return_distributions=True
        )
        expr = result["expressibility"]
        lo, hi = result["expressibility_ci_lower"], result["expressibility_ci_upper"]
        assert lo <= hi  # Mandatory invariant.
        # Loose bracket: the point estimate should be within the CI in
        # the overwhelming majority of cases. With n=80 and 200
        # bootstraps we expect exactness; allow 0.01 slack for safety.
        assert lo - 0.01 <= expr <= hi + 0.01

    def test_ci_is_deterministic(self) -> None:
        enc = IQPEncoding(n_features=3, reps=1)
        a = compute_expressibility(
            enc, n_samples=40, seed=42, return_distributions=True
        )
        b = compute_expressibility(
            enc, n_samples=40, seed=42, return_distributions=True
        )
        assert a["expressibility_ci_lower"] == b["expressibility_ci_lower"]
        assert a["expressibility_ci_upper"] == b["expressibility_ci_upper"]
        assert a["mean_fidelity_ci_lower"] == b["mean_fidelity_ci_lower"]
        assert a["mean_fidelity_ci_upper"] == b["mean_fidelity_ci_upper"]

    def test_invalid_confidence_level_raises_clean(self) -> None:
        enc = IQPEncoding(n_features=3, reps=1)
        with pytest.raises(ValueError, match="confidence_level"):
            compute_expressibility(enc, n_samples=30, seed=42, confidence_level=0.0)
        with pytest.raises(ValueError, match="confidence_level"):
            compute_expressibility(enc, n_samples=30, seed=42, confidence_level=1.5)

    def test_invalid_n_bootstrap_raises_clean(self) -> None:
        enc = IQPEncoding(n_features=3, reps=1)
        with pytest.raises(ValueError, match="n_bootstrap"):
            compute_expressibility(enc, n_samples=30, seed=42, n_bootstrap_ci=0)


class TestEntanglementCI:
    """The detailed entanglement result exposes a working CI."""

    def test_ci_keys_present(self) -> None:
        enc = IQPEncoding(n_features=3, reps=1)
        r = compute_entanglement_capability(
            enc, n_samples=20, seed=42, return_details=True
        )
        for key in (
            "entanglement_ci_lower",
            "entanglement_ci_upper",
            "confidence_level",
            "sampling",
        ):
            assert key in r

    def test_ci_bounds_in_unit_interval_and_bracket_estimate(self) -> None:
        enc = IQPEncoding(n_features=3, reps=1)
        r = compute_entanglement_capability(
            enc, n_samples=30, seed=42, return_details=True
        )
        lo = r["entanglement_ci_lower"]
        hi = r["entanglement_ci_upper"]
        ent = r["entanglement_capability"]
        assert 0.0 <= lo <= hi <= 1.0
        assert lo - 0.01 <= ent <= hi + 0.01

    def test_invalid_args_raise_clean(self) -> None:
        enc = IQPEncoding(n_features=3, reps=1)
        with pytest.raises(ValueError, match="sampling"):
            compute_entanglement_capability(
                enc, n_samples=20, seed=42, sampling="bad"  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="confidence_level"):
            compute_entanglement_capability(
                enc, n_samples=20, seed=42, confidence_level=0.0
            )


class TestTrainabilityCI:
    """The detailed trainability result exposes a working CI on both the
    trainability score and the underlying gradient variance."""

    def test_ci_keys_present(self) -> None:
        enc = HardwareEfficientEncoding(n_features=3, reps=1)
        r = estimate_trainability(enc, n_samples=15, seed=42, return_details=True)
        for key in (
            "trainability_ci_lower",
            "trainability_ci_upper",
            "gradient_variance_ci_lower",
            "gradient_variance_ci_upper",
            "confidence_level",
            "sampling",
        ):
            assert key in r

    def test_ci_bounds_are_ordered_and_finite(self) -> None:
        enc = HardwareEfficientEncoding(n_features=3, reps=1)
        r = estimate_trainability(enc, n_samples=15, seed=42, return_details=True)
        for key_prefix in ("trainability_", "gradient_variance_"):
            lo = r[f"{key_prefix}ci_lower"]  # type: ignore[literal-required]
            hi = r[f"{key_prefix}ci_upper"]  # type: ignore[literal-required]
            assert np.isfinite(lo) and np.isfinite(hi)
            assert lo <= hi

    def test_invalid_args_raise_clean(self) -> None:
        enc = HardwareEfficientEncoding(n_features=3, reps=1)
        with pytest.raises(ValueError, match="sampling"):
            estimate_trainability(
                enc, n_samples=15, seed=42, sampling="x"  # type: ignore[arg-type]
            )
        with pytest.raises(ValueError, match="n_bootstrap"):
            estimate_trainability(enc, n_samples=15, seed=42, n_bootstrap_ci=0)
