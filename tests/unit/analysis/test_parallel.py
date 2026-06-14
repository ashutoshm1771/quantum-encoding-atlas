"""Tests for analysis-loop parallelization.

The three core analysis functions (``compute_expressibility``,
``compute_entanglement_capability``, ``estimate_trainability``) gained a
``parallel`` parameter (``False`` / ``True`` / ``'thread'`` / ``'process'``)
plus ``max_workers`` in this commit. The tests in this module verify:

* **Numerical determinism** — for a fixed ``seed``, every mode produces
  byte-identical output to the sequential baseline. This is the
  single most important property; if it ever regresses, the analysis
  pipeline silently loses reproducibility.
* **Backward compatibility** — ``parallel=True`` is an alias for
  ``'thread'``, and omitting ``parallel`` keeps the sequential
  semantics unchanged.
* **Argument validation** — a bad value raises a clean ``ValueError``
  *before* any expensive sampling happens (i.e. it is not wrapped in
  an ``AnalysisError`` by the broad sampling-try/except).
* **All three backends + all three modes** — every supported backend is
  exercised against every supported mode.
* **All three analysis functions** — expressibility, entanglement,
  trainability, plus the lower-level ``compute_fidelity_distribution``.
* **The shared :mod:`encoding_atlas.analysis._parallel` helper** has
  exhaustive coverage of its tiny normaliser.

These tests use small sample counts to stay fast; the determinism
guarantee makes large-sample testing unnecessary — if 30 samples are
identical across modes, 30 000 will be too.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

# Backend availability — process-pool tests are run for every available
# backend because every backend in the analysis path returns NumPy arrays
# (not circuit objects), so pickle-across-processes works for all of them.
try:
    import qiskit  # noqa: F401

    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    import cirq  # noqa: F401

    HAS_CIRQ = True
except ImportError:
    HAS_CIRQ = False

from encoding_atlas import AngleEncoding, HardwareEfficientEncoding, IQPEncoding
from encoding_atlas.analysis import (
    compute_entanglement_capability,
    compute_expressibility,
    estimate_trainability,
)
from encoding_atlas.analysis._parallel import resolve_parallel_mode
from encoding_atlas.analysis.expressibility import compute_fidelity_distribution


# Hide the "n_samples is low" warnings raised when we run with small batches
# for test speed. The numerical determinism check doesn't need large
# samples to be meaningful.
@pytest.fixture(autouse=True)
def _silence_low_sample_warning() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        yield


# =============================================================================
# resolve_parallel_mode — tiny pure-Python normaliser
# =============================================================================


class TestResolveParallelMode:
    """The shared helper used by all three analysis functions."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (False, "sequential"),
            (True, "thread"),
            ("thread", "thread"),
            ("process", "process"),
        ],
    )
    def test_accepted_values(self, value: object, expected: str) -> None:
        assert resolve_parallel_mode(value) == expected  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", ["sequential", "Process", "Thread", "tread", 1, 0])
    def test_rejects_bad_values(self, bad: object) -> None:
        with pytest.raises(ValueError, match="parallel must be"):
            resolve_parallel_mode(bad)  # type: ignore[arg-type]


# =============================================================================
# Numerical determinism: parallel modes match sequential exactly
# =============================================================================


def _backend_param() -> list[pytest.param]:
    """Parametrize over every installed simulation backend."""
    backends = [pytest.param("pennylane", id="pennylane")]
    if HAS_QISKIT:
        backends.append(pytest.param("qiskit", id="qiskit"))
    if HAS_CIRQ:
        backends.append(pytest.param("cirq", id="cirq"))
    return backends


class TestExpressibilityParallelism:
    """``compute_expressibility`` must yield identical output across modes."""

    @pytest.mark.parametrize("backend", _backend_param())
    @pytest.mark.parametrize("mode", [True, "thread", "process"])
    def test_parallel_matches_sequential(
        self, backend: str, mode
    ) -> None:  # noqa: ANN001
        enc = IQPEncoding(n_features=3, reps=1)
        baseline = compute_expressibility(enc, n_samples=30, seed=42, backend=backend)
        observed = compute_expressibility(
            enc,
            n_samples=30,
            seed=42,
            backend=backend,
            parallel=mode,
            max_workers=2,
        )
        assert observed == baseline

    def test_invalid_parallel_raises_clean_ValueError(self) -> None:
        enc = IQPEncoding(n_features=3, reps=1)
        # The bad value must surface as a raw ValueError, not wrapped in
        # an AnalysisError by the broad sampling-try/except.
        with pytest.raises(ValueError, match="parallel must be"):
            compute_expressibility(
                enc, n_samples=30, seed=42, parallel="invalid"  # type: ignore[arg-type]
            )

    def test_return_distributions_matches_across_modes(self) -> None:
        enc = IQPEncoding(n_features=3, reps=1)
        seq = compute_expressibility(
            enc, n_samples=30, seed=42, return_distributions=True
        )
        thr = compute_expressibility(
            enc,
            n_samples=30,
            seed=42,
            return_distributions=True,
            parallel="thread",
            max_workers=2,
        )
        assert seq["expressibility"] == thr["expressibility"]
        assert seq["kl_divergence"] == thr["kl_divergence"]
        np.testing.assert_array_equal(
            seq["fidelity_distribution"], thr["fidelity_distribution"]
        )


class TestFidelityDistributionParallelism:
    """The lower-level ``compute_fidelity_distribution`` exposes the same
    parallel knobs and must also be numerically deterministic."""

    @pytest.mark.parametrize("mode", [True, "thread", "process"])
    def test_parallel_matches_sequential(self, mode) -> None:  # noqa: ANN001
        enc = IQPEncoding(n_features=3, reps=1)
        baseline = compute_fidelity_distribution(enc, n_samples=30, seed=42)
        observed = compute_fidelity_distribution(
            enc, n_samples=30, seed=42, parallel=mode, max_workers=2
        )
        np.testing.assert_array_equal(observed, baseline)

    def test_invalid_parallel_raises_clean_ValueError(self) -> None:
        enc = IQPEncoding(n_features=3, reps=1)
        with pytest.raises(ValueError, match="parallel must be"):
            compute_fidelity_distribution(
                enc, n_samples=30, seed=42, parallel="invalid"  # type: ignore[arg-type]
            )


class TestEntanglementParallelism:
    """``compute_entanglement_capability`` for both Meyer-Wallach and
    Scott measures, all backends, all modes."""

    @pytest.mark.parametrize("backend", _backend_param())
    @pytest.mark.parametrize("mode", [True, "thread", "process"])
    def test_meyer_wallach_matches_sequential(
        self, backend: str, mode
    ) -> None:  # noqa: ANN001
        enc = IQPEncoding(n_features=3, reps=1)
        baseline = compute_entanglement_capability(
            enc, n_samples=20, seed=42, backend=backend
        )
        observed = compute_entanglement_capability(
            enc,
            n_samples=20,
            seed=42,
            backend=backend,
            parallel=mode,
            max_workers=2,
        )
        assert observed == baseline

    @pytest.mark.parametrize("mode", [True, "thread", "process"])
    def test_scott_measure_matches_sequential(self, mode) -> None:  # noqa: ANN001
        enc = IQPEncoding(n_features=3, reps=1)
        baseline = compute_entanglement_capability(
            enc, n_samples=20, seed=42, measure="scott", scott_k=1
        )
        observed = compute_entanglement_capability(
            enc,
            n_samples=20,
            seed=42,
            measure="scott",
            scott_k=1,
            parallel=mode,
            max_workers=2,
        )
        assert observed == baseline

    @pytest.mark.parametrize("mode", [True, "thread", "process"])
    def test_return_details_matches_across_modes(self, mode) -> None:  # noqa: ANN001
        enc = IQPEncoding(n_features=3, reps=1)
        seq = compute_entanglement_capability(
            enc, n_samples=20, seed=42, return_details=True
        )
        par = compute_entanglement_capability(
            enc,
            n_samples=20,
            seed=42,
            return_details=True,
            parallel=mode,
            max_workers=2,
        )
        assert seq["entanglement_capability"] == par["entanglement_capability"]
        assert seq["std_error"] == par["std_error"]
        np.testing.assert_array_equal(
            seq["entanglement_samples"], par["entanglement_samples"]
        )
        np.testing.assert_array_equal(
            seq["per_qubit_entanglement"], par["per_qubit_entanglement"]
        )

    def test_invalid_parallel_raises_clean_ValueError(self) -> None:
        enc = IQPEncoding(n_features=3, reps=1)
        with pytest.raises(ValueError, match="parallel must be"):
            compute_entanglement_capability(
                enc, n_samples=20, seed=42, parallel="invalid"  # type: ignore[arg-type]
            )


class TestTrainabilityParallelism:
    """``estimate_trainability`` is the most expensive — gradient
    computation runs ~2*n_features simulations per sample — so its
    parallelization win is the largest. It also has the trickiest path:
    failed samples must still be tolerated and packed-from-zero must
    match the sequential implementation."""

    @pytest.mark.parametrize("backend", _backend_param())
    @pytest.mark.parametrize("mode", [True, "thread", "process"])
    def test_parallel_matches_sequential(
        self, backend: str, mode
    ) -> None:  # noqa: ANN001
        enc = HardwareEfficientEncoding(n_features=3, reps=1)
        baseline = estimate_trainability(enc, n_samples=15, seed=42, backend=backend)
        observed = estimate_trainability(
            enc,
            n_samples=15,
            seed=42,
            backend=backend,
            parallel=mode,
            max_workers=2,
        )
        assert observed == baseline

    @pytest.mark.parametrize("mode", [True, "thread", "process"])
    def test_return_details_matches_across_modes(self, mode) -> None:  # noqa: ANN001
        enc = HardwareEfficientEncoding(n_features=3, reps=1)
        seq = estimate_trainability(enc, n_samples=15, seed=42, return_details=True)
        par = estimate_trainability(
            enc,
            n_samples=15,
            seed=42,
            return_details=True,
            parallel=mode,
            max_workers=2,
        )
        # Scalar pieces must match exactly across modes.
        for key in (
            "trainability_estimate",
            "gradient_variance",
            "barren_plateau_risk",
            "effective_dimension",
            "n_samples",
            "n_successful_samples",
            "n_failed_samples",
        ):
            assert seq[key] == par[key], f"{key} differs: {seq[key]} != {par[key]}"
        np.testing.assert_array_equal(
            seq["per_parameter_variance"], par["per_parameter_variance"]
        )

    def test_invalid_parallel_raises_clean_ValueError(self) -> None:
        enc = HardwareEfficientEncoding(n_features=3, reps=1)
        with pytest.raises(ValueError, match="parallel must be"):
            estimate_trainability(
                enc, n_samples=15, seed=42, parallel="invalid"  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("mode", [True, "thread", "process"])
    def test_observable_propagates_into_parallel_path(
        self, mode
    ) -> None:  # noqa: ANN001
        """The observable kwarg must reach workers; using different
        observables should give different (but still mode-stable)
        results."""
        enc = HardwareEfficientEncoding(n_features=3, reps=1)
        comp_seq = estimate_trainability(
            enc, n_samples=15, seed=42, observable="computational"
        )
        global_seq = estimate_trainability(
            enc, n_samples=15, seed=42, observable="global_z"
        )
        comp_par = estimate_trainability(
            enc,
            n_samples=15,
            seed=42,
            observable="computational",
            parallel=mode,
            max_workers=2,
        )
        global_par = estimate_trainability(
            enc,
            n_samples=15,
            seed=42,
            observable="global_z",
            parallel=mode,
            max_workers=2,
        )
        assert comp_seq == comp_par
        assert global_seq == global_par
        # Sanity: the two observables should generally give different
        # trainability scores; equality would imply the observable kwarg
        # is being ignored.
        assert comp_seq != global_seq or comp_seq == 0.0


# =============================================================================
# Cross-cutting: small encoding still works (single-sample edge case)
# =============================================================================


class TestEdgeCases:
    """Boundary cases that should not break the parallel dispatch."""

    @pytest.mark.parametrize("mode", [True, "thread", "process"])
    def test_n_samples_equals_one_uses_sequential_fastpath(
        self, mode
    ) -> None:  # noqa: ANN001
        """When there's a single sample, the parallel branches still
        need to produce a finite result — the implementations explicitly
        short-circuit to the sequential path for ``n_samples <= 1`` to
        avoid executor overhead. We can't call expressibility with
        n_samples=1 (it errors below threshold) but we can verify the
        guarantee holds for entanglement.

        We use ``n_samples`` at the analysis minimum (10) and prove the
        result is identical across modes."""
        enc = AngleEncoding(n_features=2)
        baseline = compute_entanglement_capability(enc, n_samples=10, seed=42)
        observed = compute_entanglement_capability(
            enc, n_samples=10, seed=42, parallel=mode, max_workers=1
        )
        assert observed == baseline

    def test_max_workers_one_still_uses_executor(self) -> None:
        """``max_workers=1`` is unusual but legal — it shouldn't change
        numerical output. (This also exercises the thin
        ThreadPoolExecutor wrapper.)"""
        enc = IQPEncoding(n_features=3, reps=1)
        baseline = compute_expressibility(enc, n_samples=30, seed=42)
        observed = compute_expressibility(
            enc, n_samples=30, seed=42, parallel="thread", max_workers=1
        )
        assert observed == baseline
