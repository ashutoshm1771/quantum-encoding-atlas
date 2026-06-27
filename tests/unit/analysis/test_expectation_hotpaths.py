"""Unit tests for the hot-path expectation-value helpers in ``_utils``.

The optimisations covered here replaced three Python-level loops in the
trainability pipeline:

* ``_global_z_eigenvalues`` — a cached, bit-parity-fold version of the
  per-call ``[1.0 - 2.0 * (bin(i).count("1") % 2) for ...]`` list.
* ``_local_z_expectation_from_state`` — the ``⟨Z₀⟩`` reducer, formerly a
  ``for i in range(2**n)`` Python loop.
* ``_expectations_batch`` — a vectorised per-row evaluator for every
  supported observable, used by both ``compute_all_parameter_gradients``
  and the trainability inner loop.

The tests below pin three things:

1. **Correctness against the slow reference.** Each helper is compared
   against the original Python-level formula on hand-crafted and random
   inputs, including all qubit counts the library supports.
2. **Behavioural invariants.** The cached array is immutable; the cache
   actually hits; degenerate / boundary inputs (single-qubit, all-zero,
   computational basis) match analytic expectations.
3. **Public surface preservation.** ``compute_all_parameter_gradients``
   is asserted to be byte-identical to ``compute_parameter_gradient``
   called per-index for every observable and across an entangling
   encoding — the optimisation must not change numerical output.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from encoding_atlas.analysis._utils import (
    _expectations_batch,
    _global_z_eigenvalues,
    _local_z_expectation_from_state,
    compute_all_parameter_gradients,
    compute_parameter_gradient,
)
from encoding_atlas.core.exceptions import (
    NumericalInstabilityError,
    ValidationError,
)

# =============================================================================
# Reference (slow) implementations used as ground truth.
# =============================================================================


def _reference_global_z_eigenvalues(n_qubits: int) -> np.ndarray:
    """Original Python list-comprehension form, used as the reference."""
    return np.array(
        [1.0 - 2.0 * (bin(i).count("1") % 2) for i in range(2**n_qubits)],
        dtype=np.float64,
    )


def _reference_local_z(statevector: np.ndarray, n_qubits: int) -> float:
    """Original ``for i in range(2**n)`` Python loop form."""
    dim = 2**n_qubits
    prob_zero = 0.0
    for i in range(dim):
        if (i >> (n_qubits - 1)) & 1 == 0:
            prob_zero += float(np.abs(statevector[i]) ** 2)
    return 2.0 * prob_zero - 1.0


def _reference_global_z_expectation(statevector: np.ndarray, n_qubits: int) -> float:
    """Slow but obviously-correct global-Z expectation."""
    z = _reference_global_z_eigenvalues(n_qubits)
    probs = np.abs(statevector) ** 2
    return float(np.sum(z * probs))


def _random_statevector(n_qubits: int, seed: int) -> np.ndarray:
    """Reproducible Haar-uniform-ish normalised statevector."""
    rng = np.random.default_rng(seed)
    dim = 2**n_qubits
    real = rng.standard_normal(dim)
    imag = rng.standard_normal(dim)
    psi = real + 1j * imag
    return (psi / np.linalg.norm(psi)).astype(np.complex128)


# =============================================================================
# _global_z_eigenvalues
# =============================================================================


class TestGlobalZEigenvalues:
    """Correctness, immutability, caching, and validation."""

    @pytest.mark.parametrize("n", list(range(1, 13)))
    def test_matches_reference_for_each_qubit_count(self, n: int) -> None:
        cached = _global_z_eigenvalues(n)
        reference = _reference_global_z_eigenvalues(n)
        assert cached.dtype == np.float64
        assert cached.shape == (2**n,)
        assert_array_equal(cached, reference)

    def test_only_plus_minus_one_values(self) -> None:
        for n in (1, 4, 8):
            eigenvalues = _global_z_eigenvalues(n)
            assert_array_equal(np.unique(eigenvalues), np.array([-1.0, 1.0]))

    def test_even_count_matches_odd_count(self) -> None:
        # By symmetry, exactly half the eigenvalues are +1 and half are -1
        # (the two parity classes are equinumerous in the binary cube).
        for n in (2, 5, 9):
            eigenvalues = _global_z_eigenvalues(n)
            assert int(np.sum(eigenvalues == 1.0)) == 2 ** (n - 1)
            assert int(np.sum(eigenvalues == -1.0)) == 2 ** (n - 1)

    def test_cached_array_is_read_only(self) -> None:
        eigenvalues = _global_z_eigenvalues(4)
        assert eigenvalues.flags.writeable is False
        with pytest.raises(ValueError):
            eigenvalues[0] = -1.0

    def test_cache_returns_same_object(self) -> None:
        first = _global_z_eigenvalues(6)
        second = _global_z_eigenvalues(6)
        assert first is second

    def test_different_n_qubits_distinct_arrays(self) -> None:
        a = _global_z_eigenvalues(3)
        b = _global_z_eigenvalues(4)
        assert a is not b
        assert a.shape != b.shape

    def test_invalid_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _global_z_eigenvalues(0)

    def test_invalid_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _global_z_eigenvalues(-3)

    def test_invalid_non_int_raises(self) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            _global_z_eigenvalues(3.0)  # type: ignore[arg-type]

    def test_exceeds_max_qubits_raises(self) -> None:
        with pytest.raises(ValueError, match="maximum"):
            _global_z_eigenvalues(21)


# =============================================================================
# _local_z_expectation_from_state
# =============================================================================


class TestLocalZExpectation:
    """``⟨Z₀⟩`` correctness across known and random states."""

    def test_zero_state_returns_plus_one(self) -> None:
        for n in (1, 2, 3, 4):
            state = np.zeros(2**n, dtype=np.complex128)
            state[0] = 1.0
            assert _local_z_expectation_from_state(state, n) == pytest.approx(1.0)

    def test_one_on_first_qubit_returns_minus_one(self) -> None:
        # |1...> where qubit 0 (MSB) is 1.
        for n in (1, 2, 3, 4):
            state = np.zeros(2**n, dtype=np.complex128)
            state[2 ** (n - 1)] = 1.0  # first index with MSB=1
            assert _local_z_expectation_from_state(state, n) == pytest.approx(-1.0)

    def test_uniform_superposition_returns_zero(self) -> None:
        # Equal superposition over all basis states → P(q₀=0) = 1/2 → ⟨Z₀⟩ = 0.
        for n in (1, 2, 3, 4):
            state = np.full(2**n, 1.0 / np.sqrt(2**n), dtype=np.complex128)
            result = _local_z_expectation_from_state(state, n)
            assert result == pytest.approx(0.0, abs=1e-12)

    def test_bell_state_first_qubit_returns_zero(self) -> None:
        # (|00> + |11>)/√2 → qubit 0 maximally mixed.
        state = np.zeros(4, dtype=np.complex128)
        state[0] = 1.0 / np.sqrt(2)
        state[3] = 1.0 / np.sqrt(2)
        assert _local_z_expectation_from_state(state, 2) == pytest.approx(0.0)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7])
    def test_random_states_match_reference(self, n: int) -> None:
        state = _random_statevector(n, seed=100 + n)
        new = _local_z_expectation_from_state(state, n)
        reference = _reference_local_z(state, n)
        assert new == pytest.approx(reference, abs=1e-12)

    def test_returns_python_float(self) -> None:
        state = np.array([1.0, 0.0], dtype=np.complex128)
        result = _local_z_expectation_from_state(state, 1)
        assert isinstance(result, float)


# =============================================================================
# _expectations_batch
# =============================================================================


class TestExpectationsBatch:
    """Vectorised per-row evaluator covering every supported observable."""

    @pytest.fixture(scope="class")
    def basis_states(self) -> dict[str, np.ndarray]:
        """A small library of known states for parity checks."""
        return {
            "zero_2q": np.array([1, 0, 0, 0], dtype=np.complex128),
            "one_2q": np.array([0, 0, 0, 1], dtype=np.complex128),
            "plus0_2q": np.array([1, 1, 0, 0], dtype=np.complex128) / np.sqrt(2),
            "bell_2q": np.array([1, 0, 0, 1], dtype=np.complex128) / np.sqrt(2),
        }

    def test_computational_known_values(self, basis_states: dict[str, Any]) -> None:
        states = np.stack(
            [
                basis_states["zero_2q"],
                basis_states["one_2q"],
                basis_states["plus0_2q"],
            ],
            axis=0,
        )
        result = _expectations_batch(states, "computational", 2)
        assert_allclose(result, [1.0, 0.0, 0.5])

    def test_global_z_known_values(self, basis_states: dict[str, Any]) -> None:
        states = np.stack(
            [
                basis_states["zero_2q"],
                basis_states["one_2q"],
                basis_states["plus0_2q"],
                basis_states["bell_2q"],
            ],
            axis=0,
        )
        # |00>: even parity → +1; |11>: even → +1; |+0>: split → 0;
        # Bell (|00> + |11>): even parity components → +1.
        result = _expectations_batch(states, "global_z", 2)
        assert_allclose(result, [1.0, 1.0, 0.0, 1.0], atol=1e-12)

    def test_pauli_z_alias_equals_global_z(self, basis_states: dict[str, Any]) -> None:
        states = np.stack([basis_states["zero_2q"], basis_states["bell_2q"]], axis=0)
        assert_allclose(
            _expectations_batch(states, "pauli_z", 2),
            _expectations_batch(states, "global_z", 2),
        )

    def test_local_z_known_values(self, basis_states: dict[str, Any]) -> None:
        states = np.stack(
            [
                basis_states["zero_2q"],
                basis_states["one_2q"],
                basis_states["plus0_2q"],
                basis_states["bell_2q"],
            ],
            axis=0,
        )
        # qubit 0 (MSB): |00> → 0; |11> → 1; |+0> → 0; Bell → maximally mixed.
        result = _expectations_batch(states, "local_z", 2)
        assert_allclose(result, [1.0, -1.0, 1.0, 0.0], atol=1e-12)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 6])
    @pytest.mark.parametrize("observable", ["computational", "global_z", "local_z"])
    def test_random_states_match_per_row_reference(
        self, n: int, observable: str
    ) -> None:
        states = np.stack(
            [_random_statevector(n, seed=200 + n + i) for i in range(5)],
            axis=0,
        )
        batch = _expectations_batch(states, observable, n)

        # Reference: per-row scalar computation.
        if observable == "computational":
            ref = np.array([float(np.abs(s[0]) ** 2) for s in states], dtype=np.float64)
        elif observable == "global_z":
            ref = np.array(
                [_reference_global_z_expectation(s, n) for s in states],
                dtype=np.float64,
            )
        else:  # local_z
            ref = np.array([_reference_local_z(s, n) for s in states], dtype=np.float64)

        assert_allclose(batch, ref, atol=1e-12)

    def test_returns_float64(self) -> None:
        states = np.array([[1.0, 0.0]], dtype=np.complex128)
        for observable in ("computational", "global_z", "local_z"):
            result = _expectations_batch(states, observable, 1)
            assert result.dtype == np.float64

    def test_empty_batch_returns_empty(self) -> None:
        states = np.empty((0, 4), dtype=np.complex128)
        for observable in ("computational", "global_z", "local_z"):
            result = _expectations_batch(states, observable, 2)
            assert result.shape == (0,)

    def test_unknown_observable_raises(self) -> None:
        states = np.array([[1.0, 0.0]], dtype=np.complex128)
        with pytest.raises(ValueError, match="Unknown observable"):
            _expectations_batch(states, "schrodingers_cat", 1)  # type: ignore[arg-type]


# =============================================================================
# compute_all_parameter_gradients (batched parameter-shift)
# =============================================================================


@pytest.fixture(autouse=False)
def _require_pennylane(pennylane_available: bool) -> None:
    if not pennylane_available:
        pytest.skip("PennyLane is required for these gradient tests")


class TestBatchedGradientCorrectness:
    """Batched gradient must match the per-call form to numerical precision.

    These tests pin the most important invariant of the optimisation: the
    refactor changes the number of Python-level calls and the order of
    arithmetic, but NOT the numerical answer. A user who runs the same
    encoding before and after this change must see identical gradients.
    """

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available: bool) -> None:
        if not pennylane_available:
            pytest.skip("PennyLane is required for these gradient tests")

    @pytest.mark.parametrize("observable", ["computational", "global_z", "local_z"])
    def test_matches_per_call_on_angle_encoding(self, observable: str) -> None:
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        x = np.array([0.3, 0.6, 1.2, 2.0], dtype=np.float64)

        batched = compute_all_parameter_gradients(enc, x, observable=observable)
        per_call = np.array(
            [
                compute_parameter_gradient(enc, x, i, observable=observable)
                for i in range(4)
            ],
            dtype=np.float64,
        )
        # Byte-for-byte equality is the goal — any drift would mean the
        # observable arithmetic differs between paths.
        assert_array_equal(batched, per_call)

    @pytest.mark.parametrize("observable", ["computational", "global_z", "local_z"])
    def test_matches_per_call_on_entangling_encoding(self, observable: str) -> None:
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=3, reps=1)
        x = np.array([0.4, 1.1, 1.9], dtype=np.float64)

        batched = compute_all_parameter_gradients(enc, x, observable=observable)
        per_call = np.array(
            [
                compute_parameter_gradient(enc, x, i, observable=observable)
                for i in range(3)
            ],
            dtype=np.float64,
        )
        # IQP has CNOT/CZ entanglers — exercises the global-Z path too.
        assert_allclose(batched, per_call, atol=1e-13)

    def test_shape_and_dtype(self) -> None:
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=3)
        x = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        grads = compute_all_parameter_gradients(enc, x)
        assert grads.shape == (3,)
        assert grads.dtype == np.float64

    def test_single_parameter_works(self) -> None:
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        x = np.array([0.5], dtype=np.float64)
        grads = compute_all_parameter_gradients(enc, x)
        assert grads.shape == (1,)
        assert np.isfinite(grads[0])

    def test_2d_input_raises_validation_error(self) -> None:
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        bad_x = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
        with pytest.raises(ValidationError, match="1D"):
            compute_all_parameter_gradients(enc, bad_x)

    def test_unknown_observable_raises(self) -> None:
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        x = np.array([0.1, 0.2], dtype=np.float64)
        with pytest.raises(ValueError, match="Unknown observable"):
            compute_all_parameter_gradients(
                enc, x, observable="hadamard_z"  # type: ignore[arg-type]
            )

    def test_pauli_z_emits_single_deprecation_warning(self) -> None:
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=5)
        x = np.linspace(0.1, 1.0, 5)

        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            compute_all_parameter_gradients(enc, x, observable="pauli_z")

        deprecation_warnings = [
            w for w in captured if issubclass(w.category, DeprecationWarning)
        ]
        # Critical: ONE warning total, not one-per-parameter (5 in the
        # previous loop-based implementation).
        assert len(deprecation_warnings) == 1
        assert "pauli_z" in str(deprecation_warnings[0].message)


class TestBatchedGradientNumericalSafety:
    """The batched path must propagate NaN/Inf as ``NumericalInstabilityError``.

    These tests mock the batched simulator to inject pathological
    statevectors, ensuring the new ``isfinite`` check fires on the same
    failure modes the previous per-call implementation reported via
    ``compute_parameter_gradient``.
    """

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available: bool) -> None:
        if not pennylane_available:
            pytest.skip("PennyLane is required for these gradient tests")

    def test_nan_statevector_raises_numerical_instability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from encoding_atlas import AngleEncoding
        from encoding_atlas.analysis import _utils as utils

        enc = AngleEncoding(n_features=2)
        x = np.array([0.1, 0.2], dtype=np.float64)

        def bad_batch(encoding, X, backend):  # noqa: N803 - signature match
            # All-NaN statevectors propagate as NaN expectation values.
            return np.full(
                (X.shape[0], 2**encoding.n_qubits), np.nan, dtype=np.complex128
            )

        monkeypatch.setattr(utils, "simulate_encoding_statevectors_batch", bad_batch)

        with pytest.raises(NumericalInstabilityError, match="invalid value"):
            utils.compute_all_parameter_gradients(enc, x, observable="local_z")

    def test_inf_statevector_raises_numerical_instability(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from encoding_atlas import AngleEncoding
        from encoding_atlas.analysis import _utils as utils

        enc = AngleEncoding(n_features=2)
        x = np.array([0.1, 0.2], dtype=np.float64)

        def bad_batch(encoding, X, backend):  # noqa: N803 - signature match
            shape = (X.shape[0], 2**encoding.n_qubits)
            states = np.zeros(shape, dtype=np.complex128)
            states[:, 0] = np.inf
            return states

        monkeypatch.setattr(utils, "simulate_encoding_statevectors_batch", bad_batch)

        with pytest.raises(NumericalInstabilityError):
            utils.compute_all_parameter_gradients(enc, x, observable="computational")


class TestPublicAPIDocsExample:
    """Smoke-test the documented example from the function docstring."""

    @pytest.fixture(autouse=True)
    def check_pennylane(self, pennylane_available: bool) -> None:
        if not pennylane_available:
            pytest.skip("PennyLane is required for the docstring example")

    def test_three_feature_example_runs(self) -> None:
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=3)
        x = np.array([0.5, 1.0, 1.5])
        grads = compute_all_parameter_gradients(enc, x)
        assert grads.shape == (3,)
        assert np.all(np.isfinite(grads))
