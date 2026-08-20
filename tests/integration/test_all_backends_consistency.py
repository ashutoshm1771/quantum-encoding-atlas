"""Every encoding must prepare the same state on every backend.

The library advertises PennyLane, Qiskit and Cirq support and documents an
MSB/LSB conversion to keep them consistent. That guarantee was previously
spot-checked on five encodings, which let a real bug ship:
``SO2EquivariantFeatureMap`` prepared a bit-reversed state on Qiskit, because
``QuantumCircuit.initialize`` reads its amplitude vector LSB-first while
``qml.StatePrep`` reads it MSB-first. PennyLane-vs-Qiskit fidelity was 0.0006 —
near-orthogonal — on every input tried.

This module replaces the spot check with a sweep over the whole registry, so a
new encoding is covered the moment it is registered rather than when someone
remembers to add a case. The parametrisation is generated from
``BENCHMARK_PARAMS``; nothing here needs editing to cover encoding number 17.

Backends that are not installed are skipped individually, and
``TestBackendsAvailable`` records which ones actually ran, so a silently
degraded environment is visible in the report instead of passing quietly. Under
``ENCODING_ATLAS_REQUIRE_ALL_BACKENDS=1`` — which CI sets — a missing backend
fails instead of skipping; see :mod:`tests._backends`.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pytest

from encoding_atlas.analysis._utils import (
    _reverse_qubit_order,
    simulate_encoding_statevector,
)
from encoding_atlas.core.registry import get_encoding
from encoding_atlas.guide._candidates import BENCHMARK_PARAMS
from tests._backends import (
    ALL_BACKENDS,
    OPTIONAL_BACKENDS,
    backend_is_installed,
    missing_backends,
    require_backend,
)

# Widths spanning the odd/even and power-of-two cases the encodings branch on.
FEATURE_COUNTS = (2, 3, 4, 6)

# Inputs are fixed rather than random so a failure is reproducible verbatim.
INPUTS = (
    np.array([0.3, 0.7, -0.4, 0.9, 0.15, -0.6, 0.5, 0.25]),
    np.array([-0.918, -0.967, 0.274, -0.46, 0.627, 0.826, -0.21, 0.71]),
)

# Statevector agreement is exact up to floating point; this is not a tolerance
# for "close enough" physics, it is round-off headroom.
FIDELITY_TOL = 1e-9


def _cases() -> list[tuple[str, int]]:
    """Every (encoding, feature count) the registry can actually build."""
    cases = []
    for name, params in BENCHMARK_PARAMS.items():
        for n_features in FEATURE_COUNTS:
            try:
                get_encoding(name, n_features=n_features, **params)
            except Exception:
                continue
            cases.append((name, n_features))
    return cases


CASES = _cases()
CASE_IDS = [f"{name}-d{d}" for name, d in CASES]


def _statevector(name: str, n_features: int, backend: str, x: Any) -> Any:
    encoding = get_encoding(name, n_features=n_features, **BENCHMARK_PARAMS[name])
    with warnings.catch_warnings():
        # Encodings warn about input ranges; irrelevant to backend agreement.
        warnings.simplefilter("ignore")
        return simulate_encoding_statevector(encoding, x[:n_features], backend=backend)


class TestBackendsAvailable:
    """Make a degraded environment visible rather than silently green."""

    def test_pennylane_is_available(self) -> None:
        """PennyLane is a hard dependency; its absence is a broken install."""
        assert backend_is_installed("pennylane")

    @pytest.mark.parametrize("backend", OPTIONAL_BACKENDS)
    def test_optional_backend_reported(self, backend: str) -> None:
        require_backend(
            backend, reason="its cross-backend consistency tests did not run"
        )
        assert backend_is_installed(backend)

    def test_no_advertised_backend_is_missing(self) -> None:
        """One aggregate report, so a degraded environment reads at a glance.

        Under ``ENCODING_ATLAS_REQUIRE_ALL_BACKENDS=1`` the session never gets
        this far — ``pytest_configure`` refuses to start. This is the same
        check without the flag, downgraded to a skip so local runs stay usable.
        """
        missing = missing_backends(ALL_BACKENDS)
        if missing:
            pytest.skip(
                f"not every advertised backend is installed: {missing}. "
                f"The consistency guarantee is verified only for the rest."
            )
        assert not missing


@pytest.mark.parametrize(("name", "n_features"), CASES, ids=CASE_IDS)
@pytest.mark.parametrize("backend", OPTIONAL_BACKENDS)
class TestCrossBackendConsistency:
    """Each optional backend must reproduce the PennyLane reference state."""

    def test_prepares_the_same_state(
        self, backend: str, name: str, n_features: int
    ) -> None:
        require_backend(
            backend,
            reason=f"{name} was not checked for agreement with the "
            f"PennyLane reference state",
        )

        for x in INPUTS:
            reference = _statevector(name, n_features, "pennylane", x)
            other = _statevector(name, n_features, backend, x)

            assert other.shape == reference.shape, (
                f"{name} d={n_features}: {backend} returned shape {other.shape}, "
                f"PennyLane returned {reference.shape}"
            )

            # Fidelity, so a physically irrelevant global phase does not fail.
            fidelity = float(abs(np.vdot(reference, other)) ** 2)
            if fidelity >= 1.0 - FIDELITY_TOL:
                continue

            # Diagnose before failing: a bit-reversal means a qubit-ordering
            # bug (the SO2 failure mode), not a different circuit.
            n_qubits = int(np.log2(reference.shape[0]))
            reversed_fidelity = float(
                abs(np.vdot(reference, _reverse_qubit_order(other, n_qubits))) ** 2
            )
            hint = (
                " The state is BIT-REVERSED: the backend implementation is using "
                "the opposite qubit convention. See "
                "encoding_atlas.encodings._qubit_order."
                if reversed_fidelity >= 1.0 - FIDELITY_TOL
                else " The states differ genuinely, not just in qubit ordering."
            )
            pytest.fail(
                f"{name} (n_features={n_features}) disagrees between pennylane "
                f"and {backend}: fidelity={fidelity:.6f} on input "
                f"{np.round(x[:n_features], 3).tolist()}.{hint}"
            )

    def test_normalised_and_same_width(
        self, backend: str, name: str, n_features: int
    ) -> None:
        require_backend(
            backend,
            reason=f"{name} was not checked for normalisation and register "
            f"width on this backend",
        )
        state = _statevector(name, n_features, backend, INPUTS[0])
        assert np.isclose(np.linalg.norm(state), 1.0, atol=1e-9)
        encoding = get_encoding(name, n_features=n_features, **BENCHMARK_PARAMS[name])
        assert state.shape == (2**encoding.n_qubits,)


class TestRegressionSO2Qiskit:
    """The specific bug this module was written to prevent recurring."""

    def test_so2_agrees_across_backends(self) -> None:
        require_backend(
            "qiskit",
            reason="the SO(2) bit-reversal regression test did not run",
        )
        for x in INPUTS:
            reference = _statevector("so2_equivariant", 2, "pennylane", x)
            qiskit_state = _statevector("so2_equivariant", 2, "qiskit", x)
            fidelity = float(abs(np.vdot(reference, qiskit_state)) ** 2)
            assert fidelity == pytest.approx(1.0, abs=FIDELITY_TOL), (
                "SO2EquivariantFeatureMap regressed to preparing a bit-reversed "
                f"state on Qiskit (fidelity {fidelity:.6f})"
            )

    def test_state_preparation_encodings_permute_amplitudes(self) -> None:
        """Both amplitude-vector encodings must use the shared permutation.

        A future state-preparation encoding that calls ``initialize`` without
        it would fail the sweep above; this pins the two that exist today.
        """
        import inspect

        from encoding_atlas.encodings import amplitude, equivariant_feature_map

        for module in (amplitude, equivariant_feature_map):
            source = inspect.getsource(module)
            if "initialize(" in source:
                assert "msb_to_lsb_amplitudes" in source, (
                    f"{module.__name__} calls QuantumCircuit.initialize without "
                    f"the MSB->LSB amplitude permutation"
                )
