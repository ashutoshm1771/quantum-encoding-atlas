"""Noise-resilience analysis for quantum encodings.

Quantifies how well an encoding's prepared state survives realistic gate noise
by simulating the circuit under a depolarizing noise model (via PennyLane's
``default.mixed`` density-matrix simulator) and measuring the retained fidelity
between the ideal and noisy states:

    F(x) = <phi(x)| rho_noisy(x) |phi(x)> ,   resilience = mean_x F(x) in [0, 1].

This is the fidelity-based quantity that underlies the benchmark's headline
noise result (entangling encodings suffer far larger fidelity decay than
non-entangling ones). It completes the "understanding" layer: noise resilience
is the one atlas axis that previously had no shipped way to compute it for an
arbitrary encoding.

Noise model
-----------
A depolarizing channel is inserted after every gate — probability
``single_qubit`` after one-qubit gates and ``two_qubit`` after each qubit of a
two-qubit gate (an independent single-qubit approximation). The depolarizing
channel maps ``rho -> (1 - p) rho + p I / d``. Three preset levels approximate
low / medium / high hardware error rates (:data:`NOISE_LEVELS`).

Notes
-----
This ``retained_fidelity`` (and its complement ``fidelity_decay``) is bounded in
``[0, 1]`` and directly verifiable. It is distinct from the atlas's
``noise_resilience`` column, which is a derived trade-off score
(``1 - expressibility_change``) and can exceed 1; the fidelity-based metric here
is the principled, well-defined quantity the noise result is built on.

References
----------
Preskill (2018), *Quantum* 2:79; Arute et al. (2019), *Nature* 574:505.
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from encoding_atlas.analysis._utils import simulate_encoding_statevector

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding

logger = logging.getLogger(__name__)

# Preset depolarizing error rates (single-/two-qubit gate error probabilities),
# approximating low / medium / high hardware noise.
NOISE_LEVELS: dict[str, dict[str, float]] = {
    "low": {"single_qubit": 0.001, "two_qubit": 0.01},
    "medium": {"single_qubit": 0.005, "two_qubit": 0.05},
    "high": {"single_qubit": 0.01, "two_qubit": 0.10},
}

# Density-matrix simulation is O(4^n); warn beyond this and refuse beyond the cap.
_WARN_QUBITS = 8
_MAX_NOISE_QUBITS = 12

# Default feature sampling range for the fidelity average (radians).
_DEFAULT_RANGE = (0.0, 2.0 * math.pi)


@dataclass(frozen=True)
class NoiseResilienceResult:
    """Result of a noise-resilience analysis.

    Attributes
    ----------
    retained_fidelity : float
        Mean fidelity between the ideal and noisy states over the sampled
        inputs, in ``[0, 1]`` (1 = perfectly resilient).
    fidelity_decay : float
        ``1 - retained_fidelity``: the fraction of state fidelity lost to noise.
    noise_level : str
        The preset level used, or ``"custom"`` for explicit ``noise_params``.
    single_qubit_error, two_qubit_error : float
        The depolarizing probabilities applied after one- and two-qubit gates.
    n_samples : int
        Number of random inputs averaged over.
    std_fidelity : float
        Standard deviation of the per-input fidelities.
    min_fidelity, max_fidelity : float
        Extremes of the per-input fidelities.
    """

    retained_fidelity: float
    fidelity_decay: float
    noise_level: str
    single_qubit_error: float
    two_qubit_error: float
    n_samples: int
    std_fidelity: float
    min_fidelity: float
    max_fidelity: float


def _resolve_noise(
    noise_level: str, noise_params: dict[str, float] | None
) -> tuple[float, float, str]:
    """Resolve ``(single_qubit_error, two_qubit_error, label)``."""
    if noise_params is not None:
        single = float(noise_params.get("single_qubit", 0.0))
        two = float(noise_params.get("two_qubit", 0.0))
        label = "custom"
    else:
        if noise_level not in NOISE_LEVELS:
            raise ValueError(
                f"Unknown noise_level {noise_level!r}; "
                f"valid: {sorted(NOISE_LEVELS)} (or pass noise_params)."
            )
        single = NOISE_LEVELS[noise_level]["single_qubit"]
        two = NOISE_LEVELS[noise_level]["two_qubit"]
        label = noise_level
    for name, value in (("single_qubit", single), ("two_qubit", two)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} error probability must be in [0, 1], got {value}")
    return single, two, label


def _check_qubits(n_qubits: int) -> None:
    """Guard density-matrix simulation memory (O(4^n))."""
    if n_qubits > _MAX_NOISE_QUBITS:
        raise ValueError(
            f"Noisy (density-matrix) simulation is limited to "
            f"{_MAX_NOISE_QUBITS} qubits; encoding has {n_qubits}. Memory scales "
            f"as O(4^n)."
        )
    if n_qubits > _WARN_QUBITS:
        warnings.warn(
            f"Noisy simulation with {n_qubits} qubits requires a "
            f"{(4**n_qubits) * 16 / 1024**2:.0f} MB density matrix and may be slow.",
            stacklevel=3,
        )


def simulate_noisy_density_matrix(
    encoding: BaseEncoding,
    x: NDArray[np.floating[Any]],
    *,
    single_qubit_error: float,
    two_qubit_error: float,
) -> NDArray[np.complexfloating[Any, Any]]:
    """Simulate the encoding under depolarizing noise; return the density matrix.

    A depolarizing channel is inserted after each gate: ``single_qubit_error``
    after one-qubit gates, and ``two_qubit_error`` on each qubit of two-qubit
    (or larger) gates. Uses PennyLane's ``default.mixed`` device.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding to simulate.
    x : ndarray, shape (n_features,)
        Input to encode.
    single_qubit_error, two_qubit_error : float
        Depolarizing probabilities in ``[0, 1]``.

    Returns
    -------
    ndarray, shape (2**n_qubits, 2**n_qubits)
        The noisy density matrix.
    """
    import pennylane as qml

    n_qubits = encoding.n_qubits
    circuit_fn = encoding.get_circuit(x, backend="pennylane")
    if not callable(circuit_fn):
        raise ValueError(
            f"Encoding returned a non-callable circuit: {type(circuit_fn).__name__}"
        )
    device = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(device)  # type: ignore[untyped-decorator]
    def noisy_circuit() -> Any:
        with qml.queuing.AnnotatedQueue() as queue:
            circuit_fn()
        tape = qml.tape.QuantumScript.from_queue(queue)
        for op in tape.operations:
            qml.apply(op)
            wires = list(op.wires)
            if len(wires) == 1:
                if single_qubit_error > 0:
                    qml.DepolarizingChannel(single_qubit_error, wires=wires[0])
            elif two_qubit_error > 0:
                for wire in wires:
                    qml.DepolarizingChannel(two_qubit_error, wires=wire)
        return qml.density_matrix(wires=range(n_qubits))

    return np.asarray(noisy_circuit(), dtype=np.complex128)


def compute_noise_resilience(
    encoding: BaseEncoding,
    *,
    noise_level: str = "medium",
    noise_params: dict[str, float] | None = None,
    n_samples: int = 25,
    seed: int | None = None,
    feature_range: tuple[float, float] = _DEFAULT_RANGE,
) -> NoiseResilienceResult:
    """Measure an encoding's resilience to depolarizing gate noise.

    For ``n_samples`` random inputs, simulates the ideal state ``|phi(x)>`` and
    the noisy density matrix ``rho(x)`` and averages the fidelity
    ``F(x) = <phi(x)| rho(x) |phi(x)>``. Higher retained fidelity means the
    encoding degrades less under noise.

    Parameters
    ----------
    encoding : BaseEncoding
        Encoding to analyze (at most 12 qubits; density-matrix memory is
        O(4^n)).
    noise_level : {"low", "medium", "high"}, default="medium"
        Preset depolarizing error rates (ignored if ``noise_params`` is given).
    noise_params : dict or None, default=None
        Explicit ``{"single_qubit": p1, "two_qubit": p2}`` error probabilities,
        overriding ``noise_level``.
    n_samples : int, default=25
        Number of random inputs to average over. Depolarizing fidelity is
        nearly input-independent, so modest values are accurate.
    seed : int or None, default=None
        Seed for input sampling (reproducibility).
    feature_range : tuple, default=(0, 2*pi)
        Range for uniformly sampled input features.

    Returns
    -------
    NoiseResilienceResult
        Retained fidelity, fidelity decay, and per-sample statistics.

    Raises
    ------
    ValueError
        If ``n_samples < 1``, the encoding exceeds the qubit cap, or the noise
        specification is invalid.
    """
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    single, two, label = _resolve_noise(noise_level, noise_params)
    _check_qubits(encoding.n_qubits)

    rng = np.random.default_rng(seed)
    n_features = encoding.n_features
    low, high = feature_range

    fidelities: NDArray[np.float64] = np.empty(n_samples, dtype=np.float64)
    for i in range(n_samples):
        x = rng.uniform(low, high, size=n_features)
        psi = simulate_encoding_statevector(encoding, x, backend="pennylane")
        rho = simulate_noisy_density_matrix(
            encoding, x, single_qubit_error=single, two_qubit_error=two
        )
        fidelity = float(np.real(np.conj(psi) @ rho @ psi))
        fidelities[i] = min(1.0, max(0.0, fidelity))

    retained = float(fidelities.mean())
    return NoiseResilienceResult(
        retained_fidelity=retained,
        fidelity_decay=1.0 - retained,
        noise_level=label,
        single_qubit_error=single,
        two_qubit_error=two,
        n_samples=n_samples,
        std_fidelity=float(fidelities.std()),
        min_fidelity=float(fidelities.min()),
        max_fidelity=float(fidelities.max()),
    )
