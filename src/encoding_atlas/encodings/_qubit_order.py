"""Qubit-ordering conversion for amplitude-vector state preparation.

Most encodings build their circuit gate by gate. There, a wire index means the
same physical qubit in every framework, and the single global conversion in
:func:`encoding_atlas.analysis._utils._reverse_qubit_order` is enough to
reconcile Qiskit's readout convention with the library's.

State preparation is the exception. ``qml.StatePrep`` and
``QuantumCircuit.initialize`` both take a whole amplitude vector, and they
disagree about what an *index* into that vector means:

=========  =====================  =====================
Index      MSB (PennyLane, here)  LSB (Qiskit)
=========  =====================  =====================
0 (00)     q0=0, q1=0             q0=0, q1=0
1 (01)     q0=0, q1=1             q0=1, q1=0
2 (10)     q0=1, q1=0             q0=0, q1=1
3 (11)     q0=1, q1=1             q0=1, q1=1
=========  =====================  =====================

Handing the same array to both therefore prepares *different physical states*,
and the global readout conversion then compounds the error rather than fixing
it. Any encoding that prepares a state from an amplitude vector must permute it
with :func:`msb_to_lsb_amplitudes` before calling ``initialize``.

This module exists so that requirement lives in one tested place: the bug it
guards against is silent — the circuit builds, the simulation runs, and only
the state is wrong.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["msb_to_lsb_amplitudes"]


def msb_to_lsb_amplitudes(
    amplitudes: NDArray[Any],
    n_qubits: int,
) -> NDArray[Any]:
    """Reorder an amplitude vector from MSB-first to LSB-first indexing.

    The permutation is a bit-reversal of the amplitude indices, applied as a
    tensor transpose rather than an index loop so the cost stays linear in the
    vector length.

    The map is its own inverse: applying it twice returns the original vector.

    Parameters
    ----------
    amplitudes : ndarray of shape (2**n_qubits,)
        Amplitudes indexed MSB-first, the convention used throughout this
        library and by PennyLane.
    n_qubits : int
        Number of qubits. Must be a positive integer, and ``amplitudes`` must
        have exactly ``2**n_qubits`` entries.

    Returns
    -------
    ndarray of shape (2**n_qubits,)
        The same amplitudes indexed LSB-first, ready for
        :meth:`qiskit.QuantumCircuit.initialize`. A contiguous copy, so the
        caller's array is never aliased or mutated.

    Raises
    ------
    ValueError
        If ``n_qubits`` is not a positive integer, or ``amplitudes`` is not a
        1D array of length ``2**n_qubits``.

    Examples
    --------
    Index 1 (``01``) and index 2 (``10``) swap; the palindromic indices do not:

    >>> import numpy as np
    >>> msb_to_lsb_amplitudes(np.array([0.0, 1.0, 2.0, 3.0]), 2)
    array([0., 2., 1., 3.])

    The conversion is an involution:

    >>> state = np.arange(8.0)
    >>> back = msb_to_lsb_amplitudes(msb_to_lsb_amplitudes(state, 3), 3)
    >>> bool(np.array_equal(back, state))
    True
    """
    if isinstance(n_qubits, bool) or not isinstance(n_qubits, (int, np.integer)):
        raise ValueError(f"n_qubits must be a positive integer, got {n_qubits!r}")
    if n_qubits < 1:
        raise ValueError(f"n_qubits must be a positive integer, got {n_qubits!r}")

    array = np.asarray(amplitudes)
    if array.ndim != 1:
        raise ValueError(f"amplitudes must be 1D, got shape {array.shape}")
    expected = 1 << int(n_qubits)
    if array.shape[0] != expected:
        raise ValueError(
            f"amplitudes must have 2**{n_qubits} = {expected} entries, "
            f"got {array.shape[0]}"
        )

    reordered: NDArray[Any] = np.ascontiguousarray(
        array.reshape([2] * int(n_qubits))
        .transpose(list(reversed(range(int(n_qubits)))))
        .ravel()
    )
    return reordered
