"""Shared parallelization helpers for analysis sampling loops.

The three core analysis functions (``compute_expressibility``,
``compute_entanglement_capability``, ``estimate_trainability``) all sample
many independent quantum circuits and aggregate their results. Their hot
paths are embarrassingly parallel, and this module supplies the small
amount of shared infrastructure that lets each of them dispatch
sequentially, on a thread pool, or on a process pool with the same public
API.

API consistency
---------------
The ``parallel`` argument mirrors the one already exposed on
:meth:`encoding_atlas.core.base.BaseEncoding.get_circuits`:

* ``False`` (default) — sequential, no executor overhead.
* ``True`` or ``'thread'`` — :class:`concurrent.futures.ThreadPoolExecutor`.
* ``'process'`` — :class:`concurrent.futures.ProcessPoolExecutor`.

Determinism
-----------
The analysis callers are required to pre-generate every random input in
the *main* process before dispatching. Workers receive the inputs and
perform only deterministic computation (statevector simulation,
gradient evaluation, entanglement measure). This guarantees that for a
fixed seed the numerical output is identical across parallelization
modes — sequential, thread pool, and process pool all produce the same
result, byte-for-byte where floats allow.

Pickling caveats
----------------
ProcessPoolExecutor exchanges all arguments and return values via
``pickle``. The analysis workers do not return circuit objects (which
would fail for PennyLane's local-closure qfuncs); they only return
numpy arrays / floats / Python tuples, which are universally
picklable. This is what allows ``parallel='process'`` to work for all
three backends in the analysis path, unlike
:meth:`BaseEncoding.get_circuits`.
"""

from __future__ import annotations

from typing import Literal, Union

# Public type alias re-used by every analysis function's signature.
ParallelArg = Union[bool, Literal["thread", "process"]]
ParallelMode = Literal["sequential", "thread", "process"]


def resolve_parallel_mode(parallel: ParallelArg) -> ParallelMode:
    """Normalize the public ``parallel`` argument to an internal mode tag.

    Parameters
    ----------
    parallel : bool or {'thread', 'process'}
        Public-facing parallelization selector. ``True`` is preserved as
        an alias for ``'thread'`` so callers don't have to update their
        existing ``parallel=True`` invocations.

    Returns
    -------
    {'sequential', 'thread', 'process'}
        Internal mode label.

    Raises
    ------
    ValueError
        If ``parallel`` is none of the accepted values. The error message
        lists exactly what is accepted so users can self-correct quickly.
    """
    if parallel is False:
        return "sequential"
    if parallel is True or parallel == "thread":
        return "thread"
    if parallel == "process":
        return "process"
    raise ValueError(
        f"parallel must be False, True, 'thread', or 'process', " f"got {parallel!r}"
    )
