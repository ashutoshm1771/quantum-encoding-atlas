"""Data-driven encoding screening.

:func:`encoding_atlas.guide.recommend_encoding` answers from *metadata* — how
many features, which task, which hardware. It never sees the data. But the
benchmark's central positive result is that a training-free quantity computed
*on your data*, the centered kernel-target alignment, tracks downstream kernel
accuracy closely (Spearman rho = 0.91 across encodings and datasets). This
module turns that result into the workflow it implies: compute the alignment
for every candidate encoding on your own ``(X, y)``, and train only the ones
that score well.

Read the ranking as a shortlist, not an oracle
----------------------------------------------
On the benchmark's eight datasets, taking the single highest-alignment
encoding scores 0.960 mean accuracy against an oracle's 0.974 — but simply
always choosing ``angle`` scores 0.958, so the *top-1* pick is not meaningfully
better than that default. Taking the **top three** reaches 0.973, cutting the
regret against the oracle roughly twelve-fold while training three encodings
instead of sixteen.

Screening is therefore worth it for the shortlist and for adapting to data the
defaults do not suit, not as a way to skip the search entirely. That is also
exactly what the study prescribes: score, keep the encodings that do well,
train only those. :meth:`ScreeningResult.top` defaults to three for this
reason.

Cost
----
Alignment needs one statevector per sample per encoding and no training, so a
full 16-encoding screen on 100 samples takes on the order of a second. Larger
datasets are sub-sampled (stratified, seeded) to keep it that way.

Scaling is part of the search
-----------------------------
The range features are scaled into is not a fixed detail — for several
encodings it moves accuracy further than the choice of encoding does, because
a full ``2*pi`` sweep drives the kernel onto its concentration floor. Pass
``feature_ranges=`` to rank over (encoding, range) pairs rather than encodings
alone; see :mod:`encoding_atlas.analysis.scaling`.

Notes
-----
Alignment is a two-class quantity: labels are mapped to ``{-1, +1}`` by
partition, so multi-class and regression targets are rejected rather than
silently scored. Concentration is available as an annotation but is
deliberately *not* part of the ranking — at a fixed circuit width alignment
already ranks the Haar-floor encodings last, so a concentration veto would be
redundant. Its value is telling you whether the ranking will survive at wider
circuits.

References
----------
Cristianini et al. (2002), *NeurIPS* 14 — kernel-target alignment.
Cortes, Mohri & Rostamizadeh (2012), *JMLR* 13:795 — centered alignment.
Hubregtsen et al. (2022), *Phys. Rev. A* 106:042431 — training embedding
kernels by alignment.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray

from encoding_atlas.guide._candidates import BENCHMARK_PARAMS, build_candidates

if TYPE_CHECKING:
    from encoding_atlas.core.base import BaseEncoding

logger = logging.getLogger(__name__)

__all__ = [
    "ScreenedEncoding",
    "ScreeningResult",
    "screen_encodings",
]

# Alignment is O(n^2) in kernel entries and O(n) in circuit simulations, so the
# screen sub-samples above this. 200 samples give 19,900 off-diagonal pairs,
# well past the point where the ranking stabilises.
_DEFAULT_MAX_SAMPLES: int = 200

# Default shortlist size. See the module docstring: top-3 captures nearly all
# of the achievable accuracy, top-1 does not beat the always-angle default.
_DEFAULT_TOP_K: int = 3


@dataclass(frozen=True)
class ScreenedEncoding:
    """One candidate encoding, scored on the user's data.

    Attributes
    ----------
    name : str
        Canonical encoding name, e.g. ``"angle"``.
    display_name : str
        Class name of the built encoding, e.g. ``"AngleEncoding"``.
    encoding : BaseEncoding
        The built instance, ready to train with — no need to rebuild it.
    params : Mapping[str, Any]
        Fixed parameters used, matching the benchmark's configuration.
    n_qubits : int
        Circuit width at the caller's feature count.
    alignment : float
        Kernel-target alignment on the supplied data, in ``[-1, 1]``. Centred
        by default. This is the ranking key.
    rank : int
        1-based position in the screen, 1 = highest alignment.
    atlas_rank : int or None
        The encoding's overall benchmark rank, or ``None`` if not benchmarked.
    atlas_alignment : float or None
        Mean alignment the benchmark measured, for reference. A large gap
        against :attr:`alignment` means your data behaves differently from the
        benchmark's, which is exactly when screening pays off.
    feature_range : tuple[float, float] or None
        The range the features were scaled into for this measurement, when
        ``feature_ranges`` was supplied. ``None`` means the data was used as
        given. Candidates are ranked across (encoding, range) pairs, so the
        same encoding can appear more than once at different ranges.
    concentration_ratio : float or None
        Kernel variance relative to the Haar floor on your data, when
        ``include_concentration=True``. Near 1 means the kernel carries no
        usable geometry at this width.
    is_concentrated : bool or None
        Whether ``concentration_ratio`` fell below the concentration
        threshold. ``None`` when concentration was not requested.
    """

    name: str
    display_name: str
    encoding: BaseEncoding
    params: Mapping[str, Any]
    n_qubits: int
    alignment: float
    rank: int
    feature_range: tuple[float, float] | None
    atlas_rank: int | None
    atlas_alignment: float | None
    concentration_ratio: float | None
    is_concentrated: bool | None


@dataclass(frozen=True)
class ScreeningResult:
    """Ranked outcome of screening candidate encodings on a dataset.

    Attributes
    ----------
    candidates : tuple[ScreenedEncoding, ...]
        Successfully scored encodings, highest alignment first. Ties break on
        name so the order is deterministic.
    skipped : Mapping[str, str]
        Encodings that could not be built or scored at this feature count,
        mapped to the reason.
    n_samples_used : int
        Rows actually used, after any sub-sampling.
    n_samples_supplied : int
        Rows supplied by the caller.
    n_features : int
        Feature count screened at.
    centered : bool
        Whether the centred alignment variant was used.
    shots : int or None
        Shot budget per kernel entry, or ``None`` for exact kernels.
    feature_ranges : tuple[tuple[float, float], ...] or None
        Ranges swept, or ``None`` if the data was used as supplied.
    """

    candidates: tuple[ScreenedEncoding, ...]
    skipped: Mapping[str, str]
    n_samples_used: int
    n_samples_supplied: int
    n_features: int
    centered: bool
    shots: int | None
    feature_ranges: tuple[tuple[float, float], ...] | None = None

    def top(self, k: int = _DEFAULT_TOP_K) -> list[ScreenedEncoding]:
        """Return the ``k`` highest-aligned candidates.

        Defaults to three, the shortlist size the benchmark supports: it
        captures nearly all achievable accuracy, whereas the single top pick
        is no better than the ``angle`` default. See the module docstring.

        Parameters
        ----------
        k : int, default=3
            Shortlist size. Must be a positive integer. Values beyond the
            number of scored candidates return all of them.

        Returns
        -------
        list[ScreenedEncoding]

        Raises
        ------
        ValueError
            If ``k`` is not a positive integer.
        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise ValueError(f"k must be a positive integer, got {k!r}")
        if k < 1:
            raise ValueError(f"k must be a positive integer, got {k!r}")
        return list(self.candidates[:k])

    def names(self, k: int | None = None) -> list[str]:
        """Return candidate names in ranked order, optionally the top ``k``."""
        if k is None:
            return [c.name for c in self.candidates]
        return [c.name for c in self.top(k)]

    def best(self) -> ScreenedEncoding:
        """Return the highest-aligned candidate.

        Raises
        ------
        RuntimeError
            If no candidate could be scored.
        """
        if not self.candidates:
            raise RuntimeError(
                "No encoding could be screened; inspect `skipped` for reasons."
            )
        return self.candidates[0]

    def get(self, name: str) -> ScreenedEncoding | None:
        """Return the scored candidate called ``name``, or ``None``."""
        key = name.strip().lower()
        for candidate in self.candidates:
            if candidate.name.lower() == key:
                return candidate
        return None


class _Scored(NamedTuple):
    """One scored (encoding, range) pair, before ranking."""

    name: str
    encoding: BaseEncoding
    alignment: float
    feature_range: tuple[float, float] | None
    concentration_ratio: float | None
    is_concentrated: bool | None


def _validate_feature_ranges(
    feature_ranges: Sequence[tuple[float, float]] | None,
) -> tuple[tuple[float, float] | None, ...]:
    """Normalise the requested ranges to a non-empty tuple to iterate over.

    ``None`` becomes a single ``None`` entry, meaning "use the data as given".
    """
    if feature_ranges is None:
        return (None,)
    resolved = tuple((float(low), float(high)) for low, high in feature_ranges)
    if not resolved:
        raise ValueError("feature_ranges must not be empty")
    for low, high in resolved:
        if not (math.isfinite(low) and math.isfinite(high)):
            raise ValueError(f"feature range must be finite, got ({low}, {high})")
        if low >= high:
            raise ValueError(
                f"feature range must satisfy low < high, got ({low}, {high})"
            )
    return resolved


def _stratified_subsample(
    n_samples: int,
    y: NDArray[np.float64],
    max_samples: int,
    seed: int | None,
) -> NDArray[np.intp]:
    """Return indices of a class-balanced subsample, or all of them.

    Preserves each class's proportion, keeps at least one row per class, and
    is fully determined by ``seed``.
    """
    if n_samples <= max_samples:
        return np.arange(n_samples, dtype=np.intp)

    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    chosen: list[NDArray[np.intp]] = []
    for index, cls in enumerate(classes):
        members = np.flatnonzero(y == cls).astype(np.intp)
        # Proportional allocation, at least one row so no class disappears.
        quota = max(1, int(round(max_samples * members.size / n_samples)))
        quota = min(quota, members.size)
        # Deterministic per class: permute with a stream derived from `seed`.
        picked = members[rng.permutation(members.size)[:quota]]
        chosen.append(picked)
        del index
    indices = np.concatenate(chosen)
    indices.sort()
    return indices


def screen_encodings(
    X: NDArray[np.floating[Any]],
    y: NDArray[np.integer[Any] | np.floating[Any]],
    *,
    candidates: Sequence[str] | None = None,
    max_samples: int = _DEFAULT_MAX_SAMPLES,
    centered: bool = True,
    include_concentration: bool = False,
    feature_ranges: Sequence[tuple[float, float]] | None = None,
    shots: int | None = None,
    seed: int | None = None,
    backend: Literal["pennylane", "qiskit", "cirq"] = "pennylane",
) -> ScreeningResult:
    """Rank candidate encodings by kernel-target alignment on your data.

    Builds each candidate at ``X.shape[1]`` features, computes its fidelity
    kernel on the data, and scores it by alignment with the labels — no
    training. Encodings that cannot be built at this width (SO(2) equivariance
    needs exactly two features, the swap-equivariant map an even count) are
    reported in :attr:`ScreeningResult.skipped` rather than raising.

    Use the result as a **shortlist**: train the top few and compare them
    properly. The top-1 pick alone is not reliably better than the library's
    metadata recommendation — see the module docstring for the measured
    numbers.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix. Scale it as you would before training; alignment
        depends on the input distribution, not just the circuit.
    y : ndarray of shape (n_samples,)
        Two-class labels. Any two distinct values work — ``{0, 1}``,
        ``{1, 2}`` and ``{-1, +1}`` all describe the same split and score the
        same. Multi-class and continuous targets are rejected.
    candidates : sequence of str, optional
        Restrict to these encodings. Defaults to all 16 benchmarked ones, at
        the benchmark's own parameters.
    max_samples : int, default=200
        Sub-sample (stratified, seeded) above this many rows to keep the
        screen fast. Must be at least 2.
    centered : bool, default=True
        Use the centred alignment (Cortes et al., 2012). This is the variant
        the benchmark validated; leave it on unless you have a reason not to.
    include_concentration : bool, default=False
        Also annotate each candidate with its kernel concentration on your
        data. Computed from the same kernel, so it costs no extra simulation.
        It does not affect the ranking.
    feature_ranges : sequence of (float, float), optional
        Also sweep the range the features are min-max scaled into, ranking
        over (encoding, range) pairs. The range is not a minor knob: for
        several encodings it moves accuracy further than the choice of
        encoding does, because a full ``2*pi`` sweep drives the kernel onto
        its concentration floor. Pass
        :data:`encoding_atlas.analysis.DEFAULT_FEATURE_RANGES` for the usual
        sweep. When ``None`` (default) the data is used exactly as supplied.
    shots : int or None, default=None
        Screen on finite-shot kernel estimates instead of exact ones, to see
        what a device would report. Alignment aggregates over all pairs and so
        tolerates shot noise well.
    seed : int or None, default=None
        Seed for sub-sampling and shot sampling. Fixes the whole screen.
    backend : {"pennylane", "qiskit", "cirq"}, default="pennylane"
        Statevector simulation backend.

    Returns
    -------
    ScreeningResult
        Ranked candidates plus the reasons any were skipped.

    Raises
    ------
    ValueError
        If ``X``/``y`` are malformed or mismatched, ``y`` is not two-class,
        ``max_samples`` is invalid, or ``candidates`` names an unknown
        encoding.

    Examples
    --------
    >>> from encoding_atlas.benchmark import get_dataset
    >>> from encoding_atlas.guide import screen_encodings
    >>> X, y = get_dataset("moons", n_samples=60, seed=0)
    >>> result = screen_encodings(X, y, seed=0)
    >>> shortlist = result.names(3)
    >>> len(shortlist)
    3

    See Also
    --------
    encoding_atlas.guide.recommend_encoding : Metadata-based recommendation.
    encoding_atlas.analysis.compute_kernel_target_alignment : The score.
    encoding_atlas.benchmark.evaluate_encoding : Train the shortlist.
    """
    from encoding_atlas.analysis.concentration import summarize_kernel_concentration
    from encoding_atlas.analysis.generalization import (
        centered_kernel_target_alignment,
        compute_fidelity_kernel,
        kernel_target_alignment,
        validate_binary_labels,
    )
    from encoding_atlas.analysis.scaling import scale_to_range

    if isinstance(max_samples, bool) or not isinstance(max_samples, int):
        raise ValueError(f"max_samples must be an integer >= 2, got {max_samples!r}")
    if max_samples < 2:
        raise ValueError(f"max_samples must be an integer >= 2, got {max_samples!r}")

    X_array = np.asarray(X, dtype=np.float64)
    if X_array.ndim != 2:
        raise ValueError(f"X must be a 2D array, got shape {X_array.shape}")
    if X_array.shape[0] < 2:
        raise ValueError(f"X must contain at least 2 samples, got {X_array.shape[0]}")
    if not np.all(np.isfinite(X_array)):
        raise ValueError("X contains NaN or infinite values")

    y_float = validate_binary_labels(y)
    if y_float.shape[0] != X_array.shape[0]:
        raise ValueError(
            f"X and y must have the same length, got {X_array.shape[0]} and "
            f"{y_float.shape[0]}"
        )

    n_supplied, n_features = X_array.shape
    indices = _stratified_subsample(n_supplied, y_float, max_samples, seed)
    X_used, y_used = X_array[indices], y_float[indices]

    built, skipped = build_candidates(n_features, candidates)
    logger.debug(
        "Screening %d candidate(s) at n_features=%d on %d sample(s)",
        len(built),
        n_features,
        len(X_used),
    )

    resolved_ranges = _validate_feature_ranges(feature_ranges)

    score = centered_kernel_target_alignment if centered else kernel_target_alignment
    scored: list[_Scored] = []
    for name, encoding in built:
        for feature_range in resolved_ranges:
            X_scaled = (
                X_used
                if feature_range is None
                else scale_to_range(X_used, feature_range[0], feature_range[1])
            )
            try:
                K = compute_fidelity_kernel(
                    encoding, X_scaled, backend=backend, shots=shots, seed=seed
                )
                alignment = float(score(K, y_used))
                ratio: float | None = None
                concentrated: bool | None = None
                if include_concentration:
                    summary = summarize_kernel_concentration(K, int(encoding.n_qubits))
                    ratio = summary.concentration_ratio
                    concentrated = summary.is_concentrated
            except Exception as exc:  # noqa: BLE001 - one bad candidate must not abort
                skipped[name] = f"{type(exc).__name__}: {exc}"
                logger.debug("Screening skipped %s: %s", name, exc)
                continue
            scored.append(
                _Scored(name, encoding, alignment, feature_range, ratio, concentrated)
            )

    # Highest alignment first. Ties break on the narrower range (further from
    # the concentration floor) then on name, so the order is deterministic.
    scored.sort(
        key=lambda item: (
            -item.alignment,
            (
                item.feature_range[1] - item.feature_range[0]
                if item.feature_range
                else 0.0
            ),
            item.name,
        )
    )

    atlas_ranks, atlas_alignments = _atlas_reference()
    ranked = tuple(
        ScreenedEncoding(
            name=item.name,
            display_name=type(item.encoding).__name__,
            encoding=item.encoding,
            params=dict(BENCHMARK_PARAMS[item.name]),
            n_qubits=int(item.encoding.n_qubits),
            alignment=item.alignment,
            rank=position,
            feature_range=item.feature_range,
            atlas_rank=atlas_ranks.get(item.name),
            atlas_alignment=atlas_alignments.get(item.name),
            concentration_ratio=item.concentration_ratio,
            is_concentrated=item.is_concentrated,
        )
        for position, item in enumerate(scored, start=1)
    )

    return ScreeningResult(
        candidates=ranked,
        skipped=skipped,
        n_samples_used=int(len(X_used)),
        n_samples_supplied=int(n_supplied),
        n_features=int(n_features),
        centered=bool(centered),
        shots=shots,
        feature_ranges=(
            None if feature_ranges is None else tuple(resolved_ranges)  # type: ignore[arg-type]
        ),
    )


def _atlas_reference() -> tuple[dict[str, int], dict[str, float]]:
    """Return the atlas's benchmark rank and measured alignment per encoding.

    Degrades to empty mappings if the rule base and the bundled atlas ever
    drift apart, so screening never fails on a lookup.
    """
    from encoding_atlas.atlas import get_encoding_profile

    ranks: dict[str, int] = {}
    alignments: dict[str, float] = {}
    for name in BENCHMARK_PARAMS:
        try:
            profile = get_encoding_profile(name)
        except KeyError:  # pragma: no cover - rule base and atlas agree today
            continue
        ranks[name] = profile.rank
        measured = profile.metric("kernel_target_alignment")
        if measured is not None:
            alignments[name] = float(measured)
    return ranks, alignments
