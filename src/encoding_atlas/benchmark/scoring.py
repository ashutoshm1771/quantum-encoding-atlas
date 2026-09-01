"""Weighted scoring of encodings, with an explicit reason for every absence.

Ranking encodings means combining metrics that are not all available for every
encoding, and *why* a metric is missing decides what the correct substitute is.
There are two genuinely different kinds of absence, and conflating them is how
a headline result gets manufactured:

``structurally_zero``
    The quantity is defined, known, and equal to zero — so nothing was
    measured. ``AngleEncoding`` applies only single-qubit rotations, so its
    entanglement capability is exactly 0; the pipeline never ran it through the
    entanglement stage because there was nothing to run.
``not_measured``
    The quantity is undefined or was never computed, and no number is
    defensible. ``BasisEncoding`` prepares computational basis states, so its
    fidelity distribution is degenerate and expressibility was never attempted.

Imputing a column statistic covers both cases and is wrong for both. On the
atlas's own data, filling the missing entanglement capability with the median
of the encodings that *do* entangle (0.605, versus a true value of 0) is what
produced its only robustness claim:

===================================  ==============  ==============
Monte Carlo weight sweep                  published    this module
===================================  ==============  ==============
``angle`` mean rank                            1.71            3.26
``angle`` % of draws in the top 3              97.8            80.8
methods above the 90% threshold        ``['angle']``          ``[]``
===================================  ==============  ==============

(Re-running the published code reproduces 97.2 rather than 97.8 for the reason
given under *Reproducibility* below; both clear the 90% bar, so the correction
does not depend on which figure is taken as the baseline.)

So nothing is imputed here. A :class:`MetricReading` carries its availability
alongside its value; ``structurally_zero`` resolves to ``0.0`` and takes part
in normalisation like any other number, while ``not_measured`` yields no value
at all and is *reported* rather than silently filled or silently dropped.

Scoring an encoding on fewer metrics than its rivals
----------------------------------------------------
When a metric is genuinely unmeasurable, the encoding is scored on the weight
that remains, renormalised. That is defensible only if it is visible, so
:attr:`ScoringResult.effective_weight` records the fraction of the intended
weight each encoding was actually judged on, and
:attr:`ScoringResult.unusable` names the metrics it lacked. Pass
``on_unusable="exclude"`` to drop such encodings instead of ranking them
against a different objective.

Reproducibility
---------------
:func:`weight_sensitivity` draws random weight vectors with
:class:`numpy.random.RandomState` rather than :func:`numpy.random.default_rng`.
NumPy guarantees the legacy generator's stream across releases and explicitly
does not guarantee it for ``Generator``; a published statistic that shifts when
NumPy is upgraded is not reproducible. The atlas's sensitivity numbers were
observed to move (97.8% to 97.2%) from that cause alone, with byte-identical
inputs and the same seed.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Availability",
    "MetricReading",
    "MetricSpec",
    "ScoringResult",
    "SensitivityResult",
    "describe_unusable",
    "measured",
    "not_measured",
    "readings_from_values",
    "score_encodings",
    "structurally_zero",
    "weight_sensitivity",
]

#: Why a metric reading holds the value it does.
Availability = Literal["measured", "structurally_zero", "not_measured"]

#: What to do with an encoding that lacks a usable value for some metric.
UnusablePolicy = Literal["report", "exclude"]

_RESOLVES_TO_ZERO: frozenset[str] = frozenset({"structurally_zero"})

#: Decimal places the ranking order is decided on. Weighted sums accumulate in
#: different orders for different encodings, so mathematically identical scores
#: can differ by an ULP; without this, rounding noise would decide rank 1 from
#: rank 3 among genuinely tied encodings. Scores themselves are not rounded.
_RANK_PRECISION = 12


@dataclass(frozen=True)
class MetricReading:
    """One metric for one encoding, carrying why it is absent if it is.

    Prefer the constructors :func:`measured`, :func:`structurally_zero` and
    :func:`not_measured` over building this directly.

    Attributes
    ----------
    value : float or None
        The measurement. Must be finite when ``availability`` is
        ``"measured"``, and ``None`` otherwise.
    availability : {"measured", "structurally_zero", "not_measured"}
        Why the reading holds that value. This is the field that decides how
        the metric enters scoring, so it is required rather than inferred.
    reason : str
        Optional human-readable justification, surfaced in reports. For a
        structural zero this should say *why* it is zero.
    """

    value: float | None
    availability: Availability = "measured"
    reason: str = ""

    def __post_init__(self) -> None:
        if self.availability == "measured":
            if self.value is None or not math.isfinite(float(self.value)):
                raise ValueError(
                    f"a 'measured' reading needs a finite value, got {self.value!r}"
                )
        elif self.availability in ("structurally_zero", "not_measured"):
            if self.value is not None:
                raise ValueError(
                    f"a {self.availability!r} reading must not carry a value, "
                    f"got {self.value!r}"
                )
        else:
            raise ValueError(
                f"availability must be 'measured', 'structurally_zero' or "
                f"'not_measured', got {self.availability!r}"
            )

    @property
    def resolved(self) -> float | None:
        """The number this reading contributes, or ``None`` if it cannot.

        A structural zero resolves to ``0.0`` and participates in
        normalisation; ``not_measured`` resolves to ``None`` and is reported.
        """
        if self.availability == "measured":
            return float(self.value) if self.value is not None else None
        if self.availability in _RESOLVES_TO_ZERO:
            return 0.0
        return None

    @property
    def is_usable(self) -> bool:
        """Whether this reading contributes a number to the score."""
        return self.resolved is not None


def measured(value: float) -> MetricReading:
    """A metric that was measured. ``value`` must be finite."""
    return MetricReading(value=float(value), availability="measured")


def structurally_zero(reason: str = "") -> MetricReading:
    """A metric that is *known* to be zero, so it was never measured.

    Use for quantities that are defined and vanish by construction — the
    entanglement capability of a circuit with no entangling gates, say. It
    resolves to ``0.0`` and normalises alongside measured values.
    """
    return MetricReading(value=None, availability="structurally_zero", reason=reason)


def not_measured(reason: str = "") -> MetricReading:
    """A metric with no defensible value: undefined, or never computed.

    It contributes nothing, and the encoding is reported as having been scored
    on less than the full weight rather than being quietly filled in.
    """
    return MetricReading(value=None, availability="not_measured", reason=reason)


@dataclass(frozen=True)
class MetricSpec:
    """One axis of a weighted score.

    Attributes
    ----------
    name : str
        Key into each encoding's reading mapping.
    weight : float
        Relative importance. Must be positive; weights need not sum to 1, as
        the score divides by the weight actually used.
    higher_is_better : bool
        Whether a larger value is a better one. ``False`` inverts the
        normalised value, so every axis contributes "more is better".
    """

    name: str
    weight: float
    higher_is_better: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError(
                f"weight for {self.name!r} must be finite and positive, "
                f"got {self.weight!r}"
            )


_READINGS = Mapping[str, Mapping[str, MetricReading]]


def describe_unusable(unusable: Mapping[str, Sequence[str]]) -> str:
    """Render ``{encoding: unusable metrics}`` as a readable clause."""
    if not unusable:
        return "none"
    return "; ".join(
        f"{name} lacks {list(metrics)}" for name, metrics in sorted(unusable.items())
    )


def _resolve(
    readings: _READINGS,
    specs: Sequence[MetricSpec],
) -> tuple[
    dict[str, dict[str, float]], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]
]:
    """Turn readings into usable numbers plus the two absence registers."""
    values: dict[str, dict[str, float]] = {}
    unusable: dict[str, tuple[str, ...]] = {}
    zeros: dict[str, tuple[str, ...]] = {}

    for name in sorted(readings):
        row = readings[name]
        usable: dict[str, float] = {}
        missing: list[str] = []
        structural: list[str] = []
        for spec in specs:
            reading = row.get(spec.name)
            if reading is None:
                missing.append(spec.name)
                continue
            resolved = reading.resolved
            if resolved is None:
                missing.append(spec.name)
                continue
            usable[spec.name] = resolved
            if reading.availability in _RESOLVES_TO_ZERO:
                structural.append(spec.name)
        values[name] = usable
        if missing:
            unusable[name] = tuple(missing)
        if structural:
            zeros[name] = tuple(structural)
    return values, unusable, zeros


def _normalisation(
    values: Mapping[str, Mapping[str, float]],
    specs: Sequence[MetricSpec],
) -> dict[str, tuple[float, float]]:
    """Min and max of each metric over the encodings that have it."""
    ranges: dict[str, tuple[float, float]] = {}
    for spec in specs:
        column = [row[spec.name] for row in values.values() if spec.name in row]
        ranges[spec.name] = (
            (float(min(column)), float(max(column))) if column else (0.0, 1.0)
        )
    return ranges


def _normalise(value: float, low: float, high: float, higher_is_better: bool) -> float:
    """Min-max a value into ``[0, 1]``, oriented so larger is always better."""
    if high - low <= 1e-12:
        return 0.5
    scaled = (value - low) / (high - low)
    return scaled if higher_is_better else 1.0 - scaled


@dataclass(frozen=True)
class ScoringResult:
    """A weighted ranking together with what each encoding was judged on.

    Attributes
    ----------
    scores : dict
        ``{encoding: score}`` in ``[0, 1]``, best first.
    ranks : dict
        ``{encoding: rank}``, 1 = best. Scores equal to 12 decimal places count
        as tied and are then ordered by name, so floating-point accumulation
        noise cannot decide the ranking.
    effective_weight : dict
        ``{encoding: fraction of the intended weight it was scored on}``. A
        value below 1 means this encoding was ranked against a *different*
        objective than its rivals; that is why the figure is reported rather
        than absorbed.
    unusable : dict
        ``{encoding: metrics with no defensible value}``. Never imputed.
    structural_zeros : dict
        ``{encoding: metrics resolved to 0.0 because they vanish by
        construction}``. These do take part in normalisation.
    normalization : dict
        ``{metric: (low, high)}`` used for min-max scaling.
    specs : tuple of MetricSpec
        The axes and weights the score was computed from.
    excluded : dict
        ``{encoding: metrics it lacked}`` for encodings dropped entirely under
        ``on_unusable="exclude"``. Empty under the default policy.
    """

    scores: dict[str, float]
    ranks: dict[str, int]
    effective_weight: dict[str, float]
    unusable: dict[str, tuple[str, ...]]
    structural_zeros: dict[str, tuple[str, ...]]
    normalization: dict[str, tuple[float, float]]
    specs: tuple[MetricSpec, ...]
    excluded: dict[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def best(self) -> str:
        """Encoding with the highest score."""
        return next(iter(self.scores))

    @property
    def fully_scored(self) -> tuple[str, ...]:
        """Encodings judged on the complete objective, in rank order."""
        return tuple(
            name for name in self.scores if self.effective_weight[name] >= 1.0 - 1e-12
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "scores": dict(self.scores),
            "ranks": dict(self.ranks),
            "effective_weight": dict(self.effective_weight),
            "unusable": {k: list(v) for k, v in self.unusable.items()},
            "structural_zeros": {k: list(v) for k, v in self.structural_zeros.items()},
            "normalization": {k: list(v) for k, v in self.normalization.items()},
            "excluded": {k: list(v) for k, v in self.excluded.items()},
            "specs": [
                {
                    "name": s.name,
                    "weight": s.weight,
                    "higher_is_better": s.higher_is_better,
                }
                for s in self.specs
            ],
            "best": self.best if self.scores else None,
        }

    def summary(self) -> str:
        """A short report naming every encoding not judged on the full set."""
        lines = [f"{len(self.scores)} encodings scored on {len(self.specs)} metrics"]
        # score_encodings never yields an empty result; this guards a
        # ScoringResult built directly.
        if self.scores:  # pragma: no branch
            lines.append(f"  best: {self.best} ({self.scores[self.best]:.3f})")
        partial = [
            (n, w) for n, w in sorted(self.effective_weight.items()) if w < 1.0 - 1e-12
        ]
        if partial:
            detail = ", ".join(f"{n} ({w:.0%} of the weight)" for n, w in partial)
            lines.append(f"  scored on a reduced objective: {detail}")
        if self.structural_zeros:
            # Not "lacks": these metrics are present and equal to zero.
            detail = "; ".join(
                f"{name}: {list(metrics)}"
                for name, metrics in sorted(self.structural_zeros.items())
            )
            lines.append(f"  known to be zero, not missing: {detail}")
        if self.excluded:
            lines.append(f"  excluded: {describe_unusable(self.excluded)}")
        return "\n".join(lines)


def score_encodings(
    readings: _READINGS,
    specs: Sequence[MetricSpec],
    *,
    on_unusable: UnusablePolicy = "report",
) -> ScoringResult:
    """Rank encodings by a weighted, min-max normalised score.

    Parameters
    ----------
    readings : mapping
        ``{encoding: {metric: MetricReading}}``. A metric absent from an
        encoding's mapping is treated as :func:`not_measured`.
    specs : sequence of MetricSpec
        The axes, their weights, and their orientation. Must be non-empty with
        unique names.
    on_unusable : {"report", "exclude"}, default="report"
        ``"report"`` scores the encoding on the weight that remains and records
        the shortfall in :attr:`ScoringResult.effective_weight`. ``"exclude"``
        drops it and lists it in :attr:`ScoringResult.excluded`. Nothing is
        ever imputed under either policy.

    Returns
    -------
    ScoringResult

    Raises
    ------
    ValueError
        If ``specs`` is empty or has duplicate names, ``readings`` is empty,
        ``on_unusable`` is unknown, or no encoding retains a usable metric.

    Examples
    --------
    A structural zero counts as zero; an unmeasured metric does not count:

    >>> specs = [MetricSpec("accuracy", 1.0), MetricSpec("entanglement", 1.0)]
    >>> readings = {
    ...     "plain":  {"accuracy": measured(0.9),
    ...                "entanglement": structurally_zero("no entangling gates")},
    ...     "fancy":  {"accuracy": measured(0.8), "entanglement": measured(0.6)},
    ... }
    >>> result = score_encodings(readings, specs)
    >>> result.effective_weight["plain"]
    1.0
    >>> result.structural_zeros["plain"]
    ('entanglement',)
    """
    if not specs:
        raise ValueError("specs must not be empty")
    names = [spec.name for spec in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate metric names in specs: {names}")
    if not readings:
        raise ValueError("readings must not be empty")
    if on_unusable not in ("report", "exclude"):
        raise ValueError(
            f"on_unusable must be 'report' or 'exclude', got {on_unusable!r}"
        )

    values, unusable, zeros = _resolve(readings, specs)

    excluded: dict[str, tuple[str, ...]] = {}
    if on_unusable == "exclude":
        excluded = dict(unusable)
        values = {n: row for n, row in values.items() if n not in excluded}
        unusable = {}
        zeros = {n: z for n, z in zeros.items() if n not in excluded}

    values = {n: row for n, row in values.items() if row}
    if not values:
        raise ValueError(
            "no encoding has a usable value for any metric "
            f"({describe_unusable(excluded or unusable)})"
        )

    ranges = _normalisation(values, specs)
    total = sum(spec.weight for spec in specs)

    scores: dict[str, float] = {}
    effective: dict[str, float] = {}
    for name, row in values.items():
        weighted = 0.0
        used = 0.0
        for spec in specs:
            if spec.name not in row:
                continue
            low, high = ranges[spec.name]
            weighted += spec.weight * _normalise(
                row[spec.name], low, high, spec.higher_is_better
            )
            used += spec.weight
        scores[name] = weighted / used if used > 0 else 0.0
        effective[name] = used / total

    ordered = sorted(scores, key=lambda n: (-round(scores[n], _RANK_PRECISION), n))
    return ScoringResult(
        scores={n: scores[n] for n in ordered},
        ranks={n: i + 1 for i, n in enumerate(ordered)},
        effective_weight={n: effective[n] for n in ordered},
        unusable={n: unusable[n] for n in ordered if n in unusable},
        structural_zeros={n: zeros[n] for n in ordered if n in zeros},
        normalization=ranges,
        specs=tuple(specs),
        excluded=excluded,
    )


@dataclass(frozen=True)
class SensitivityResult:
    """How a ranking holds up when the weights are drawn at random.

    Attributes
    ----------
    mean_rank, std_rank : dict
        ``{encoding: statistic}`` over the sampled weight vectors.
    best_rank, worst_rank : dict
        ``{encoding: rank}`` extremes observed.
    top_k_fraction : dict
        ``{encoding: fraction of draws placing it in the top ``k``}``, in
        ``[0, 1]``.
    robust : tuple of str
        Encodings whose ``top_k_fraction`` reached ``threshold``, best mean
        rank first. Empty is a meaningful result: it says no encoding survives
        reweighting.
    k, threshold : int, float
        The criterion ``robust`` was evaluated against.
    n_samples : int
        Number of weight vectors drawn.
    seed : int
        Seed for the legacy generator, whose stream NumPy holds stable across
        releases.
    equal_weight_ranking : tuple of str
        The deterministic ranking under equal weights, for reference.
    effective_weight : dict
        Carried through from scoring; an encoding below 1 was compared on a
        reduced objective in every draw.
    unusable : dict
        ``{encoding: metrics with no defensible value}``.
    structural_zeros : dict
        ``{encoding: metrics resolved to 0.0 by construction}``.
    """

    mean_rank: dict[str, float]
    std_rank: dict[str, float]
    best_rank: dict[str, int]
    worst_rank: dict[str, int]
    top_k_fraction: dict[str, float]
    robust: tuple[str, ...]
    k: int
    threshold: float
    n_samples: int
    seed: int
    equal_weight_ranking: tuple[str, ...]
    effective_weight: dict[str, float]
    unusable: dict[str, tuple[str, ...]]
    structural_zeros: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view."""
        return {
            "mean_rank": dict(self.mean_rank),
            "std_rank": dict(self.std_rank),
            "best_rank": dict(self.best_rank),
            "worst_rank": dict(self.worst_rank),
            "top_k_fraction": dict(self.top_k_fraction),
            "robust": list(self.robust),
            "k": self.k,
            "threshold": self.threshold,
            "n_samples": self.n_samples,
            "seed": self.seed,
            "equal_weight_ranking": list(self.equal_weight_ranking),
            "effective_weight": dict(self.effective_weight),
            "unusable": {k: list(v) for k, v in self.unusable.items()},
            "structural_zeros": {k: list(v) for k, v in self.structural_zeros.items()},
        }

    def summary(self) -> str:
        """A short report, including the case where nothing is robust."""
        lines = [
            f"{len(self.mean_rank)} encodings over {self.n_samples} random "
            f"weight vectors (seed {self.seed})"
        ]
        for name in sorted(self.mean_rank, key=lambda n: self.mean_rank[n])[:3]:
            lines.append(
                f"  {name}: mean rank {self.mean_rank[name]:.2f}, "
                f"top-{self.k} in {self.top_k_fraction[name]:.1%} of draws"
            )
        if self.robust:
            lines.append(
                f"  robustly top-{self.k} (>= {self.threshold:.0%}): "
                f"{list(self.robust)}"
            )
        else:
            lines.append(
                f"  no encoding reaches the top {self.k} in {self.threshold:.0%} "
                f"of draws: the ranking does not survive reweighting"
            )
        partial = [n for n, w in self.effective_weight.items() if w < 1.0 - 1e-12]
        if partial:
            lines.append(f"  scored on a reduced objective: {partial}")
        return "\n".join(lines)


def _rank_descending(scores: NDArray[np.float64]) -> NDArray[np.int_]:
    """Ranks with 1 = highest score; ties resolved by position, as argsort does."""
    order: NDArray[np.int_] = np.argsort(np.argsort(-scores)) + 1
    return order


def weight_sensitivity(
    readings: _READINGS,
    specs: Sequence[MetricSpec],
    *,
    n_samples: int = 1000,
    seed: int = 42,
    k: int = 3,
    threshold: float = 0.90,
    on_unusable: UnusablePolicy = "report",
) -> SensitivityResult:
    """Resample the weights at random and see which rankings survive.

    Weight vectors are drawn from a flat Dirichlet over the metrics, so every
    convex combination is equally likely. An encoding that stays near the top
    across the sweep is preferred for reasons stronger than one particular
    choice of weights.

    Parameters
    ----------
    readings : mapping
        ``{encoding: {metric: MetricReading}}``, as for :func:`score_encodings`.
        Absences are resolved by the same rules; nothing is imputed.
    specs : sequence of MetricSpec
        Axes and orientations. The declared ``weight`` values are ignored here —
        the point is to replace them — but the names and orientations are used.
    n_samples : int, default=1000
        Number of weight vectors. Must be positive.
    seed : int, default=42
        Seed for :class:`numpy.random.RandomState`, chosen over
        :func:`numpy.random.default_rng` because NumPy guarantees the legacy
        stream across releases. Results are therefore reproducible on any
        supported NumPy.
    k : int, default=3
        Size of the "top" band that ``top_k_fraction`` counts.
    threshold : float, default=0.90
        Fraction of draws an encoding must reach the top ``k`` in to count as
        robust. In ``(0, 1]``.
    on_unusable : {"report", "exclude"}, default="report"
        As for :func:`score_encodings`.

    Returns
    -------
    SensitivityResult

    Raises
    ------
    ValueError
        If ``n_samples`` or ``k`` is not positive, ``threshold`` is outside
        ``(0, 1]``, or the inputs are unusable.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold!r}")

    # Reuse the scoring path so absences resolve identically in both analyses.
    baseline = score_encodings(readings, specs, on_unusable=on_unusable)
    values, _, _ = _resolve(readings, specs)
    values = {n: values[n] for n in baseline.scores}

    names = list(baseline.scores)
    ranges = baseline.normalization

    # Normalised design matrix, with NaN marking "this encoding lacks this
    # metric" so the weights can be renormalised per encoding per draw.
    matrix = np.full((len(names), len(specs)), np.nan, dtype=np.float64)
    for i, name in enumerate(names):
        for j, spec in enumerate(specs):
            if spec.name in values[name]:
                low, high = ranges[spec.name]
                matrix[i, j] = _normalise(
                    values[name][spec.name], low, high, spec.higher_is_better
                )
    present = ~np.isnan(matrix)
    filled = np.where(present, matrix, 0.0)

    def _score(weights: NDArray[np.float64]) -> NDArray[np.float64]:
        used = present @ weights
        raw = filled @ weights
        scores: NDArray[np.float64] = np.divide(
            raw, used, out=np.zeros_like(raw), where=used > 0
        )
        return scores

    random_state = np.random.RandomState(seed)
    ranks = np.zeros((n_samples, len(names)), dtype=np.int_)
    for sample in range(n_samples):
        weights = random_state.dirichlet(np.ones(len(specs)))
        ranks[sample] = _rank_descending(_score(weights))

    equal = _rank_descending(_score(np.ones(len(specs)) / len(specs)))
    equal_order = tuple(names[i] for i in np.argsort(equal))

    mean_rank = {n: float(ranks[:, i].mean()) for i, n in enumerate(names)}
    top_k = {n: float((ranks[:, i] <= k).mean()) for i, n in enumerate(names)}
    robust = tuple(
        sorted(
            (n for n, f in top_k.items() if f >= threshold), key=lambda n: mean_rank[n]
        )
    )

    return SensitivityResult(
        mean_rank=mean_rank,
        std_rank={n: float(ranks[:, i].std()) for i, n in enumerate(names)},
        best_rank={n: int(ranks[:, i].min()) for i, n in enumerate(names)},
        worst_rank={n: int(ranks[:, i].max()) for i, n in enumerate(names)},
        top_k_fraction=top_k,
        robust=robust,
        k=k,
        threshold=threshold,
        n_samples=n_samples,
        seed=seed,
        equal_weight_ranking=equal_order,
        effective_weight=dict(baseline.effective_weight),
        unusable=dict(baseline.unusable),
        structural_zeros=dict(baseline.structural_zeros),
    )


def readings_from_values(
    values: Mapping[str, Mapping[str, float | None]],
    *,
    structural_zeros: Mapping[str, Iterable[str]] | None = None,
    reason: str = "",
) -> dict[str, dict[str, MetricReading]]:
    """Build readings from a plain value table plus a structural-zero register.

    A convenience for callers holding ``{encoding: {metric: value_or_None}}``.
    Every ``None`` becomes :func:`not_measured` unless the encoding/metric pair
    is listed in ``structural_zeros``, in which case it becomes
    :func:`structurally_zero`.

    Parameters
    ----------
    values : mapping
        ``{encoding: {metric: value or None}}``.
    structural_zeros : mapping, optional
        ``{encoding: iterable of metric names that vanish by construction}``.
    reason : str, optional
        Justification attached to every structural zero created here.

    Returns
    -------
    dict
        ``{encoding: {metric: MetricReading}}``.

    Examples
    --------
    >>> readings = readings_from_values(
    ...     {"plain": {"ent": None}, "fancy": {"ent": 0.6}},
    ...     structural_zeros={"plain": ["ent"]},
    ...     reason="no entangling gates",
    ... )
    >>> plain = readings.get("plain", {}).get("ent")
    >>> plain.availability
    'structurally_zero'
    >>> fancy = readings.get("fancy", {}).get("ent")
    >>> fancy.availability
    'measured'
    """
    zeros = {k: set(v) for k, v in (structural_zeros or {}).items()}
    out: dict[str, dict[str, MetricReading]] = {}
    for name, row in values.items():
        built: dict[str, MetricReading] = {}
        for metric, value in row.items():
            if value is not None:
                built[metric] = measured(value)
            elif metric in zeros.get(name, ()):
                built[metric] = structurally_zero(reason)
            else:
                built[metric] = not_measured()
        out[name] = built
    return out
