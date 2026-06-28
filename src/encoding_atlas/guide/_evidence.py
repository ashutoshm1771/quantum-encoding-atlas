"""Empirical priority scores derived from the bundled benchmark atlas.

The recommender's central design goal is to be *evidence-based*: a user's stated
priority should be scored by what the benchmark actually measured, not by
hand-tuned tags. This module maps each priority onto a measured axis of
:mod:`encoding_atlas.atlas` and normalises it to ``[0, 1]`` across all
encodings, so that the best-measured encoding on an axis scores ``1.0`` and the
worst scores ``0.0``.

Priority → measured axis:

==================  ===========================================================
Priority            Empirical axis (higher score = better match)
==================  ===========================================================
``accuracy``        best of measured VQC / quantum-kernel test accuracy
``trainability``    measured gradient-variance trainability estimate
``speed``           circuit depth, inverted (shallower = faster)
``noise_resilience``measured retained fidelity under depolarising noise
==================  ===========================================================

These scores are what allow the guide to stop recommending high-expressibility
but low-accuracy encodings (e.g. IQP, ZZ) for accuracy-driven queries — a direct
consequence of the refuted expressibility–accuracy hypothesis (H1).
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from encoding_atlas.atlas import list_profiles

# Score assigned when a metric is undefined for an encoding (e.g. noise
# resilience for the basis encoding). A neutral midpoint avoids both crediting
# and penalising an encoding on an axis the benchmark did not measure for it.
NEUTRAL_SCORE = 0.5

# priority -> (metric key, higher-is-better). The synthetic ``_best_accuracy``
# key is computed as max(vqc_accuracy, kernel_accuracy).
_PRIORITY_AXIS: dict[str, tuple[str, bool]] = {
    "accuracy": ("_best_accuracy", True),
    "trainability": ("trainability_estimate", True),
    "speed": ("depth", False),
    "noise_resilience": ("noise_resilience", True),
}


def _raw_value(metrics: Mapping[str, Any], metric: str) -> float | None:
    """Extract the raw axis value for one encoding (``None`` if undefined)."""
    if metric == "_best_accuracy":
        candidates = [
            metrics.get("vqc_accuracy"),
            metrics.get("kernel_accuracy"),
        ]
        defined = [v for v in candidates if v is not None]
        return max(defined) if defined else None
    return metrics.get(metric)


@lru_cache(maxsize=1)
def empirical_priority_scores() -> dict[str, dict[str, float]]:
    """Return ``{priority: {encoding_name: score in [0, 1]}}`` from the atlas.

    Scores are min-max normalised per axis across all benchmarked encodings.
    For ``speed`` (lower depth is better) the normalised value is inverted.
    Undefined values map to :data:`NEUTRAL_SCORE`.
    """
    profiles = list_profiles()
    table: dict[str, dict[str, float]] = {}

    for priority, (metric, higher_is_better) in _PRIORITY_AXIS.items():
        raw: dict[str, float | None] = {
            profile.name: _raw_value(profile.metrics, metric) for profile in profiles
        }
        defined = [v for v in raw.values() if v is not None]
        scores: dict[str, float] = {}

        if not defined or max(defined) == min(defined):
            # Degenerate axis: nothing to discriminate on.
            scores = dict.fromkeys(raw, NEUTRAL_SCORE)
        else:
            low, high = min(defined), max(defined)
            span = high - low
            for name, value in raw.items():
                if value is None:
                    scores[name] = NEUTRAL_SCORE
                    continue
                normalised = (value - low) / span
                scores[name] = normalised if higher_is_better else 1.0 - normalised

        table[priority] = scores

    return table


def evidence_score(name: str, priority: str) -> float:
    """Return the empirical ``[0, 1]`` score for ``name`` on ``priority``.

    Falls back to :data:`NEUTRAL_SCORE` for unknown priorities or encodings, so
    the recommender degrades gracefully if the atlas and rule base ever drift.
    """
    return empirical_priority_scores().get(priority, {}).get(name, NEUTRAL_SCORE)
