"""Encoding recommendation system.

Provides a two-phase recommendation pipeline:

1. **Hard filter** — structurally impossible encodings are eliminated
   (wrong data type, wrong feature count, missing symmetry, etc.).
2. **Soft scoring** — remaining candidates are scored based on how well
   they match the user's priority, problem structure, hardware target,
   and other preferences.  The highest-scoring encoding becomes the
   primary recommendation; the next three become alternatives.

All new parameters introduced after ``priority`` are keyword-only with
backward-compatible defaults so that existing callers are not broken.

Scoring categories (in descending order of weight):

1. **Hard-precondition bonuses** — data type, symmetry, feature count,
   trainable match.
2. **Binary / discrete penalty** — non-binary encodings on binary data.
3. **Priority matching** — user-stated optimisation goal.
4. **Problem structure** — domain-specific encodings.
5. **Feature interactions** — custom interaction structures.
6. **Task matching** — classification vs regression suitability.
7. **Hardware suitability** — NISQ bonus, deep-circuit penalty,
   ``avoid_when`` penalty.
8. **Feature count** — qubit efficiency and default routing.
9. **Sample count** — simulability for tiny datasets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from encoding_atlas.guide.rules import (
    ENCODING_RULES,
    VALID_DATA_TYPES,
    VALID_FEATURE_INTERACTIONS,
    VALID_PRIORITIES,
    VALID_PROBLEM_STRUCTURES,
    VALID_SYMMETRIES,
    VALID_TASKS,
    EncodingRule,
    _passes_hard_constraints,
)

# ---------------------------------------------------------------------------
# Public data structure
# ---------------------------------------------------------------------------


@dataclass
class Recommendation:
    """Encoding recommendation result.

    Attributes
    ----------
    encoding_name : str
        Canonical name of the recommended encoding (matches registry keys).
    explanation : str
        Human-readable rationale for the recommendation.
    alternatives : list[str]
        Up to three runner-up encoding names, ranked by score.
    confidence : float
        Confidence in the recommendation, in ``[0, 1]``.
    """

    encoding_name: str
    explanation: str
    alternatives: list[str]
    confidence: float


# ---------------------------------------------------------------------------
# Explanation templates
# ---------------------------------------------------------------------------

_EXPLANATION_TEMPLATES: dict[str, str] = {
    "angle": (
        "Angle encoding provides O(1) depth with simple rotations, "
        "ideal for {reason}"
    ),
    "basis": (
        "Basis encoding directly maps binary/discrete features to "
        "computational basis states"
    ),
    "higher_order_angle": (
        "Higher-order angle encoding captures polynomial feature "
        "interactions (order-k products) without entanglement"
    ),
    "iqp": (
        "IQP encoding creates highly entangled states with provable "
        "classical simulation hardness, well-suited for kernel methods"
    ),
    "zz_feature_map": (
        "ZZ Feature Map provides standard pairwise feature interactions "
        "via (pi-x_i)(pi-x_j) phase encoding for kernel methods"
    ),
    "pauli_feature_map": (
        "Pauli Feature Map enables configurable Pauli-string rotation "
        "structures for custom feature interactions"
    ),
    "data_reuploading": (
        "Data re-uploading achieves universal approximation capability "
        "through repeated data encoding with entanglement layers"
    ),
    "hardware_efficient": (
        "Hardware-efficient encoding minimises gate decomposition "
        "overhead on real quantum devices"
    ),
    "amplitude": (
        "Amplitude encoding provides exponential compression "
        "({n_qubits} qubits for {n_features} features)"
    ),
    "qaoa": (
        "QAOA-inspired encoding uses cost-mixer layer structure "
        "suited for combinatorial and graph-structured problems"
    ),
    "hamiltonian": (
        "Hamiltonian encoding applies Trotterised time evolution "
        "under a data-dependent Hamiltonian for physics-inspired ML"
    ),
    "trainable": (
        "Trainable encoding interleaves data rotations with learnable "
        "parameter layers for task-specific optimisation"
    ),
    "symmetry_inspired": (
        "Symmetry-inspired encoding provides a heuristic "
        "symmetry-aware inductive bias for the given problem"
    ),
    "so2_equivariant": (
        "SO(2) equivariant encoding guarantees mathematically rigorous "
        "2D rotational equivariance for the 2-feature input"
    ),
    "cyclic_equivariant": (
        "Cyclic equivariant encoding guarantees rigorous Z_n cyclic "
        "shift symmetry with ring-topology circuits"
    ),
    "swap_equivariant": (
        "Swap equivariant encoding guarantees rigorous S_2 pair-swap "
        "symmetry over feature pairs"
    ),
}


# ---------------------------------------------------------------------------
# Scoring weight constants
# ---------------------------------------------------------------------------
# Extracted as named constants to simplify tuning, sensitivity analysis,
# and future empirical calibration.  See the design document for rationale.

# Hard-precondition bonuses — dominate soft heuristics so that specialised
# encodings reliably outrank general-purpose ones.
_W_DATA_TYPE_BONUS: float = 0.50
_W_SYMMETRY_BONUS: float = 0.45
_W_N_FEATURES_BONUS: float = 0.10
_W_TRAINABLE_BONUS: float = 0.40

# Penalty for non-binary encodings when user declares binary/discrete data.
_W_BINARY_PENALTY: float = 0.20

# Priority matching — per matched ``best_for`` tag (capped at 2).
_W_PRIORITY_PER_TAG: float = 0.10
_PRIORITY_TAG_CAP: int = 2

# Problem structure bonus for domain-specific encodings.
_W_STRUCTURE_BONUS: float = 0.36

# Feature interaction bonus.
_W_INTERACTION_BONUS: float = 0.35

# Task matching — small bonus for task-relevant encodings.
# Applied only when no domain-specific parameter (problem_structure,
# feature_interactions) is active, to preserve the priority hierarchy.
_W_TASK_BONUS: float = 0.04

# Hardware suitability.
_W_HARDWARE_NISQ_BONUS: float = 0.10
_W_HARDWARE_DEEP_PENALTY: float = 0.15
_W_AVOID_WHEN_PENALTY: float = 0.08

# Feature count suitability.
_W_LOGARITHMIC_BONUS: float = 0.15
_W_ACCURACY_DEFAULT_BONUS: float = 0.12
_W_SMALL_FEATURE_BONUS: float = 0.03

# Sample count factor.
_W_SMALL_SAMPLE_BONUS: float = 0.03

# ``avoid_when`` tags triggered by real (non-simulator) hardware.
# IQP (full entanglement) is more noise-sensitive than ZZ/Pauli (structured
# entanglement), hence IQP uses "noisy_hardware" while ZZ/Pauli use
# "very_noisy_hardware".  Only the former set triggers automatic penalties;
# "very_noisy_hardware" is only matched via ``get_matching_encodings``.
_HARDWARE_AVOID_TAGS: frozenset[str] = frozenset({"noisy_hardware", "nisq_hardware"})


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

# Maps user-facing priority values to relevant ``best_for`` tags.
_PRIORITY_TAG_MAP: dict[str, list[str]] = {
    "speed": ["speed", "simplicity"],
    "noise_resilience": ["nisq_hardware", "native_gates", "noise_resilience"],
    "trainability": ["trainability", "task_specific", "optimization"],
    "accuracy": [
        "expressibility",
        "quantum_advantage",
        "universal_approximation",
        "kernel_methods",
    ],
}

# Maps ML task type to relevant ``best_for`` tags.
_TASK_TAG_MAP: dict[str, list[str]] = {
    "classification": ["kernel_methods"],
    "regression": ["universal_approximation"],
}

# Default encoding for each feature-count range when priority is accuracy.
# This mirrors the decision tree's feature-count-based fallback (level 7).
_ACCURACY_DEFAULT_BY_FEATURE_RANGE: dict[str, str] = {
    "small": "iqp",  # n_features <= 4
    "medium": "zz_feature_map",  # 5 <= n_features <= 8
    "large": "amplitude",  # n_features > 8
}

# Maps user-facing problem_structure values to relevant ``best_for`` tags.
_STRUCTURE_TAG_MAP: dict[str, list[str]] = {
    "combinatorial": ["combinatorial", "graph_optimization", "qaoa_structure"],
    "physics_simulation": ["physics_simulation", "time_evolution"],
    "time_series": ["periodic_data", "cyclic_symmetry", "time_series"],
}


def _compute_score(
    name: str,
    rules: EncodingRule,
    *,
    n_features: int,
    n_samples: int,
    task: str,
    hardware: str,
    priority: str,
    data_type: str,
    symmetry: str | None,
    trainable: bool,
    problem_structure: str | None,
    feature_interactions: str | None,
) -> float:
    """Compute a suitability score in ``[0, 1]`` for one candidate encoding.

    The score is a weighted sum of bonuses (and penalties) reflecting how
    well the encoding matches the user's stated requirements.  Weights are
    chosen so that *hard-precondition matches* (data type, symmetry, exact
    feature count) dominate, followed by priority/structure matching, then
    softer heuristics for hardware, task, and feature count.

    Parameters
    ----------
    name : str
        Canonical encoding name.
    rules : EncodingRule
        Rule entry for this encoding.
    n_features, n_samples, task, hardware, priority, data_type, symmetry,
    trainable, problem_structure, feature_interactions
        User-provided recommendation parameters.

    Returns
    -------
    float
        Score clamped to ``[0, 1]``.
    """
    score = 0.0

    # --- 1. Hard-precondition bonus (high-confidence signals) ---------------
    if (
        rules["requires_data_type"] is not None
        and data_type in rules["requires_data_type"]
    ):
        score += _W_DATA_TYPE_BONUS

    if (
        rules["requires_symmetry"] is not None
        and symmetry == rules["requires_symmetry"]
    ):
        score += _W_SYMMETRY_BONUS

    if (
        rules["requires_n_features"] is not None
        and n_features == rules["requires_n_features"]
    ):
        score += _W_N_FEATURES_BONUS

    if rules["requires_trainable"] and trainable:
        score += _W_TRAINABLE_BONUS

    # --- 2. Penalty for non-binary encodings on binary/discrete data --------
    if data_type in ("binary", "discrete") and rules["requires_data_type"] is None:
        score -= _W_BINARY_PENALTY

    # --- 3. Priority matching -----------------------------------------------
    priority_tags = _PRIORITY_TAG_MAP.get(priority, [])
    matched_priority = sum(1 for t in priority_tags if t in rules["best_for"])
    score += _W_PRIORITY_PER_TAG * min(matched_priority, _PRIORITY_TAG_CAP)

    # --- 4. Problem structure matching --------------------------------------
    if problem_structure is not None:
        struct_tags = _STRUCTURE_TAG_MAP.get(problem_structure, [])
        if any(t in rules["best_for"] for t in struct_tags):
            score += _W_STRUCTURE_BONUS

    # --- 5. Feature interaction matching ------------------------------------
    if feature_interactions == "polynomial":
        if "polynomial_features" in rules["best_for"]:
            score += _W_INTERACTION_BONUS
    elif feature_interactions == "custom_pauli" and "custom_pauli" in rules["best_for"]:
        score += _W_INTERACTION_BONUS

    # --- 6. Task matching ---------------------------------------------------
    # Applied only when no domain-specific parameter is active, so that
    # problem_structure and feature_interactions take precedence.
    if problem_structure is None and feature_interactions is None:
        task_tags = _TASK_TAG_MAP.get(task, [])
        if any(t in rules["best_for"] for t in task_tags):
            score += _W_TASK_BONUS

    # --- 7. Hardware suitability --------------------------------------------
    if hardware != "simulator":
        if any(
            t in rules["best_for"]
            for t in ("nisq_hardware", "native_gates", "noise_resilience")
        ):
            score += _W_HARDWARE_NISQ_BONUS
        if rules["circuit_depth"] == "deep":
            score -= _W_HARDWARE_DEEP_PENALTY
        # Penalise encodings explicitly marked to avoid noisy hardware.
        if any(t in rules["avoid_when"] for t in _HARDWARE_AVOID_TAGS):
            score -= _W_AVOID_WHEN_PENALTY

    # --- 8. Feature count suitability ---------------------------------------
    if n_features > 8 and rules["qubit_scaling"] == "logarithmic":
        score += _W_LOGARITHMIC_BONUS

    # Default-for-feature-range bonus (accuracy priority only).
    if priority == "accuracy":
        size = "small" if n_features <= 4 else "medium" if n_features <= 8 else "large"
        if name == _ACCURACY_DEFAULT_BY_FEATURE_RANGE.get(size):
            score += _W_ACCURACY_DEFAULT_BONUS

    # Small-feature bonus for encodings with an explicit max_features limit.
    if (
        n_features <= 4
        and rules["max_features"] is not None
        and rules["max_features"] >= n_features
    ):
        score += _W_SMALL_FEATURE_BONUS

    # --- 9. Sample count factor ---------------------------------------------
    if n_samples < 100 and rules["simulable"]:
        score += _W_SMALL_SAMPLE_BONUS

    return max(0.0, min(1.0, score))


def _score_to_confidence(score: float) -> float:
    """Map a raw score to a human-interpretable confidence value.

    The mapping is a continuous piecewise-linear function with three bands:

    * ``score >= 0.50`` → high confidence  (0.85 – 0.95)
    * ``0.30 <= score < 0.50`` → medium confidence  (0.65 – 0.85)
    * ``score < 0.30`` → lower confidence  (0.50 – 0.65)

    The function is continuous at both band boundaries:
    ``score = 0.30`` → ``0.65``, ``score = 0.50`` → ``0.85``.

    Parameters
    ----------
    score : float
        Raw suitability score in ``[0, 1]``.

    Returns
    -------
    float
        Confidence value in ``[0.50, 0.95]``.
    """
    if score >= 0.50:
        # High band: 0.85 at score=0.50, 0.95 at score=1.0.
        return min(0.95, 0.85 + (score - 0.50) * 0.20)
    if score >= 0.30:
        # Mid band: 0.65 at score=0.30, 0.85 at score=0.50.
        return 0.65 + (score - 0.30) * 1.00
    # Low band: 0.50 at score=0.0, 0.65 at score=0.30.
    return 0.50 + score * 0.50


def _generate_explanation(
    name: str,
    rules: EncodingRule,
    *,
    priority: str,
    n_features: int,
) -> str:
    """Produce a human-readable explanation for the recommendation.

    Parameters
    ----------
    name : str
        Canonical encoding name.
    rules : EncodingRule
        Rule entry (unused currently, reserved for future enrichment).
    priority : str
        User's stated priority (used in some templates).
    n_features : int
        Feature count (used for amplitude encoding qubit calculation).

    Returns
    -------
    str
        Explanation string.
    """
    template = _EXPLANATION_TEMPLATES.get(
        name, f"{name} encoding matches your requirements"
    )
    n_qubits = (
        max(1, math.ceil(math.log2(max(n_features, 1)))) if name == "amplitude" else 0
    )
    try:
        return template.format(
            reason=priority,
            n_features=n_features,
            n_qubits=n_qubits,
        )
    except (KeyError, ValueError, IndexError):
        return template


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_recommend_inputs(
    *,
    n_features: int,
    n_samples: int,
    task: str,
    priority: str,
    data_type: str,
    symmetry: str | None,
    problem_structure: str | None,
    feature_interactions: str | None,
) -> None:
    """Validate ``recommend_encoding`` inputs, raising ``ValueError`` on bad values."""
    if not isinstance(n_features, int) or n_features < 1:
        raise ValueError(f"n_features must be a positive integer, got {n_features!r}")
    if not isinstance(n_samples, int) or n_samples < 1:
        raise ValueError(f"n_samples must be a positive integer, got {n_samples!r}")
    if task not in VALID_TASKS:
        raise ValueError(f"task must be one of {sorted(VALID_TASKS)}, got {task!r}")
    if priority not in VALID_PRIORITIES:
        raise ValueError(
            f"priority must be one of {sorted(VALID_PRIORITIES)}, got {priority!r}"
        )
    if data_type not in VALID_DATA_TYPES:
        raise ValueError(
            f"data_type must be one of {sorted(VALID_DATA_TYPES)}, got {data_type!r}"
        )
    if symmetry is not None and symmetry not in VALID_SYMMETRIES:
        raise ValueError(
            f"symmetry must be one of {sorted(VALID_SYMMETRIES)} or None, "
            f"got {symmetry!r}"
        )
    if (
        problem_structure is not None
        and problem_structure not in VALID_PROBLEM_STRUCTURES
    ):
        raise ValueError(
            f"problem_structure must be one of "
            f"{sorted(VALID_PROBLEM_STRUCTURES)} or None, "
            f"got {problem_structure!r}"
        )
    if (
        feature_interactions is not None
        and feature_interactions not in VALID_FEATURE_INTERACTIONS
    ):
        raise ValueError(
            f"feature_interactions must be one of "
            f"{sorted(VALID_FEATURE_INTERACTIONS)} or None, "
            f"got {feature_interactions!r}"
        )


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def recommend_encoding(
    n_features: int,
    n_samples: int = 500,
    task: Literal["classification", "regression"] = "classification",
    hardware: str = "simulator",
    priority: Literal[
        "accuracy", "trainability", "speed", "noise_resilience"
    ] = "accuracy",
    *,
    data_type: Literal["continuous", "binary", "discrete"] = "continuous",
    symmetry: (
        Literal["rotation", "cyclic", "permutation_pairs", "general"] | None
    ) = None,
    trainable: bool = False,
    problem_structure: (
        Literal["combinatorial", "physics_simulation", "time_series"] | None
    ) = None,
    feature_interactions: Literal["polynomial", "custom_pauli"] | None = None,
) -> Recommendation:
    """Recommend an encoding based on problem characteristics.

    The recommendation is produced in two phases:

    1. **Hard filter** — encodings whose structural preconditions are not
       satisfied (data type, feature count, symmetry, trainability) are
       eliminated.
    2. **Soft scoring** — remaining candidates are scored on priority match,
       problem structure, task type, hardware suitability, and feature count.
       The highest-scoring encoding becomes the primary recommendation.

    Parameters
    ----------
    n_features : int
        Number of input features (must be >= 1).
    n_samples : int
        Number of training samples (default 500, must be >= 1).
    task : {"classification", "regression"}
        Machine learning task type.  Classification boosts kernel-method
        encodings; regression boosts universal-approximation encodings.
    hardware : str
        Target hardware (``"simulator"``, ``"ibm"``, ``"ionq"``, etc.).
    priority : {"accuracy", "trainability", "speed", "noise_resilience"}
        Optimisation priority.
    data_type : {"continuous", "binary", "discrete"}
        Nature of the input features.
    symmetry : {"rotation", "cyclic", "permutation_pairs", "general"} | None
        Known symmetry in the data.  ``None`` means no known symmetry.
    trainable : bool
        Whether the encoding should have learnable parameters.
    problem_structure : {"combinatorial", "physics_simulation", "time_series"} | None
        Domain structure of the problem.
    feature_interactions : {"polynomial", "custom_pauli"} | None
        Desired feature interaction type.

    Returns
    -------
    Recommendation
        The recommended encoding with explanation, alternatives, and
        confidence score.

    Raises
    ------
    ValueError
        If any parameter value is outside its valid set.
    """
    _validate_recommend_inputs(
        n_features=n_features,
        n_samples=n_samples,
        task=task,
        priority=priority,
        data_type=data_type,
        symmetry=symmetry,
        problem_structure=problem_structure,
        feature_interactions=feature_interactions,
    )

    # Phase A — hard filter
    candidates: dict[str, EncodingRule] = {}
    for name, rules in ENCODING_RULES.items():
        if _passes_hard_constraints(
            rules,
            n_features=n_features,
            data_type=data_type,
            symmetry=symmetry,
            trainable=trainable,
        ):
            candidates[name] = rules

    # Defensive fallback — unreachable in practice because ``angle`` has no
    # hard constraints, but kept for safety.  Uses ``_score_to_confidence``
    # to remain consistent with the documented confidence range [0.50, 0.95].
    if not candidates:
        return Recommendation(
            encoding_name="angle",
            explanation=(
                "No encoding matches all constraints; angle encoding "
                "is the safest general-purpose fallback"
            ),
            alternatives=[],
            confidence=_score_to_confidence(0.0),
        )

    # Phase B — score candidates
    scores: dict[str, float] = {}
    for name, rules in candidates.items():
        scores[name] = _compute_score(
            name,
            rules,
            n_features=n_features,
            n_samples=n_samples,
            task=task,
            hardware=hardware,
            priority=priority,
            data_type=data_type,
            symmetry=symmetry,
            trainable=trainable,
            problem_structure=problem_structure,
            feature_interactions=feature_interactions,
        )

    # Rank and select
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_name, best_score = ranked[0]
    alternatives = [name for name, _ in ranked[1:4]]

    confidence = _score_to_confidence(best_score)

    explanation = _generate_explanation(
        best_name,
        candidates[best_name],
        priority=priority,
        n_features=n_features,
    )

    return Recommendation(
        encoding_name=best_name,
        explanation=explanation,
        alternatives=alternatives,
        confidence=confidence,
    )
