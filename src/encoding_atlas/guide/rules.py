"""Rule-based encoding recommendations.

This module defines the encoding knowledge base used by the recommender
and decision tree. Each encoding has:

- **Soft tags** (``best_for`` / ``avoid_when``): influence scoring but are not
  hard gates.
- **Hard constraints**: boolean preconditions that *must* be satisfied for the
  encoding to be a valid candidate (e.g., ``requires_even_features``).

The canonical names used as dictionary keys correspond exactly to the primary
registry names in ``encoding_atlas.encodings._registry``.

Public constants
----------------
``ENCODING_RULES``
    Complete encoding knowledge base (16 entries).
``VALID_DATA_TYPES``, ``VALID_SYMMETRIES``, ``VALID_PRIORITIES``,
``VALID_TASKS``, ``VALID_PROBLEM_STRUCTURES``, ``VALID_FEATURE_INTERACTIONS``
    Canonical sets of accepted parameter values, used by the recommender and
    decision tree for runtime validation.
"""

from __future__ import annotations

from typing import Literal, TypedDict

# ---------------------------------------------------------------------------
# Typed schema for every encoding's rule entry
# ---------------------------------------------------------------------------


class EncodingRule(TypedDict):
    """Schema for a single encoding's recommendation rules.

    Attributes
    ----------
    best_for : list[str]
        Tags describing scenarios where this encoding excels.
    avoid_when : list[str]
        Tags describing scenarios where this encoding should be avoided.
    max_features : int | None
        Upper bound on feature count (``None`` = no limit).
    simulable : bool
        Whether the encoding is efficiently classically simulable.
    requires_data_type : list[str] | None
        Hard constraint — if set, the user's ``data_type`` must be in this
        list for the encoding to be eligible.
    requires_symmetry : str | None
        Hard constraint — if set, the user must specify this exact symmetry
        type for the encoding to be eligible.
    requires_n_features : int | None
        Hard constraint — if set, ``n_features`` must equal this value.
    requires_even_features : bool
        Hard constraint — if ``True``, ``n_features`` must be even.
    requires_trainable : bool
        Hard constraint — if ``True``, the user must opt in with
        ``trainable=True``.
    qubit_scaling : Literal["linear", "logarithmic"]
        How qubit count scales with feature count.
    circuit_depth : Literal["constant", "shallow", "moderate", "deep"]
        Qualitative circuit depth category.
    """

    best_for: list[str]
    avoid_when: list[str]
    max_features: int | None
    simulable: bool
    requires_data_type: list[str] | None
    requires_symmetry: str | None
    requires_n_features: int | None
    requires_even_features: bool
    requires_trainable: bool
    qubit_scaling: Literal["linear", "logarithmic"]
    circuit_depth: Literal["constant", "shallow", "moderate", "deep"]


# ---------------------------------------------------------------------------
# Valid parameter values — canonical source for runtime validation
# ---------------------------------------------------------------------------

VALID_DATA_TYPES: frozenset[str] = frozenset({"continuous", "binary", "discrete"})
VALID_SYMMETRIES: frozenset[str] = frozenset(
    {"rotation", "cyclic", "permutation_pairs", "general"}
)
VALID_PRIORITIES: frozenset[str] = frozenset(
    {"accuracy", "trainability", "speed", "noise_resilience"}
)
VALID_TASKS: frozenset[str] = frozenset({"classification", "regression"})
VALID_PROBLEM_STRUCTURES: frozenset[str] = frozenset(
    {"combinatorial", "physics_simulation", "time_series"}
)
VALID_FEATURE_INTERACTIONS: frozenset[str] = frozenset({"polynomial", "custom_pauli"})


# ---------------------------------------------------------------------------
# Complete encoding knowledge base — all 16 encodings
# ---------------------------------------------------------------------------

ENCODING_RULES: dict[str, EncodingRule] = {
    # ------------------------------------------------------------------
    # Non-entangling, classically simulable
    # ------------------------------------------------------------------
    "angle": {
        "best_for": [
            "speed",
            "simplicity",
            "product_states",
            "nisq_hardware",
        ],
        "avoid_when": [
            "need_entanglement",
            "quantum_advantage",
            "feature_interactions",
        ],
        "max_features": None,
        "simulable": True,
        "requires_data_type": None,
        "requires_symmetry": None,
        "requires_n_features": None,
        "requires_even_features": False,
        "requires_trainable": False,
        "qubit_scaling": "linear",
        "circuit_depth": "constant",
    },
    "basis": {
        "best_for": [
            "binary_data",
            "combinatorial",
            "simplicity",
            "speed",
        ],
        "avoid_when": [
            "continuous_data",
            "feature_interactions",
            "need_entanglement",
        ],
        "max_features": None,
        "simulable": True,
        "requires_data_type": ["binary", "discrete"],
        "requires_symmetry": None,
        "requires_n_features": None,
        "requires_even_features": False,
        "requires_trainable": False,
        "qubit_scaling": "linear",
        "circuit_depth": "constant",
    },
    "higher_order_angle": {
        "best_for": [
            "polynomial_features",
            "feature_interactions",
            "product_states",
        ],
        "avoid_when": [
            "many_features",
            "need_entanglement",
            "quantum_advantage",
        ],
        "max_features": 10,
        "simulable": True,
        "requires_data_type": None,
        "requires_symmetry": None,
        "requires_n_features": None,
        "requires_even_features": False,
        "requires_trainable": False,
        "qubit_scaling": "linear",
        "circuit_depth": "shallow",
    },
    # ------------------------------------------------------------------
    # Standard entangling encodings
    # ------------------------------------------------------------------
    "iqp": {
        "best_for": [
            "quantum_advantage",
            "expressibility",
            "kernel_methods",
        ],
        "avoid_when": [
            "many_features",
            "noisy_hardware",
            "nisq_hardware",
        ],
        "max_features": 12,
        "simulable": False,
        "requires_data_type": None,
        "requires_symmetry": None,
        "requires_n_features": None,
        "requires_even_features": False,
        "requires_trainable": False,
        "qubit_scaling": "linear",
        "circuit_depth": "moderate",
    },
    "zz_feature_map": {
        "best_for": [
            "kernel_methods",
            "standard_benchmark",
            "balanced",
        ],
        "avoid_when": [
            "very_noisy_hardware",
            "many_features",
        ],
        "max_features": 12,
        "simulable": False,
        "requires_data_type": None,
        "requires_symmetry": None,
        "requires_n_features": None,
        "requires_even_features": False,
        "requires_trainable": False,
        "qubit_scaling": "linear",
        "circuit_depth": "moderate",
    },
    "pauli_feature_map": {
        "best_for": [
            "custom_pauli",
            "kernel_methods",
            "research",
            "feature_interactions",
        ],
        "avoid_when": [
            "simplicity",
            "very_noisy_hardware",
        ],
        "max_features": 12,
        "simulable": False,
        "requires_data_type": None,
        "requires_symmetry": None,
        "requires_n_features": None,
        "requires_even_features": False,
        "requires_trainable": False,
        "qubit_scaling": "linear",
        "circuit_depth": "moderate",
    },
    "data_reuploading": {
        "best_for": [
            "universal_approximation",
            "expressibility",
            "trainability",
            "time_series",
        ],
        "avoid_when": [
            "limited_depth",
            "nisq_hardware",
            "speed",
        ],
        "max_features": 8,
        "simulable": False,
        "requires_data_type": None,
        "requires_symmetry": None,
        "requires_n_features": None,
        "requires_even_features": False,
        "requires_trainable": False,
        "qubit_scaling": "linear",
        "circuit_depth": "deep",
    },
    "hardware_efficient": {
        "best_for": [
            "nisq_hardware",
            "native_gates",
            "noise_resilience",
        ],
        "avoid_when": [
            "simulator_only",
            "quantum_advantage",
        ],
        "max_features": None,
        "simulable": False,
        "requires_data_type": None,
        "requires_symmetry": None,
        "requires_n_features": None,
        "requires_even_features": False,
        "requires_trainable": False,
        "qubit_scaling": "linear",
        "circuit_depth": "shallow",
    },
    # ------------------------------------------------------------------
    # Specialised encodings
    # ------------------------------------------------------------------
    "amplitude": {
        "best_for": [
            "many_features",
            "compression",
            "exponential_compression",
        ],
        "avoid_when": [
            "nisq_hardware",
            "shallow_circuits",
            "speed",
        ],
        "max_features": None,
        "simulable": False,
        "requires_data_type": None,
        "requires_symmetry": None,
        "requires_n_features": None,
        "requires_even_features": False,
        "requires_trainable": False,
        "qubit_scaling": "logarithmic",
        "circuit_depth": "deep",
    },
    "qaoa": {
        "best_for": [
            "combinatorial",
            "graph_optimization",
            "qaoa_structure",
        ],
        "avoid_when": [
            "continuous_features_only",
            "speed",
        ],
        "max_features": None,
        "simulable": False,
        "requires_data_type": None,
        "requires_symmetry": None,
        "requires_n_features": None,
        "requires_even_features": False,
        "requires_trainable": False,
        "qubit_scaling": "linear",
        "circuit_depth": "moderate",
    },
    "hamiltonian": {
        "best_for": [
            "physics_simulation",
            "time_evolution",
            "expressibility",
        ],
        "avoid_when": [
            "speed",
            "simplicity",
            "nisq_hardware",
        ],
        "max_features": None,
        "simulable": False,
        "requires_data_type": None,
        "requires_symmetry": None,
        "requires_n_features": None,
        "requires_even_features": False,
        "requires_trainable": False,
        "qubit_scaling": "linear",
        "circuit_depth": "deep",
    },
    "trainable": {
        "best_for": [
            "task_specific",
            "optimization",
            "trainability",
            "qnn",
        ],
        "avoid_when": [
            "no_optimization_budget",
            "simplicity",
        ],
        "max_features": None,
        "simulable": False,
        "requires_data_type": None,
        "requires_symmetry": None,
        "requires_n_features": None,
        "requires_even_features": False,
        "requires_trainable": True,
        "qubit_scaling": "linear",
        "circuit_depth": "moderate",
    },
    # ------------------------------------------------------------------
    # Symmetry-aware (heuristic)
    # ------------------------------------------------------------------
    "symmetry_inspired": {
        "best_for": [
            "symmetry_general",
            "inductive_bias",
            "heuristic_symmetry",
        ],
        "avoid_when": [
            "rigorous_equivariance",
            "speed",
        ],
        "max_features": None,
        "simulable": False,
        "requires_data_type": None,
        "requires_symmetry": "general",
        "requires_n_features": None,
        "requires_even_features": False,
        "requires_trainable": False,
        "qubit_scaling": "linear",
        "circuit_depth": "moderate",
    },
    # ------------------------------------------------------------------
    # Rigorous equivariant encodings
    # ------------------------------------------------------------------
    "so2_equivariant": {
        "best_for": [
            "rotation_equivariance",
            "2d_rotation",
            "rigorous_equivariance",
        ],
        "avoid_when": [
            "many_features",
            "non_2d_data",
        ],
        "max_features": 2,
        "simulable": False,
        "requires_data_type": None,
        "requires_symmetry": "rotation",
        "requires_n_features": 2,
        "requires_even_features": False,
        "requires_trainable": False,
        "qubit_scaling": "linear",
        "circuit_depth": "moderate",
    },
    "cyclic_equivariant": {
        "best_for": [
            "cyclic_symmetry",
            "periodic_data",
            "rigorous_equivariance",
        ],
        "avoid_when": [
            "non_periodic_data",
        ],
        "max_features": None,
        "simulable": False,
        "requires_data_type": None,
        "requires_symmetry": "cyclic",
        "requires_n_features": None,
        "requires_even_features": False,
        "requires_trainable": False,
        "qubit_scaling": "linear",
        "circuit_depth": "moderate",
    },
    "swap_equivariant": {
        "best_for": [
            "permutation_pairs",
            "paired_features",
            "rigorous_equivariance",
        ],
        "avoid_when": [
            "odd_features",
            "non_paired_data",
        ],
        "max_features": None,
        "simulable": False,
        "requires_data_type": None,
        "requires_symmetry": "permutation_pairs",
        "requires_n_features": None,
        "requires_even_features": True,
        "requires_trainable": False,
        "qubit_scaling": "linear",
        "circuit_depth": "moderate",
    },
}


# ---------------------------------------------------------------------------
# Hard constraint checker
# ---------------------------------------------------------------------------


def _passes_hard_constraints(
    rules: EncodingRule,
    *,
    n_features: int | None = None,
    data_type: str = "continuous",
    symmetry: str | None = None,
    trainable: bool = False,
) -> bool:
    """Return ``True`` if the encoding is not disqualified by hard constraints.

    Hard constraints are *exclusion-based*: if any single constraint fails the
    encoding is eliminated from the candidate set regardless of how well it
    matches on soft tags.

    Parameters
    ----------
    rules : EncodingRule
        The rule entry for the encoding under test.
    n_features : int | None
        Number of input features (``None`` skips feature-count checks).
    data_type : str
        One of ``"continuous"``, ``"binary"``, ``"discrete"``.
    symmetry : str | None
        Symmetry type requested by the user (``None`` = no symmetry).
    trainable : bool
        Whether the user has opted into trainable encoding parameters.

    Returns
    -------
    bool
        ``True`` if all hard constraints pass, ``False`` otherwise.
    """
    # 1. Data-type constraint
    if (
        rules["requires_data_type"] is not None
        and data_type not in rules["requires_data_type"]
    ):
        return False

    # 2. Exact feature-count constraint (e.g. SO2 requires exactly 2)
    if (
        rules["requires_n_features"] is not None
        and n_features is not None
        and n_features != rules["requires_n_features"]
    ):
        return False

    # 3. Even-features constraint (e.g. SwapEquivariant)
    if (
        rules["requires_even_features"]
        and n_features is not None
        and n_features % 2 != 0
    ):
        return False

    # 4. Maximum feature-count constraint
    if (
        rules["max_features"] is not None
        and n_features is not None
        and n_features > rules["max_features"]
    ):
        return False

    # 5. Symmetry constraint — encoding requires a specific symmetry type
    if rules["requires_symmetry"] is not None and (
        symmetry is None or symmetry != rules["requires_symmetry"]
    ):
        return False

    # 6. Trainable constraint
    if rules["requires_trainable"] and not trainable:
        return False

    return True


# ---------------------------------------------------------------------------
# Public matching API
# ---------------------------------------------------------------------------


def get_matching_encodings(
    requirements: list[str],
    constraints: list[str] | None = None,
    *,
    n_features: int | None = None,
    data_type: str = "continuous",
    symmetry: str | None = None,
    trainable: bool = False,
) -> list[str]:
    """Get encodings matching requirements, constraints, and hard filters.

    This performs a two-phase check for each encoding:

    1. **Hard filter** — eliminates encodings whose structural preconditions
       are not met (data type, feature count, symmetry, trainability).
    2. **Soft match** — among survivors, selects those whose ``best_for`` tags
       overlap with *requirements* and whose ``avoid_when`` tags do not overlap
       with *constraints*.

    Parameters
    ----------
    requirements : list[str]
        Tags the encoding should be good at (matched against ``best_for``).
    constraints : list[str] | None
        Tags the encoding should *not* be associated with (matched against
        ``avoid_when``).  ``None`` means no soft constraints.
    n_features : int | None
        Number of input features for hard-filter checks.
    data_type : str
        Data type for hard-filter checks.
    symmetry : str | None
        Symmetry type for hard-filter checks.
    trainable : bool
        Trainable flag for hard-filter checks.

    Returns
    -------
    list[str]
        Encoding names that pass both hard and soft filters.
    """
    matches: list[str] = []

    for name, rules in ENCODING_RULES.items():
        # Phase A — hard filter
        if not _passes_hard_constraints(
            rules,
            n_features=n_features,
            data_type=data_type,
            symmetry=symmetry,
            trainable=trainable,
        ):
            continue

        # Phase B — soft tag match
        if any(req in rules["best_for"] for req in requirements):
            if constraints:
                if not any(c in rules["avoid_when"] for c in constraints):
                    matches.append(name)
            else:
                matches.append(name)

    return matches
