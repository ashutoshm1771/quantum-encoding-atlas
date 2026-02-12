"""Decision tree for encoding selection.

Provides a deterministic, interpretable encoding selection path based on
user-specified problem characteristics.  All 16 encodings are reachable
through the tree, with *data_reuploading* reachable by two distinct paths
(``problem_structure="time_series"`` and ``priority="trainability"``).

Decision priority (highest to lowest):

1. **Data type** — binary / discrete data routes to *basis*.
2. **Symmetry** — specific symmetry groups route to the corresponding
   equivariant encoding; ``"general"`` routes to *symmetry_inspired*.
3. **Trainable flag** — explicit opt-in routes to *trainable*.
4. **Problem structure** — combinatorial → *qaoa*,
   physics_simulation → *hamiltonian*, time_series → *data_reuploading*.
5. **Feature interactions** — polynomial → *higher_order_angle*,
   custom_pauli → *pauli_feature_map*.
6. **Priority** — speed → *angle*, noise_resilience → *hardware_efficient*,
   trainability → *data_reuploading*.
7. **Feature count** (accuracy fallback) — ≤4 → *iqp*, 5–8 → *zz_feature_map*,
   >8 → *amplitude*.
"""

from __future__ import annotations

from typing import Any, Literal

from encoding_atlas.guide.rules import (
    VALID_DATA_TYPES,
    VALID_FEATURE_INTERACTIONS,
    VALID_PRIORITIES,
    VALID_PROBLEM_STRUCTURES,
    VALID_SYMMETRIES,
)


class EncodingDecisionTree:
    """Decision tree for encoding selection.

    The tree is constructed once on instantiation and can be queried
    repeatedly via :meth:`decide`.  The nested dictionary representation
    is available as the :attr:`tree` attribute for inspection or
    visualisation.
    """

    def __init__(self) -> None:
        self.tree = self._build_tree()

    # -----------------------------------------------------------------
    # Tree structure
    # -----------------------------------------------------------------

    @staticmethod
    def _build_tree() -> dict[str, Any]:
        """Build the decision tree as a nested dictionary.

        Leaf values are encoding canonical names (strings).  Internal
        nodes are dictionaries with ``"question"`` and ``"options"`` keys.
        """
        return {
            "question": "What is your data type?",
            "options": {
                "binary": "basis",
                "discrete": "basis",
                "continuous": {
                    "question": "Does your data have a known symmetry?",
                    "options": {
                        "rotation (2D, n_features=2)": "so2_equivariant",
                        "cyclic": "cyclic_equivariant",
                        "permutation_pairs (even n_features)": "swap_equivariant",
                        "general (heuristic)": "symmetry_inspired",
                        "none": {
                            "question": "Do you want trainable encoding parameters?",
                            "options": {
                                "yes": "trainable",
                                "no": {
                                    "question": "What is the problem structure?",
                                    "options": {
                                        "combinatorial / graph": "qaoa",
                                        "physics simulation": "hamiltonian",
                                        "time_series / periodic": "data_reuploading",
                                        "none / general": {
                                            "question": "Do you need specific feature interactions?",
                                            "options": {
                                                "polynomial (no entanglement)": "higher_order_angle",
                                                "custom Pauli strings": "pauli_feature_map",
                                                "none": {
                                                    "question": "What is your optimisation priority?",
                                                    "options": {
                                                        "speed": "angle",
                                                        "noise_resilience": "hardware_efficient",
                                                        "trainability": "data_reuploading",
                                                        "accuracy": {
                                                            "question": "How many features?",
                                                            "options": {
                                                                "few (<= 4)": "iqp",
                                                                "medium (5-8)": "zz_feature_map",
                                                                "many (> 8)": "amplitude",
                                                            },
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }

    # -----------------------------------------------------------------
    # Programmatic decision
    # -----------------------------------------------------------------

    def decide(
        self,
        *,
        data_type: Literal["continuous", "binary", "discrete"] = "continuous",
        n_features: int = 4,
        symmetry: (
            Literal["rotation", "cyclic", "permutation_pairs", "general"] | None
        ) = None,
        trainable: bool = False,
        priority: Literal[
            "accuracy", "speed", "noise_resilience", "trainability"
        ] = "accuracy",
        problem_structure: (
            Literal["combinatorial", "physics_simulation", "time_series"] | None
        ) = None,
        feature_interactions: Literal["polynomial", "custom_pauli"] | None = None,
    ) -> str:
        """Make a deterministic encoding selection based on inputs.

        Parameters
        ----------
        data_type : {"continuous", "binary", "discrete"}
            Input data type (default ``"continuous"``).
        n_features : int
            Number of input features (default 4, must be >= 1).
        symmetry : {"rotation", "cyclic", "permutation_pairs", "general"} | None
            Known symmetry in the data (default ``None``).
        trainable : bool
            Whether learnable parameters are desired (default ``False``).
        priority : {"accuracy", "speed", "noise_resilience", "trainability"}
            Optimisation priority (default ``"accuracy"``).
        problem_structure : {"combinatorial", "physics_simulation", "time_series"} | None
            Domain structure (default ``None``).
        feature_interactions : {"polynomial", "custom_pauli"} | None
            Desired interaction type (default ``None``).

        Returns
        -------
        str
            Canonical encoding name.

        Raises
        ------
        ValueError
            If any parameter value is outside its valid set.
        """
        self._validate_inputs(
            data_type=data_type,
            n_features=n_features,
            symmetry=symmetry,
            priority=priority,
            problem_structure=problem_structure,
            feature_interactions=feature_interactions,
        )

        # Level 1 — data type
        if data_type in ("binary", "discrete"):
            return "basis"

        # Level 2 — symmetry
        if symmetry == "rotation" and n_features == 2:
            return "so2_equivariant"
        if symmetry == "cyclic":
            return "cyclic_equivariant"
        if symmetry == "permutation_pairs" and n_features % 2 == 0:
            return "swap_equivariant"
        if symmetry == "general":
            return "symmetry_inspired"

        # Level 3 — trainable
        if trainable:
            return "trainable"

        # Level 4 — problem structure
        if problem_structure == "combinatorial":
            return "qaoa"
        if problem_structure == "physics_simulation":
            return "hamiltonian"
        if problem_structure == "time_series":
            return "data_reuploading"

        # Level 5 — feature interactions
        if feature_interactions == "polynomial":
            return "higher_order_angle"
        if feature_interactions == "custom_pauli":
            return "pauli_feature_map"

        # Level 6 — priority
        if priority == "speed":
            return "angle"
        if priority == "noise_resilience":
            return "hardware_efficient"
        if priority == "trainability":
            return "data_reuploading"

        # Level 7 — accuracy-based feature count selection (default)
        if n_features <= 4:
            return "iqp"
        if n_features <= 8:
            return "zz_feature_map"
        return "amplitude"

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        *,
        data_type: str,
        n_features: int,
        symmetry: str | None,
        priority: str,
        problem_structure: str | None,
        feature_interactions: str | None,
    ) -> None:
        """Validate ``decide`` inputs, raising ``ValueError`` on bad values."""
        if not isinstance(n_features, int) or n_features < 1:
            raise ValueError(
                f"n_features must be a positive integer, got {n_features!r}"
            )
        if data_type not in VALID_DATA_TYPES:
            raise ValueError(
                f"data_type must be one of {sorted(VALID_DATA_TYPES)}, "
                f"got {data_type!r}"
            )
        if symmetry is not None and symmetry not in VALID_SYMMETRIES:
            raise ValueError(
                f"symmetry must be one of {sorted(VALID_SYMMETRIES)} or None, "
                f"got {symmetry!r}"
            )
        if priority not in VALID_PRIORITIES:
            raise ValueError(
                f"priority must be one of {sorted(VALID_PRIORITIES)}, "
                f"got {priority!r}"
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
