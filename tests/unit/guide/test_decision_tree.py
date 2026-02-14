"""Tests for encoding_atlas.guide.decision_tree.

Covers:
- Tree construction and structural validity.
- Every encoding reachable through the ``decide()`` method.
- Edge cases (wrong feature counts for symmetry, defaults).
- Input validation (ValueError on bad parameter values).
- ``time_series`` problem structure routing.
- Consistency between the decision tree and the recommender for
  unambiguous cases.

Run with: pytest tests/unit/guide/test_decision_tree.py -v
"""

from __future__ import annotations

from typing import Any

import pytest

from encoding_atlas.guide.decision_tree import EncodingDecisionTree
from encoding_atlas.guide.recommender import recommend_encoding
from encoding_atlas.guide.rules import ENCODING_RULES
from tests.unit.guide.conftest import ENCODING_TRIGGER_PARAMS

# =========================================================================
# Helpers
# =========================================================================


def _collect_leaves(node: Any) -> list[str]:
    """Recursively collect all leaf strings from the tree."""
    if isinstance(node, str):
        return [node]
    if isinstance(node, dict) and "options" in node:
        leaves: list[str] = []
        for child in node["options"].values():
            leaves.extend(_collect_leaves(child))
        return leaves
    return []


# =========================================================================
# Tree construction
# =========================================================================


class TestDecisionTreeConstruction:
    """Verify the tree's structural integrity."""

    def test_tree_is_dict(self, decision_tree: EncodingDecisionTree) -> None:
        assert isinstance(decision_tree.tree, dict)

    def test_root_has_question(self, decision_tree: EncodingDecisionTree) -> None:
        assert "question" in decision_tree.tree
        assert "options" in decision_tree.tree

    def test_all_leaves_are_valid_encoding_names(
        self, decision_tree: EncodingDecisionTree
    ) -> None:
        """Every leaf string must be a key in ENCODING_RULES."""
        leaves = _collect_leaves(decision_tree.tree)
        assert len(leaves) > 0, "Tree has no leaves"
        for leaf in leaves:
            assert leaf in ENCODING_RULES, f"Leaf '{leaf}' is not a valid encoding name"

    def test_all_16_encodings_appear_in_tree(
        self, decision_tree: EncodingDecisionTree
    ) -> None:
        """Every encoding must appear at least once as a tree leaf."""
        leaves = set(_collect_leaves(decision_tree.tree))
        missing = set(ENCODING_RULES.keys()) - leaves
        assert not missing, f"Encodings missing from tree: {missing}"


# =========================================================================
# Reachability — every encoding via decide()
# =========================================================================


class TestDecisionTreeReachability:
    """Each of the 16 encodings must be reachable via decide()."""

    @pytest.mark.parametrize(
        "encoding_name,params",
        sorted(ENCODING_TRIGGER_PARAMS.items()),
        ids=sorted(ENCODING_TRIGGER_PARAMS.keys()),
    )
    def test_decide_reaches_encoding(
        self,
        decision_tree: EncodingDecisionTree,
        encoding_name: str,
        params: dict,
    ) -> None:
        result = decision_tree.decide(**params)
        assert result == encoding_name, (
            f"Expected '{encoding_name}' but got '{result}' " f"for params {params}"
        )


# =========================================================================
# Edge cases
# =========================================================================


class TestDecisionTreeEdgeCases:
    """Boundary conditions and fall-through behaviour."""

    def test_default_params(self, decision_tree: EncodingDecisionTree) -> None:
        """decide() with no arguments should return a valid encoding."""
        result = decision_tree.decide()
        assert result in ENCODING_RULES

    def test_binary_data(self, decision_tree: EncodingDecisionTree) -> None:
        assert decision_tree.decide(data_type="binary") == "basis"

    def test_discrete_data(self, decision_tree: EncodingDecisionTree) -> None:
        assert decision_tree.decide(data_type="discrete") == "basis"

    def test_rotation_wrong_features_falls_through(
        self, decision_tree: EncodingDecisionTree
    ) -> None:
        """symmetry='rotation' with n_features=5 cannot match SO2;
        the tree should fall through to a general encoding."""
        result = decision_tree.decide(symmetry="rotation", n_features=5)
        assert result != "so2_equivariant"
        assert result in ENCODING_RULES

    def test_permutation_pairs_odd_features_falls_through(
        self, decision_tree: EncodingDecisionTree
    ) -> None:
        """symmetry='permutation_pairs' with odd n_features cannot match
        swap_equivariant; should fall through."""
        result = decision_tree.decide(symmetry="permutation_pairs", n_features=3)
        assert result != "swap_equivariant"
        assert result in ENCODING_RULES

    def test_n_features_1(self, decision_tree: EncodingDecisionTree) -> None:
        """Single feature should not crash."""
        result = decision_tree.decide(n_features=1)
        assert result in ENCODING_RULES

    def test_n_features_100(self, decision_tree: EncodingDecisionTree) -> None:
        """Very large feature count."""
        result = decision_tree.decide(n_features=100)
        assert result == "amplitude"

    def test_combined_flags_priority_order(
        self, decision_tree: EncodingDecisionTree
    ) -> None:
        """When multiple conditions are set, higher-priority levels win.
        e.g., symmetry='cyclic' + problem_structure='combinatorial'
        → cyclic wins because symmetry is checked before problem_structure."""
        result = decision_tree.decide(
            symmetry="cyclic", problem_structure="combinatorial", n_features=4
        )
        assert result == "cyclic_equivariant"

    def test_time_series_routes_to_data_reuploading(
        self, decision_tree: EncodingDecisionTree
    ) -> None:
        """problem_structure='time_series' should route to data_reuploading."""
        result = decision_tree.decide(n_features=4, problem_structure="time_series")
        assert result == "data_reuploading"

    def test_time_series_lower_priority_than_symmetry(
        self, decision_tree: EncodingDecisionTree
    ) -> None:
        """Symmetry check (level 2) should take precedence over
        problem_structure (level 4)."""
        result = decision_tree.decide(
            n_features=4,
            symmetry="cyclic",
            problem_structure="time_series",
        )
        assert result == "cyclic_equivariant"

    def test_trainable_takes_precedence_over_time_series(
        self, decision_tree: EncodingDecisionTree
    ) -> None:
        """Trainable check (level 3) should beat problem_structure (level 4)."""
        result = decision_tree.decide(
            n_features=4,
            trainable=True,
            problem_structure="time_series",
        )
        assert result == "trainable"


# =========================================================================
# Input validation
# =========================================================================


class TestDecisionTreeInputValidation:
    """Verify that invalid inputs to ``decide()`` raise ``ValueError``."""

    def test_invalid_data_type(self, decision_tree: EncodingDecisionTree) -> None:
        with pytest.raises(ValueError, match="data_type"):
            decision_tree.decide(data_type="float")  # type: ignore[arg-type]

    def test_invalid_priority(self, decision_tree: EncodingDecisionTree) -> None:
        with pytest.raises(ValueError, match="priority"):
            decision_tree.decide(priority="invalid")  # type: ignore[arg-type]

    def test_invalid_symmetry(self, decision_tree: EncodingDecisionTree) -> None:
        with pytest.raises(ValueError, match="symmetry"):
            decision_tree.decide(symmetry="translational")  # type: ignore[arg-type]

    def test_invalid_problem_structure(
        self, decision_tree: EncodingDecisionTree
    ) -> None:
        with pytest.raises(ValueError, match="problem_structure"):
            decision_tree.decide(problem_structure="unknown")  # type: ignore[arg-type]

    def test_invalid_feature_interactions(
        self, decision_tree: EncodingDecisionTree
    ) -> None:
        with pytest.raises(ValueError, match="feature_interactions"):
            decision_tree.decide(feature_interactions="quadratic")  # type: ignore[arg-type]

    def test_n_features_zero(self, decision_tree: EncodingDecisionTree) -> None:
        with pytest.raises(ValueError, match="n_features"):
            decision_tree.decide(n_features=0)

    def test_n_features_negative(self, decision_tree: EncodingDecisionTree) -> None:
        with pytest.raises(ValueError, match="n_features"):
            decision_tree.decide(n_features=-1)

    def test_invalid_trainable_type(self, decision_tree: EncodingDecisionTree) -> None:
        with pytest.raises(ValueError, match="trainable"):
            decision_tree.decide(trainable="yes")  # type: ignore[arg-type]

    def test_none_symmetry_is_valid(self, decision_tree: EncodingDecisionTree) -> None:
        """symmetry=None should NOT raise."""
        result = decision_tree.decide(symmetry=None)
        assert result in ENCODING_RULES

    def test_none_problem_structure_is_valid(
        self, decision_tree: EncodingDecisionTree
    ) -> None:
        """problem_structure=None should NOT raise."""
        result = decision_tree.decide(problem_structure=None)
        assert result in ENCODING_RULES

    def test_none_feature_interactions_is_valid(
        self, decision_tree: EncodingDecisionTree
    ) -> None:
        """feature_interactions=None should NOT raise."""
        result = decision_tree.decide(feature_interactions=None)
        assert result in ENCODING_RULES


# =========================================================================
# Consistency with recommender
# =========================================================================


class TestDecisionTreeConsistencyWithRecommender:
    """For the canonical trigger parameter sets, the decision tree and
    the recommender should agree."""

    @pytest.mark.parametrize(
        "encoding_name,params",
        sorted(ENCODING_TRIGGER_PARAMS.items()),
        ids=sorted(ENCODING_TRIGGER_PARAMS.keys()),
    )
    def test_tree_and_recommender_agree(
        self,
        decision_tree: EncodingDecisionTree,
        encoding_name: str,
        params: dict,
    ) -> None:
        tree_result = decision_tree.decide(**params)
        rec_result = recommend_encoding(**params).encoding_name
        assert tree_result == rec_result, (
            f"Tree says '{tree_result}', recommender says '{rec_result}' "
            f"for '{encoding_name}' with params {params}"
        )
