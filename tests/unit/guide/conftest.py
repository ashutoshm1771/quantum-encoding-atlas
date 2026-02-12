"""Pytest fixtures for the guide (recommender) module tests."""

from __future__ import annotations

import pytest

from encoding_atlas.guide.decision_tree import EncodingDecisionTree
from encoding_atlas.guide.recommender import Recommendation, recommend_encoding
from encoding_atlas.guide.rules import ENCODING_RULES

# ---------------------------------------------------------------------------
# Canonical names
# ---------------------------------------------------------------------------

ALL_ENCODING_NAMES: list[str] = sorted(ENCODING_RULES.keys())
"""All 16 canonical encoding names, sorted alphabetically."""


# ---------------------------------------------------------------------------
# Trigger parameter sets — one per encoding
# ---------------------------------------------------------------------------
# Each mapping is guaranteed to cause the named encoding to be the *primary*
# recommendation from both ``recommend_encoding()`` and
# ``EncodingDecisionTree.decide()``.

ENCODING_TRIGGER_PARAMS: dict[str, dict] = {
    "angle": dict(n_features=4, priority="speed"),
    "basis": dict(n_features=4, data_type="binary"),
    "higher_order_angle": dict(n_features=4, feature_interactions="polynomial"),
    "iqp": dict(n_features=4, priority="accuracy"),
    "zz_feature_map": dict(n_features=6, priority="accuracy"),
    "pauli_feature_map": dict(n_features=4, feature_interactions="custom_pauli"),
    "data_reuploading": dict(n_features=4, priority="trainability"),
    "hardware_efficient": dict(n_features=4, priority="noise_resilience"),
    "amplitude": dict(n_features=16, priority="accuracy"),
    "qaoa": dict(n_features=4, problem_structure="combinatorial"),
    "hamiltonian": dict(n_features=4, problem_structure="physics_simulation"),
    "trainable": dict(n_features=4, trainable=True),
    "symmetry_inspired": dict(n_features=4, symmetry="general"),
    "so2_equivariant": dict(n_features=2, symmetry="rotation"),
    "cyclic_equivariant": dict(n_features=4, symmetry="cyclic"),
    "swap_equivariant": dict(n_features=4, symmetry="permutation_pairs"),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def all_encoding_names() -> list[str]:
    """Complete list of all 16 canonical encoding names."""
    return ALL_ENCODING_NAMES


@pytest.fixture
def decision_tree() -> EncodingDecisionTree:
    """Fresh EncodingDecisionTree instance."""
    return EncodingDecisionTree()


@pytest.fixture
def encoding_trigger_params() -> dict[str, dict]:
    """Trigger param sets keyed by encoding name."""
    return {k: dict(v) for k, v in ENCODING_TRIGGER_PARAMS.items()}


@pytest.fixture
def default_recommendation() -> Recommendation:
    """Recommendation produced with all-default params (n_features=4)."""
    return recommend_encoding(n_features=4)
