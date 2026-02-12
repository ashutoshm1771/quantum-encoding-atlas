"""Tests for encoding_atlas.guide.rules.

Covers:
- Completeness of the ENCODING_RULES dictionary (all 16 encodings present).
- TypedDict field presence and type correctness.
- Hard constraint enforcement via ``_passes_hard_constraints()``.
- Public ``get_matching_encodings()`` API with soft and hard filters.

Run with: pytest tests/unit/guide/test_rules.py -v
"""

from __future__ import annotations

import pytest

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
    get_matching_encodings,
)

# =========================================================================
# Expected canonical names (must stay in sync with _registry.py)
# =========================================================================

EXPECTED_ENCODING_NAMES: frozenset[str] = frozenset(
    {
        "angle",
        "basis",
        "higher_order_angle",
        "iqp",
        "zz_feature_map",
        "pauli_feature_map",
        "data_reuploading",
        "hardware_efficient",
        "amplitude",
        "qaoa",
        "hamiltonian",
        "trainable",
        "symmetry_inspired",
        "so2_equivariant",
        "cyclic_equivariant",
        "swap_equivariant",
    }
)

# All fields required by the EncodingRule TypedDict
REQUIRED_FIELDS: frozenset[str] = frozenset(EncodingRule.__annotations__.keys())


# =========================================================================
# Completeness tests
# =========================================================================


class TestEncodingRulesCompleteness:
    """Ensure every encoding is present with a valid rule entry."""

    def test_all_16_encodings_present(self) -> None:
        """ENCODING_RULES must contain exactly 16 canonical entries."""
        assert len(ENCODING_RULES) == 16

    def test_exact_encoding_names(self) -> None:
        """The key set must match the expected canonical names exactly."""
        assert set(ENCODING_RULES.keys()) == EXPECTED_ENCODING_NAMES

    @pytest.mark.parametrize("name", sorted(EXPECTED_ENCODING_NAMES))
    def test_all_required_fields_present(self, name: str) -> None:
        """Every entry must contain all EncodingRule fields."""
        entry = ENCODING_RULES[name]
        missing = REQUIRED_FIELDS - set(entry.keys())
        assert not missing, f"{name} is missing fields: {missing}"

    @pytest.mark.parametrize("name", sorted(EXPECTED_ENCODING_NAMES))
    def test_best_for_non_empty(self, name: str) -> None:
        """Every encoding must have at least one best_for tag."""
        assert len(ENCODING_RULES[name]["best_for"]) >= 1

    @pytest.mark.parametrize("name", sorted(EXPECTED_ENCODING_NAMES))
    def test_qubit_scaling_valid(self, name: str) -> None:
        """qubit_scaling must be 'linear' or 'logarithmic'."""
        assert ENCODING_RULES[name]["qubit_scaling"] in ("linear", "logarithmic")

    @pytest.mark.parametrize("name", sorted(EXPECTED_ENCODING_NAMES))
    def test_circuit_depth_valid(self, name: str) -> None:
        """circuit_depth must be one of the four valid categories."""
        assert ENCODING_RULES[name]["circuit_depth"] in (
            "constant",
            "shallow",
            "moderate",
            "deep",
        )

    @pytest.mark.parametrize("name", sorted(EXPECTED_ENCODING_NAMES))
    def test_simulable_is_bool(self, name: str) -> None:
        """simulable must be a boolean."""
        assert isinstance(ENCODING_RULES[name]["simulable"], bool)

    @pytest.mark.parametrize("name", sorted(EXPECTED_ENCODING_NAMES))
    def test_max_features_is_int_or_none(self, name: str) -> None:
        """max_features must be a positive int or None."""
        val = ENCODING_RULES[name]["max_features"]
        assert val is None or (isinstance(val, int) and val > 0)


# =========================================================================
# Hard constraint tests
# =========================================================================


class TestHardConstraints:
    """Test _passes_hard_constraints() for each constraint category."""

    # -- data type ---------------------------------------------------------

    def test_basis_requires_binary_data(self) -> None:
        """Basis encoding must fail for continuous data."""
        rules = ENCODING_RULES["basis"]
        assert _passes_hard_constraints(rules, data_type="binary")
        assert _passes_hard_constraints(rules, data_type="discrete")
        assert not _passes_hard_constraints(rules, data_type="continuous")

    def test_angle_accepts_any_data_type(self) -> None:
        """Angle encoding has no data-type constraint."""
        rules = ENCODING_RULES["angle"]
        for dt in ("continuous", "binary", "discrete"):
            assert _passes_hard_constraints(rules, data_type=dt)

    # -- exact feature count -----------------------------------------------

    def test_so2_requires_exactly_2_features(self) -> None:
        """SO2 equivariant must fail for n_features != 2."""
        rules = ENCODING_RULES["so2_equivariant"]
        assert _passes_hard_constraints(rules, n_features=2, symmetry="rotation")
        assert not _passes_hard_constraints(rules, n_features=3, symmetry="rotation")
        assert not _passes_hard_constraints(rules, n_features=1, symmetry="rotation")

    def test_so2_passes_when_n_features_none(self) -> None:
        """Feature-count checks are skipped when n_features is None."""
        rules = ENCODING_RULES["so2_equivariant"]
        assert _passes_hard_constraints(rules, n_features=None, symmetry="rotation")

    # -- even features -----------------------------------------------------

    def test_swap_requires_even_features(self) -> None:
        """Swap equivariant must fail for odd n_features."""
        rules = ENCODING_RULES["swap_equivariant"]
        assert _passes_hard_constraints(
            rules, n_features=4, symmetry="permutation_pairs"
        )
        assert not _passes_hard_constraints(
            rules, n_features=3, symmetry="permutation_pairs"
        )
        assert _passes_hard_constraints(
            rules, n_features=6, symmetry="permutation_pairs"
        )

    # -- symmetry ----------------------------------------------------------

    def test_so2_requires_rotation_symmetry(self) -> None:
        """SO2 equivariant must fail when symmetry is not 'rotation'."""
        rules = ENCODING_RULES["so2_equivariant"]
        assert _passes_hard_constraints(rules, n_features=2, symmetry="rotation")
        assert not _passes_hard_constraints(rules, n_features=2, symmetry=None)
        assert not _passes_hard_constraints(rules, n_features=2, symmetry="cyclic")

    def test_cyclic_requires_cyclic_symmetry(self) -> None:
        """Cyclic equivariant must fail when symmetry is not 'cyclic'."""
        rules = ENCODING_RULES["cyclic_equivariant"]
        assert _passes_hard_constraints(rules, n_features=4, symmetry="cyclic")
        assert not _passes_hard_constraints(rules, n_features=4, symmetry=None)
        assert not _passes_hard_constraints(rules, n_features=4, symmetry="rotation")

    def test_symmetry_inspired_requires_general(self) -> None:
        """Symmetry-inspired must fail when symmetry is not 'general'."""
        rules = ENCODING_RULES["symmetry_inspired"]
        assert _passes_hard_constraints(rules, symmetry="general")
        assert not _passes_hard_constraints(rules, symmetry=None)
        assert not _passes_hard_constraints(rules, symmetry="rotation")

    def test_non_symmetry_encodings_ignore_symmetry(self) -> None:
        """Encodings without requires_symmetry accept any symmetry value."""
        rules = ENCODING_RULES["angle"]
        assert _passes_hard_constraints(rules, symmetry=None)
        assert _passes_hard_constraints(rules, symmetry="rotation")
        assert _passes_hard_constraints(rules, symmetry="anything_else")

    # -- trainable ---------------------------------------------------------

    def test_trainable_requires_trainable_flag(self) -> None:
        """Trainable encoding must fail when trainable=False."""
        rules = ENCODING_RULES["trainable"]
        assert _passes_hard_constraints(rules, trainable=True)
        assert not _passes_hard_constraints(rules, trainable=False)

    def test_non_trainable_encodings_accept_either_flag(self) -> None:
        """Encodings without requires_trainable accept both flag values."""
        rules = ENCODING_RULES["iqp"]
        assert _passes_hard_constraints(rules, trainable=True)
        assert _passes_hard_constraints(rules, trainable=False)

    # -- max features ------------------------------------------------------

    @pytest.mark.parametrize(
        "name,max_val",
        [
            ("iqp", 12),
            ("zz_feature_map", 12),
            ("pauli_feature_map", 12),
            ("data_reuploading", 8),
            ("higher_order_angle", 10),
        ],
    )
    def test_max_features_enforced(self, name: str, max_val: int) -> None:
        """Encodings with max_features must fail when exceeded."""
        rules = ENCODING_RULES[name]
        assert _passes_hard_constraints(rules, n_features=max_val)
        assert not _passes_hard_constraints(rules, n_features=max_val + 1)

    def test_unlimited_features_for_angle(self) -> None:
        """Angle encoding accepts arbitrarily many features."""
        rules = ENCODING_RULES["angle"]
        assert _passes_hard_constraints(rules, n_features=1000)


# =========================================================================
# get_matching_encodings tests
# =========================================================================


class TestGetMatchingEncodings:
    """Test the public get_matching_encodings() API."""

    def test_returns_list(self) -> None:
        """Return value is always a list."""
        result = get_matching_encodings(["speed"])
        assert isinstance(result, list)

    def test_matching_single_requirement(self) -> None:
        """Requiring 'speed' should include 'angle'."""
        result = get_matching_encodings(["speed"])
        assert "angle" in result

    def test_constraint_filters_out(self) -> None:
        """An avoid_when constraint should exclude matching encodings."""
        # IQP has "noisy_hardware" in avoid_when
        with_constraint = get_matching_encodings(
            ["quantum_advantage"],
            constraints=["noisy_hardware"],
        )
        assert "iqp" not in with_constraint

        without_constraint = get_matching_encodings(["quantum_advantage"])
        assert "iqp" in without_constraint

    def test_hard_filter_applied(self) -> None:
        """data_type='binary' should exclude continuous-only encodings."""
        result = get_matching_encodings(
            ["speed"],
            data_type="binary",
        )
        # Basis should appear (it has "speed" in best_for and accepts binary)
        assert "basis" in result
        # No encoding requiring a symmetry should appear
        for name in result:
            rules = ENCODING_RULES[name]
            if rules["requires_data_type"] is not None:
                assert "binary" in rules["requires_data_type"]

    def test_empty_requirements_returns_empty(self) -> None:
        """No requirement tags → no soft-match → empty result."""
        result = get_matching_encodings([])
        assert result == []

    def test_nonexistent_requirement_returns_empty(self) -> None:
        """A tag that no encoding has → empty result."""
        result = get_matching_encodings(["nonexistent_tag_xyz"])
        assert result == []

    def test_trainable_filter(self) -> None:
        """trainable=True should include 'trainable' for matching tags."""
        result = get_matching_encodings(
            ["trainability"],
            trainable=True,
        )
        assert "trainable" in result

    def test_trainable_excluded_by_default(self) -> None:
        """Without trainable=True, 'trainable' is filtered out."""
        result = get_matching_encodings(["trainability"])
        assert "trainable" not in result

    def test_symmetry_filter(self) -> None:
        """symmetry='cyclic' should enable cyclic_equivariant."""
        result = get_matching_encodings(
            ["cyclic_symmetry"],
            symmetry="cyclic",
        )
        assert "cyclic_equivariant" in result

    def test_n_features_filter(self) -> None:
        """n_features=20 should exclude encodings with max_features < 20."""
        result = get_matching_encodings(
            ["expressibility"],
            n_features=20,
        )
        # IQP has max_features=12, should be excluded
        assert "iqp" not in result


# =========================================================================
# VALID_* constant tests
# =========================================================================


class TestValidConstants:
    """Verify the canonical VALID_* frozenset constants."""

    def test_valid_data_types(self) -> None:
        """VALID_DATA_TYPES must contain exactly the expected values."""
        assert frozenset({"continuous", "binary", "discrete"}) == VALID_DATA_TYPES

    def test_valid_symmetries(self) -> None:
        assert (
            frozenset({"rotation", "cyclic", "permutation_pairs", "general"})
            == VALID_SYMMETRIES
        )

    def test_valid_priorities(self) -> None:
        assert (
            frozenset({"accuracy", "trainability", "speed", "noise_resilience"})
            == VALID_PRIORITIES
        )

    def test_valid_tasks(self) -> None:
        assert frozenset({"classification", "regression"}) == VALID_TASKS

    def test_valid_problem_structures(self) -> None:
        assert (
            frozenset({"combinatorial", "physics_simulation", "time_series"})
            == VALID_PROBLEM_STRUCTURES
        )

    def test_valid_feature_interactions(self) -> None:
        assert frozenset({"polynomial", "custom_pauli"}) == VALID_FEATURE_INTERACTIONS

    def test_all_constants_are_frozensets(self) -> None:
        """Canonical constants must be frozensets (immutable)."""
        for const in (
            VALID_DATA_TYPES,
            VALID_SYMMETRIES,
            VALID_PRIORITIES,
            VALID_TASKS,
            VALID_PROBLEM_STRUCTURES,
            VALID_FEATURE_INTERACTIONS,
        ):
            assert isinstance(const, frozenset)

    def test_encoding_rules_symmetry_values_in_valid_set(self) -> None:
        """Every requires_symmetry value must be in VALID_SYMMETRIES."""
        for name, rules in ENCODING_RULES.items():
            if rules["requires_symmetry"] is not None:
                assert rules["requires_symmetry"] in VALID_SYMMETRIES, (
                    f"{name} has requires_symmetry="
                    f"'{rules['requires_symmetry']}' not in VALID_SYMMETRIES"
                )

    def test_encoding_rules_data_type_values_in_valid_set(self) -> None:
        """Every requires_data_type value must be in VALID_DATA_TYPES."""
        for name, rules in ENCODING_RULES.items():
            if rules["requires_data_type"] is not None:
                for dt in rules["requires_data_type"]:
                    assert dt in VALID_DATA_TYPES, (
                        f"{name} has requires_data_type '{dt}' "
                        f"not in VALID_DATA_TYPES"
                    )


# =========================================================================
# time_series tag in data_reuploading
# =========================================================================


class TestTimeSeriestag:
    """Verify that data_reuploading supports time_series."""

    def test_data_reuploading_has_time_series_tag(self) -> None:
        """data_reuploading best_for must include 'time_series'."""
        assert "time_series" in ENCODING_RULES["data_reuploading"]["best_for"]

    def test_time_series_matching(self) -> None:
        """get_matching_encodings with 'time_series' should include
        data_reuploading."""
        result = get_matching_encodings(["time_series"])
        assert "data_reuploading" in result
