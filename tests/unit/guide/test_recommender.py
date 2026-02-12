"""Tests for encoding_atlas.guide.recommender.

Covers:
- Recommendation dataclass field validation.
- Backward compatibility with the original 5-parameter signature.
- Every encoding reachable as primary recommendation.
- Hard constraint enforcement (no invalid encoding is ever recommended).
- Alternatives list correctness.
- Confidence value ranges and continuity.
- Each new parameter value.
- Task parameter impact on scoring.
- ``avoid_when`` penalty on real hardware.
- Input validation (ValueError on bad inputs).
- Edge cases (extreme feature counts, conflicting constraints).

Run with: pytest tests/unit/guide/test_recommender.py -v
"""

from __future__ import annotations

import pytest

from encoding_atlas.guide.recommender import (
    _compute_score,
    _generate_explanation,
    _score_to_confidence,
    recommend_encoding,
)
from encoding_atlas.guide.rules import ENCODING_RULES
from tests.unit.guide.conftest import (
    ENCODING_TRIGGER_PARAMS,
)

# =========================================================================
# Recommendation dataclass
# =========================================================================


class TestRecommendationDataclass:
    """Verify the Recommendation result structure."""

    def test_has_all_fields(self) -> None:
        """Recommendation must expose all four documented fields."""
        rec = recommend_encoding(n_features=4)
        assert hasattr(rec, "encoding_name")
        assert hasattr(rec, "explanation")
        assert hasattr(rec, "alternatives")
        assert hasattr(rec, "confidence")

    def test_encoding_name_is_string(self) -> None:
        rec = recommend_encoding(n_features=4)
        assert isinstance(rec.encoding_name, str)

    def test_explanation_is_non_empty_string(self) -> None:
        rec = recommend_encoding(n_features=4)
        assert isinstance(rec.explanation, str)
        assert len(rec.explanation) > 0

    def test_alternatives_is_list_of_strings(self) -> None:
        rec = recommend_encoding(n_features=4)
        assert isinstance(rec.alternatives, list)
        for alt in rec.alternatives:
            assert isinstance(alt, str)

    def test_confidence_is_float(self) -> None:
        rec = recommend_encoding(n_features=4)
        assert isinstance(rec.confidence, float)


# =========================================================================
# Backward compatibility
# =========================================================================


class TestBackwardCompatibility:
    """Existing callers using only the original 5 positional args must work."""

    def test_positional_args(self) -> None:
        """Calling with 5 positional args must not raise."""
        rec = recommend_encoding(4, 500, "classification", "simulator", "accuracy")
        assert rec.encoding_name in ENCODING_RULES

    def test_speed_priority(self) -> None:
        rec = recommend_encoding(n_features=4, priority="speed")
        assert rec.encoding_name in ENCODING_RULES

    def test_noise_resilience_priority(self) -> None:
        rec = recommend_encoding(n_features=4, priority="noise_resilience")
        assert rec.encoding_name in ENCODING_RULES

    def test_small_features(self) -> None:
        rec = recommend_encoding(n_features=4, n_samples=200)
        assert rec.encoding_name in ENCODING_RULES

    def test_large_features(self) -> None:
        rec = recommend_encoding(n_features=16)
        assert rec.encoding_name in ENCODING_RULES

    def test_default_call(self) -> None:
        rec = recommend_encoding(n_features=6)
        assert rec.encoding_name in ENCODING_RULES


# =========================================================================
# Every encoding reachable
# =========================================================================


class TestEveryEncodingReachable:
    """Each of the 16 encodings must be the *primary* recommendation
    for at least one parameter combination."""

    @pytest.mark.parametrize(
        "encoding_name,params",
        sorted(ENCODING_TRIGGER_PARAMS.items()),
        ids=sorted(ENCODING_TRIGGER_PARAMS.keys()),
    )
    def test_encoding_is_primary(self, encoding_name: str, params: dict) -> None:
        rec = recommend_encoding(**params)
        assert rec.encoding_name == encoding_name, (
            f"Expected '{encoding_name}' but got '{rec.encoding_name}' "
            f"for params {params}"
        )


# =========================================================================
# Hard constraint enforcement
# =========================================================================


class TestHardConstraintEnforcement:
    """Encodings with hard constraints must *never* appear (as primary or
    alternative) when those constraints are violated."""

    def test_basis_never_for_continuous(self) -> None:
        """Basis encoding must not appear for data_type='continuous'."""
        for priority in ("accuracy", "speed", "trainability", "noise_resilience"):
            rec = recommend_encoding(
                n_features=4, priority=priority, data_type="continuous"
            )
            assert rec.encoding_name != "basis"
            assert "basis" not in rec.alternatives

    def test_so2_never_for_3_features(self) -> None:
        """SO2 equivariant must not appear when n_features != 2."""
        rec = recommend_encoding(n_features=3, symmetry="rotation")
        assert rec.encoding_name != "so2_equivariant"
        assert "so2_equivariant" not in rec.alternatives

    def test_so2_never_without_rotation_symmetry(self) -> None:
        """SO2 equivariant must not appear without symmetry='rotation'."""
        rec = recommend_encoding(n_features=2)
        assert rec.encoding_name != "so2_equivariant"
        assert "so2_equivariant" not in rec.alternatives

    def test_swap_never_for_odd_features(self) -> None:
        """Swap equivariant must not appear when n_features is odd."""
        rec = recommend_encoding(n_features=3, symmetry="permutation_pairs")
        assert rec.encoding_name != "swap_equivariant"
        assert "swap_equivariant" not in rec.alternatives

    def test_trainable_never_without_flag(self) -> None:
        """Trainable encoding must not appear when trainable=False."""
        for priority in ("accuracy", "trainability", "speed"):
            rec = recommend_encoding(n_features=4, priority=priority)
            assert rec.encoding_name != "trainable"
            assert "trainable" not in rec.alternatives

    def test_cyclic_never_without_cyclic_symmetry(self) -> None:
        """Cyclic equivariant must not appear without symmetry='cyclic'."""
        rec = recommend_encoding(n_features=4)
        assert rec.encoding_name != "cyclic_equivariant"
        assert "cyclic_equivariant" not in rec.alternatives

    def test_symmetry_inspired_never_without_general_symmetry(self) -> None:
        """Symmetry-inspired must not appear without symmetry='general'."""
        rec = recommend_encoding(n_features=4)
        assert rec.encoding_name != "symmetry_inspired"
        assert "symmetry_inspired" not in rec.alternatives

    @pytest.mark.parametrize(
        "name,max_val",
        [
            ("iqp", 12),
            ("zz_feature_map", 12),
            ("data_reuploading", 8),
            ("higher_order_angle", 10),
        ],
    )
    def test_max_features_respected(self, name: str, max_val: int) -> None:
        """Encoding must not appear when n_features exceeds its max."""
        rec = recommend_encoding(n_features=max_val + 5)
        assert rec.encoding_name != name
        assert name not in rec.alternatives


# =========================================================================
# Alternatives
# =========================================================================


class TestAlternatives:
    """Verify alternatives list properties."""

    def test_primary_not_in_alternatives(self) -> None:
        rec = recommend_encoding(n_features=4)
        assert rec.encoding_name not in rec.alternatives

    def test_max_three_alternatives(self) -> None:
        rec = recommend_encoding(n_features=4)
        assert len(rec.alternatives) <= 3

    @pytest.mark.parametrize(
        "params",
        [
            dict(n_features=4),
            dict(n_features=4, priority="speed"),
            dict(n_features=4, data_type="binary"),
            dict(n_features=2, symmetry="rotation"),
        ],
    )
    def test_alternatives_are_valid_encoding_names(self, params: dict) -> None:
        rec = recommend_encoding(**params)
        for alt in rec.alternatives:
            assert alt in ENCODING_RULES, f"'{alt}' is not a valid encoding"

    def test_alternatives_pass_hard_constraints(self) -> None:
        """Every alternative must satisfy the same hard constraints."""
        from encoding_atlas.guide.rules import _passes_hard_constraints

        test_cases = [
            dict(n_features=4, data_type="binary"),
            dict(n_features=2, symmetry="rotation"),
            dict(n_features=4, symmetry="cyclic"),
            dict(n_features=4, trainable=True),
        ]
        for params in test_cases:
            rec = recommend_encoding(**params)
            for alt in rec.alternatives:
                rules = ENCODING_RULES[alt]
                assert _passes_hard_constraints(
                    rules,
                    n_features=params.get("n_features"),
                    data_type=params.get("data_type", "continuous"),
                    symmetry=params.get("symmetry"),
                    trainable=params.get("trainable", False),
                ), f"Alternative '{alt}' violates hard constraints for {params}"


# =========================================================================
# Confidence values
# =========================================================================


class TestConfidenceValues:
    """Verify confidence scores are well-behaved."""

    @pytest.mark.parametrize(
        "encoding_name,params",
        sorted(ENCODING_TRIGGER_PARAMS.items()),
        ids=sorted(ENCODING_TRIGGER_PARAMS.keys()),
    )
    def test_confidence_in_valid_range(self, encoding_name: str, params: dict) -> None:
        """Confidence must be in [0, 1] for every trigger set."""
        rec = recommend_encoding(**params)
        assert 0.0 <= rec.confidence <= 1.0

    def test_hard_match_gives_high_confidence(self) -> None:
        """Binary data -> basis should yield confidence >= 0.75."""
        rec = recommend_encoding(n_features=4, data_type="binary")
        assert rec.confidence >= 0.75

    def test_symmetry_match_gives_high_confidence(self) -> None:
        """SO2 match should yield confidence >= 0.85."""
        rec = recommend_encoding(n_features=2, symmetry="rotation")
        assert rec.confidence >= 0.85

    def test_fallback_gives_lower_confidence(self) -> None:
        """No constraints -> lower-end confidence."""
        rec = recommend_encoding(n_features=6)
        assert rec.confidence <= 0.80


# =========================================================================
# New parameters — each value produces the expected encoding
# =========================================================================


class TestNewParameters:
    """Verify that each new parameter value steers the recommendation."""

    def test_data_type_binary(self) -> None:
        rec = recommend_encoding(n_features=4, data_type="binary")
        assert rec.encoding_name == "basis"

    def test_data_type_discrete(self) -> None:
        rec = recommend_encoding(n_features=4, data_type="discrete")
        assert rec.encoding_name == "basis"

    def test_symmetry_rotation_2d(self) -> None:
        rec = recommend_encoding(n_features=2, symmetry="rotation")
        assert rec.encoding_name == "so2_equivariant"

    def test_symmetry_cyclic(self) -> None:
        rec = recommend_encoding(n_features=4, symmetry="cyclic")
        assert rec.encoding_name == "cyclic_equivariant"

    def test_symmetry_permutation_pairs_even(self) -> None:
        rec = recommend_encoding(n_features=4, symmetry="permutation_pairs")
        assert rec.encoding_name == "swap_equivariant"

    def test_symmetry_general(self) -> None:
        rec = recommend_encoding(n_features=4, symmetry="general")
        assert rec.encoding_name == "symmetry_inspired"

    def test_trainable_flag(self) -> None:
        rec = recommend_encoding(n_features=4, trainable=True)
        assert rec.encoding_name == "trainable"

    def test_problem_structure_combinatorial(self) -> None:
        rec = recommend_encoding(n_features=4, problem_structure="combinatorial")
        assert rec.encoding_name == "qaoa"

    def test_problem_structure_physics(self) -> None:
        rec = recommend_encoding(n_features=4, problem_structure="physics_simulation")
        assert rec.encoding_name == "hamiltonian"

    def test_problem_structure_time_series(self) -> None:
        """time_series should recommend data_reuploading."""
        rec = recommend_encoding(n_features=4, problem_structure="time_series")
        assert rec.encoding_name == "data_reuploading"

    def test_feature_interactions_polynomial(self) -> None:
        rec = recommend_encoding(n_features=4, feature_interactions="polynomial")
        assert rec.encoding_name == "higher_order_angle"

    def test_feature_interactions_custom_pauli(self) -> None:
        rec = recommend_encoding(n_features=4, feature_interactions="custom_pauli")
        assert rec.encoding_name == "pauli_feature_map"


# =========================================================================
# Task parameter
# =========================================================================


class TestTaskParameter:
    """Verify that the ``task`` parameter influences scoring."""

    def test_classification_boosts_kernel_methods(self) -> None:
        """Classification should boost IQP (kernel_methods) over
        data_reuploading (universal_approximation) compared to
        regression."""
        kwargs = dict(n_features=4, priority="accuracy")
        score_cls = _compute_score(
            "iqp",
            ENCODING_RULES["iqp"],
            n_samples=500,
            hardware="simulator",
            data_type="continuous",
            symmetry=None,
            trainable=False,
            problem_structure=None,
            feature_interactions=None,
            task="classification",
            **kwargs,
        )
        score_reg = _compute_score(
            "iqp",
            ENCODING_RULES["iqp"],
            n_samples=500,
            hardware="simulator",
            data_type="continuous",
            symmetry=None,
            trainable=False,
            problem_structure=None,
            feature_interactions=None,
            task="regression",
            **kwargs,
        )
        assert score_cls > score_reg

    def test_regression_boosts_universal_approximation(self) -> None:
        """Regression should boost data_reuploading
        (universal_approximation) vs classification."""
        kwargs = dict(n_features=4, priority="accuracy")
        score_reg = _compute_score(
            "data_reuploading",
            ENCODING_RULES["data_reuploading"],
            n_samples=500,
            hardware="simulator",
            data_type="continuous",
            symmetry=None,
            trainable=False,
            problem_structure=None,
            feature_interactions=None,
            task="regression",
            **kwargs,
        )
        score_cls = _compute_score(
            "data_reuploading",
            ENCODING_RULES["data_reuploading"],
            n_samples=500,
            hardware="simulator",
            data_type="continuous",
            symmetry=None,
            trainable=False,
            problem_structure=None,
            feature_interactions=None,
            task="classification",
            **kwargs,
        )
        assert score_reg > score_cls

    def test_task_not_applied_with_problem_structure(self) -> None:
        """Task bonus should NOT apply when problem_structure is set."""
        kwargs = dict(
            n_features=4,
            priority="accuracy",
            n_samples=500,
            hardware="simulator",
            data_type="continuous",
            symmetry=None,
            trainable=False,
            feature_interactions=None,
        )
        score_cls = _compute_score(
            "iqp",
            ENCODING_RULES["iqp"],
            task="classification",
            problem_structure="combinatorial",
            **kwargs,
        )
        score_reg = _compute_score(
            "iqp",
            ENCODING_RULES["iqp"],
            task="regression",
            problem_structure="combinatorial",
            **kwargs,
        )
        assert score_cls == score_reg

    def test_task_not_applied_with_feature_interactions(self) -> None:
        """Task bonus should NOT apply when feature_interactions is set."""
        kwargs = dict(
            n_features=4,
            priority="accuracy",
            n_samples=500,
            hardware="simulator",
            data_type="continuous",
            symmetry=None,
            trainable=False,
            problem_structure=None,
        )
        score_cls = _compute_score(
            "iqp",
            ENCODING_RULES["iqp"],
            task="classification",
            feature_interactions="polynomial",
            **kwargs,
        )
        score_reg = _compute_score(
            "iqp",
            ENCODING_RULES["iqp"],
            task="regression",
            feature_interactions="polynomial",
            **kwargs,
        )
        assert score_cls == score_reg


# =========================================================================
# avoid_when penalty
# =========================================================================


class TestAvoidWhenPenalty:
    """Verify that ``avoid_when`` tags are penalised on real hardware."""

    def _default_kwargs(self, **overrides: object) -> dict:
        base = dict(
            n_features=4,
            n_samples=500,
            task="classification",
            priority="accuracy",
            data_type="continuous",
            symmetry=None,
            trainable=False,
            problem_structure=None,
            feature_interactions=None,
        )
        base.update(overrides)
        return base

    def test_iqp_penalised_on_real_hardware(self) -> None:
        """IQP has 'noisy_hardware' in avoid_when — should score lower
        on real hardware than on simulator."""
        iqp_sim = _compute_score(
            "iqp",
            ENCODING_RULES["iqp"],
            **self._default_kwargs(hardware="simulator"),
        )
        iqp_hw = _compute_score(
            "iqp",
            ENCODING_RULES["iqp"],
            **self._default_kwargs(hardware="ibm"),
        )
        assert iqp_sim > iqp_hw

    def test_zz_not_penalised_on_generic_hardware(self) -> None:
        """ZZ uses 'very_noisy_hardware' which does not trigger the
        automatic hardware penalty."""
        zz_sim = _compute_score(
            "zz_feature_map",
            ENCODING_RULES["zz_feature_map"],
            **self._default_kwargs(hardware="simulator"),
        )
        zz_hw = _compute_score(
            "zz_feature_map",
            ENCODING_RULES["zz_feature_map"],
            **self._default_kwargs(hardware="ibm"),
        )
        # ZZ should not get the avoid_when penalty (only "very_noisy_hardware"),
        # but also gets no NISQ bonus — so scores should be equal.
        assert zz_sim == zz_hw

    def test_hardware_efficient_not_penalised(self) -> None:
        """Hardware-efficient has no hardware-avoidance tags."""
        he_sim = _compute_score(
            "hardware_efficient",
            ENCODING_RULES["hardware_efficient"],
            **self._default_kwargs(hardware="simulator"),
        )
        he_hw = _compute_score(
            "hardware_efficient",
            ENCODING_RULES["hardware_efficient"],
            **self._default_kwargs(hardware="ibm"),
        )
        # hardware_efficient gets NISQ bonus on real hw, no penalty.
        assert he_hw > he_sim


# =========================================================================
# Input validation
# =========================================================================


class TestInputValidation:
    """Verify that invalid inputs raise ValueError."""

    def test_n_features_zero(self) -> None:
        with pytest.raises(ValueError, match="n_features"):
            recommend_encoding(n_features=0)

    def test_n_features_negative(self) -> None:
        with pytest.raises(ValueError, match="n_features"):
            recommend_encoding(n_features=-1)

    def test_n_samples_zero(self) -> None:
        with pytest.raises(ValueError, match="n_samples"):
            recommend_encoding(n_features=4, n_samples=0)

    def test_invalid_task(self) -> None:
        with pytest.raises(ValueError, match="task"):
            recommend_encoding(n_features=4, task="clustering")  # type: ignore[arg-type]

    def test_invalid_priority(self) -> None:
        with pytest.raises(ValueError, match="priority"):
            recommend_encoding(n_features=4, priority="invalid")  # type: ignore[arg-type]

    def test_invalid_data_type(self) -> None:
        with pytest.raises(ValueError, match="data_type"):
            recommend_encoding(n_features=4, data_type="float")  # type: ignore[arg-type]

    def test_invalid_symmetry(self) -> None:
        with pytest.raises(ValueError, match="symmetry"):
            recommend_encoding(n_features=4, symmetry="translational")  # type: ignore[arg-type]

    def test_invalid_problem_structure(self) -> None:
        with pytest.raises(ValueError, match="problem_structure"):
            recommend_encoding(n_features=4, problem_structure="unknown")  # type: ignore[arg-type]

    def test_invalid_feature_interactions(self) -> None:
        with pytest.raises(ValueError, match="feature_interactions"):
            recommend_encoding(n_features=4, feature_interactions="quadratic")  # type: ignore[arg-type]


# =========================================================================
# Edge cases
# =========================================================================


class TestEdgeCases:
    """Boundary conditions and unusual parameter combinations."""

    def test_n_features_1(self) -> None:
        """Single feature must not crash."""
        rec = recommend_encoding(n_features=1)
        assert rec.encoding_name in ENCODING_RULES

    def test_n_features_very_large(self) -> None:
        """100 features should favour amplitude encoding."""
        rec = recommend_encoding(n_features=100)
        assert rec.encoding_name == "amplitude"

    def test_n_features_2_no_symmetry(self) -> None:
        """2 features without symmetry should not pick SO2."""
        rec = recommend_encoding(n_features=2)
        assert rec.encoding_name != "so2_equivariant"

    def test_conflicting_binary_and_rotation(self) -> None:
        """data_type='binary' + symmetry='rotation' — basis should win
        because data_type is a hard constraint for basis, while SO2
        has no data_type constraint but requires exactly 2 features
        *and* rotation symmetry. The scoring should favour basis."""
        rec = recommend_encoding(n_features=2, data_type="binary", symmetry="rotation")
        assert rec.encoding_name in ENCODING_RULES
        # Importantly: the result must not violate any hard constraint
        rules = ENCODING_RULES[rec.encoding_name]
        if rules["requires_data_type"] is not None:
            assert "binary" in rules["requires_data_type"]

    def test_small_samples(self) -> None:
        """Very small dataset (n_samples=10)."""
        rec = recommend_encoding(n_features=4, n_samples=10)
        assert rec.encoding_name in ENCODING_RULES

    def test_hardware_ibm(self) -> None:
        """Real hardware target should not crash."""
        rec = recommend_encoding(n_features=4, hardware="ibm")
        assert rec.encoding_name in ENCODING_RULES

    def test_regression_task(self) -> None:
        """Regression task should produce a valid recommendation."""
        rec = recommend_encoding(n_features=4, task="regression")
        assert rec.encoding_name in ENCODING_RULES

    def test_all_defaults(self) -> None:
        """Calling with only n_features should work."""
        rec = recommend_encoding(n_features=4)
        assert rec.encoding_name in ENCODING_RULES
        assert len(rec.explanation) > 0
        assert 0.0 <= rec.confidence <= 1.0


# =========================================================================
# Scoring helpers unit tests
# =========================================================================


class TestComputeScore:
    """Unit tests for the _compute_score helper."""

    def _default_kwargs(self, **overrides: object) -> dict:
        """Build a default kwargs dict for _compute_score."""
        base = dict(
            n_features=4,
            n_samples=500,
            task="classification",
            hardware="simulator",
            priority="accuracy",
            data_type="continuous",
            symmetry=None,
            trainable=False,
            problem_structure=None,
            feature_interactions=None,
        )
        base.update(overrides)
        return base

    def test_score_clamped_to_unit_interval(self) -> None:
        """Score must be in [0, 1] regardless of inputs."""
        for name, rules in ENCODING_RULES.items():
            score = _compute_score(name, rules, **self._default_kwargs())
            assert 0.0 <= score <= 1.0, f"{name} score={score}"

    def test_hard_precondition_bonus(self) -> None:
        """Matching a hard precondition should yield a higher score."""
        basis_score = _compute_score(
            "basis",
            ENCODING_RULES["basis"],
            **self._default_kwargs(data_type="binary"),
        )
        angle_score = _compute_score(
            "angle",
            ENCODING_RULES["angle"],
            **self._default_kwargs(data_type="binary"),
        )
        assert basis_score > angle_score

    def test_priority_matching_increases_score(self) -> None:
        """Matching priority tags should increase the score."""
        angle_speed = _compute_score(
            "angle",
            ENCODING_RULES["angle"],
            **self._default_kwargs(priority="speed"),
        )
        angle_accuracy = _compute_score(
            "angle",
            ENCODING_RULES["angle"],
            **self._default_kwargs(priority="accuracy"),
        )
        assert angle_speed > angle_accuracy

    def test_hardware_penalty_for_deep_circuits(self) -> None:
        """Deep circuits should be penalised on real hardware."""
        amp_sim = _compute_score(
            "amplitude",
            ENCODING_RULES["amplitude"],
            **self._default_kwargs(n_features=16, hardware="simulator"),
        )
        amp_hw = _compute_score(
            "amplitude",
            ENCODING_RULES["amplitude"],
            **self._default_kwargs(n_features=16, hardware="ibm"),
        )
        assert amp_sim > amp_hw

    def test_symmetry_bonus(self) -> None:
        """Symmetry match should increase score."""
        with_sym = _compute_score(
            "cyclic_equivariant",
            ENCODING_RULES["cyclic_equivariant"],
            **self._default_kwargs(symmetry="cyclic"),
        )
        # Without symmetry match, cyclic_equivariant is filtered by hard
        # constraints. Instead, compare cyclic vs a general encoding at
        # the same settings.
        general_score = _compute_score(
            "angle",
            ENCODING_RULES["angle"],
            **self._default_kwargs(symmetry="cyclic"),
        )
        assert with_sym > general_score

    def test_trainable_bonus(self) -> None:
        """Trainable match should increase score."""
        trainable_score = _compute_score(
            "trainable",
            ENCODING_RULES["trainable"],
            **self._default_kwargs(trainable=True),
        )
        angle_score = _compute_score(
            "angle",
            ENCODING_RULES["angle"],
            **self._default_kwargs(trainable=True),
        )
        assert trainable_score > angle_score

    def test_binary_penalty(self) -> None:
        """Non-binary encodings should be penalised on binary data.
        Use priority='speed' so angle gets a non-zero base score;
        with priority='accuracy' both scores clamp to 0.0."""
        angle_continuous = _compute_score(
            "angle",
            ENCODING_RULES["angle"],
            **self._default_kwargs(data_type="continuous", priority="speed"),
        )
        angle_binary = _compute_score(
            "angle",
            ENCODING_RULES["angle"],
            **self._default_kwargs(data_type="binary", priority="speed"),
        )
        assert angle_continuous > angle_binary

    def test_structure_bonus(self) -> None:
        """Problem structure should boost matching encodings."""
        qaoa_struct = _compute_score(
            "qaoa",
            ENCODING_RULES["qaoa"],
            **self._default_kwargs(problem_structure="combinatorial"),
        )
        qaoa_no_struct = _compute_score(
            "qaoa",
            ENCODING_RULES["qaoa"],
            **self._default_kwargs(problem_structure=None),
        )
        assert qaoa_struct > qaoa_no_struct

    def test_interaction_bonus(self) -> None:
        """Feature interaction should boost matching encodings."""
        pauli_int = _compute_score(
            "pauli_feature_map",
            ENCODING_RULES["pauli_feature_map"],
            **self._default_kwargs(feature_interactions="custom_pauli"),
        )
        pauli_no_int = _compute_score(
            "pauli_feature_map",
            ENCODING_RULES["pauli_feature_map"],
            **self._default_kwargs(feature_interactions=None),
        )
        assert pauli_int > pauli_no_int

    def test_logarithmic_bonus(self) -> None:
        """Logarithmic scaling should boost amplitude at high n_features."""
        amp_large = _compute_score(
            "amplitude",
            ENCODING_RULES["amplitude"],
            **self._default_kwargs(n_features=16),
        )
        amp_small = _compute_score(
            "amplitude",
            ENCODING_RULES["amplitude"],
            **self._default_kwargs(n_features=4),
        )
        assert amp_large > amp_small

    def test_small_sample_bonus(self) -> None:
        """Small datasets should boost simulable encodings."""
        angle_small = _compute_score(
            "angle",
            ENCODING_RULES["angle"],
            **self._default_kwargs(n_samples=50),
        )
        angle_large = _compute_score(
            "angle",
            ENCODING_RULES["angle"],
            **self._default_kwargs(n_samples=500),
        )
        assert angle_small > angle_large


class TestScoreToConfidence:
    """Unit tests for _score_to_confidence."""

    def test_zero_score(self) -> None:
        assert _score_to_confidence(0.0) == 0.50

    def test_mid_score(self) -> None:
        conf = _score_to_confidence(0.40)
        assert 0.65 <= conf <= 0.80

    def test_high_score(self) -> None:
        conf = _score_to_confidence(0.70)
        assert 0.85 <= conf <= 0.95

    def test_max_score(self) -> None:
        conf = _score_to_confidence(1.0)
        assert conf <= 0.95

    def test_monotonic(self) -> None:
        """Higher score must produce higher or equal confidence."""
        scores = [i / 20 for i in range(21)]
        confidences = [_score_to_confidence(s) for s in scores]
        for i in range(len(confidences) - 1):
            assert confidences[i] <= confidences[i + 1]

    def test_continuous_at_lower_boundary(self) -> None:
        """Confidence must be continuous at score=0.30."""
        below = _score_to_confidence(0.30 - 1e-9)
        at = _score_to_confidence(0.30)
        assert abs(below - at) < 0.001

    def test_continuous_at_upper_boundary(self) -> None:
        """Confidence must be continuous at score=0.50."""
        below = _score_to_confidence(0.50 - 1e-9)
        at = _score_to_confidence(0.50)
        assert abs(below - at) < 0.001


class TestGenerateExplanation:
    """Unit tests for _generate_explanation."""

    @pytest.mark.parametrize("name", sorted(ENCODING_RULES.keys()))
    def test_explanation_is_non_empty(self, name: str) -> None:
        result = _generate_explanation(
            name,
            ENCODING_RULES[name],
            priority="accuracy",
            n_features=4,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_amplitude_includes_qubit_count(self) -> None:
        result = _generate_explanation(
            "amplitude",
            ENCODING_RULES["amplitude"],
            priority="accuracy",
            n_features=16,
        )
        assert "4 qubits" in result
        assert "16 features" in result
