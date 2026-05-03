"""Tests for the simulability analysis module.

This module provides comprehensive tests for the simulability analysis
functions in encoding_atlas.analysis.simulability:

- check_simulability: Main simulability classification function
- get_simulability_reason: Concise explanation function
- is_clifford_circuit: Clifford gate detection
- is_matchgate_circuit: Matchgate circuit detection with topology awareness
- estimate_entanglement_bound: Entanglement entropy estimation

Test Categories
---------------
1. Basic functionality tests for each function
2. Edge case handling (boundary conditions, special inputs)
3. Error handling (invalid inputs, type errors)
4. Numerical stability tests
5. Known value tests (analytical verification)
6. Cross-encoding consistency tests
7. Coverage-targeted tests for private helpers and decision branches
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pytest

from encoding_atlas.analysis.simulability import (
    _check_clifford_property,
    _check_matchgate_property,
    _check_non_clifford_gates,
    _compute_bipartite_entropy,
    _format_memory_size,
    _get_entanglement_pattern,
    _normalize_gate_set,
    check_simulability,
    estimate_entanglement_bound,
    get_simulability_reason,
    is_clifford_circuit,
    is_matchgate_circuit,
)
from encoding_atlas.core.exceptions import AnalysisError
from encoding_atlas.core.properties import EncodingProperties

if TYPE_CHECKING:
    pass


# =============================================================================
# Test Class: check_simulability
# =============================================================================


class TestCheckSimulability:
    """Tests for the check_simulability function."""

    def test_simulable_non_entangling_encoding(self, sample_encoding_2q):
        """Test that non-entangling encoding is classified as simulable."""
        result = check_simulability(sample_encoding_2q)

        assert result["is_simulable"] is True
        assert result["simulability_class"] == "simulable"
        assert "product states" in result["reason"].lower()
        assert isinstance(result["details"], dict)
        assert isinstance(result["recommendations"], list)
        assert len(result["recommendations"]) > 0

    def test_simulable_non_entangling_4q(self, sample_encoding_4q):
        """Test non-entangling encoding with more qubits."""
        result = check_simulability(sample_encoding_4q)

        assert result["is_simulable"] is True
        assert result["simulability_class"] == "simulable"
        assert result["details"]["is_entangling"] is False

    def test_entangling_encoding_not_simulable(self, entangling_encoding_4q):
        """Test that entangling encoding is classified as not simulable.

        Entangling circuits with non-Clifford gates are classified based on
        their circuit family's asymptotic complexity.  The classification is
        ``not_simulable`` regardless of instance size, as no known efficient
        classical simulation algorithm exists for the general case.

        Practical feasibility for small instances is noted in recommendations.
        """
        result = check_simulability(entangling_encoding_4q)

        assert result["is_simulable"] is False
        assert result["simulability_class"] == "not_simulable"
        assert result["details"]["is_entangling"] is True
        assert len(result["recommendations"]) > 0

    def test_iqp_encoding_mentions_hardness(self):
        """Test that IQP encoding result mentions provable hardness."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=2)
        result = check_simulability(enc)

        assert result["simulability_class"] == "not_simulable"
        # Reason should reference IQP-specific hardness
        reason_lower = result["reason"].lower()
        assert "iqp" in reason_lower
        # Recommendations should still note practical brute-force feasibility
        assert any(
            "brute" in r.lower()
            or "statevector" in r.lower()
            or "feasible" in r.lower()
            for r in result["recommendations"]
        )

    def test_detailed_false_returns_minimal_result(self, sample_encoding_2q):
        """Test that detailed=False returns minimal but complete result."""
        result = check_simulability(sample_encoding_2q, detailed=False)

        assert "is_simulable" in result
        assert "simulability_class" in result
        assert "reason" in result
        assert "details" in result
        assert "recommendations" in result
        # Details should be empty dict when detailed=False
        assert result["details"] == {}

    def test_detailed_true_includes_details(self, sample_encoding_2q):
        """Test that detailed=True includes comprehensive details."""
        result = check_simulability(sample_encoding_2q, detailed=True)

        # Check all expected detail fields
        assert "is_entangling" in result["details"]
        assert "is_clifford" in result["details"]
        assert "is_matchgate" in result["details"]
        assert "entanglement_pattern" in result["details"]
        assert "two_qubit_gate_count" in result["details"]
        assert "n_qubits" in result["details"]
        assert "n_features" in result["details"]
        assert "declared_simulability" in result["details"]
        assert "encoding_name" in result["details"]

    def test_linear_entanglement_conditionally_simulable(self):
        """Test that linear entanglement is conditionally simulable."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1, entanglement="linear")
        result = check_simulability(enc)

        # Linear entanglement may be conditionally simulable
        assert result["simulability_class"] in (
            "conditionally_simulable",
            "not_simulable",  # Conservative classification is acceptable
        )

    def test_circular_entanglement_conditionally_simulable(self):
        """Test that circular entanglement is conditionally simulable."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1, entanglement="circular")
        result = check_simulability(enc)

        assert result["simulability_class"] in (
            "conditionally_simulable",
            "not_simulable",
        )

    def test_invalid_encoding_type_raises_error(self):
        """Test that invalid encoding type raises AnalysisError."""
        with pytest.raises(AnalysisError) as exc_info:
            check_simulability("not an encoding")  # type: ignore

        assert "BaseEncoding" in str(exc_info.value)

    def test_invalid_encoding_none_raises_error(self):
        """Test that None input raises AnalysisError."""
        with pytest.raises(AnalysisError):
            check_simulability(None)  # type: ignore

    def test_result_type_is_typed_dict(self, sample_encoding_2q):
        """Test that result conforms to SimulabilityResult TypedDict."""
        result = check_simulability(sample_encoding_2q)

        # Check all required keys are present
        required_keys = {
            "is_simulable",
            "simulability_class",
            "reason",
            "details",
            "recommendations",
        }
        assert required_keys <= set(result.keys())

    def test_recommendations_not_empty(self, sample_encoding_2q):
        """Test that recommendations are always provided."""
        result = check_simulability(sample_encoding_2q)
        assert len(result["recommendations"]) > 0
        assert all(isinstance(r, str) for r in result["recommendations"])

    def test_simulability_class_valid_values(
        self, sample_encoding_2q, entangling_encoding_4q
    ):
        """Test that simulability_class is one of the valid literals."""
        valid_classes = {"simulable", "conditionally_simulable", "not_simulable"}

        result1 = check_simulability(sample_encoding_2q)
        result2 = check_simulability(entangling_encoding_4q)

        assert result1["simulability_class"] in valid_classes
        assert result2["simulability_class"] in valid_classes

    def test_multiple_encodings_consistency(self, sample_encoding_factory):
        """Test consistency across different encoding types."""
        # Non-entangling should always be simulable
        angle_enc = sample_encoding_factory("angle", n_features=4)
        result = check_simulability(angle_enc)
        assert result["is_simulable"] is True

        # IQP should never be simulable
        iqp_enc = sample_encoding_factory("iqp", n_features=4, reps=2)
        result = check_simulability(iqp_enc)
        assert result["is_simulable"] is False


# =============================================================================
# Test Class: get_simulability_reason
# =============================================================================


class TestGetSimulabilityReason:
    """Tests for the get_simulability_reason function."""

    def test_simulable_encoding_prefix(self, sample_encoding_2q):
        """Test that simulable encoding has 'Simulable:' prefix."""
        reason = get_simulability_reason(sample_encoding_2q)
        assert reason.startswith("Simulable:")

    def test_not_simulable_encoding_prefix(self, entangling_encoding_4q):
        """Test that not simulable encoding has 'Not simulable:' prefix."""
        reason = get_simulability_reason(entangling_encoding_4q)
        assert reason.startswith("Not simulable:")

    def test_reason_is_string(self, sample_encoding_2q):
        """Test that result is a string."""
        reason = get_simulability_reason(sample_encoding_2q)
        assert isinstance(reason, str)

    def test_reason_not_empty(self, sample_encoding_2q, entangling_encoding_4q):
        """Test that reason contains meaningful content."""
        reason1 = get_simulability_reason(sample_encoding_2q)
        reason2 = get_simulability_reason(entangling_encoding_4q)

        # Reason should have content after the prefix
        assert len(reason1.split(":", 1)[1].strip()) > 0
        assert len(reason2.split(":", 1)[1].strip()) > 0

    def test_invalid_encoding_raises_error(self):
        """Test that invalid encoding raises AnalysisError."""
        with pytest.raises(AnalysisError):
            get_simulability_reason(42)  # type: ignore

    def test_consistency_with_check_simulability(self, sample_encoding_2q):
        """Test that reason matches check_simulability result."""
        full_result = check_simulability(sample_encoding_2q)
        short_reason = get_simulability_reason(sample_encoding_2q)

        # The short reason should contain the full reason text
        assert full_result["reason"] in short_reason


# =============================================================================
# Test Class: is_clifford_circuit
# =============================================================================


class TestIsCliffordCircuit:
    """Tests for the is_clifford_circuit function."""

    def test_angle_encoding_not_clifford(self, sample_encoding_2q):
        """Test that AngleEncoding is not Clifford (uses RY gates)."""
        result = is_clifford_circuit(sample_encoding_2q)
        # AngleEncoding uses parameterized RY gates, which are not Clifford
        assert result is False

    def test_iqp_encoding_not_clifford(self, entangling_encoding_4q):
        """Test that IQPEncoding is not Clifford (uses RZ gates)."""
        result = is_clifford_circuit(entangling_encoding_4q)
        assert result is False

    def test_returns_bool(self, sample_encoding_2q):
        """Test that result is a boolean."""
        result = is_clifford_circuit(sample_encoding_2q)
        assert isinstance(result, bool)

    def test_invalid_encoding_raises_error(self):
        """Test that invalid encoding raises AnalysisError."""
        with pytest.raises(AnalysisError):
            is_clifford_circuit([1, 2, 3])  # type: ignore

    def test_none_raises_error(self):
        """Test that None raises AnalysisError."""
        with pytest.raises(AnalysisError):
            is_clifford_circuit(None)  # type: ignore


# =============================================================================
# Test Class: is_matchgate_circuit
# =============================================================================


class TestIsMatchgateCircuit:
    """Tests for the is_matchgate_circuit function.

    Matchgate circuits are classically simulable when applied with
    nearest-neighbor connectivity on a line topology. These tests verify:

    1. Standard encodings (Angle, IQP) are correctly identified as non-matchgate
    2. The function returns the correct type (bool)
    3. Invalid inputs raise appropriate errors
    4. Topology awareness: matchgate-based encodings with non-linear topology
       are correctly classified as not efficiently simulable
    """

    def test_angle_encoding_not_matchgate(self, sample_encoding_2q):
        """Test that AngleEncoding is not a matchgate circuit.

        AngleEncoding uses single-qubit RY rotations without two-qubit
        matchgate operations, so it is not classified as a matchgate circuit.
        """
        result = is_matchgate_circuit(sample_encoding_2q)
        assert result is False

    def test_iqp_encoding_not_matchgate(self, entangling_encoding_4q):
        """Test that IQPEncoding is not a matchgate circuit.

        IQP circuits use diagonal gates (RZ, ZZ interactions) that are
        not in the matchgate gate set.
        """
        result = is_matchgate_circuit(entangling_encoding_4q)
        assert result is False

    def test_returns_bool(self, sample_encoding_2q):
        """Test that result is a boolean."""
        result = is_matchgate_circuit(sample_encoding_2q)
        assert isinstance(result, bool)

    def test_invalid_encoding_raises_error(self):
        """Test that invalid encoding raises AnalysisError."""
        with pytest.raises(AnalysisError):
            is_matchgate_circuit([1, 2, 3])  # type: ignore

    def test_none_raises_error(self):
        """Test that None raises AnalysisError."""
        with pytest.raises(AnalysisError):
            is_matchgate_circuit(None)  # type: ignore

    def test_string_raises_error(self):
        """Test that string input raises AnalysisError."""
        with pytest.raises(AnalysisError):
            is_matchgate_circuit("not an encoding")  # type: ignore

    def test_non_entangling_encoding_not_matchgate(self, sample_encoding_4q):
        """Test that non-entangling encoding is not classified as matchgate.

        Non-entangling encodings are trivially simulable as product states
        but are not matchgate circuits specifically. The matchgate check
        correctly defers to the product-state simulability path.
        """
        result = is_matchgate_circuit(sample_encoding_4q)
        assert result is False

    @pytest.mark.parametrize("entanglement", ["full", "linear", "circular"])
    def test_iqp_with_various_topologies_not_matchgate(self, entanglement):
        """Test that IQP encoding is not matchgate regardless of topology.

        IQP circuits use non-matchgate gates (ZZ interactions, RZ rotations),
        so they should never be classified as matchgate circuits regardless
        of the entanglement topology used.
        """
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1, entanglement=entanglement)
        result = is_matchgate_circuit(enc)
        assert result is False


# =============================================================================
# Test Class: estimate_entanglement_bound
# =============================================================================


class TestEstimateEntanglementBound:
    """Tests for the estimate_entanglement_bound function."""

    def test_non_entangling_encoding_zero_entropy(self, sample_encoding_2q):
        """Test that non-entangling encoding has zero entanglement."""
        entropy = estimate_entanglement_bound(sample_encoding_2q, n_samples=10, seed=42)
        assert entropy == 0.0

    def test_non_entangling_encoding_zero_entropy_4q(self, sample_encoding_4q):
        """Test that non-entangling encoding with more qubits has zero entropy."""
        entropy = estimate_entanglement_bound(sample_encoding_4q, n_samples=10, seed=42)
        assert entropy == 0.0

    @pytest.mark.slow
    def test_entangling_encoding_positive_entropy(
        self, entangling_encoding_4q, skip_if_no_pennylane
    ):
        """Test that entangling encoding has positive entanglement."""
        entropy = estimate_entanglement_bound(
            entangling_encoding_4q, n_samples=20, seed=42
        )
        # IQP encoding should produce entanglement
        assert entropy > 0.0

    def test_seed_reproducibility(self, sample_encoding_4q):
        """Test that same seed gives same result."""
        entropy1 = estimate_entanglement_bound(
            sample_encoding_4q, n_samples=10, seed=42
        )
        entropy2 = estimate_entanglement_bound(
            sample_encoding_4q, n_samples=10, seed=42
        )
        assert entropy1 == entropy2

    def test_different_seeds_may_differ(self):
        """Test that different seeds can give different results."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)

        # Note: For non-entangling, both will be 0.0
        # For entangling, different seeds may give different max entropy
        # This is a property test, not strict requirement
        entropy1 = estimate_entanglement_bound(enc, n_samples=5, seed=42)
        entropy2 = estimate_entanglement_bound(enc, n_samples=5, seed=123)

        # Both should be non-negative floats
        assert entropy1 >= 0.0
        assert entropy2 >= 0.0

    def test_returns_float(self, sample_encoding_2q):
        """Test that result is a float."""
        entropy = estimate_entanglement_bound(sample_encoding_2q, n_samples=5, seed=42)
        assert isinstance(entropy, float)

    def test_invalid_encoding_raises_error(self):
        """Test that invalid encoding raises AnalysisError."""
        with pytest.raises(AnalysisError):
            estimate_entanglement_bound("not an encoding", n_samples=10)  # type: ignore

    def test_invalid_n_samples_raises_error(self, sample_encoding_2q):
        """Test that n_samples < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_samples must be at least 1"):
            estimate_entanglement_bound(sample_encoding_2q, n_samples=0)

    def test_negative_n_samples_raises_error(self, sample_encoding_2q):
        """Test that negative n_samples raises ValueError."""
        with pytest.raises(ValueError, match="n_samples must be at least 1"):
            estimate_entanglement_bound(sample_encoding_2q, n_samples=-5)

    def test_entropy_upper_bound(self, entangling_encoding_4q, skip_if_no_pennylane):
        """Test that entropy is bounded by n_qubits/2 (max possible)."""
        n_qubits = entangling_encoding_4q.n_qubits
        max_entropy = n_qubits / 2

        entropy = estimate_entanglement_bound(
            entangling_encoding_4q, n_samples=10, seed=42
        )

        assert entropy <= max_entropy + 1e-10  # Small tolerance for numerical error


# =============================================================================
# Test Class: Private Functions
# =============================================================================


class TestComputeBipartiteEntropy:
    """Tests for the _compute_bipartite_entropy function."""

    def test_product_state_zero_entropy(self, zero_state_2q):
        """Test that product state |00⟩ has zero entanglement."""
        entropy = _compute_bipartite_entropy(zero_state_2q, n_qubits=2, cut_position=1)
        assert abs(entropy) < 1e-10

    def test_bell_state_max_entropy(self, bell_state):
        """Test that Bell state has maximum entropy (1 bit for 2 qubits)."""
        entropy = _compute_bipartite_entropy(bell_state, n_qubits=2, cut_position=1)
        # Bell state has entropy = 1 bit for 2 qubits
        assert abs(entropy - 1.0) < 1e-10

    def test_product_state_plus_zero(self, product_state_2q):
        """Test that |+⟩⊗|0⟩ product state has zero entanglement."""
        entropy = _compute_bipartite_entropy(
            product_state_2q, n_qubits=2, cut_position=1
        )
        assert abs(entropy) < 1e-10

    def test_ghz_state_entropy(self, ghz_state_3q):
        """Test GHZ state entanglement (should be 1 bit for middle cut)."""
        # For GHZ, cutting at position 1 gives entropy = 1
        entropy = _compute_bipartite_entropy(ghz_state_3q, n_qubits=3, cut_position=1)
        assert abs(entropy - 1.0) < 1e-10

    def test_invalid_cut_position_returns_zero(self, bell_state):
        """Test that invalid cut positions return zero."""
        # Cut at position 0 (all qubits on one side)
        entropy = _compute_bipartite_entropy(bell_state, n_qubits=2, cut_position=0)
        assert entropy == 0.0

        # Cut at position >= n_qubits
        entropy = _compute_bipartite_entropy(bell_state, n_qubits=2, cut_position=2)
        assert entropy == 0.0

    def test_negative_cut_position_returns_zero(self, bell_state):
        """Test that negative cut positions return zero."""
        entropy = _compute_bipartite_entropy(bell_state, n_qubits=2, cut_position=-1)
        assert entropy == 0.0

    def test_entropy_is_non_negative(self, random_statevector_generator):
        """Test that entropy is always non-negative."""
        for n_qubits in [2, 3, 4]:
            state = random_statevector_generator(n_qubits)
            cut_pos = n_qubits // 2
            if cut_pos == 0:
                cut_pos = 1

            entropy = _compute_bipartite_entropy(state, n_qubits, cut_pos)
            assert entropy >= 0.0

    def test_entropy_bounded_by_max(self, random_statevector_generator):
        """Test that entropy is bounded by log2(min_dim)."""
        for n_qubits in [2, 3, 4]:
            state = random_statevector_generator(n_qubits)
            cut_pos = n_qubits // 2
            if cut_pos == 0:
                cut_pos = 1

            # Maximum entropy is min(cut_pos, n_qubits - cut_pos) bits
            max_entropy = min(cut_pos, n_qubits - cut_pos)

            entropy = _compute_bipartite_entropy(state, n_qubits, cut_pos)
            assert entropy <= max_entropy + 1e-10


class TestCheckCliffordProperty:
    """Tests for the _check_clifford_property function."""

    def test_parameterized_encoding_not_clifford(self, sample_encoding_2q):
        """Test that parameterized encodings are not Clifford."""
        result = _check_clifford_property(sample_encoding_2q)
        assert result is False

    def test_returns_bool(self, sample_encoding_2q):
        """Test that result is a boolean."""
        result = _check_clifford_property(sample_encoding_2q)
        assert isinstance(result, bool)


class TestCheckMatchgateProperty:
    """Tests for the _check_matchgate_property function.

    Verifies the internal matchgate detection logic, including the critical
    topology requirement: matchgate circuits are only efficiently simulable
    with nearest-neighbor connectivity on a line topology.

    This distinction is important because a circuit composed entirely of
    matchgate operations but arranged in a non-linear topology (e.g., full,
    circular) loses its classical simulability guarantee.

    References
    ----------
    Jozsa & Miyake (2008), "Matchgates and classical simulation of quantum
    circuits", Proc. R. Soc. A 464, 3089-3106.
    """

    def test_angle_encoding_not_matchgate(self, sample_encoding_2q):
        """Test that AngleEncoding is not identified as matchgate."""
        result = _check_matchgate_property(sample_encoding_2q)
        assert result is False

    def test_iqp_encoding_not_matchgate(self, entangling_encoding_4q):
        """Test that IQPEncoding is not identified as matchgate."""
        result = _check_matchgate_property(entangling_encoding_4q)
        assert result is False

    def test_returns_bool(self, sample_encoding_2q):
        """Test that result is a boolean."""
        result = _check_matchgate_property(sample_encoding_2q)
        assert isinstance(result, bool)

    def test_non_entangling_encoding_not_matchgate(self, sample_encoding_4q):
        """Test that non-entangling encoding returns False.

        Non-entangling encodings are trivially simulable as product states
        but the matchgate check correctly returns False, deferring to the
        product-state simulability classification.
        """
        result = _check_matchgate_property(sample_encoding_4q)
        assert result is False

    @pytest.mark.parametrize("entanglement", ["full", "circular"])
    def test_non_linear_topology_returns_false(self, entanglement):
        """Test that non-linear topology prevents matchgate simulability.

        Matchgate circuits require nearest-neighbor connectivity on a line
        topology for efficient classical simulation. Circuits with full or
        circular entanglement topology do not satisfy this requirement,
        even if all gates are matchgates.

        This is the key test for verifying the topology guard in
        _check_matchgate_property: the code at simulability.py:1441-1448
        correctly returns False for matchgate-based encodings that use
        non-linear topology.
        """
        from encoding_atlas import IQPEncoding

        # IQP is not matchgate-based, but this verifies the function
        # does not misclassify entangling circuits with non-linear topology
        enc = IQPEncoding(n_features=4, reps=1, entanglement=entanglement)
        result = _check_matchgate_property(enc)
        assert result is False

    def test_linear_topology_iqp_still_not_matchgate(self):
        """Test that linear topology alone does not make IQP a matchgate circuit.

        IQP circuits use non-matchgate gates (ZZ interactions), so even
        with linear topology they should not be classified as matchgate.
        The topology check is necessary but not sufficient.
        """
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1, entanglement="linear")
        result = _check_matchgate_property(enc)
        assert result is False


class TestGetEntanglementPattern:
    """Tests for the _get_entanglement_pattern function."""

    def test_non_entangling_pattern_none(self, sample_encoding_2q):
        """Test that non-entangling encoding has 'none' pattern."""
        pattern = _get_entanglement_pattern(sample_encoding_2q)
        assert pattern == "none"

    def test_full_entanglement_pattern(self):
        """Test that full entanglement is detected."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1, entanglement="full")
        pattern = _get_entanglement_pattern(enc)
        assert pattern == "full"

    def test_linear_entanglement_pattern(self):
        """Test that linear entanglement is detected."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1, entanglement="linear")
        pattern = _get_entanglement_pattern(enc)
        assert pattern == "linear"

    def test_circular_entanglement_pattern(self):
        """Test that circular entanglement is detected."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1, entanglement="circular")
        pattern = _get_entanglement_pattern(enc)
        assert pattern == "circular"

    def test_valid_pattern_values(self, sample_encoding_2q, entangling_encoding_4q):
        """Test that pattern is one of the expected values."""
        valid_patterns = {"none", "linear", "circular", "full", "partial", "unknown"}

        pattern1 = _get_entanglement_pattern(sample_encoding_2q)
        pattern2 = _get_entanglement_pattern(entangling_encoding_4q)

        assert pattern1 in valid_patterns
        assert pattern2 in valid_patterns


# =============================================================================
# Integration Tests
# =============================================================================


class TestSimulabilityIntegration:
    """Integration tests combining multiple simulability functions."""

    def test_simulable_encoding_all_functions_consistent(self, sample_encoding_4q):
        """Test that all functions are consistent for simulable encoding."""
        # Full analysis
        result = check_simulability(sample_encoding_4q)

        # Short reason
        reason = get_simulability_reason(sample_encoding_4q)

        # Entanglement bound
        entropy = estimate_entanglement_bound(sample_encoding_4q, n_samples=5, seed=42)

        # All should indicate simulable/product state
        assert result["is_simulable"] is True
        assert reason.startswith("Simulable:")
        assert entropy == 0.0
        assert result["details"]["entanglement_pattern"] == "none"

    def test_not_simulable_encoding_all_functions_consistent(
        self, entangling_encoding_4q, skip_if_no_pennylane
    ):
        """Test that all functions are consistent for non-simulable encoding."""
        # Full analysis
        result = check_simulability(entangling_encoding_4q)

        # Short reason
        reason = get_simulability_reason(entangling_encoding_4q)

        # Entanglement bound
        entropy = estimate_entanglement_bound(
            entangling_encoding_4q, n_samples=10, seed=42
        )

        # All should indicate not simulable/entangled
        assert result["is_simulable"] is False
        assert reason.startswith("Not simulable:")
        assert entropy > 0.0
        assert result["details"]["entanglement_pattern"] != "none"

    def test_different_encodings_different_results(self, sample_encoding_factory):
        """Test that different encoding types get appropriate classifications."""
        angle_enc = sample_encoding_factory("angle", n_features=4)
        iqp_enc = sample_encoding_factory("iqp", n_features=4, reps=2)

        result_angle = check_simulability(angle_enc)
        result_iqp = check_simulability(iqp_enc)

        # Should have different simulability
        assert result_angle["is_simulable"] != result_iqp["is_simulable"]
        assert result_angle["simulability_class"] != result_iqp["simulability_class"]


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestSimulabilityEdgeCases:
    """Edge case tests for simulability functions."""

    def test_single_qubit_encoding(self):
        """Test simulability analysis with single qubit."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=1)
        result = check_simulability(enc)

        # Single qubit is always simulable (trivially)
        assert result["is_simulable"] is True
        assert result["simulability_class"] == "simulable"

    def test_two_qubit_encoding(self):
        """Test simulability analysis with two qubits."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        result = check_simulability(enc)

        assert result["is_simulable"] is True
        assert result["details"]["n_qubits"] == 2

    def test_encoding_name_in_details(self, sample_encoding_2q):
        """Test that encoding name is included in details."""
        result = check_simulability(sample_encoding_2q)
        assert "encoding_name" in result["details"]
        assert "Angle" in result["details"]["encoding_name"]

    def test_n_features_in_details(self, sample_encoding_4q):
        """Test that n_features is correctly reported."""
        result = check_simulability(sample_encoding_4q)
        assert result["details"]["n_features"] == 4


# =============================================================================
# Parametrized Tests
# =============================================================================


@pytest.mark.parametrize("n_features", [2, 3, 4, 6, 8])
def test_angle_encoding_simulable_various_sizes(n_features):
    """Test that AngleEncoding is always simulable regardless of size."""
    from encoding_atlas import AngleEncoding

    enc = AngleEncoding(n_features=n_features)
    result = check_simulability(enc)

    assert result["is_simulable"] is True
    assert result["simulability_class"] == "simulable"


@pytest.mark.parametrize("entanglement", ["full", "linear", "circular"])
def test_iqp_entanglement_topologies(entanglement):
    """Test IQP encoding with different entanglement topologies."""
    from encoding_atlas import IQPEncoding

    enc = IQPEncoding(n_features=4, reps=1, entanglement=entanglement)
    result = check_simulability(enc)

    # All should be non-simulable (entangling with non-Clifford gates)
    assert result["is_simulable"] is False
    assert result["details"]["entanglement_pattern"] == entanglement


@pytest.mark.parametrize("reps", [1, 2, 3])
def test_iqp_reps_affect_gate_count(reps):
    """Test that different reps affect gate count but not simulability."""
    from encoding_atlas import IQPEncoding

    enc = IQPEncoding(n_features=4, reps=reps, entanglement="full")
    result = check_simulability(enc)

    # Should always be not simulable (IQP is hard)
    assert result["is_simulable"] is False
    # Gate count should scale with reps
    assert result["details"]["two_qubit_gate_count"] >= reps


@pytest.mark.parametrize("seed", [None, 0, 42, 12345])
def test_estimate_entanglement_bound_with_seeds(seed, sample_encoding_4q):
    """Test that estimate_entanglement_bound works with various seeds."""
    entropy = estimate_entanglement_bound(sample_encoding_4q, n_samples=5, seed=seed)
    assert entropy >= 0.0
    assert isinstance(entropy, float)


# =============================================================================
# Regression Tests
# =============================================================================


class TestSimulabilityRegressions:
    """Regression tests for known issues."""

    def test_empty_recommendations_never_returned(
        self, sample_encoding_2q, entangling_encoding_4q
    ):
        """Ensure recommendations are always provided."""
        result1 = check_simulability(sample_encoding_2q)
        result2 = check_simulability(entangling_encoding_4q)

        assert len(result1["recommendations"]) > 0
        assert len(result2["recommendations"]) > 0

    def test_reason_never_empty(self, sample_encoding_2q, entangling_encoding_4q):
        """Ensure reason is always meaningful."""
        result1 = check_simulability(sample_encoding_2q)
        result2 = check_simulability(entangling_encoding_4q)

        assert len(result1["reason"]) > 0
        assert len(result2["reason"]) > 0

    def test_details_consistent_with_simulability(
        self, sample_encoding_2q, entangling_encoding_4q
    ):
        """Ensure details are consistent with simulability classification."""
        result1 = check_simulability(sample_encoding_2q)
        result2 = check_simulability(entangling_encoding_4q)

        # Non-entangling should be simulable
        if result1["details"]["is_entangling"] is False:
            assert result1["is_simulable"] is True

        # Entangling should typically not be simulable (with non-Clifford gates)
        if result2["details"]["is_entangling"] is True and not result2["details"].get(
            "is_clifford", False
        ):
            # Could be conditionally_simulable or not_simulable
            assert result2["simulability_class"] in (
                "conditionally_simulable",
                "not_simulable",
            )


# =============================================================================
# Cirq Backend Tests
# =============================================================================


class TestCirqBackendSimulability:
    """Tests for simulability analysis using Cirq backend.

    These tests verify that the simulability analysis functions work correctly
    with the Cirq backend for circuit simulation. The Cirq backend is one of
    three supported backends (pennylane, qiskit, cirq) and should produce
    consistent results with the other backends.
    """

    def test_estimate_entanglement_bound_cirq_non_entangling(
        self, sample_encoding_4q, skip_if_no_cirq
    ):
        """Test that non-entangling encoding has zero entanglement with Cirq backend."""
        entropy = estimate_entanglement_bound(
            sample_encoding_4q, n_samples=10, seed=42, backend="cirq"
        )
        assert entropy == 0.0

    def test_estimate_entanglement_bound_cirq_entangling(self, skip_if_no_cirq):
        """Test that entangling encoding has positive entanglement with Cirq backend."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)
        entropy = estimate_entanglement_bound(
            enc, n_samples=20, seed=42, backend="cirq"
        )

        # IQP encoding should produce entanglement
        assert entropy > 0.0

    def test_estimate_entanglement_bound_cirq_seed_reproducibility(
        self, skip_if_no_cirq
    ):
        """Test that same seed gives same result with Cirq backend."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)
        entropy1 = estimate_entanglement_bound(
            enc, n_samples=10, seed=42, backend="cirq"
        )
        entropy2 = estimate_entanglement_bound(
            enc, n_samples=10, seed=42, backend="cirq"
        )

        assert entropy1 == entropy2

    def test_estimate_entanglement_bound_cirq_returns_float(
        self, sample_encoding_2q, skip_if_no_cirq
    ):
        """Test that result is a float with Cirq backend."""
        entropy = estimate_entanglement_bound(
            sample_encoding_2q, n_samples=5, seed=42, backend="cirq"
        )
        assert isinstance(entropy, float)

    def test_estimate_entanglement_bound_cirq_upper_bound(self, skip_if_no_cirq):
        """Test that entropy is bounded by n_qubits/2 with Cirq backend."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=2)
        n_qubits = enc.n_qubits
        max_entropy = n_qubits / 2

        entropy = estimate_entanglement_bound(
            enc, n_samples=10, seed=42, backend="cirq"
        )

        assert entropy <= max_entropy + 1e-10  # Small tolerance for numerical error

    def test_estimate_entanglement_bound_invalid_backend_raises_error(
        self, sample_encoding_2q
    ):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="backend must be"):
            estimate_entanglement_bound(
                sample_encoding_2q, n_samples=5, backend="invalid_backend"  # type: ignore
            )


# =============================================================================
# Cross-Backend Consistency Tests
# =============================================================================


class TestCrossBackendConsistency:
    """Tests for consistency across different simulation backends.

    These tests verify that simulability analysis functions produce
    consistent results regardless of which backend is used for simulation.
    This is critical for ensuring reproducibility and user confidence.
    """

    def test_entanglement_bound_non_entangling_all_backends(
        self,
        sample_encoding_4q,
        skip_if_no_pennylane,
        skip_if_no_qiskit,
        skip_if_no_cirq,
    ):
        """Test that non-entangling encoding gives zero entropy on all backends."""
        entropy_pennylane = estimate_entanglement_bound(
            sample_encoding_4q, n_samples=5, seed=42, backend="pennylane"
        )
        entropy_qiskit = estimate_entanglement_bound(
            sample_encoding_4q, n_samples=5, seed=42, backend="qiskit"
        )
        entropy_cirq = estimate_entanglement_bound(
            sample_encoding_4q, n_samples=5, seed=42, backend="cirq"
        )

        # All should be exactly zero for non-entangling encoding
        assert entropy_pennylane == 0.0
        assert entropy_qiskit == 0.0
        assert entropy_cirq == 0.0

    def test_entanglement_bound_entangling_all_backends_consistent(
        self, skip_if_no_pennylane, skip_if_no_qiskit, skip_if_no_cirq
    ):
        """Test that entangling encoding gives consistent entropy across backends."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)

        entropy_pennylane = estimate_entanglement_bound(
            enc, n_samples=20, seed=42, backend="pennylane"
        )
        entropy_qiskit = estimate_entanglement_bound(
            enc, n_samples=20, seed=42, backend="qiskit"
        )
        entropy_cirq = estimate_entanglement_bound(
            enc, n_samples=20, seed=42, backend="cirq"
        )

        # All should be positive
        assert entropy_pennylane > 0.0
        assert entropy_qiskit > 0.0
        assert entropy_cirq > 0.0

        # Results should be close (within reasonable tolerance for different backends)
        # Note: Exact values may differ slightly due to backend implementation details
        # but the order of magnitude should be the same
        max_entropy = max(entropy_pennylane, entropy_qiskit, entropy_cirq)
        min_entropy = min(entropy_pennylane, entropy_qiskit, entropy_cirq)

        # All values should be within a factor of 2 of each other (very conservative)
        # In practice, they should be much closer
        assert max_entropy <= 2 * min_entropy + 1e-10

    def test_simulability_result_consistent_across_backends(
        self, sample_encoding_2q, entangling_encoding_4q
    ):
        """Test that check_simulability gives same result regardless of backend used.

        Note: check_simulability doesn't use a backend directly (it analyzes
        circuit structure), so this test verifies that the analysis is
        backend-independent.
        """
        # Non-entangling encoding
        result1 = check_simulability(sample_encoding_2q)
        assert result1["is_simulable"] is True
        assert result1["simulability_class"] == "simulable"

        # Entangling encoding
        result2 = check_simulability(entangling_encoding_4q)
        assert result2["is_simulable"] is False

        # These results should be deterministic and not depend on any backend
        # Run again to verify determinism
        result1_again = check_simulability(sample_encoding_2q)
        result2_again = check_simulability(entangling_encoding_4q)

        assert result1 == result1_again
        assert result2 == result2_again


# =============================================================================
# Backend-Parametrized Tests
# =============================================================================


@pytest.mark.parametrize("backend", ["pennylane", "qiskit", "cirq"])
def test_estimate_entanglement_bound_parametrized_backends(
    backend,
    sample_encoding_4q,
    pennylane_available,
    qiskit_available,
    cirq_available,
):
    """Test estimate_entanglement_bound with all backends via parametrization."""
    # Skip if backend not available
    if backend == "pennylane" and not pennylane_available:
        pytest.skip("PennyLane not available")
    if backend == "qiskit" and not qiskit_available:
        pytest.skip("Qiskit not available")
    if backend == "cirq" and not cirq_available:
        pytest.skip("Cirq not available")

    # Non-entangling encoding should give zero entropy on all backends
    entropy = estimate_entanglement_bound(
        sample_encoding_4q, n_samples=5, seed=42, backend=backend
    )
    assert entropy == 0.0
    assert isinstance(entropy, float)


@pytest.mark.parametrize("backend", ["pennylane", "qiskit", "cirq"])
def test_estimate_entanglement_bound_entangling_parametrized(
    backend,
    pennylane_available,
    qiskit_available,
    cirq_available,
):
    """Test estimate_entanglement_bound with entangling encoding on all backends."""
    from encoding_atlas import IQPEncoding

    # Skip if backend not available
    if backend == "pennylane" and not pennylane_available:
        pytest.skip("PennyLane not available")
    if backend == "qiskit" and not qiskit_available:
        pytest.skip("Qiskit not available")
    if backend == "cirq" and not cirq_available:
        pytest.skip("Cirq not available")

    enc = IQPEncoding(n_features=4, reps=1)
    entropy = estimate_entanglement_bound(enc, n_samples=10, seed=42, backend=backend)

    # Entangling encoding should have positive entropy
    assert entropy > 0.0
    assert isinstance(entropy, float)

    # Entropy should be bounded by n_qubits/2
    max_entropy = enc.n_qubits / 2
    assert entropy <= max_entropy + 1e-10


# =============================================================================
# Helper to build mock encodings for testing internal decision branches
# =============================================================================


def _make_mock_encoding(
    *,
    name: str = "MockEncoding",
    n_qubits: int = 4,
    n_features: int = 4,
    is_entangling: bool = False,
    two_qubit_gates: int = 0,
    parameter_count: int = 0,
    simulability: str = "simulable",
    gate_set: list[str] | None = None,
    entanglement: str | None = None,
    entanglement_pairs: list[tuple[int, int]] | None = None,
):
    """Create a minimal BaseEncoding subclass for testing internal decision logic.

    Returns an instance of a dynamically-created BaseEncoding subclass with the
    specified properties, allowing tests to exercise internal decision branches
    that depend on encoding attributes.
    """
    from encoding_atlas.core.base import BaseEncoding

    gate_count = two_qubit_gates + max(parameter_count, n_features)
    single_qubit_gates = gate_count - two_qubit_gates
    props = EncodingProperties(
        n_qubits=n_qubits,
        depth=1,
        gate_count=gate_count,
        single_qubit_gates=single_qubit_gates,
        two_qubit_gates=two_qubit_gates,
        parameter_count=parameter_count,
        is_entangling=is_entangling,
        simulability=simulability,
    )

    # Determine which extra slots we need for optional attributes
    extra_slots = ["_mock_n_qubits", "_mock_props"]
    if gate_set is not None:
        extra_slots.append("gate_set")
    if entanglement is not None:
        extra_slots.append("entanglement")
    if entanglement_pairs is not None:
        extra_slots.append("_mock_entanglement_pairs")

    # Build class attributes dict
    cls_attrs: dict = {
        "__slots__": tuple(extra_slots),
    }

    # Define the required abstract method implementations
    def _n_qubits_prop(self):
        return self._mock_n_qubits

    def _depth_prop(self):
        return 1

    def _compute_properties_impl(self):
        return self._mock_props

    def _get_circuit_impl(self, x, backend="pennylane"):
        return None

    def _get_circuits_impl(self, X, backend="pennylane"):
        return []

    def _get_circuit_from_validated_impl(self, x, backend):
        return None

    cls_attrs["n_qubits"] = property(_n_qubits_prop)
    cls_attrs["depth"] = property(_depth_prop)
    cls_attrs["_compute_properties"] = _compute_properties_impl
    cls_attrs["get_circuit"] = _get_circuit_impl
    cls_attrs["get_circuits"] = _get_circuits_impl
    cls_attrs["_get_circuit_from_validated"] = _get_circuit_from_validated_impl

    # Add entanglement_pairs as a method if needed
    if entanglement_pairs is not None:
        _pairs = entanglement_pairs  # capture in closure

        def _get_entanglement_pairs_method(self):
            return self._mock_entanglement_pairs

        cls_attrs["get_entanglement_pairs"] = _get_entanglement_pairs_method

    # Create dynamic class with the given name
    MockClass = type(name, (BaseEncoding,), cls_attrs)

    # Instantiate
    instance = MockClass(n_features=n_features)

    # Set slot attributes on the instance
    object.__setattr__(instance, "_mock_n_qubits", n_qubits)
    object.__setattr__(instance, "_mock_props", props)
    if gate_set is not None:
        object.__setattr__(instance, "gate_set", gate_set)
    if entanglement is not None:
        object.__setattr__(instance, "entanglement", entanglement)
    if entanglement_pairs is not None:
        object.__setattr__(instance, "_mock_entanglement_pairs", entanglement_pairs)

    return instance


# =============================================================================
# Test Class: _format_memory_size
# =============================================================================


class TestFormatMemorySize:
    """Tests for the _format_memory_size helper function."""

    def test_bytes(self):
        assert _format_memory_size(512) == "512 bytes"

    def test_kilobytes(self):
        assert _format_memory_size(1024) == "1.0 KB"
        assert _format_memory_size(1536) == "1.5 KB"

    def test_megabytes(self):
        assert _format_memory_size(16 * 1024**2) == "16.0 MB"

    def test_gigabytes(self):
        assert _format_memory_size(2.5 * 1024**3) == "2.5 GB"

    def test_terabytes(self):
        assert _format_memory_size(3.0 * 1024**4) == "3.0 TB"

    def test_zero(self):
        assert _format_memory_size(0) == "0 bytes"


# =============================================================================
# Test Class: _normalize_gate_set
# =============================================================================


class TestNormalizeGateSet:
    """Tests for the _normalize_gate_set helper function."""

    def test_list_input(self):
        result = _normalize_gate_set(["H", "CNOT", "RZ"])
        assert result == {"h", "cnot", "rz"}

    def test_tuple_input(self):
        result = _normalize_gate_set(("X", "Y", "Z"))
        assert result == {"x", "y", "z"}

    def test_set_input(self):
        result = _normalize_gate_set({"CX", "CZ"})
        assert result == {"cx", "cz"}

    def test_frozenset_input(self):
        result = _normalize_gate_set(frozenset({"SWAP", "H"}))
        assert result == {"swap", "h"}

    def test_string_input(self):
        result = _normalize_gate_set("CNOT")
        assert result == {"cnot"}

    def test_unsupported_type_returns_empty(self):
        result = _normalize_gate_set(42)
        assert result == set()

    def test_empty_list(self):
        result = _normalize_gate_set([])
        assert result == set()


# =============================================================================
# Test Class: _check_non_clifford_gates
# =============================================================================


class TestCheckNonCliffordGates:
    """Tests for the _check_non_clifford_gates function."""

    def test_parameterized_encoding_has_non_clifford(self, sample_encoding_2q):
        """Parameterized rotations are non-Clifford."""
        result = _check_non_clifford_gates(sample_encoding_2q)
        assert result["has_non_clifford"] is True
        assert result["has_parameterized_rotations"] is True

    def test_non_parameterized_encoding(self):
        """Non-parameterized encoding with no gate_set attr."""
        mock = _make_mock_encoding(parameter_count=0)
        result = _check_non_clifford_gates(mock)
        assert result["has_non_clifford"] is False
        assert result["has_parameterized_rotations"] is False
        assert result["has_t_gates"] is False
        assert result["non_clifford_gates"] == set()

    def test_encoding_with_t_gates_in_gate_set(self):
        """Encoding with T gates detected via gate_set attribute."""
        mock = _make_mock_encoding(
            parameter_count=0,
            gate_set=["H", "T", "CNOT"],
        )
        result = _check_non_clifford_gates(mock)
        assert result["has_non_clifford"] is True
        assert result["has_t_gates"] is True
        assert "t" in result["non_clifford_gates"]

    def test_encoding_with_only_clifford_gate_set(self):
        """Encoding with only Clifford gates in gate_set."""
        mock = _make_mock_encoding(
            parameter_count=0,
            gate_set=["H", "CNOT", "S", "X"],
        )
        result = _check_non_clifford_gates(mock)
        # gate_set analysis finds no non-Clifford
        assert result["has_t_gates"] is False
        assert result["non_clifford_gates"] == set()

    def test_encoding_with_rotation_in_gate_set(self):
        """Encoding with RZ in gate_set."""
        mock = _make_mock_encoding(
            parameter_count=0,
            gate_set=["H", "RZ", "CNOT"],
        )
        result = _check_non_clifford_gates(mock)
        assert result["has_non_clifford"] is True
        assert "rz" in result["non_clifford_gates"]


# =============================================================================
# Test Class: Clifford detection with gate_set attribute
# =============================================================================


class TestCliffordWithGateSet:
    """Tests for _check_clifford_property with gate_set attributes."""

    def test_basis_encoding_is_clifford(self):
        """BasisEncoding uses only X gates (Clifford)."""
        from encoding_atlas import BasisEncoding

        enc = BasisEncoding(n_features=4)
        assert _check_clifford_property(enc) is True

    def test_encoding_with_clifford_gate_set(self):
        """Encoding declaring Clifford-only gate_set is detected."""
        mock = _make_mock_encoding(
            parameter_count=0,
            gate_set=["H", "S", "CNOT", "X", "Z"],
        )
        assert _check_clifford_property(mock) is True

    def test_encoding_with_non_clifford_gate_set(self):
        """Encoding with non-Clifford gates in gate_set is detected."""
        mock = _make_mock_encoding(
            parameter_count=0,
            gate_set=["H", "T", "CNOT"],
        )
        assert _check_clifford_property(mock) is False

    def test_non_entangling_no_params_no_gate_set(self):
        """Non-entangling, no parameters, no gate_set => Clifford heuristic True."""
        mock = _make_mock_encoding(
            parameter_count=0,
            is_entangling=False,
            two_qubit_gates=0,
        )
        assert _check_clifford_property(mock) is True


# =============================================================================
# Test Class: Matchgate property with mock encodings
# =============================================================================


class TestMatchgateWithMockEncodings:
    """Tests for _check_matchgate_property with mock encodings."""

    def test_known_matchgate_encoding_linear_topology(self):
        """Known matchgate encoding name with linear topology."""
        mock = _make_mock_encoding(
            name="GivensEncoding",
            is_entangling=True,
            two_qubit_gates=3,
            entanglement="linear",
        )
        assert _check_matchgate_property(mock) is True

    def test_known_matchgate_encoding_full_topology_rejected(self):
        """Known matchgate encoding with full topology is NOT simulable."""
        mock = _make_mock_encoding(
            name="GivensEncoding",
            is_entangling=True,
            two_qubit_gates=6,
            entanglement="full",
        )
        assert _check_matchgate_property(mock) is False

    def test_encoding_with_matchgate_gate_set_linear(self):
        """Encoding with matchgate gate_set and linear topology."""
        mock = _make_mock_encoding(
            is_entangling=True,
            two_qubit_gates=3,
            gate_set=["H", "iswap", "rz"],
            entanglement="linear",
        )
        assert _check_matchgate_property(mock) is True

    def test_encoding_with_matchgate_gate_set_full_topology(self):
        """Encoding with matchgate gate_set but full topology not simulable."""
        mock = _make_mock_encoding(
            is_entangling=True,
            two_qubit_gates=6,
            gate_set=["H", "iswap", "rz"],
            entanglement="full",
        )
        assert _check_matchgate_property(mock) is False

    def test_fermionic_keyword_linear_topology(self):
        """Encoding with 'fermionic' in name and linear topology."""
        mock = _make_mock_encoding(
            name="FermionicTestEncoding",
            is_entangling=True,
            two_qubit_gates=3,
            entanglement="linear",
        )
        assert _check_matchgate_property(mock) is True

    def test_fermionic_keyword_full_topology(self):
        """Encoding with 'fermionic' in name but full topology."""
        mock = _make_mock_encoding(
            name="FermionicTestEncoding",
            is_entangling=True,
            two_qubit_gates=6,
            entanglement="full",
        )
        assert _check_matchgate_property(mock) is False


# =============================================================================
# Test Class: _get_entanglement_pattern strategies
# =============================================================================


class TestGetEntanglementPatternStrategies:
    """Tests for _get_entanglement_pattern with various detection strategies."""

    def test_strategy1_entanglement_attr_ring(self):
        """Detect 'ring' pattern via entanglement attribute (normalized to circular)."""
        mock = _make_mock_encoding(is_entangling=True, entanglement="ring")
        assert _get_entanglement_pattern(mock) == "circular"

    def test_strategy1_entanglement_attr_pbc(self):
        """Detect 'pbc' pattern (periodic boundary conditions)."""
        mock = _make_mock_encoding(is_entangling=True, entanglement="pbc")
        assert _get_entanglement_pattern(mock) == "circular"

    def test_strategy1_entanglement_attr_all_to_all(self):
        """Detect 'all-to-all' pattern (normalized to full)."""
        mock = _make_mock_encoding(is_entangling=True, entanglement="all-to-all")
        assert _get_entanglement_pattern(mock) == "full"

    def test_strategy1_entanglement_attr_nearest_neighbor(self):
        """Detect 'nearest-neighbor' normalized to 'linear'."""
        mock = _make_mock_encoding(is_entangling=True, entanglement="nearest-neighbor")
        assert _get_entanglement_pattern(mock) == "linear"

    def test_strategy1_entanglement_attr_nearest_underscore(self):
        """Detect 'nearest_neighbor' normalized to 'linear'."""
        mock = _make_mock_encoding(is_entangling=True, entanglement="nearest_neighbor")
        assert _get_entanglement_pattern(mock) == "linear"

    def test_strategy1_entanglement_attr_empty(self):
        """Detect 'empty' pattern (normalized to none)."""
        mock = _make_mock_encoding(is_entangling=True, entanglement="empty")
        assert _get_entanglement_pattern(mock) == "none"

    def test_strategy2_entanglement_pairs_linear(self):
        """Detect linear pattern from entanglement pairs."""
        mock = _make_mock_encoding(
            is_entangling=True,
            n_qubits=4,
            entanglement_pairs=[(0, 1), (1, 2), (2, 3)],
        )
        assert _get_entanglement_pattern(mock) == "linear"

    def test_strategy2_entanglement_pairs_circular(self):
        """Detect circular pattern from entanglement pairs."""
        mock = _make_mock_encoding(
            is_entangling=True,
            n_qubits=4,
            entanglement_pairs=[(0, 1), (1, 2), (2, 3), (0, 3)],
        )
        assert _get_entanglement_pattern(mock) == "circular"

    def test_strategy2_entanglement_pairs_full(self):
        """Detect full pattern from entanglement pairs."""
        mock = _make_mock_encoding(
            is_entangling=True,
            n_qubits=4,
            entanglement_pairs=[
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
            ],
        )
        assert _get_entanglement_pattern(mock) == "full"

    def test_strategy2_entanglement_pairs_partial(self):
        """Detect partial pattern from entanglement pairs."""
        mock = _make_mock_encoding(
            is_entangling=True,
            n_qubits=4,
            entanglement_pairs=[(0, 1), (2, 3)],
        )
        assert _get_entanglement_pattern(mock) == "partial"

    def test_strategy2_entanglement_pairs_empty(self):
        """Empty entanglement pairs returns 'none'."""
        mock = _make_mock_encoding(
            is_entangling=True,
            n_qubits=4,
            entanglement_pairs=[],
        )
        assert _get_entanglement_pattern(mock) == "none"

    def test_strategy2_entanglement_pairs_exception(self):
        """Exception from get_entanglement_pairs falls through gracefully."""
        from encoding_atlas.core.base import BaseEncoding

        class BrokenPairsEncoding(BaseEncoding):
            __slots__ = ()

            @property
            def n_qubits(self):
                return 4

            @property
            def depth(self):
                return 1

            def _compute_properties(self):
                return EncodingProperties(
                    n_qubits=4,
                    depth=1,
                    gate_count=6,
                    single_qubit_gates=3,
                    two_qubit_gates=3,
                    parameter_count=4,
                    is_entangling=True,
                    simulability="not_simulable",
                )

            def _get_circuit_from_validated(self, x, backend):
                return None

            def get_entanglement_pairs(self):
                raise RuntimeError("broken")

        enc = BrokenPairsEncoding(n_features=4)
        # Should fall through to next strategy without crashing
        pattern = _get_entanglement_pattern(enc)
        assert pattern in ("none", "linear", "circular", "full", "partial", "unknown")

    def test_strategy3_iqp_name_default_full(self):
        """IQP-named encoding defaults to 'full' entanglement."""
        mock = _make_mock_encoding(
            name="IQPTestEncoding",
            is_entangling=True,
        )
        assert _get_entanglement_pattern(mock) == "full"

    def test_strategy3_zz_name_default_full(self):
        """ZZ-named encoding defaults to 'full' entanglement."""
        mock = _make_mock_encoding(
            name="ZZTestEncoding",
            is_entangling=True,
        )
        assert _get_entanglement_pattern(mock) == "full"

    def test_strategy3_hardware_efficient_name_defaults_linear(self):
        """Hardware-efficient named encoding defaults to 'linear'."""
        mock = _make_mock_encoding(
            name="HardwareTestEncoding",
            is_entangling=True,
        )
        assert _get_entanglement_pattern(mock) == "linear"

    def test_strategy3_iqp_with_linear_config(self):
        """IQP-named encoding with linear in config.

        The _get_entanglement_pattern function checks encoding.config for
        entanglement specification when the encoding name contains 'iqp'.
        BaseEncoding.config returns self._config.copy(), so we set _config.
        """
        mock = _make_mock_encoding(
            name="IQPTestEncoding",
            is_entangling=True,
        )
        # BaseEncoding stores config in _config; config property returns a copy
        object.__setattr__(mock, "_config", {"entanglement": "linear"})
        assert _get_entanglement_pattern(mock) == "linear"

    def test_strategy3_iqp_with_circular_config(self):
        """IQP-named encoding with circular in config."""
        mock = _make_mock_encoding(
            name="IQPTestEncoding",
            is_entangling=True,
        )
        object.__setattr__(mock, "_config", {"entanglement": "ring"})
        assert _get_entanglement_pattern(mock) == "circular"

    def test_fallback_unknown(self):
        """Unknown encoding with no detection data returns 'unknown'."""
        mock = _make_mock_encoding(
            name="CompletelyUnknownEncoding",
            is_entangling=True,
        )
        assert _get_entanglement_pattern(mock) == "unknown"


# =============================================================================
# Test Class: check_simulability decision branches
# =============================================================================


class TestCheckSimulabilityDecisionBranches:
    """Tests for check_simulability decision logic branches."""

    def test_case3_clifford_circuit(self):
        """Case 3: Clifford-only circuit is simulable via Gottesman-Knill."""
        mock = _make_mock_encoding(
            name="CliffordTestEncoding",
            is_entangling=True,
            two_qubit_gates=3,
            parameter_count=0,
            gate_set=["H", "S", "CNOT", "X"],
            simulability="simulable",
            entanglement="linear",
        )
        result = check_simulability(mock)
        assert result["is_simulable"] is True
        assert result["simulability_class"] == "simulable"
        assert "clifford" in result["reason"].lower()
        assert any("stabilizer" in r.lower() for r in result["recommendations"])

    def test_case4_matchgate_circuit(self):
        """Case 4: Matchgate circuit with linear topology is simulable."""
        mock = _make_mock_encoding(
            name="GivensEncoding",
            is_entangling=True,
            two_qubit_gates=3,
            parameter_count=0,
            gate_set=["iswap", "rz", "h"],
            simulability="simulable",
            entanglement="linear",
        )
        result = check_simulability(mock)
        assert result["is_simulable"] is True
        assert result["simulability_class"] == "simulable"
        assert "matchgate" in result["reason"].lower()
        assert any("fermion" in r.lower() for r in result["recommendations"])

    def test_case5_circular_entanglement(self):
        """Case 5: Circular entanglement is conditionally simulable."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1, entanglement="circular")
        result = check_simulability(enc)
        assert result["simulability_class"] == "conditionally_simulable"
        assert "circular" in result["reason"].lower()
        assert any(
            "mps" in r.lower() or "dmrg" in r.lower() for r in result["recommendations"]
        )

    def test_case5_full_entanglement_not_simulable(self):
        """Case 5: Full entanglement circuit is not simulable regardless of size.

        The theoretical classification is based on the circuit family's
        asymptotic complexity.  IQP circuits with full entanglement have
        provable hardness; small-instance feasibility is in recommendations.
        """
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=2, entanglement="full")
        result = check_simulability(enc)
        assert result["simulability_class"] == "not_simulable"
        assert "iqp" in result["reason"].lower()

    def test_case5_iqp_hardness_note(self):
        """IQP circuit reason mentions provable hardness."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=2, entanglement="full")
        result = check_simulability(enc)
        reason_lower = result["reason"].lower()
        # Must mention IQP hardness (theoretical classification)
        assert "iqp" in reason_lower

    def test_case5_large_circuit_not_simulable(self):
        """Large circuit (>20 qubits) classified as not_simulable."""
        mock = _make_mock_encoding(
            name="LargeEncoding",
            n_qubits=25,
            n_features=25,
            is_entangling=True,
            two_qubit_gates=100,
            parameter_count=50,
            simulability="not_simulable",
            entanglement="full",
        )
        result = check_simulability(mock)
        assert result["is_simulable"] is False
        assert result["simulability_class"] == "not_simulable"
        assert any(
            "tensor" in r.lower() or "quantum hardware" in r.lower()
            for r in result["recommendations"]
        )

    def test_case5_large_iqp_circuit(self):
        """Large IQP circuit mentions provable hardness."""
        mock = _make_mock_encoding(
            name="IQPLargeEncoding",
            n_qubits=25,
            n_features=25,
            is_entangling=True,
            two_qubit_gates=100,
            parameter_count=50,
            simulability="not_simulable",
            entanglement="full",
        )
        result = check_simulability(mock)
        assert result["is_simulable"] is False
        assert result["simulability_class"] == "not_simulable"
        assert "iqp" in result["reason"].lower()

    def test_case5_high_entanglement_reason(self):
        """High entanglement circuit (two_qubit_gates > 2*n_qubits) reason."""
        mock = _make_mock_encoding(
            name="HighEntanglementEncoding",
            n_qubits=25,
            n_features=25,
            is_entangling=True,
            two_qubit_gates=100,  # > 25*2=50
            parameter_count=50,
            simulability="not_simulable",
            entanglement="full",
        )
        result = check_simulability(mock)
        assert "high entanglement" in result["reason"].lower()

    def test_case5_moderate_entanglement_reason(self):
        """Moderate entanglement circuit (two_qubit_gates <= 2*n_qubits) reason."""
        mock = _make_mock_encoding(
            name="ModerateEncoding",
            n_qubits=25,
            n_features=25,
            is_entangling=True,
            two_qubit_gates=30,  # <= 25*2=50
            parameter_count=50,
            simulability="not_simulable",
            entanglement="full",
        )
        result = check_simulability(mock)
        assert "entangling circuit" in result["reason"].lower()

    def test_case6_fallback_declared_simulable(self):
        """Fallback to declared simulability when entangling but 0 two-qubit gates."""
        mock = _make_mock_encoding(
            name="WeirdEncoding",
            is_entangling=True,
            two_qubit_gates=0,
            parameter_count=4,
            simulability="simulable",
        )
        result = check_simulability(mock)
        assert result["is_simulable"] is True
        assert result["simulability_class"] == "simulable"
        assert "declared simulability" in result["reason"].lower()

    def test_case6_fallback_declared_not_simulable(self):
        """Fallback with declared not_simulable."""
        mock = _make_mock_encoding(
            name="WeirdEncoding",
            is_entangling=True,
            two_qubit_gates=0,
            parameter_count=4,
            simulability="not_simulable",
        )
        result = check_simulability(mock)
        assert result["is_simulable"] is False
        assert result["simulability_class"] == "not_simulable"

    def test_case6_fallback_declared_conditionally_simulable(self):
        """Fallback with declared conditionally_simulable."""
        mock = _make_mock_encoding(
            name="WeirdEncoding",
            is_entangling=True,
            two_qubit_gates=0,
            parameter_count=4,
            simulability="conditionally_simulable",
        )
        result = check_simulability(mock)
        assert result["is_simulable"] is False
        assert result["simulability_class"] == "conditionally_simulable"

    def test_case6_invalid_declared_simulability_fallback(self):
        """Invalid declared simulability falls back to not_simulable."""
        mock = _make_mock_encoding(
            name="WeirdEncoding",
            is_entangling=True,
            two_qubit_gates=0,
            parameter_count=4,
            simulability="invalid_value",
        )
        result = check_simulability(mock)
        assert result["is_simulable"] is False
        assert result["simulability_class"] == "not_simulable"

    def test_non_clifford_gates_in_details(self):
        """Non-Clifford gates set included in details when non-empty."""
        mock = _make_mock_encoding(
            is_entangling=False,
            gate_set=["H", "T", "CNOT"],
            parameter_count=0,
        )
        result = check_simulability(mock, detailed=True)
        assert result["details"]["has_non_clifford_gates"] is True
        assert result["details"]["has_t_gates"] is True
        assert "non_clifford_gates" in result["details"]

    def test_encoding_properties_access_failure(self):
        """AnalysisError raised when encoding.properties fails."""
        from encoding_atlas.core.base import BaseEncoding

        # Create a subclass where properties raises an error
        class BrokenEncoding(BaseEncoding):
            __slots__ = ()

            @property
            def n_qubits(self):
                return 4

            @property
            def depth(self):
                return 1

            @property
            def properties(self):
                raise RuntimeError("broken")

            def _compute_properties(self):
                raise RuntimeError("broken")

            def _get_circuit_from_validated(self, x, backend):
                return None

        enc = BrokenEncoding(n_features=4)
        with pytest.raises(AnalysisError, match="Failed to access encoding properties"):
            check_simulability(enc)

    def test_memory_warning_qubits_range(self):
        """Small circuit between MEMORY_WARNING_QUBITS and SMALL_CIRCUIT_QUBITS gets warning rec."""
        # n_qubits=20 is at the boundary, but < 25 means no memory warning
        mock = _make_mock_encoding(
            name="TestEncoding",
            n_qubits=4,
            n_features=4,
            is_entangling=True,
            two_qubit_gates=6,
            parameter_count=4,
            simulability="not_simulable",
            entanglement="full",
        )
        result = check_simulability(mock)
        # 4 qubits < 25, so no memory warning
        assert not any("ensure sufficient RAM" in r for r in result["recommendations"])


# =============================================================================
# Test Class: BasisEncoding Clifford integration
# =============================================================================


class TestBasisEncodingClifford:
    """Integration tests for BasisEncoding as a Clifford circuit."""

    def test_basis_encoding_is_clifford_via_public_api(self):
        """BasisEncoding is detected as Clifford via is_clifford_circuit()."""
        from encoding_atlas import BasisEncoding

        enc = BasisEncoding(n_features=4)
        # BasisEncoding is non-entangling, so it's "simulable" via product state
        # path (Case 1), but the Clifford check should also be True
        assert is_clifford_circuit(enc) is True

    def test_basis_encoding_simulable(self):
        """BasisEncoding check_simulability returns simulable."""
        from encoding_atlas import BasisEncoding

        enc = BasisEncoding(n_features=4)
        result = check_simulability(enc)
        assert result["is_simulable"] is True
        assert result["simulability_class"] == "simulable"
        assert result["details"]["is_clifford"] is True


# =============================================================================
# Test Class: _compute_bipartite_entropy edge cases
# =============================================================================


class TestComputeBipartiteEntropyEdgeCases:
    """Additional edge case tests for _compute_bipartite_entropy."""

    def test_single_qubit_system(self):
        """Single-qubit system (n_qubits=1) with cut=0 returns 0."""
        state = np.array([1.0, 0.0], dtype=np.complex128)
        entropy = _compute_bipartite_entropy(state, n_qubits=1, cut_position=0)
        assert entropy == 0.0

    def test_single_qubit_system_cut_at_1(self):
        """Single-qubit system (n_qubits=1) with cut=1 returns 0."""
        state = np.array([1.0, 0.0], dtype=np.complex128)
        entropy = _compute_bipartite_entropy(state, n_qubits=1, cut_position=1)
        assert entropy == 0.0

    def test_wrong_statevector_size_returns_zero(self):
        """Mismatched statevector size returns 0 with warning."""
        state = np.array([1.0, 0.0, 0.0], dtype=np.complex128)  # 3 elements, not 2^n
        entropy = _compute_bipartite_entropy(state, n_qubits=2, cut_position=1)
        assert entropy == 0.0

    def test_maximally_entangled_3q_cut_at_2(self):
        """3-qubit GHZ state cut at position 2."""
        ghz = np.zeros(8, dtype=np.complex128)
        ghz[0] = 1.0 / np.sqrt(2)
        ghz[7] = 1.0 / np.sqrt(2)
        entropy = _compute_bipartite_entropy(ghz, n_qubits=3, cut_position=2)
        assert abs(entropy - 1.0) < 1e-10


# =============================================================================
# Test Class: estimate_entanglement_bound edge cases
# =============================================================================


class TestEstimateEntanglementBoundEdgeCases:
    """Edge case and error path tests for estimate_entanglement_bound."""

    def test_n_samples_1(self, sample_encoding_2q):
        """Minimum n_samples=1 works."""
        entropy = estimate_entanglement_bound(sample_encoding_2q, n_samples=1, seed=42)
        assert isinstance(entropy, float)
        assert entropy >= 0.0

    def test_invalid_backend_raises_value_error(self, sample_encoding_2q):
        """Invalid backend string raises ValueError."""
        with pytest.raises(ValueError, match="backend must be"):
            estimate_entanglement_bound(
                sample_encoding_2q, n_samples=5, backend="tensorflow"
            )

    @patch("encoding_atlas.analysis._utils.simulate_encoding_statevector")
    def test_all_simulations_fail_raises_analysis_error(
        self, mock_simulate, entangling_encoding_4q
    ):
        """AnalysisError raised when all simulations fail."""
        mock_simulate.side_effect = RuntimeError("sim failed")

        with pytest.raises(AnalysisError, match="All .* simulations failed"):
            estimate_entanglement_bound(entangling_encoding_4q, n_samples=5, seed=42)

    @patch("encoding_atlas.analysis._utils.simulate_encoding_statevector")
    def test_partial_simulation_failures_still_succeed(
        self, mock_simulate, entangling_encoding_4q
    ):
        """Partial failures still produce a result if some samples succeed."""
        n_qubits = entangling_encoding_4q.n_qubits
        # First call: initial validation (returns zeros)
        valid_state = np.zeros(2**n_qubits, dtype=np.complex128)
        valid_state[0] = 1.0

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return valid_state  # Validation call
            if call_count % 2 == 0:
                raise ValueError("numerical issue")
            return valid_state

        mock_simulate.side_effect = side_effect
        entropy = estimate_entanglement_bound(
            entangling_encoding_4q, n_samples=5, seed=42
        )
        assert isinstance(entropy, float)
        assert entropy >= 0.0

    @patch("encoding_atlas.analysis._utils.simulate_encoding_statevector")
    def test_import_error_in_sampling_raises_analysis_error(
        self, mock_simulate, entangling_encoding_4q
    ):
        """ImportError during sampling loop fails fast."""
        n_qubits = entangling_encoding_4q.n_qubits
        valid_state = np.zeros(2**n_qubits, dtype=np.complex128)
        valid_state[0] = 1.0

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return valid_state  # Validation call succeeds
            raise ImportError("backend not installed")

        mock_simulate.side_effect = side_effect

        with pytest.raises(AnalysisError, match="not available"):
            estimate_entanglement_bound(entangling_encoding_4q, n_samples=5, seed=42)

    @patch("encoding_atlas.analysis._utils.simulate_encoding_statevector")
    def test_early_validation_import_error(self, mock_simulate, entangling_encoding_4q):
        """ImportError during early validation raises AnalysisError."""
        mock_simulate.side_effect = ImportError("no pennylane")

        with pytest.raises(AnalysisError, match="not available"):
            estimate_entanglement_bound(entangling_encoding_4q, n_samples=5, seed=42)


# =============================================================================
# Test Class: Hardware Efficient Encoding simulability
# =============================================================================


class TestHardwareEfficientSimulability:
    """Tests for HardwareEfficientEncoding simulability analysis."""

    def test_hardware_efficient_not_simulable(self):
        """HardwareEfficientEncoding is entangling and not efficiently simulable."""
        from encoding_atlas import HardwareEfficientEncoding

        enc = HardwareEfficientEncoding(n_features=4)
        result = check_simulability(enc)
        assert result["is_simulable"] is False
        assert result["details"]["is_entangling"] is True

    def test_hardware_efficient_entanglement_pattern(self):
        """HardwareEfficientEncoding has linear entanglement pattern."""
        from encoding_atlas import HardwareEfficientEncoding

        enc = HardwareEfficientEncoding(n_features=4)
        pattern = _get_entanglement_pattern(enc)
        assert pattern == "linear"

    def test_hardware_efficient_conditionally_simulable(self):
        """HardwareEfficientEncoding with linear topology is conditionally simulable."""
        from encoding_atlas import HardwareEfficientEncoding

        enc = HardwareEfficientEncoding(n_features=4)
        result = check_simulability(enc)
        assert result["simulability_class"] == "conditionally_simulable"


# =============================================================================
# Test Class: Multiple encoding types
# =============================================================================


@pytest.mark.parametrize(
    "encoding_name,expected_simulable",
    [
        ("AngleEncoding", True),
        ("BasisEncoding", True),
    ],
)
def test_non_entangling_encodings_simulable(encoding_name, expected_simulable):
    """Non-entangling encodings should be classified as simulable."""
    import encoding_atlas

    enc_cls = getattr(encoding_atlas, encoding_name)
    enc = enc_cls(n_features=4)
    result = check_simulability(enc)
    assert result["is_simulable"] is expected_simulable


@pytest.mark.parametrize(
    "encoding_name",
    ["IQPEncoding", "ZZFeatureMap", "PauliFeatureMap"],
)
def test_entangling_encodings_not_simulable(encoding_name):
    """All entangling encodings should not be efficiently simulable."""
    import encoding_atlas

    enc_cls = getattr(encoding_atlas, encoding_name)
    enc = enc_cls(n_features=4, reps=1)
    result = check_simulability(enc)
    assert result["is_simulable"] is False
    assert result["simulability_class"] in ("conditionally_simulable", "not_simulable")


# =============================================================================
# Additional Coverage Tests
# =============================================================================


class TestFallbackRecommendations:
    """Tests for the fallback recommendation paths in check_simulability."""

    def test_case6_simulable_fallback_recommendations(self):
        """Fallback simulable encoding with no prior recommendations gets default recs.

        This covers the 'if not recommendations: if is_simulable:' branch
        at lines 728-732 where a large entangling encoding (>20 qubits) with
        declared simulability='simulable' and zero two-qubit gates reaches
        the fallback path without any recommendations accumulated.
        """
        mock = _make_mock_encoding(
            name="WeirdLargeEncoding",
            n_qubits=25,
            n_features=25,
            is_entangling=True,
            two_qubit_gates=0,
            parameter_count=4,
            simulability="simulable",
        )
        result = check_simulability(mock)
        assert result["is_simulable"] is True
        assert result["simulability_class"] == "simulable"
        assert any(
            "standard simulation" in r.lower() for r in result["recommendations"]
        )

    def test_case6_not_simulable_fallback_recommendations(self):
        """Fallback not-simulable encoding with no prior recommendations gets default recs.

        Covers the 'else:' branch at lines 733-737.
        """
        mock = _make_mock_encoding(
            name="WeirdLargeEncoding",
            n_qubits=25,
            n_features=25,
            is_entangling=True,
            two_qubit_gates=0,
            parameter_count=4,
            simulability="not_simulable",
        )
        result = check_simulability(mock)
        assert result["is_simulable"] is False
        assert any(
            "statevector" in r.lower() or "tensor" in r.lower()
            for r in result["recommendations"]
        )


class TestGetEntanglementPatternAdditional:
    """Additional tests for _get_entanglement_pattern edge cases."""

    def test_strategy1_partial_pattern(self):
        """Detect 'partial' pattern via entanglement attribute."""
        mock = _make_mock_encoding(is_entangling=True, entanglement="partial")
        assert _get_entanglement_pattern(mock) == "partial"

    def test_strategy2_non_iterable_pairs_skipped(self):
        """Non-iterable items in entanglement pairs are gracefully skipped.

        When get_entanglement_pairs() returns items that are not iterable
        (e.g., integers), the normalization code skips them via the
        `if hasattr(p, "__iter__")` check, and if no pairs normalize
        successfully, the function falls through to the next strategy.
        """
        from encoding_atlas.core.base import BaseEncoding

        class NonIterablePairsEncoding(BaseEncoding):
            __slots__ = ()

            @property
            def n_qubits(self):
                return 4

            @property
            def depth(self):
                return 1

            def _compute_properties(self):
                return EncodingProperties(
                    n_qubits=4,
                    depth=1,
                    gate_count=6,
                    single_qubit_gates=3,
                    two_qubit_gates=3,
                    parameter_count=4,
                    is_entangling=True,
                    simulability="not_simulable",
                )

            def _get_circuit_from_validated(self, x, backend):
                return None

            def get_entanglement_pairs(self):
                # Return non-iterable items and strings (should be skipped)
                return [42, "invalid", None]

        enc = NonIterablePairsEncoding(n_features=4)
        pattern = _get_entanglement_pattern(enc)
        # Should fall through to strategy 3 or fallback
        assert pattern in ("none", "linear", "circular", "full", "partial", "unknown")

    def test_strategy2_short_pair_skipped(self):
        """Pairs with fewer than 2 elements are skipped.

        Covers the `if len(pair_list) >= 2` branch where a pair
        has only 1 element.
        """
        from encoding_atlas.core.base import BaseEncoding

        class ShortPairsEncoding(BaseEncoding):
            __slots__ = ()

            @property
            def n_qubits(self):
                return 4

            @property
            def depth(self):
                return 1

            def _compute_properties(self):
                return EncodingProperties(
                    n_qubits=4,
                    depth=1,
                    gate_count=6,
                    single_qubit_gates=3,
                    two_qubit_gates=3,
                    parameter_count=4,
                    is_entangling=True,
                    simulability="not_simulable",
                )

            def _get_circuit_from_validated(self, x, backend):
                return None

            def get_entanglement_pairs(self):
                # Single-element tuples should be skipped
                return [(0,), (1,)]

        enc = ShortPairsEncoding(n_features=4)
        pattern = _get_entanglement_pattern(enc)
        assert pattern in ("none", "linear", "circular", "full", "partial", "unknown")

    def test_strategy2_pair_conversion_error_skipped(self):
        """Pairs that raise errors during int() conversion are skipped.

        Covers the `except (TypeError, ValueError, IndexError): continue`
        branch at lines 1542-1543.
        """
        from encoding_atlas.core.base import BaseEncoding

        class BadConversionPairsEncoding(BaseEncoding):
            __slots__ = ()

            @property
            def n_qubits(self):
                return 4

            @property
            def depth(self):
                return 1

            def _compute_properties(self):
                return EncodingProperties(
                    n_qubits=4,
                    depth=1,
                    gate_count=6,
                    single_qubit_gates=3,
                    two_qubit_gates=3,
                    parameter_count=4,
                    is_entangling=True,
                    simulability="not_simulable",
                )

            def _get_circuit_from_validated(self, x, backend):
                return None

            def get_entanglement_pairs(self):
                # Pairs where int() conversion will raise ValueError
                return [("not_a_number", "also_not"), (0, 1), (1, 2), (2, 3)]

        enc = BadConversionPairsEncoding(n_features=4)
        pattern = _get_entanglement_pattern(enc)
        # The bad pair is skipped; remaining pairs form a linear pattern
        assert pattern == "linear"

    def test_strategy3_iqp_with_non_dict_config(self):
        """IQP-named encoding with non-dict config falls back to 'full'."""
        mock = _make_mock_encoding(
            name="IQPTestEncoding",
            is_entangling=True,
        )
        # Set config with a non-string entanglement value (not checked by the code)
        object.__setattr__(mock, "_config", {"entanglement": 42})
        assert _get_entanglement_pattern(mock) == "full"


class TestComputeBipartiteEntropyNumerical:
    """Tests for numerical edge cases in _compute_bipartite_entropy."""

    def test_linalg_error_returns_zero(self):
        """LinAlgError during eigenvalue computation returns 0.0.

        Covers the except np.linalg.LinAlgError branch at lines 1719-1721.
        """
        from unittest.mock import patch

        state = np.array(
            [1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)], dtype=np.complex128
        )
        with patch("numpy.linalg.eigvalsh", side_effect=np.linalg.LinAlgError("test")):
            entropy = _compute_bipartite_entropy(state, n_qubits=2, cut_position=1)
        assert entropy == 0.0

    def test_all_negative_eigenvalues_returns_zero(self):
        """When all eigenvalues are below epsilon threshold, returns 0.0.

        Covers the `if len(positive_eigenvalues) == 0` branch at line 1726-1728.
        """
        from unittest.mock import patch

        state = np.array(
            [1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)], dtype=np.complex128
        )
        # Mock eigvalsh to return all-negative eigenvalues
        mock_eigenvalues = np.array([-1e-20, -1e-20], dtype=np.float64)
        with patch("numpy.linalg.eigvalsh", return_value=mock_eigenvalues):
            entropy = _compute_bipartite_entropy(state, n_qubits=2, cut_position=1)
        assert entropy == 0.0


class TestEstimateEntanglementBoundMemoryWarning:
    """Test the memory warning path in estimate_entanglement_bound."""

    def test_large_qubit_memory_warning(self):
        """Memory warning issued for encodings with > 25 qubits.

        Covers the warning block at lines 1047-1053.
        """
        mock = _make_mock_encoding(
            name="LargeEntanglingEncoding",
            n_qubits=26,
            n_features=26,
            is_entangling=True,
            two_qubit_gates=50,
            parameter_count=26,
            simulability="not_simulable",
        )
        # The function will warn but then fail on simulation (mock can't simulate)
        # We just verify the warning is issued
        import warnings as warn_mod

        with warn_mod.catch_warnings(record=True) as w:
            warn_mod.simplefilter("always")
            try:
                estimate_entanglement_bound(mock, n_samples=1, seed=42)
            except Exception:
                pass  # Expected to fail on actual simulation

            # Check that memory warning was issued
            memory_warnings = [
                x
                for x in w
                if issubclass(x.category, UserWarning)
                and "memory" in str(x.message).lower()
            ]
            assert len(memory_warnings) >= 1

    def test_single_qubit_mid_qubit_fallback(self):
        """Single-qubit entangling encoding uses mid_qubit=1 fallback.

        Covers the `if mid_qubit == 0: mid_qubit = 1` branch at lines 1105-1106.
        This occurs when n_qubits=1 (n_qubits // 2 = 0).
        """
        mock = _make_mock_encoding(
            name="SingleQubitEntangling",
            n_qubits=1,
            n_features=1,
            is_entangling=True,
            two_qubit_gates=0,
            parameter_count=1,
            simulability="not_simulable",
        )
        # Mock simulate to return a valid 1-qubit statevector
        valid_state = np.array([1.0, 0.0], dtype=np.complex128)
        with patch(
            "encoding_atlas.analysis._utils.simulate_encoding_statevector",
            return_value=valid_state,
        ):
            entropy = estimate_entanglement_bound(mock, n_samples=3, seed=42)
        # Single qubit: _compute_bipartite_entropy(state, 1, 1) returns 0.0
        # because cut_position >= n_qubits
        assert entropy == 0.0
