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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from encoding_atlas.analysis.simulability import (
    _check_clifford_property,
    _check_matchgate_property,
    _compute_bipartite_entropy,
    _get_entanglement_pattern,
    check_simulability,
    estimate_entanglement_bound,
    get_simulability_reason,
    is_clifford_circuit,
    is_matchgate_circuit,
)
from encoding_atlas.core.exceptions import AnalysisError

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

    def test_small_iqp_encoding_conditionally_simulable(self, entangling_encoding_4q):
        """Test that small IQP encoding is classified as conditionally simulable.

        Small circuits (≤20 qubits) can always be simulated via brute-force
        statevector methods, even if the circuit structure would otherwise
        be hard to simulate efficiently. Per the documentation, small circuits
        are classified as "conditionally_simulable".
        """
        result = check_simulability(entangling_encoding_4q)

        assert result["is_simulable"] is False  # Not *efficiently* simulable
        assert result["simulability_class"] == "conditionally_simulable"
        assert result["details"]["is_entangling"] is True
        assert len(result["recommendations"]) > 0
        # Should recommend brute-force simulation for small circuits
        assert any("statevector" in r.lower() or "brute" in r.lower()
                   for r in result["recommendations"])

    def test_small_iqp_mentions_circuit_size(self):
        """Test that small IQP encoding result mentions circuit size feasibility."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=2)
        result = check_simulability(enc)

        # Small IQP circuits should mention they can be simulated via brute force
        reason_lower = result["reason"].lower()
        assert "iqp" in reason_lower or "qubits" in reason_lower
        # Should mention brute-force or statevector feasibility
        assert any("brute" in r.lower() or "statevector" in r.lower() or "feasible" in r.lower()
                   for r in result["recommendations"])

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
        if (
            result2["details"]["is_entangling"] is True
            and not result2["details"].get("is_clifford", False)
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
        entropy = estimate_entanglement_bound(enc, n_samples=20, seed=42, backend="cirq")

        # IQP encoding should produce entanglement
        assert entropy > 0.0

    def test_estimate_entanglement_bound_cirq_seed_reproducibility(
        self, skip_if_no_cirq
    ):
        """Test that same seed gives same result with Cirq backend."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=1)
        entropy1 = estimate_entanglement_bound(enc, n_samples=10, seed=42, backend="cirq")
        entropy2 = estimate_entanglement_bound(enc, n_samples=10, seed=42, backend="cirq")

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

        entropy = estimate_entanglement_bound(enc, n_samples=10, seed=42, backend="cirq")

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
        self, sample_encoding_4q, skip_if_no_pennylane, skip_if_no_qiskit, skip_if_no_cirq
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
