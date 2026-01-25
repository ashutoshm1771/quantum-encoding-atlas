"""Tests for EncodingProperties."""

import pytest
from encoding_atlas.core.properties import EncodingProperties, ResourceSummary


class TestEncodingProperties:
    """Test suite for EncodingProperties dataclass."""

    def test_valid_properties(self):
        """Test that valid properties can be created."""
        props = EncodingProperties(
            n_qubits=4,
            depth=3,
            gate_count=12,
            single_qubit_gates=8,
            two_qubit_gates=4,
            parameter_count=4,
            is_entangling=True,
            simulability="not_simulable",
        )
        assert props.n_qubits == 4
        assert props.depth == 3
        assert props.is_entangling is True

    def test_immutability(self):
        """Properties should be immutable (frozen dataclass)."""
        props = EncodingProperties(
            n_qubits=4, depth=3, gate_count=8,
            single_qubit_gates=4, two_qubit_gates=4,
            parameter_count=4, is_entangling=True,
            simulability="not_simulable",
        )
        with pytest.raises(AttributeError):
            props.n_qubits = 5

    def test_invalid_qubit_count(self):
        """n_qubits must be at least 1."""
        with pytest.raises(ValueError, match="n_qubits must be at least 1"):
            EncodingProperties(
                n_qubits=0,
                depth=1, gate_count=0,
                single_qubit_gates=0, two_qubit_gates=0,
                parameter_count=0, is_entangling=False,
                simulability="simulable",
            )

    def test_gate_count_consistency(self):
        """single + two qubit gates must equal total."""
        with pytest.raises(ValueError, match="must equal gate_count"):
            EncodingProperties(
                n_qubits=4, depth=3, gate_count=10,
                single_qubit_gates=4, two_qubit_gates=4,
                parameter_count=4, is_entangling=True,
                simulability="not_simulable",
            )

    def test_to_dict(self):
        """Test conversion to dictionary."""
        props = EncodingProperties(
            n_qubits=4, depth=3, gate_count=8,
            single_qubit_gates=4, two_qubit_gates=4,
            parameter_count=4, is_entangling=True,
            simulability="not_simulable",
        )
        d = props.to_dict()
        assert isinstance(d, dict)
        assert d["n_qubits"] == 4
        assert d["is_entangling"] is True


class TestResourceSummary:
    """Tests for ResourceSummary."""

    def test_two_qubit_ratio(self):
        """Test two-qubit gate ratio computation."""
        summary = ResourceSummary(
            n_qubits=4,
            depth=3,
            gate_count=10,
            two_qubit_gates=4,
        )
        assert summary.two_qubit_ratio == 0.4

    def test_zero_gates(self):
        """Test with zero gate count."""
        summary = ResourceSummary(
            n_qubits=4,
            depth=0,
            gate_count=0,
            two_qubit_gates=0,
        )
        assert summary.two_qubit_ratio == 0.0
