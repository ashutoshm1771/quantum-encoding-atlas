"""Integration tests for encoding workflows."""

import numpy as np


class TestEncodingWorkflow:
    """Test complete encoding workflows."""

    def test_angle_encoding_workflow(self):
        """Test complete angle encoding workflow."""
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=4)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        # Should not raise
        assert enc.n_qubits == 4
        assert enc.depth == 1
        assert enc.properties.is_entangling is False

    def test_iqp_encoding_workflow(self):
        """Test complete IQP encoding workflow."""
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=4, reps=2)
        x = np.array([0.1, 0.2, 0.3, 0.4])

        assert enc.n_qubits == 4
        assert enc.properties.is_entangling is True

    def test_registry_workflow(self):
        """Test getting encodings from registry."""
        from encoding_atlas import get_encoding, list_encodings

        encodings = list_encodings()
        assert len(encodings) > 0

        enc = get_encoding("angle", n_features=4)
        assert enc.n_features == 4
