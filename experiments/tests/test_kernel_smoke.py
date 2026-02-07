"""Smoke tests for quantum kernel computation (Stage 6b).

These tests verify that the quantum kernel implementation works end-to-end
on small examples, without testing statistical validity or performance.
"""

import pytest
import numpy as np


class TestKernelComputation:
    """Tests for kernel matrix computation."""

    def test_kernel_matrix_computation(self):
        """Test that kernel matrix is computed correctly."""
        from experiments.kernel import compute_kernel_matrix, simulate_encoding_state
        from encoding_atlas import IQPEncoding

        encoding = IQPEncoding(n_features=2, reps=1)
        X = np.array([[0.1, 0.2], [0.5, 0.6], [1.0, 1.1]])

        K = compute_kernel_matrix(encoding, X)

        # Check shape
        assert K.shape == (3, 3), f"Expected shape (3, 3), got {K.shape}"

        # Check symmetry
        assert np.allclose(K, K.T), "Kernel matrix should be symmetric"

        # Check diagonal is 1 (self-fidelity)
        assert np.allclose(np.diag(K), 1.0), "Diagonal should be 1 (self-fidelity)"

        # Check all entries in [0, 1]
        assert np.all(K >= 0), "All kernel entries should be >= 0"
        assert np.all(K <= 1), "All kernel entries should be <= 1"

    def test_kernel_entry_identical_states(self):
        """Test that identical states give kernel value 1."""
        from experiments.kernel import compute_kernel_entry

        state = np.array([1, 0, 0, 0], dtype=complex)

        entry = compute_kernel_entry(state, state)
        assert np.isclose(entry, 1.0), f"Self-fidelity should be 1, got {entry}"

    def test_kernel_entry_orthogonal_states(self):
        """Test that orthogonal states give kernel value 0."""
        from experiments.kernel import compute_kernel_entry

        state1 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        state2 = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩

        entry = compute_kernel_entry(state1, state2)
        assert np.isclose(entry, 0.0), f"Orthogonal states should give 0, got {entry}"

    def test_kernel_entry_bell_state(self):
        """Test kernel computation with Bell state."""
        from experiments.kernel import compute_kernel_entry

        # Bell state: (|00⟩ + |11⟩) / sqrt(2)
        bell = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)

        # Self-fidelity should be 1
        entry = compute_kernel_entry(bell, bell)
        assert np.isclose(entry, 1.0), f"Self-fidelity of Bell state should be 1, got {entry}"

    def test_simulate_encoding_state(self):
        """Test statevector simulation from encoding."""
        from experiments.kernel import simulate_encoding_state
        from encoding_atlas import AngleEncoding

        encoding = AngleEncoding(n_features=2)
        x = np.array([0.5, 1.0])

        state = simulate_encoding_state(encoding, x)

        # Check dimension
        expected_dim = 2 ** encoding.n_qubits
        assert len(state) == expected_dim, f"Expected dim {expected_dim}, got {len(state)}"

        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0), f"State should be normalized, got norm {norm}"


class TestKernelTargetAlignment:
    """Tests for kernel-target alignment computation."""

    def test_kernel_target_alignment(self):
        """Test kernel-target alignment computation."""
        from experiments.kernel import kernel_target_alignment

        # Labels: y = [0, 0, 1, 1] -> y_signed = [-1, -1, 1, 1]
        # Ideal kernel yy^T has +1 for same class, -1 for different class
        y = np.array([0, 0, 1, 1])

        # Create the ideal kernel that perfectly matches yy^T
        # y_signed = 2*y - 1 = [-1, -1, 1, 1]
        y_signed = 2.0 * y - 1.0
        K_ideal = np.outer(y_signed, y_signed)  # This is yy^T

        alignment = kernel_target_alignment(K_ideal, y)
        assert np.isclose(alignment, 1.0), f"Ideal kernel should have alignment 1.0, got {alignment}"

        # Test a kernel with block structure (high but not perfect alignment)
        K_block = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ], dtype=float)

        alignment_block = kernel_target_alignment(K_block, y)
        # Block kernel should have positive alignment (same-class entries are high)
        assert alignment_block > 0.5, f"Block kernel should have positive alignment, got {alignment_block}"

    def test_alignment_random_kernel(self):
        """Test that random kernel has lower alignment than perfect."""
        from experiments.kernel import kernel_target_alignment

        y = np.array([0, 0, 1, 1])

        # Perfect alignment kernel
        K_perfect = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ], dtype=float)

        # Random kernel (symmetric)
        np.random.seed(42)
        K_random = np.random.rand(4, 4)
        K_random = (K_random + K_random.T) / 2

        alignment_perfect = kernel_target_alignment(K_perfect, y)
        alignment_random = kernel_target_alignment(K_random, y)

        assert abs(alignment_random) < alignment_perfect, (
            f"Random alignment ({alignment_random}) should be lower than "
            f"perfect alignment ({alignment_perfect})"
        )

    def test_alignment_range(self):
        """Test that alignment is in valid range [-1, 1]."""
        from experiments.kernel import kernel_target_alignment

        y = np.array([0, 0, 1, 1])
        np.random.seed(123)
        K = np.random.rand(4, 4)
        K = (K + K.T) / 2

        alignment = kernel_target_alignment(K, y)
        assert -1 <= alignment <= 1, f"Alignment should be in [-1, 1], got {alignment}"


class TestEnsurePSD:
    """Tests for PSD enforcement."""

    def test_ensure_psd_already_psd(self):
        """Test that already PSD matrix is not modified."""
        from experiments.kernel import ensure_psd

        # Already PSD matrix
        K_psd = np.array([[1.0, 0.5], [0.5, 1.0]])
        K_out, modified = ensure_psd(K_psd)

        assert not modified, "Already PSD matrix should not be modified"
        assert np.allclose(K_out, K_psd), "Output should match input for PSD matrix"

    def test_ensure_psd_fixes_negative_eigenvalues(self):
        """Test that non-PSD matrix is corrected."""
        from experiments.kernel import ensure_psd

        # Non-PSD matrix (has eigenvalues 3 and -1)
        K_bad = np.array([[1.0, 2.0], [2.0, 1.0]])
        K_out, modified = ensure_psd(K_bad)

        assert modified, "Non-PSD matrix should be modified"

        # Check output is PSD
        eigenvalues = np.linalg.eigvalsh(K_out)
        assert np.all(eigenvalues >= 0), f"Output should be PSD, got eigenvalues {eigenvalues}"

    def test_ensure_psd_preserves_symmetry(self):
        """Test that output is symmetric."""
        from experiments.kernel import ensure_psd

        K = np.array([[1.0, 2.0], [2.0, 1.0]])
        K_out, _ = ensure_psd(K)

        assert np.allclose(K_out, K_out.T), "Output should be symmetric"


class TestKernelCrossMatrix:
    """Tests for cross kernel matrix computation."""

    def test_cross_kernel_matrix(self):
        """Test computation of test-train kernel matrix."""
        from experiments.kernel import compute_kernel_matrix_cross
        from encoding_atlas import IQPEncoding

        encoding = IQPEncoding(n_features=2, reps=1)

        X_train = np.array([[0.1, 0.2], [0.5, 0.6], [1.0, 1.1]])
        X_test = np.array([[0.3, 0.4], [0.8, 0.9]])

        K_cross = compute_kernel_matrix_cross(encoding, X_train, X_test)

        # Check shape: (n_test, n_train)
        assert K_cross.shape == (2, 3), f"Expected shape (2, 3), got {K_cross.shape}"

        # Check values in [0, 1]
        assert np.all(K_cross >= 0), "All entries should be >= 0"
        assert np.all(K_cross <= 1), "All entries should be <= 1"

    def test_cross_kernel_with_precomputed_states(self):
        """Test cross kernel with pre-computed training states."""
        from experiments.kernel import (
            compute_kernel_matrix_cross,
            simulate_encoding_state,
        )
        from encoding_atlas import IQPEncoding

        encoding = IQPEncoding(n_features=2, reps=1)

        X_train = np.array([[0.1, 0.2], [0.5, 0.6]])
        X_test = np.array([[0.3, 0.4]])

        # Pre-compute training states
        train_states = [simulate_encoding_state(encoding, x) for x in X_train]

        # Compute cross kernel
        K_cross = compute_kernel_matrix_cross(
            encoding, X_train, X_test, train_states=train_states
        )

        assert K_cross.shape == (1, 2), f"Expected shape (1, 2), got {K_cross.shape}"


class TestKernelSVMIntegration:
    """Integration tests for kernel SVM pipeline."""

    def test_kernel_svm_integration(self):
        """Test full kernel SVM pipeline."""
        from experiments.kernel import compute_kernel_matrix, ensure_psd
        from encoding_atlas import AngleEncoding
        from sklearn.svm import SVC

        encoding = AngleEncoding(n_features=2, reps=1)

        # Simple linearly separable data
        X = np.array([
            [0.1, 0.1], [0.2, 0.2], [0.3, 0.1],  # Class 0
            [2.0, 2.0], [2.1, 2.2], [2.2, 2.0],  # Class 1
        ])
        y = np.array([0, 0, 0, 1, 1, 1])

        K = compute_kernel_matrix(encoding, X)
        K_psd, _ = ensure_psd(K)

        svm = SVC(kernel="precomputed")
        svm.fit(K_psd, y)

        predictions = svm.predict(K_psd)
        accuracy = np.mean(predictions == y)

        # Should achieve high accuracy on training data
        assert accuracy >= 0.8, f"Expected accuracy >= 0.8, got {accuracy}"

    def test_kernel_svm_with_test_set(self):
        """Test kernel SVM with separate test set."""
        from experiments.kernel import (
            compute_kernel_matrix,
            compute_kernel_matrix_cross,
            ensure_psd,
            simulate_encoding_state,
        )
        from encoding_atlas import IQPEncoding
        from sklearn.svm import SVC

        encoding = IQPEncoding(n_features=2, reps=1)

        # Training data
        X_train = np.array([
            [0.1, 0.1], [0.2, 0.2],  # Class 0
            [2.0, 2.0], [2.1, 2.1],  # Class 1
        ])
        y_train = np.array([0, 0, 1, 1])

        # Test data
        X_test = np.array([[0.15, 0.15], [2.05, 2.05]])
        y_test = np.array([0, 1])

        # Compute kernels
        K_train = compute_kernel_matrix(encoding, X_train)
        K_train_psd, _ = ensure_psd(K_train)

        train_states = [simulate_encoding_state(encoding, x) for x in X_train]
        K_test = compute_kernel_matrix_cross(
            encoding, X_train, X_test, train_states=train_states
        )

        # Train and predict
        svm = SVC(kernel="precomputed")
        svm.fit(K_train_psd, y_train)
        predictions = svm.predict(K_test)

        # Check predictions are valid
        assert len(predictions) == 2
        assert set(predictions).issubset({0, 1})


class TestKernelStatistics:
    """Tests for kernel statistics computation."""

    def test_compute_kernel_statistics(self):
        """Test kernel statistics computation."""
        from experiments.kernel import compute_kernel_statistics

        K = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ])

        stats = compute_kernel_statistics(K)

        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "rank" in stats
        assert "trace" in stats

        # Check trace equals n for normalized kernel
        assert np.isclose(stats["trace"], 3.0)

        # Check rank is reasonable
        assert stats["rank"] == 3


class TestKernelHandler:
    """Tests for the kernel stage handler."""

    @pytest.mark.slow
    def test_handle_kernel_basic(self):
        """Test that the kernel handler runs without errors."""
        from experiments.runner import _handle_kernel
        from experiments.config import ExperimentConfig, EncodingSpec
        from encoding_atlas import IQPEncoding

        # Create a minimal config
        config = ExperimentConfig(
            stage="kernel",
            base_seed=42,
            backend="pennylane",
            encoding_specs=[EncodingSpec(name="iqp", params={"n_features": 2, "reps": 1})],
            analysis_params={
                "datasets": ["moons"],
                "n_runs": 1,
                "n_folds": 2,
                "compute_alignment": True,
            },
            output_dir="experiments/results/raw/test_kernel",
        )

        encoding = IQPEncoding(n_features=2, reps=1)
        # Use task_seed(0) so encoding_index is non-negative
        result = _handle_kernel(encoding, config, seed=config.task_seed(0))

        assert "encoding_name" in result
        assert "datasets" in result
        assert "moons" in result["datasets"]

        moons_result = result["datasets"]["moons"]
        assert moons_result["status"] == "success", (
            f"Expected success, got: {moons_result}"
        )

    def test_handle_kernel_feature_mismatch(self):
        """Test that handler skips mismatched feature dimensions."""
        from experiments.runner import _handle_kernel
        from experiments.config import ExperimentConfig, EncodingSpec
        from encoding_atlas import IQPEncoding

        config = ExperimentConfig(
            stage="kernel",
            base_seed=42,
            backend="pennylane",
            encoding_specs=[EncodingSpec(name="iqp", params={"n_features": 2, "reps": 1})],
            analysis_params={
                "datasets": ["iris"],  # iris has 4 features
                "n_runs": 1,
                "n_folds": 2,
                "compute_alignment": True,
            },
            output_dir="experiments/results/raw/test_kernel",
        )

        # 2-feature encoding vs 4-feature dataset
        encoding = IQPEncoding(n_features=2, reps=1)
        result = _handle_kernel(encoding, config, seed=config.task_seed(0))

        assert result["datasets"]["iris"]["status"] == "skipped"
        assert "Feature mismatch" in result["datasets"]["iris"]["reason"]

    @pytest.mark.slow
    def test_handle_kernel_alignment_computed(self):
        """Test that kernel-target alignment is computed when requested."""
        from experiments.runner import _handle_kernel
        from experiments.config import ExperimentConfig, EncodingSpec
        from encoding_atlas import IQPEncoding

        config = ExperimentConfig(
            stage="kernel",
            base_seed=42,
            backend="pennylane",
            encoding_specs=[EncodingSpec(name="iqp", params={"n_features": 2, "reps": 1})],
            analysis_params={
                "datasets": ["moons"],
                "n_runs": 1,
                "n_folds": 2,
                "compute_alignment": True,
            },
            output_dir="experiments/results/raw/test_kernel",
        )

        encoding = IQPEncoding(n_features=2, reps=1)
        result = _handle_kernel(encoding, config, seed=config.task_seed(0))

        moons_result = result["datasets"]["moons"]
        assert moons_result["status"] == "success", (
            f"Expected success, got: {moons_result}"
        )
        assert "kernel_target_alignment" in moons_result
        if moons_result["kernel_target_alignment"] is not None:
            alignment = moons_result["kernel_target_alignment"]
            assert -1 <= alignment <= 1, f"Alignment should be in [-1, 1], got {alignment}"


class TestDifferentEncodings:
    """Test kernel computation with different encoding types."""

    @pytest.mark.parametrize("encoding_name,params", [
        ("iqp", {"n_features": 2, "reps": 1}),
        ("angle", {"n_features": 2}),
        ("zz_feature_map", {"n_features": 2, "reps": 1}),
    ])
    def test_kernel_with_encoding(self, encoding_name, params):
        """Test kernel computation works with various encodings."""
        from experiments.kernel import compute_kernel_matrix
        from encoding_atlas import get_encoding

        encoding = get_encoding(encoding_name, **params)
        X = np.array([[0.1, 0.2], [0.5, 0.6], [1.0, 1.1]])

        K = compute_kernel_matrix(encoding, X)

        # Basic checks
        assert K.shape == (3, 3)
        assert np.allclose(K, K.T)
        assert np.allclose(np.diag(K), 1.0)
        assert np.all(K >= 0) and np.all(K <= 1)
