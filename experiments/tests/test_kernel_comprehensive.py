"""Comprehensive tests for quantum kernel computation (Stage 6b).

Covers:
- kernel.py: all public functions, edge cases, mathematical properties
- _handle_kernel: runner integration, config paths, output schema
- Cross-consistency: kernel matrix vs cross-kernel, return_states reuse
- Encoding breadth: entangling, non-entangling, amplitude, equivariant

Run with::

    pytest experiments/tests/test_kernel_comprehensive.py -v
    pytest experiments/tests/test_kernel_comprehensive.py -v -m "not slow"
"""

import time

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(
    datasets=("moons",),
    n_runs=1,
    n_folds=2,
    compute_alignment=True,
    n_features=2,
):
    """Build a minimal ExperimentConfig for kernel stage tests."""
    from experiments.config import ExperimentConfig, EncodingSpec

    return ExperimentConfig(
        stage="kernel",
        base_seed=42,
        backend="pennylane",
        encoding_specs=[
            EncodingSpec(name="iqp", params={"n_features": n_features, "reps": 1}),
        ],
        analysis_params={
            "datasets": list(datasets),
            "n_runs": n_runs,
            "n_folds": n_folds,
            "compute_alignment": compute_alignment,
        },
        output_dir="experiments/results/raw/test_kernel",
    )


def _handler_seed(config=None, task_index=0):
    """Return the correct seed to pass to _handle_kernel.

    The handler expects ``config.task_seed(task_index)`` which equals
    ``config.stage_seed + task_index``.  Using a raw seed like 42 would
    yield a negative encoding_index and invalid SVM random_state.
    """
    if config is None:
        config = _make_config()
    return config.task_seed(task_index)


# =========================================================================
# 1. compute_kernel_entry — mathematical properties
# =========================================================================

class TestKernelEntryMath:
    """Mathematical invariants of the fidelity kernel entry."""

    def test_global_phase_invariance(self):
        """K should be invariant under global phase: e^{iθ}|ψ⟩."""
        from experiments.kernel import compute_kernel_entry

        state = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
        phase = np.exp(1j * np.pi / 3)
        state_phased = phase * state

        k_original = compute_kernel_entry(state, state)
        k_phased = compute_kernel_entry(state, state_phased)

        assert np.isclose(k_original, 1.0)
        assert np.isclose(k_phased, 1.0), (
            f"Kernel should be phase-invariant, got {k_phased}"
        )

    def test_symmetry(self):
        """K(ψ, φ) == K(φ, ψ)."""
        from experiments.kernel import compute_kernel_entry

        s1 = np.array([1, 1j, 0, 0], dtype=complex) / np.sqrt(2)
        s2 = np.array([0, 0, 1, 1j], dtype=complex) / np.sqrt(2)

        assert np.isclose(
            compute_kernel_entry(s1, s2),
            compute_kernel_entry(s2, s1),
        )

    def test_known_overlap(self):
        """|⟨+|0⟩|² = 1/2 for single-qubit |+⟩ and |0⟩."""
        from experiments.kernel import compute_kernel_entry

        zero = np.array([1, 0], dtype=complex)
        plus = np.array([1, 1], dtype=complex) / np.sqrt(2)

        k = compute_kernel_entry(zero, plus)
        assert np.isclose(k, 0.5), f"Expected 0.5, got {k}"

    def test_numerical_clamp(self):
        """Fidelity should never exceed [0, 1] even with float noise."""
        from experiments.kernel import compute_kernel_entry

        # Unnormalised state (norm > 1) — could produce fidelity > 1 without clamp
        s = np.array([1.00000001, 0, 0, 0], dtype=complex)
        k = compute_kernel_entry(s, s)
        assert 0.0 <= k <= 1.0


# =========================================================================
# 2. compute_kernel_matrix — coverage of all branches
# =========================================================================

class TestKernelMatrixBranches:
    """Test every code path in compute_kernel_matrix."""

    def test_return_states_true(self):
        """return_states=True should return (K, states) tuple."""
        from experiments.kernel import compute_kernel_matrix
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        X = np.array([[0.1, 0.2], [0.5, 0.6]])

        result = compute_kernel_matrix(enc, X, return_states=True)
        assert isinstance(result, tuple) and len(result) == 2

        K, states = result
        assert K.shape == (2, 2)
        assert len(states) == 2
        # Each state should be a normalised complex vector
        for s in states:
            assert np.isclose(np.linalg.norm(s), 1.0)

    def test_return_states_false(self):
        """return_states=False (default) should return just K."""
        from experiments.kernel import compute_kernel_matrix
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        X = np.array([[0.1, 0.2], [0.5, 0.6]])

        K = compute_kernel_matrix(enc, X, return_states=False)
        assert isinstance(K, np.ndarray)
        assert K.shape == (2, 2)

    def test_verbose_logging(self, caplog):
        """verbose=True should produce log messages."""
        import logging
        from experiments.kernel import compute_kernel_matrix
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        X = np.array([[0.1, 0.2], [0.5, 0.6]])

        with caplog.at_level(logging.INFO, logger="experiments.kernel"):
            compute_kernel_matrix(enc, X, verbose=True)

        assert any("statevector" in r.message.lower() for r in caplog.records)

    def test_single_sample(self):
        """Kernel of a single sample should be [[1.0]]."""
        from experiments.kernel import compute_kernel_matrix
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        X = np.array([[0.3, 0.7]])

        K = compute_kernel_matrix(enc, X)
        assert K.shape == (1, 1)
        assert np.isclose(K[0, 0], 1.0)

    def test_two_identical_samples(self):
        """Two identical inputs should give an all-ones kernel."""
        from experiments.kernel import compute_kernel_matrix
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        x = np.array([0.3, 0.7])
        X = np.vstack([x, x])

        K = compute_kernel_matrix(enc, X)
        assert np.allclose(K, np.ones((2, 2)))

    def test_psd_property(self):
        """The kernel matrix should be positive semi-definite."""
        from experiments.kernel import compute_kernel_matrix
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=2, reps=1)
        rng = np.random.default_rng(99)
        X = rng.uniform(0, 2 * np.pi, (6, 2))

        K = compute_kernel_matrix(enc, X)
        eigenvalues = np.linalg.eigvalsh(K)
        assert np.all(eigenvalues >= -1e-10), (
            f"Kernel should be PSD, min eigenvalue = {eigenvalues.min()}"
        )


# =========================================================================
# 3. compute_kernel_matrix_cross — consistency with full kernel
# =========================================================================

class TestCrossKernelConsistency:
    """Cross-kernel entries should match the full kernel sub-block."""

    def test_cross_matches_full_kernel(self):
        """K_cross[i,j] should equal K_full[test_i, train_j]."""
        from experiments.kernel import (
            compute_kernel_matrix,
            compute_kernel_matrix_cross,
        )
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=2, reps=1)
        X_all = np.array([
            [0.1, 0.2],   # train 0
            [0.5, 0.6],   # train 1
            [1.0, 1.1],   # train 2
            [0.3, 0.4],   # test 0
            [0.8, 0.9],   # test 1
        ])
        X_train = X_all[:3]
        X_test = X_all[3:]

        K_full = compute_kernel_matrix(enc, X_all)
        K_cross = compute_kernel_matrix_cross(enc, X_train, X_test)

        # K_cross should match the sub-block K_full[3:, :3]
        expected = K_full[3:, :3]
        assert np.allclose(K_cross, expected, atol=1e-12), (
            f"Cross-kernel mismatch.\nGot:\n{K_cross}\nExpected:\n{expected}"
        )

    def test_precomputed_states_match_recomputed(self):
        """Pre-computed states should give identical cross-kernel."""
        from experiments.kernel import (
            compute_kernel_matrix,
            compute_kernel_matrix_cross,
        )
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=2, reps=1)
        X_train = np.array([[0.1, 0.2], [0.5, 0.6]])
        X_test = np.array([[0.3, 0.4]])

        # Path 1: no pre-computed states (states computed inside cross)
        K_cross_a = compute_kernel_matrix_cross(enc, X_train, X_test)

        # Path 2: with pre-computed states from full kernel
        _, states = compute_kernel_matrix(enc, X_train, return_states=True)
        K_cross_b = compute_kernel_matrix_cross(
            enc, X_train, X_test, train_states=states,
        )

        assert np.allclose(K_cross_a, K_cross_b, atol=1e-12)

    def test_single_test_single_train(self):
        """1×1 cross-kernel should have shape (1, 1)."""
        from experiments.kernel import compute_kernel_matrix_cross
        from encoding_atlas import AngleEncoding

        enc = AngleEncoding(n_features=2)
        K = compute_kernel_matrix_cross(
            enc, np.array([[0.1, 0.2]]), np.array([[0.3, 0.4]]),
        )
        assert K.shape == (1, 1)
        assert 0.0 <= K[0, 0] <= 1.0


# =========================================================================
# 4. kernel_target_alignment — edge cases
# =========================================================================

class TestKTAEdgeCases:
    """Edge cases and additional mathematical checks for KTA."""

    def test_anti_correlated_kernel(self):
        """A kernel anti-correlated with labels should have negative KTA."""
        from experiments.kernel import kernel_target_alignment

        y = np.array([0, 0, 1, 1])
        # Anti-correlated: high values for different-class pairs
        K_anti = np.array([
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [2, 2, 1, 1],
            [2, 2, 1, 1],
        ], dtype=float)

        alignment = kernel_target_alignment(K_anti, y)
        assert alignment < 0, f"Anti-correlated kernel should be negative, got {alignment}"

    def test_identity_kernel_alignment(self):
        """Identity kernel should have a small positive alignment (known bias)."""
        from experiments.kernel import kernel_target_alignment

        n = 10
        K_identity = np.eye(n)
        y = np.array([0] * 5 + [1] * 5)

        alignment = kernel_target_alignment(K_identity, y)
        # Uncentered KTA of identity = 1/sqrt(n) for balanced labels
        assert alignment > 0, "Identity kernel alignment should be positive"
        assert alignment < 0.5, (
            f"Identity kernel alignment should be modest, got {alignment}"
        )

    def test_zero_kernel(self):
        """All-zero kernel should return 0.0 (degenerate guard)."""
        from experiments.kernel import kernel_target_alignment

        K = np.zeros((4, 4))
        y = np.array([0, 0, 1, 1])
        assert kernel_target_alignment(K, y) == 0.0

    def test_single_class_labels(self):
        """All same-class labels should still return a value without crash."""
        from experiments.kernel import kernel_target_alignment

        K = np.array([[1.0, 0.5], [0.5, 1.0]])
        y = np.array([1, 1])

        alignment = kernel_target_alignment(K, y)
        assert isinstance(alignment, float)
        assert -1 <= alignment <= 1

    def test_large_balanced_dataset(self):
        """KTA should stay in [-1, 1] for a moderately sized random kernel."""
        from experiments.kernel import kernel_target_alignment

        rng = np.random.default_rng(77)
        n = 50
        K = rng.uniform(0, 1, (n, n))
        K = (K + K.T) / 2
        np.fill_diagonal(K, 1.0)
        y = np.array([0] * 25 + [1] * 25)

        alignment = kernel_target_alignment(K, y)
        assert -1.0 <= alignment <= 1.0

    def test_imbalanced_labels(self):
        """Imbalanced labels (1 vs 3) should not crash."""
        from experiments.kernel import kernel_target_alignment

        K = np.array([
            [1.0, 0.3, 0.2, 0.1],
            [0.3, 1.0, 0.4, 0.2],
            [0.2, 0.4, 1.0, 0.5],
            [0.1, 0.2, 0.5, 1.0],
        ])
        y = np.array([0, 1, 1, 1])

        alignment = kernel_target_alignment(K, y)
        assert isinstance(alignment, float)
        assert -1 <= alignment <= 1


# =========================================================================
# 5. ensure_psd — additional edge cases
# =========================================================================

class TestEnsurePSDEdgeCases:
    """Edge cases for PSD enforcement."""

    def test_custom_epsilon(self):
        """Custom epsilon should be used for eigenvalue clipping."""
        from experiments.kernel import ensure_psd

        K_bad = np.array([[1.0, 2.0], [2.0, 1.0]])  # eigenvalues: 3, -1
        K_out, modified = ensure_psd(K_bad, epsilon=0.5)

        assert modified
        eigenvalues = np.linalg.eigvalsh(K_out)
        assert np.min(eigenvalues) >= 0.5 - 1e-10

    def test_borderline_negative_eigenvalue_within_tolerance(self):
        """Eigenvalue at -epsilon is within tolerance and should NOT be modified."""
        from experiments.kernel import ensure_psd

        # Build a matrix with eigenvalue = -1e-8 (exactly at default epsilon)
        # eigenvalues of [[a, b],[b, a]] are a+b and a-b
        # Want a-b = -1e-8, a+b = 2.0 => a = 1.0 - 5e-9, b = 1.0 + 5e-9
        a = 1.0 - 5e-9
        b = 1.0 + 5e-9
        K = np.array([[a, b], [b, a]])

        K_out, modified = ensure_psd(K)
        # min_eigenvalue >= -epsilon means "within tolerance" → not modified
        assert not modified

    def test_eigenvalue_beyond_tolerance_is_clipped(self):
        """Eigenvalue below -epsilon should be clipped to epsilon."""
        from experiments.kernel import ensure_psd

        # Build a matrix with eigenvalue = -1e-6 (well below default epsilon=1e-8)
        # eigenvalues of [[a, b],[b, a]] are a+b and a-b
        # Want a-b = -1e-6, a+b = 2.0 => a = 1.0 - 5e-7, b = 1.0 + 5e-7
        a = 1.0 - 5e-7
        b = 1.0 + 5e-7
        K = np.array([[a, b], [b, a]])

        K_out, modified = ensure_psd(K)
        assert modified
        eigenvalues = np.linalg.eigvalsh(K_out)
        assert np.all(eigenvalues >= 0)

    def test_asymmetric_input_symmetrized(self):
        """Slightly asymmetric input should be symmetrised."""
        from experiments.kernel import ensure_psd

        K = np.array([[1.0, 0.5001], [0.4999, 1.0]])
        K_out, _ = ensure_psd(K)
        assert np.allclose(K_out, K_out.T)

    def test_identity_matrix_unchanged(self):
        """Identity matrix should pass through unmodified."""
        from experiments.kernel import ensure_psd

        K = np.eye(5)
        K_out, modified = ensure_psd(K)
        assert not modified
        assert np.allclose(K_out, K)

    def test_large_negative_eigenvalues(self):
        """Multiple large negative eigenvalues should all be clipped."""
        from experiments.kernel import ensure_psd

        # Random symmetric matrix likely has negative eigenvalues
        rng = np.random.default_rng(42)
        A = rng.normal(size=(5, 5))
        K = (A + A.T) / 2  # symmetric, but not PSD

        K_out, modified = ensure_psd(K)
        eigenvalues = np.linalg.eigvalsh(K_out)
        assert np.all(eigenvalues >= 0)
        assert modified


# =========================================================================
# 6. compute_kernel_statistics — coverage gaps
# =========================================================================

class TestKernelStatisticsEdgeCases:
    """Edge cases for kernel statistics computation."""

    def test_identity_kernel(self):
        """Identity kernel should have mean off-diag=0, rank=n."""
        from experiments.kernel import compute_kernel_statistics

        K = np.eye(4)
        stats = compute_kernel_statistics(K)

        assert np.isclose(stats["mean"], 0.0)
        assert np.isclose(stats["std"], 0.0)
        assert np.isclose(stats["min"], 0.0)
        assert np.isclose(stats["max"], 0.0)
        assert stats["rank"] == 4
        assert np.isclose(stats["trace"], 4.0)

    def test_rank_deficient_kernel(self):
        """A rank-1 kernel (all entries 1) should have rank=1."""
        from experiments.kernel import compute_kernel_statistics

        K = np.ones((4, 4))
        stats = compute_kernel_statistics(K)

        assert stats["rank"] == 1
        assert np.isclose(stats["trace"], 4.0)
        assert np.isclose(stats["mean"], 1.0)

    def test_two_by_two(self):
        """Minimal 2×2 kernel should work."""
        from experiments.kernel import compute_kernel_statistics

        K = np.array([[1.0, 0.7], [0.7, 1.0]])
        stats = compute_kernel_statistics(K)

        assert np.isclose(stats["mean"], 0.7)
        assert np.isclose(stats["std"], 0.0)   # only two off-diag, both 0.7
        assert np.isclose(stats["min"], 0.7)
        assert np.isclose(stats["max"], 0.7)
        assert stats["rank"] == 2

    def test_statistics_values_correct(self):
        """Verify off-diagonal statistics against manual computation."""
        from experiments.kernel import compute_kernel_statistics

        K = np.array([
            [1.0, 0.2, 0.8],
            [0.2, 1.0, 0.5],
            [0.8, 0.5, 1.0],
        ])
        stats = compute_kernel_statistics(K)

        off_diag = [0.2, 0.8, 0.2, 0.5, 0.8, 0.5]
        assert np.isclose(stats["mean"], np.mean(off_diag))
        assert np.isclose(stats["std"], np.std(off_diag))
        assert np.isclose(stats["min"], 0.2)
        assert np.isclose(stats["max"], 0.8)


# =========================================================================
# 7. Encoding breadth — non-entangling, amplitude, equivariant
# =========================================================================

class TestKernelEncodingBreadth:
    """Kernel computation works for diverse encoding families."""

    @pytest.mark.parametrize("encoding_name,params", [
        ("angle", {"n_features": 2}),
        ("basis", {"n_features": 2}),
        ("higher_order_angle", {"n_features": 2, "order": 2}),
    ])
    def test_non_entangling_encodings(self, encoding_name, params):
        """Non-entangling encodings should produce valid kernels."""
        from experiments.kernel import compute_kernel_matrix
        from encoding_atlas import get_encoding

        enc = get_encoding(encoding_name, **params)
        X = np.array([[0.1, 0.2], [0.9, 1.0], [2.0, 3.0]])
        K = compute_kernel_matrix(enc, X)

        assert K.shape == (3, 3)
        assert np.allclose(K, K.T)
        assert np.allclose(np.diag(K), 1.0)
        assert np.all(K >= -1e-10) and np.all(K <= 1.0 + 1e-10)

    @pytest.mark.parametrize("encoding_name,params", [
        ("iqp", {"n_features": 2, "reps": 1}),
        ("zz_feature_map", {"n_features": 2, "reps": 1}),
        ("data_reuploading", {"n_features": 2, "n_layers": 1}),
        ("hardware_efficient", {"n_features": 2, "reps": 1}),
        ("qaoa_encoding", {"n_features": 2, "reps": 1}),
    ])
    def test_entangling_encodings(self, encoding_name, params):
        """Entangling encodings should produce valid kernels."""
        from experiments.kernel import compute_kernel_matrix
        from encoding_atlas import get_encoding

        enc = get_encoding(encoding_name, **params)
        X = np.array([[0.1, 0.2], [0.9, 1.0], [2.0, 3.0]])
        K = compute_kernel_matrix(enc, X)

        assert K.shape == (3, 3)
        assert np.allclose(K, K.T)
        assert np.allclose(np.diag(K), 1.0)
        assert np.all(K >= -1e-10) and np.all(K <= 1.0 + 1e-10)

    def test_amplitude_encoding(self):
        """AmplitudeEncoding (logarithmic qubits) should produce valid kernel."""
        from experiments.kernel import compute_kernel_matrix
        from encoding_atlas import AmplitudeEncoding

        enc = AmplitudeEncoding(n_features=4)
        # AmplitudeEncoding normalises input, so values can be arbitrary
        X = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])
        K = compute_kernel_matrix(enc, X)

        assert K.shape == (2, 2)
        assert np.allclose(K, K.T)
        assert np.allclose(np.diag(K), 1.0)
        assert np.all(K >= -1e-10) and np.all(K <= 1.0 + 1e-10)


# =========================================================================
# 8. Full SVM pipeline — train/predict cycle
# =========================================================================

class TestKernelSVMPipeline:
    """End-to-end kernel SVM tests matching the handler flow."""

    def test_pipeline_with_return_states_reuse(self):
        """Mimic the exact handler flow: return_states → cross-kernel → SVM."""
        from experiments.kernel import (
            compute_kernel_matrix,
            compute_kernel_matrix_cross,
            ensure_psd,
        )
        from encoding_atlas import IQPEncoding
        from sklearn.svm import SVC

        enc = IQPEncoding(n_features=2, reps=1)

        # Well-separated data
        X_train = np.array([
            [0.1, 0.1], [0.2, 0.15], [0.15, 0.2],
            [3.0, 3.0], [3.1, 3.05], [3.05, 3.1],
        ])
        y_train = np.array([0, 0, 0, 1, 1, 1])
        X_test = np.array([[0.12, 0.12], [3.02, 3.02]])
        y_test = np.array([0, 1])

        # Step 1: compute K_train + states
        K_train, states = compute_kernel_matrix(enc, X_train, return_states=True)
        K_train_psd, _ = ensure_psd(K_train)

        # Step 2: cross-kernel reusing states
        K_test = compute_kernel_matrix_cross(
            enc, X_train, X_test, train_states=states,
        )

        # Step 3: SVM
        svm = SVC(kernel="precomputed")
        svm.fit(K_train_psd, y_train)
        preds = svm.predict(K_test)

        assert len(preds) == 2
        # Well-separated data should be classified correctly
        assert preds[0] == 0
        assert preds[1] == 1

    def test_svm_with_regularised_kernel(self):
        """SVM should still work after PSD regularisation."""
        from experiments.kernel import compute_kernel_matrix, ensure_psd
        from encoding_atlas import AngleEncoding
        from sklearn.svm import SVC

        enc = AngleEncoding(n_features=2)
        X = np.array([[0.0, 0.0], [0.1, 0.1], [3.0, 3.0], [3.1, 3.1]])
        y = np.array([0, 0, 1, 1])

        K = compute_kernel_matrix(enc, X)
        K_psd, _ = ensure_psd(K)

        svm = SVC(kernel="precomputed")
        svm.fit(K_psd, y)
        preds = svm.predict(K_psd)

        # Should at least not crash
        assert len(preds) == 4
        assert set(preds).issubset({0, 1})


# =========================================================================
# 9. _handle_kernel — runner integration
# =========================================================================

class TestHandleKernelIntegration:
    """Integration tests for the _handle_kernel stage handler."""

    @pytest.mark.slow
    def test_schema_version_present(self):
        """Result must contain schema_version per experiment plan."""
        from experiments.runner import _handle_kernel
        from encoding_atlas import IQPEncoding

        config = _make_config(datasets=["moons"], n_runs=1, n_folds=2)
        enc = IQPEncoding(n_features=2, reps=1)
        result = _handle_kernel(enc, config, seed=_handler_seed(config))

        assert "schema_version" in result
        assert result["schema_version"] == "1.0"

    @pytest.mark.slow
    def test_output_structure_complete(self):
        """Verify the full output structure matches expectations."""
        from experiments.runner import _handle_kernel
        from encoding_atlas import IQPEncoding

        config = _make_config(datasets=["moons"], n_runs=1, n_folds=2)
        enc = IQPEncoding(n_features=2, reps=1)
        result = _handle_kernel(enc, config, seed=_handler_seed(config))

        # Top-level keys
        assert result["encoding_name"] == "IQPEncoding"
        assert result["n_qubits"] == 2
        assert result["n_features"] == 2
        assert "datasets" in result

        ds = result["datasets"]["moons"]
        assert ds["status"] == "success", f"Expected success, got: {ds}"

        # Aggregated metrics
        assert isinstance(ds["mean_test_accuracy"], float)
        assert isinstance(ds["std_test_accuracy"], float)
        assert isinstance(ds["mean_precision"], float)
        assert isinstance(ds["mean_recall"], float)
        assert isinstance(ds["mean_f1"], float)
        assert isinstance(ds["ci_95_lower"], float)
        assert isinstance(ds["ci_95_upper"], float)
        assert ds["ci_95_lower"] <= ds["mean_test_accuracy"] <= ds["ci_95_upper"]

        # Kernel-target alignment
        assert "kernel_target_alignment" in ds
        if ds["kernel_target_alignment"] is not None:
            assert -1 <= ds["kernel_target_alignment"] <= 1

        # Run structure
        assert len(ds["runs"]) == 1
        run = ds["runs"][0]
        assert "run" in run
        assert "cv_fold_seed" in run
        assert "kernel_run_seed" in run
        assert "folds" in run
        assert "mean_accuracy" in run

        # Fold structure
        fold = run["folds"][0]
        assert "fold" in fold
        assert fold["status"] == "success"
        assert "test_accuracy" in fold
        assert "precision" in fold
        assert "recall" in fold
        assert "f1" in fold
        assert "kernel_regularized" in fold
        assert "wall_time_seconds" in fold
        assert isinstance(fold["wall_time_seconds"], float)

    @pytest.mark.slow
    def test_multiple_runs_aggregation(self):
        """Multiple runs should aggregate correctly."""
        from experiments.runner import _handle_kernel
        from encoding_atlas import AngleEncoding

        config = _make_config(datasets=["moons"], n_runs=3, n_folds=2)
        enc = AngleEncoding(n_features=2)
        result = _handle_kernel(enc, config, seed=_handler_seed(config))

        ds = result["datasets"]["moons"]
        assert ds["status"] == "success", f"Expected success, got: {ds}"
        assert len(ds["runs"]) == 3
        # std should be >= 0 (could be 0 if all folds have identical accuracy)
        assert ds["std_test_accuracy"] >= 0.0

    def test_unknown_dataset_skipped(self):
        """Unknown dataset name should be skipped, not crash."""
        from experiments.runner import _handle_kernel
        from encoding_atlas import IQPEncoding

        config = _make_config(datasets=["nonexistent_dataset"])
        enc = IQPEncoding(n_features=2, reps=1)
        result = _handle_kernel(enc, config, seed=_handler_seed(config))

        ds = result["datasets"]["nonexistent_dataset"]
        assert ds["status"] == "skipped"
        assert "Unknown dataset" in ds["reason"]

    @pytest.mark.slow
    def test_compute_alignment_false(self):
        """When compute_alignment=False, KTA should not be computed."""
        from experiments.runner import _handle_kernel
        from encoding_atlas import IQPEncoding

        config = _make_config(
            datasets=["moons"], n_runs=1, n_folds=2,
            compute_alignment=False,
        )
        enc = IQPEncoding(n_features=2, reps=1)
        result = _handle_kernel(enc, config, seed=_handler_seed(config))

        ds = result["datasets"]["moons"]
        assert ds["status"] == "success", f"Expected success, got: {ds}"
        # KTA should remain None when alignment not requested
        assert ds["kernel_target_alignment"] is None
        # kernel_was_regularized key should NOT be present
        assert "kernel_was_regularized" not in ds

    def test_feature_mismatch_4feat_on_2feat_dataset(self):
        """4-feature encoding should skip 2-feature datasets."""
        from experiments.runner import _handle_kernel
        from encoding_atlas import IQPEncoding

        config = _make_config(datasets=["moons"], n_features=2)
        enc = IQPEncoding(n_features=4, reps=1)  # 4 features vs 2-feature dataset
        result = _handle_kernel(enc, config, seed=_handler_seed(config))

        ds = result["datasets"]["moons"]
        assert ds["status"] == "skipped"
        assert "Feature mismatch" in ds["reason"]

    @pytest.mark.slow
    def test_multiple_datasets_mixed(self):
        """Handler should process compatible datasets and skip incompatible ones."""
        from experiments.runner import _handle_kernel
        from encoding_atlas import IQPEncoding

        config = _make_config(
            datasets=["moons", "iris"],  # moons=2feat, iris=4feat
            n_runs=1,
            n_folds=2,
        )
        enc = IQPEncoding(n_features=2, reps=1)
        result = _handle_kernel(enc, config, seed=_handler_seed(config))

        # moons (2 features) should succeed
        assert result["datasets"]["moons"]["status"] == "success"
        # iris (4 features) should be skipped
        assert result["datasets"]["iris"]["status"] == "skipped"

    @pytest.mark.slow
    def test_kernel_regularized_flag(self):
        """kernel_regularized should be a boolean in each fold result."""
        from experiments.runner import _handle_kernel
        from encoding_atlas import IQPEncoding

        config = _make_config(datasets=["moons"], n_runs=1, n_folds=2)
        enc = IQPEncoding(n_features=2, reps=1)
        result = _handle_kernel(enc, config, seed=_handler_seed(config))

        ds = result["datasets"]["moons"]
        assert ds["status"] == "success", f"Expected success, got: {ds}"
        for run in ds["runs"]:
            for fold in run["folds"]:
                assert fold["status"] == "success", f"Fold failed: {fold}"
                assert isinstance(fold["kernel_regularized"], bool)

    @pytest.mark.slow
    def test_accuracy_in_valid_range(self):
        """All accuracy values should be in [0, 1]."""
        from experiments.runner import _handle_kernel
        from encoding_atlas import IQPEncoding

        config = _make_config(datasets=["moons"], n_runs=2, n_folds=2)
        enc = IQPEncoding(n_features=2, reps=1)
        result = _handle_kernel(enc, config, seed=_handler_seed(config))

        ds = result["datasets"]["moons"]
        assert ds["status"] == "success", f"Expected success, got: {ds}"
        assert 0.0 <= ds["mean_test_accuracy"] <= 1.0
        for run in ds["runs"]:
            for fold in run["folds"]:
                assert fold["status"] == "success", f"Fold failed: {fold}"
                assert 0.0 <= fold["test_accuracy"] <= 1.0
                assert 0.0 <= fold["precision"] <= 1.0
                assert 0.0 <= fold["recall"] <= 1.0
                assert 0.0 <= fold["f1"] <= 1.0

    @pytest.mark.slow
    def test_bootstrap_ci_bounds(self):
        """Bootstrap CI lower should be <= mean <= upper."""
        from experiments.runner import _handle_kernel
        from encoding_atlas import AngleEncoding

        config = _make_config(datasets=["moons"], n_runs=3, n_folds=2)
        enc = AngleEncoding(n_features=2)
        result = _handle_kernel(enc, config, seed=_handler_seed(config))

        ds = result["datasets"]["moons"]
        assert ds["status"] == "success", f"Expected success, got: {ds}"
        assert ds["ci_95_lower"] <= ds["mean_test_accuracy"]
        assert ds["mean_test_accuracy"] <= ds["ci_95_upper"]


# =========================================================================
# 10. Reproducibility and determinism
# =========================================================================

class TestKernelReproducibility:
    """Kernel computation should be deterministic for same inputs."""

    def test_deterministic_kernel_matrix(self):
        """Same encoding + same data → identical kernel matrix."""
        from experiments.kernel import compute_kernel_matrix
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=2, reps=1)
        X = np.array([[0.1, 0.2], [0.5, 0.6], [1.0, 1.1]])

        K1 = compute_kernel_matrix(enc, X)
        K2 = compute_kernel_matrix(enc, X)

        assert np.array_equal(K1, K2), "Kernel should be exactly deterministic"

    def test_different_inputs_different_kernel(self):
        """Different inputs should produce different kernel matrices."""
        from experiments.kernel import compute_kernel_matrix
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=2, reps=1)
        X1 = np.array([[0.1, 0.2], [0.5, 0.6]])
        X2 = np.array([[1.0, 2.0], [3.0, 4.0]])

        K1 = compute_kernel_matrix(enc, X1)
        K2 = compute_kernel_matrix(enc, X2)

        assert not np.allclose(K1, K2), "Different inputs should give different kernels"

    @pytest.mark.slow
    def test_handler_deterministic(self):
        """Same config + same seed → identical handler results."""
        from experiments.runner import _handle_kernel
        from encoding_atlas import AngleEncoding

        config = _make_config(datasets=["moons"], n_runs=1, n_folds=2)
        enc = AngleEncoding(n_features=2)
        seed = _handler_seed(config)

        r1 = _handle_kernel(enc, config, seed=seed)
        r2 = _handle_kernel(enc, config, seed=seed)

        ds1 = r1["datasets"]["moons"]
        ds2 = r2["datasets"]["moons"]

        assert ds1["status"] == "success", f"Run 1 failed: {ds1}"
        assert ds2["status"] == "success", f"Run 2 failed: {ds2}"
        assert ds1["mean_test_accuracy"] == ds2["mean_test_accuracy"]
        assert ds1["kernel_target_alignment"] == ds2["kernel_target_alignment"]


# =========================================================================
# 11. simulate_encoding_state — additional coverage
# =========================================================================

class TestSimulateEncodingState:
    """Additional tests for the simulation wrapper."""

    def test_state_dimension_matches_qubits(self):
        """State dimension should be 2^n_qubits."""
        from experiments.kernel import simulate_encoding_state
        from encoding_atlas import IQPEncoding

        for n in [2, 3]:
            enc = IQPEncoding(n_features=n, reps=1)
            x = np.random.default_rng(0).uniform(0, 2 * np.pi, n)
            state = simulate_encoding_state(enc, x)
            assert state.shape == (2**n,)

    def test_state_is_complex(self):
        """Statevector should be complex-valued."""
        from experiments.kernel import simulate_encoding_state
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=2, reps=1)
        x = np.array([0.5, 1.0])
        state = simulate_encoding_state(enc, x)

        assert np.iscomplexobj(state)

    def test_different_inputs_give_different_states(self):
        """Different feature vectors should produce different states."""
        from experiments.kernel import simulate_encoding_state
        from encoding_atlas import IQPEncoding

        enc = IQPEncoding(n_features=2, reps=1)
        s1 = simulate_encoding_state(enc, np.array([0.1, 0.2]))
        s2 = simulate_encoding_state(enc, np.array([1.5, 2.5]))

        assert not np.allclose(s1, s2)

    def test_amplitude_encoding_state(self):
        """AmplitudeEncoding should produce valid normalised state."""
        from experiments.kernel import simulate_encoding_state
        from encoding_atlas import AmplitudeEncoding

        enc = AmplitudeEncoding(n_features=4)
        x = np.array([1.0, 2.0, 3.0, 4.0])
        state = simulate_encoding_state(enc, x)

        assert np.isclose(np.linalg.norm(state), 1.0)
        assert state.shape == (2**enc.n_qubits,)


# =========================================================================
# 12. Integration: ExperimentRunner dispatch to kernel stage
# =========================================================================

class TestRunnerKernelDispatch:
    """Verify the ExperimentRunner correctly dispatches to _handle_kernel."""

    def test_kernel_stage_registered(self):
        """'kernel' stage should be in the handler registry."""
        from experiments.runner import _STAGE_HANDLERS
        assert "kernel" in _STAGE_HANDLERS

    def test_handler_is_callable(self):
        """The kernel handler should be a callable."""
        from experiments.runner import _STAGE_HANDLERS
        assert callable(_STAGE_HANDLERS["kernel"])

    def test_handler_signature(self):
        """The kernel handler should accept (encoding, config, seed)."""
        import inspect
        from experiments.runner import _handle_kernel

        sig = inspect.signature(_handle_kernel)
        params = list(sig.parameters.keys())
        assert params == ["encoding", "config", "seed"]


# =========================================================================
# 13. Datasets module interaction for kernel stage
# =========================================================================

class TestKernelDatasetInteraction:
    """Verify dataset loading and feature matching logic."""

    def test_all_2feat_datasets_match(self):
        """All 2-feature datasets should be processable by 2-feature encoding."""
        from experiments.datasets import DATASET_N_FEATURES

        two_feat = [name for name, n in DATASET_N_FEATURES.items() if n == 2]
        assert len(two_feat) == 4  # moons, circles, linear, xor

    def test_all_4feat_datasets_match(self):
        """All 4-feature datasets should be processable by 4-feature encoding."""
        from experiments.datasets import DATASET_N_FEATURES

        four_feat = [name for name, n in DATASET_N_FEATURES.items() if n == 4]
        assert len(four_feat) == 4  # iris, wine, breast_cancer, digits_01

    def test_dataset_features_scaled_to_2pi(self):
        """Dataset features should be in [0, 2π] range."""
        from experiments.datasets import load_dataset

        for name in ["moons", "iris"]:
            X, y = load_dataset(name, seed=42)
            assert X.min() >= 0.0 - 1e-10
            assert X.max() <= 2 * np.pi + 1e-10

    def test_dataset_labels_binary(self):
        """Dataset labels should be in {0, 1}."""
        from experiments.datasets import load_dataset

        for name in ["moons", "iris", "breast_cancer"]:
            X, y = load_dataset(name, seed=42)
            assert set(np.unique(y)).issubset({0, 1})


# =========================================================================
# 14. Config validation for kernel stage
# =========================================================================

class TestKernelConfigValidation:
    """Verify kernel stage config is valid and consistent."""

    def test_stage6b_config_loads(self):
        """The stage6b config file should load without errors."""
        import os
        from experiments.config import load_config

        config_path = os.path.join(
            "experiments", "configs", "stage6b_kernel.json",
        )
        if os.path.isfile(config_path):
            config = load_config(config_path)
            assert config.stage == "kernel"
            assert len(config.encoding_specs) > 0

    def test_stage_seed_computation(self):
        """Kernel stage seed should be base_seed + 7*1000."""
        config = _make_config()
        assert config.stage_seed == 42 + 7 * 1000  # 7042

    def test_task_seed_varies_per_task(self):
        """Each task should get a distinct seed."""
        config = _make_config()
        seeds = [config.task_seed(i) for i in range(5)]
        assert len(set(seeds)) == 5  # all unique

    def test_quick_mode_overrides(self):
        """Quick mode should reduce n_runs and n_folds for kernel stage."""
        import os
        from experiments.config import load_config

        config_path = os.path.join(
            "experiments", "configs", "stage6b_kernel.json",
        )
        if os.path.isfile(config_path):
            config = load_config(config_path, quick=True)
            assert config.analysis_params["n_runs"] == 2
            assert config.analysis_params["n_folds"] == 3
