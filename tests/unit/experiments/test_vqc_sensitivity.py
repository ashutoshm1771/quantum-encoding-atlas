"""Comprehensive tests for VQC hyperparameter sensitivity analysis.

This module tests the sensitivity analysis pipeline defined in
:mod:`experiments.vqc_sensitivity`, including:

1. **Grid construction**: Verify grid builds correct cells with right keys
2. **Seed computation**: Verify determinism and collision avoidance
3. **Checkpoint integration**: Verify crash-safe resume and skip
4. **Cell evaluation**: Verify training/evaluation loop (mocked VQC)
5. **Analysis summary**: Verify best config, LR/layer sensitivity, ranking
6. **CLI entry point**: Verify argparse, quick mode, exit codes
7. **JSON report structure**: Verify schema compliance
8. **Edge cases**: Empty results, all-failed folds, single-cell grid

The tests mock heavy dependencies (PennyLane, VQCClassifier, datasets) to
run quickly without quantum simulation.

Note on patching: ``_evaluate_cell`` and ``run_sensitivity_analysis`` use
lazy imports (``from X import Y`` inside the function body).  These must be
patched at their **source module** (e.g. ``sklearn.metrics.f1_score``), not
at ``experiments.vqc_sensitivity.f1_score`` (which does not exist at module
level).
"""

from __future__ import annotations

import json
import os
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from experiments.vqc_sensitivity import (
    _DATASETS,
    _DEFAULT_BASE_SEED,
    _DEFAULT_EPOCHS,
    _DEFAULT_N_FOLDS,
    _DEFAULT_N_RUNS,
    _ENCODING_CONFIGS,
    _LEARNING_RATES,
    _N_VAR_LAYERS,
    _QUICK_N_FOLDS,
    _QUICK_N_RUNS,
    _SCHEMA_VERSION,
    _SEED_OFFSET,
    _build_grid,
    _cell_id,
    _compute_analysis,
    _cv_fold_seed,
    _evaluate_cell,
    _fold_init_seed,
    _json_default,
    _vqc_run_seed,
    main,
    run_sensitivity_analysis,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_grid():
    """A minimal grid with 1 encoding, 1 dataset, 1 lr, 1 layer count."""
    return _build_grid(
        learning_rates=[0.01],
        n_var_layers=[2],
        encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
        datasets=["moons"],
    )


@pytest.fixture
def full_grid():
    """The default full grid (36 cells)."""
    return _build_grid()


@pytest.fixture
def mock_cell():
    """A single grid cell specification."""
    return {
        "encoding_name": "angle",
        "encoding_params": {"n_features": 2, "rotation": "Y"},
        "dataset": "moons",
        "lr": 0.01,
        "n_var_layers": 2,
        "cell_index": 0,
    }


@pytest.fixture
def sample_results():
    """Sample cell results for analysis testing."""
    results = []
    for enc in ["angle", "iqp"]:
        for ds in ["moons", "circles"]:
            for lr in [0.001, 0.01, 0.05]:
                for layers in [1, 2, 3]:
                    # Simulate accuracy that varies with hyperparams.
                    base_acc = 0.75
                    if enc == "iqp":
                        base_acc += 0.05
                    if ds == "circles":
                        base_acc -= 0.02
                    # lr=0.01 is best, layers=2 is best.
                    lr_bonus = {0.001: 0.0, 0.01: 0.05, 0.05: 0.02}[lr]
                    layer_bonus = {1: 0.0, 2: 0.04, 3: 0.02}[layers]

                    acc = base_acc + lr_bonus + layer_bonus
                    results.append(
                        {
                            "encoding": enc,
                            "dataset": ds,
                            "lr": lr,
                            "n_var_layers": layers,
                            "mean_accuracy": round(acc, 4),
                            "std_accuracy": 0.03,
                            "ci_lower": round(acc - 0.05, 4),
                            "ci_upper": round(acc + 0.05, 4),
                            "mean_final_loss": 0.3,
                            "mean_epochs_to_converge": 80.0,
                            "n_observations": 15,
                            "status": "success",
                        }
                    )
    return results


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Temporary output directory for test runs."""
    return str(tmp_path / "sensitivity_output")


def _make_vqc_mock(status="success", final_loss=0.3, n_epochs=5):
    """Helper: build a mock VQCClassifier instance."""
    vqc = MagicMock()
    vqc.status_ = status
    vqc.loss_history_ = [0.6 - 0.06 * i for i in range(n_epochs)]
    vqc.get_final_loss.return_value = final_loss
    vqc.fit.return_value = vqc
    vqc.predict.return_value = np.array([0, 1])
    return vqc


# Common patch paths for _evaluate_cell's lazy imports.
_PATCH_GET_ENCODING = "encoding_atlas.get_encoding"
_PATCH_LOAD_DATASET = "experiments.datasets.load_dataset"
_PATCH_GET_CV_FOLDS = "experiments.datasets.get_cv_folds"
_PATCH_VQC_CLS = "experiments.vqc.VQCClassifier"
_PATCH_CI_METRICS = "experiments.statistical.compute_ci_metrics"
_PATCH_PRECISION = "sklearn.metrics.precision_score"
_PATCH_RECALL = "sklearn.metrics.recall_score"
_PATCH_F1 = "sklearn.metrics.f1_score"

# Common patch path for run_sensitivity_analysis's lazy import.
_PATCH_CKPT_MGR = "experiments.checkpoint.CheckpointManager"


# =============================================================================
# Grid Construction Tests
# =============================================================================


class TestBuildGrid:
    """Tests for _build_grid()."""

    def test_default_grid_size(self, full_grid):
        """Default grid has 3 LRs * 3 layers * 2 encodings * 2 datasets = 36."""
        assert len(full_grid) == 36

    def test_default_grid_cell_keys(self, full_grid):
        """Each cell has all required keys."""
        required_keys = {
            "encoding_name",
            "encoding_params",
            "dataset",
            "lr",
            "n_var_layers",
            "cell_index",
        }
        for cell in full_grid:
            assert set(cell.keys()) == required_keys

    def test_cell_indices_are_sequential(self, full_grid):
        """Cell indices must be 0, 1, ..., n-1."""
        indices = [c["cell_index"] for c in full_grid]
        assert indices == list(range(len(full_grid)))

    def test_custom_grid_size(self):
        """Custom parameters produce the correct grid size."""
        grid = _build_grid(
            learning_rates=[0.01, 0.05],
            n_var_layers=[1, 3],
            encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
            datasets=["moons"],
        )
        assert len(grid) == 2 * 2 * 1 * 1

    def test_single_cell_grid(self, simple_grid):
        """A single-cell grid contains exactly one cell."""
        assert len(simple_grid) == 1
        cell = simple_grid[0]
        assert cell["encoding_name"] == "angle"
        assert cell["dataset"] == "moons"
        assert cell["lr"] == 0.01
        assert cell["n_var_layers"] == 2
        assert cell["cell_index"] == 0

    def test_encoding_params_preserved(self, full_grid):
        """Encoding params are passed through correctly."""
        angle_cells = [c for c in full_grid if c["encoding_name"] == "angle"]
        iqp_cells = [c for c in full_grid if c["encoding_name"] == "iqp"]
        assert len(angle_cells) > 0
        assert len(iqp_cells) > 0
        assert angle_cells[0]["encoding_params"]["n_features"] == 2
        assert iqp_cells[0]["encoding_params"]["n_features"] == 2

    def test_all_combinations_present(self, full_grid):
        """All combinations of (encoding, dataset, lr, layers) appear."""
        combos = {
            (c["encoding_name"], c["dataset"], c["lr"], c["n_var_layers"])
            for c in full_grid
        }
        expected_count = (
            len(_ENCODING_CONFIGS)
            * len(_DATASETS)
            * len(_LEARNING_RATES)
            * len(_N_VAR_LAYERS)
        )
        assert len(combos) == expected_count

    def test_empty_params_handled(self):
        """Grid with empty encoding params still works."""
        grid = _build_grid(
            learning_rates=[0.01],
            n_var_layers=[2],
            encoding_configs=[{"name": "angle", "params": {}}],
            datasets=["moons"],
        )
        assert len(grid) == 1
        assert grid[0]["encoding_params"] == {}


# =============================================================================
# Cell ID Tests
# =============================================================================


class TestCellId:
    """Tests for _cell_id()."""

    def test_deterministic(self, mock_cell):
        """Same cell always produces the same ID."""
        id1 = _cell_id(mock_cell)
        id2 = _cell_id(mock_cell)
        assert id1 == id2

    def test_format(self, mock_cell):
        """Cell ID has the expected format."""
        cid = _cell_id(mock_cell)
        assert cid == "sensitivity/angle/moons/lr0.01_layers2"

    def test_different_cells_different_ids(self):
        """Different cells produce different IDs."""
        cell_a = {
            "encoding_name": "angle",
            "dataset": "moons",
            "lr": 0.01,
            "n_var_layers": 2,
        }
        cell_b = {
            "encoding_name": "iqp",
            "dataset": "moons",
            "lr": 0.01,
            "n_var_layers": 2,
        }
        assert _cell_id(cell_a) != _cell_id(cell_b)

    def test_lr_precision_in_id(self):
        """LR values are preserved precisely in the ID."""
        cell = {
            "encoding_name": "angle",
            "dataset": "moons",
            "lr": 0.001,
            "n_var_layers": 3,
        }
        cid = _cell_id(cell)
        assert "lr0.001" in cid
        assert "layers3" in cid


# =============================================================================
# Seed Computation Tests
# =============================================================================


class TestSeedComputation:
    """Tests for seed helper functions."""

    def test_cv_fold_seed_encoding_independent(self):
        """CV fold seed depends only on base_seed and dataset."""
        seed_moons = _cv_fold_seed(42, "moons")
        seed_circles = _cv_fold_seed(42, "circles")
        assert seed_moons == 42  # moons is index 0
        assert seed_circles == 43  # circles is index 1

    def test_cv_fold_seed_deterministic(self):
        """Same inputs produce same seed."""
        assert _cv_fold_seed(42, "moons") == _cv_fold_seed(42, "moons")

    def test_cv_fold_seed_unknown_dataset(self):
        """Unknown dataset defaults to index 0."""
        assert _cv_fold_seed(42, "unknown") == 42

    def test_vqc_run_seed_offset(self):
        """VQC run seed uses the 9000 offset."""
        seed = _vqc_run_seed(42, cell_index=0, run_index=0)
        assert seed == 42 + _SEED_OFFSET

    def test_vqc_run_seed_varies_with_cell(self):
        """Different cells produce different run seeds."""
        s0 = _vqc_run_seed(42, cell_index=0, run_index=0)
        s1 = _vqc_run_seed(42, cell_index=1, run_index=0)
        assert s0 != s1
        assert s1 - s0 == 10000  # cell_index * 10000

    def test_vqc_run_seed_varies_with_run(self):
        """Different runs produce different seeds."""
        s0 = _vqc_run_seed(42, cell_index=0, run_index=0)
        s1 = _vqc_run_seed(42, cell_index=0, run_index=1)
        assert s1 - s0 == 1

    def test_vqc_run_seed_no_collision_with_stage6a(self):
        """9000 offset avoids collision with stage 6a seeds."""
        # Stage 6a uses stage_seed + encoding_idx*100 + dataset_idx*10 + run_idx
        # stage_seed for vqc stage is base_seed + 6*1000 = 6042
        # Max encoding_idx ~= 15, so max = 6042 + 15*100 + 8*10 + 10 = 7632
        # Our seeds start at base_seed + 9000 = 9042
        max_stage6a_seed = 42 + 6000 + 15 * 100 + 8 * 10 + 10
        min_sensitivity_seed = _vqc_run_seed(42, cell_index=0, run_index=0)
        assert min_sensitivity_seed > max_stage6a_seed

    def test_fold_init_seed_formula(self):
        """Per-fold seed matches ExperimentConfig.fold_init_seed formula."""
        run_seed = 9042
        fold_seed = _fold_init_seed(run_seed, fold_index=2)
        assert fold_seed == run_seed + 2 * 2000

    def test_fold_init_seed_unique_per_fold(self):
        """Each fold gets a distinct seed."""
        run_seed = 9042
        fold_seeds = [_fold_init_seed(run_seed, i) for i in range(5)]
        assert len(set(fold_seeds)) == 5

    def test_all_seeds_unique_across_full_grid(self):
        """No two (cell, run, fold) combinations share a seed."""
        grid = _build_grid()
        all_seeds = set()
        n_runs = 3
        n_folds = 5
        for cell in grid:
            for run_idx in range(n_runs):
                run_seed = _vqc_run_seed(42, cell["cell_index"], run_idx)
                for fold_idx in range(n_folds):
                    fold_seed = _fold_init_seed(run_seed, fold_idx)
                    all_seeds.add(fold_seed)
        # 36 cells * 3 runs * 5 folds = 540 unique seeds
        assert len(all_seeds) == 36 * n_runs * n_folds

    def test_cell_spacing_exceeds_fold_range(self):
        """Cell seed spacing (10000) exceeds max per-fold offset (8002)."""
        max_fold_offset = (_DEFAULT_N_FOLDS - 1) * 2000 + (_DEFAULT_N_RUNS - 1)
        cell_spacing = _vqc_run_seed(0, 1, 0) - _vqc_run_seed(0, 0, 0)
        assert cell_spacing > max_fold_offset


# =============================================================================
# Cell Evaluation Tests (mocked VQC)
# =============================================================================


class TestEvaluateCell:
    """Tests for _evaluate_cell() with mocked VQC.

    The lazy imports inside ``_evaluate_cell`` require patching at the
    **source module**, not at ``experiments.vqc_sensitivity``.
    """

    @patch(_PATCH_F1, return_value=0.82)
    @patch(_PATCH_RECALL, return_value=0.80)
    @patch(_PATCH_PRECISION, return_value=0.85)
    @patch(_PATCH_CI_METRICS)
    @patch(_PATCH_VQC_CLS)
    @patch(_PATCH_GET_CV_FOLDS)
    @patch(_PATCH_LOAD_DATASET)
    @patch(_PATCH_GET_ENCODING)
    def test_successful_cell(
        self,
        mock_enc,
        mock_load,
        mock_folds,
        mock_vqc_cls,
        mock_ci,
        mock_prec,
        mock_recall,
        mock_f1,
        mock_cell,
    ):
        """Successful cell evaluation populates all metrics."""
        mock_enc.return_value = MagicMock()
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        y = np.array([0, 0, 1, 1])
        mock_load.return_value = (X, y)
        mock_folds.return_value = [
            (X[:2], X[2:], y[:2], y[2:]),
            (X[2:], X[:2], y[2:], y[:2]),
        ]
        mock_vqc_cls.return_value = _make_vqc_mock()
        mock_ci.return_value = {
            "mean": 0.85,
            "std": 0.03,
            "ci_lower": 0.82,
            "ci_upper": 0.88,
        }

        result = _evaluate_cell(
            mock_cell,
            n_runs=1,
            n_folds=2,
            epochs=10,
            base_seed=42,
        )

        assert result["status"] == "success"
        assert result["encoding"] == "angle"
        assert result["dataset"] == "moons"
        assert result["lr"] == 0.01
        assert result["n_var_layers"] == 2
        assert result["mean_accuracy"] == 0.85
        assert result["std_accuracy"] == 0.03
        assert result["ci_lower"] == 0.82
        assert result["ci_upper"] == 0.88
        assert result["n_observations"] == 2  # 1 run * 2 folds
        assert len(result["runs"]) == 1
        assert len(result["runs"][0]["folds"]) == 2

    @patch(_PATCH_F1, return_value=0.0)
    @patch(_PATCH_RECALL, return_value=0.0)
    @patch(_PATCH_PRECISION, return_value=0.0)
    @patch(_PATCH_VQC_CLS)
    @patch(_PATCH_GET_CV_FOLDS)
    @patch(_PATCH_LOAD_DATASET)
    @patch(_PATCH_GET_ENCODING)
    def test_all_folds_fail(
        self,
        mock_enc,
        mock_load,
        mock_folds,
        mock_vqc_cls,
        mock_prec,
        mock_recall,
        mock_f1,
        mock_cell,
    ):
        """When all folds fail, result status is 'failed'."""
        mock_enc.return_value = MagicMock()
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        y = np.array([0, 1])
        mock_load.return_value = (X, y)
        mock_folds.return_value = [
            (X[:1], X[1:], y[:1], y[1:]),
        ]
        mock_vqc_cls.return_value.fit.side_effect = RuntimeError("PennyLane error")

        result = _evaluate_cell(
            mock_cell,
            n_runs=1,
            n_folds=1,
            epochs=5,
            base_seed=42,
        )

        assert result["status"] == "failed"
        assert result["mean_accuracy"] is None
        assert result["n_observations"] == 0
        assert result["runs"][0]["folds"][0]["status"] == "failed"

    @patch(_PATCH_F1, return_value=0.5)
    @patch(_PATCH_RECALL, return_value=0.5)
    @patch(_PATCH_PRECISION, return_value=0.5)
    @patch(_PATCH_CI_METRICS)
    @patch(_PATCH_VQC_CLS)
    @patch(_PATCH_GET_CV_FOLDS)
    @patch(_PATCH_LOAD_DATASET)
    @patch(_PATCH_GET_ENCODING)
    def test_partial_failure(
        self,
        mock_enc,
        mock_load,
        mock_folds,
        mock_vqc_cls,
        mock_ci,
        mock_prec,
        mock_recall,
        mock_f1,
        mock_cell,
    ):
        """When some folds fail, result still succeeds with reduced observations."""
        mock_enc.return_value = MagicMock()
        X = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        y = np.array([0, 0, 1, 1])
        mock_load.return_value = (X, y)
        mock_folds.return_value = [
            (X[:2], X[2:], y[:2], y[2:]),
            (X[2:], X[:2], y[2:], y[:2]),
        ]

        call_count = [0]

        def side_effect_fit(X_train, y_train):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("First fold fails")
            return mock_vqc_cls.return_value

        vqc_instance = _make_vqc_mock()
        vqc_instance.fit.side_effect = side_effect_fit
        mock_vqc_cls.return_value = vqc_instance

        mock_ci.return_value = {
            "mean": 0.75,
            "std": 0.0,
            "ci_lower": 0.75,
            "ci_upper": 0.75,
        }

        result = _evaluate_cell(
            mock_cell,
            n_runs=1,
            n_folds=2,
            epochs=5,
            base_seed=42,
        )

        assert result["status"] == "success"
        assert result["n_observations"] == 1  # Only 1 of 2 folds succeeded
        folds = result["runs"][0]["folds"]
        assert folds[0]["status"] == "failed"
        assert folds[1]["status"] == "success"

    @patch(_PATCH_F1, return_value=0.82)
    @patch(_PATCH_RECALL, return_value=0.80)
    @patch(_PATCH_PRECISION, return_value=0.85)
    @patch(_PATCH_CI_METRICS)
    @patch(_PATCH_VQC_CLS)
    @patch(_PATCH_GET_CV_FOLDS)
    @patch(_PATCH_LOAD_DATASET)
    @patch(_PATCH_GET_ENCODING)
    def test_wall_time_recorded(
        self,
        mock_enc,
        mock_load,
        mock_folds,
        mock_vqc_cls,
        mock_ci,
        mock_prec,
        mock_recall,
        mock_f1,
        mock_cell,
    ):
        """Wall time is recorded for each fold and the overall cell."""
        mock_enc.return_value = MagicMock()
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        y = np.array([0, 1])
        mock_load.return_value = (X, y)
        mock_folds.return_value = [
            (X[:1], X[1:], y[:1], y[1:]),
        ]
        mock_vqc_cls.return_value = _make_vqc_mock()
        mock_ci.return_value = {
            "mean": 0.85,
            "std": 0.0,
            "ci_lower": 0.85,
            "ci_upper": 0.85,
        }

        result = _evaluate_cell(
            mock_cell,
            n_runs=1,
            n_folds=1,
            epochs=1,
            base_seed=42,
        )

        assert "wall_time_seconds" in result
        assert result["wall_time_seconds"] >= 0
        assert "wall_time_seconds" in result["runs"][0]["folds"][0]

    @patch(_PATCH_F1, return_value=0.82)
    @patch(_PATCH_RECALL, return_value=0.80)
    @patch(_PATCH_PRECISION, return_value=0.85)
    @patch(_PATCH_CI_METRICS)
    @patch(_PATCH_VQC_CLS)
    @patch(_PATCH_GET_CV_FOLDS)
    @patch(_PATCH_LOAD_DATASET)
    @patch(_PATCH_GET_ENCODING)
    def test_multiple_runs(
        self,
        mock_enc,
        mock_load,
        mock_folds,
        mock_vqc_cls,
        mock_ci,
        mock_prec,
        mock_recall,
        mock_f1,
        mock_cell,
    ):
        """Multiple runs accumulate fold results correctly."""
        mock_enc.return_value = MagicMock()
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        y = np.array([0, 1])
        mock_load.return_value = (X, y)
        mock_folds.return_value = [
            (X[:1], X[1:], y[:1], y[1:]),
        ]
        mock_vqc_cls.return_value = _make_vqc_mock()
        mock_ci.return_value = {
            "mean": 0.85,
            "std": 0.02,
            "ci_lower": 0.83,
            "ci_upper": 0.87,
        }

        result = _evaluate_cell(
            mock_cell,
            n_runs=3,
            n_folds=1,
            epochs=1,
            base_seed=42,
        )

        assert len(result["runs"]) == 3
        assert result["n_observations"] == 3  # 3 runs * 1 fold

    @patch(_PATCH_F1, return_value=0.0)
    @patch(_PATCH_RECALL, return_value=0.0)
    @patch(_PATCH_PRECISION, return_value=0.0)
    @patch(_PATCH_CI_METRICS)
    @patch(_PATCH_VQC_CLS)
    @patch(_PATCH_GET_CV_FOLDS)
    @patch(_PATCH_LOAD_DATASET)
    @patch(_PATCH_GET_ENCODING)
    def test_diverged_vqc_recorded(
        self,
        mock_enc,
        mock_load,
        mock_folds,
        mock_vqc_cls,
        mock_ci,
        mock_prec,
        mock_recall,
        mock_f1,
        mock_cell,
    ):
        """A VQC that diverges is recorded with status 'diverged'."""
        mock_enc.return_value = MagicMock()
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        y = np.array([0, 1])
        mock_load.return_value = (X, y)
        mock_folds.return_value = [
            (X[:1], X[1:], y[:1], y[1:]),
        ]
        mock_vqc_cls.return_value = _make_vqc_mock(
            status="diverged",
            final_loss=11.0,
        )
        mock_ci.return_value = {
            "mean": 0.5,
            "std": 0.0,
            "ci_lower": 0.5,
            "ci_upper": 0.5,
        }

        result = _evaluate_cell(
            mock_cell,
            n_runs=1,
            n_folds=1,
            epochs=5,
            base_seed=42,
        )

        fold_result = result["runs"][0]["folds"][0]
        assert fold_result["status"] == "diverged"
        assert fold_result["final_loss"] == 11.0

    @patch(_PATCH_F1, return_value=0.82)
    @patch(_PATCH_RECALL, return_value=0.80)
    @patch(_PATCH_PRECISION, return_value=0.85)
    @patch(_PATCH_CI_METRICS)
    @patch(_PATCH_VQC_CLS)
    @patch(_PATCH_GET_CV_FOLDS)
    @patch(_PATCH_LOAD_DATASET)
    @patch(_PATCH_GET_ENCODING)
    def test_encoding_instantiated_correctly(
        self,
        mock_enc,
        mock_load,
        mock_folds,
        mock_vqc_cls,
        mock_ci,
        mock_prec,
        mock_recall,
        mock_f1,
        mock_cell,
    ):
        """get_encoding is called with the cell's encoding name and params."""
        mock_enc.return_value = MagicMock()
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        y = np.array([0, 1])
        mock_load.return_value = (X, y)
        mock_folds.return_value = [
            (X[:1], X[1:], y[:1], y[1:]),
        ]
        mock_vqc_cls.return_value = _make_vqc_mock()
        mock_ci.return_value = {
            "mean": 0.85,
            "std": 0.0,
            "ci_lower": 0.85,
            "ci_upper": 0.85,
        }

        _evaluate_cell(
            mock_cell,
            n_runs=1,
            n_folds=1,
            epochs=1,
            base_seed=42,
        )

        mock_enc.assert_called_once_with(
            "angle",
            n_features=2,
            rotation="Y",
        )

    @patch(_PATCH_F1, return_value=0.82)
    @patch(_PATCH_RECALL, return_value=0.80)
    @patch(_PATCH_PRECISION, return_value=0.85)
    @patch(_PATCH_CI_METRICS)
    @patch(_PATCH_VQC_CLS)
    @patch(_PATCH_GET_CV_FOLDS)
    @patch(_PATCH_LOAD_DATASET)
    @patch(_PATCH_GET_ENCODING)
    def test_vqc_seed_passed_correctly(
        self,
        mock_enc,
        mock_load,
        mock_folds,
        mock_vqc_cls,
        mock_ci,
        mock_prec,
        mock_recall,
        mock_f1,
        mock_cell,
    ):
        """VQC is constructed with the correct fold-level seed."""
        mock_enc.return_value = MagicMock()
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        y = np.array([0, 1])
        mock_load.return_value = (X, y)
        mock_folds.return_value = [
            (X[:1], X[1:], y[:1], y[1:]),
        ]
        mock_vqc_cls.return_value = _make_vqc_mock()
        mock_ci.return_value = {
            "mean": 0.85,
            "std": 0.0,
            "ci_lower": 0.85,
            "ci_upper": 0.85,
        }

        _evaluate_cell(
            mock_cell,
            n_runs=1,
            n_folds=1,
            epochs=10,
            base_seed=42,
        )

        # Verify VQCClassifier was instantiated with expected seed.
        call_kwargs = mock_vqc_cls.call_args[1]
        expected_run_seed = _vqc_run_seed(42, 0, 0)
        expected_fold_seed = _fold_init_seed(expected_run_seed, 0)
        assert call_kwargs["seed"] == expected_fold_seed
        assert call_kwargs["lr"] == 0.01
        assert call_kwargs["n_var_layers"] == 2
        assert call_kwargs["epochs"] == 10


# =============================================================================
# Analysis Summary Tests
# =============================================================================


class TestComputeAnalysis:
    """Tests for _compute_analysis()."""

    def test_best_config_identified(self, sample_results):
        """Best config is identified correctly for each encoding/dataset."""
        analysis = _compute_analysis(sample_results)
        best = analysis["best_config_per_encoding_dataset"]

        # With our fixture data: lr=0.01 and layers=2 is best.
        for key in best:
            assert best[key]["lr"] == 0.01
            assert best[key]["n_var_layers"] == 2

    def test_lr_sensitivity_computed(self, sample_results):
        """LR sensitivity contains all learning rates for each group."""
        analysis = _compute_analysis(sample_results)
        lr_sens = analysis["lr_sensitivity"]

        for key in lr_sens:
            assert set(lr_sens[key].keys()) == {"0.001", "0.01", "0.05"}
            # LR 0.01 should be highest.
            assert lr_sens[key]["0.01"] >= lr_sens[key]["0.001"]
            assert lr_sens[key]["0.01"] >= lr_sens[key]["0.05"]

    def test_layer_sensitivity_computed(self, sample_results):
        """Layer sensitivity contains all layer counts for each group."""
        analysis = _compute_analysis(sample_results)
        layer_sens = analysis["layer_sensitivity"]

        for key in layer_sens:
            assert set(layer_sens[key].keys()) == {"1", "2", "3"}
            # Layers=2 should be highest.
            assert layer_sens[key]["2"] >= layer_sens[key]["1"]
            assert layer_sens[key]["2"] >= layer_sens[key]["3"]

    def test_default_config_rank(self, sample_results):
        """Default config (lr=0.01, layers=2) is ranked for each group."""
        analysis = _compute_analysis(sample_results)
        ranks = analysis["default_config_rank"]

        for key in ranks:
            assert ranks[key]["rank"] == 1  # It's the best in our fixture
            assert ranks[key]["total_configs"] == 9  # 3 LRs * 3 layers

    def test_all_encoding_dataset_groups_present(self, sample_results):
        """All 4 groups (2 encodings * 2 datasets) are present."""
        analysis = _compute_analysis(sample_results)
        best = analysis["best_config_per_encoding_dataset"]
        expected_keys = {"angle/moons", "angle/circles", "iqp/moons", "iqp/circles"}
        assert set(best.keys()) == expected_keys

    def test_empty_results(self):
        """Empty results produce empty analysis."""
        analysis = _compute_analysis([])
        assert analysis["best_config_per_encoding_dataset"] == {}
        assert analysis["lr_sensitivity"] == {}
        assert analysis["layer_sensitivity"] == {}
        assert analysis["default_config_rank"] == {}

    def test_failed_results_excluded(self):
        """Results with mean_accuracy=None are excluded from analysis."""
        results = [
            {
                "encoding": "angle",
                "dataset": "moons",
                "lr": 0.01,
                "n_var_layers": 2,
                "mean_accuracy": None,
                "status": "failed",
            },
            {
                "encoding": "angle",
                "dataset": "moons",
                "lr": 0.05,
                "n_var_layers": 2,
                "mean_accuracy": 0.8,
                "status": "success",
            },
        ]
        analysis = _compute_analysis(results)
        best = analysis["best_config_per_encoding_dataset"]
        assert "angle/moons" in best
        assert best["angle/moons"]["lr"] == 0.05

    def test_single_config_per_group(self):
        """Single config per group gets rank 1."""
        results = [
            {
                "encoding": "iqp",
                "dataset": "circles",
                "lr": 0.01,
                "n_var_layers": 2,
                "mean_accuracy": 0.9,
            }
        ]
        analysis = _compute_analysis(results)
        rank = analysis["default_config_rank"]["iqp/circles"]
        assert rank["rank"] == 1
        assert rank["total_configs"] == 1

    def test_default_config_missing_from_grid(self):
        """When default config (lr=0.01, layers=2) is not in grid."""
        results = [
            {
                "encoding": "angle",
                "dataset": "moons",
                "lr": 0.05,
                "n_var_layers": 3,
                "mean_accuracy": 0.8,
            }
        ]
        analysis = _compute_analysis(results)
        rank = analysis["default_config_rank"].get("angle/moons", {})
        assert rank.get("rank") is None

    def test_analysis_with_tied_accuracies(self):
        """Analysis handles tied mean accuracies gracefully."""
        results = [
            {
                "encoding": "angle",
                "dataset": "moons",
                "lr": 0.01,
                "n_var_layers": 2,
                "mean_accuracy": 0.85,
            },
            {
                "encoding": "angle",
                "dataset": "moons",
                "lr": 0.05,
                "n_var_layers": 2,
                "mean_accuracy": 0.85,
            },
        ]
        analysis = _compute_analysis(results)
        best = analysis["best_config_per_encoding_dataset"]["angle/moons"]
        assert best["mean_accuracy"] == 0.85

    def test_analysis_lr_sensitivity_averages_over_layers(self):
        """LR sensitivity correctly averages over layer counts."""
        results = [
            {
                "encoding": "angle",
                "dataset": "moons",
                "lr": 0.01,
                "n_var_layers": 1,
                "mean_accuracy": 0.80,
            },
            {
                "encoding": "angle",
                "dataset": "moons",
                "lr": 0.01,
                "n_var_layers": 2,
                "mean_accuracy": 0.90,
            },
        ]
        analysis = _compute_analysis(results)
        lr_sens = analysis["lr_sensitivity"]["angle/moons"]
        expected_avg = round((0.80 + 0.90) / 2, 6)
        assert lr_sens["0.01"] == expected_avg


# =============================================================================
# Checkpoint Integration Tests
# =============================================================================


class TestCheckpointIntegration:
    """Tests for checkpoint-based resume in run_sensitivity_analysis.

    ``CheckpointManager`` is imported lazily, so we patch at
    ``experiments.checkpoint.CheckpointManager``.
    """

    @patch("experiments.vqc_sensitivity._evaluate_cell")
    @patch(_PATCH_CKPT_MGR)
    def test_completed_cells_skipped(
        self,
        mock_ckpt_cls,
        mock_eval,
        tmp_output_dir,
    ):
        """Completed cells are loaded from checkpoint and not re-evaluated."""
        cached_result = {
            "encoding": "angle",
            "dataset": "moons",
            "lr": 0.01,
            "n_var_layers": 2,
            "mean_accuracy": 0.85,
            "status": "success",
        }
        mock_ckpt = MagicMock()
        mock_ckpt.is_completed.return_value = True
        mock_ckpt.get_result.return_value = cached_result
        mock_ckpt_cls.return_value = mock_ckpt

        report = run_sensitivity_analysis(
            output_dir=tmp_output_dir,
            learning_rates=[0.01],
            n_var_layers=[2],
            encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
            datasets=["moons"],
        )

        mock_eval.assert_not_called()
        assert report["summary"]["skipped"] == 1
        assert report["summary"]["completed"] == 0

    @patch("experiments.vqc_sensitivity._evaluate_cell")
    @patch(_PATCH_CKPT_MGR)
    def test_new_cells_evaluated_and_checkpointed(
        self,
        mock_ckpt_cls,
        mock_eval,
        tmp_output_dir,
    ):
        """New cells are evaluated and saved to checkpoint."""
        cell_result = {
            "encoding": "angle",
            "dataset": "moons",
            "lr": 0.01,
            "n_var_layers": 2,
            "mean_accuracy": 0.85,
            "status": "success",
        }
        mock_eval.return_value = cell_result

        mock_ckpt = MagicMock()
        mock_ckpt.is_completed.return_value = False
        mock_ckpt_cls.return_value = mock_ckpt

        report = run_sensitivity_analysis(
            output_dir=tmp_output_dir,
            learning_rates=[0.01],
            n_var_layers=[2],
            encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
            datasets=["moons"],
        )

        mock_eval.assert_called_once()
        mock_ckpt.mark_completed.assert_called_once()
        assert report["summary"]["completed"] == 1

    @patch("experiments.vqc_sensitivity._evaluate_cell")
    @patch(_PATCH_CKPT_MGR)
    def test_evaluation_error_caught(
        self,
        mock_ckpt_cls,
        mock_eval,
        tmp_output_dir,
    ):
        """Runtime errors during evaluation are caught, not propagated."""
        mock_eval.side_effect = RuntimeError("Unexpected error")

        mock_ckpt = MagicMock()
        mock_ckpt.is_completed.return_value = False
        mock_ckpt_cls.return_value = mock_ckpt

        report = run_sensitivity_analysis(
            output_dir=tmp_output_dir,
            learning_rates=[0.01],
            n_var_layers=[2],
            encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
            datasets=["moons"],
        )

        assert report["summary"]["failed"] == 1
        assert report["summary"]["completed"] == 0
        assert report["results"][0]["status"] == "failed"
        assert "Unexpected error" in report["results"][0]["reason"]


# =============================================================================
# Report Structure Tests
# =============================================================================


class TestReportStructure:
    """Tests for the output JSON report structure."""

    @patch("experiments.vqc_sensitivity._evaluate_cell")
    @patch(_PATCH_CKPT_MGR)
    def test_schema_version(self, mock_ckpt_cls, mock_eval, tmp_output_dir):
        """Report contains correct schema version."""
        mock_eval.return_value = {
            "encoding": "angle",
            "dataset": "moons",
            "lr": 0.01,
            "n_var_layers": 2,
            "mean_accuracy": 0.85,
            "status": "success",
        }
        mock_ckpt = MagicMock()
        mock_ckpt.is_completed.return_value = False
        mock_ckpt_cls.return_value = mock_ckpt

        report = run_sensitivity_analysis(
            output_dir=tmp_output_dir,
            learning_rates=[0.01],
            n_var_layers=[2],
            encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
            datasets=["moons"],
        )

        assert report["schema_version"] == _SCHEMA_VERSION

    @patch("experiments.vqc_sensitivity._evaluate_cell")
    @patch(_PATCH_CKPT_MGR)
    def test_grid_metadata(self, mock_ckpt_cls, mock_eval, tmp_output_dir):
        """Report contains correct grid metadata."""
        mock_eval.return_value = {
            "encoding": "angle",
            "dataset": "moons",
            "lr": 0.01,
            "n_var_layers": 2,
            "mean_accuracy": 0.85,
            "status": "success",
        }
        mock_ckpt = MagicMock()
        mock_ckpt.is_completed.return_value = False
        mock_ckpt_cls.return_value = mock_ckpt

        report = run_sensitivity_analysis(
            output_dir=tmp_output_dir,
            learning_rates=[0.01],
            n_var_layers=[2],
            encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
            datasets=["moons"],
            n_runs=3,
            n_folds=5,
            epochs=100,
        )

        grid = report["grid"]
        assert grid["learning_rates"] == [0.01]
        assert grid["n_var_layers"] == [2]
        assert grid["encodings"] == ["angle"]
        assert grid["datasets"] == ["moons"]
        assert grid["n_runs"] == 3
        assert grid["n_folds"] == 5
        assert grid["epochs"] == 100

    @patch("experiments.vqc_sensitivity._evaluate_cell")
    @patch(_PATCH_CKPT_MGR)
    def test_report_has_analysis_section(
        self,
        mock_ckpt_cls,
        mock_eval,
        tmp_output_dir,
    ):
        """Report contains analysis section with expected keys."""
        mock_eval.return_value = {
            "encoding": "angle",
            "dataset": "moons",
            "lr": 0.01,
            "n_var_layers": 2,
            "mean_accuracy": 0.85,
            "status": "success",
        }
        mock_ckpt = MagicMock()
        mock_ckpt.is_completed.return_value = False
        mock_ckpt_cls.return_value = mock_ckpt

        report = run_sensitivity_analysis(
            output_dir=tmp_output_dir,
            learning_rates=[0.01],
            n_var_layers=[2],
            encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
            datasets=["moons"],
        )

        analysis = report["analysis"]
        assert "best_config_per_encoding_dataset" in analysis
        assert "lr_sensitivity" in analysis
        assert "layer_sensitivity" in analysis
        assert "default_config_rank" in analysis

    @patch("experiments.vqc_sensitivity._evaluate_cell")
    @patch(_PATCH_CKPT_MGR)
    def test_report_has_summary(
        self,
        mock_ckpt_cls,
        mock_eval,
        tmp_output_dir,
    ):
        """Report contains a summary section."""
        mock_eval.return_value = {
            "encoding": "angle",
            "dataset": "moons",
            "lr": 0.01,
            "n_var_layers": 2,
            "mean_accuracy": 0.85,
            "status": "success",
        }
        mock_ckpt = MagicMock()
        mock_ckpt.is_completed.return_value = False
        mock_ckpt_cls.return_value = mock_ckpt

        report = run_sensitivity_analysis(
            output_dir=tmp_output_dir,
            learning_rates=[0.01],
            n_var_layers=[2],
            encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
            datasets=["moons"],
        )

        summary = report["summary"]
        assert "total_cells" in summary
        assert "completed" in summary
        assert "skipped" in summary
        assert "failed" in summary
        assert "wall_time_seconds" in summary

    @patch("experiments.vqc_sensitivity._evaluate_cell")
    @patch(_PATCH_CKPT_MGR)
    def test_report_saved_to_disk(
        self,
        mock_ckpt_cls,
        mock_eval,
        tmp_output_dir,
    ):
        """Report JSON is saved to the output directory."""
        mock_eval.return_value = {
            "encoding": "angle",
            "dataset": "moons",
            "lr": 0.01,
            "n_var_layers": 2,
            "mean_accuracy": 0.85,
            "status": "success",
        }
        mock_ckpt = MagicMock()
        mock_ckpt.is_completed.return_value = False
        mock_ckpt_cls.return_value = mock_ckpt

        run_sensitivity_analysis(
            output_dir=tmp_output_dir,
            learning_rates=[0.01],
            n_var_layers=[2],
            encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
            datasets=["moons"],
        )

        report_path = os.path.join(tmp_output_dir, "sensitivity_report.json")
        assert os.path.isfile(report_path)

        with open(report_path) as fh:
            saved = json.load(fh)
        assert saved["schema_version"] == _SCHEMA_VERSION

    @patch("experiments.vqc_sensitivity._evaluate_cell")
    @patch(_PATCH_CKPT_MGR)
    def test_runs_stripped_from_summary_results(
        self,
        mock_ckpt_cls,
        mock_eval,
        tmp_output_dir,
    ):
        """Per-run detail is stripped from top-level results."""
        mock_eval.return_value = {
            "encoding": "angle",
            "dataset": "moons",
            "lr": 0.01,
            "n_var_layers": 2,
            "mean_accuracy": 0.85,
            "status": "success",
            "runs": [{"run": 0, "folds": [{"fold": 0}]}],
        }
        mock_ckpt = MagicMock()
        mock_ckpt.is_completed.return_value = False
        mock_ckpt_cls.return_value = mock_ckpt

        report = run_sensitivity_analysis(
            output_dir=tmp_output_dir,
            learning_rates=[0.01],
            n_var_layers=[2],
            encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
            datasets=["moons"],
        )

        for r in report["results"]:
            assert "runs" not in r


# =============================================================================
# Quick Mode Tests
# =============================================================================


class TestQuickMode:
    """Tests for the --quick flag."""

    @patch("experiments.vqc_sensitivity._evaluate_cell")
    @patch(_PATCH_CKPT_MGR)
    def test_quick_mode_reduces_runs_and_folds(
        self,
        mock_ckpt_cls,
        mock_eval,
        tmp_output_dir,
    ):
        """Quick mode uses reduced n_runs and n_folds."""
        mock_eval.return_value = {
            "encoding": "angle",
            "dataset": "moons",
            "lr": 0.01,
            "n_var_layers": 2,
            "mean_accuracy": 0.85,
            "status": "success",
        }
        mock_ckpt = MagicMock()
        mock_ckpt.is_completed.return_value = False
        mock_ckpt_cls.return_value = mock_ckpt

        report = run_sensitivity_analysis(
            output_dir=tmp_output_dir,
            quick=True,
            learning_rates=[0.01],
            n_var_layers=[2],
            encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
            datasets=["moons"],
        )

        assert report["grid"]["n_runs"] == _QUICK_N_RUNS
        assert report["grid"]["n_folds"] == _QUICK_N_FOLDS

        call_kwargs = mock_eval.call_args[1]
        assert call_kwargs["n_runs"] == _QUICK_N_RUNS
        assert call_kwargs["n_folds"] == _QUICK_N_FOLDS


# =============================================================================
# CLI Tests
# =============================================================================


class TestCLI:
    """Tests for the CLI entry point (main function)."""

    @patch("experiments.vqc_sensitivity.run_sensitivity_analysis")
    def test_default_args(self, mock_run):
        """CLI with no args uses default output dir."""
        mock_run.return_value = {
            "summary": {
                "completed": 1,
                "skipped": 0,
                "failed": 0,
                "wall_time_seconds": 10.0,
            },
            "analysis": {
                "best_config_per_encoding_dataset": {},
                "default_config_rank": {},
            },
        }

        exit_code = main([])

        assert exit_code == 0
        call_kwargs = mock_run.call_args[1]
        assert (
            call_kwargs["output_dir"] == "experiments/results/raw/stage6a5_sensitivity"
        )
        assert call_kwargs["quick"] is False

    @patch("experiments.vqc_sensitivity.run_sensitivity_analysis")
    def test_quick_flag(self, mock_run):
        """CLI --quick flag is passed through."""
        mock_run.return_value = {
            "summary": {
                "completed": 1,
                "skipped": 0,
                "failed": 0,
                "wall_time_seconds": 1.0,
            },
            "analysis": {
                "best_config_per_encoding_dataset": {},
                "default_config_rank": {},
            },
        }

        exit_code = main(["--quick"])

        assert exit_code == 0
        assert mock_run.call_args[1]["quick"] is True

    @patch("experiments.vqc_sensitivity.run_sensitivity_analysis")
    def test_custom_output_dir(self, mock_run):
        """CLI --output-dir is forwarded."""
        mock_run.return_value = {
            "summary": {
                "completed": 1,
                "skipped": 0,
                "failed": 0,
                "wall_time_seconds": 1.0,
            },
            "analysis": {
                "best_config_per_encoding_dataset": {},
                "default_config_rank": {},
            },
        }

        main(["--output-dir", "/tmp/custom_dir"])

        assert mock_run.call_args[1]["output_dir"] == "/tmp/custom_dir"

    @patch("experiments.vqc_sensitivity.run_sensitivity_analysis")
    def test_exit_code_1_on_partial_failure(self, mock_run):
        """CLI returns 1 when some cells fail."""
        mock_run.return_value = {
            "summary": {
                "completed": 5,
                "skipped": 0,
                "failed": 2,
                "wall_time_seconds": 50.0,
            },
            "analysis": {
                "best_config_per_encoding_dataset": {},
                "default_config_rank": {},
            },
        }

        exit_code = main([])

        assert exit_code == 1

    @patch("experiments.vqc_sensitivity.run_sensitivity_analysis")
    def test_exit_code_2_on_fatal_error(self, mock_run):
        """CLI returns 2 on fatal exception."""
        mock_run.side_effect = RuntimeError("Fatal error")

        exit_code = main([])

        assert exit_code == 2

    @patch("experiments.vqc_sensitivity.run_sensitivity_analysis")
    def test_log_level_debug(self, mock_run):
        """CLI --log-level DEBUG configures logging."""
        mock_run.return_value = {
            "summary": {
                "completed": 1,
                "skipped": 0,
                "failed": 0,
                "wall_time_seconds": 1.0,
            },
            "analysis": {
                "best_config_per_encoding_dataset": {},
                "default_config_rank": {},
            },
        }

        exit_code = main(["--log-level", "DEBUG"])

        assert exit_code == 0


# =============================================================================
# JSON Serialization Tests
# =============================================================================


class TestJsonDefault:
    """Tests for the _json_default fallback serializer."""

    def test_numpy_int(self):
        """Numpy integers are serialized to Python int."""
        assert _json_default(np.int64(42)) == 42
        assert isinstance(_json_default(np.int64(42)), int)

    def test_numpy_float(self):
        """Numpy floats are serialized to Python float."""
        assert _json_default(np.float64(3.14)) == pytest.approx(3.14)
        assert isinstance(_json_default(np.float64(3.14)), float)

    def test_numpy_array(self):
        """Numpy arrays are serialized to Python lists."""
        result = _json_default(np.array([1, 2, 3]))
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_numpy_bool(self):
        """Numpy booleans are serialized to Python bool."""
        val = np.bool_(True)
        result = _json_default(val)
        assert result is True
        assert isinstance(result, bool)

    def test_unsupported_type_raises(self):
        """Unsupported types raise TypeError."""
        with pytest.raises(TypeError, match="not JSON serializable"):
            _json_default(object())

    def test_full_report_json_serializable(self, sample_results):
        """A report with numpy types can be serialized via _json_default."""
        sample_results[0]["mean_accuracy"] = np.float64(0.85)
        sample_results[0]["n_observations"] = np.int64(15)

        report = {
            "schema_version": _SCHEMA_VERSION,
            "results": sample_results[:2],
            "analysis": _compute_analysis(sample_results[:2]),
        }

        json_str = json.dumps(report, default=_json_default)
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["schema_version"] == _SCHEMA_VERSION


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_default_parameters(self):
        """Default parameters match specification."""
        assert _DEFAULT_BASE_SEED == 42
        assert _DEFAULT_N_RUNS == 3
        assert _DEFAULT_N_FOLDS == 5
        assert _DEFAULT_EPOCHS == 100

    def test_quick_parameters(self):
        """Quick mode parameters are reduced."""
        assert _QUICK_N_RUNS < _DEFAULT_N_RUNS
        assert _QUICK_N_FOLDS < _DEFAULT_N_FOLDS

    def test_learning_rates(self):
        """Learning rates match specification."""
        assert _LEARNING_RATES == [0.001, 0.01, 0.05]

    def test_var_layers(self):
        """Layer counts match specification."""
        assert _N_VAR_LAYERS == [1, 2, 3]

    def test_datasets(self):
        """Datasets match specification."""
        assert _DATASETS == ["moons", "circles"]

    def test_encoding_configs(self):
        """Encoding configs match specification."""
        assert len(_ENCODING_CONFIGS) == 2
        names = [c["name"] for c in _ENCODING_CONFIGS]
        assert "angle" in names
        assert "iqp" in names

    def test_seed_offset(self):
        """Seed offset is 9000."""
        assert _SEED_OFFSET == 9000


# =============================================================================
# Integration-like Tests (with real CheckpointManager, mocked VQC)
# =============================================================================


class TestIntegrationWithCheckpoint:
    """Tests using real CheckpointManager but mocked VQC evaluation."""

    @patch("experiments.vqc_sensitivity._evaluate_cell")
    def test_resume_after_interruption(self, mock_eval, tmp_output_dir):
        """After interruption, completed cells are not re-evaluated."""
        cell_result = {
            "encoding": "angle",
            "dataset": "moons",
            "lr": 0.01,
            "n_var_layers": 2,
            "mean_accuracy": 0.85,
            "status": "success",
        }
        mock_eval.return_value = cell_result

        # First run: evaluate 1 cell.
        report1 = run_sensitivity_analysis(
            output_dir=tmp_output_dir,
            learning_rates=[0.01],
            n_var_layers=[2],
            encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
            datasets=["moons"],
        )
        assert report1["summary"]["completed"] == 1
        assert mock_eval.call_count == 1

        # Second run: cell should be skipped (from checkpoint).
        mock_eval.reset_mock()
        report2 = run_sensitivity_analysis(
            output_dir=tmp_output_dir,
            learning_rates=[0.01],
            n_var_layers=[2],
            encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
            datasets=["moons"],
        )
        assert report2["summary"]["skipped"] == 1
        assert report2["summary"]["completed"] == 0
        mock_eval.assert_not_called()

    @patch("experiments.vqc_sensitivity._evaluate_cell")
    def test_partial_grid_resume(self, mock_eval, tmp_output_dir):
        """After completing some cells, new cells are evaluated on resume."""
        call_count = [0]

        def eval_side_effect(cell, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "encoding": cell["encoding_name"],
                    "dataset": cell["dataset"],
                    "lr": cell["lr"],
                    "n_var_layers": cell["n_var_layers"],
                    "mean_accuracy": 0.85,
                    "status": "success",
                }
            # Simulate crash on second cell.
            raise KeyboardInterrupt("Simulated interruption")

        mock_eval.side_effect = eval_side_effect

        # First run: 1st cell succeeds, 2nd crashes.
        with pytest.raises(KeyboardInterrupt):
            run_sensitivity_analysis(
                output_dir=tmp_output_dir,
                learning_rates=[0.01],
                n_var_layers=[2],
                encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
                datasets=["moons", "circles"],
            )

        # Reset mock for resume.
        mock_eval.side_effect = lambda cell, **kwargs: {
            "encoding": cell["encoding_name"],
            "dataset": cell["dataset"],
            "lr": cell["lr"],
            "n_var_layers": cell["n_var_layers"],
            "mean_accuracy": 0.80,
            "status": "success",
        }

        report = run_sensitivity_analysis(
            output_dir=tmp_output_dir,
            learning_rates=[0.01],
            n_var_layers=[2],
            encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
            datasets=["moons", "circles"],
        )

        # First cell skipped (from checkpoint), second cell evaluated.
        assert report["summary"]["skipped"] == 1
        assert report["summary"]["completed"] == 1


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_cell_id_with_small_lr(self):
        """Cell ID handles very small learning rates."""
        cell = {
            "encoding_name": "angle",
            "dataset": "moons",
            "lr": 0.001,
            "n_var_layers": 1,
        }
        cid = _cell_id(cell)
        assert "lr0.001" in cid

    def test_cell_id_with_different_datasets(self):
        """Cell IDs differ across datasets."""
        cell_a = {
            "encoding_name": "angle",
            "dataset": "moons",
            "lr": 0.01,
            "n_var_layers": 2,
        }
        cell_b = {
            "encoding_name": "angle",
            "dataset": "circles",
            "lr": 0.01,
            "n_var_layers": 2,
        }
        assert _cell_id(cell_a) != _cell_id(cell_b)

    def test_fold_init_seed_no_overflow(self):
        """Fold init seed doesn't overflow with max parameters."""
        run_seed = _vqc_run_seed(42, cell_index=35, run_index=2)
        fold_seed = _fold_init_seed(run_seed, fold_index=4)
        assert isinstance(fold_seed, int)
        assert fold_seed > 0

    @patch("experiments.vqc_sensitivity._evaluate_cell")
    @patch(_PATCH_CKPT_MGR)
    def test_report_with_all_failures(
        self,
        mock_ckpt_cls,
        mock_eval,
        tmp_output_dir,
    ):
        """Report is valid even when all cells fail."""
        mock_eval.side_effect = ValueError("Total failure")
        mock_ckpt = MagicMock()
        mock_ckpt.is_completed.return_value = False
        mock_ckpt_cls.return_value = mock_ckpt

        report = run_sensitivity_analysis(
            output_dir=tmp_output_dir,
            learning_rates=[0.01],
            n_var_layers=[2],
            encoding_configs=[{"name": "angle", "params": {"n_features": 2}}],
            datasets=["moons"],
        )

        assert report["summary"]["failed"] == 1
        assert report["summary"]["completed"] == 0
        assert report["analysis"]["best_config_per_encoding_dataset"] == {}
