"""Smoke tests for Stage 1: Resource Profiling.

These tests verify the ``_handle_resources`` handler and the end-to-end
runner pipeline on representative encodings without running the full
170-task sweep. They cover:

- Handler output structure and required fields
- Normal encodings (protocol path)
- Data-dependent encodings (BasisEncoding fallback path)
- Non-entangling vs entangling encodings
- Equivariant encodings with non-standard params (SO2, Swap)
- The new HigherOrderAngle order=3 split
- End-to-end runner with checkpointing and resume
- Output field type and value constraints

Run with: pytest experiments/tests/test_stage1_smoke.py -v
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest

from encoding_atlas import get_encoding
from experiments.config import ExperimentConfig, load_config
from experiments.runner import _handle_resources


# =============================================================================
# Helpers
# =============================================================================

_CONFIGS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, "configs")
)

_STAGE1_CONFIG = os.path.join(_CONFIGS_DIR, "stage1_resources.json")


def _make_dummy_config() -> ExperimentConfig:
    """Load the real Stage 1 config for use in handler calls."""
    return load_config(_STAGE1_CONFIG)


# Fields that every successful _handle_resources result MUST contain.
_REQUIRED_SUMMARY_FIELDS = {
    "n_qubits",
    "depth",
    "gate_count",
    "single_qubit_gates",
    "two_qubit_gates",
    "parameter_count",
    "estimated_time_us",
    "encoding_name",
}


def _assert_valid_resource_result(result: dict[str, Any], *, encoding_name: str) -> None:
    """Assert that *result* has the correct structure and sane values."""
    missing = _REQUIRED_SUMMARY_FIELDS - result.keys()
    assert not missing, f"Missing fields for {encoding_name}: {missing}"

    assert isinstance(result["n_qubits"], int)
    assert result["n_qubits"] >= 1

    assert isinstance(result["depth"], int)
    assert result["depth"] >= 0

    assert isinstance(result["gate_count"], int)
    assert result["gate_count"] >= 0

    assert isinstance(result["single_qubit_gates"], int)
    assert isinstance(result["two_qubit_gates"], int)
    assert result["single_qubit_gates"] + result["two_qubit_gates"] >= result["gate_count"] or \
           result["gate_count"] == result["single_qubit_gates"] + result["two_qubit_gates"]

    assert isinstance(result["parameter_count"], int)
    assert result["parameter_count"] >= 0

    assert isinstance(result["estimated_time_us"], (int, float))
    assert result["estimated_time_us"] >= 0

    assert isinstance(result["encoding_name"], str)
    assert len(result["encoding_name"]) > 0


# =============================================================================
# Handler: normal encodings (ResourceAnalyzable protocol path)
# =============================================================================


class TestHandleResourcesNormal:
    """Test _handle_resources on encodings that implement ResourceAnalyzable."""

    config = _make_dummy_config()

    @pytest.mark.parametrize("name, params", [
        ("angle", {"n_features": 4}),
        ("iqp", {"n_features": 4, "reps": 2, "entanglement": "full"}),
        ("zz_feature_map", {"n_features": 4, "reps": 1, "entanglement": "linear"}),
        ("data_reuploading", {"n_features": 4, "n_layers": 2}),
        ("hardware_efficient", {"n_features": 4, "reps": 2}),
    ])
    def test_returns_valid_result(self, name: str, params: dict):
        enc = get_encoding(name, **params)
        result = _handle_resources(enc, self.config, seed=1042)
        _assert_valid_resource_result(result, encoding_name=name)

    def test_entangling_encoding_has_two_qubit_gates(self):
        enc = get_encoding("iqp", n_features=4, reps=2, entanglement="full")
        result = _handle_resources(enc, self.config, seed=1042)
        assert result["two_qubit_gates"] > 0

    def test_non_entangling_encoding_has_no_two_qubit_gates(self):
        enc = get_encoding("angle", n_features=4)
        result = _handle_resources(enc, self.config, seed=1042)
        assert result["two_qubit_gates"] == 0

    def test_more_reps_means_more_gates(self):
        enc1 = get_encoding("iqp", n_features=4, reps=1, entanglement="full")
        enc2 = get_encoding("iqp", n_features=4, reps=2, entanglement="full")
        r1 = _handle_resources(enc1, self.config, seed=1042)
        r2 = _handle_resources(enc2, self.config, seed=1042)
        assert r2["gate_count"] > r1["gate_count"]
        assert r2["depth"] > r1["depth"]


# =============================================================================
# Handler: data-dependent encoding (BasisEncoding fallback path)
# =============================================================================


class TestHandleResourcesDataDependent:
    """Test _handle_resources on BasisEncoding (properties fallback)."""

    config = _make_dummy_config()

    def test_basis_encoding_returns_valid_result(self):
        enc = get_encoding("basis", n_features=4)
        result = _handle_resources(enc, self.config, seed=1042)
        _assert_valid_resource_result(result, encoding_name="basis")

    def test_basis_encoding_marked_data_dependent(self):
        enc = get_encoding("basis", n_features=4)
        result = _handle_resources(enc, self.config, seed=1042)
        assert result.get("is_data_dependent") is True

    def test_basis_encoding_has_note(self):
        enc = get_encoding("basis", n_features=4)
        result = _handle_resources(enc, self.config, seed=1042)
        assert "note" in result
        assert "worst-case" in result["note"].lower()


# =============================================================================
# Handler: equivariant encodings with non-standard params
# =============================================================================


class TestHandleResourcesEquivariant:
    """Test _handle_resources on equivariant encodings."""

    config = _make_dummy_config()

    def test_so2_equivariant(self):
        enc = get_encoding("so2_equivariant", n_features=2, max_angular_momentum=2)
        result = _handle_resources(enc, self.config, seed=1042)
        _assert_valid_resource_result(result, encoding_name="so2_equivariant")
        assert result["n_qubits"] >= 2

    def test_cyclic_equivariant(self):
        enc = get_encoding("cyclic_equivariant", n_features=4, reps=2)
        result = _handle_resources(enc, self.config, seed=1042)
        _assert_valid_resource_result(result, encoding_name="cyclic_equivariant")

    def test_swap_equivariant_even_features(self):
        enc = get_encoding("swap_equivariant", n_features=4, reps=2)
        result = _handle_resources(enc, self.config, seed=1042)
        _assert_valid_resource_result(result, encoding_name="swap_equivariant")


# =============================================================================
# Handler: HigherOrderAngle order=3 split
# =============================================================================


class TestHandleResourcesHigherOrderAngle:
    """Test the HigherOrderAngle order=3 configurations added by the split."""

    config = _make_dummy_config()

    @pytest.mark.parametrize("n_features", [4, 6, 8])
    def test_order_3_valid_for_large_n_features(self, n_features: int):
        enc = get_encoding("higher_order_angle", n_features=n_features, order=3)
        result = _handle_resources(enc, self.config, seed=1042)
        _assert_valid_resource_result(result, encoding_name="higher_order_angle")

    def test_order_3_invalid_for_n_features_2(self):
        """order=3 with n_features=2 must fail at encoding instantiation."""
        with pytest.raises(ValueError, match="cannot exceed n_features"):
            get_encoding("higher_order_angle", n_features=2, order=3)

    @pytest.mark.parametrize("order", [1, 2, 3])
    def test_all_orders_at_n_features_4(self, order: int):
        """All three orders work at n_features=4."""
        enc = get_encoding("higher_order_angle", n_features=4, order=order)
        result = _handle_resources(enc, self.config, seed=1042)
        _assert_valid_resource_result(result, encoding_name="higher_order_angle")


# =============================================================================
# Handler: all 16 encodings at default config
# =============================================================================


class TestHandleResourcesAllEncodings:
    """Smoke test: _handle_resources succeeds for every encoding family."""

    config = _make_dummy_config()

    # One representative configuration per encoding.
    _REPRESENTATIVE_ENCODINGS = [
        ("angle", {"n_features": 4}),
        ("amplitude", {"n_features": 4}),
        ("basis", {"n_features": 4}),
        ("iqp", {"n_features": 4, "reps": 1}),
        ("zz_feature_map", {"n_features": 4, "reps": 1}),
        ("pauli_feature_map", {"n_features": 4, "reps": 1}),
        ("data_reuploading", {"n_features": 4, "n_layers": 1}),
        ("hardware_efficient", {"n_features": 4, "reps": 1}),
        ("higher_order_angle", {"n_features": 4, "order": 2}),
        ("qaoa_encoding", {"n_features": 4, "reps": 1}),
        ("hamiltonian_encoding", {"n_features": 4, "reps": 1}),
        ("symmetry_inspired", {"n_features": 4, "reps": 1}),
        ("trainable_encoding", {"n_features": 4, "n_layers": 1}),
        ("so2_equivariant", {"n_features": 2, "max_angular_momentum": 1}),
        ("cyclic_equivariant", {"n_features": 4, "reps": 1}),
        ("swap_equivariant", {"n_features": 4, "reps": 1}),
    ]

    @pytest.mark.parametrize(
        "name, params",
        _REPRESENTATIVE_ENCODINGS,
        ids=[e[0] for e in _REPRESENTATIVE_ENCODINGS],
    )
    def test_all_16_encodings(self, name: str, params: dict):
        enc = get_encoding(name, **params)
        result = _handle_resources(enc, self.config, seed=1042)
        _assert_valid_resource_result(result, encoding_name=name)


# =============================================================================
# End-to-end: runner with checkpointing
# =============================================================================


class TestStage1EndToEnd:
    """Integration test: run Stage 1 on a minimal config through the full
    runner pipeline (config -> instantiation -> handler -> checkpoint)."""

    def _write_mini_config(self, tmp_path) -> str:
        """Write a minimal Stage 1 config with 3 tasks."""
        config = {
            "stage": "resources",
            "seed": 42,
            "backend": "pennylane",
            "encodings": [
                {"name": "angle", "params": {"n_features": [2, 4]}},
                {"name": "iqp", "params": {"n_features": 4, "reps": 1}},
            ],
            "analysis_params": {},
            "output_dir": os.path.join(str(tmp_path), "results"),
        }
        path = os.path.join(str(tmp_path), "mini_stage1.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(config, fh)
        return path

    def test_runner_completes_all_tasks(self, tmp_path):
        from experiments.runner import ExperimentRunner

        config_path = self._write_mini_config(tmp_path)
        runner = ExperimentRunner(config_path, resume=False)
        summary = runner.run()

        assert summary["total"] == 3
        assert summary["completed"] == 3
        assert summary["failed"] == 0
        assert summary["skipped"] == 0

    def test_runner_writes_checkpoint(self, tmp_path):
        from experiments.runner import ExperimentRunner

        config_path = self._write_mini_config(tmp_path)
        runner = ExperimentRunner(config_path, resume=False)
        runner.run()

        ckpt_dir = os.path.join(str(tmp_path), "results", "checkpoints")
        assert os.path.isdir(ckpt_dir)
        log_file = os.path.join(ckpt_dir, "completed.jsonl")
        assert os.path.isfile(log_file)

        with open(log_file, "r", encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]
        assert len(lines) == 3

    def test_runner_resumes_from_checkpoint(self, tmp_path):
        from experiments.runner import ExperimentRunner

        config_path = self._write_mini_config(tmp_path)

        # First run: completes all 3 tasks.
        runner1 = ExperimentRunner(config_path, resume=False)
        runner1.run()

        # Second run with resume=True: all 3 should be skipped.
        runner2 = ExperimentRunner(config_path, resume=True)
        summary = runner2.run()

        assert summary["completed"] == 0
        assert summary["skipped"] == 3

    def test_runner_writes_summary_json(self, tmp_path):
        from experiments.runner import ExperimentRunner

        config_path = self._write_mini_config(tmp_path)
        runner = ExperimentRunner(config_path, resume=False)
        runner.run()

        summary_path = os.path.join(str(tmp_path), "results", "summary.json")
        assert os.path.isfile(summary_path)

        with open(summary_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["total_results"] == 3
        assert data["success_count"] == 3

    def test_runner_result_values_are_sane(self, tmp_path):
        from experiments.runner import ExperimentRunner

        config_path = self._write_mini_config(tmp_path)
        runner = ExperimentRunner(config_path, resume=False)
        runner.run()

        # Verify IQP has more gates than Angle.
        iqp_results = runner.store.filter(encoding_name="iqp", status="success")
        angle_results = runner.store.filter(encoding_name="angle", status="success")

        assert len(iqp_results) == 1
        assert len(angle_results) == 2

        iqp_gates = iqp_results[0].result["gate_count"]
        angle_gates = angle_results[0].result["gate_count"]
        assert iqp_gates > angle_gates

    def test_runner_writes_effective_config(self, tmp_path):
        from experiments.runner import ExperimentRunner

        config_path = self._write_mini_config(tmp_path)
        ExperimentRunner(config_path, resume=False)

        effective_path = os.path.join(str(tmp_path), "results", "effective_config.json")
        assert os.path.isfile(effective_path)

        with open(effective_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["stage"] == "resources"
        assert len(data["encoding_specs"]) == 3


# =============================================================================
# Config: Stage 1 task count verification
# =============================================================================


class TestStage1ConfigIntegrity:
    """Verify that the shipped stage1_resources.json produces exactly
    170 encoding specs and that every spec instantiates."""

    @pytest.fixture
    def stage1_config(self):
        return load_config(_STAGE1_CONFIG)

    def test_total_task_count(self, stage1_config):
        assert len(stage1_config.encoding_specs) == 170

    def test_higher_order_angle_has_11_specs(self, stage1_config):
        ho_specs = [s for s in stage1_config.encoding_specs
                    if s.name == "higher_order_angle"]
        assert len(ho_specs) == 11

    def test_higher_order_angle_order_3_excludes_n_features_2(self, stage1_config):
        order3 = [s for s in stage1_config.encoding_specs
                  if s.name == "higher_order_angle" and s.params.get("order") == 3]
        assert len(order3) == 3
        for spec in order3:
            assert spec.params["n_features"] >= 4

    def test_so2_has_3_specs(self, stage1_config):
        so2 = [s for s in stage1_config.encoding_specs
               if s.name == "so2_equivariant"]
        assert len(so2) == 3
        for spec in so2:
            assert spec.params["n_features"] == 2

    def test_all_specs_instantiate(self, stage1_config):
        for spec in stage1_config.encoding_specs:
            enc = get_encoding(spec.name, **spec.params)
            if "n_features" in spec.params:
                assert enc.n_features == spec.params["n_features"]

    def test_no_duplicate_specs(self, stage1_config):
        seen = set()
        for spec in stage1_config.encoding_specs:
            key = (spec.name, tuple(sorted(spec.params.items())))
            assert key not in seen, f"Duplicate: {spec.name}({spec.params})"
            seen.add(key)
