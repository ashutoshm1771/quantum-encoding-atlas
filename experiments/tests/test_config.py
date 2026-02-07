"""Tests for experiment configuration loading, validation, and expansion.

This module tests the critical config infrastructure that sits between
JSON config files and encoding instantiation. A misconfigured parameter
name (e.g. ``"steps"`` instead of ``"reps"``) causes the encoding to be
silently skipped at runtime, so we test the full chain:

    JSON file  ->  load_config()  ->  EncodingSpec  ->  get_encoding(**params)

Run with: pytest experiments/tests/test_config.py -v
"""

import json
import os

import pytest

from experiments.config import (
    EncodingSpec,
    ExperimentConfig,
    STAGE_OFFSETS,
    VALID_BACKENDS,
    VALID_STAGES,
    _expand_param_sweep,
    _validate_raw_config,
    config_to_dict,
    load_config,
)

# =============================================================================
# Directory containing the shipped experiment configs.
# =============================================================================

_CONFIGS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), os.pardir, "configs")
)

# Every shipped config file that should be validated.
_ALL_CONFIG_FILES = [
    "stage1_resources.json",
    "stage2_simulability.json",
    "stage3_expressibility.json",
    "stage4_entanglement.json",
    "stage5_trainability.json",
    "stage5b_noise.json",
    "stage6a_vqc.json",
    "stage6b_kernel.json",
]

# Filter to only configs that actually exist on disk (avoids hard failures
# when individual stages haven't been created yet).
_EXISTING_CONFIG_FILES = [
    f for f in _ALL_CONFIG_FILES
    if os.path.isfile(os.path.join(_CONFIGS_DIR, f))
]


# =============================================================================
# Helpers
# =============================================================================


def _write_config(config_dict: dict, tmp_path) -> str:
    """Write a config dict to a temporary JSON file and return the path."""
    path = os.path.join(str(tmp_path), "test_config.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(config_dict, fh)
    return path


# =============================================================================
# _expand_param_sweep
# =============================================================================


class TestExpandParamSweep:
    """Tests for Cartesian-product parameter sweep expansion."""

    def test_empty_params_returns_single_empty_dict(self):
        assert _expand_param_sweep({}) == [{}]

    def test_all_scalars_returns_single_dict(self):
        result = _expand_param_sweep({"n_features": 4, "reps": 2})
        assert result == [{"n_features": 4, "reps": 2}]

    def test_single_list_axis(self):
        result = _expand_param_sweep({"n_features": [2, 4]})
        assert result == [{"n_features": 2}, {"n_features": 4}]

    def test_two_list_axes_cartesian_product(self):
        result = _expand_param_sweep({"n_features": [2, 4], "reps": [1, 2]})
        assert len(result) == 4
        assert {"n_features": 2, "reps": 1} in result
        assert {"n_features": 2, "reps": 2} in result
        assert {"n_features": 4, "reps": 1} in result
        assert {"n_features": 4, "reps": 2} in result

    def test_mixed_scalar_and_list(self):
        result = _expand_param_sweep({"n_features": [2, 4], "reps": 2})
        assert len(result) == 2
        for combo in result:
            assert combo["reps"] == 2

    def test_single_element_list_treated_as_sweep(self):
        result = _expand_param_sweep({"n_features": [4]})
        assert result == [{"n_features": 4}]

    def test_empty_list_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty list"):
            _expand_param_sweep({"n_features": []})

    def test_string_scalar_not_expanded(self):
        result = _expand_param_sweep({"rotation": "Y", "n_features": 4})
        assert result == [{"rotation": "Y", "n_features": 4}]

    def test_list_of_strings_expanded(self):
        result = _expand_param_sweep({"rotation": ["X", "Y", "Z"]})
        assert len(result) == 3
        assert {"rotation": "X"} in result
        assert {"rotation": "Y"} in result
        assert {"rotation": "Z"} in result

    def test_three_axes_produces_correct_count(self):
        result = _expand_param_sweep({
            "n_features": [2, 4],
            "reps": [1, 2, 3],
            "entanglement": ["full", "linear"],
        })
        assert len(result) == 2 * 3 * 2  # 12

    def test_output_contains_only_scalars(self):
        result = _expand_param_sweep({
            "n_features": [2, 4],
            "reps": [1, 2],
            "fixed": 42,
        })
        for combo in result:
            for value in combo.values():
                assert not isinstance(value, list)


# =============================================================================
# EncodingSpec
# =============================================================================


class TestEncodingSpec:
    """Tests for the EncodingSpec frozen dataclass."""

    def test_valid_creation(self):
        spec = EncodingSpec(name="iqp", params={"n_features": 4, "reps": 2})
        assert spec.name == "iqp"
        assert spec.params == {"n_features": 4, "reps": 2}

    def test_default_empty_params(self):
        spec = EncodingSpec(name="basis")
        assert spec.params == {}

    def test_empty_name_raises(self):
        with pytest.raises(ValueError, match="non-empty string"):
            EncodingSpec(name="", params={})

    def test_list_in_params_raises(self):
        """Params with list values indicate unexpanded sweeps — a bug."""
        with pytest.raises(ValueError, match="list"):
            EncodingSpec(name="iqp", params={"n_features": [2, 4]})

    def test_frozen(self):
        spec = EncodingSpec(name="iqp", params={"n_features": 4})
        with pytest.raises(AttributeError):
            spec.name = "angle"  # type: ignore[misc]


# =============================================================================
# ExperimentConfig — seed computation
# =============================================================================


class TestExperimentConfigSeeds:
    """Tests for deterministic seed computation."""

    @staticmethod
    def _make_config(stage: str, seed: int = 42) -> ExperimentConfig:
        return ExperimentConfig(
            stage=stage,
            base_seed=seed,
            backend="pennylane",
            encoding_specs=[],
            analysis_params={},
            output_dir="test",
        )

    def test_stage_seed_expressibility(self):
        config = self._make_config("expressibility")
        assert config.stage_seed == 42 + 3 * 1000  # 3042

    def test_stage_seed_noise_half_integer_offset(self):
        config = self._make_config("noise")
        assert config.stage_seed == int(42 + 5.5 * 1000)  # 5542

    def test_task_seed_offset(self):
        config = self._make_config("resources")
        assert config.task_seed(0) == config.stage_seed
        assert config.task_seed(5) == config.stage_seed + 5
        assert config.task_seed(100) == config.stage_seed + 100

    def test_all_stage_seeds_unique(self):
        """No two stages should produce the same seed."""
        seeds = {
            stage: 42 + int(offset * 1000)
            for stage, offset in STAGE_OFFSETS.items()
        }
        assert len(set(seeds.values())) == len(seeds), (
            f"Stage seed collision detected: {seeds}"
        )

    def test_all_stages_have_offsets(self):
        """Every valid stage must have a seed offset defined."""
        for stage in VALID_STAGES:
            assert stage in STAGE_OFFSETS, (
                f"Stage '{stage}' missing from STAGE_OFFSETS"
            )


# =============================================================================
# _validate_raw_config
# =============================================================================


class TestValidateRawConfig:
    """Tests for raw JSON config structure validation."""

    @staticmethod
    def _minimal_raw(*, stage="resources", backend="pennylane"):
        return {
            "stage": stage,
            "backend": backend,
            "encodings": [{"name": "angle", "params": {"n_features": 4}}],
        }

    def test_valid_minimal(self):
        _validate_raw_config(self._minimal_raw())

    def test_missing_stage_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            _validate_raw_config({"encodings": [{"name": "angle"}]})

    def test_missing_encodings_raises(self):
        with pytest.raises(ValueError, match="missing required keys"):
            _validate_raw_config({"stage": "resources"})

    def test_unknown_stage_raises(self):
        with pytest.raises(ValueError, match="Unknown stage"):
            _validate_raw_config(self._minimal_raw(stage="nonexistent"))

    def test_empty_encodings_list_raises(self):
        raw = {"stage": "resources", "encodings": []}
        with pytest.raises(ValueError, match="non-empty list"):
            _validate_raw_config(raw)

    def test_encoding_missing_name_raises(self):
        raw = {
            "stage": "resources",
            "encodings": [{"params": {"n_features": 4}}],
        }
        with pytest.raises(ValueError, match="missing 'name'"):
            _validate_raw_config(raw)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            _validate_raw_config(self._minimal_raw(backend="bad_backend"))

    def test_all_valid_stages_accepted(self):
        for stage in VALID_STAGES:
            _validate_raw_config(self._minimal_raw(stage=stage))

    def test_all_valid_backends_accepted(self):
        for backend in VALID_BACKENDS:
            _validate_raw_config(self._minimal_raw(backend=backend))


# =============================================================================
# load_config
# =============================================================================


class TestLoadConfig:
    """Tests for the full config loading pipeline."""

    def test_load_minimal(self, tmp_path):
        path = _write_config({
            "stage": "resources",
            "seed": 42,
            "encodings": [
                {"name": "angle", "params": {"n_features": 4}},
            ],
        }, tmp_path)

        config = load_config(path)
        assert config.stage == "resources"
        assert config.base_seed == 42
        assert len(config.encoding_specs) == 1
        assert config.encoding_specs[0].name == "angle"
        assert config.encoding_specs[0].params == {"n_features": 4}

    def test_load_with_sweep_expansion(self, tmp_path):
        path = _write_config({
            "stage": "expressibility",
            "encodings": [{
                "name": "iqp",
                "params": {"n_features": [2, 4], "reps": [1, 2]},
            }],
        }, tmp_path)

        config = load_config(path)
        assert len(config.encoding_specs) == 4

        combos = {
            (s.params["n_features"], s.params["reps"])
            for s in config.encoding_specs
        }
        assert combos == {(2, 1), (2, 2), (4, 1), (4, 2)}

    def test_load_multiple_encodings(self, tmp_path):
        path = _write_config({
            "stage": "resources",
            "encodings": [
                {"name": "angle", "params": {"n_features": 4}},
                {"name": "iqp", "params": {"n_features": [2, 4]}},
            ],
        }, tmp_path)

        config = load_config(path)
        assert len(config.encoding_specs) == 3  # 1 + 2

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/config.json")

    def test_default_seed_is_42(self, tmp_path):
        path = _write_config({
            "stage": "resources",
            "encodings": [{"name": "angle", "params": {"n_features": 4}}],
        }, tmp_path)
        assert load_config(path).base_seed == 42

    def test_default_backend_is_pennylane(self, tmp_path):
        path = _write_config({
            "stage": "resources",
            "encodings": [{"name": "angle", "params": {"n_features": 4}}],
        }, tmp_path)
        assert load_config(path).backend == "pennylane"

    def test_analysis_params_forwarded(self, tmp_path):
        path = _write_config({
            "stage": "expressibility",
            "encodings": [{"name": "iqp", "params": {"n_features": 4}}],
            "analysis_params": {"n_samples": 5000, "n_bins": 75},
        }, tmp_path)
        config = load_config(path)
        assert config.analysis_params["n_samples"] == 5000
        assert config.analysis_params["n_bins"] == 75

    def test_quick_mode_overrides_analysis_params(self, tmp_path):
        path = _write_config({
            "stage": "expressibility",
            "encodings": [{"name": "iqp", "params": {"n_features": 4, "reps": 2}}],
            "analysis_params": {"n_samples": 5000, "n_bins": 75},
        }, tmp_path)

        config = load_config(path, quick=True)
        assert config.quick is True
        assert config.analysis_params["n_samples"] == 500  # overridden

    def test_quick_mode_reduces_n_features_sweep(self, tmp_path):
        path = _write_config({
            "stage": "expressibility",
            "encodings": [{
                "name": "iqp",
                "params": {"n_features": [2, 4, 6, 8], "reps": [1, 2, 3, 4]},
            }],
        }, tmp_path)

        config = load_config(path, quick=True)
        n_features_values = {s.params["n_features"] for s in config.encoding_specs}
        reps_values = {s.params["reps"] for s in config.encoding_specs}
        assert n_features_values <= {2, 4}
        assert reps_values <= {1, 2}

    def test_quick_mode_vqc_epochs_propagated(self, tmp_path):
        """Quick-mode epochs override must propagate into nested vqc_params."""
        path = _write_config({
            "stage": "vqc",
            "encodings": [{"name": "angle", "params": {"n_features": 2}}],
            "analysis_params": {
                "datasets": ["moons"],
                "n_runs": 10,
                "n_folds": 5,
                "vqc_params": {"n_var_layers": 2, "lr": 0.01, "epochs": 100},
            },
        }, tmp_path)

        config = load_config(path, quick=True)
        assert config.analysis_params["vqc_params"]["epochs"] == 20


# =============================================================================
# config_to_dict
# =============================================================================


class TestConfigToDict:
    """Tests for config serialization."""

    def test_round_trip_is_json_serializable(self, tmp_path):
        path = _write_config({
            "stage": "resources",
            "seed": 42,
            "backend": "pennylane",
            "encodings": [
                {"name": "angle", "params": {"n_features": 4}},
            ],
            "analysis_params": {"n_samples": 1000},
            "output_dir": "test_output",
        }, tmp_path)

        config = load_config(path)
        d = config_to_dict(config)

        assert d["stage"] == "resources"
        assert d["base_seed"] == 42
        assert isinstance(d["encoding_specs"], list)

        # Must be JSON-serializable without error.
        json.dumps(d)

    def test_preserves_encoding_specs(self, tmp_path):
        path = _write_config({
            "stage": "expressibility",
            "encodings": [
                {"name": "iqp", "params": {"n_features": [2, 4], "reps": [1, 2]}},
            ],
        }, tmp_path)

        config = load_config(path)
        d = config_to_dict(config)
        assert len(d["encoding_specs"]) == 4
        for spec_dict in d["encoding_specs"]:
            assert "name" in spec_dict
            assert "params" in spec_dict


# =============================================================================
# Shipped config file integrity tests
# =============================================================================


class TestConfigFileIntegrity:
    """Validate that every shipped config file loads, expands, and
    produces encoding specs that can be instantiated without error.

    This is the test class that would have caught the ``"steps"`` vs
    ``"reps"`` bug in ``stage6a_vqc.json`` and ``stage6b_kernel.json``
    before any experiment was run.
    """

    @pytest.fixture(params=_EXISTING_CONFIG_FILES)
    def config_path(self, request):
        return os.path.join(_CONFIGS_DIR, request.param)

    def test_config_loads_without_error(self, config_path):
        config = load_config(config_path)
        assert config.stage in VALID_STAGES
        assert config.backend in VALID_BACKENDS
        assert len(config.encoding_specs) > 0

    def test_config_loads_in_quick_mode(self, config_path):
        config = load_config(config_path, quick=True)
        assert config.quick is True
        assert len(config.encoding_specs) > 0

    def test_all_encoding_specs_instantiate(self, config_path):
        """Every encoding spec produced by the config must instantiate.

        This catches parameter name mismatches (e.g. ``"steps"`` passed
        to a constructor that only accepts ``"reps"``), unknown encoding
        registry names, and invalid parameter values.
        """
        from encoding_atlas import get_encoding

        config = load_config(config_path)

        for spec in config.encoding_specs:
            enc = get_encoding(spec.name, **spec.params)
            # If n_features was specified, the instantiated encoding
            # must agree on the number of features.
            if "n_features" in spec.params:
                assert enc.n_features == spec.params["n_features"], (
                    f"{spec.name}(**{spec.params}): expected n_features="
                    f"{spec.params['n_features']}, got {enc.n_features}"
                )

    def test_config_serialization_round_trip(self, config_path):
        config = load_config(config_path)
        d = config_to_dict(config)
        # Must be JSON-serializable.
        serialized = json.dumps(d)
        assert isinstance(json.loads(serialized), dict)

    def test_no_duplicate_encoding_specs(self, config_path):
        """Check for accidental duplicate tasks in the expanded spec list."""
        config = load_config(config_path)
        seen = set()
        for spec in config.encoding_specs:
            key = (spec.name, tuple(sorted(spec.params.items())))
            assert key not in seen, (
                f"Duplicate encoding spec: {spec.name}({spec.params})"
            )
            seen.add(key)
