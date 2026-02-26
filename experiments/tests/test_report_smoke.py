"""Smoke tests for Stage 8 report generation.

These tests verify that the report module can load existing experiment
results and produce the expected output files without errors.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results" / "raw"
_TRADEOFF_DIR = _RESULTS_DIR / "stage7_tradeoff"
_SENSITIVITY_DIR = _RESULTS_DIR / "stage6a5_sensitivity"
_FIGURE_DIR = _PROJECT_ROOT / "experiments" / "results" / "figures"

_STAGE_DIRS = {
    "resources": str(_RESULTS_DIR / "stage1_resources"),
    "simulability": str(_RESULTS_DIR / "stage2_simulability"),
    "expressibility": str(_RESULTS_DIR / "stage3_expressibility"),
    "entanglement": str(_RESULTS_DIR / "stage4_entanglement"),
    "trainability": str(_RESULTS_DIR / "stage5_trainability"),
    "noise": str(_RESULTS_DIR / "stage5b_noise"),
    "vqc": str(_RESULTS_DIR / "stage6a_vqc"),
    "kernel": str(_RESULTS_DIR / "stage6b_kernel"),
}


def _results_available() -> bool:
    """Check if experiment results are available for testing."""
    required = [
        _TRADEOFF_DIR / "rankings.json",
        _TRADEOFF_DIR / "hypothesis_verdicts.json",
    ]
    return all(p.exists() for p in required)


skip_if_no_results = pytest.mark.skipif(
    not _results_available(),
    reason="Experiment results not available (run stages 1-7 first)",
)


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

class TestHelpers:
    """Test module-level helper functions."""

    def test_json_default_numpy_int(self) -> None:
        """Numpy integers are serialized as Python int."""
        from experiments.report import _json_default

        try:
            import numpy as np
            val = np.int64(42)
            assert _json_default(val) == 42
        except ImportError:
            pytest.skip("numpy not available")

    def test_json_default_numpy_float(self) -> None:
        """Numpy floats are serialized as Python float."""
        from experiments.report import _json_default

        try:
            import numpy as np
            val = np.float64(3.14)
            assert abs(_json_default(val) - 3.14) < 1e-10
        except ImportError:
            pytest.skip("numpy not available")

    def test_json_default_raises_on_unknown(self) -> None:
        """Unknown types raise TypeError."""
        from experiments.report import _json_default

        with pytest.raises(TypeError):
            _json_default(object())

    def test_fmt_none(self) -> None:
        """None values format as dash."""
        from experiments.report import _fmt
        assert _fmt(None) == "—"

    def test_fmt_float(self) -> None:
        """Float values format with precision."""
        from experiments.report import _fmt
        assert _fmt(0.12345, 3) == "0.123"

    def test_fmt_bool(self) -> None:
        """Boolean values format as Yes/No."""
        from experiments.report import _fmt
        assert _fmt(True) == "Yes"
        assert _fmt(False) == "No"

    def test_tex_escape(self) -> None:
        """LaTeX special characters are escaped."""
        from experiments.report import _tex_escape
        assert _tex_escape("zz_feature_map") == r"zz\_feature\_map"

    def test_load_json_missing(self) -> None:
        """Missing file returns None."""
        from experiments.report import _load_json
        assert _load_json("/nonexistent/file.json") is None


# ---------------------------------------------------------------------------
# Unit tests: table generation with mock data
# ---------------------------------------------------------------------------

class TestTableGeneration:
    """Test table generation with minimal mock data."""

    def test_accuracy_matrix_md(self) -> None:
        """Accuracy matrix markdown generation."""
        from experiments.report import _generate_accuracy_matrix_md

        enc_names = ["angle", "basis"]
        ds_names = ["moons", "circles"]
        matrix = {
            "angle": {"moons": "0.850 +/- 0.030", "circles": "0.780 +/- 0.040"},
            "basis": {"moons": "0.550 +/- 0.050", "circles": "0.520 +/- 0.060"},
        }
        md = _generate_accuracy_matrix_md(enc_names, ds_names, matrix, "Test Title")
        assert "# Test Title" in md
        assert "angle" in md
        assert "0.850 +/- 0.030" in md

    def test_accuracy_matrix_tex(self) -> None:
        """Accuracy matrix LaTeX generation."""
        from experiments.report import _generate_accuracy_matrix_tex

        enc_names = ["angle"]
        ds_names = ["moons"]
        matrix = {"angle": {"moons": "0.850 +/- 0.030"}}
        tex = _generate_accuracy_matrix_tex(
            enc_names, ds_names, matrix, "Test", "tab:test",
        )
        assert r"\begin{table}" in tex
        assert r"\label{tab:test}" in tex

    def test_sensitivity_table_md_empty(self) -> None:
        """Sensitivity table handles empty results."""
        from experiments.report import _generate_sensitivity_table_md
        md = _generate_sensitivity_table_md({"results": []})
        assert "No results available" in md


# ---------------------------------------------------------------------------
# Unit tests: narrative generation
# ---------------------------------------------------------------------------

class TestNarrativeGeneration:
    """Test narrative document generation with mock data."""

    def _mock_verdicts(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "hypothesis_verdicts": {
                "H1": {
                    "verdict": "refuted",
                    "confidence": "moderate",
                    "evidence": "Test evidence for H1.",
                    "test_statistic": {"rho": -0.5, "p_value": 0.01},
                },
                "H2": {
                    "verdict": "inconclusive",
                    "confidence": "low",
                    "evidence": "Test evidence for H2.",
                    "test_statistic": {"wins": 2},
                },
                "H3": {
                    "verdict": "supported",
                    "confidence": "high",
                    "evidence": "Test evidence for H3.",
                    "test_statistic": {},
                },
                "H4": {
                    "verdict": "supported",
                    "confidence": "high",
                    "evidence": "Test evidence.",
                    "test_statistic": {},
                },
                "H5": {
                    "verdict": "supported",
                    "confidence": "moderate",
                    "evidence": "Test evidence.",
                    "test_statistic": {},
                },
                "H6": {
                    "verdict": "inconclusive",
                    "confidence": "low",
                    "evidence": "Test evidence.",
                    "test_statistic": {},
                },
                "H7": {
                    "verdict": "supported",
                    "confidence": "moderate",
                    "evidence": "Test evidence.",
                    "test_statistic": {"n_pareto": 4},
                },
            },
        }

    def test_hypotheses_md_contains_all_hypotheses(self) -> None:
        """Hypotheses narrative includes all H1-H7."""
        from experiments.report import _generate_hypotheses_md

        verdicts = self._mock_verdicts()
        md = _generate_hypotheses_md(verdicts, None)
        for h_id in ["H1", "H2", "H3", "H4", "H5", "H6", "H7"]:
            assert h_id in md

    def test_hypotheses_md_verdict_labels(self) -> None:
        """Verdict labels appear in the narrative."""
        from experiments.report import _generate_hypotheses_md

        verdicts = self._mock_verdicts()
        md = _generate_hypotheses_md(verdicts, None)
        assert "REFUTED" in md
        assert "SUPPORTED" in md
        assert "INCONCLUSIVE" in md

    def test_ranking_md_structure(self) -> None:
        """Ranking narrative has expected sections."""
        from experiments.report import _generate_ranking_md

        rankings_data = {
            "rankings": [
                {
                    "encoding": "angle",
                    "rank": 1,
                    "score": 0.772,
                    "vqc_accuracy": 0.848,
                    "kernel_accuracy": 0.958,
                    "is_pareto": True,
                    "is_simulable": True,
                },
            ],
        }
        pareto_data = {
            "pareto_optimal": ["angle"],
            "objective_names": ["accuracy", "inv_depth"],
            "n_encodings_analyzed": 16,
            "encodings": {
                "angle": {"objectives": [0.848, 0.5]},
            },
        }
        md = _generate_ranking_md(rankings_data, pareto_data, None)
        assert "# Final Encoding Rankings" in md
        assert "Pareto Front" in md
        assert "Key Findings" in md
        assert "Practical Guidance" in md


# ---------------------------------------------------------------------------
# Integration tests with real data
# ---------------------------------------------------------------------------

@skip_if_no_results
class TestReportIntegration:
    """Integration tests using actual experiment results."""

    def test_generate_report_produces_all_outputs(self) -> None:
        """Full report generation produces expected files."""
        from experiments.report import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_report(
                stage_dirs=_STAGE_DIRS,
                tradeoff_dir=str(_TRADEOFF_DIR),
                output_dir=tmpdir,
                figure_dir=str(_FIGURE_DIR),
                sensitivity_dir=str(_SENSITIVITY_DIR),
            )

            assert result["status"] in ("success", "partial")
            assert result["n_files_generated"] > 0

            # Check core files exist.
            assert os.path.isfile(os.path.join(tmpdir, "master_summary.json"))
            assert os.path.isfile(os.path.join(tmpdir, "hypotheses.md"))
            assert os.path.isfile(os.path.join(tmpdir, "ranking.md"))

    def test_master_summary_schema(self) -> None:
        """Master summary has expected schema."""
        from experiments.report import _build_master_summary

        summary = _build_master_summary(
            _STAGE_DIRS, str(_TRADEOFF_DIR),
            str(_SENSITIVITY_DIR),
        )
        assert summary["schema_version"] == "1.0"
        assert summary["n_encodings"] == 16
        assert len(summary["encoding_profiles"]) == 16
        assert "hypothesis_verdicts" in summary
        assert "pareto_front" in summary

        # Verify each profile has required keys.
        for profile in summary["encoding_profiles"]:
            assert "encoding" in profile
            assert "rank" in profile
            assert "metrics" in profile
            assert "vqc_accuracy" in profile["metrics"]

    def test_tables_directory_populated(self) -> None:
        """Table generation creates files in the tables directory."""
        from experiments.report import _generate_tables

        with tempfile.TemporaryDirectory() as tmpdir:
            table_dir = os.path.join(tmpdir, "tables")
            files = _generate_tables(
                _STAGE_DIRS, str(_TRADEOFF_DIR),
                str(_SENSITIVITY_DIR), table_dir,
            )
            assert len(files) > 0
            assert os.path.isdir(table_dir)

            # Should have VQC and kernel accuracy matrices.
            table_names = [os.path.basename(f) for f in files]
            assert "vqc_accuracy_matrix.md" in table_names
            assert "kernel_accuracy_matrix.md" in table_names

    def test_figure_verification(self) -> None:
        """Figure verification reports existing figures."""
        from experiments.report import _verify_figures

        report = _verify_figures(str(_FIGURE_DIR))
        assert report["status"] in ("ok", "empty")
        if report["status"] == "ok":
            assert report["png_count"] > 0

    def test_master_summary_json_is_valid(self) -> None:
        """Generated master_summary.json is valid JSON with correct encoding count."""
        from experiments.report import generate_report

        with tempfile.TemporaryDirectory() as tmpdir:
            generate_report(
                stage_dirs=_STAGE_DIRS,
                tradeoff_dir=str(_TRADEOFF_DIR),
                output_dir=tmpdir,
                figure_dir=str(_FIGURE_DIR),
                sensitivity_dir=str(_SENSITIVITY_DIR),
                generate_tables=False,
            )

            with open(os.path.join(tmpdir, "master_summary.json"), "r") as fh:
                data = json.load(fh)

            assert data["schema_version"] == "1.0"
            assert data["n_encodings"] == 16


# ---------------------------------------------------------------------------
# Config integration test
# ---------------------------------------------------------------------------

class TestConfigIntegration:
    """Test that the report stage is properly registered."""

    def test_report_in_valid_stages(self) -> None:
        """Report stage is registered in VALID_STAGES."""
        from experiments.config import VALID_STAGES
        assert "report" in VALID_STAGES

    def test_report_in_stage_offsets(self) -> None:
        """Report stage has a seed offset."""
        from experiments.config import STAGE_OFFSETS
        assert "report" in STAGE_OFFSETS
        assert STAGE_OFFSETS["report"] == 9

    def test_report_config_loads(self) -> None:
        """Stage 8 config file loads without error."""
        from experiments.config import load_config

        config_path = str(
            _PROJECT_ROOT / "experiments" / "configs" / "stage8_report.json"
        )
        if os.path.isfile(config_path):
            config = load_config(config_path)
            assert config.stage == "report"
            assert "__report__" in config.encoding_specs[0].name

    def test_report_handler_registered(self) -> None:
        """Report handler is in the _STAGE_HANDLERS registry."""
        from experiments.runner import _STAGE_HANDLERS
        assert "report" in _STAGE_HANDLERS
