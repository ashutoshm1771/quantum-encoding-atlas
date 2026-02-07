"""Tests for result aggregation, export, and querying utilities.

Covers the full :class:`ResultStore` API surface:

- Construction from checkpoint and from JSON
- Filtering by stage, encoding, status, metric
- Metric value extraction and grouping
- Summary JSON export and reload round trip
- Markdown table rendering
- Edge cases (empty store, all-failed results, missing keys)

Run with: pytest experiments/tests/test_results.py -v
"""

import json
import os

import pytest

from experiments.checkpoint import CheckpointManager
from experiments.results import ResultStore, TaskResult


# =============================================================================
# Helpers
# =============================================================================


def _task(
    encoding: str = "iqp",
    params: dict | None = None,
    result: dict | None = None,
    status: str = "success",
    stage: str = "resources",
    wall_time: float = 0.1,
) -> TaskResult:
    """Build a TaskResult with sensible defaults."""
    if params is None:
        params = {"n_features": 4}
    if result is None:
        result = {"gate_count": 52, "depth": 6}
    exp_id = f"{stage}/{encoding}/params_hash/{stage}"
    return TaskResult(
        experiment_id=exp_id,
        stage=stage,
        encoding_name=encoding,
        encoding_params=params,
        metric=stage,
        result=result,
        status=status,
        wall_time=wall_time,
    )


def _populated_store() -> ResultStore:
    """Create a store with a known set of results for query tests."""
    return ResultStore([
        _task("iqp", {"n_features": 4, "reps": 1}, {"gate_count": 26, "depth": 3}),
        _task("iqp", {"n_features": 4, "reps": 2}, {"gate_count": 52, "depth": 6}),
        _task("angle", {"n_features": 4}, {"gate_count": 4, "depth": 1}),
        _task("basis", {"n_features": 4}, {"gate_count": 4, "depth": 1}, status="failed"),
        _task(
            "iqp", {"n_features": 4, "reps": 1},
            {"expressibility": 0.83},
            stage="expressibility",
        ),
    ])


# =============================================================================
# Construction
# =============================================================================


class TestResultStoreConstruction:
    """Tests for building a ResultStore."""

    def test_empty_store(self):
        store = ResultStore()
        assert store.count == 0

    def test_from_list(self):
        tasks = [_task("iqp"), _task("angle")]
        store = ResultStore(tasks)
        assert store.count == 2

    def test_from_checkpoint(self, tmp_path):
        """Build a store by replaying a checkpoint log."""
        ckpt = CheckpointManager(str(tmp_path))

        exp_id = ckpt.get_experiment_id("resources", "iqp", {"n_features": 4}, "resources")
        ckpt.mark_completed(exp_id, {
            "status": "success",
            "gate_count": 52,
            "encoding_params": {"n_features": 4},
        })

        store = ResultStore.from_checkpoint(ckpt)
        assert store.count == 1

        results = store.filter(encoding_name="iqp")
        assert len(results) == 1
        assert results[0].result["gate_count"] == 52

    def test_add(self):
        store = ResultStore()
        store.add(_task("iqp"))
        assert store.count == 1
        store.add(_task("angle"))
        assert store.count == 2


# =============================================================================
# Filtering
# =============================================================================


class TestResultStoreFiltering:
    """Tests for :meth:`ResultStore.filter`."""

    def test_filter_by_encoding_name(self):
        store = _populated_store()
        results = store.filter(encoding_name="iqp")
        assert all(r.encoding_name == "iqp" for r in results)
        assert len(results) == 3  # 2 resources + 1 expressibility

    def test_filter_by_stage(self):
        store = _populated_store()
        results = store.filter(stage="resources")
        assert all(r.stage == "resources" for r in results)
        assert len(results) == 4  # 2 iqp + 1 angle + 1 basis(failed)

    def test_filter_by_status(self):
        store = _populated_store()
        assert len(store.filter(status="success")) == 4
        assert len(store.filter(status="failed")) == 1

    def test_filter_combined(self):
        store = _populated_store()
        results = store.filter(encoding_name="iqp", stage="resources", status="success")
        assert len(results) == 2

    def test_filter_no_match(self):
        store = _populated_store()
        assert store.filter(encoding_name="nonexistent") == []


# =============================================================================
# Metric extraction
# =============================================================================


class TestGetMetricValues:
    """Tests for :meth:`ResultStore.get_metric_values`."""

    def test_groups_by_encoding(self):
        store = _populated_store()
        grouped = store.get_metric_values("gate_count")
        assert "iqp" in grouped
        assert "angle" in grouped
        # basis is failed — excluded by default.
        assert "basis" not in grouped

    def test_values_are_floats(self):
        store = _populated_store()
        grouped = store.get_metric_values("gate_count")
        for values in grouped.values():
            assert all(isinstance(v, float) for v in values)

    def test_filter_by_encoding(self):
        store = _populated_store()
        grouped = store.get_metric_values("gate_count", encoding_name="angle")
        assert list(grouped.keys()) == ["angle"]
        assert grouped["angle"] == [4.0]

    def test_missing_key_excluded(self):
        store = _populated_store()
        # "expressibility" key only exists in one result.
        grouped = store.get_metric_values("expressibility")
        assert "iqp" in grouped
        assert len(grouped["iqp"]) == 1
        assert grouped["iqp"][0] == pytest.approx(0.83)


# =============================================================================
# JSON export and reload
# =============================================================================


class TestJsonExport:
    """Tests for :meth:`ResultStore.save_summary_json` and :meth:`from_json`."""

    def test_round_trip(self, tmp_path):
        store = _populated_store()
        path = os.path.join(str(tmp_path), "summary.json")
        store.save_summary_json(path)

        assert os.path.isfile(path)

        reloaded = ResultStore.from_json(path)
        assert reloaded.count == store.count

    def test_json_structure(self, tmp_path):
        store = _populated_store()
        path = os.path.join(str(tmp_path), "summary.json")
        store.save_summary_json(path)

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        assert "total_results" in data
        assert "success_count" in data
        assert "failed_count" in data
        assert "results" in data
        assert data["total_results"] == 5
        assert data["success_count"] == 4
        assert data["failed_count"] == 1

    def test_creates_parent_directories(self, tmp_path):
        store = ResultStore([_task("iqp")])
        nested = os.path.join(str(tmp_path), "a", "b", "summary.json")
        store.save_summary_json(nested)
        assert os.path.isfile(nested)

    def test_empty_store_exports(self, tmp_path):
        store = ResultStore()
        path = os.path.join(str(tmp_path), "summary.json")
        store.save_summary_json(path)

        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert data["total_results"] == 0
        assert data["results"] == []

    def test_numpy_values_serialise(self, tmp_path):
        """Numpy scalars in results must survive JSON serialisation."""
        import numpy as np

        task = _task("iqp", result={"gate_count": np.int64(52), "ratio": np.float64(0.5)})
        store = ResultStore([task])
        path = os.path.join(str(tmp_path), "summary.json")
        store.save_summary_json(path)

        reloaded = ResultStore.from_json(path)
        r = reloaded.filter(encoding_name="iqp")[0].result
        assert r["gate_count"] == 52
        assert abs(r["ratio"] - 0.5) < 1e-10


# =============================================================================
# Markdown table
# =============================================================================


class TestMarkdownTable:
    """Tests for :meth:`ResultStore.to_markdown_table`."""

    def test_basic_table(self):
        store = _populated_store()
        md = store.to_markdown_table(["gate_count", "depth"])

        lines = md.strip().split("\n")
        # Header, separator, data rows.
        assert len(lines) >= 3
        assert "Encoding" in lines[0]
        assert "gate_count" in lines[0]
        assert "---" in lines[1]

    def test_excludes_failed_tasks(self):
        store = _populated_store()
        md = store.to_markdown_table(["gate_count"])
        # "basis" is the only failed task — should not appear.
        assert "basis" not in md.lower()

    def test_sort_by_metric(self):
        store = _populated_store()
        md = store.to_markdown_table(["gate_count"], sort_by="gate_count")
        lines = md.strip().split("\n")
        # Data rows (skip header + separator).
        data_lines = lines[2:]
        assert len(data_lines) >= 2

    def test_empty_store_produces_header_only(self):
        store = ResultStore()
        md = store.to_markdown_table(["gate_count"])
        lines = md.strip().split("\n")
        # Only header + separator, no data rows.
        assert len(lines) == 2
