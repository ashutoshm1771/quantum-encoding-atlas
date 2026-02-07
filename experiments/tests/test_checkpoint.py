"""Tests for crash-safe, append-only checkpoint management.

Covers the full :class:`CheckpointManager` API surface:

- Experiment ID generation (determinism, uniqueness, structure)
- Mark-completed / is-completed round trip
- Result retrieval
- Append-only JSONL persistence
- Crash recovery (truncated last line)
- Thread safety (concurrent writes)
- Clear / reset behaviour
- Numpy type serialisation

Run with: pytest experiments/tests/test_checkpoint.py -v
"""

import json
import os
import threading

import numpy as np
import pytest

from experiments.checkpoint import CheckpointManager


# =============================================================================
# Helpers
# =============================================================================


def _make_ckpt(tmp_path: str, filename: str = "completed.jsonl") -> CheckpointManager:
    """Create a CheckpointManager rooted at *tmp_path*."""
    return CheckpointManager(str(tmp_path), filename=filename)


# =============================================================================
# Experiment ID generation
# =============================================================================


class TestExperimentId:
    """Tests for :meth:`CheckpointManager.get_experiment_id`."""

    def test_deterministic_same_inputs(self):
        """Same inputs always produce the same ID."""
        a = CheckpointManager.get_experiment_id(
            "resources", "iqp", {"n_features": 4, "reps": 2}, "resources",
        )
        b = CheckpointManager.get_experiment_id(
            "resources", "iqp", {"n_features": 4, "reps": 2}, "resources",
        )
        assert a == b

    def test_deterministic_key_order_independent(self):
        """Parameter dict key order does not affect the ID."""
        a = CheckpointManager.get_experiment_id(
            "resources", "iqp", {"reps": 2, "n_features": 4}, "resources",
        )
        b = CheckpointManager.get_experiment_id(
            "resources", "iqp", {"n_features": 4, "reps": 2}, "resources",
        )
        assert a == b

    def test_different_params_produce_different_ids(self):
        """Distinct parameters must produce distinct IDs."""
        a = CheckpointManager.get_experiment_id(
            "resources", "iqp", {"n_features": 4, "reps": 1}, "resources",
        )
        b = CheckpointManager.get_experiment_id(
            "resources", "iqp", {"n_features": 4, "reps": 2}, "resources",
        )
        assert a != b

    def test_different_encodings_produce_different_ids(self):
        a = CheckpointManager.get_experiment_id(
            "resources", "iqp", {"n_features": 4}, "resources",
        )
        b = CheckpointManager.get_experiment_id(
            "resources", "angle", {"n_features": 4}, "resources",
        )
        assert a != b

    def test_different_stages_produce_different_ids(self):
        a = CheckpointManager.get_experiment_id(
            "resources", "iqp", {"n_features": 4}, "resources",
        )
        b = CheckpointManager.get_experiment_id(
            "expressibility", "iqp", {"n_features": 4}, "expressibility",
        )
        assert a != b

    def test_structure_has_four_parts(self):
        """ID format is ``stage/encoding/param_slug/metric``."""
        exp_id = CheckpointManager.get_experiment_id(
            "resources", "iqp", {"n_features": 4, "reps": 2}, "resources",
        )
        parts = exp_id.split("/")
        assert len(parts) == 4
        assert parts[0] == "resources"
        assert parts[1] == "iqp"
        assert parts[3] == "resources"

    def test_metric_defaults_to_stage(self):
        """When metric is None, it falls back to the stage name."""
        exp_id = CheckpointManager.get_experiment_id(
            "resources", "angle", {"n_features": 4}, None,
        )
        parts = exp_id.split("/")
        assert parts[3] == "resources"

    def test_empty_params(self):
        """Empty params dict produces a valid ID."""
        exp_id = CheckpointManager.get_experiment_id(
            "resources", "angle", {}, "resources",
        )
        assert "resources/angle/" in exp_id


# =============================================================================
# Core CRUD
# =============================================================================


class TestCheckpointCrud:
    """Mark-completed, is-completed, and get-result round trip."""

    def test_empty_checkpoint_has_no_completions(self, tmp_path):
        ckpt = _make_ckpt(tmp_path)
        assert ckpt.completed_count == 0

    def test_is_completed_false_for_unknown_id(self, tmp_path):
        ckpt = _make_ckpt(tmp_path)
        assert ckpt.is_completed("resources/iqp/nf4/resources") is False

    def test_mark_and_check(self, tmp_path):
        ckpt = _make_ckpt(tmp_path)
        exp_id = "resources/iqp/nf4_abc12345/resources"
        result = {"gate_count": 52, "depth": 6}

        ckpt.mark_completed(exp_id, result)

        assert ckpt.is_completed(exp_id) is True
        assert ckpt.completed_count == 1

    def test_get_result_returns_stored_data(self, tmp_path):
        ckpt = _make_ckpt(tmp_path)
        exp_id = "resources/angle/nf4_abc12345/resources"
        result = {"gate_count": 4, "depth": 1}

        ckpt.mark_completed(exp_id, result)

        retrieved = ckpt.get_result(exp_id)
        assert retrieved == result

    def test_get_result_returns_none_for_unknown(self, tmp_path):
        ckpt = _make_ckpt(tmp_path)
        assert ckpt.get_result("nonexistent/id/foo/bar") is None

    def test_multiple_tasks(self, tmp_path):
        ckpt = _make_ckpt(tmp_path)
        for i in range(5):
            exp_id = f"resources/enc{i}/nf4_abc/resources"
            ckpt.mark_completed(exp_id, {"value": i})

        assert ckpt.completed_count == 5
        for i in range(5):
            exp_id = f"resources/enc{i}/nf4_abc/resources"
            assert ckpt.is_completed(exp_id)
            assert ckpt.get_result(exp_id) == {"value": i}

    def test_get_all_completed(self, tmp_path):
        ckpt = _make_ckpt(tmp_path)
        ckpt.mark_completed("a/b/c/d", {"v": 1})
        ckpt.mark_completed("e/f/g/h", {"v": 2})

        all_records = ckpt.get_all_completed()
        assert len(all_records) == 2
        assert "a/b/c/d" in all_records
        assert "e/f/g/h" in all_records


# =============================================================================
# Persistence and crash recovery
# =============================================================================


class TestCheckpointPersistence:
    """JSONL file persistence, reload, and crash recovery."""

    def test_persists_to_disk(self, tmp_path):
        ckpt = _make_ckpt(tmp_path)
        ckpt.mark_completed("a/b/c/d", {"v": 1})

        log_file = os.path.join(str(tmp_path), "completed.jsonl")
        assert os.path.isfile(log_file)

        with open(log_file, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["id"] == "a/b/c/d"
        assert record["result"]["v"] == 1
        assert "ts" in record

    def test_reload_from_disk(self, tmp_path):
        """A fresh CheckpointManager reads back previously persisted results."""
        ckpt1 = _make_ckpt(tmp_path)
        ckpt1.mark_completed("a/b/c/d", {"v": 42})

        # Create a new manager pointing at the same directory.
        ckpt2 = _make_ckpt(tmp_path)
        assert ckpt2.is_completed("a/b/c/d")
        assert ckpt2.get_result("a/b/c/d") == {"v": 42}

    def test_append_only(self, tmp_path):
        """Successive writes append, never overwrite prior lines."""
        ckpt = _make_ckpt(tmp_path)
        ckpt.mark_completed("id1", {"v": 1})
        ckpt.mark_completed("id2", {"v": 2})

        log_file = os.path.join(str(tmp_path), "completed.jsonl")
        with open(log_file, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        assert len(lines) == 2

    def test_truncated_last_line_recovery(self, tmp_path):
        """A truncated last line (crash mid-write) is silently ignored."""
        log_file = os.path.join(str(tmp_path), "completed.jsonl")
        os.makedirs(str(tmp_path), exist_ok=True)

        # Write a valid line followed by a truncated line.
        with open(log_file, "w", encoding="utf-8") as fh:
            valid = json.dumps({"id": "good/id/x/y", "result": {"v": 1}, "ts": "t"})
            fh.write(valid + "\n")
            fh.write('{"id": "bad/id/x/y", "result": {"v": 2')  # truncated

        ckpt = _make_ckpt(tmp_path)
        assert ckpt.is_completed("good/id/x/y") is True
        assert ckpt.is_completed("bad/id/x/y") is False
        assert ckpt.completed_count == 1

    def test_empty_lines_ignored(self, tmp_path):
        """Blank lines in the log file are harmlessly skipped."""
        log_file = os.path.join(str(tmp_path), "completed.jsonl")
        os.makedirs(str(tmp_path), exist_ok=True)

        valid = json.dumps({"id": "a/b/c/d", "result": {"v": 1}, "ts": "t"})
        with open(log_file, "w", encoding="utf-8") as fh:
            fh.write("\n")
            fh.write(valid + "\n")
            fh.write("\n\n")

        ckpt = _make_ckpt(tmp_path)
        assert ckpt.completed_count == 1

    def test_last_write_wins_for_duplicate_ids(self, tmp_path):
        """If an ID appears twice, the later record takes precedence."""
        log_file = os.path.join(str(tmp_path), "completed.jsonl")
        os.makedirs(str(tmp_path), exist_ok=True)

        line1 = json.dumps({"id": "dup/id/x/y", "result": {"v": "old"}, "ts": "t1"})
        line2 = json.dumps({"id": "dup/id/x/y", "result": {"v": "new"}, "ts": "t2"})
        with open(log_file, "w", encoding="utf-8") as fh:
            fh.write(line1 + "\n")
            fh.write(line2 + "\n")

        ckpt = _make_ckpt(tmp_path)
        assert ckpt.get_result("dup/id/x/y") == {"v": "new"}
        # Still counts as 1 unique ID.
        assert ckpt.completed_count == 1


# =============================================================================
# Clear
# =============================================================================


class TestCheckpointClear:
    """Tests for the destructive clear() method."""

    def test_clear_removes_file(self, tmp_path):
        ckpt = _make_ckpt(tmp_path)
        ckpt.mark_completed("a/b/c/d", {"v": 1})

        log_file = os.path.join(str(tmp_path), "completed.jsonl")
        assert os.path.isfile(log_file)

        ckpt.clear()
        assert not os.path.isfile(log_file)
        assert ckpt.completed_count == 0

    def test_clear_then_mark(self, tmp_path):
        """After clear, new completions work normally."""
        ckpt = _make_ckpt(tmp_path)
        ckpt.mark_completed("a/b/c/d", {"v": 1})
        ckpt.clear()

        ckpt.mark_completed("e/f/g/h", {"v": 2})
        assert ckpt.completed_count == 1
        assert ckpt.is_completed("a/b/c/d") is False
        assert ckpt.is_completed("e/f/g/h") is True


# =============================================================================
# Numpy serialisation
# =============================================================================


class TestNumpySerialisation:
    """Checkpoint must handle numpy scalars and arrays in result dicts."""

    def test_numpy_int(self, tmp_path):
        ckpt = _make_ckpt(tmp_path)
        ckpt.mark_completed("a/b/c/d", {"gate_count": np.int64(52)})
        assert ckpt.get_result("a/b/c/d")["gate_count"] == 52

    def test_numpy_float(self, tmp_path):
        ckpt = _make_ckpt(tmp_path)
        ckpt.mark_completed("a/b/c/d", {"ratio": np.float64(0.692)})
        assert abs(ckpt.get_result("a/b/c/d")["ratio"] - 0.692) < 1e-10

    def test_numpy_array(self, tmp_path):
        ckpt = _make_ckpt(tmp_path)
        ckpt.mark_completed("a/b/c/d", {"values": np.array([1.0, 2.0, 3.0])})
        assert list(ckpt.get_result("a/b/c/d")["values"]) == [1.0, 2.0, 3.0]

    def test_numpy_bool(self, tmp_path):
        ckpt = _make_ckpt(tmp_path)
        ckpt.mark_completed("a/b/c/d", {"flag": np.bool_(True)})
        assert ckpt.get_result("a/b/c/d")["flag"] == True  # noqa: E712


# =============================================================================
# Thread safety
# =============================================================================


class TestCheckpointThreadSafety:
    """Concurrent writes must not corrupt the log or lose data."""

    def test_concurrent_writes(self, tmp_path):
        ckpt = _make_ckpt(tmp_path)
        n_threads = 8
        ids_per_thread = 10
        errors: list[str] = []

        def worker(thread_id: int) -> None:
            for i in range(ids_per_thread):
                exp_id = f"stage/enc{thread_id}/task{i}_hash/metric"
                try:
                    ckpt.mark_completed(exp_id, {"tid": thread_id, "i": i})
                except Exception as exc:
                    errors.append(f"Thread {thread_id}, task {i}: {exc}")

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Errors during concurrent writes: {errors}"
        assert ckpt.completed_count == n_threads * ids_per_thread

        # Verify all data is retrievable.
        for tid in range(n_threads):
            for i in range(ids_per_thread):
                exp_id = f"stage/enc{tid}/task{i}_hash/metric"
                assert ckpt.is_completed(exp_id), f"Missing: {exp_id}"
