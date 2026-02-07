"""Result aggregation, export, and querying utilities.

This module provides a :class:`ResultStore` that collects raw results
from completed checkpoint records, aggregates them by encoding or
parameter, and exports summary tables in JSON and Markdown formats.

It operates purely on the data already persisted by the checkpoint
system — no new simulations are triggered.

Usage
-----
::

    from experiments.checkpoint import CheckpointManager
    from experiments.results import ResultStore

    ckpt = CheckpointManager("experiments/results/raw/stage3")
    store = ResultStore.from_checkpoint(ckpt)
    store.save_summary_json("experiments/results/raw/stage3/summary.json")
    print(store.to_markdown_table(metric_keys=["expressibility"]))
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)


@dataclass
class TaskResult:
    """Container for one completed experiment task.

    Attributes
    ----------
    experiment_id : str
        Deterministic experiment identifier.
    stage : str
        Experiment stage (parsed from *experiment_id*).
    encoding_name : str
        Registry name of the encoding.
    encoding_params : dict[str, Any]
        Parameters used for encoding instantiation.
    metric : str
        Analysis metric name.
    result : dict[str, Any]
        Raw result dictionary from the analysis function.
    status : str
        Execution status: ``"success"``, ``"failed"``, or ``"skipped"``.
    wall_time : float
        Wall-clock time in seconds for this task.
    """

    experiment_id: str
    stage: str
    encoding_name: str
    encoding_params: dict[str, Any]
    metric: str
    result: dict[str, Any]
    status: str
    wall_time: float = 0.0


class ResultStore:
    """In-memory store of experiment results with query and export methods.

    Parameters
    ----------
    results : list[TaskResult], optional
        Pre-populated list of results (default empty).
    """

    def __init__(self, results: list[TaskResult] | None = None) -> None:
        self._results: list[TaskResult] = list(results) if results else []

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, checkpoint: Any) -> ResultStore:
        """Build a ResultStore from a :class:`CheckpointManager`.

        Iterates over all completed records in the checkpoint and
        parses each into a :class:`TaskResult`.

        Parameters
        ----------
        checkpoint : CheckpointManager
            A loaded checkpoint manager instance.

        Returns
        -------
        ResultStore
            Populated store ready for querying and export.
        """
        store = cls()
        all_records = checkpoint.get_all_completed()

        for exp_id, record in all_records.items():
            result_data = record.get("result", {})
            task = _parse_task_result(exp_id, result_data)
            store._results.append(task)

        logger.info(
            "ResultStore loaded %d task results from checkpoint.", len(store._results)
        )
        return store

    @classmethod
    def from_json(cls, path: str) -> ResultStore:
        """Load a ResultStore from a previously saved summary JSON.

        Parameters
        ----------
        path : str
            Path to the summary JSON file.

        Returns
        -------
        ResultStore
            Populated store.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        results: list[TaskResult] = []
        for entry in data.get("results", []):
            results.append(TaskResult(
                experiment_id=entry.get("experiment_id", ""),
                stage=entry.get("stage", ""),
                encoding_name=entry.get("encoding_name", ""),
                encoding_params=entry.get("encoding_params", {}),
                metric=entry.get("metric", ""),
                result=entry.get("result", {}),
                status=entry.get("status", "success"),
                wall_time=entry.get("wall_time", 0.0),
            ))

        return cls(results)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of results in the store."""
        return len(self._results)

    def filter(
        self,
        *,
        stage: str | None = None,
        encoding_name: str | None = None,
        status: str | None = None,
        metric: str | None = None,
    ) -> list[TaskResult]:
        """Filter results by field values.

        Parameters
        ----------
        stage : str or None
            Filter by stage name.
        encoding_name : str or None
            Filter by encoding registry name.
        status : str or None
            Filter by status (``"success"``, ``"failed"``, ``"skipped"``).
        metric : str or None
            Filter by metric name.

        Returns
        -------
        list[TaskResult]
            Matching results (may be empty).
        """
        out: list[TaskResult] = []
        for r in self._results:
            if stage is not None and r.stage != stage:
                continue
            if encoding_name is not None and r.encoding_name != encoding_name:
                continue
            if status is not None and r.status != status:
                continue
            if metric is not None and r.metric != metric:
                continue
            out.append(r)
        return out

    def get_metric_values(
        self,
        metric_key: str,
        *,
        encoding_name: str | None = None,
    ) -> dict[str, list[float]]:
        """Extract a specific metric value grouped by encoding name.

        Parameters
        ----------
        metric_key : str
            Key to look up in each task's ``result`` dict
            (e.g. ``"expressibility"``, ``"gate_count"``).
        encoding_name : str or None
            If provided, restrict to this encoding only.

        Returns
        -------
        dict[str, list[float]]
            Mapping from encoding name to list of metric values
            across all matching tasks.
        """
        grouped: dict[str, list[float]] = {}
        for r in self._results:
            if r.status != "success":
                continue
            if encoding_name is not None and r.encoding_name != encoding_name:
                continue
            val = r.result.get(metric_key)
            if val is not None:
                grouped.setdefault(r.encoding_name, []).append(float(val))
        return grouped

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def save_summary_json(self, path: str) -> None:
        """Save all results to a structured JSON file.

        Parameters
        ----------
        path : str
            Output file path. Parent directories are created as needed.
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        data = {
            "total_results": len(self._results),
            "success_count": sum(1 for r in self._results if r.status == "success"),
            "failed_count": sum(1 for r in self._results if r.status == "failed"),
            "skipped_count": sum(1 for r in self._results if r.status == "skipped"),
            "results": [
                {
                    "experiment_id": r.experiment_id,
                    "stage": r.stage,
                    "encoding_name": r.encoding_name,
                    "encoding_params": r.encoding_params,
                    "metric": r.metric,
                    "result": r.result,
                    "status": r.status,
                    "wall_time": r.wall_time,
                }
                for r in self._results
            ],
        }

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, default=_json_default)

        logger.info("Summary saved to %s (%d results)", path, len(self._results))

    def to_markdown_table(
        self,
        metric_keys: Sequence[str],
        *,
        sort_by: str | None = None,
        descending: bool = True,
    ) -> str:
        """Render results as a Markdown table.

        One row per (encoding_name, encoding_params) combination.
        Columns are encoding name, parameters, and the requested metrics.

        Parameters
        ----------
        metric_keys : sequence of str
            Which metric keys to include as columns.
        sort_by : str or None
            Metric key to sort rows by. If ``None``, rows appear in
            insertion order.
        descending : bool
            Sort direction (default descending / highest first).

        Returns
        -------
        str
            Markdown-formatted table string.
        """
        # Collect rows: one per successful task.
        rows: list[dict[str, Any]] = []
        for r in self._results:
            if r.status != "success":
                continue
            row: dict[str, Any] = {
                "encoding": r.encoding_name,
                "params": _format_params(r.encoding_params),
            }
            for key in metric_keys:
                val = r.result.get(key)
                row[key] = f"{val:.4f}" if isinstance(val, float) else str(val)
            rows.append(row)

        if sort_by is not None and rows:
            rows.sort(
                key=lambda r: _safe_float(r.get(sort_by, "0")),
                reverse=descending,
            )

        # Build Markdown.
        headers = ["Encoding", "Params"] + list(metric_keys)
        lines: list[str] = []
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")

        for row in rows:
            cells = [row["encoding"], row["params"]]
            cells.extend(row.get(k, "—") for k in metric_keys)
            lines.append("| " + " | ".join(cells) + " |")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def add(self, task: TaskResult) -> None:
        """Add a single task result to the store.

        Parameters
        ----------
        task : TaskResult
            The result to add.
        """
        self._results.append(task)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _parse_task_result(experiment_id: str, result_data: dict[str, Any]) -> TaskResult:
    """Parse a checkpoint record into a TaskResult.

    The experiment_id format is ``stage/encoding/param_slug/metric``.
    """
    parts = experiment_id.split("/")
    if len(parts) >= 4:
        stage, encoding_name, _, metric = parts[0], parts[1], parts[2], parts[3]
    elif len(parts) == 3:
        stage, encoding_name, metric = parts[0], parts[1], parts[2]
    else:
        stage = parts[0] if parts else "unknown"
        encoding_name = parts[1] if len(parts) > 1 else "unknown"
        metric = "unknown"

    return TaskResult(
        experiment_id=experiment_id,
        stage=stage,
        encoding_name=encoding_name,
        encoding_params=result_data.get("encoding_params", {}),
        metric=metric,
        result=result_data,
        status=result_data.get("status", "success"),
        wall_time=result_data.get("wall_time", 0.0),
    )


def _format_params(params: dict[str, Any]) -> str:
    """Format a params dict into a short human-readable string."""
    if not params:
        return "default"
    parts = []
    for k in sorted(params.keys()):
        v = params[k]
        parts.append(f"{k}={v}")
    return ", ".join(parts)


def _safe_float(val: Any) -> float:
    """Try to parse a value as float; return 0.0 on failure."""
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _json_default(obj: Any) -> Any:
    """Fallback serializer for non-standard types."""
    type_name = type(obj).__name__
    if "int" in type_name and hasattr(obj, "item"):
        return int(obj.item())
    if "float" in type_name and hasattr(obj, "item"):
        return float(obj.item())
    if "ndarray" in type_name and hasattr(obj, "tolist"):
        return obj.tolist()
    if "bool_" in type_name and hasattr(obj, "item"):
        return bool(obj.item())
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
