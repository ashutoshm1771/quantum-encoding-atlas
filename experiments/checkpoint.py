"""Crash-safe, append-only checkpoint management.

This module provides a :class:`CheckpointManager` that tracks which
experiment tasks have been completed. It uses a simple append-only
JSONL (JSON Lines) file where each line is one completed task record.

Design Goals
------------
- **Crash-safe**: Only complete lines are parsed; a truncated last line
  (from a crash mid-write) is silently ignored.
- **Append-only**: Completed tasks are appended, never overwritten.
  This avoids data loss from concurrent writes or interrupted flushes.
- **Deterministic IDs**: Experiment IDs are derived from
  ``(stage, encoding_name, params, metric)`` so that the same task
  always maps to the same ID, enabling reliable resume.

File Format
-----------
Each line in the checkpoint file is a JSON object::

    {"id": "expressibility/iqp/nf4_reps2/expressibility", "result": {...}, "ts": "..."}

The ``id`` field is the deterministic experiment ID. The ``result``
field stores the analysis output. The ``ts`` field is an ISO-8601
timestamp for human inspection.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage experiment checkpoints via an append-only JSONL log.

    Parameters
    ----------
    checkpoint_dir : str
        Directory where the checkpoint file is stored. Created
        automatically if it does not exist.
    filename : str, optional
        Name of the JSONL log file within *checkpoint_dir*
        (default ``"completed.jsonl"``).

    Examples
    --------
    >>> ckpt = CheckpointManager("/tmp/my_experiment")
    >>> exp_id = ckpt.get_experiment_id("expr", "iqp", {"n_features": 4}, "expressibility")
    >>> if not ckpt.is_completed(exp_id):
    ...     result = run_experiment(...)
    ...     ckpt.mark_completed(exp_id, result)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        filename: str = "completed.jsonl",
    ) -> None:
        self._checkpoint_dir = os.path.abspath(checkpoint_dir)
        self._filepath = os.path.join(self._checkpoint_dir, filename)
        self._lock = threading.Lock()

        # In-memory index of completed experiment IDs.
        # Populated lazily on first access via _ensure_loaded().
        self._completed: dict[str, dict[str, Any]] | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_completed(self, experiment_id: str) -> bool:
        """Check whether a task has already been completed.

        Parameters
        ----------
        experiment_id : str
            Deterministic experiment identifier (see :meth:`get_experiment_id`).

        Returns
        -------
        bool
            ``True`` if the experiment was previously marked completed.
        """
        self._ensure_loaded()
        assert self._completed is not None  # guaranteed by _ensure_loaded
        return experiment_id in self._completed

    def get_result(self, experiment_id: str) -> dict[str, Any] | None:
        """Retrieve the cached result for a completed experiment.

        Parameters
        ----------
        experiment_id : str
            Deterministic experiment identifier.

        Returns
        -------
        dict or None
            The stored result dict, or ``None`` if not completed.
        """
        self._ensure_loaded()
        assert self._completed is not None
        record = self._completed.get(experiment_id)
        if record is None:
            return None
        return record.get("result")

    def mark_completed(
        self,
        experiment_id: str,
        result: dict[str, Any],
    ) -> None:
        """Record a task as completed with its result.

        Appends a single JSON line to the checkpoint file and updates
        the in-memory index. The write is flushed immediately.

        Parameters
        ----------
        experiment_id : str
            Deterministic experiment identifier.
        result : dict[str, Any]
            JSON-serializable result dictionary to persist.

        Raises
        ------
        TypeError
            If *result* is not JSON-serializable.
        """
        self._ensure_loaded()
        assert self._completed is not None

        record: dict[str, Any] = {
            "id": experiment_id,
            "result": result,
            "ts": datetime.now(timezone.utc).isoformat(),
        }

        # Validate serializability before touching the file.
        line = json.dumps(record, default=_json_default)

        with self._lock:
            os.makedirs(self._checkpoint_dir, exist_ok=True)
            with open(self._filepath, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
                fh.flush()
                os.fsync(fh.fileno())

            self._completed[experiment_id] = record

        logger.debug("Checkpoint saved: %s", experiment_id)

    def get_all_completed(self) -> dict[str, dict[str, Any]]:
        """Return all completed experiment records.

        Returns
        -------
        dict[str, dict]
            Mapping from experiment ID to full record dict
            (including ``result`` and ``ts`` fields).
        """
        self._ensure_loaded()
        assert self._completed is not None
        # Return a shallow copy so callers cannot mutate the index.
        return dict(self._completed)

    @property
    def completed_count(self) -> int:
        """Number of completed experiments in the checkpoint."""
        self._ensure_loaded()
        assert self._completed is not None
        return len(self._completed)

    def clear(self) -> None:
        """Delete the checkpoint file and reset in-memory state.

        .. warning::
            This is destructive. Use only for testing or intentional resets.
        """
        with self._lock:
            if os.path.isfile(self._filepath):
                os.remove(self._filepath)
            self._completed = {}
        logger.info("Checkpoint cleared: %s", self._filepath)

    # ------------------------------------------------------------------
    # Experiment ID construction
    # ------------------------------------------------------------------

    @staticmethod
    def get_experiment_id(
        stage: str,
        encoding_name: str,
        encoding_params: dict[str, Any],
        metric: str | None = None,
    ) -> str:
        """Build a deterministic experiment identifier.

        The ID has the form ``stage/encoding_name/param_hash[/metric]``
        where *param_hash* is a short, human-readable summary of the
        encoding parameters plus a truncated SHA-256 for uniqueness.

        Parameters
        ----------
        stage : str
            Experiment stage name (e.g. ``"expressibility"``).
        encoding_name : str
            Registry name of the encoding (e.g. ``"iqp"``).
        encoding_params : dict[str, Any]
            Parameters used to instantiate the encoding.
        metric : str or None, optional
            Analysis metric name. If ``None``, defaults to *stage*.

        Returns
        -------
        str
            Deterministic experiment identifier string.

        Examples
        --------
        >>> CheckpointManager.get_experiment_id(
        ...     "expressibility", "iqp",
        ...     {"n_features": 4, "reps": 2}, "expressibility"
        ... )
        'expressibility/iqp/nf4_reps2_a1b2c3/expressibility'
        """
        if metric is None:
            metric = stage

        # Build a human-readable param summary (sorted for determinism).
        parts: list[str] = []
        for key in sorted(encoding_params.keys()):
            val = encoding_params[key]
            # Shorten common keys for readability.
            short_key = _SHORT_KEYS.get(key, key)
            parts.append(f"{short_key}{val}")

        readable = "_".join(parts) if parts else "default"

        # Append a short hash to guarantee uniqueness even if the
        # readable summary collides (e.g. truncated float values).
        param_json = json.dumps(encoding_params, sort_keys=True, default=str)
        digest = hashlib.sha256(param_json.encode("utf-8")).hexdigest()[:8]
        param_slug = f"{readable}_{digest}"

        return f"{stage}/{encoding_name}/{param_slug}/{metric}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load the checkpoint file into memory if not already loaded.

        Handles missing files (fresh start) and truncated last lines
        (crash recovery) gracefully.
        """
        if self._completed is not None:
            return

        with self._lock:
            # Double-check after acquiring lock.
            if self._completed is not None:
                return

            loaded: dict[str, dict[str, Any]] = {}

            if os.path.isfile(self._filepath):
                with open(self._filepath, "r", encoding="utf-8") as fh:
                    for line_no, raw_line in enumerate(fh, start=1):
                        raw_line = raw_line.strip()
                        if not raw_line:
                            continue
                        try:
                            record = json.loads(raw_line)
                        except json.JSONDecodeError:
                            # Truncated last line from a crash — skip it.
                            logger.warning(
                                "Ignoring corrupt checkpoint line %d in %s",
                                line_no,
                                self._filepath,
                            )
                            continue

                        exp_id = record.get("id")
                        if exp_id is None:
                            logger.warning(
                                "Checkpoint line %d missing 'id' field, skipping.",
                                line_no,
                            )
                            continue

                        # Last-write-wins if an ID appears multiple times
                        # (e.g. manual re-run after deleting a line).
                        loaded[exp_id] = record

                logger.info(
                    "Loaded %d completed experiments from %s",
                    len(loaded),
                    self._filepath,
                )
            else:
                logger.info(
                    "No checkpoint file found at %s — starting fresh.",
                    self._filepath,
                )

            self._completed = loaded


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# Abbreviations for common parameter names to keep IDs readable.
_SHORT_KEYS: dict[str, str] = {
    "n_features": "nf",
    "reps": "reps",
    "n_layers": "nl",
    "entanglement": "ent",
    "rotation": "rot",
    "order": "ord",
    "steps": "steps",
    "normalize": "norm",
}


def _json_default(obj: Any) -> Any:
    """Fallback serializer for non-standard types (e.g. numpy scalars)."""
    # Handle numpy types without importing numpy at module level.
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
