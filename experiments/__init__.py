"""Phase 4 experiment framework for Quantum Encoding Atlas.

This package provides a unified, checkpointed experiment runner for
systematic evaluation of quantum encoding properties. It supports
resource profiling, simulability classification, expressibility analysis,
entanglement capability measurement, and trainability estimation.

Modules
-------
config : Experiment configuration loading and validation.
checkpoint : Crash-safe, append-only checkpoint management.
results : Result aggregation and export utilities.
runner : Unified experiment runner with automatic dispatch.

Usage
-----
Run experiments via the CLI entry point::

    python -m experiments.run_stage --config experiments/configs/stage1_resources.json

Or programmatically::

    from experiments.runner import ExperimentRunner
    runner = ExperimentRunner("experiments/configs/stage1_resources.json")
    runner.run()
"""

from __future__ import annotations

__all__ = [
    "CheckpointManager",
    "ExperimentConfig",
    "ExperimentRunner",
    "ResultStore",
    "load_config",
]

from experiments.checkpoint import CheckpointManager
from experiments.config import ExperimentConfig, load_config
from experiments.results import ResultStore
from experiments.runner import ExperimentRunner
