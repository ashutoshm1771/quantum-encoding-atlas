"""Experiment configuration loading and validation.

This module handles loading JSON configuration files that define which
encodings to benchmark, what parameter sweeps to perform, and which
analysis functions to invoke. It also provides quick-mode overrides
for fast validation runs and deterministic seed computation.

Configuration Format
--------------------
Each JSON config file specifies a single experiment stage::

    {
        "stage": "expressibility",
        "seed": 42,
        "backend": "pennylane",
        "encodings": [
            {
                "name": "iqp",
                "params": {
                    "n_features": [2, 4, 6, 8],
                    "reps": [1, 2, 3]
                }
            }
        ],
        "analysis_params": {
            "n_samples": 5000,
            "n_bins": 75
        },
        "output_dir": "experiments/results/raw/stage3"
    }

Parameter values that are lists are treated as sweep axes; the runner
generates the Cartesian product of all list-valued parameters. Scalar
values are fixed across all tasks.

Seed Strategy
-------------
All randomness is seeded deterministically for reproducibility::

    stage_seed  = base_seed + STAGE_OFFSETS[stage] * 1000
    task_seed   = stage_seed + task_index

This ensures independence between tasks and reproducibility after
checkpoint resume.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from dataclasses import dataclass, field
from itertools import product
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STAGES = frozenset({
    "resources",
    "simulability",
    "expressibility",
    "entanglement",
    "trainability",
    "noise",
    "vqc",
    "kernel",
    "tradeoff",
})

VALID_BACKENDS = frozenset({"pennylane", "qiskit", "cirq"})

# Deterministic offset per stage so seeds never collide across stages.
# Each stage_seed = base_seed + offset * 1000, so adjacent integer offsets
# are separated by 1000.  Half-integer offsets (e.g. 5.5) position sub-stages
# between their parent stages, giving a 500-unit gap on each side.
STAGE_OFFSETS: dict[str, int | float] = {
    "resources": 1,       # Stage 1  → seed 1042
    "simulability": 2,    # Stage 2  → seed 2042
    "expressibility": 3,  # Stage 3  → seed 3042
    "entanglement": 4,    # Stage 4  → seed 4042
    "trainability": 5,    # Stage 5  → seed 5042
    "noise": 5.5,         # Stage 5b → seed 5542
    "vqc": 6,             # Stage 6a → seed 6042
    "kernel": 7,          # Stage 6b → seed 7042
    "tradeoff": 8,        # Stage 7  → seed 8042
}

# Quick-mode overrides reduce sample counts for fast validation.
_QUICK_OVERRIDES: dict[str, dict[str, Any]] = {
    "expressibility": {"n_samples": 500, "n_bins": 25},
    "entanglement": {"n_samples": 100},
    "trainability": {"n_samples": 50},
    "noise": {
        "n_samples_expressibility": 100,
        "n_samples_entanglement": 50,
        "n_samples_fidelity": 25,
        "noise_levels": ["medium"],
    },
    "vqc": {
        "epochs": 20,
        "n_runs": 2,
        "n_folds": 3,
        "datasets": ["moons", "iris", "breast_cancer"],
        "classical_baselines": ["svm_rbf"],
    },
    "kernel": {
        "n_runs": 2,
        "n_folds": 3,
        "datasets": ["moons", "iris", "breast_cancer"],
    },
    "tradeoff": {
        "sensitivity_n_samples": 100,
        "generate_plots": False,
    },
}

# Required top-level keys in every config file.
_REQUIRED_KEYS = {"stage", "encodings"}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EncodingSpec:
    """A single (encoding_name, param_dict) pair ready for instantiation.

    Attributes
    ----------
    name : str
        Registry name for the encoding (e.g. ``"iqp"``, ``"angle"``).
    params : dict[str, Any]
        Keyword arguments passed to ``get_encoding(name, **params)``.
        All values are scalars (sweep already expanded).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("EncodingSpec.name must be a non-empty string.")
        for key, value in self.params.items():
            if isinstance(value, list):
                raise ValueError(
                    f"EncodingSpec.params['{key}'] is a list; sweep expansion "
                    "should have already been applied. This is a bug."
                )


@dataclass
class ExperimentConfig:
    """Validated, fully-expanded experiment configuration.

    Attributes
    ----------
    stage : str
        Analysis stage name (one of :data:`VALID_STAGES`).
    base_seed : int
        Base random seed for the entire experiment.
    backend : str
        Quantum simulation backend name.
    encoding_specs : list[EncodingSpec]
        Fully expanded list of (encoding, params) to evaluate.
        Each list element is one task unit.
    analysis_params : dict[str, Any]
        Parameters forwarded to the analysis function
        (e.g. ``n_samples``, ``n_bins``).
    output_dir : str
        Directory for writing results and checkpoints.
    quick : bool
        Whether quick-mode overrides are active.
    """

    stage: str
    base_seed: int
    backend: str
    encoding_specs: list[EncodingSpec]
    analysis_params: dict[str, Any]
    output_dir: str
    quick: bool = False
    smoke: bool = False

    # -- Seed helpers -------------------------------------------------------

    @property
    def stage_seed(self) -> int:
        """Deterministic seed for this stage.

        Returns ``int(base_seed + STAGE_OFFSETS[stage] * 1000)``.

        The ``int()`` cast is needed because half-integer offsets
        (e.g. 5.5 for the noise stage) produce a float intermediate.
        """
        return int(self.base_seed + STAGE_OFFSETS.get(self.stage, 0) * 1000)

    def task_seed(self, task_index: int) -> int:
        """Deterministic seed for a specific task.

        Parameters
        ----------
        task_index : int
            Zero-based index of the task within this stage.

        Returns
        -------
        int
            ``stage_seed + task_index``
        """
        return self.stage_seed + task_index

    # -- Classification-stage seed helpers (Stages 6a / 6b) ----------------
    #
    # These implement the seed strategy from PHASE_4_EXPERIMENT_PLAN.md §Seed
    # Strategy.  The key invariant is that **CV fold generation** uses a seed
    # that depends only on (stage, dataset) — never on encoding — so that all
    # encodings and classical baselines are evaluated on identical train/test
    # splits, enabling valid paired statistical tests (Wilcoxon signed-rank).

    def cv_fold_seed(self, dataset_index: int) -> int:
        """Seed for CV fold generation — shared across all encodings *and* stages.

        .. math::
            \\text{cv\\_fold\\_seed} = \\text{base\\_seed} + \\text{dataset\\_index}

        Uses ``base_seed`` (not ``stage_seed``) so that VQC (Stage 6a),
        kernel (Stage 6b), and classical baselines all generate **identical**
        CV folds and synthetic data realisations for the same dataset.
        This is required for:

        - Paired statistical tests (Wilcoxon signed-rank) within each stage.
        - Cross-paradigm ranking comparison (H6) between VQC and kernel
          stages, which the experiment plan specifies as "same folds as 6a".

        The seed is also used for ``load_dataset()`` so that synthetic
        datasets (moons, circles, linear, xor) produce the same noise
        realisation regardless of which stage or encoding is being evaluated.

        Parameters
        ----------
        dataset_index : int
            Zero-based position of the dataset in the experiment's dataset
            list.  Must be consistent across stages (i.e. the ``datasets``
            list in each stage config should have the same order).

        Returns
        -------
        int
        """
        return self.base_seed + dataset_index

    def vqc_run_seed(
        self,
        encoding_index: int,
        dataset_index: int,
        run_index: int,
    ) -> int:
        """Seed for VQC parameter initialisation (Stage 6a).

        .. math::
            \\text{vqc\\_run\\_seed} = \\text{stage\\_seed}
            + \\text{encoding\\_index} \\times 100
            + \\text{dataset\\_index} \\times 10
            + \\text{run\\_index}

        Parameters
        ----------
        encoding_index : int
            Zero-based index of the encoding in the task list.
        dataset_index : int
            Zero-based position of the dataset.
        run_index : int
            Zero-based repetition index.

        Returns
        -------
        int
        """
        return (
            self.stage_seed
            + encoding_index * 100
            + dataset_index * 10
            + run_index
        )

    def kernel_run_seed(
        self,
        encoding_index: int,
        dataset_index: int,
        run_index: int,
    ) -> int:
        """Seed for kernel SVM random state (Stage 6b).

        .. math::
            \\text{kernel\\_run\\_seed} = \\text{stage\\_seed} + 500
            + \\text{encoding\\_index} \\times 100
            + \\text{dataset\\_index} \\times 10
            + \\text{run\\_index}

        The +500 offset separates kernel seeds from VQC seeds so that
        the two stages never share a seed for the same logical position.

        Parameters
        ----------
        encoding_index : int
            Zero-based index of the encoding in the task list.
        dataset_index : int
            Zero-based position of the dataset.
        run_index : int
            Zero-based repetition index.

        Returns
        -------
        int
        """
        return (
            self.stage_seed
            + 500
            + encoding_index * 100
            + dataset_index * 10
            + run_index
        )

    @staticmethod
    def fold_init_seed(run_seed: int, fold_index: int) -> int:
        """Per-fold seed for model parameter initialisation.

        Offsets the *run_seed* (from :meth:`vqc_run_seed` or
        :meth:`kernel_run_seed`) by ``fold_index * 2000`` to guarantee
        non-overlapping seeds across folds while keeping them
        deterministic.

        Parameters
        ----------
        run_seed : int
            The run-level seed (encoding + dataset + run dependent).
        fold_index : int
            Zero-based fold index within the CV loop.

        Returns
        -------
        int
        """
        return run_seed + fold_index * 2000


# ---------------------------------------------------------------------------
# Sweep expansion
# ---------------------------------------------------------------------------

def _expand_param_sweep(params: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand a parameter dict with list values into all combinations.

    Scalar values are kept fixed. List values are treated as sweep axes
    and the Cartesian product of all lists is generated.

    Parameters
    ----------
    params : dict[str, Any]
        Parameter dictionary. Values may be scalars or lists.

    Returns
    -------
    list[dict[str, Any]]
        List of parameter dicts, one per combination. Each dict
        contains only scalar values.

    Examples
    --------
    >>> _expand_param_sweep({"n_features": [2, 4], "reps": [1, 2]})
    [
        {"n_features": 2, "reps": 1},
        {"n_features": 2, "reps": 2},
        {"n_features": 4, "reps": 1},
        {"n_features": 4, "reps": 2},
    ]

    >>> _expand_param_sweep({"n_features": 4, "reps": 2})
    [{"n_features": 4, "reps": 2}]
    """
    if not params:
        return [{}]

    sweep_keys: list[str] = []
    sweep_values: list[list[Any]] = []
    fixed: dict[str, Any] = {}

    for key, value in sorted(params.items()):
        if isinstance(value, list):
            if not value:
                raise ValueError(
                    f"Parameter '{key}' is an empty list. "
                    "Sweep axes must have at least one value."
                )
            sweep_keys.append(key)
            sweep_values.append(value)
        else:
            fixed[key] = value

    if not sweep_keys:
        return [dict(fixed)]

    expanded: list[dict[str, Any]] = []
    for combo in product(*sweep_values):
        entry = dict(fixed)
        for key, val in zip(sweep_keys, combo):
            entry[key] = val
        expanded.append(entry)

    return expanded


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _validate_raw_config(raw: dict[str, Any]) -> None:
    """Validate the structure of a raw config dict.

    Parameters
    ----------
    raw : dict[str, Any]
        Parsed JSON config.

    Raises
    ------
    ValueError
        If required keys are missing, stage is unknown, or encodings
        list is malformed.
    """
    missing = _REQUIRED_KEYS - raw.keys()
    if missing:
        raise ValueError(
            f"Config is missing required keys: {sorted(missing)}"
        )

    stage = raw["stage"]
    if stage not in VALID_STAGES:
        raise ValueError(
            f"Unknown stage '{stage}'. Valid stages: {sorted(VALID_STAGES)}"
        )

    encodings = raw["encodings"]
    if not isinstance(encodings, list) or not encodings:
        raise ValueError("'encodings' must be a non-empty list.")

    for i, enc in enumerate(encodings):
        if not isinstance(enc, dict):
            raise ValueError(f"encodings[{i}] must be a dict, got {type(enc).__name__}.")
        if "name" not in enc:
            raise ValueError(f"encodings[{i}] is missing 'name' key.")

    backend = raw.get("backend", "pennylane")
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"Unknown backend '{backend}'. Valid: {sorted(VALID_BACKENDS)}"
        )


def load_config(
    path: str,
    *,
    quick: bool = False,
    smoke: bool = False,
) -> ExperimentConfig:
    """Load and validate an experiment configuration from a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON configuration file.
    quick : bool, optional
        If ``True``, override sample counts and sweep ranges with
        reduced values for fast validation (default ``False``).
    smoke : bool, optional
        If ``True``, apply quick-mode overrides **plus** cap dataset
        sizes to 20 samples (via ``max_samples`` in analysis_params).
        Designed for fast end-to-end pipeline validation where
        statistical results are meaningless but every code path is
        exercised (default ``False``).

    Returns
    -------
    ExperimentConfig
        Fully validated and sweep-expanded configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    json.JSONDecodeError
        If the file is not valid JSON.
    ValueError
        If the config fails structural validation.
    """
    # Smoke implies quick — all quick-mode overrides are applied first,
    # then the additional smoke-mode dataset cap is layered on top.
    if smoke:
        quick = True
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as fh:
        raw: dict[str, Any] = json.load(fh)

    _validate_raw_config(raw)

    stage: str = raw["stage"]
    base_seed: int = int(raw.get("seed", 42))
    backend: str = raw.get("backend", "pennylane")
    analysis_params: dict[str, Any] = dict(raw.get("analysis_params", {}))
    output_dir: str = raw.get(
        "output_dir",
        os.path.join("experiments", "results", "raw", stage),
    )

    # Apply quick-mode overrides before expansion.
    if quick and stage in _QUICK_OVERRIDES:
        overrides = _QUICK_OVERRIDES[stage]
        for key, val in overrides.items():
            analysis_params[key] = val
        # Propagate epochs override into nested vqc_params if present.
        # The VQC runner reads epochs from vqc_params (the nested dict),
        # so a top-level override alone would be silently ignored.
        if "epochs" in overrides and "vqc_params" in analysis_params:
            analysis_params["vqc_params"]["epochs"] = overrides["epochs"]
        logger.info("Quick-mode overrides applied for stage '%s': %s", stage, overrides)

    # Smoke-mode: additionally cap dataset samples for e2e validation.
    # This applies to stages that load datasets (vqc, kernel).  The
    # ``max_samples`` key is read by the stage handlers and forwarded
    # to ``load_dataset()``.
    if smoke:
        _SMOKE_MAX_SAMPLES = 20
        analysis_params.setdefault("max_samples", _SMOKE_MAX_SAMPLES)
        logger.info(
            "Smoke-mode: max_samples capped at %d",
            analysis_params["max_samples"],
        )

    # Expand encoding param sweeps into individual specs.
    encoding_specs: list[EncodingSpec] = []
    for enc_entry in raw["encodings"]:
        name: str = enc_entry["name"]
        raw_params: dict[str, Any] = dict(enc_entry.get("params", {}))

        # In quick mode, reduce n_features sweeps to [2, 4] and reps to [1, 2].
        if quick:
            if "n_features" in raw_params and isinstance(raw_params["n_features"], list):
                raw_params["n_features"] = [v for v in raw_params["n_features"] if v <= 4]
                if not raw_params["n_features"]:
                    raw_params["n_features"] = [2]
            if "reps" in raw_params and isinstance(raw_params["reps"], list):
                raw_params["reps"] = [v for v in raw_params["reps"] if v <= 2]
                if not raw_params["reps"]:
                    raw_params["reps"] = [1]

        for param_combo in _expand_param_sweep(raw_params):
            encoding_specs.append(EncodingSpec(name=name, params=param_combo))

    if not encoding_specs:
        raise ValueError("No encoding specs generated after sweep expansion.")

    config = ExperimentConfig(
        stage=stage,
        base_seed=base_seed,
        backend=backend,
        encoding_specs=encoding_specs,
        analysis_params=analysis_params,
        output_dir=output_dir,
        quick=quick,
        smoke=smoke,
    )

    logger.info(
        "Loaded config: stage=%s, %d encoding specs, output=%s",
        stage,
        len(encoding_specs),
        output_dir,
    )

    return config


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    """Serialize an ExperimentConfig to a JSON-compatible dict.

    Useful for saving the effective config alongside results for
    full reproducibility.

    Parameters
    ----------
    config : ExperimentConfig
        The configuration to serialize.

    Returns
    -------
    dict[str, Any]
        JSON-serializable dictionary.
    """
    return {
        "stage": config.stage,
        "base_seed": config.base_seed,
        "backend": config.backend,
        "encoding_specs": [
            {"name": spec.name, "params": dict(spec.params)}
            for spec in config.encoding_specs
        ],
        "analysis_params": dict(config.analysis_params),
        "output_dir": config.output_dir,
        "quick": config.quick,
        "smoke": config.smoke,
    }
