"""Generate the bundled fidelity-kernel concentration dataset.

Measures how each of the 16 atlas encodings' fidelity kernel concentrates as
the circuit widens, and writes the result to the package data file that
:mod:`encoding_atlas.atlas` serves.

The scan answers the question the eight-stage pipeline does not: the accuracy
stages ran at ``n_features`` in ``{2, 4}`` (see
``experiments/configs/stage6b_kernel.json``), so the published ranking is
measured entirely in the regime *before* kernel concentration switches on. This
scan sweeps 2-8 qubits and records where each encoding's kernel reaches the
Haar floor.

Protocol
--------
Encoding parameters match ``stage6b_kernel.json`` so the concentration numbers
describe the same circuits the accuracy numbers were measured on. Inputs are
drawn uniformly from ``[0, 2*pi)`` — the analysis package's data-free
convention — with a fixed seed, so the scan is exactly reproducible.

Usage
-----
::

    python -m experiments.concentration_scan            # write package data
    python -m experiments.concentration_scan --check    # verify, do not write
    python -m experiments.concentration_scan -o out.json

Regenerate and commit the output whenever an encoding's circuit definition
changes; ``--check`` is the CI-friendly form that fails if the committed data
no longer matches what the current code produces.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

from encoding_atlas.analysis.concentration import (
    CONCENTRATION_THRESHOLD,
    ScalingResult,
    estimate_concentration_scaling,
)
from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.core.registry import get_encoding

SCHEMA_VERSION = "1.0"

# Default destination: the package data file the atlas API reads.
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "encoding_atlas"
    / "atlas"
    / "data"
    / "concentration.json"
)

# Measurement protocol. Kept in one place so the emitted provenance block and
# the actual measurement can never drift apart.
FEATURE_COUNTS: tuple[int, ...] = (2, 4, 6, 8)
# 200 inputs = 19,900 off-diagonal pairs per point. Validated against the
# closed form for angle encoding (Var = (3/8)^n - (1/4)^n under U[0, 2*pi)
# inputs): agreement is within ~2% at this sample count, versus ~30% error at
# 60. The dataset ships permanently, so it is measured at the accurate end.
N_SAMPLES: int = 200
INPUT_RANGE: tuple[float, float] = (0.0, 2.0 * math.pi)
SAMPLING = "uniform"
SEED: int = 42
BACKEND = "pennylane"

# Fixed (non-width) parameters per encoding, mirroring stage6b_kernel.json so
# the concentration scan describes the same circuits the benchmark measured.
#
# ``trainable_encoding`` additionally pins ``seed``: its variational parameters
# are drawn at construction time, so without a seed the scan would not be
# reproducible. Holding them fixed also matches how the benchmark treats it —
# as a feature map, not a jointly-optimised model.
ENCODING_PARAMS: dict[str, dict[str, Any]] = {
    "angle": {"rotation": "Y"},
    "amplitude": {},
    "basis": {},
    "iqp": {"reps": 2},
    "zz_feature_map": {"reps": 2},
    "pauli_feature_map": {"reps": 2},
    "data_reuploading": {"n_layers": 2},
    "hardware_efficient": {"reps": 2},
    "higher_order_angle": {"order": 2},
    "qaoa_encoding": {"reps": 2},
    "hamiltonian_encoding": {"reps": 2},
    "symmetry_inspired": {"reps": 2},
    "trainable_encoding": {"n_layers": 2, "seed": SEED},
    "so2_equivariant": {"max_angular_momentum": 2},
    "cyclic_equivariant": {"reps": 2},
    "swap_equivariant": {"reps": 2},
}


def _factory(name: str, params: dict[str, Any]) -> Callable[[int], BaseEncoding]:
    """Return an ``n_features -> BaseEncoding`` factory for one encoding."""

    def build(n_features: int) -> BaseEncoding:
        return get_encoding(name, n_features=n_features, **params)

    return build


def _json_number(value: float) -> float | None:
    """Convert a float to a JSON-safe value (``nan``/``inf`` become ``None``)."""
    return float(value) if math.isfinite(value) else None


def _serialize(name: str, params: dict[str, Any], scaling: ScalingResult) -> dict:
    """Reduce a :class:`ScalingResult` to its JSON record."""
    return {
        "encoding": name,
        "display_name": scaling.encoding_name,
        "params": dict(params),
        "points": [
            {
                "n_features": n_features,
                "n_qubits": result.n_qubits,
                "concentration_ratio": _json_number(result.concentration_ratio),
                "mean_ratio": _json_number(result.mean_ratio),
                "offdiagonal_mean": _json_number(result.offdiagonal_mean),
                "offdiagonal_variance": _json_number(result.offdiagonal_variance),
                "shots_per_entry": _json_number(result.shots_per_entry),
                "is_concentrated": bool(result.is_concentrated),
            }
            for n_features, result in zip(scaling.feature_counts, scaling.results)
        ],
        "decay_rate": _json_number(scaling.decay_rate),
        "mean_decay_rate": _json_number(scaling.mean_decay_rate),
        "haar_normalized_slope": _json_number(scaling.haar_normalized_slope),
        "r_squared": _json_number(scaling.r_squared),
        "concentration_horizon": scaling.concentration_horizon(),
        "skipped": {str(k): v for k, v in sorted(scaling.skipped.items())},
    }


def build_dataset(*, verbose: bool = True) -> dict:
    """Run the full scan and return the dataset as a plain dictionary."""
    encodings: dict[str, dict] = {}
    for name, params in ENCODING_PARAMS.items():
        scaling = estimate_concentration_scaling(
            _factory(name, params),
            feature_counts=FEATURE_COUNTS,
            n_samples=N_SAMPLES,
            input_range=INPUT_RANGE,
            sampling=SAMPLING,
            threshold=CONCENTRATION_THRESHOLD,
            seed=SEED,
            backend=BACKEND,
        )
        encodings[name] = _serialize(name, params, scaling)
        if verbose:
            ratios = ", ".join(f"{r:.2f}" for r in scaling.concentration_ratios)
            print(
                f"{name:24s} qubits={scaling.n_qubits} ratios=[{ratios}] "
                f"decay={scaling.decay_rate:.2f} "
                f"horizon={scaling.concentration_horizon()}",
                flush=True,
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "n_encodings": len(encodings),
        "protocol": {
            "feature_counts": list(FEATURE_COUNTS),
            "n_samples": N_SAMPLES,
            "input_range": list(INPUT_RANGE),
            "sampling": SAMPLING,
            "threshold": CONCENTRATION_THRESHOLD,
            "seed": SEED,
            "backend": BACKEND,
        },
        "generated_by": "experiments/concentration_scan.py",
        "encodings": encodings,
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point. Returns a process exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="destination JSON file (default: the bundled package data file)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the existing file matches a fresh scan; do not write",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="suppress progress")
    args = parser.parse_args(argv)

    dataset = build_dataset(verbose=not args.quiet)
    payload = json.dumps(dataset, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not args.output.exists():
            print(f"MISSING: {args.output}", file=sys.stderr)
            return 1
        if args.output.read_text(encoding="utf-8") != payload:
            print(f"STALE: {args.output} differs from a fresh scan", file=sys.stderr)
            return 1
        print(f"OK: {args.output} is up to date")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    if not args.quiet:
        print(f"\nWrote {args.output} ({len(dataset['encodings'])} encodings)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
