"""Generate the bundled feature-scaling sensitivity dataset.

Measures how each of the 16 atlas encodings responds to the range its features
are scaled into, and — the reason this scan exists — how the benchmark's
headline expressibility-versus-accuracy correlation moves with that choice.

Why
---
The empirical pipeline min-max scales every dataset into ``[0, 2*pi]`` at load
time (``experiments/datasets.py``), and hypothesis H1 is evaluated under that
choice. Rotation gates have period ``2*pi``, so that range drives each feature
over a full period — the regime ``experiments/concentration_scan.py`` shows is
maximally Haar-like, where fidelity kernels collapse onto the concentration
floor. Circuits that scramble fastest are pushed hardest, which is exactly the
set H1 is about. The range is therefore a variable the study should report, not
a constant it can assume, and this scan measures it.

The scan records, per encoding and range: mean kernel-target alignment, mean
kernel accuracy, and mean kernel concentration across the benchmark's datasets;
plus, per range, the Spearman correlation between measured expressibility and
measured accuracy across encodings.

Protocol
--------
Encoding parameters match ``experiments/configs/stage6b_kernel.json`` via
``encoding_atlas.guide._candidates.BENCHMARK_PARAMS``. Datasets are the
library's own binary classification sets, loaded unscaled and rescaled per
range. Expressibility is re-measured at each range so both axes of the
correlation describe the same regime. Everything is seeded.

Usage
-----
::

    python -m experiments.scaling_scan            # write package data
    python -m experiments.scaling_scan --check    # verify, do not write
    python -m experiments.scaling_scan -o out.json

Regenerate and commit whenever an encoding's circuit definition changes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from encoding_atlas.analysis.concentration import summarize_kernel_concentration
from encoding_atlas.analysis.expressibility import compute_expressibility
from encoding_atlas.analysis.generalization import (
    centered_kernel_target_alignment,
    compute_fidelity_kernel,
)
from encoding_atlas.analysis.scaling import scale_to_range
from encoding_atlas.benchmark.datasets import get_dataset
from encoding_atlas.benchmark.kernel import ensure_psd
from encoding_atlas.core.registry import get_encoding
from encoding_atlas.guide._candidates import BENCHMARK_PARAMS

SCHEMA_VERSION = "1.0"

DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "encoding_atlas"
    / "atlas"
    / "data"
    / "scaling_sensitivity.json"
)

# Measurement protocol, kept in one place so the emitted provenance block and
# the actual measurement cannot drift apart.
RANGES: tuple[tuple[float, float], ...] = (
    (0.0, math.pi / 4),
    (0.0, math.pi / 2),
    (0.0, math.pi),
    (0.0, 2.0 * math.pi),
)
DATASETS: tuple[str, ...] = ("moons", "circles", "linear", "xor", "iris")
N_SAMPLES: int = 120
N_FOLDS: int = 5
KERNEL_C: float = 1.0
EXPRESSIBILITY_SAMPLES: int = 1000
SEED: int = 42
BACKEND = "pennylane"

# The range the empirical pipeline uses, recorded so consumers can identify it.
PUBLISHED_RANGE: tuple[float, float] = (0.0, 2.0 * math.pi)


def _fold_accuracy(K: np.ndarray, y: np.ndarray) -> float:
    """Mean cross-validated accuracy of a precomputed-kernel SVM."""
    folds = []
    for train, test in StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(
        K, y
    ):
        K_train, _ = ensure_psd(K[np.ix_(train, train)])
        svm = SVC(kernel="precomputed", C=KERNEL_C, random_state=SEED)
        svm.fit(K_train, y[train])
        folds.append(float(np.mean(svm.predict(K[np.ix_(test, train)]) == y[test])))
    return float(np.mean(folds))


def _load_datasets() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load the benchmark datasets unscaled, keyed by name."""
    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in DATASETS:
        X, y = get_dataset(name, n_samples=N_SAMPLES, seed=SEED)
        loaded[name] = (np.asarray(X, dtype=np.float64), np.asarray(y))
    return loaded


def _json_number(value: float) -> float | None:
    """Convert a float to a JSON-safe value (``nan``/``inf`` become ``None``)."""
    return float(value) if math.isfinite(value) else None


def _atlas_expressibility_names() -> set[str]:
    """Encodings the bundled atlas reports an expressibility for.

    The published H1 analysis is restricted to these; ``basis`` and
    ``amplitude`` are state-preparation circuits whose expressibility the
    pipeline records as null.
    """
    from encoding_atlas.atlas import get_encoding_profile

    names: set[str] = set()
    for name in BENCHMARK_PARAMS:
        try:
            profile = get_encoding_profile(name)
        except KeyError:  # pragma: no cover - rule base and atlas agree today
            continue
        if profile.metrics.get("expressibility") is not None:
            names.add(name)
    return names


def build_dataset(*, verbose: bool = True) -> dict[str, Any]:
    """Run the full scan and return the dataset as a plain dictionary."""
    data = _load_datasets()
    n_features = {name: X.shape[1] for name, (X, _) in data.items()}

    encodings: dict[str, dict[str, Any]] = {}
    # accuracy[range_index][encoding] and expressibility[range_index][encoding]
    accuracy_by_range: list[dict[str, float]] = [{} for _ in RANGES]
    expressibility_by_range: list[dict[str, float]] = [{} for _ in RANGES]

    for name, params in BENCHMARK_PARAMS.items():
        points: list[dict[str, Any]] = []
        display_name = ""
        for index, (low, high) in enumerate(RANGES):
            alignments, accuracies, ratios = [], [], []
            for ds_name, (X_raw, y) in data.items():
                try:
                    encoding = get_encoding(
                        name, n_features=n_features[ds_name], **params
                    )
                except Exception:
                    continue
                display_name = display_name or type(encoding).__name__
                X = scale_to_range(X_raw, low, high)
                K = compute_fidelity_kernel(encoding, X, backend=BACKEND)
                alignments.append(float(centered_kernel_target_alignment(K, y)))
                accuracies.append(_fold_accuracy(K, y))
                ratios.append(
                    summarize_kernel_concentration(
                        K, int(encoding.n_qubits)
                    ).concentration_ratio
                )
            if not alignments:
                continue

            # Expressibility re-measured in the same regime, so both axes of
            # the correlation below describe the same input distribution.
            expressibility: float | None = None
            try:
                probe = get_encoding(name, n_features=2, **params)
                expressibility = float(
                    compute_expressibility(
                        probe,
                        n_samples=EXPRESSIBILITY_SAMPLES,
                        input_range=(low, high),
                        seed=SEED,
                    )
                )
            except Exception:
                expressibility = None

            mean_accuracy = float(np.mean(accuracies))
            accuracy_by_range[index][name] = mean_accuracy
            if expressibility is not None:
                expressibility_by_range[index][name] = expressibility

            points.append(
                {
                    "low": low,
                    "high": high,
                    "mean_alignment": _json_number(float(np.mean(alignments))),
                    "mean_accuracy": _json_number(mean_accuracy),
                    "mean_concentration_ratio": _json_number(float(np.mean(ratios))),
                    "expressibility": (
                        None if expressibility is None else _json_number(expressibility)
                    ),
                    "n_datasets": len(alignments),
                }
            )

        if not points:
            continue
        best = max(points, key=lambda p: (p["mean_alignment"] or -2.0, -p["high"]))
        accuracies_only = [p["mean_accuracy"] for p in points if p["mean_accuracy"]]
        encodings[name] = {
            "encoding": name,
            "display_name": display_name,
            "params": dict(params),
            "points": points,
            "best_range": [best["low"], best["high"]],
            "accuracy_spread": _json_number(
                max(accuracies_only) - min(accuracies_only) if accuracies_only else 0.0
            ),
        }
        if verbose:
            accs = ", ".join(f"{p['mean_accuracy']:.3f}" for p in points)
            print(
                f"{name:24s} acc=[{accs}] best={tuple(encodings[name]['best_range'])}",
                flush=True,
            )

    # The scientific payload: how H1 moves with the scaling range.
    #
    # Reported twice. The full set covers every encoding measurable here. The
    # "atlas subset" restricts to the encodings the bundled atlas reports an
    # expressibility for, which is the set the published H1 analysis used --
    # basis and amplitude are state-preparation circuits whose expressibility
    # the pipeline records as null. Only the subset is comparable with the
    # published number.
    atlas_expressibility = _atlas_expressibility_names()
    correlations: list[dict[str, Any]] = []
    for index, (low, high) in enumerate(RANGES):
        shared = sorted(
            set(accuracy_by_range[index]) & set(expressibility_by_range[index])
        )
        if len(shared) < 3:
            continue

        def _correlate(names: list[str]) -> tuple[float | None, float | None, int]:
            if len(names) < 3:
                return None, None, len(names)
            rho, p_value = spearmanr(
                [expressibility_by_range[index][n] for n in names],
                [accuracy_by_range[index][n] for n in names],
            )
            return _json_number(float(rho)), _json_number(float(p_value)), len(names)

        rho_all, p_all, n_all = _correlate(shared)
        subset = [n for n in shared if n in atlas_expressibility]
        rho_sub, p_sub, n_sub = _correlate(subset)

        correlations.append(
            {
                "low": low,
                "high": high,
                "spearman_rho": rho_all,
                "p_value": p_all,
                "n_encodings": n_all,
                "spearman_rho_atlas_subset": rho_sub,
                "p_value_atlas_subset": p_sub,
                "n_encodings_atlas_subset": n_sub,
                "is_published_range": [low, high] == list(PUBLISHED_RANGE),
            }
        )
        if verbose:
            print(
                f"  expressibility vs accuracy @ [{low:.3f}, {high:.3f}]: "
                f"all rho={rho_all:+.3f} p={p_all:.3g} (n={n_all}) | "
                f"atlas subset rho={rho_sub:+.3f} p={p_sub:.3g} (n={n_sub})",
                flush=True,
            )

    return {
        "schema_version": SCHEMA_VERSION,
        "n_encodings": len(encodings),
        "protocol": {
            "ranges": [list(r) for r in RANGES],
            "datasets": list(DATASETS),
            "n_samples": N_SAMPLES,
            "n_folds": N_FOLDS,
            "kernel_C": KERNEL_C,
            "expressibility_samples": EXPRESSIBILITY_SAMPLES,
            "seed": SEED,
            "backend": BACKEND,
            "published_range": list(PUBLISHED_RANGE),
        },
        "generated_by": "experiments/scaling_scan.py",
        "encodings": encodings,
        "expressibility_accuracy_correlation": correlations,
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
