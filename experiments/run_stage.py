#!/usr/bin/env python
"""CLI entry point for running experiment stages.

Usage
-----
::

    # Run a stage (resumes from checkpoint by default)
    python -m experiments.run_stage --config experiments/configs/stage1_resources.json

    # Force re-run, ignoring existing checkpoints
    python -m experiments.run_stage --config experiments/configs/stage1_resources.json --no-resume

    # Quick validation run with reduced samples
    python -m experiments.run_stage --config experiments/configs/stage3_expressibility.json --quick

    # Smoke test: quick mode + datasets capped at 20 samples (fastest e2e validation)
    python -m experiments.run_stage --config experiments/configs/stage6a_vqc.json --smoke

    # Combine flags
    python -m experiments.run_stage --config experiments/configs/stage3_expressibility.json --quick --no-resume

Exit Codes
----------
- 0: All tasks completed or skipped successfully.
- 1: One or more tasks failed (results still saved).
- 2: Fatal error (bad config, missing file, etc.).
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, configure logging, and run the experiment.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments. If ``None``, reads from ``sys.argv``.

    Returns
    -------
    int
        Exit code (0 = success, 1 = partial failure, 2 = fatal error).
    """
    parser = argparse.ArgumentParser(
        prog="run_stage",
        description="Run a Phase 4 experiment stage with checkpointing.",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON config file for this stage.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Ignore existing checkpoints and re-run all tasks.",
    )

    speed_group = parser.add_mutually_exclusive_group()
    speed_group.add_argument(
        "--quick",
        action="store_true",
        default=False,
        help="Use reduced sample counts for fast validation.",
    )
    speed_group.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help=(
            "Smoke test: quick-mode overrides plus dataset samples capped "
            "at 20. Exercises every code path in under a minute for Stages "
            "6a/6b. Results are not statistically meaningful."
        ),
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )

    args = parser.parse_args(argv)

    # Configure logging.
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        from experiments.runner import ExperimentRunner

        runner = ExperimentRunner(
            config_path=args.config,
            resume=not args.no_resume,
            quick=args.quick,
            smoke=args.smoke,
        )
        summary = runner.run()

    except (FileNotFoundError, ValueError) as exc:
        print(f"\nFatal error: {exc}", file=sys.stderr)
        return 2

    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress has been checkpointed.")
        return 1

    except Exception as exc:
        logging.exception("Unexpected error: %s", exc)
        return 2

    # Return non-zero if any tasks failed.
    if summary.get("failed", 0) > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
