"""Allow running as ``python -m experiments``."""

from __future__ import annotations

import sys

from experiments.run_stage import main

sys.exit(main())
