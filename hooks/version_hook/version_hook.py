"""MkDocs hook to inject the package version into template context."""

from __future__ import annotations

import re
from pathlib import Path


def _read_version() -> str:
    """Read version from _version.py or the installed package."""
    # Try 1: Read from source tree (editable install / local dev)
    version_file = (
        Path(__file__).resolve().parents[2] / "src" / "encoding_atlas" / "_version.py"
    )
    if version_file.exists():
        match = re.search(
            r'__version__\s*=\s*version\s*=\s*["\']([^"\']+)["\']',
            version_file.read_text(),
        )
        if match:
            return match.group(1)

    # Try 2: Import from installed package (non-editable pip install in CI)
    try:
        from encoding_atlas._version import __version__

        return __version__
    except ImportError:
        pass

    return "dev"


def on_config(config, **kwargs):
    """Inject package_version into extra config for Jinja2 templates."""
    config["extra"]["package_version"] = _read_version()
    return config
