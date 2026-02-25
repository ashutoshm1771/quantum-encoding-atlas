"""MkDocs hook to inject the package version into template context."""

from __future__ import annotations

import re
from pathlib import Path


def _strip_dev_suffix(version: str) -> str:
    """Strip setuptools-scm dev/local suffixes to show the last release version.

    Examples: '0.3.1.dev1+gbd09888de' -> '0.3.0', '0.3.0' -> '0.3.0'
    """
    match = re.match(r"(\d+\.\d+\.\d+)", version)
    if match:
        base = match.group(1)
        # If this is a dev version (e.g. 0.3.1.dev1), the base patch was
        # bumped by setuptools-scm; roll it back to the actual tagged release.
        if ".dev" in version:
            major, minor, patch = base.split(".")
            patch = str(max(int(patch) - 1, 0))
            return f"{major}.{minor}.{patch}"
        return base
    return version


def _read_version() -> str:
    """Read version from _version.py or the installed package."""
    raw = None

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
            raw = match.group(1)

    # Try 2: Import from installed package (non-editable pip install in CI)
    if raw is None:
        try:
            from encoding_atlas._version import __version__

            raw = __version__
        except ImportError:
            pass

    if raw is None:
        return "dev"

    return _strip_dev_suffix(raw)


def on_config(config, **kwargs):
    """Inject package_version into extra config for Jinja2 templates."""
    config["extra"]["package_version"] = _read_version()
    return config
