"""What it means for an advertised backend to be missing.

The library advertises PennyLane, Qiskit and Cirq. Tests for a backend that is
not installed *skip*, which is right for a contributor who installed only what
they need, and wrong for CI: a green run that quietly skipped 157 Cirq tests is
indistinguishable from a green run that verified them. That is not a
hypothetical — it is how ``SO2EquivariantFeatureMap`` shipped a bit-reversed
state on Qiskit.

So the policy is environment-dependent, and this module is the one place that
decides it:

``ENCODING_ATLAS_REQUIRE_ALL_BACKENDS``
    Unset, ``0``, ``false``, ``no``, ``off``
        A missing optional backend skips. The developer default.
    ``1``, ``true``, ``yes``, ``on``
        A missing optional backend is a *failure*. What CI sets.

Strict mode is enforced twice, deliberately, at two different granularities:

* :func:`tests.conftest.pytest_configure` refuses to start the session at all
  when a backend is missing, so the whole suite — including the many modules
  that guard with ``pytest.importorskip`` — is covered by one loud message.
* :func:`require_backend` turns an individual skip into a failure, so a module
  run outside that session (imported into another harness, or collected with a
  different rootdir) still honours the flag.

This lives in Python rather than in a CI shell step on purpose. The previous
version was a bash heredoc in ``ci.yml``; Windows runners default to PowerShell,
could not parse it, and the Windows half of the test matrix failed before it
ever ran a test. A policy expressed as a test runs identically on every
platform, and can be reproduced locally with::

    ENCODING_ATLAS_REQUIRE_ALL_BACKENDS=1 pytest
"""

from __future__ import annotations

import os
from collections.abc import Iterable

import pytest

__all__ = [
    "ALL_BACKENDS",
    "INSTALL_HINT",
    "OPTIONAL_BACKENDS",
    "REQUIRED_BACKENDS",
    "REQUIRE_ALL_BACKENDS_ENV",
    "backend_is_installed",
    "missing_backends",
    "require_all_backends",
    "require_backend",
]

#: Environment variable that promotes "backend missing" from skip to failure.
REQUIRE_ALL_BACKENDS_ENV = "ENCODING_ATLAS_REQUIRE_ALL_BACKENDS"

#: Backends installed by the base package; absence means a broken install.
REQUIRED_BACKENDS: tuple[str, ...] = ("pennylane",)

#: Backends behind an extra; absence is normal locally, never acceptable in CI.
OPTIONAL_BACKENDS: tuple[str, ...] = ("qiskit", "cirq")

#: Every backend the package advertises support for.
ALL_BACKENDS: tuple[str, ...] = (*REQUIRED_BACKENDS, *OPTIONAL_BACKENDS)

INSTALL_HINT = (
    "pip install 'encoding-atlas[all]'  (from a checkout: pip install '.[all]')"
)

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def require_all_backends() -> bool:
    """Whether a missing backend must fail the run instead of skipping it.

    Reads :data:`REQUIRE_ALL_BACKENDS_ENV` on every call rather than at import
    time, so a test can toggle the policy with ``monkeypatch.setenv``.
    """
    return os.environ.get(REQUIRE_ALL_BACKENDS_ENV, "").strip().lower() in _TRUTHY


def backend_is_installed(backend: str) -> bool:
    """Whether ``backend`` can actually be imported.

    Deliberately imports rather than inspecting :mod:`importlib.util` specs: a
    package whose spec resolves but whose import raises is, for the purposes of
    every test in this suite, missing.
    """
    try:
        __import__(backend)
    except ImportError:
        return False
    return True


def missing_backends(backends: Iterable[str] = ALL_BACKENDS) -> list[str]:
    """Those of ``backends`` that cannot be imported, in the given order."""
    return [backend for backend in backends if not backend_is_installed(backend)]


def require_backend(backend: str, *, reason: str = "") -> None:
    """Skip the calling test if ``backend`` is absent — or fail, in strict mode.

    Parameters
    ----------
    backend : str
        Import name of the backend, e.g. ``"qiskit"``.
    reason : str, optional
        What goes unverified without it. Appended to the skip/failure message,
        because "skipped: qiskit not installed" tells a reader nothing about
        which guarantee lapsed.

    Raises
    ------
    Skipped
        If ``backend`` is missing and strict mode is off.
    Failed
        If ``backend`` is missing and strict mode is on.
    """
    if backend_is_installed(backend):
        return

    detail = f" {reason.rstrip('.')}." if reason else ""
    head = f"Advertised backend {backend!r} is not installed.{detail}"

    if require_all_backends():
        pytest.fail(
            f"{head} {REQUIRE_ALL_BACKENDS_ENV} is set, so a missing backend is "
            f"a failure rather than a skip: this guarantee would otherwise have "
            f"gone unverified while the run stayed green. Install with: "
            f"{INSTALL_HINT}"
        )
    pytest.skip(
        f"{head} Set {REQUIRE_ALL_BACKENDS_ENV}=1 to make this a failure "
        f"instead of a skip. Install with: {INSTALL_HINT}"
    )
