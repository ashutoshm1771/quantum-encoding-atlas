"""The backend-availability policy is itself load-bearing, so it is tested.

This policy decides whether CI notices a backend that failed to install. Its
previous incarnation was a bash heredoc in ``ci.yml`` that no test covered and
no one could run locally; it was wrong (unparseable on Windows) for a full
commit before anyone found out. Expressing it in Python is only an improvement
if the Python is actually exercised — hence this module.

Both branches are checked: the skip that keeps a partial local install usable,
and the failure that makes CI honest.
"""

from __future__ import annotations

import pytest

import tests.conftest as root_conftest
from tests._backends import (
    ALL_BACKENDS,
    OPTIONAL_BACKENDS,
    REQUIRE_ALL_BACKENDS_ENV,
    REQUIRED_BACKENDS,
    backend_is_installed,
    missing_backends,
    require_all_backends,
    require_backend,
)

# A name that will never resolve, standing in for an uninstalled backend.
ABSENT = "encoding_atlas_no_such_backend_xyz"


class TestStrictModeFlag:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on", " 1 ", "On"])
    def test_truthy_values_enable_strict_mode(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(REQUIRE_ALL_BACKENDS_ENV, value)
        assert require_all_backends()

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "maybe", "2"])
    def test_other_values_leave_strict_mode_off(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv(REQUIRE_ALL_BACKENDS_ENV, value)
        assert not require_all_backends()

    def test_unset_is_permissive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The default must suit a contributor with a partial install."""
        monkeypatch.delenv(REQUIRE_ALL_BACKENDS_ENV, raising=False)
        assert not require_all_backends()

    def test_read_per_call_not_at_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Caching the flag at import time would make it untestable."""
        monkeypatch.setenv(REQUIRE_ALL_BACKENDS_ENV, "1")
        assert require_all_backends()
        monkeypatch.setenv(REQUIRE_ALL_BACKENDS_ENV, "0")
        assert not require_all_backends()


class TestDetection:
    def test_detects_an_installed_package(self) -> None:
        assert backend_is_installed("numpy")

    def test_detects_an_absent_package(self) -> None:
        assert not backend_is_installed(ABSENT)

    def test_pennylane_is_a_hard_dependency(self) -> None:
        assert backend_is_installed("pennylane")

    def test_missing_backends_preserves_order_and_filters(self) -> None:
        assert missing_backends(["numpy", ABSENT, "pennylane"]) == [ABSENT]

    def test_missing_backends_empty_when_all_present(self) -> None:
        assert missing_backends(["numpy", "pennylane"]) == []

    def test_backend_lists_are_disjoint_and_complete(self) -> None:
        assert set(REQUIRED_BACKENDS).isdisjoint(OPTIONAL_BACKENDS)
        assert list(ALL_BACKENDS) == [*REQUIRED_BACKENDS, *OPTIONAL_BACKENDS]


class TestRequireBackend:
    def test_present_backend_is_a_no_op(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Strict mode must not disturb a backend that *is* installed."""
        monkeypatch.setenv(REQUIRE_ALL_BACKENDS_ENV, "1")
        require_backend("numpy")

    def test_skips_when_permissive(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(REQUIRE_ALL_BACKENDS_ENV, "0")
        with pytest.raises(pytest.skip.Exception) as excinfo:
            require_backend(ABSENT)
        assert REQUIRE_ALL_BACKENDS_ENV in str(excinfo.value)

    def test_fails_when_strict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(REQUIRE_ALL_BACKENDS_ENV, "1")
        with pytest.raises(pytest.fail.Exception) as excinfo:
            require_backend(ABSENT)
        assert ABSENT in str(excinfo.value)

    def test_failure_is_not_a_skip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ``Failed`` that is also caught as ``Skipped`` would defeat this.

        Both derive from ``OutcomeException``; the distinction is the whole
        point of the flag, so it is pinned rather than assumed.
        """
        monkeypatch.setenv(REQUIRE_ALL_BACKENDS_ENV, "1")
        with pytest.raises(BaseException) as excinfo:
            require_backend(ABSENT)
        assert isinstance(excinfo.value, pytest.fail.Exception)
        assert not isinstance(excinfo.value, pytest.skip.Exception)

    @pytest.mark.parametrize("strict", ["0", "1"])
    def test_reason_reaches_the_message(
        self, monkeypatch: pytest.MonkeyPatch, strict: str
    ) -> None:
        """A bare "qiskit not installed" hides *which* guarantee lapsed."""
        monkeypatch.setenv(REQUIRE_ALL_BACKENDS_ENV, strict)
        with pytest.raises(BaseException) as excinfo:
            require_backend(ABSENT, reason="the widget round trip did not run")
        assert "the widget round trip did not run" in str(excinfo.value)

    @pytest.mark.parametrize("strict", ["0", "1"])
    def test_message_names_the_install_command(
        self, monkeypatch: pytest.MonkeyPatch, strict: str
    ) -> None:
        monkeypatch.setenv(REQUIRE_ALL_BACKENDS_ENV, strict)
        with pytest.raises(BaseException) as excinfo:
            require_backend(ABSENT)
        assert "pip install" in str(excinfo.value)


class TestSessionGate:
    """``pytest_configure`` is what actually stops a degraded CI run."""

    def test_permissive_session_starts_with_a_backend_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(REQUIRE_ALL_BACKENDS_ENV, raising=False)
        monkeypatch.setattr(root_conftest, "missing_backends", lambda _: ["cirq"])
        root_conftest.pytest_configure(None)  # type: ignore[arg-type]

    def test_strict_session_is_refused_with_a_backend_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(REQUIRE_ALL_BACKENDS_ENV, "1")
        monkeypatch.setattr(root_conftest, "missing_backends", lambda _: ["cirq"])
        with pytest.raises(pytest.UsageError) as excinfo:
            root_conftest.pytest_configure(None)  # type: ignore[arg-type]
        message = str(excinfo.value)
        assert "cirq" in message
        assert REQUIRE_ALL_BACKENDS_ENV in message

    def test_strict_session_starts_when_nothing_is_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(REQUIRE_ALL_BACKENDS_ENV, "1")
        monkeypatch.setattr(root_conftest, "missing_backends", lambda _: [])
        root_conftest.pytest_configure(None)  # type: ignore[arg-type]

    def test_this_session_passed_the_gate(self) -> None:
        """Sanity: the suite you are reading ran, so the gate let it.

        Under ``ENCODING_ATLAS_REQUIRE_ALL_BACKENDS=1`` that is a positive
        statement — every advertised backend is importable right now.
        """
        if require_all_backends():
            assert missing_backends(ALL_BACKENDS) == []
