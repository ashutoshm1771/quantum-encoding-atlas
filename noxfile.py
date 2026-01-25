"""Nox automation for testing across Python versions."""

import nox

PYTHON_VERSIONS = ["3.9", "3.10", "3.11", "3.12"]


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run the test suite."""
    session.install(".[dev]")
    session.run(
        "pytest",
        "--cov=encoding_atlas",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "-v",
        *session.posargs,
    )


@nox.session(python="3.11")
def lint(session: nox.Session) -> None:
    """Run linters."""
    session.install(".[dev]")
    session.run("ruff", "check", "src", "tests")
    session.run("black", "--check", "src", "tests")
    session.run("mypy", "src")


@nox.session(python="3.11")
def format(session: nox.Session) -> None:
    """Format code with black and ruff."""
    session.install("black", "ruff")
    session.run("black", "src", "tests")
    session.run("ruff", "check", "--fix", "src", "tests")


@nox.session(python="3.11")
def docs(session: nox.Session) -> None:
    """Build documentation."""
    session.install(".[docs]")
    session.run("mkdocs", "build", "--strict")


@nox.session(python="3.11")
def docs_serve(session: nox.Session) -> None:
    """Serve documentation locally."""
    session.install(".[docs]")
    session.run("mkdocs", "serve")


@nox.session(python="3.11")
def coverage(session: nox.Session) -> None:
    """Generate coverage report."""
    session.install(".[dev]")
    session.run(
        "pytest",
        "--cov=encoding_atlas",
        "--cov-report=html",
        "--cov-report=term-missing",
    )
    session.log("Coverage report: htmlcov/index.html")


@nox.session(python="3.11")
def typecheck(session: nox.Session) -> None:
    """Run type checking with mypy."""
    session.install(".[dev]")
    session.run("mypy", "src", "--strict")


@nox.session(python="3.11")
def build(session: nox.Session) -> None:
    """Build distribution packages."""
    session.install("build", "twine")
    session.run("python", "-m", "build")
    session.run("twine", "check", "dist/*")
