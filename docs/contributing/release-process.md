# Release Process

This page documents how new versions of the Quantum Encoding Atlas are released. The process is automated via GitHub Actions — a human only needs to create a git tag.

---

## Versioning

The project follows [Semantic Versioning](https://semver.org/):

```
MAJOR.MINOR.PATCH

MAJOR  — Breaking API changes (renamed classes, removed methods)
MINOR  — New features, new encodings, backward-compatible changes
PATCH  — Bug fixes, documentation updates, no API changes
```

Versions are derived automatically from git tags by `setuptools-scm`. You never edit a version number manually.

---

## How to Release

### 1. Ensure master is clean

```bash
git checkout master
git pull origin master
pytest                       # All tests pass
ruff check src tests         # No lint errors
black --check src tests      # Formatting clean
mypy src                     # Types check
mkdocs build --strict        # Docs build
```

### 2. Update the changelog

Edit `CHANGELOG.md` to move items from "Unreleased" to the new version section:

```markdown
## [0.2.0] - 2026-02-15

### Added
- Cyclic Equivariant Feature Map
- Swap Equivariant Feature Map
- ...

### Changed
- ...

### Fixed
- ...
```

Commit the changelog update:

```bash
git add CHANGELOG.md
git commit -m "Release v0.2.0"
git push origin master
```

### 3. Create and push the tag

```bash
git tag v0.2.0
git push origin v0.2.0
```

### 4. GitHub Actions takes over

The tag push triggers the `publish-pypi.yml` workflow, which automatically:

1. Builds the source distribution and wheel
2. Verifies the version matches the tag
3. Publishes to PyPI via OIDC trusted publishing
4. Creates a GitHub Release with auto-generated release notes

### 5. Verify

- Check [PyPI](https://pypi.org/project/encoding-atlas/) for the new version
- Check the [GitHub Releases](https://github.com/ashutoshm1771/quantum-encoding-atlas/releases) page
- Test installation: `pip install encoding-atlas==0.2.0`

---

## Pre-Release Versions

For release candidates:

```bash
git tag v0.3.0-rc.1
git push origin v0.3.0-rc.1
```

Pre-releases are automatically marked as such on GitHub and PyPI. They are not installed by default (`pip install encoding-atlas` installs only stable versions).

---

## Hotfix Process

For urgent fixes to a released version:

```bash
# Create a hotfix branch from the release tag
git checkout -b hotfix/fix-description v0.2.0

# Make the fix, commit, then tag
git tag v0.2.1
git push origin hotfix/fix-description v0.2.1

# Merge back to master
git checkout master
git merge hotfix/fix-description
git push origin master
```
