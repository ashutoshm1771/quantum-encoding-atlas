#!/usr/bin/env python
"""Compatibility shim for older pip versions.

This file allows installation with older versions of pip that don't fully
support PEP 517/518. All configuration is in pyproject.toml.
"""

from setuptools import setup

if __name__ == "__main__":
    setup()
