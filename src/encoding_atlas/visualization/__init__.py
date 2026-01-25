"""Visualization tools for quantum encodings.

This module provides functions to visualize and compare quantum data encodings,
helping users understand encoding properties, trade-offs, and circuit structures.

Quick Start
-----------
>>> from encoding_atlas.visualization import compare_encodings
>>> compare_encodings(['angle', 'iqp', 'amplitude'], n_features=4)

>>> # For matplotlib output
>>> fig = compare_encodings(['angle', 'iqp'], n_features=8, output='matplotlib')
>>> fig.savefig('comparison.png')

Available Functions
-------------------
compare_encodings
    Compare multiple encodings side-by-side with ASCII or matplotlib output.

Examples
--------
Compare encodings with text output (ASCII art):

>>> from encoding_atlas.visualization import compare_encodings
>>> compare_encodings(
...     ['angle', 'iqp', 'zz_feature_map'],
...     n_features=4
... )

Generate a publication-ready comparison figure:

>>> fig = compare_encodings(
...     ['angle', 'amplitude', 'iqp', 'hardware_efficient'],
...     n_features=8,
...     output='matplotlib',
...     figsize=(14, 10)
... )
>>> fig.savefig('encoding_comparison.pdf', dpi=300, bbox_inches='tight')

Compare with custom configurations:

>>> compare_encodings(
...     ['angle', 'iqp'],
...     n_features=4,
...     configs={
...         'angle': {'rotation': 'X', 'reps': 3},
...         'iqp': {'entanglement': 'full', 'reps': 2},
...     }
... )
"""

from encoding_atlas.visualization.comparison import (
    EncodingComparison,
    compare_encodings,
)

__all__ = [
    "compare_encodings",
    "EncodingComparison",
]
