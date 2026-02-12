"""Utility functions for encoding_atlas."""

from encoding_atlas.utils.preprocessing import normalize_features, scale_features
from encoding_atlas.utils.validation import validate_backend, validate_features

__all__ = [
    "validate_features",
    "validate_backend",
    "scale_features",
    "normalize_features",
]
