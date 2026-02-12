"""Encoding selection guide and recommendations."""

from encoding_atlas.guide.decision_tree import EncodingDecisionTree
from encoding_atlas.guide.recommender import Recommendation, recommend_encoding
from encoding_atlas.guide.rules import ENCODING_RULES, get_matching_encodings

__all__ = [
    "recommend_encoding",
    "Recommendation",
    "EncodingDecisionTree",
    "ENCODING_RULES",
    "get_matching_encodings",
]
