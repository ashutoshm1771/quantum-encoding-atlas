"""Encoding selection guide and recommendations.

Two complementary entry points:

:func:`recommend_encoding`
    Answers from problem *metadata* — feature count, task, hardware, priority
    — using the rule base plus the bundled benchmark evidence. Instant, and
    needs no data.

:func:`screen_encodings`
    Answers from your *data*. Scores every candidate encoding by kernel-target
    alignment on your own ``(X, y)`` — the training-free quantity the
    benchmark found tracks kernel accuracy (Spearman rho = 0.91) — and returns
    a ranked shortlist to train.

Use the recommendation to start, and the screen once you have data in hand.
"""

from encoding_atlas.guide.decision_tree import EncodingDecisionTree
from encoding_atlas.guide.recommender import Recommendation, recommend_encoding
from encoding_atlas.guide.rules import ENCODING_RULES, get_matching_encodings
from encoding_atlas.guide.screening import (
    ScreenedEncoding,
    ScreeningResult,
    screen_encodings,
)

__all__ = [
    "recommend_encoding",
    "Recommendation",
    "EncodingDecisionTree",
    "ENCODING_RULES",
    "get_matching_encodings",
    # Data-driven screening
    "screen_encodings",
    "ScreeningResult",
    "ScreenedEncoding",
]
