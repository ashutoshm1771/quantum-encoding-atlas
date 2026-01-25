"""Encoding recommendation system."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Recommendation:
    """Encoding recommendation result."""

    encoding_name: str
    explanation: str
    alternatives: list[str]
    confidence: float


def recommend_encoding(
    n_features: int,
    n_samples: int = 500,
    task: Literal["classification", "regression"] = "classification",
    hardware: str = "simulator",
    priority: Literal["accuracy", "trainability", "speed", "noise_resilience"] = "accuracy",
) -> Recommendation:
    """Recommend an encoding based on problem characteristics.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_samples : int
        Number of training samples.
    task : str
        Type of ML task.
    hardware : str
        Target hardware ('simulator', 'ibm', 'ionq', etc.).
    priority : str
        Optimization priority.

    Returns
    -------
    Recommendation
        Encoding recommendation with explanation.
    """
    # Simple rule-based recommendation
    if priority == "speed":
        return Recommendation(
            encoding_name="angle",
            explanation="Angle encoding is fastest with O(1) depth",
            alternatives=["basis"],
            confidence=0.8,
        )

    if priority == "noise_resilience" or hardware != "simulator":
        return Recommendation(
            encoding_name="hardware_efficient",
            explanation="Hardware-efficient encoding minimizes errors on real devices",
            alternatives=["angle", "data_reuploading"],
            confidence=0.7,
        )

    if n_features <= 4 and n_samples >= 200:
        return Recommendation(
            encoding_name="iqp",
            explanation="IQP encoding is expressive for small feature sets with sufficient data",
            alternatives=["zz_feature_map", "data_reuploading"],
            confidence=0.75,
        )

    if n_features > 8:
        return Recommendation(
            encoding_name="amplitude",
            explanation="Amplitude encoding provides exponential compression for many features",
            alternatives=["angle"],
            confidence=0.6,
        )

    return Recommendation(
        encoding_name="zz_feature_map",
        explanation="ZZ Feature Map is a good default for medium-sized problems",
        alternatives=["iqp", "data_reuploading"],
        confidence=0.7,
    )
