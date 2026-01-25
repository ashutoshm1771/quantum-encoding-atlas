"""Decision tree for encoding selection."""

from typing import Dict, Any


class EncodingDecisionTree:
    """Decision tree for encoding selection."""

    def __init__(self) -> None:
        self.tree = self._build_tree()

    def _build_tree(self) -> Dict[str, Any]:
        """Build the decision tree."""
        return {
            "question": "What is your data type?",
            "options": {
                "continuous": {
                    "question": "How many features?",
                    "options": {
                        "few (<= 4)": "iqp",
                        "medium (5-8)": "zz_feature_map",
                        "many (> 8)": "amplitude",
                    },
                },
                "binary": "basis",
                "discrete": "basis",
            },
        }

    def decide(self, **kwargs: Any) -> str:
        """Make a decision based on inputs."""
        data_type = kwargs.get("data_type", "continuous")
        n_features = kwargs.get("n_features", 4)

        if data_type in ("binary", "discrete"):
            return "basis"

        if n_features <= 4:
            return "iqp"
        elif n_features <= 8:
            return "zz_feature_map"
        else:
            return "amplitude"
