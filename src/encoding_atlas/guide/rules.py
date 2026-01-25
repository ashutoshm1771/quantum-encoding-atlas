"""Rule-based encoding recommendations."""

from typing import Dict, List


ENCODING_RULES: Dict[str, Dict[str, any]] = {
    "angle": {
        "best_for": ["speed", "simplicity", "product_states"],
        "avoid_when": ["need_entanglement", "quantum_advantage"],
        "max_features": None,
        "simulable": True,
    },
    "amplitude": {
        "best_for": ["many_features", "compression"],
        "avoid_when": ["nisq_hardware", "shallow_circuits"],
        "max_features": None,
        "simulable": False,
    },
    "iqp": {
        "best_for": ["quantum_advantage", "expressibility"],
        "avoid_when": ["many_features", "noisy_hardware"],
        "max_features": 8,
        "simulable": False,
    },
    "zz_feature_map": {
        "best_for": ["balanced", "standard_benchmark"],
        "avoid_when": ["very_noisy_hardware"],
        "max_features": 10,
        "simulable": False,
    },
    "data_reuploading": {
        "best_for": ["universal_approximation", "trainability"],
        "avoid_when": ["limited_depth"],
        "max_features": 8,
        "simulable": False,
    },
    "hardware_efficient": {
        "best_for": ["nisq_hardware", "native_gates"],
        "avoid_when": ["simulator_only"],
        "max_features": None,
        "simulable": False,
    },
}


def get_matching_encodings(
    requirements: List[str],
    constraints: List[str] | None = None,
) -> List[str]:
    """Get encodings matching requirements and constraints."""
    matches = []

    for name, rules in ENCODING_RULES.items():
        # Check requirements
        if any(req in rules["best_for"] for req in requirements):
            # Check constraints
            if constraints:
                if not any(c in rules["avoid_when"] for c in constraints):
                    matches.append(name)
            else:
                matches.append(name)

    return matches
