"""Visualization tools for benchmark results."""

from typing import Any


def plot_accuracy_comparison(results: dict[str, list[float]]) -> Any:
    """Plot accuracy comparison bar chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for visualization")

    names = list(results.keys())
    means = [sum(v) / len(v) for v in results.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(names, means)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Encoding")
    ax.set_title("Encoding Accuracy Comparison")
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig
