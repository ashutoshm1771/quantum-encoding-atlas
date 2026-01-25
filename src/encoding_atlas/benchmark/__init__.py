"""Benchmarking framework for encoding comparison."""

from encoding_atlas.benchmark.datasets import get_dataset, list_datasets
from encoding_atlas.benchmark.runner import EncodingBenchmark
from encoding_atlas.benchmark.metrics import compute_metrics

__all__ = [
    "get_dataset",
    "list_datasets",
    "EncodingBenchmark",
    "compute_metrics",
]
