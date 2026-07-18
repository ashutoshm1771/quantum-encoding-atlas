"""Benchmarking framework for encoding comparison.

Evaluate quantum encodings on classification tasks with variational quantum
classifiers and quantum-kernel SVMs, paired stratified cross-validation,
classical baselines, and statistical comparison.

>>> from encoding_atlas import AngleEncoding
>>> from encoding_atlas.benchmark import EncodingBenchmark
>>> bench = EncodingBenchmark(
...     [AngleEncoding(n_features=2)], ["moons"],
...     methods=("kernel",), n_runs=1, n_folds=3, seed=0,
... )
>>> results = bench.run()  # doctest: +SKIP
"""

from encoding_atlas.benchmark.baselines import (
    CLASSICAL_BASELINE_NAMES,
    REGRESSION_BASELINE_NAMES,
    get_classical_baseline,
    get_regression_baseline,
    run_baseline_single_fold,
    run_regression_baseline_single_fold,
)
from encoding_atlas.benchmark.datasets import (
    get_dataset,
    get_regression_dataset,
    list_datasets,
    list_multiclass_datasets,
    list_regression_datasets,
)
from encoding_atlas.benchmark.kernel import (
    QuantumKernelClassifier,
    QuantumKernelRegressor,
    centered_kernel_target_alignment,
    compute_kernel_matrix,
    ensure_psd,
    kernel_target_alignment,
    run_kernel_regression_fold,
    run_kernel_single_fold,
)
from encoding_atlas.benchmark.metrics import compute_metrics, compute_regression_metrics
from encoding_atlas.benchmark.runner import EncodingBenchmark, evaluate_encoding
from encoding_atlas.benchmark.statistical import (
    cliffs_delta,
    compare_encodings,
    compare_encodings_corrected,
    holm_bonferroni,
    wilcoxon_test,
)
from encoding_atlas.benchmark.vqc import (
    VQCClassifier,
    VQCRegressor,
    run_vqc_regression_fold,
    run_vqc_single_fold,
)

__all__ = [
    # Orchestration
    "EncodingBenchmark",
    "evaluate_encoding",
    # Datasets & metrics
    "get_dataset",
    "list_datasets",
    "list_multiclass_datasets",
    "get_regression_dataset",
    "list_regression_datasets",
    "compute_metrics",
    "compute_regression_metrics",
    # Classifiers
    "VQCClassifier",
    "QuantumKernelClassifier",
    "run_vqc_single_fold",
    "run_kernel_single_fold",
    # Regressors
    "VQCRegressor",
    "QuantumKernelRegressor",
    "run_vqc_regression_fold",
    "run_kernel_regression_fold",
    # Quantum kernel utilities
    "compute_kernel_matrix",
    "kernel_target_alignment",
    "centered_kernel_target_alignment",
    "ensure_psd",
    # Classical baselines
    "get_classical_baseline",
    "run_baseline_single_fold",
    "CLASSICAL_BASELINE_NAMES",
    "get_regression_baseline",
    "run_regression_baseline_single_fold",
    "REGRESSION_BASELINE_NAMES",
    # Statistics
    "wilcoxon_test",
    "compare_encodings",
    "compare_encodings_corrected",
    "cliffs_delta",
    "holm_bonferroni",
]
