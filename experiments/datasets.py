"""Dataset loading and preprocessing for Stage 6 experiments.

This module provides centralized dataset loading with consistent preprocessing
for the VQC classification benchmark (Stage 6a) and quantum kernel benchmark
(Stage 6b).

All datasets are preprocessed identically:
1. Binary classification (converted from multi-class where needed)
2. Feature scaling to [0, 2π] via min-max normalization
3. Stratified k-fold cross-validation with deterministic seed

Dataset Specifications
----------------------
Toy datasets (n_features=2):
- moons: sklearn.datasets.make_moons with noise=0.1
- circles: sklearn.datasets.make_circles with noise=0.1, factor=0.5
- linear: sklearn.datasets.make_classification (linearly separable)
- xor: Synthetic XOR pattern with 4 clusters at corners

Real-world datasets (n_features=4 after PCA):
- iris: Setosa vs rest (first 4 features, no PCA needed)
- wine: Class 0 vs rest (PCA to 4 features)
- breast_cancer: Binary classification (PCA to 4 features)
- digits_01: Digits 0 vs 1 only (PCA to 4 features)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# All available datasets for Stage 6 experiments
AVAILABLE_DATASETS: List[str] = [
    # Toy datasets (n_features=2)
    "moons",
    "circles",
    "linear",
    "xor",
    # Real-world datasets (n_features=4, after PCA where needed)
    "iris",
    "wine",
    "breast_cancer",
    "digits_01",
]

# Dataset feature dimensions
DATASET_N_FEATURES: dict[str, int] = {
    "moons": 2,
    "circles": 2,
    "linear": 2,
    "xor": 2,
    "iris": 4,
    "wine": 4,
    "breast_cancer": 4,
    "digits_01": 4,
}

# Synthetic datasets accept an ``n_samples`` parameter at generation time.
# Real-world datasets have a fixed size and must be subsampled after loading.
_SYNTHETIC_DATASETS = frozenset({"moons", "circles", "linear", "xor"})


def load_dataset(
    name: str,
    n_samples: int = 200,
    seed: int = 42,
    max_samples: Optional[int] = None,
) -> Tuple[NDArray[np.floating], NDArray[np.intp]]:
    """Load and preprocess a dataset for VQC/kernel experiments.

    All datasets are preprocessed consistently:
    1. Converted to binary classification
    2. Features scaled to [0, 2π] via min-max normalization
    3. Deterministic based on seed

    Parameters
    ----------
    name : str
        Dataset name. One of: 'moons', 'circles', 'linear', 'xor',
        'iris', 'wine', 'breast_cancer', 'digits_01'.
    n_samples : int, default=200
        Number of samples (for synthetic datasets). Ignored for
        real-world datasets which use their natural size.
    seed : int, default=42
        Random seed for reproducibility.
    max_samples : int or None, default=None
        If set, cap the dataset to at most this many samples.
        Synthetic datasets are generated with fewer samples directly;
        real-world datasets are stratified-subsampled after loading.
        Used by ``--smoke`` mode for fast end-to-end validation.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Features scaled to [0, 2π].
    y : np.ndarray, shape (n_samples,)
        Binary labels {0, 1}.

    Raises
    ------
    ValueError
        If dataset name is not recognized.
    """
    if name not in AVAILABLE_DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {AVAILABLE_DATASETS}"
        )

    logger.debug("Loading dataset '%s' with seed=%d", name, seed)

    # For synthetic datasets, cap generation size with max_samples.
    effective_n_samples = n_samples
    if max_samples is not None and name in _SYNTHETIC_DATASETS:
        effective_n_samples = min(n_samples, max_samples)

    # Load raw data based on dataset type
    if name == "moons":
        X, y = _load_moons(effective_n_samples, seed)
    elif name == "circles":
        X, y = _load_circles(effective_n_samples, seed)
    elif name == "linear":
        X, y = _load_linear(effective_n_samples, seed)
    elif name == "xor":
        X, y = _load_xor(effective_n_samples, seed)
    elif name == "iris":
        X, y = _load_iris()
    elif name == "wine":
        X, y = _load_wine()
    elif name == "breast_cancer":
        X, y = _load_breast_cancer()
    elif name == "digits_01":
        X, y = _load_digits_01()
    else:
        raise ValueError(f"Dataset '{name}' not implemented")

    # For real-world datasets, stratified-subsample if max_samples is set.
    if max_samples is not None and len(X) > max_samples:
        X, y = _stratified_subsample(X, y, max_samples, seed)

    # Scale features to [0, 2π]
    X_scaled = _scale_to_range(X, low=0.0, high=2 * np.pi)

    logger.debug(
        "Loaded dataset '%s': shape=%s, classes=%s",
        name, X_scaled.shape, np.unique(y).tolist()
    )

    return X_scaled, y


def get_cv_folds(
    X: NDArray[np.floating],
    y: NDArray[np.intp],
    n_folds: int = 5,
    seed: int = 42,
) -> List[Tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.intp], NDArray[np.intp]]]:
    """Generate stratified k-fold cross-validation splits.

    Uses scikit-learn's StratifiedKFold to ensure class proportions are
    maintained in each fold. The folds are deterministic for a given seed,
    ensuring the same splits are used across all encodings for fair
    paired statistical comparison.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Labels.
    n_folds : int, default=5
        Number of cross-validation folds.
    seed : int, default=42
        Random seed for fold generation. Same seed produces same folds.

    Returns
    -------
    list of tuples
        Each tuple contains (X_train, X_test, y_train, y_test) for one fold.
        The list has length n_folds.

    Notes
    -----
    The same seed must be used across all encodings to ensure paired
    statistical comparisons are valid (same test samples for each encoding).
    """
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        folds.append((X_train, X_test, y_train, y_test))

    logger.debug(
        "Generated %d stratified folds with seed=%d",
        n_folds, seed
    )

    return folds


def get_dataset_n_features(name: str) -> int:
    """Get the number of features for a dataset.

    Parameters
    ----------
    name : str
        Dataset name.

    Returns
    -------
    int
        Number of features (2 for toy datasets, 4 for real-world).

    Raises
    ------
    ValueError
        If dataset name is not recognized.
    """
    if name not in DATASET_N_FEATURES:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {list(DATASET_N_FEATURES.keys())}"
        )
    return DATASET_N_FEATURES[name]


# =============================================================================
# Internal dataset loaders
# =============================================================================


def _load_moons(
    n_samples: int,
    seed: int,
) -> Tuple[NDArray[np.floating], NDArray[np.intp]]:
    """Load two interleaving half circles (moons) dataset."""
    from sklearn.datasets import make_moons

    X, y = make_moons(n_samples=n_samples, noise=0.1, random_state=seed)
    return X, y.astype(np.intp)


def _load_circles(
    n_samples: int,
    seed: int,
) -> Tuple[NDArray[np.floating], NDArray[np.intp]]:
    """Load concentric circles dataset with rotational symmetry."""
    from sklearn.datasets import make_circles

    X, y = make_circles(
        n_samples=n_samples,
        noise=0.1,
        factor=0.5,
        random_state=seed,
    )
    return X, y.astype(np.intp)


def _load_linear(
    n_samples: int,
    seed: int,
) -> Tuple[NDArray[np.floating], NDArray[np.intp]]:
    """Load linearly separable 2D dataset."""
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        flip_y=0.0,
        class_sep=1.5,
        random_state=seed,
    )
    return X, y.astype(np.intp)


def _load_xor(
    n_samples: int,
    seed: int,
) -> Tuple[NDArray[np.floating], NDArray[np.intp]]:
    """Load synthetic XOR dataset with swap (Z_2) symmetry.

    Creates 4 clusters at the corners of a square, with labels
    determined by XOR of quadrant coordinates:
    - (0,0) and (1,1) -> class 0
    - (0,1) and (1,0) -> class 1

    This dataset has swap symmetry relevant to hypothesis H2.
    """
    rng = np.random.default_rng(seed)

    # Samples per cluster
    n_per_cluster = n_samples // 4
    remainder = n_samples % 4

    # Cluster centers at corners
    centers = np.array([
        [0.0, 0.0],   # class 0 (XOR = 0)
        [1.0, 1.0],   # class 0 (XOR = 0)
        [0.0, 1.0],   # class 1 (XOR = 1)
        [1.0, 0.0],   # class 1 (XOR = 1)
    ])
    labels = np.array([0, 0, 1, 1])

    X_list = []
    y_list = []

    for i, (center, label) in enumerate(zip(centers, labels)):
        # Add extra sample to first clusters if needed
        n = n_per_cluster + (1 if i < remainder else 0)
        cluster_samples = rng.normal(loc=center, scale=0.15, size=(n, 2))
        X_list.append(cluster_samples)
        y_list.append(np.full(n, label))

    X = np.vstack(X_list)
    y = np.concatenate(y_list).astype(np.intp)

    # Shuffle the data
    shuffle_idx = rng.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]

    return X, y


def _load_iris() -> Tuple[NDArray[np.floating], NDArray[np.intp]]:
    """Load Iris dataset as binary classification: setosa vs rest.

    Uses first 4 features (no PCA needed as Iris has 4 features).
    """
    from sklearn.datasets import load_iris

    data = load_iris()
    X = data.data[:, :4]  # First 4 features
    y = (data.target != 0).astype(np.intp)  # Setosa=0 vs rest=1

    return X, y


def _load_wine() -> Tuple[NDArray[np.floating], NDArray[np.intp]]:
    """Load Wine dataset as binary classification: class 0 vs rest.

    Applies PCA to reduce from 13 features to 4.
    """
    from sklearn.datasets import load_wine
    from sklearn.decomposition import PCA

    data = load_wine()
    X_full = data.data
    y = (data.target != 0).astype(np.intp)  # Class 0 vs rest

    # PCA to 4 features
    pca = PCA(n_components=4, random_state=42)
    X = pca.fit_transform(X_full)

    return X, y


def _load_breast_cancer() -> Tuple[NDArray[np.floating], NDArray[np.intp]]:
    """Load Breast Cancer Wisconsin dataset with PCA to 4 features.

    Already binary classification (malignant vs benign).
    """
    from sklearn.datasets import load_breast_cancer
    from sklearn.decomposition import PCA

    data = load_breast_cancer()
    X_full = data.data
    y = data.target.astype(np.intp)

    # PCA to 4 features
    pca = PCA(n_components=4, random_state=42)
    X = pca.fit_transform(X_full)

    return X, y


def _load_digits_01() -> Tuple[NDArray[np.floating], NDArray[np.intp]]:
    """Load Digits dataset filtered to 0 vs 1 only, with PCA to 4 features.

    Uses only digits 0 and 1 for binary classification.
    """
    from sklearn.datasets import load_digits
    from sklearn.decomposition import PCA

    data = load_digits()

    # Filter to digits 0 and 1
    mask = (data.target == 0) | (data.target == 1)
    X_full = data.data[mask]
    y = data.target[mask].astype(np.intp)

    # PCA to 4 features
    pca = PCA(n_components=4, random_state=42)
    X = pca.fit_transform(X_full)

    return X, y


# =============================================================================
# Utility functions
# =============================================================================


def _stratified_subsample(
    X: NDArray[np.floating],
    y: NDArray[np.intp],
    max_samples: int,
    seed: int,
) -> Tuple[NDArray[np.floating], NDArray[np.intp]]:
    """Reduce dataset size while maintaining class balance.

    Uses scikit-learn's ``train_test_split`` with stratification to
    select a representative subset. This is used by ``--smoke`` mode
    to cap real-world dataset sizes for fast end-to-end validation.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Labels.
    max_samples : int
        Target number of samples.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    X_sub, y_sub : tuple of np.ndarray
        Subsampled feature matrix and labels.
    """
    if max_samples >= len(X):
        return X, y

    from sklearn.model_selection import train_test_split

    X_sub, _, y_sub, _ = train_test_split(
        X, y,
        train_size=max_samples,
        stratify=y,
        random_state=seed,
    )
    logger.debug(
        "Stratified subsample: %d -> %d samples", len(X), len(X_sub),
    )
    return X_sub, y_sub


def _scale_to_range(
    X: NDArray[np.floating],
    low: float = 0.0,
    high: float = 2 * np.pi,
) -> NDArray[np.floating]:
    """Scale features to [low, high] range via min-max normalization.

    Parameters
    ----------
    X : np.ndarray
        Input features, shape (n_samples, n_features).
    low : float, default=0.0
        Lower bound of output range.
    high : float, default=2π
        Upper bound of output range.

    Returns
    -------
    np.ndarray
        Scaled features in [low, high].
    """
    X_min = X.min(axis=0, keepdims=True)
    X_max = X.max(axis=0, keepdims=True)

    # Avoid division by zero for constant features
    scale = X_max - X_min
    scale = np.where(scale == 0, 1.0, scale)

    X_normalized = (X - X_min) / scale  # [0, 1]
    X_scaled = X_normalized * (high - low) + low  # [low, high]

    return X_scaled
