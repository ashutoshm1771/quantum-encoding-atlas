"""Tests for scikit-learn estimator-contract compliance.

The four estimators are meant to compose with the scikit-learn ecosystem, so
these tests exercise the ecosystem itself — ``clone``, ``cross_val_score``,
``GridSearchCV``, ``Pipeline`` and the meta-estimators — rather than only
asserting that the right base classes are present. If a future change breaks
the contract, a real integration is what should fail.

The contract also constrains ``__init__``: it must store its arguments
verbatim, with no validation and no attributes ending in an underscore, so
that ``clone`` and ``set_params`` can rebuild an estimator from
``get_params()``. Both halves are pinned here.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import numpy as np
import pytest
from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    cross_val_score,
    cross_validate,
    learning_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.validation import NotFittedError

from encoding_atlas import AngleEncoding, IQPEncoding
from encoding_atlas.benchmark import (
    QuantumKernelClassifier,
    QuantumKernelRegressor,
    VQCClassifier,
    VQCRegressor,
)

# Small and fast: the point is the plumbing, not the quantum model quality.
N = 24
EPOCHS = 2


def _encoding() -> AngleEncoding:
    return AngleEncoding(n_features=2, rotation="Y")


@pytest.fixture(scope="module")
def classification() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = np.vstack(
        [rng.normal(0.5, 0.2, (N // 2, 2)), rng.normal(2.5, 0.2, (N // 2, 2))]
    )
    y = np.array([0] * (N // 2) + [1] * (N // 2), dtype=np.intp)
    return X, y


@pytest.fixture(scope="module")
def regression() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(1)
    X = rng.uniform(0.0, np.pi, (N, 2))
    return X, np.sin(X[:, 0]) + 0.1 * X[:, 1]


CLASSIFIERS: list[tuple[str, Callable[[], Any]]] = [
    ("QuantumKernelClassifier", lambda: QuantumKernelClassifier(_encoding())),
    ("VQCClassifier", lambda: VQCClassifier(_encoding(), epochs=EPOCHS)),
]
REGRESSORS: list[tuple[str, Callable[[], Any]]] = [
    ("QuantumKernelRegressor", lambda: QuantumKernelRegressor(_encoding())),
    ("VQCRegressor", lambda: VQCRegressor(_encoding(), epochs=EPOCHS)),
]
ALL = CLASSIFIERS + REGRESSORS
ALL_IDS = [name for name, _ in ALL]


# =====================================================================
# The __init__ contract
# =====================================================================


class TestConstructorContract:
    """``__init__`` stores arguments verbatim and does nothing else."""

    @pytest.mark.parametrize(("name", "factory"), ALL, ids=ALL_IDS)
    def test_inherits_base_estimator(
        self, name: str, factory: Callable[[], Any]
    ) -> None:
        assert isinstance(factory(), BaseEstimator)

    @pytest.mark.parametrize(("name", "factory"), ALL, ids=ALL_IDS)
    def test_no_fitted_attributes_before_fit(
        self, name: str, factory: Callable[[], Any]
    ) -> None:
        """Trailing-underscore attributes are what ``check_is_fitted`` keys on."""
        estimator = factory()
        premature = [
            attr
            for attr in vars(estimator)
            if attr.endswith("_") and not attr.startswith("__")
        ]
        assert premature == [], f"{name} exposes {premature} before fit"

    @pytest.mark.parametrize(("name", "factory"), ALL, ids=ALL_IDS)
    def test_get_params_matches_init_signature(
        self, name: str, factory: Callable[[], Any]
    ) -> None:
        estimator = factory()
        expected = {
            p
            for p in inspect.signature(type(estimator).__init__).parameters
            if p != "self"
        }
        assert set(estimator.get_params()) == expected

    @pytest.mark.parametrize(("name", "factory"), ALL, ids=ALL_IDS)
    def test_params_stored_unmodified(
        self, name: str, factory: Callable[[], Any]
    ) -> None:
        estimator = factory()
        for key, value in estimator.get_params().items():
            assert getattr(estimator, key) is value

    @pytest.mark.parametrize(("name", "factory"), ALL, ids=ALL_IDS)
    def test_constructor_does_not_validate(
        self, name: str, factory: Callable[[], Any]
    ) -> None:
        """Invalid values must construct; ``GridSearchCV`` relies on this."""
        estimator = factory()
        bad = {"C": -1.0, "alpha": -1.0, "n_var_layers": 0, "lr": -1.0, "epochs": 0}
        applicable = {k: v for k, v in bad.items() if k in estimator.get_params()}
        assert applicable, f"{name} has no numeric hyper-parameter to test"
        type(estimator)(**{**estimator.get_params(), **applicable})

    @pytest.mark.parametrize(("name", "factory"), ALL, ids=ALL_IDS)
    def test_encoding_read_at_fit_not_construction(
        self, name: str, factory: Callable[[], Any]
    ) -> None:
        """A cached circuit width would go stale under ``set_params``."""
        estimator = factory()
        estimator.set_params(encoding=IQPEncoding(n_features=2, reps=1))
        assert isinstance(estimator.encoding, IQPEncoding)


# =====================================================================
# clone / get_params / set_params
# =====================================================================


class TestCloneRoundTrip:
    @pytest.mark.parametrize(("name", "factory"), ALL, ids=ALL_IDS)
    def test_clone_produces_an_equivalent_unfitted_estimator(
        self, name: str, factory: Callable[[], Any]
    ) -> None:
        original = factory()
        copy = clone(original)
        assert copy is not original
        assert type(copy) is type(original)
        assert copy.get_params()["encoding"] == original.get_params()["encoding"]

    @pytest.mark.parametrize(("name", "factory"), ALL, ids=ALL_IDS)
    def test_clone_of_a_fitted_estimator_is_unfitted(
        self,
        name: str,
        factory: Callable[[], Any],
        classification: tuple[np.ndarray, np.ndarray],
        regression: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = classification if "Classifier" in name else regression
        fitted = factory().fit(X, y)
        fresh = clone(fitted)
        with pytest.raises(NotFittedError):
            fresh.predict(X)

    @pytest.mark.parametrize(("name", "factory"), ALL, ids=ALL_IDS)
    def test_set_params_round_trips(
        self, name: str, factory: Callable[[], Any]
    ) -> None:
        estimator = factory()
        params = estimator.get_params()
        assert estimator.set_params(**params).get_params() == params

    def test_repr_is_informative(self) -> None:
        text = repr(QuantumKernelClassifier(_encoding(), C=2.0))
        assert "QuantumKernelClassifier" in text
        assert "C=2.0" in text


# =====================================================================
# Estimator type tags and fitted attributes
# =====================================================================


class TestEstimatorTags:
    @pytest.mark.parametrize(("name", "factory"), CLASSIFIERS)
    def test_classifiers_are_tagged(
        self, name: str, factory: Callable[[], Any]
    ) -> None:
        assert is_classifier(factory())
        assert not is_regressor(factory())

    @pytest.mark.parametrize(("name", "factory"), REGRESSORS)
    def test_regressors_are_tagged(self, name: str, factory: Callable[[], Any]) -> None:
        assert is_regressor(factory())
        assert not is_classifier(factory())

    @pytest.mark.parametrize(("name", "factory"), CLASSIFIERS)
    def test_classes_and_n_features_set_by_fit(
        self,
        name: str,
        factory: Callable[[], Any],
        classification: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = classification
        model = factory().fit(X, y)
        assert np.array_equal(model.classes_, np.unique(y))
        assert model.n_features_in_ == X.shape[1]

    @pytest.mark.parametrize(("name", "factory"), REGRESSORS)
    def test_n_features_set_by_fit(
        self,
        name: str,
        factory: Callable[[], Any],
        regression: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = regression
        assert factory().fit(X, y).n_features_in_ == X.shape[1]


# =====================================================================
# The ecosystem itself
# =====================================================================


class TestEcosystemIntegration:
    @pytest.mark.parametrize(("name", "factory"), CLASSIFIERS)
    def test_cross_val_score(
        self,
        name: str,
        factory: Callable[[], Any],
        classification: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = classification
        scores = cross_val_score(factory(), X, y, cv=3)
        assert len(scores) == 3
        assert np.all((scores >= 0.0) & (scores <= 1.0))

    @pytest.mark.parametrize(("name", "factory"), REGRESSORS)
    def test_cross_val_score_regression(
        self,
        name: str,
        factory: Callable[[], Any],
        regression: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = regression
        scores = cross_val_score(factory(), X, y, cv=3)
        assert len(scores) == 3
        assert np.all(np.isfinite(scores))

    def test_cross_validate(
        self, classification: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = classification
        out = cross_validate(QuantumKernelClassifier(_encoding()), X, y, cv=3)
        assert len(out["test_score"]) == 3

    def test_grid_search_over_hyperparameters(
        self, classification: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = classification
        search = GridSearchCV(
            QuantumKernelClassifier(_encoding()), {"C": [0.5, 2.0]}, cv=2
        ).fit(X, y)
        assert search.best_params_["C"] in (0.5, 2.0)
        assert search.predict(X).shape == y.shape

    def test_grid_search_over_encodings(
        self, classification: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The payoff: the encoding itself becomes a tunable hyper-parameter."""
        X, y = classification
        candidates = [_encoding(), IQPEncoding(n_features=2, reps=1)]
        search = GridSearchCV(
            QuantumKernelClassifier(_encoding()), {"encoding": candidates}, cv=2
        ).fit(X, y)
        assert type(search.best_params_["encoding"]) in {AngleEncoding, IQPEncoding}

    def test_learning_curve(
        self, classification: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = classification
        sizes, train, test = learning_curve(
            QuantumKernelClassifier(_encoding()), X, y, cv=2, train_sizes=[0.5, 1.0]
        )
        assert len(sizes) == 2 and train.shape[0] == 2 and test.shape[0] == 2

    def test_pipeline_with_scaler(
        self, classification: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Scaling in a pipeline is how the feature-range finding gets applied."""
        X, y = classification
        pipe = make_pipeline(
            MinMaxScaler((0.0, np.pi / 2)), QuantumKernelClassifier(_encoding())
        ).fit(X, y)
        assert pipe.predict(X).shape == y.shape
        assert cross_val_score(pipe, X, y, cv=2).shape == (2,)

    def test_voting_classifier(
        self, classification: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = classification
        ensemble = VotingClassifier(
            [
                ("kernel", QuantumKernelClassifier(_encoding())),
                ("vqc", VQCClassifier(_encoding(), epochs=EPOCHS)),
            ]
        ).fit(X, y)
        assert ensemble.predict(X).shape == y.shape

    def test_calibration(self, classification: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = classification
        calibrated = CalibratedClassifierCV(
            QuantumKernelClassifier(_encoding()), cv=2
        ).fit(X, y)
        probs = calibrated.predict_proba(X)
        assert probs.shape == (len(y), 2)
        assert np.allclose(probs.sum(axis=1), 1.0)


class TestDecisionFunction:
    """``decision_function`` unlocks threshold-based metrics for the SVM path."""

    def test_shape_and_sign_agree_with_predict(
        self, classification: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = classification
        model = QuantumKernelClassifier(_encoding()).fit(X, y)
        scores = model.decision_function(X)
        assert scores.shape == (len(y),)
        predicted_from_scores = model.classes_[(scores > 0).astype(int)]
        assert np.array_equal(predicted_from_scores, model.predict(X))

    def test_supports_roc_auc(
        self, classification: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = classification
        model = QuantumKernelClassifier(_encoding()).fit(X, y)
        assert 0.0 <= roc_auc_score(y, model.decision_function(X)) <= 1.0

    def test_requires_fit(self) -> None:
        with pytest.raises(NotFittedError):
            QuantumKernelClassifier(_encoding()).decision_function(np.zeros((2, 2)))


# =====================================================================
# Fit-time validation
# =====================================================================


class TestFitTimeValidation:
    @pytest.mark.parametrize(
        ("factory", "kwargs", "match"),
        [
            (lambda: QuantumKernelClassifier(_encoding()), {"C": 0.0}, "C must be"),
            (lambda: QuantumKernelRegressor(_encoding()), {"alpha": 0.0}, "alpha"),
            (
                lambda: VQCClassifier(_encoding(), epochs=EPOCHS),
                {"n_var_layers": 0},
                "n_var_layers",
            ),
            (lambda: VQCClassifier(_encoding(), epochs=EPOCHS), {"lr": 0.0}, "lr"),
            (lambda: VQCClassifier(_encoding()), {"epochs": 0}, "epochs"),
        ],
    )
    def test_bad_hyperparameter_raises_at_fit(
        self,
        factory: Callable[[], Any],
        kwargs: dict,
        match: str,
        classification: tuple[np.ndarray, np.ndarray],
        regression: tuple[np.ndarray, np.ndarray],
    ) -> None:
        estimator = factory().set_params(**kwargs)
        X, y = regression if is_regressor(estimator) else classification
        with pytest.raises(ValueError, match=match):
            estimator.fit(X, y)

    @pytest.mark.parametrize(("name", "factory"), ALL, ids=ALL_IDS)
    def test_missing_encoding_raises_at_fit(
        self,
        name: str,
        factory: Callable[[], Any],
        classification: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = classification
        estimator = factory().set_params(encoding=None)
        with pytest.raises(ValueError, match="encoding must be set"):
            estimator.fit(X, y)

    @pytest.mark.parametrize(("name", "factory"), ALL, ids=ALL_IDS)
    def test_predict_before_fit_raises_not_fitted(
        self, name: str, factory: Callable[[], Any]
    ) -> None:
        with pytest.raises(NotFittedError) as excinfo:
            factory().predict(np.zeros((2, 2)))
        # NotFittedError subclasses ValueError, so older callers still catch it.
        assert isinstance(excinfo.value, ValueError)
        assert "not fitted" in str(excinfo.value)

    @pytest.mark.parametrize(("name", "factory"), ALL, ids=ALL_IDS)
    def test_feature_count_mismatch_raises(
        self,
        name: str,
        factory: Callable[[], Any],
        classification: tuple[np.ndarray, np.ndarray],
        regression: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = classification if "Classifier" in name else regression
        with pytest.raises(ValueError, match="encoding expects"):
            factory().fit(np.hstack([X, X]), y)

    @pytest.mark.parametrize(("name", "factory"), ALL, ids=ALL_IDS)
    def test_inconsistent_sample_counts_raise(
        self, name: str, factory: Callable[[], Any]
    ) -> None:
        with pytest.raises(ValueError, match="inconsistent numbers of samples"):
            factory().fit(np.zeros((4, 2)), np.array([0, 1]))

    @pytest.mark.parametrize(("name", "factory"), ALL, ids=ALL_IDS)
    def test_non_finite_features_raise(
        self, name: str, factory: Callable[[], Any]
    ) -> None:
        X = np.array([[0.1, 0.2], [np.nan, 0.4]])
        with pytest.raises(ValueError, match="NaN or infinite"):
            factory().fit(X, np.array([0, 1]))

    @pytest.mark.parametrize(("name", "factory"), ALL, ids=ALL_IDS)
    def test_predict_with_wrong_width_raises(
        self,
        name: str,
        factory: Callable[[], Any],
        classification: tuple[np.ndarray, np.ndarray],
        regression: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = classification if "Classifier" in name else regression
        model = factory().fit(X, y)
        with pytest.raises(ValueError, match="features"):
            model.predict(np.zeros((3, 5)))


# =====================================================================
# Score semantics are unchanged by inheriting the mixins
# =====================================================================


class TestScoreEquivalence:
    @pytest.mark.parametrize(("name", "factory"), CLASSIFIERS)
    def test_classifier_score_is_accuracy(
        self,
        name: str,
        factory: Callable[[], Any],
        classification: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = classification
        model = factory().fit(X, y)
        assert model.score(X, y) == pytest.approx(accuracy_score(y, model.predict(X)))

    @pytest.mark.parametrize(("name", "factory"), REGRESSORS)
    def test_regressor_score_is_r2(
        self,
        name: str,
        factory: Callable[[], Any],
        regression: tuple[np.ndarray, np.ndarray],
    ) -> None:
        X, y = regression
        model = factory().fit(X, y)
        assert model.score(X, y) == pytest.approx(r2_score(y, model.predict(X)))


# =====================================================================
# Backward compatibility of the public surface
# =====================================================================


class TestBackwardCompatibility:
    def test_positional_encoding_still_accepted(self) -> None:
        assert QuantumKernelClassifier(_encoding()).C == 1.0
        assert VQCClassifier(_encoding(), 3, 0.1, 5, 0).n_var_layers == 3

    def test_documented_fitted_attributes_survive(
        self, classification: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = classification
        vqc = VQCClassifier(_encoding(), epochs=EPOCHS).fit(X, y)
        assert vqc.params_.shape == (vqc.n_var_layers, _encoding().n_qubits)
        assert len(vqc.loss_history_) >= 1
        assert vqc.status_ in {"success", "diverged"}
        assert vqc.get_final_loss() is not None

        kernel = QuantumKernelClassifier(_encoding()).fit(X, y)
        assert isinstance(kernel.kernel_regularized_, bool)

    def test_get_final_loss_is_none_before_fit(self) -> None:
        assert VQCClassifier(_encoding()).get_final_loss() is None
        assert VQCRegressor(_encoding()).get_final_loss() is None

    def test_multiclass_path_still_works(self) -> None:
        rng = np.random.default_rng(0)
        X = np.vstack(
            [
                rng.normal(0.4, 0.15, (6, 2)),
                rng.normal(1.6, 0.15, (6, 2)),
                rng.normal(2.8, 0.15, (6, 2)),
            ]
        )
        y = np.array([0] * 6 + [1] * 6 + [2] * 6, dtype=np.intp)
        model = VQCClassifier(_encoding(), epochs=EPOCHS, seed=0).fit(X, y)
        assert np.array_equal(model.classes_, np.array([0, 1, 2]))
        assert set(model.predict(X)).issubset({0, 1, 2})
        assert model.predict_proba(X).shape == (len(y), 3)
