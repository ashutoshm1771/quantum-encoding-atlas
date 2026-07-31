"""Tests for data-driven encoding screening.

The load-bearing test is :class:`TestPredictsAccuracy`: screening is only worth
shipping if the alignment ranking actually tracks measured kernel accuracy, so
that is asserted directly against SVMs trained on the same kernels rather than
taken on faith. The rest covers ranking determinism, label handling,
sub-sampling, graceful skipping, and validation.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from encoding_atlas.analysis.generalization import compute_fidelity_kernel
from encoding_atlas.benchmark import get_dataset
from encoding_atlas.benchmark.kernel import ensure_psd
from encoding_atlas.core.base import BaseEncoding
from encoding_atlas.guide import ScreenedEncoding, ScreeningResult, screen_encodings
from encoding_atlas.guide._candidates import (
    BENCHMARK_PARAMS,
    build_candidates,
    default_candidate_names,
)

# Small but statistically meaningful: 60 samples give 1,770 off-diagonal pairs.
N_SMALL = 60


@pytest.fixture(scope="module")
def moons() -> tuple[np.ndarray, np.ndarray]:
    X, y = get_dataset("moons", n_samples=N_SMALL, seed=0)
    return np.asarray(X), np.asarray(y)


@pytest.fixture(scope="module")
def screened(moons: tuple[np.ndarray, np.ndarray]) -> ScreeningResult:
    X, y = moons
    return screen_encodings(X, y, seed=0)


# =====================================================================
# Candidate construction
# =====================================================================


class TestCandidates:
    def test_default_set_is_the_benchmarked_sixteen(self) -> None:
        assert len(default_candidate_names()) == 16
        assert set(default_candidate_names()) == set(BENCHMARK_PARAMS)

    def test_all_build_at_two_features(self) -> None:
        built, skipped = build_candidates(2)
        assert len(built) == 16
        assert skipped == {}
        assert all(isinstance(e, BaseEncoding) for _, e in built)

    @pytest.mark.parametrize(
        ("n_features", "expected_failures"),
        [
            (3, {"symmetry_inspired", "so2_equivariant", "swap_equivariant"}),
            (4, {"so2_equivariant"}),
            (5, {"symmetry_inspired", "so2_equivariant", "swap_equivariant"}),
        ],
    )
    def test_unsupported_widths_are_reported_not_raised(
        self, n_features: int, expected_failures: set[str]
    ) -> None:
        built, skipped = build_candidates(n_features)
        assert set(skipped) == expected_failures
        assert len(built) == 16 - len(expected_failures)
        assert all(":" in reason for reason in skipped.values())

    def test_restriction_preserves_requested_order(self) -> None:
        built, _ = build_candidates(2, ["iqp", "angle"])
        assert [name for name, _ in built] == ["iqp", "angle"]

    def test_duplicates_collapse(self) -> None:
        built, _ = build_candidates(2, ["angle", "angle", "iqp"])
        assert [name for name, _ in built] == ["angle", "iqp"]

    def test_unknown_name_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown encoding"):
            build_candidates(2, ["not_an_encoding"])

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True])
    def test_bad_feature_count_raises(self, bad: object) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            build_candidates(bad)  # type: ignore[arg-type]

    def test_trainable_encoding_is_seeded(self) -> None:
        """Unseeded variational parameters would make screening irreproducible."""
        assert "seed" in BENCHMARK_PARAMS["trainable_encoding"]


# =====================================================================
# Result structure and ranking
# =====================================================================


class TestScreeningResult:
    def test_scores_every_candidate(self, screened: ScreeningResult) -> None:
        assert len(screened.candidates) == 16
        assert screened.skipped == {}
        assert screened.n_features == 2
        assert screened.n_samples_used == N_SMALL
        assert screened.n_samples_supplied == N_SMALL
        assert screened.centered is True
        assert screened.shots is None

    def test_ranked_by_descending_alignment(self, screened: ScreeningResult) -> None:
        alignments = [c.alignment for c in screened.candidates]
        assert alignments == sorted(alignments, reverse=True)
        assert [c.rank for c in screened.candidates] == list(range(1, 17))

    def test_alignment_in_valid_range(self, screened: ScreeningResult) -> None:
        assert all(-1.0 <= c.alignment <= 1.0 for c in screened.candidates)

    def test_fields_are_populated(self, screened: ScreeningResult) -> None:
        best = screened.best()
        assert isinstance(best, ScreenedEncoding)
        assert best.rank == 1
        assert best.name in BENCHMARK_PARAMS
        assert best.display_name == type(best.encoding).__name__
        assert best.n_qubits == best.encoding.n_qubits
        assert best.params == BENCHMARK_PARAMS[best.name]
        assert best.atlas_rank is not None and 1 <= best.atlas_rank <= 16
        assert best.atlas_alignment is not None

    def test_returned_encoding_is_ready_to_use(
        self, screened: ScreeningResult, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The point of carrying the instance: train without rebuilding."""
        X, _ = moons
        K = compute_fidelity_kernel(screened.best().encoding, X)
        assert K.shape == (len(X), len(X))

    def test_concentration_absent_by_default(self, screened: ScreeningResult) -> None:
        assert all(c.concentration_ratio is None for c in screened.candidates)
        assert all(c.is_concentrated is None for c in screened.candidates)

    def test_concentration_annotation_when_requested(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = moons
        result = screen_encodings(X, y, seed=0, include_concentration=True)
        for c in result.candidates:
            assert c.concentration_ratio is not None and c.concentration_ratio > 0.0
            assert c.is_concentrated == (c.concentration_ratio < 2.0)

    def test_concentration_does_not_change_the_ranking(
        self, screened: ScreeningResult, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Annotation only — concentration is deliberately not a ranking input."""
        X, y = moons
        annotated = screen_encodings(X, y, seed=0, include_concentration=True)
        assert annotated.names() == screened.names()

    def test_top_defaults_to_three(self, screened: ScreeningResult) -> None:
        assert len(screened.top()) == 3
        assert screened.top() == list(screened.candidates[:3])

    def test_top_clamps_to_available(self, screened: ScreeningResult) -> None:
        assert len(screened.top(999)) == len(screened.candidates)

    @pytest.mark.parametrize("bad", [0, -1, 1.5, True])
    def test_top_validates_k(self, screened: ScreeningResult, bad: object) -> None:
        with pytest.raises(ValueError, match="k must be a positive integer"):
            screened.top(bad)  # type: ignore[arg-type]

    def test_names_and_get(self, screened: ScreeningResult) -> None:
        assert screened.names(3) == [c.name for c in screened.top(3)]
        assert screened.names() == [c.name for c in screened.candidates]
        assert screened.get("ANGLE") is screened.get("angle")
        assert screened.get("nope") is None

    def test_result_is_immutable(self, screened: ScreeningResult) -> None:
        with pytest.raises(FrozenInstanceError):
            screened.n_features = 9  # type: ignore[misc]
        with pytest.raises(FrozenInstanceError):
            screened.candidates[0].alignment = 1.0  # type: ignore[misc]

    def test_best_raises_when_nothing_scored(self) -> None:
        empty = ScreeningResult(
            candidates=(),
            skipped={},
            n_samples_used=0,
            n_samples_supplied=0,
            n_features=2,
            centered=True,
            shots=None,
        )
        with pytest.raises(RuntimeError, match="No encoding could be screened"):
            empty.best()


# =====================================================================
# The claim worth testing: alignment tracks accuracy
# =====================================================================


class TestPredictsAccuracy:
    """Screening exists because alignment predicts kernel accuracy."""

    @pytest.mark.parametrize("dataset", ["moons", "linear", "xor"])
    def test_ranking_correlates_with_measured_accuracy(self, dataset: str) -> None:
        X, y = get_dataset(dataset, n_samples=80, seed=0)
        X, y = np.asarray(X), np.asarray(y)
        result = screen_encodings(X, y, seed=0)

        accuracies = []
        for candidate in result.candidates:
            K = compute_fidelity_kernel(candidate.encoding, X)
            folds = []
            for train, test in StratifiedKFold(3, shuffle=True, random_state=0).split(
                X, y
            ):
                K_train, _ = ensure_psd(K[np.ix_(train, train)])
                svm = SVC(kernel="precomputed", C=1.0).fit(K_train, y[train])
                folds.append(np.mean(svm.predict(K[np.ix_(test, train)]) == y[test]))
            accuracies.append(float(np.mean(folds)))

        rho, _ = spearmanr([c.alignment for c in result.candidates], accuracies)
        assert rho > 0.3, f"{dataset}: alignment should track accuracy, got {rho}"

    def test_shortlist_beats_a_random_pick(self) -> None:
        """Top-3 should land near the best achievable, well above the mean."""
        X, y = get_dataset("xor", n_samples=80, seed=0)
        X, y = np.asarray(X), np.asarray(y)
        result = screen_encodings(X, y, seed=0)

        accuracies = {}
        for candidate in result.candidates:
            K = compute_fidelity_kernel(candidate.encoding, X)
            folds = []
            for train, test in StratifiedKFold(3, shuffle=True, random_state=0).split(
                X, y
            ):
                K_train, _ = ensure_psd(K[np.ix_(train, train)])
                svm = SVC(kernel="precomputed", C=1.0).fit(K_train, y[train])
                folds.append(np.mean(svm.predict(K[np.ix_(test, train)]) == y[test]))
            accuracies[candidate.name] = float(np.mean(folds))

        shortlist = max(accuracies[n] for n in result.names(3))
        assert shortlist >= np.mean(list(accuracies.values()))
        assert shortlist >= max(accuracies.values()) - 0.05

    def test_ranks_haar_floor_encodings_below_angle(self) -> None:
        """The benchmark's bottom encodings should not top a screen on easy data.

        This is the property that makes screening safe to act on: the maps
        whose kernels sit at the Haar floor must not outrank a shallow
        non-entangling one on linearly separable data.
        """
        X, y = get_dataset("linear", n_samples=80, seed=0)
        result = screen_encodings(np.asarray(X), np.asarray(y), seed=0)
        order = result.names()
        assert order.index("angle") < order.index("iqp")
        assert order.index("angle") < order.index("zz_feature_map")


# =====================================================================
# Determinism, sub-sampling, label handling
# =====================================================================


class TestDeterminism:
    def test_same_seed_same_ranking(self, moons: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = moons
        a = screen_encodings(X, y, seed=7)
        b = screen_encodings(X, y, seed=7)
        assert a.names() == b.names()
        assert [c.alignment for c in a.candidates] == [
            c.alignment for c in b.candidates
        ]

    def test_exact_kernels_ignore_seed(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Without sub-sampling or shots there is nothing random left."""
        X, y = moons
        assert [c.alignment for c in screen_encodings(X, y, seed=1).candidates] == [
            c.alignment for c in screen_encodings(X, y, seed=2).candidates
        ]


class TestSubsampling:
    def test_large_input_is_subsampled(self) -> None:
        X, y = get_dataset("moons", n_samples=300, seed=0)
        result = screen_encodings(np.asarray(X), np.asarray(y), max_samples=50, seed=0)
        assert result.n_samples_supplied == 300
        assert result.n_samples_used <= 50

    def test_small_input_is_used_whole(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = moons
        result = screen_encodings(X, y, max_samples=1000, seed=0)
        assert result.n_samples_used == len(X)

    def test_subsample_keeps_both_classes(self) -> None:
        X, y = get_dataset("moons", n_samples=300, seed=0)
        # Reach into the helper directly to assert the class balance property.
        from encoding_atlas.guide.screening import _stratified_subsample

        y_arr = np.asarray(y, dtype=np.float64)
        idx = _stratified_subsample(len(y_arr), y_arr, 40, seed=0)
        assert len(np.unique(y_arr[idx])) == 2
        assert len(idx) <= 44  # proportional allocation, rounding-tolerant

    def test_subsample_is_deterministic(self) -> None:
        from encoding_atlas.guide.screening import _stratified_subsample

        y = np.asarray([0] * 50 + [1] * 50, dtype=np.float64)
        a = _stratified_subsample(100, y, 30, seed=3)
        b = _stratified_subsample(100, y, 30, seed=3)
        assert np.array_equal(a, b)


class TestLabelHandling:
    @pytest.mark.parametrize("shift", [0, 1, 5])
    def test_label_convention_does_not_change_the_ranking(
        self, moons: tuple[np.ndarray, np.ndarray], shift: int
    ) -> None:
        """{0,1}, {1,2} and {5,6} describe the same split and must score alike."""
        X, y = moons
        base = screen_encodings(X, y, seed=0)
        shifted = screen_encodings(X, y + shift, seed=0)
        assert shifted.names() == base.names()
        assert [c.alignment for c in shifted.candidates] == pytest.approx(
            [c.alignment for c in base.candidates]
        )

    def test_signed_labels_also_work(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = moons
        base = screen_encodings(X, y, seed=0)
        signed = screen_encodings(X, 2 * y - 1, seed=0)
        assert signed.names() == base.names()

    def test_uncentered_variant_available(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = moons
        result = screen_encodings(X, y, centered=False, seed=0)
        assert result.centered is False
        assert len(result.candidates) == 16

    def test_multiclass_rejected(self) -> None:
        X, y = get_dataset("moons", n_samples=30, seed=0)
        y = np.asarray(y).copy()
        y[:5] = 2
        with pytest.raises(ValueError, match="two-class"):
            screen_encodings(np.asarray(X), y)

    def test_single_class_rejected(self) -> None:
        X, _ = get_dataset("moons", n_samples=30, seed=0)
        with pytest.raises(ValueError, match="two-class"):
            screen_encodings(np.asarray(X), np.zeros(30, dtype=int))

    def test_continuous_target_rejected(self) -> None:
        X, _ = get_dataset("moons", n_samples=30, seed=0)
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="two-class"):
            screen_encodings(np.asarray(X), rng.random(30))


class TestShotAwareScreening:
    def test_shots_recorded_and_ranking_survives(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Alignment aggregates over all pairs, so it tolerates shot noise."""
        X, y = moons
        exact = screen_encodings(X, y, seed=0)
        sampled = screen_encodings(X, y, shots=4000, seed=0)
        assert sampled.shots == 4000
        for candidate in sampled.candidates:
            reference = exact.get(candidate.name)
            assert reference is not None
            assert candidate.alignment == pytest.approx(reference.alignment, abs=0.1)

    def test_invalid_shots_rejected(self, moons: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = moons
        result = screen_encodings(X, y, shots=0, seed=0)
        # Every candidate fails identically, and the reason is reported.
        assert result.candidates == ()
        assert len(result.skipped) == 16
        assert all("shots" in reason for reason in result.skipped.values())


class TestSkippingAndRestriction:
    def test_odd_feature_count_skips_gracefully(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.uniform(0, np.pi, (30, 3))
        y = np.array([0] * 15 + [1] * 15)
        result = screen_encodings(X, y, seed=0)
        assert set(result.skipped) == {
            "symmetry_inspired",
            "so2_equivariant",
            "swap_equivariant",
        }
        assert len(result.candidates) == 13

    def test_restricted_candidate_set(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = moons
        result = screen_encodings(X, y, candidates=["angle", "iqp"], seed=0)
        assert set(result.names()) == {"angle", "iqp"}

    def test_unknown_candidate_raises(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = moons
        with pytest.raises(ValueError, match="Unknown encoding"):
            screen_encodings(X, y, candidates=["nope"])


class TestValidation:
    def test_one_dimensional_X_raises(self) -> None:
        with pytest.raises(ValueError, match="2D array"):
            screen_encodings(np.array([1.0, 2.0]), np.array([0, 1]))

    def test_length_mismatch_raises(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError, match="same length"):
            screen_encodings(rng.random((10, 2)), np.array([0, 1]))

    def test_too_few_samples_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 samples"):
            screen_encodings(np.array([[0.1, 0.2]]), np.array([0]))

    def test_non_finite_X_raises(self) -> None:
        X = np.array([[0.1, 0.2], [np.nan, 0.4]])
        with pytest.raises(ValueError, match="NaN or infinite"):
            screen_encodings(X, np.array([0, 1]))

    def test_non_finite_y_raises(self) -> None:
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        with pytest.raises(ValueError, match="NaN or infinite"):
            screen_encodings(X, np.array([0.0, np.nan]))

    def test_two_dimensional_y_raises(self) -> None:
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        with pytest.raises(ValueError, match="1D label vector"):
            screen_encodings(X, np.array([[0], [1]]))

    @pytest.mark.parametrize("bad", [0, 1, -5, 2.5, True])
    def test_bad_max_samples_raises(self, bad: object) -> None:
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        with pytest.raises(ValueError, match="max_samples"):
            screen_encodings(X, np.array([0, 1]), max_samples=bad)  # type: ignore[arg-type]
