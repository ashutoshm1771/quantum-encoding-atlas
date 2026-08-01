"""Tests for feature-scaling sensitivity analysis.

The load-bearing test is :class:`TestTheMechanism`: widening the range an
encoding's features are scaled into drives its kernel towards the Haar
concentration floor, and the effect is strongest for the circuits that
scramble fastest. That is asserted directly, because it is the claim the whole
module exists to make measurable.

The closed-form anchor is angle encoding, whose kernel under inputs uniform on
``[0, w]`` is a product of ``cos^2((x - x')/2)`` terms; at ``w = 2*pi`` the
per-qubit mean overlap is exactly 1/2, giving the ``2^-n`` Haar floor.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from encoding_atlas import AngleEncoding, IQPEncoding
from encoding_atlas.analysis.scaling import (
    DEFAULT_FEATURE_RANGES,
    FeatureRangeResult,
    FeatureRangeScan,
    recommend_feature_range,
    scale_to_range,
    scan_feature_ranges,
)
from encoding_atlas.benchmark import get_dataset

N_STAT = 120


@pytest.fixture(scope="module")
def moons() -> tuple[np.ndarray, np.ndarray]:
    X, y = get_dataset("moons", n_samples=N_STAT, seed=0)
    return np.asarray(X), np.asarray(y)


# =====================================================================
# scale_to_range
# =====================================================================


class TestScaleToRange:
    def test_maps_each_feature_onto_the_full_range(self) -> None:
        X = np.array([[0.0, 10.0], [1.0, 20.0], [0.5, 15.0]])
        scaled = scale_to_range(X, 0.0, math.pi)
        assert np.allclose(scaled.min(axis=0), 0.0)
        assert np.allclose(scaled.max(axis=0), math.pi)

    def test_is_per_feature_not_global(self) -> None:
        """Each column is scaled independently, matching MinMaxScaler.

        A global min-max (as in ``utils.preprocessing.scale_features``) would
        leave the narrow column compressed.
        """
        X = np.array([[0.0, 100.0], [1.0, 200.0]])
        scaled = scale_to_range(X, 0.0, 1.0)
        assert np.allclose(scaled, np.array([[0.0, 0.0], [1.0, 1.0]]))

    def test_matches_sklearn(self) -> None:
        from sklearn.preprocessing import MinMaxScaler

        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 3))
        assert np.allclose(
            scale_to_range(X, 0.0, math.pi),
            MinMaxScaler((0.0, math.pi)).fit_transform(X),
        )

    def test_constant_feature_maps_to_the_floor(self) -> None:
        X = np.array([[5.0, 1.0], [5.0, 2.0], [5.0, 3.0]])
        scaled = scale_to_range(X, 0.0, math.pi)
        assert np.allclose(scaled[:, 0], 0.0)
        assert np.allclose(scaled[:, 1], [0.0, math.pi / 2, math.pi])

    def test_reference_fits_on_another_array(self) -> None:
        """The seam that prevents CV leakage: fit on train, apply to test."""
        train = np.array([[0.0], [1.0]])
        test = np.array([[-1.0], [2.0]])
        scaled = scale_to_range(test, 0.0, math.pi, reference=train)
        # Test values outside the training span must escape the target range.
        assert scaled.min() < 0.0
        assert scaled.max() > math.pi

    def test_reference_defaults_to_X(self) -> None:
        X = np.array([[0.0], [1.0]])
        assert np.allclose(
            scale_to_range(X, 0.0, 1.0), scale_to_range(X, 0.0, 1.0, reference=X)
        )

    @pytest.mark.parametrize(
        ("low", "high", "match"),
        [
            (1.0, 0.0, "low < high"),
            (1.0, 1.0, "low < high"),
            (0.0, math.inf, "finite"),
            (math.nan, 1.0, "finite"),
        ],
    )
    def test_invalid_range_raises(self, low: float, high: float, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            scale_to_range(np.zeros((3, 2)), low, high)

    def test_non_2d_raises(self) -> None:
        with pytest.raises(ValueError, match="2D array"):
            scale_to_range(np.zeros(4), 0.0, 1.0)

    def test_reference_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="reference has"):
            scale_to_range(np.zeros((3, 2)), 0.0, 1.0, reference=np.zeros((3, 5)))

    def test_empty_reference_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one row"):
            scale_to_range(np.zeros((3, 2)), 0.0, 1.0, reference=np.zeros((0, 2)))


# =====================================================================
# The mechanism the module exists to expose
# =====================================================================


class TestTheMechanism:
    @pytest.mark.parametrize(
        "encoding_factory",
        [
            lambda: IQPEncoding(n_features=2, reps=2),
            lambda: AngleEncoding(n_features=2, rotation="Y"),
        ],
        ids=["iqp", "angle"],
    )
    def test_wider_range_moves_the_kernel_towards_the_haar_mean(
        self, moons: tuple[np.ndarray, np.ndarray], encoding_factory: object
    ) -> None:
        """The mechanism: widening drives overlaps down towards ``1/2**n``."""
        from encoding_atlas.analysis.concentration import haar_kernel_moments

        X, y = moons
        encoding = encoding_factory()  # type: ignore[operator]
        scan = scan_feature_ranges(encoding, X, y, seed=0)
        haar_mean, _ = haar_kernel_moments(scan.n_qubits)
        # DEFAULT_FEATURE_RANGES is ordered narrow -> wide.
        narrowest, widest = scan.results[0], scan.results[-1]
        assert abs(widest.offdiagonal_mean - haar_mean) < abs(
            narrowest.offdiagonal_mean - haar_mean
        )

    @pytest.mark.parametrize(
        "encoding_factory",
        [
            lambda: IQPEncoding(n_features=2, reps=2),
            lambda: AngleEncoding(n_features=2, rotation="Y"),
        ],
        ids=["iqp", "angle"],
    )
    def test_alignment_falls_monotonically_as_the_range_widens(
        self, moons: tuple[np.ndarray, np.ndarray], encoding_factory: object
    ) -> None:
        """The consequence: less usable geometry, so less predictable accuracy."""
        X, y = moons
        scan = scan_feature_ranges(encoding_factory(), X, y, seed=0)  # type: ignore[operator]
        alignments = [r.alignment for r in scan.results]
        assert alignments == sorted(alignments, reverse=True)

    def test_concentration_ratio_is_not_monotone_in_range(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """A documented trap, pinned so the docs stay honest.

        The ratio is a *variance* relative to Haar. A very narrow range makes
        every kernel entry close to 1 — degenerate, but with tiny variance and
        so a *low* ratio. Only a ratio near 1 combined with a mean near
        ``1/2**n`` means the Haar floor. Read the mean alongside it.
        """
        X, y = moons
        scan = scan_feature_ranges(
            AngleEncoding(n_features=2, rotation="Y"), X, y, seed=0
        )
        ratios = [r.concentration_ratio for r in scan.results]
        assert ratios != sorted(ratios)
        assert ratios != sorted(ratios, reverse=True)

    def test_angle_reaches_the_haar_mean_at_a_full_period(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Closed form: uniform inputs on [0, 2*pi] give mean overlap 2^-n.

        This is why the pipeline's own scaling choice is the maximally
        scrambling one.
        """
        X, y = moons
        scan = scan_feature_ranges(
            AngleEncoding(n_features=2, rotation="Y"),
            X,
            y,
            ranges=[(0.0, 2.0 * math.pi)],
            seed=0,
        )
        assert scan.results[0].offdiagonal_mean == pytest.approx(0.25, rel=0.25)

    def test_entangling_encoding_is_more_scaling_sensitive(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The effect concentrates in the circuits that scramble fastest."""
        X, y = moons
        iqp = scan_feature_ranges(IQPEncoding(n_features=2, reps=2), X, y, seed=0)
        angle = scan_feature_ranges(AngleEncoding(n_features=2), X, y, seed=0)
        assert iqp.alignment_spread > angle.alignment_spread

    def test_full_period_is_not_the_best_range_for_an_entangling_map(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """The published [0, 2*pi] choice is not what alignment would pick."""
        X, y = moons
        scan = scan_feature_ranges(IQPEncoding(n_features=2, reps=2), X, y, seed=0)
        assert scan.best_range != (0.0, 2.0 * math.pi)
        published = scan.at(0.0, 2.0 * math.pi)
        assert published is not None
        assert scan.best.alignment > published.alignment


# =====================================================================
# scan_feature_ranges
# =====================================================================


class TestScanFeatureRanges:
    def test_default_ranges_are_ordered_narrow_to_wide(self) -> None:
        widths = [high - low for low, high in DEFAULT_FEATURE_RANGES]
        assert widths == sorted(widths)
        assert DEFAULT_FEATURE_RANGES[-1] == (0.0, 2.0 * math.pi)

    def test_result_structure(self, moons: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = moons
        scan = scan_feature_ranges(AngleEncoding(n_features=2), X, y, seed=0)
        assert isinstance(scan, FeatureRangeScan)
        assert scan.encoding_name == "AngleEncoding"
        assert scan.n_qubits == 2
        assert scan.n_features == 2
        assert scan.n_samples_used == N_STAT
        assert scan.n_samples_supplied == N_STAT
        assert scan.centered is True
        assert len(scan.results) == len(DEFAULT_FEATURE_RANGES)
        for result in scan.results:
            assert isinstance(result, FeatureRangeResult)
            assert -1.0 <= result.alignment <= 1.0
            assert result.width == pytest.approx(result.high - result.low)
            assert result.concentration_ratio >= 0.0

    def test_custom_ranges_are_honoured(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = moons
        scan = scan_feature_ranges(
            AngleEncoding(n_features=2), X, y, ranges=[(0.0, 1.0), (-1.0, 1.0)], seed=0
        )
        assert [(r.low, r.high) for r in scan.results] == [(0.0, 1.0), (-1.0, 1.0)]

    def test_best_maximises_alignment(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = moons
        scan = scan_feature_ranges(AngleEncoding(n_features=2), X, y, seed=0)
        assert scan.best.alignment == max(r.alignment for r in scan.results)
        assert scan.best_range == (scan.best.low, scan.best.high)

    def test_alignment_spread_is_the_range(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = moons
        scan = scan_feature_ranges(AngleEncoding(n_features=2), X, y, seed=0)
        alignments = [r.alignment for r in scan.results]
        assert scan.alignment_spread == pytest.approx(max(alignments) - min(alignments))

    def test_at_looks_up_a_range(self, moons: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = moons
        scan = scan_feature_ranges(AngleEncoding(n_features=2), X, y, seed=0)
        assert scan.at(0.0, math.pi) is not None
        assert scan.at(0.0, 12345.0) is None

    def test_results_are_immutable(self, moons: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = moons
        scan = scan_feature_ranges(AngleEncoding(n_features=2), X, y, seed=0)
        with pytest.raises(FrozenInstanceError):
            scan.results[0].alignment = 0.0  # type: ignore[misc]

    def test_determinism(self, moons: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = moons
        a = scan_feature_ranges(AngleEncoding(n_features=2), X, y, seed=3)
        b = scan_feature_ranges(AngleEncoding(n_features=2), X, y, seed=3)
        assert a == b

    def test_subsampling(self) -> None:
        X, y = get_dataset("moons", n_samples=300, seed=0)
        scan = scan_feature_ranges(
            AngleEncoding(n_features=2),
            np.asarray(X),
            np.asarray(y),
            max_samples=40,
            seed=0,
        )
        assert scan.n_samples_supplied == 300
        assert scan.n_samples_used == 40

    def test_uncentered_variant(self, moons: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = moons
        scan = scan_feature_ranges(
            AngleEncoding(n_features=2), X, y, centered=False, seed=0
        )
        assert scan.centered is False

    def test_label_convention_does_not_matter(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = moons
        base = scan_feature_ranges(AngleEncoding(n_features=2), X, y, seed=0)
        shifted = scan_feature_ranges(AngleEncoding(n_features=2), X, y + 3, seed=0)
        assert [r.alignment for r in shifted.results] == pytest.approx(
            [r.alignment for r in base.results]
        )


class TestScanValidation:
    def test_multiclass_rejected(self) -> None:
        X, y = get_dataset("moons", n_samples=30, seed=0)
        y = np.asarray(y).copy()
        y[:5] = 2
        with pytest.raises(ValueError, match="two-class"):
            scan_feature_ranges(AngleEncoding(n_features=2), np.asarray(X), y)

    def test_empty_ranges_rejected(self, moons: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = moons
        with pytest.raises(ValueError, match="must not be empty"):
            scan_feature_ranges(AngleEncoding(n_features=2), X, y, ranges=[])

    def test_inverted_range_rejected(
        self, moons: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, y = moons
        with pytest.raises(ValueError, match="low < high"):
            scan_feature_ranges(AngleEncoding(n_features=2), X, y, ranges=[(1.0, 0.0)])

    @pytest.mark.parametrize("bad", [0, 1, -3, True])
    def test_bad_max_samples_rejected(self, bad: object) -> None:
        X, y = get_dataset("moons", n_samples=20, seed=0)
        with pytest.raises(ValueError, match="max_samples"):
            scan_feature_ranges(
                AngleEncoding(n_features=2),
                np.asarray(X),
                np.asarray(y),
                max_samples=bad,  # type: ignore[arg-type]
            )

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            scan_feature_ranges(
                AngleEncoding(n_features=2),
                np.zeros((10, 2)),
                np.array([0, 1]),
            )

    def test_non_finite_X_rejected(self) -> None:
        X = np.array([[0.1, 0.2], [np.nan, 0.4]])
        with pytest.raises(ValueError, match="NaN or infinite"):
            scan_feature_ranges(AngleEncoding(n_features=2), X, np.array([0, 1]))


class TestRecommendFeatureRange:
    def test_matches_the_scan(self, moons: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = moons
        enc = IQPEncoding(n_features=2, reps=2)
        assert (
            recommend_feature_range(enc, X, y, seed=0)
            == scan_feature_ranges(enc, X, y, seed=0).best_range
        )

    def test_returns_a_usable_range(self, moons: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = moons
        low, high = recommend_feature_range(
            IQPEncoding(n_features=2, reps=2), X, y, seed=0
        )
        assert low < high
        scaled = scale_to_range(X, low, high)
        assert np.allclose(scaled.min(axis=0), low)
