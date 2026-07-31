"""Tests for the unified encoding profiler.

Covers full-profile completeness, data-dependent axes, custom (non-built-in)
encodings, atlas comparison, and graceful per-axis failure handling. Small
sample counts keep the quantum simulations fast.
"""

from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from encoding_atlas import AngleEncoding, IQPEncoding
from encoding_atlas.analysis.profile import (
    EncodingCharacterization,
    atlas_comparable_metrics,
    compare_to_atlas,
    profile_encoding,
)

# Small, fast profiling configuration for tests.
_FAST = dict(
    expressibility_samples=30,
    entanglement_samples=30,
    trainability_samples=15,
    noise_samples=3,
    concentration_samples=20,
    seed=0,
)

_DATA_FREE_KEYS = {
    "depth",
    "expressibility",
    "entanglement_capability",
    "trainability_estimate",
    "noise_retained_fidelity",
    "kernel_concentration_ratio",
    "kernel_offdiagonal_mean",
    "kernel_shots_per_entry",
    "kernel_is_concentrated",
}


@pytest.fixture(scope="module")
def base_profile() -> EncodingCharacterization:
    return profile_encoding(AngleEncoding(n_features=2, rotation="Y"), **_FAST)


@pytest.fixture
def data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0.5, 0.2, (8, 2)), rng.normal(2.7, 0.2, (8, 2))])
    y = np.array([0] * 8 + [1] * 8, dtype=np.intp)
    return X, y


class TestProfileCompleteness:
    def test_identity_fields(self, base_profile: EncodingCharacterization) -> None:
        assert base_profile.encoding_name == "AngleEncoding"
        assert base_profile.n_qubits == 2
        assert base_profile.n_features == 2
        assert base_profile.is_simulable is True

    def test_data_free_axes_present(
        self, base_profile: EncodingCharacterization
    ) -> None:
        assert set(base_profile.metrics) >= _DATA_FREE_KEYS
        for key in _DATA_FREE_KEYS:
            assert base_profile.metrics[key] is not None

    def test_no_data_axes_without_data(
        self, base_profile: EncodingCharacterization
    ) -> None:
        for key in (
            "kernel_target_alignment",
            "geometric_difference",
            "effective_dimension",
        ):
            assert key not in base_profile.metrics

    def test_metrics_are_read_only(
        self, base_profile: EncodingCharacterization
    ) -> None:
        assert isinstance(base_profile.metrics, MappingProxyType)
        with pytest.raises(TypeError):
            base_profile.metrics["depth"] = 0  # type: ignore[index]

    def test_metric_accessor_none_safe(
        self, base_profile: EncodingCharacterization
    ) -> None:
        assert base_profile.metric("nope") is None
        assert base_profile.metric("nope", default=-1.0) == -1.0

    def test_non_entangling_has_zero_entanglement(
        self, base_profile: EncodingCharacterization
    ) -> None:
        assert base_profile.metrics["entanglement_capability"] == pytest.approx(
            0.0, abs=1e-6
        )

    def test_no_failures_for_valid_encoding(
        self, base_profile: EncodingCharacterization
    ) -> None:
        assert dict(base_profile.notes) == {}


class TestDataDependentAxes:
    def test_geometry_axes_with_X_only(
        self, data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        X, _ = data
        p = profile_encoding(AngleEncoding(n_features=2), X=X, **_FAST)
        assert p.metrics["geometric_difference"] is not None
        assert p.metrics["effective_dimension"] is not None
        # KTA needs labels, which were not supplied.
        assert "kernel_target_alignment" not in p.metrics

    def test_kta_with_X_and_y(self, data: tuple[np.ndarray, np.ndarray]) -> None:
        X, y = data
        p = profile_encoding(AngleEncoding(n_features=2), X=X, y=y, **_FAST)
        assert p.metrics["kernel_target_alignment"] is not None
        assert -1.0 <= p.metrics["kernel_target_alignment"] <= 1.0


class TestConcentrationAxis:
    """The concentration axis is data-free, so it is always present."""

    def test_present_without_data(self, base_profile: EncodingCharacterization) -> None:
        assert base_profile.metrics["kernel_concentration_ratio"] > 0.0
        assert 0.0 <= base_profile.metrics["kernel_offdiagonal_mean"] <= 1.0
        assert base_profile.metrics["kernel_shots_per_entry"] > 0.0
        assert isinstance(base_profile.metrics["kernel_is_concentrated"], bool)

    def test_flag_agrees_with_ratio(
        self, base_profile: EncodingCharacterization
    ) -> None:
        assert base_profile.metrics["kernel_is_concentrated"] == (
            base_profile.metrics["kernel_concentration_ratio"] < 2.0
        )

    def test_uses_supplied_data(self, data: tuple[np.ndarray, np.ndarray]) -> None:
        """With X given the axis describes the user's data, not random inputs."""
        X, _ = data
        with_data = profile_encoding(AngleEncoding(n_features=2), X=X, **_FAST)
        without = profile_encoding(AngleEncoding(n_features=2), **_FAST)
        assert (
            with_data.metrics["kernel_concentration_ratio"]
            != without.metrics["kernel_concentration_ratio"]
        )

    def test_entangling_map_is_more_concentrated(self) -> None:
        angle = profile_encoding(AngleEncoding(n_features=6), **_FAST)
        iqp = profile_encoding(IQPEncoding(n_features=6, reps=2), **_FAST)
        assert (
            iqp.metrics["kernel_concentration_ratio"]
            < angle.metrics["kernel_concentration_ratio"]
        )

    def test_failure_nulls_the_whole_axis(self) -> None:
        p = profile_encoding(AngleEncoding(n_features=2), X=np.zeros((5, 3)), **_FAST)
        for key in (
            "kernel_concentration_ratio",
            "kernel_offdiagonal_mean",
            "kernel_shots_per_entry",
            "kernel_is_concentrated",
        ):
            assert p.metrics[key] is None
        assert "kernel_concentration_ratio" in p.notes


class TestCustomEncoding:
    def test_custom_subclass_profiled(self) -> None:
        class MyCustomAngle(AngleEncoding):
            pass

        p = profile_encoding(MyCustomAngle(n_features=2), **_FAST)
        assert p.encoding_name == "MyCustomAngle"
        assert p.metrics["expressibility"] is not None


class TestGracefulFailure:
    def test_bad_data_isolates_failure(self) -> None:
        # X with the wrong feature count fails only the data-dependent axes;
        # the data-free axes must still be computed.
        p = profile_encoding(AngleEncoding(n_features=2), X=np.zeros((5, 3)), **_FAST)
        assert p.metrics["geometric_difference"] is None
        assert "geometric_difference" in p.notes
        assert p.metrics["expressibility"] is not None


class TestAtlasComparison:
    def test_comparable_metric_list(self) -> None:
        assert atlas_comparable_metrics() == [
            "depth",
            "expressibility",
            "entanglement_capability",
            "trainability_estimate",
        ]

    def test_compare_structure(self, base_profile: EncodingCharacterization) -> None:
        c = compare_to_atlas(base_profile, "expressibility")
        assert {
            "metric",
            "value",
            "rank",
            "n_atlas",
            "percentile",
            "beats",
            "higher_is_better",
        } <= set(c)
        assert 1 <= c["rank"] <= c["n_atlas"] + 1
        assert 0.0 <= c["percentile"] <= 100.0
        assert c["higher_is_better"] is True

    def test_depth_is_lower_is_better(
        self, base_profile: EncodingCharacterization
    ) -> None:
        c = compare_to_atlas(base_profile, "depth")
        assert c["higher_is_better"] is False
        # Angle's depth is shallow, so it beats most of the atlas.
        assert c["beats"] >= c["n_atlas"] // 2

    def test_non_comparable_metric_raises(
        self, base_profile: EncodingCharacterization
    ) -> None:
        with pytest.raises(ValueError, match="not atlas-comparable"):
            compare_to_atlas(base_profile, "noise_retained_fidelity")

    def test_uncomputed_metric_raises(self) -> None:
        # Profile without the noise axis, then request an unavailable one.
        p = profile_encoding(AngleEncoding(n_features=2), include_noise=False, **_FAST)
        # depth is computed; force an uncomputed comparable metric via bad data
        # is complex, so assert the error path for a value that is None.
        empty = EncodingCharacterization(
            encoding_name="X",
            n_qubits=2,
            n_features=2,
            is_simulable=True,
            metrics=MappingProxyType({"expressibility": None}),
            notes=MappingProxyType({"expressibility": "forced"}),
        )
        with pytest.raises(ValueError, match="was not computed"):
            compare_to_atlas(empty, "expressibility")
        assert p.metrics["depth"] is not None


class TestEntanglingProfile:
    def test_iqp_profiles(self) -> None:
        p = profile_encoding(IQPEncoding(n_features=2, reps=1), **_FAST)
        assert p.encoding_name == "IQPEncoding"
        # IQP is entangling and noise-fragile.
        assert p.metrics["entanglement_capability"] > 0.0
        assert p.metrics["noise_retained_fidelity"] < 1.0
