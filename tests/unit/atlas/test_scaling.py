"""Tests for the bundled feature-scaling sensitivity scan and its query API.

The scan exists to record one thing the study could not: that its headline
expressibility-versus-accuracy correlation depends on the range features are
scaled into. :class:`TestTheContingency` pins that directly, because a shipped
artifact making a claim about the project's own conclusions must be checked,
not trusted.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from encoding_atlas.atlas import (
    ScalingPoint,
    ScalingProfile,
    available_encodings,
    expressibility_accuracy_correlation,
    get_scaling_profile,
    list_scaling_profiles,
    scaling_metadata,
    scaling_sensitive_encodings,
)

PUBLISHED_RANGE = (0.0, 2.0 * math.pi)


class TestDatasetIntegrity:
    def test_covers_every_benchmarked_encoding(self) -> None:
        assert {p.name for p in list_scaling_profiles()} == set(available_encodings())

    def test_metadata_records_the_protocol(self) -> None:
        meta = scaling_metadata()
        assert meta["schema_version"] == "1.0"
        assert meta["n_encodings"] == 16
        assert meta["generated_by"] == "experiments/scaling_scan.py"
        protocol = meta["protocol"]
        assert protocol["published_range"] == pytest.approx(list(PUBLISHED_RANGE))
        assert len(protocol["ranges"]) >= 3
        assert protocol["datasets"]
        assert isinstance(protocol["seed"], int)

    def test_metadata_is_a_defensive_copy(self) -> None:
        scaling_metadata()["protocol"]["seed"] = -1
        assert scaling_metadata()["protocol"]["seed"] != -1

    def test_every_profile_is_well_formed(self) -> None:
        for profile in list_scaling_profiles():
            assert profile.points, profile.name
            assert profile.accuracy_spread >= 0.0
            low, high = profile.best_range
            assert low < high
            for point in profile.points:
                assert -1.0 <= point.mean_alignment <= 1.0
                assert 0.0 <= point.mean_accuracy <= 1.0
                assert point.mean_concentration_ratio >= 0.0
                assert point.n_datasets >= 1
                assert point.width == pytest.approx(point.high - point.low)

    def test_points_ordered_narrow_to_wide(self) -> None:
        for profile in list_scaling_profiles():
            widths = [p.width for p in profile.points]
            assert widths == sorted(widths), profile.name

    def test_best_range_maximises_alignment(self) -> None:
        for profile in list_scaling_profiles():
            best = max(p.mean_alignment for p in profile.points)
            chosen = profile.at_range(*profile.best_range)
            assert chosen is not None
            assert chosen.mean_alignment == pytest.approx(best), profile.name

    def test_published_range_is_present_for_every_encoding(self) -> None:
        for profile in list_scaling_profiles():
            assert profile.published is not None, profile.name


class TestTheContingency:
    """The reason the scan is shipped."""

    def test_correlation_reported_at_every_range(self) -> None:
        rows = expressibility_accuracy_correlation()
        assert len(rows) == len(scaling_metadata()["protocol"]["ranges"])
        assert sum(r["is_published_range"] for r in rows) == 1

    def test_published_range_reproduces_a_negative_association(self) -> None:
        """[0, 2*pi] is where the study measured H1, and the sign matches."""
        published = [
            r for r in expressibility_accuracy_correlation() if r["is_published_range"]
        ][0]
        assert published["spearman_rho_atlas_subset"] < 0.0
        assert published["p_value_atlas_subset"] < 0.05

    def test_a_narrower_range_reverses_the_sign(self) -> None:
        """The finding: the conclusion is contingent on the preprocessing."""
        rows = expressibility_accuracy_correlation()
        narrow = [r for r in rows if not r["is_published_range"]]
        positive = [
            r
            for r in narrow
            if r["spearman_rho_atlas_subset"] is not None
            and r["spearman_rho_atlas_subset"] > 0.0
            and r["p_value_atlas_subset"] < 0.05
        ]
        assert positive, "expected at least one range with a significant +ve rho"

    def test_atlas_subset_matches_the_published_encoding_set(self) -> None:
        """The published analysis excludes encodings with no expressibility."""
        from encoding_atlas.atlas import list_profiles

        expected = sum(
            1 for p in list_profiles() if p.metrics.get("expressibility") is not None
        )
        for row in expressibility_accuracy_correlation():
            assert row["n_encodings_atlas_subset"] == expected

    def test_correlation_is_a_defensive_copy(self) -> None:
        expressibility_accuracy_correlation()[0]["spearman_rho"] = 999.0
        assert expressibility_accuracy_correlation()[0]["spearman_rho"] != 999.0


class TestScalingSensitivity:
    def test_entangling_maps_are_the_most_sensitive(self) -> None:
        """The circuits that scramble fastest lose the most to a wide range."""
        sensitive = {p.name for p in scaling_sensitive_encodings(threshold=0.2)}
        assert "iqp" in sensitive
        assert "angle" not in sensitive

    def test_ordered_most_sensitive_first(self) -> None:
        spreads = [p.accuracy_spread for p in scaling_sensitive_encodings()]
        assert spreads == sorted(spreads, reverse=True)

    def test_threshold_filters(self) -> None:
        assert len(scaling_sensitive_encodings(threshold=0.0)) >= len(
            scaling_sensitive_encodings(threshold=0.3)
        )
        assert scaling_sensitive_encodings(threshold=10.0) == []

    def test_negative_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            scaling_sensitive_encodings(threshold=-0.1)

    def test_published_range_is_rarely_the_best(self) -> None:
        """If [0, 2*pi] were optimal the scan would be uninteresting."""
        best_is_published = [
            p
            for p in list_scaling_profiles()
            if p.best_range == pytest.approx(PUBLISHED_RANGE)
        ]
        assert len(best_is_published) < len(list_scaling_profiles()) / 2


class TestLookup:
    @pytest.mark.parametrize(
        "alias", ["iqp", "IQP", "  iqp  ", "IQPEncoding", "iqpencoding"]
    )
    def test_aliases_resolve(self, alias: str) -> None:
        assert get_scaling_profile(alias).name == "iqp"

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("qaoa_encoding", "qaoa"),
            ("hamiltonian_encoding", "hamiltonian"),
            ("trainable_encoding", "trainable"),
        ],
    )
    def test_dataset_aliases_normalise(self, alias: str, canonical: str) -> None:
        assert get_scaling_profile(alias).name == canonical

    def test_unknown_name_lists_alternatives(self) -> None:
        with pytest.raises(KeyError, match="Scanned encodings"):
            get_scaling_profile("does_not_exist")

    def test_profiles_are_immutable(self) -> None:
        profile = get_scaling_profile("angle")
        assert isinstance(profile, ScalingProfile)
        with pytest.raises(FrozenInstanceError):
            profile.accuracy_spread = 0.0  # type: ignore[misc]
        with pytest.raises(TypeError):
            profile.params["reps"] = 99  # type: ignore[index]

    def test_points_are_immutable(self) -> None:
        point = get_scaling_profile("angle").points[0]
        assert isinstance(point, ScalingPoint)
        with pytest.raises(FrozenInstanceError):
            point.mean_accuracy = 0.0  # type: ignore[misc]

    def test_at_range_exact_match_only(self) -> None:
        profile = get_scaling_profile("iqp")
        assert profile.at_range(*PUBLISHED_RANGE) is not None
        assert profile.at_range(0.0, 12345.0) is None

    def test_cached_object_is_shared(self) -> None:
        assert get_scaling_profile("angle") is get_scaling_profile("angle")


@pytest.mark.slow
class TestReproducibility:
    """The committed dataset must be exactly regenerable from the code."""

    def test_full_scan_is_byte_identical_to_the_committed_file(self) -> None:
        import json
        from pathlib import Path

        from experiments.scaling_scan import DEFAULT_OUTPUT, build_dataset

        fresh = json.dumps(build_dataset(verbose=False), indent=2, sort_keys=True)
        committed = Path(DEFAULT_OUTPUT).read_text(encoding="utf-8")
        assert fresh + "\n" == committed, (
            "src/encoding_atlas/atlas/data/scaling_sensitivity.json is stale. "
            "Regenerate with: python -m experiments.scaling_scan"
        )
