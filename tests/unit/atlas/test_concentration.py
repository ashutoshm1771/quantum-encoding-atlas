"""Tests for the bundled kernel-concentration scan and its query API.

Beyond the usual API-shape checks, this module pins the scientific claim the
dataset exists to support: the encodings whose fidelity kernel reaches the Haar
floor are exactly the high-expressibility feature maps the benchmark ranks
worst. That correspondence is the mechanism behind the atlas's refuted H1
("expressibility predicts accuracy"), so it is asserted directly.
"""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest

from encoding_atlas.atlas import (
    ConcentrationPoint,
    ConcentrationProfile,
    available_encodings,
    concentrated_encodings,
    concentration_metadata,
    get_concentration_profile,
    get_encoding_profile,
    list_concentration_profiles,
)


class TestDatasetIntegrity:
    def test_covers_every_benchmarked_encoding(self) -> None:
        scanned = {p.name for p in list_concentration_profiles()}
        assert scanned == set(available_encodings())

    def test_metadata_records_the_protocol(self) -> None:
        meta = concentration_metadata()
        assert meta["schema_version"] == "1.0"
        assert meta["n_encodings"] == 16
        assert meta["generated_by"] == "experiments/concentration_scan.py"
        protocol = meta["protocol"]
        assert protocol["feature_counts"] == [2, 4, 6, 8]
        assert protocol["n_samples"] >= 100
        assert protocol["sampling"] == "uniform"
        assert protocol["input_range"] == pytest.approx([0.0, 2 * math.pi])
        assert isinstance(protocol["seed"], int)

    def test_metadata_is_a_defensive_copy(self) -> None:
        concentration_metadata()["protocol"]["seed"] = -1
        assert concentration_metadata()["protocol"]["seed"] != -1

    def test_every_profile_has_measured_points(self) -> None:
        for profile in list_concentration_profiles():
            assert profile.points, f"{profile.name} has no measured points"
            for point in profile.points:
                assert point.n_qubits >= 1
                assert point.concentration_ratio > 0.0
                assert 0.0 <= point.offdiagonal_mean <= 1.0
                assert point.offdiagonal_variance >= 0.0
                assert point.shots_per_entry > 0.0

    def test_points_are_ordered_by_width(self) -> None:
        for profile in list_concentration_profiles():
            widths = [p.n_qubits for p in profile.points]
            assert widths == sorted(widths)

    def test_is_concentrated_flag_matches_the_ratio(self) -> None:
        for profile in list_concentration_profiles():
            for point in profile.points:
                assert point.is_concentrated == (point.concentration_ratio < 2.0)

    def test_fits_are_present_where_enough_widths_were_measured(self) -> None:
        for profile in list_concentration_profiles():
            if len(profile.points) < 2:
                continue
            assert profile.decay_rate is not None and profile.decay_rate > 0.0
            assert profile.r_squared is not None and 0.0 <= profile.r_squared <= 1.0
            assert profile.haar_normalized_slope is not None

    def test_skipped_widths_are_explained(self) -> None:
        so2 = get_concentration_profile("so2_equivariant")
        assert len(so2.points) == 1  # requires exactly two features
        assert set(so2.skipped) == {"4", "6", "8"}
        assert all("n_features" in reason for reason in so2.skipped.values())


class TestPhysicalBounds:
    def test_variance_decay_never_beats_the_haar_rate(self) -> None:
        """Haar scrambling decays the kernel variance by 4x per qubit.

        Nothing can concentrate faster than the maximally scrambling ensemble,
        so a fitted rate materially above 4 would mean the scan is wrong.
        """
        for profile in list_concentration_profiles():
            if profile.decay_rate is None:
                continue
            assert profile.decay_rate <= 4.5, profile.name

    def test_mean_decay_never_beats_the_haar_rate(self) -> None:
        # The Haar mean 1/d halves with every added qubit.
        for profile in list_concentration_profiles():
            if profile.mean_decay_rate is None:
                continue
            assert profile.mean_decay_rate <= 2.3, profile.name

    def test_concentrated_encodings_decay_near_the_haar_rate(self) -> None:
        for profile in concentrated_encodings():
            assert profile.decay_rate is not None
            assert profile.decay_rate > 3.0, profile.name


class TestTheMechanism:
    """The scan's reason for existing, asserted against the benchmark."""

    def test_concentrated_set_is_the_known_four(self) -> None:
        assert sorted(p.name for p in concentrated_encodings()) == [
            "hamiltonian",
            "iqp",
            "pauli_feature_map",
            "zz_feature_map",
        ]

    def test_concentrated_encodings_are_the_worst_ranked(self) -> None:
        """All four sit in the bottom third of the 16-encoding ranking."""
        for profile in concentrated_encodings():
            assert get_encoding_profile(profile.name).rank >= 12, profile.name

    def test_concentrated_encodings_are_the_most_expressible(self) -> None:
        """Haar-likeness is what expressibility measures; concentration is its
        consequence for the kernel. The encodings at the floor are exactly the
        ones scoring ~1.0 on expressibility."""
        for profile in concentrated_encodings():
            expressibility = get_encoding_profile(profile.name).metric("expressibility")
            assert expressibility is not None
            assert expressibility > 0.99, profile.name

    def test_top_ranked_encodings_are_not_concentrated(self) -> None:
        concentrated = {p.name for p in concentrated_encodings()}
        for profile in list_concentration_profiles():
            if get_encoding_profile(profile.name).rank <= 4:
                assert profile.name not in concentrated

    def test_concentrated_encodings_cost_more_shots(self) -> None:
        """At the widest measured circuit, the floor is far more expensive."""
        concentrated = {p.name for p in concentrated_encodings()}
        at_floor = [
            p.points[-1].shots_per_entry
            for p in list_concentration_profiles()
            if p.name in concentrated
        ]
        angle = get_concentration_profile("angle").points[-1].shots_per_entry
        assert min(at_floor) > angle


class TestLookup:
    @pytest.mark.parametrize(
        "alias", ["iqp", "IQP", "  iqp  ", "IQPEncoding", "iqpencoding"]
    )
    def test_aliases_resolve(self, alias: str) -> None:
        assert get_concentration_profile(alias).name == "iqp"

    @pytest.mark.parametrize(
        ("alias", "canonical"),
        [
            ("qaoa_encoding", "qaoa"),
            ("hamiltonian_encoding", "hamiltonian"),
            ("trainable_encoding", "trainable"),
        ],
    )
    def test_dataset_aliases_normalise(self, alias: str, canonical: str) -> None:
        assert get_concentration_profile(alias).name == canonical

    def test_unknown_name_lists_alternatives(self) -> None:
        with pytest.raises(KeyError, match="Scanned encodings"):
            get_concentration_profile("does_not_exist")

    def test_profiles_are_immutable(self) -> None:
        profile = get_concentration_profile("angle")
        assert isinstance(profile, ConcentrationProfile)
        with pytest.raises(FrozenInstanceError):
            profile.horizon = 3  # type: ignore[misc]
        # ``params`` and ``skipped`` are mapping proxies, so the shared cached
        # dataset cannot be mutated through a returned profile either.
        with pytest.raises(TypeError):
            profile.params["reps"] = 99  # type: ignore[index]
        with pytest.raises(TypeError):
            profile.skipped["4"] = "nope"  # type: ignore[index]

    def test_repeated_lookups_share_the_cached_object(self) -> None:
        assert get_concentration_profile("angle") is get_concentration_profile("angle")


class TestAtFeatures:
    def test_exact_match_is_returned(self) -> None:
        point = get_concentration_profile("iqp").at_features(6)
        assert isinstance(point, ConcentrationPoint)
        assert point.n_features == 6

    def test_snaps_to_the_nearest_measured_width(self) -> None:
        profile = get_concentration_profile("iqp")
        assert profile.at_features(5).n_features == 6  # tie breaks wider
        assert profile.at_features(7).n_features == 8
        assert profile.at_features(1).n_features == 2
        assert profile.at_features(100).n_features == 8

    def test_single_point_profile_always_returns_it(self) -> None:
        profile = get_concentration_profile("so2_equivariant")
        assert profile.at_features(50).n_features == 2

    @pytest.mark.parametrize("bad", [0, -1, 2.5, True])
    def test_validates_feature_count(self, bad: object) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            get_concentration_profile("angle").at_features(bad)  # type: ignore[arg-type]

    def test_is_concentrated_at_matches_the_point(self) -> None:
        assert get_concentration_profile("iqp").is_concentrated_at(4)
        assert not get_concentration_profile("angle").is_concentrated_at(8)

    def test_threshold_is_honoured(self) -> None:
        angle = get_concentration_profile("angle")
        assert angle.is_concentrated_at(8, threshold=1e6)
        assert not angle.is_concentrated_at(8, threshold=1e-6)

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be positive"):
            get_concentration_profile("angle").is_concentrated_at(4, threshold=0.0)


class TestConcentratedEncodings:
    def test_ordered_by_name(self) -> None:
        names = [p.name for p in concentrated_encodings()]
        assert names == sorted(names)

    def test_generous_threshold_admits_everything(self) -> None:
        assert len(concentrated_encodings(threshold=1e9)) == 16

    def test_strict_threshold_admits_nothing(self) -> None:
        assert concentrated_encodings(threshold=1e-9) == []

    def test_invalid_threshold_raises(self) -> None:
        with pytest.raises(ValueError, match="threshold must be positive"):
            concentrated_encodings(threshold=-1.0)


@pytest.mark.slow
class TestReproducibility:
    """The committed dataset must be exactly regenerable from the code.

    Marked slow: re-runs the whole scan (~20 s). Covering every encoding, not a
    sample, is deliberate — the failure this guards against is a single
    encoding introducing unseeded randomness (``TrainableEncoding`` draws its
    variational parameters at construction time), which a spot check would
    miss.
    """

    def test_full_scan_is_byte_identical_to_the_committed_file(self) -> None:
        import json
        from pathlib import Path

        from experiments.concentration_scan import DEFAULT_OUTPUT, build_dataset

        fresh = json.dumps(build_dataset(verbose=False), indent=2, sort_keys=True)
        committed = Path(DEFAULT_OUTPUT).read_text(encoding="utf-8")
        assert fresh + "\n" == committed, (
            "src/encoding_atlas/atlas/data/concentration.json is stale. "
            "Regenerate with: python -m experiments.concentration_scan"
        )

    def test_scan_is_deterministic_across_runs(self) -> None:
        """Two independent runs must agree exactly, not merely closely."""
        from experiments.concentration_scan import build_dataset

        first = build_dataset(verbose=False)["encodings"]
        second = build_dataset(verbose=False)["encodings"]
        assert first == second

    def test_query_api_reflects_the_committed_numbers(self) -> None:
        from experiments.concentration_scan import ENCODING_PARAMS, build_dataset

        fresh = build_dataset(verbose=False)["encodings"]
        for key in ENCODING_PARAMS:
            record = fresh[key]
            profile = get_concentration_profile(key)
            assert profile.display_name == record["display_name"]
            assert profile.horizon == record["concentration_horizon"]
            assert len(profile.points) == len(record["points"])
            for point, raw in zip(profile.points, record["points"]):
                assert point.n_qubits == raw["n_qubits"]
                assert point.concentration_ratio == pytest.approx(
                    raw["concentration_ratio"]
                )
