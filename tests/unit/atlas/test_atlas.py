"""Tests for the bundled empirical atlas API.

Covers:
- ``EncodingProfile`` structure, immutability, and the ``metric()`` accessor.
- ``get_encoding_profile`` lookup by canonical name, dataset alias, and display
  name (case-insensitive), plus error handling.
- ``list_profiles`` / ``available_encodings`` completeness and rank ordering.
- ``rank_encodings`` ordering, direction defaults, limiting, ``None`` filtering,
  and input validation.
- ``pareto_front``, ``hypothesis_verdicts``, ``atlas_metadata`` content and the
  read-only / deep-copy guarantees.
- Cross-system consistency: the atlas vocabulary matches the recommender's
  canonical names and the encoding registry.

Run with: pytest tests/unit/atlas/test_atlas.py -v
"""

from __future__ import annotations

from types import MappingProxyType

import pytest

from encoding_atlas.atlas import (
    EncodingProfile,
    atlas_metadata,
    available_encodings,
    get_encoding_profile,
    hypothesis_verdicts,
    list_metrics,
    list_profiles,
    pareto_front,
    rank_encodings,
)

# Ground-truth facts that the bundled dataset must encode. These are taken
# directly from the consolidated benchmark results and must remain stable.
EXPECTED_N_ENCODINGS = 16
EXPECTED_CANONICAL_NAMES = frozenset(
    {
        "angle",
        "basis",
        "higher_order_angle",
        "iqp",
        "zz_feature_map",
        "pauli_feature_map",
        "data_reuploading",
        "hardware_efficient",
        "amplitude",
        "qaoa",
        "hamiltonian",
        "trainable",
        "symmetry_inspired",
        "so2_equivariant",
        "cyclic_equivariant",
        "swap_equivariant",
    }
)
EXPECTED_PARETO = frozenset(
    {"angle", "basis", "higher_order_angle", "swap_equivariant"}
)


# =========================================================================
# EncodingProfile structure
# =========================================================================


class TestEncodingProfile:
    """Structure, typing, immutability, and accessors of EncodingProfile."""

    def test_fields_have_expected_types(self) -> None:
        profile = get_encoding_profile("angle")
        assert isinstance(profile, EncodingProfile)
        assert isinstance(profile.name, str)
        assert isinstance(profile.display_name, str)
        assert isinstance(profile.family, str)
        assert isinstance(profile.rank, int)
        assert isinstance(profile.score, float)
        assert isinstance(profile.is_pareto, bool)
        assert isinstance(profile.is_simulable, bool)

    def test_metrics_is_read_only_mapping(self) -> None:
        profile = get_encoding_profile("angle")
        assert isinstance(profile.metrics, MappingProxyType)
        with pytest.raises(TypeError):
            profile.metrics["depth"] = 0  # type: ignore[index]

    def test_profile_is_frozen(self) -> None:
        profile = get_encoding_profile("angle")
        with pytest.raises((AttributeError, TypeError)):
            profile.rank = 99  # type: ignore[misc]

    def test_metric_returns_value(self) -> None:
        angle = get_encoding_profile("angle")
        assert angle.metric("kernel_accuracy") == pytest.approx(0.9581781795935266)
        assert angle.metric("depth") == 2

    def test_metric_default_for_missing_key(self) -> None:
        angle = get_encoding_profile("angle")
        assert angle.metric("nonexistent_metric") is None
        assert angle.metric("nonexistent_metric", default=-1.0) == -1.0

    def test_metric_default_for_none_value(self) -> None:
        # angle has no entanglement_capability (non-entangling) -> None in data.
        angle = get_encoding_profile("angle")
        assert angle.metrics.get("entanglement_capability") is None
        assert angle.metric("entanglement_capability", default=0.0) == 0.0


# =========================================================================
# get_encoding_profile lookup
# =========================================================================


class TestGetEncodingProfile:
    """Lookup by canonical name, alias, display name, and error handling."""

    def test_canonical_name(self) -> None:
        assert get_encoding_profile("iqp").name == "iqp"

    @pytest.mark.parametrize(
        "alias,canonical",
        [
            ("qaoa_encoding", "qaoa"),
            ("hamiltonian_encoding", "hamiltonian"),
            ("trainable_encoding", "trainable"),
        ],
    )
    def test_dataset_alias_normalised(self, alias: str, canonical: str) -> None:
        assert get_encoding_profile(alias).name == canonical

    @pytest.mark.parametrize(
        "display,canonical",
        [
            ("AngleEncoding", "angle"),
            ("QAOAEncoding", "qaoa"),
            ("IQPEncoding", "iqp"),
        ],
    )
    def test_display_name_lookup(self, display: str, canonical: str) -> None:
        assert get_encoding_profile(display).name == canonical

    def test_case_insensitive_and_trimmed(self) -> None:
        assert get_encoding_profile("  IQP  ").name == "iqp"
        assert get_encoding_profile("AnGlE").name == "angle"

    def test_unknown_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="Unknown encoding"):
            get_encoding_profile("not_a_real_encoding")

    def test_every_canonical_name_resolvable(self) -> None:
        for name in EXPECTED_CANONICAL_NAMES:
            assert get_encoding_profile(name).name == name


# =========================================================================
# list_profiles / available_encodings
# =========================================================================


class TestListProfiles:
    """Completeness and ordering of the profile collection."""

    def test_sixteen_profiles(self) -> None:
        assert len(list_profiles()) == EXPECTED_N_ENCODINGS

    def test_ranked_best_first_and_contiguous(self) -> None:
        ranks = [p.rank for p in list_profiles()]
        assert ranks == list(range(1, EXPECTED_N_ENCODINGS + 1))

    def test_rank_one_is_angle(self) -> None:
        assert list_profiles()[0].name == "angle"

    def test_names_match_expected_canonical_set(self) -> None:
        assert {p.name for p in list_profiles()} == EXPECTED_CANONICAL_NAMES

    def test_available_encodings_sorted_and_complete(self) -> None:
        names = available_encodings()
        assert names == sorted(EXPECTED_CANONICAL_NAMES)

    def test_returned_list_is_a_copy(self) -> None:
        first = list_profiles()
        first.clear()
        assert len(list_profiles()) == EXPECTED_N_ENCODINGS


# =========================================================================
# list_metrics
# =========================================================================


class TestListMetrics:
    """Scalar metric key inventory."""

    def test_expected_scalar_metrics(self) -> None:
        assert list_metrics() == [
            "depth",
            "expressibility",
            "entanglement_capability",
            "trainability_estimate",
            "noise_resilience",
            "vqc_accuracy",
            "kernel_accuracy",
            "kernel_target_alignment",
        ]

    def test_ci_keys_excluded_from_scalar_metrics(self) -> None:
        # CI pairs exist in the data but are not single sortable scalars.
        assert "vqc_ci" not in list_metrics()
        assert "kernel_ci" not in list_metrics()
        assert get_encoding_profile("angle").metrics.get("vqc_ci") is not None


# =========================================================================
# rank_encodings
# =========================================================================


class TestRankEncodings:
    """Ordering, direction defaults, limiting, and validation."""

    def test_default_by_score_descending(self) -> None:
        ranked = rank_encodings()
        scores = [p.score for p in ranked]
        assert scores == sorted(scores, reverse=True)
        assert ranked[0].name == "angle"

    def test_by_rank_ascending(self) -> None:
        ranked = rank_encodings(by="rank")
        assert [p.rank for p in ranked] == list(range(1, EXPECTED_N_ENCODINGS + 1))

    def test_metric_default_descending(self) -> None:
        ranked = rank_encodings(by="kernel_accuracy")
        values = [p.metric("kernel_accuracy") for p in ranked]
        assert values == sorted(values, reverse=True)
        assert ranked[0].name == "angle"

    def test_depth_defaults_ascending(self) -> None:
        ranked = rank_encodings(by="depth")
        depths = [p.metric("depth") for p in ranked]
        assert depths == sorted(depths)
        assert depths[0] == 1  # shallowest circuits first

    def test_ascending_override(self) -> None:
        asc = rank_encodings(by="kernel_accuracy", ascending=True)
        values = [p.metric("kernel_accuracy") for p in asc]
        assert values == sorted(values)

    def test_limit(self) -> None:
        top3 = rank_encodings(by="kernel_accuracy", limit=3)
        assert len(top3) == 3
        assert [p.name for p in top3] == ["angle", "cyclic_equivariant", "qaoa"]

    def test_limit_zero(self) -> None:
        assert rank_encodings(limit=0) == []

    def test_none_values_filtered_out(self) -> None:
        # expressibility is undefined (None) for amplitude and basis.
        ranked = rank_encodings(by="expressibility")
        names = {p.name for p in ranked}
        assert len(ranked) == EXPECTED_N_ENCODINGS - 2
        assert "amplitude" not in names
        assert "basis" not in names

    def test_invalid_key_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot rank by"):
            rank_encodings(by="not_a_metric")

    def test_negative_limit_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            rank_encodings(limit=-1)


# =========================================================================
# pareto_front
# =========================================================================


class TestParetoFront:
    """Pareto-optimal set content and properties."""

    def test_pareto_membership(self) -> None:
        assert {p.name for p in pareto_front()} == EXPECTED_PARETO

    def test_all_marked_pareto(self) -> None:
        assert all(p.is_pareto for p in pareto_front())

    def test_pareto_is_rank_ordered(self) -> None:
        ranks = [p.rank for p in pareto_front()]
        assert ranks == sorted(ranks)

    def test_non_pareto_excluded(self) -> None:
        assert "iqp" not in {p.name for p in pareto_front()}


# =========================================================================
# hypothesis_verdicts
# =========================================================================


class TestHypothesisVerdicts:
    """Pre-registered hypothesis outcomes."""

    def test_seven_hypotheses(self) -> None:
        verdicts = hypothesis_verdicts()
        assert set(verdicts) == {"H1", "H2", "H3", "H4", "H5", "H6", "H7"}

    def test_h1_refuted(self) -> None:
        # The expressibility-accuracy hypothesis is refuted (rho < 0).
        assert hypothesis_verdicts()["H1"]["verdict"] == "refuted"

    def test_each_verdict_has_required_fields(self) -> None:
        for name, verdict in hypothesis_verdicts().items():
            assert "verdict" in verdict, name
            assert "confidence" in verdict, name
            assert "evidence" in verdict, name

    def test_returns_independent_deep_copy(self) -> None:
        first = hypothesis_verdicts()
        first["H1"]["verdict"] = "MUTATED"
        assert hypothesis_verdicts()["H1"]["verdict"] == "refuted"


# =========================================================================
# atlas_metadata
# =========================================================================


class TestAtlasMetadata:
    """Provenance and summary metadata."""

    def test_core_fields(self) -> None:
        md = atlas_metadata()
        assert md["schema_version"] == "1.0"
        assert md["n_encodings"] == EXPECTED_N_ENCODINGS
        assert md["n_pareto_optimal"] == len(EXPECTED_PARETO)

    def test_objective_names(self) -> None:
        assert atlas_metadata()["objective_names"] == [
            "accuracy",
            "inv_depth",
            "trainability",
            "noise_resilience",
        ]

    def test_source_mentions_experiments(self) -> None:
        assert "experiments/" in atlas_metadata()["source"]

    def test_returns_independent_deep_copy(self) -> None:
        md = atlas_metadata()
        md["stage_counts"].clear()
        assert atlas_metadata()["stage_counts"]  # still populated


# =========================================================================
# Cross-system consistency
# =========================================================================


class TestCrossSystemConsistency:
    """The atlas vocabulary must align with the rest of the package."""

    def test_canonical_names_match_recommender_rules(self) -> None:
        from encoding_atlas.guide.rules import ENCODING_RULES

        assert {p.name for p in list_profiles()} == set(ENCODING_RULES.keys())

    def test_canonical_names_registered_in_registry(self) -> None:
        from encoding_atlas.core.registry import list_encodings

        registry_names = set(list_encodings())
        for name in available_encodings():
            assert name in registry_names, f"{name} not in encoding registry"

    def test_atlas_exposed_on_package(self) -> None:
        import encoding_atlas

        assert hasattr(encoding_atlas, "atlas")
        assert encoding_atlas.atlas.get_encoding_profile("angle").rank == 1


# =========================================================================
# Kernel-target alignment column
# =========================================================================


class TestKernelTargetAlignmentColumn:
    """The benchmark's *validated* predictor, carried into the shipped atlas.

    Expressibility (which the study refutes) was already queryable; alignment
    (which it validates) was measured but dropped during consolidation. These
    tests pin both the presence of the column and the claim it encodes.
    """

    def test_present_for_every_encoding(self) -> None:
        for profile in list_profiles():
            value = profile.metrics["kernel_target_alignment"]
            assert value is not None, profile.name
            assert -1.0 <= value <= 1.0

    def test_is_rankable(self) -> None:
        ranked = rank_encodings(by="kernel_target_alignment")
        assert len(ranked) == EXPECTED_N_ENCODINGS
        values = [p.metric("kernel_target_alignment") for p in ranked]
        assert values == sorted(values, reverse=True)  # higher is better

    def test_predicts_kernel_accuracy_better_than_expressibility(self) -> None:
        """The paper's headline pair of results, reproduced from the atlas.

        Alignment tracks accuracy; expressibility is if anything negatively
        associated with it. This is the reason the column exists.
        """
        from scipy.stats import spearmanr

        profiles = list_profiles()
        alignment = [p.metric("kernel_target_alignment") for p in profiles]
        accuracy = [p.metric("kernel_accuracy") for p in profiles]
        rho_alignment, p_alignment = spearmanr(alignment, accuracy)

        with_expr = [p for p in profiles if p.metrics.get("expressibility") is not None]
        rho_expr, _ = spearmanr(
            [p.metric("expressibility") for p in with_expr],
            [p.metric("kernel_accuracy") for p in with_expr],
        )

        assert rho_alignment > 0.85 and p_alignment < 0.001
        assert rho_expr < 0.0
        assert rho_alignment > abs(rho_expr)

    def test_matches_the_raw_stage_measurement(self) -> None:
        """The column is an aggregate of measured values, not a re-derivation.

        Recomputes the mean over (configuration, dataset) pairs straight from
        the Stage 6b output and requires an exact match — the same rule the
        ``kernel_accuracy`` column uses.
        """
        import json
        import pathlib

        raw_path = pathlib.Path("experiments/results/raw/stage6b_kernel/summary.json")
        if not raw_path.exists():  # pragma: no cover - raw data not packaged
            pytest.skip("raw stage6b output not available")

        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for entry in raw["results"]:
            if entry.get("status") != "success":
                continue
            name = entry["encoding_name"]
            for ds in entry["result"]["datasets"].values():
                if ds.get("status") != "success":
                    continue
                value = ds.get("centered_kernel_target_alignment")
                if value is None:
                    continue
                sums[name] = sums.get(name, 0.0) + float(value)
                counts[name] = counts.get(name, 0) + 1

        for name, total in sums.items():
            profile = get_encoding_profile(name)
            assert profile.metric("kernel_target_alignment") == pytest.approx(
                total / counts[name]
            )

    def test_orders_the_haar_floor_encodings_last(self) -> None:
        """Encodings whose kernels concentrate should score lowest on alignment."""
        from encoding_atlas.atlas import concentrated_encodings

        ranked = [p.name for p in rank_encodings(by="kernel_target_alignment")]
        for profile in concentrated_encodings():
            assert ranked.index(profile.name) >= 10, profile.name
