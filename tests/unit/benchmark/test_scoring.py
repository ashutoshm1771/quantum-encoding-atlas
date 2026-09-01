"""Tests for weighted scoring with an explicit missing-metric policy.

The defect these guard against is specific and was live: the atlas's Monte
Carlo weight sweep filled a missing entanglement capability with the *column
median* (0.605) for the three encodings whose true capability is exactly 0,
because they contain no entangling gates. That single substitution produced the
suite's only robustness claim — ``robustly_top3 == ['angle']`` at 97.8% — which
disappears entirely once the structural zero is used instead.

``TestStructuralZeroVsImputation`` pins the distinction on a synthetic case, and
``TestAgainstTheRealAtlas`` pins the corrected numbers where the raw pipeline
output is available.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pathlib

import numpy as np
import pytest

from encoding_atlas.benchmark.scoring import (
    MetricReading,
    MetricSpec,
    ScoringResult,
    describe_unusable,
    measured,
    not_measured,
    readings_from_values,
    score_encodings,
    structurally_zero,
    weight_sensitivity,
)

SPECS = [MetricSpec("acc", 0.5), MetricSpec("ent", 0.5)]

#: "plain" has no entangling gates (a true zero); "partial" was never measured.
FIXTURE = {
    "plain": {"acc": measured(0.90), "ent": structurally_zero("no entangling gates")},
    "mid": {"acc": measured(0.80), "ent": measured(0.50)},
    "deep": {"acc": measured(0.70), "ent": measured(1.00)},
    "partial": {"acc": measured(0.95), "ent": not_measured("never run")},
}

RAW_TRADEOFF = pathlib.Path("experiments/results/raw/stage7_tradeoff/summary.json")
STAGE_CONFIG = pathlib.Path("experiments/configs/stage7_tradeoff.json")


class TestMetricReading:
    def test_measured_carries_its_value(self) -> None:
        assert measured(0.75).resolved == pytest.approx(0.75)

    def test_structural_zero_resolves_to_zero(self) -> None:
        reading = structurally_zero("no entangling gates")
        assert reading.resolved == 0.0
        assert reading.is_usable

    def test_not_measured_resolves_to_nothing(self) -> None:
        reading = not_measured("stage never run")
        assert reading.resolved is None
        assert not reading.is_usable

    def test_the_two_absences_are_distinguishable(self) -> None:
        """Both have value None; only the availability separates them."""
        zero, unknown = structurally_zero(), not_measured()
        assert zero.value is unknown.value is None
        assert zero.availability != unknown.availability
        assert zero.resolved != unknown.resolved

    def test_reason_is_retained(self) -> None:
        assert structurally_zero("no entangling gates").reason == "no entangling gates"

    @pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
    def test_measured_requires_a_finite_value(self, bad: float | None) -> None:
        with pytest.raises(ValueError, match="finite value"):
            MetricReading(value=bad, availability="measured")

    @pytest.mark.parametrize("availability", ["structurally_zero", "not_measured"])
    def test_absent_readings_must_not_carry_a_value(self, availability: str) -> None:
        with pytest.raises(ValueError, match="must not carry a value"):
            MetricReading(value=0.0, availability=availability)  # type: ignore[arg-type]

    def test_unknown_availability_raises(self) -> None:
        with pytest.raises(ValueError, match="availability must be"):
            MetricReading(value=None, availability="probably")  # type: ignore[arg-type]

    def test_is_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            measured(0.5).value = 0.9  # type: ignore[misc]


class TestMetricSpec:
    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_weight_must_be_finite_and_positive(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite and positive"):
            MetricSpec("acc", bad)

    def test_defaults_to_higher_is_better(self) -> None:
        assert MetricSpec("acc", 1.0).higher_is_better


class TestStructuralZeroVsImputation:
    """The defect this module exists to prevent recurring."""

    def test_structural_zero_sets_the_normalisation_floor(self) -> None:
        """A true zero is data: it widens the range every rival is scaled by."""
        result = score_encodings(FIXTURE, SPECS)
        assert result.normalization["ent"] == (0.0, 1.0)

    def test_dropping_the_zero_would_shift_every_rival(self) -> None:
        """Without it the floor is 0.5 and 'mid' normalises to 0 instead of 0.5."""
        without = {k: v for k, v in FIXTURE.items() if k != "plain"}
        assert score_encodings(without, SPECS).normalization["ent"] == (0.5, 1.0)

    def test_structural_zero_is_not_the_column_median(self) -> None:
        """Median imputation is what manufactured the atlas robustness claim.

        The median of the measured entanglement values here is 0.75; the true
        value for a non-entangling encoding is 0. Pin that we use the latter.
        """
        measured_values = [0.50, 1.00]
        assert float(np.median(measured_values)) == pytest.approx(0.75)
        assert structurally_zero().resolved == 0.0

    def test_median_imputation_would_change_the_ranking(self) -> None:
        imputed = dict(FIXTURE)
        imputed["plain"] = {"acc": measured(0.90), "ent": measured(0.75)}
        assert score_encodings(FIXTURE, SPECS).ranks["plain"] == 4
        assert score_encodings(imputed, SPECS).ranks["plain"] == 2

    def test_a_structural_zero_is_never_reported_as_unusable(self) -> None:
        result = score_encodings(FIXTURE, SPECS)
        assert "plain" not in result.unusable
        assert result.structural_zeros["plain"] == ("ent",)
        assert result.effective_weight["plain"] == pytest.approx(1.0)


class TestScoreEncodings:
    def test_ranks_and_scores_are_pinned(self) -> None:
        result = score_encodings(FIXTURE, SPECS)
        assert result.ranks == {"partial": 1, "deep": 2, "mid": 3, "plain": 4}
        assert result.scores["deep"] == pytest.approx(0.50)
        assert result.scores["mid"] == pytest.approx(0.45)
        assert result.scores["plain"] == pytest.approx(0.40)

    def test_partial_coverage_is_reported_not_absorbed(self) -> None:
        """'partial' wins on half the objective; the score alone hides that."""
        result = score_encodings(FIXTURE, SPECS)
        assert result.scores["partial"] == pytest.approx(1.0)
        assert result.ranks["partial"] == 1
        assert result.effective_weight["partial"] == pytest.approx(0.5)
        assert result.unusable["partial"] == ("ent",)
        assert "partial" in result.summary()

    def test_fully_scored_excludes_the_partial_encoding(self) -> None:
        assert score_encodings(FIXTURE, SPECS).fully_scored == ("deep", "mid", "plain")

    def test_exclude_policy_drops_and_names(self) -> None:
        result = score_encodings(FIXTURE, SPECS, on_unusable="exclude")
        assert "partial" not in result.scores
        assert result.excluded == {"partial": ("ent",)}
        assert set(result.scores) == {"plain", "mid", "deep"}

    def test_excluding_recomputes_the_normalisation(self) -> None:
        """Dropping the top scorer narrows the range the rest are scaled by."""
        full = score_encodings(FIXTURE, SPECS)
        pruned = score_encodings(FIXTURE, SPECS, on_unusable="exclude")
        assert full.normalization["acc"] == (0.7, 0.95)
        assert pruned.normalization["acc"] == (0.7, 0.90)

    def test_mathematically_tied_scores_order_by_name(self) -> None:
        """Weighted sums accumulate in different orders; an ULP must not rank.

        With 'partial' excluded these three are exactly balanced, and 'mid'
        lands one ULP high purely from summation order.
        """
        result = score_encodings(FIXTURE, SPECS, on_unusable="exclude")
        assert list(result.scores) == ["deep", "mid", "plain"]
        assert all(v == pytest.approx(0.5) for v in result.scores.values())

    def test_lower_is_better_inverts_an_axis(self) -> None:
        specs = [MetricSpec("acc", 1.0, higher_is_better=False)]
        readings = {"a": {"acc": measured(0.9)}, "b": {"acc": measured(0.1)}}
        assert score_encodings(readings, specs).best == "b"

    def test_constant_metric_scores_neutrally(self) -> None:
        specs = [MetricSpec("acc", 1.0)]
        readings = {"a": {"acc": measured(0.5)}, "b": {"acc": measured(0.5)}}
        result = score_encodings(readings, specs)
        assert set(result.scores.values()) == {0.5}

    def test_missing_key_counts_as_not_measured(self) -> None:
        readings = {
            "a": {"acc": measured(0.9)},
            "b": {"acc": measured(0.5), "ent": measured(1.0)},
        }
        result = score_encodings(readings, SPECS)
        assert result.unusable["a"] == ("ent",)

    def test_ties_break_deterministically_by_name(self) -> None:
        specs = [MetricSpec("acc", 1.0)]
        readings = {"z": {"acc": measured(0.5)}, "a": {"acc": measured(0.5)}}
        assert list(score_encodings(readings, specs).scores) == ["a", "z"]

    def test_weights_need_not_sum_to_one(self) -> None:
        a = score_encodings(FIXTURE, [MetricSpec("acc", 1.0), MetricSpec("ent", 1.0)])
        b = score_encodings(FIXTURE, [MetricSpec("acc", 5.0), MetricSpec("ent", 5.0)])
        assert a.scores == pytest.approx(b.scores)

    def test_result_is_json_serialisable(self) -> None:
        payload = score_encodings(FIXTURE, SPECS).to_dict()
        assert json.loads(json.dumps(payload))["best"] == "partial"

    def test_returns_the_documented_type(self) -> None:
        assert isinstance(score_encodings(FIXTURE, SPECS), ScoringResult)

    def test_empty_specs_raise(self) -> None:
        with pytest.raises(ValueError, match="specs must not be empty"):
            score_encodings(FIXTURE, [])

    def test_duplicate_spec_names_raise(self) -> None:
        with pytest.raises(ValueError, match="duplicate metric names"):
            score_encodings(FIXTURE, [MetricSpec("acc", 1.0), MetricSpec("acc", 1.0)])

    def test_empty_readings_raise(self) -> None:
        with pytest.raises(ValueError, match="readings must not be empty"):
            score_encodings({}, SPECS)

    def test_unknown_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="on_unusable must be"):
            score_encodings(FIXTURE, SPECS, on_unusable="impute")  # type: ignore[arg-type]

    def test_nothing_usable_raises(self) -> None:
        readings = {"a": {"acc": not_measured(), "ent": not_measured()}}
        with pytest.raises(ValueError, match="no encoding has a usable value"):
            score_encodings(readings, SPECS)


class TestSummaries:
    """The summaries are what a reader actually sees, so pin their branches."""

    CLEAN = {
        "a": {"acc": measured(0.9), "ent": measured(0.2)},
        "b": {"acc": measured(0.5), "ent": measured(0.8)},
        "c": {"acc": measured(0.1), "ent": measured(0.4)},
    }

    def test_clean_run_mentions_no_caveats(self) -> None:
        text = score_encodings(self.CLEAN, SPECS).summary()
        assert "reduced objective" not in text
        assert "known to be zero" not in text
        assert "excluded" not in text

    def test_excluded_encodings_are_named(self) -> None:
        text = score_encodings(FIXTURE, SPECS, on_unusable="exclude").summary()
        assert "excluded: partial lacks ['ent']" in text

    def test_structural_zeros_are_not_described_as_lacking(self) -> None:
        """They are present and equal to zero; 'lacks' would be wrong."""
        text = score_encodings(FIXTURE, SPECS).summary()
        assert "known to be zero, not missing: plain: ['ent']" in text
        assert "plain lacks" not in text

    def test_sensitivity_names_robust_encodings(self) -> None:
        text = weight_sensitivity(
            FIXTURE, SPECS, n_samples=100, seed=7, k=2, threshold=0.5
        ).summary()
        assert "robustly top-2" in text
        assert "partial" in text

    def test_sensitivity_reports_reduced_objectives(self) -> None:
        text = weight_sensitivity(FIXTURE, SPECS, n_samples=50, seed=0).summary()
        assert "scored on a reduced objective: ['partial']" in text

    def test_sensitivity_omits_caveats_when_none_apply(self) -> None:
        text = weight_sensitivity(self.CLEAN, SPECS, n_samples=50, seed=0).summary()
        assert "reduced objective" not in text


class TestWeightSensitivity:
    def test_pinned_output(self) -> None:
        """Pins the RandomState stream; any RNG change fails here."""
        result = weight_sensitivity(
            FIXTURE, SPECS, n_samples=200, seed=7, k=2, threshold=0.5
        )
        assert result.mean_rank["partial"] == pytest.approx(1.0)
        assert result.mean_rank["deep"] == pytest.approx(2.9)
        assert result.mean_rank["mid"] == pytest.approx(3.0)
        assert result.mean_rank["plain"] == pytest.approx(3.1)
        assert result.top_k_fraction["deep"] == pytest.approx(0.55)
        assert result.robust == ("partial", "deep")

    def test_same_seed_reproduces_exactly(self) -> None:
        a = weight_sensitivity(FIXTURE, SPECS, n_samples=100, seed=3)
        b = weight_sensitivity(FIXTURE, SPECS, n_samples=100, seed=3)
        assert a.mean_rank == b.mean_rank
        assert a.top_k_fraction == b.top_k_fraction

    def test_different_seeds_differ(self) -> None:
        a = weight_sensitivity(FIXTURE, SPECS, n_samples=200, seed=1)
        b = weight_sensitivity(FIXTURE, SPECS, n_samples=200, seed=2)
        assert a.mean_rank != b.mean_rank

    def test_does_not_use_the_unstable_generator(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """default_rng streams are not guaranteed across NumPy releases.

        Published statistics must not depend on one, so make any use of it an
        immediate failure rather than a silent drift years later.
        """

        def _forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError("weight_sensitivity must not use default_rng")

        monkeypatch.setattr(np.random, "default_rng", _forbidden)
        result = weight_sensitivity(FIXTURE, SPECS, n_samples=50, seed=0)
        assert result.n_samples == 50

    def test_empty_robust_is_a_real_answer(self) -> None:
        """No encoding dominating under reweighting is a finding, not an error.

        'partial' is dropped because, being scored on one axis only, it takes
        rank 1 in every draw and would mask the effect being tested.
        """
        contested = {k: v for k, v in FIXTURE.items() if k != "partial"}
        result = weight_sensitivity(
            contested, SPECS, n_samples=200, seed=7, k=1, threshold=0.99
        )
        assert result.robust == ()
        assert "does not survive reweighting" in result.summary()

    def test_a_one_axis_encoding_wins_every_draw(self) -> None:
        """Which is exactly why effective_weight has to be reported."""
        result = weight_sensitivity(FIXTURE, SPECS, n_samples=200, seed=7, k=1)
        assert result.top_k_fraction["partial"] == pytest.approx(1.0)
        assert result.effective_weight["partial"] == pytest.approx(0.5)

    def test_reports_reduced_objectives(self) -> None:
        result = weight_sensitivity(FIXTURE, SPECS, n_samples=50, seed=0)
        assert result.effective_weight["partial"] == pytest.approx(0.5)
        assert result.unusable["partial"] == ("ent",)
        assert result.structural_zeros["plain"] == ("ent",)

    def test_rank_bounds_are_consistent(self) -> None:
        result = weight_sensitivity(FIXTURE, SPECS, n_samples=100, seed=5)
        for name in result.mean_rank:
            assert 1 <= result.best_rank[name] <= result.worst_rank[name] <= 4
            assert (
                result.best_rank[name]
                <= result.mean_rank[name]
                <= result.worst_rank[name]
            )

    def test_equal_weight_ranking_matches_equal_weight_scoring(self) -> None:
        equal = [MetricSpec(s.name, 1.0) for s in SPECS]
        expected = tuple(score_encodings(FIXTURE, equal).scores)
        result = weight_sensitivity(FIXTURE, SPECS, n_samples=10, seed=0)
        assert result.equal_weight_ranking == expected

    def test_declared_weights_are_ignored(self) -> None:
        """The point of the sweep is to replace the chosen weights."""
        skewed = [MetricSpec("acc", 99.0), MetricSpec("ent", 0.01)]
        a = weight_sensitivity(FIXTURE, SPECS, n_samples=100, seed=4)
        b = weight_sensitivity(FIXTURE, skewed, n_samples=100, seed=4)
        assert a.mean_rank == b.mean_rank

    def test_result_is_json_serialisable(self) -> None:
        payload = weight_sensitivity(FIXTURE, SPECS, n_samples=20, seed=0).to_dict()
        assert json.loads(json.dumps(payload))["n_samples"] == 20

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"n_samples": 0}, "n_samples must be positive"),
            ({"k": 0}, "k must be positive"),
            ({"threshold": 0.0}, "threshold must be in"),
            ({"threshold": 1.5}, "threshold must be in"),
        ],
    )
    def test_invalid_arguments_raise(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            weight_sensitivity(FIXTURE, SPECS, **kwargs)


class TestReadingsFromValues:
    def test_none_becomes_not_measured_by_default(self) -> None:
        readings = readings_from_values({"a": {"ent": None}})
        assert readings["a"]["ent"].availability == "not_measured"

    def test_listed_pairs_become_structural_zeros(self) -> None:
        readings = readings_from_values(
            {"a": {"ent": None}, "b": {"ent": 0.6}},
            structural_zeros={"a": ["ent"]},
            reason="no entangling gates",
        )
        assert readings["a"]["ent"].availability == "structurally_zero"
        assert readings["a"]["ent"].reason == "no entangling gates"
        assert readings["b"]["ent"].availability == "measured"

    def test_present_values_are_never_overridden(self) -> None:
        readings = readings_from_values(
            {"a": {"ent": 0.4}}, structural_zeros={"a": ["ent"]}
        )
        assert readings["a"]["ent"].resolved == pytest.approx(0.4)

    def test_round_trips_into_scoring(self) -> None:
        readings = readings_from_values(
            {"a": {"acc": 0.9, "ent": None}, "b": {"acc": 0.5, "ent": 1.0}},
            structural_zeros={"a": ["ent"]},
        )
        assert score_encodings(readings, SPECS).normalization["ent"] == (0.0, 1.0)


class TestDescribeUnusable:
    def test_empty_reads_as_none(self) -> None:
        assert describe_unusable({}) == "none"

    def test_sorted_and_named(self) -> None:
        text = describe_unusable({"b": ("y",), "a": ("x",)})
        assert text.index("a lacks") < text.index("b lacks")


class TestPipelineReadings:
    """The atlas pipeline must classify its own absences correctly.

    Synthetic profiles, so this runs in CI where the raw pipeline output is not
    packaged — otherwise the classification would only be covered by the
    raw-data tests below, which skip there.
    """

    @staticmethod
    def _profiles() -> dict:
        return {
            "plain": {
                "vqc_accuracy": {"d1": 0.9},
                "depth": 2,
                "expressibility": 0.93,
                "trainability_estimate": 0.1,
                "entanglement_capability": None,
                "noise_resilience": 1.0,
                "is_entangling": False,
            },
            "deep": {
                "vqc_accuracy": {"d1": 0.7},
                "depth": 10,
                "expressibility": 0.99,
                "trainability_estimate": 0.05,
                "entanglement_capability": 0.8,
                "noise_resilience": 0.0,
                "is_entangling": True,
            },
            "unmeasured": {
                "vqc_accuracy": {"d1": 0.8},
                "depth": 4,
                "expressibility": None,
                "trainability_estimate": 0.2,
                "entanglement_capability": 0.3,
                "noise_resilience": 0.5,
                "is_entangling": True,
            },
            "no_accuracy": {"vqc_accuracy": {}, "depth": 3, "is_entangling": True},
        }

    def test_non_entangling_absence_is_a_structural_zero(self) -> None:
        from experiments.tradeoff import build_metric_readings

        readings = build_metric_readings(self._profiles())
        reading = readings["plain"]["entanglement"]
        assert reading.availability == "structurally_zero"
        assert reading.resolved == 0.0
        assert "exactly 0" in reading.reason

    def test_entangling_absence_stays_unknown(self) -> None:
        """Only a *non*-entangling encoding earns the structural zero."""
        from experiments.tradeoff import build_metric_readings

        readings = build_metric_readings(self._profiles())
        assert readings["unmeasured"]["expressibility"].availability == "not_measured"

    def test_encodings_without_accuracy_are_left_out(self) -> None:
        from experiments.tradeoff import build_metric_readings

        assert "no_accuracy" not in build_metric_readings(self._profiles())

    def test_readings_feed_the_shared_scorer(self) -> None:
        from experiments.tradeoff import SCORING_SPECS, build_metric_readings

        result = score_encodings(build_metric_readings(self._profiles()), SCORING_SPECS)
        assert result.structural_zeros["plain"] == ("entanglement",)
        assert result.unusable["unmeasured"] == ("expressibility",)
        assert result.effective_weight["plain"] == pytest.approx(1.0)
        assert result.effective_weight["unmeasured"] == pytest.approx(0.85)

    def test_both_stage7_analyses_share_one_policy(self) -> None:
        """7.8 and 7.9 previously resolved absences differently."""
        from experiments.tradeoff import (
            analyze_ranking_sensitivity,
            build_metric_readings,
        )

        profiles = self._profiles()
        sensitivity = analyze_ranking_sensitivity(profiles, n_samples=20, seed=0)
        readings = build_metric_readings(profiles)
        assert sensitivity["structural_zeros"] == {"plain": ["entanglement"]}
        assert set(sensitivity["per_encoding"]) == set(readings)


@pytest.mark.skipif(
    not (RAW_TRADEOFF.exists() and STAGE_CONFIG.exists()),
    reason="raw stage7 output not available",
)
class TestAgainstTheRealAtlas:
    """Pins the corrected atlas numbers where the pipeline output is present.

    Not packaged, so this skips in CI exactly as the other raw-data tests do.
    """

    @staticmethod
    def _profiles() -> dict:
        from experiments.tradeoff import build_encoding_profiles

        stage_dirs = json.loads(STAGE_CONFIG.read_text(encoding="utf-8"))[
            "analysis_params"
        ]["stage_dirs"]
        return build_encoding_profiles(stage_dirs)

    def test_non_entangling_encodings_are_structural_zeros(self) -> None:
        from experiments.tradeoff import build_metric_readings

        readings = build_metric_readings(self._profiles())
        for name in ("angle", "higher_order_angle", "basis"):
            assert readings[name]["entanglement"].availability == "structurally_zero"

    def test_the_robustness_claim_does_not_survive(self) -> None:
        """Published: robustly_top3 == ['angle'] at 97.8%. It was an artefact."""
        from experiments.tradeoff import analyze_ranking_sensitivity

        published = json.loads(RAW_TRADEOFF.read_text(encoding="utf-8"))["results"][0][
            "result"
        ]["sub_analyses"]["7.8"]
        assert published["robustly_top3"] == ["angle"]
        assert published["per_encoding"]["angle"]["pct_top3"] == pytest.approx(
            97.8, abs=0.1
        )

        corrected = analyze_ranking_sensitivity(
            self._profiles(), n_samples=1000, seed=42
        )
        assert corrected["robustly_top3"] == []
        assert corrected["per_encoding"]["angle"]["pct_top3"] == pytest.approx(
            80.8, abs=0.1
        )

    def test_legacy_output_schema_is_preserved(self) -> None:
        from experiments.tradeoff import analyze_ranking_sensitivity

        result = analyze_ranking_sensitivity(self._profiles(), n_samples=50, seed=42)
        assert {
            "per_encoding",
            "robustly_top3",
            "default_ranking",
            "n_samples",
            "metric_keys",
        } <= set(result)
        entry = next(iter(result["per_encoding"].values()))
        assert {"mean_rank", "std_rank", "min_rank", "max_rank", "pct_top3"} == set(
            entry
        )

    def test_reduced_objectives_are_reported(self) -> None:
        from experiments.tradeoff import analyze_ranking_sensitivity

        result = analyze_ranking_sensitivity(self._profiles(), n_samples=50, seed=42)
        assert result["effective_weight"]["basis"] == pytest.approx(0.70)
        assert result["effective_weight"]["amplitude"] == pytest.approx(0.85)

    def test_final_ranking_still_has_angle_first(self) -> None:
        """The correction moves ranks; it does not overturn the headline."""
        from experiments.tradeoff import analyze_pareto_front, compute_final_ranking

        profiles = self._profiles()
        ranking = compute_final_ranking(profiles, analyze_pareto_front(profiles))
        order = [r["encoding"] for r in ranking["rankings"]]
        assert order[0] == "angle"
        assert order.index("higher_order_angle") == 4  # was 2 (rank 3)
        assert math.isclose(ranking["rankings"][0]["score"], 0.733, abs_tol=5e-4)
