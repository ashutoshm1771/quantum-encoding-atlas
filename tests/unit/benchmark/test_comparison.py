"""Tests for the Demsar multiple-comparison procedure.

Critical values are checked against the tables published in Demsar (2006)
rather than against the implementation's own output, because a
self-consistent-but-wrong critical value would pass any internal check while
silently changing every conclusion drawn from it.

The rank-handling tests exist because of a specific defect: the atlas's
critical-difference figures imputed a missing (encoding, dataset) cell as
accuracy 0.0 and ranked it, which assigned worst rank for being *inapplicable*
rather than for performing badly. ``TestNoImputation`` pins the corrected
behaviour on a synthetic reproduction of that exact shape.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from encoding_atlas.benchmark.comparison import (
    ComparisonResult,
    average_ranks,
    bonferroni_dunn_quantile,
    compare_over_datasets,
    complete_cases,
    critical_difference,
    describe_exclusions,
    friedman_test,
    iman_davenport,
    rank_matrix,
    studentized_range_quantile,
)

# Demsar (2006) Table 5: Nemenyi q_0.05, k = 2..20.
DEMSAR_TABLE_5 = {
    2: 1.960,
    3: 2.343,
    4: 2.569,
    5: 2.728,
    6: 2.850,
    7: 2.949,
    8: 3.031,
    9: 3.102,
    10: 3.164,
    11: 3.219,
    12: 3.268,
    13: 3.313,
    14: 3.354,
    15: 3.391,
    16: 3.426,
    17: 3.458,
    18: 3.489,
    19: 3.517,
    20: 3.544,
}

# Demsar (2006) Table 6: two-tailed Bonferroni-Dunn, alpha = 0.05.
# k = 9 is omitted: the published 2.724 does not follow from
# alpha / (k - 1) = 0.00625, which gives 2.7344. The neighbouring entries
# (2.690 at k = 8, 2.773 at k = 10) bracket 2.7344 and not 2.724, so the table
# entry is a typo. TestBonferroniDunnQuantile pins the computed value.
DEMSAR_TABLE_6 = {
    2: 1.960,
    3: 2.241,
    4: 2.394,
    5: 2.498,
    6: 2.576,
    7: 2.638,
    8: 2.690,
    10: 2.773,
}

# A clean three-method suite: strictly ordered on every dataset.
ORDERED_SCORES = {
    "good": {"d1": 0.95, "d2": 0.93, "d3": 0.97, "d4": 0.94, "d5": 0.96},
    "middle": {"d1": 0.85, "d2": 0.83, "d3": 0.87, "d4": 0.84, "d5": 0.86},
    "poor": {"d1": 0.60, "d2": 0.62, "d3": 0.58, "d4": 0.61, "d5": 0.59},
}

RAW_KERNEL = pathlib.Path("experiments/results/raw/stage6b_kernel/summary.json")


def _friedman_closed_form(ranks: list[float], n_datasets: int) -> float:
    """Friedman's chi-square from average ranks, computed from the definition.

    ``chi2 = 12N / (k(k+1)) * (sum R_j^2 - k(k+1)^2 / 4)``. Valid without a tie
    correction, so callers must supply untied data.
    """
    k = len(ranks)
    total = sum(r**2 for r in ranks)
    return (12.0 * n_datasets / (k * (k + 1))) * (total - k * (k + 1) ** 2 / 4.0)


class TestStudentizedRangeQuantile:
    """Checked against Demsar Table 5, not against itself."""

    @pytest.mark.parametrize(("k", "expected"), sorted(DEMSAR_TABLE_5.items()))
    def test_matches_demsar_table_5(self, k: int, expected: float) -> None:
        assert studentized_range_quantile(k, 0.05) == pytest.approx(expected, abs=1e-3)

    def test_table_agreement_is_within_the_tables_own_rounding(self) -> None:
        """The published table has 3 decimals; agreement must be that tight."""
        worst = max(
            abs(studentized_range_quantile(k, 0.05) - v)
            for k, v in DEMSAR_TABLE_5.items()
        )
        assert worst < 1e-3

    def test_increases_with_more_methods(self) -> None:
        """More comparisons demand a wider critical value."""
        values = [studentized_range_quantile(k, 0.05) for k in range(2, 21)]
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_stricter_alpha_gives_larger_quantile(self) -> None:
        assert studentized_range_quantile(10, 0.01) > studentized_range_quantile(
            10, 0.05
        )

    def test_extends_beyond_the_published_table(self) -> None:
        """The point of computing rather than looking up: k > 20 works."""
        value = studentized_range_quantile(40, 0.05)
        assert value > DEMSAR_TABLE_5[20]
        assert math.isfinite(value)

    @pytest.mark.parametrize("bad_k", [1, 0, -3])
    def test_too_few_methods_raises(self, bad_k: int) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            studentized_range_quantile(bad_k)

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.1, 1.5])
    def test_bad_alpha_raises(self, bad_alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            studentized_range_quantile(5, bad_alpha)


class TestBonferroniDunnQuantile:
    @pytest.mark.parametrize(("k", "expected"), sorted(DEMSAR_TABLE_6.items()))
    def test_matches_demsar_table_6(self, k: int, expected: float) -> None:
        assert bonferroni_dunn_quantile(k, 0.05) == pytest.approx(expected, abs=1e-3)

    def test_k9_follows_the_definition_not_the_published_typo(self) -> None:
        """alpha/(k-1) = 0.00625 gives 2.7344; the table's 2.724 does not follow."""
        value = bonferroni_dunn_quantile(9, 0.05)
        assert value == pytest.approx(2.7344, abs=1e-3)
        assert (
            bonferroni_dunn_quantile(8, 0.05)
            < value
            < bonferroni_dunn_quantile(10, 0.05)
        )

    def test_is_more_powerful_than_nemenyi(self) -> None:
        """Fewer comparisons, so a smaller critical value, for every k > 2."""
        for k in range(3, 21):
            assert bonferroni_dunn_quantile(k, 0.05) < studentized_range_quantile(
                k, 0.05
            )

    @pytest.mark.parametrize("bad_k", [1, 0])
    def test_too_few_methods_raises(self, bad_k: int) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            bonferroni_dunn_quantile(bad_k)

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0])
    def test_bad_alpha_raises(self, bad_alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            bonferroni_dunn_quantile(5, bad_alpha)


class TestImanDavenport:
    def test_reproduces_demsar_worked_example(self) -> None:
        """Demsar Section 3.2.2: chi2 = 9.28, k = 4, N = 14 -> F = 3.69."""
        f_statistic, _ = iman_davenport(9.28, 4, 14)
        assert f_statistic == pytest.approx(3.69, abs=5e-3)

    def test_zero_chi_square_gives_zero_f(self) -> None:
        f_statistic, p_value = iman_davenport(0.0, 5, 10)
        assert f_statistic == 0.0
        assert p_value == pytest.approx(1.0)

    def test_maximal_chi_square_is_infinite_and_decisive(self) -> None:
        """Perfect rank agreement saturates chi2 at N(k-1); F diverges."""
        f_statistic, p_value = iman_davenport(10 * 3, 4, 10)
        assert math.isinf(f_statistic)
        assert p_value == 0.0

    def test_is_monotone_in_chi_square(self) -> None:
        values = [iman_davenport(c, 5, 10)[0] for c in (1.0, 5.0, 10.0, 20.0)]
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_p_value_decreases_as_f_grows(self) -> None:
        assert iman_davenport(20.0, 5, 10)[1] < iman_davenport(2.0, 5, 10)[1]

    def test_chi_square_above_its_maximum_raises(self) -> None:
        with pytest.raises(ValueError, match="exceeds its maximum"):
            iman_davenport(31.0, 4, 10)

    def test_negative_chi_square_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            iman_davenport(-1.0, 4, 10)

    @pytest.mark.parametrize(("k", "n"), [(1, 10), (4, 1)])
    def test_degenerate_counts_raise(self, k: int, n: int) -> None:
        with pytest.raises(ValueError, match="at least 2"):
            iman_davenport(1.0, k, n)


class TestCriticalDifference:
    def test_matches_the_formula(self) -> None:
        k, n = 15, 8
        expected = studentized_range_quantile(k, 0.05) * math.sqrt(
            k * (k + 1) / (6.0 * n)
        )
        assert critical_difference(k, n) == pytest.approx(expected)

    def test_atlas_configuration(self) -> None:
        """15 encodings over 8 datasets — the atlas's own shape."""
        assert critical_difference(15, 8) == pytest.approx(7.583, abs=1e-3)

    def test_shrinks_as_datasets_are_added(self) -> None:
        """More datasets is the only way to buy resolution."""
        values = [critical_difference(15, n) for n in (4, 8, 16, 32)]
        assert all(b < a for a, b in zip(values, values[1:]))

    def test_grows_with_more_methods(self) -> None:
        values = [critical_difference(k, 8) for k in (4, 8, 16)]
        assert all(b > a for a, b in zip(values, values[1:]))

    def test_bonferroni_dunn_is_tighter(self) -> None:
        assert critical_difference(15, 8, test="bonferroni-dunn") < critical_difference(
            15, 8, test="nemenyi"
        )

    def test_unknown_test_raises(self) -> None:
        with pytest.raises(ValueError, match="nemenyi"):
            critical_difference(5, 8, test="tukey")  # type: ignore[arg-type]

    def test_zero_datasets_raises(self) -> None:
        with pytest.raises(ValueError, match="n_datasets"):
            critical_difference(5, 0)


class TestCompleteCases:
    def test_complete_design_excludes_nothing(self) -> None:
        methods, datasets, excluded = complete_cases(ORDERED_SCORES)
        assert methods == ["good", "middle", "poor"]
        assert datasets == ["d1", "d2", "d3", "d4", "d5"]
        assert excluded == {}

    def test_reports_which_datasets_are_missing(self) -> None:
        methods, _, excluded = complete_cases(
            {"a": {"d1": 1.0, "d2": 1.0}, "b": {"d1": 0.5}}
        )
        assert methods == ["a"]
        assert excluded == {"b": ("d2",)}

    def test_datasets_are_the_union_not_the_intersection(self) -> None:
        _, datasets, _ = complete_cases({"a": {"d1": 1.0}, "b": {"d2": 1.0}})
        assert datasets == ["d1", "d2"]

    def test_empty_scores_raise(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            complete_cases({})

    def test_no_datasets_raises(self) -> None:
        with pytest.raises(ValueError, match="no datasets"):
            complete_cases({"a": {}})

    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_score_raises(self, bad: float) -> None:
        with pytest.raises(ValueError, match="finite"):
            complete_cases({"a": {"d1": bad}, "b": {"d1": 0.5}})


class TestRankMatrix:
    def test_best_score_gets_rank_one(self) -> None:
        matrix = rank_matrix(ORDERED_SCORES, ["good", "middle", "poor"], ["d1"])
        assert matrix.tolist() == [[1.0, 2.0, 3.0]]

    def test_lower_is_better_inverts_the_order(self) -> None:
        matrix = rank_matrix(
            ORDERED_SCORES, ["good", "middle", "poor"], ["d1"], higher_is_better=False
        )
        assert matrix.tolist() == [[3.0, 2.0, 1.0]]

    def test_ties_share_the_average_rank(self) -> None:
        scores = {"a": {"d": 0.9}, "b": {"d": 0.9}, "c": {"d": 0.1}}
        matrix = rank_matrix(scores, ["a", "b", "c"], ["d"])
        assert matrix.tolist() == [[1.5, 1.5, 3.0]]

    def test_every_row_sums_to_the_same_total(self) -> None:
        """k(k+1)/2 regardless of ties — the invariant Friedman relies on."""
        scores = {
            "a": {"d1": 0.9, "d2": 0.5},
            "b": {"d1": 0.9, "d2": 0.7},
            "c": {"d1": 0.1, "d2": 0.7},
        }
        matrix = rank_matrix(scores, ["a", "b", "c"], ["d1", "d2"])
        assert matrix.sum(axis=1).tolist() == [6.0, 6.0]

    def test_shape_is_datasets_by_methods(self) -> None:
        matrix = rank_matrix(
            ORDERED_SCORES, ["good", "middle", "poor"], ["d1", "d2", "d3"]
        )
        assert matrix.shape == (3, 3)

    def test_missing_cell_raises_rather_than_imputing(self) -> None:
        with pytest.raises(ValueError, match="complete block design"):
            rank_matrix({"a": {"d1": 1.0}, "b": {}}, ["a", "b"], ["d1"])


class TestAverageRanks:
    def test_orders_best_first(self) -> None:
        ranks = average_ranks(ORDERED_SCORES)
        assert list(ranks) == ["good", "middle", "poor"]
        assert ranks["good"] == pytest.approx(1.0)
        assert ranks["poor"] == pytest.approx(3.0)

    def test_complete_policy_drops_incomplete_methods(self) -> None:
        scores = dict(ORDERED_SCORES, partial={"d1": 0.99})
        assert "partial" not in average_ranks(scores, missing="complete")

    def test_available_policy_keeps_them(self) -> None:
        scores = dict(ORDERED_SCORES, partial={"d1": 0.99})
        ranks = average_ranks(scores, missing="available")
        assert ranks["partial"] == pytest.approx(1.0)

    def test_error_policy_raises_and_names_the_gap(self) -> None:
        scores = dict(ORDERED_SCORES, partial={"d1": 0.99})
        with pytest.raises(ValueError, match="partial lacks"):
            average_ranks(scores, missing="error")

    def test_unknown_policy_raises(self) -> None:
        with pytest.raises(ValueError, match="missing must be"):
            average_ranks(ORDERED_SCORES, missing="drop")  # type: ignore[arg-type]

    def test_no_complete_method_raises(self) -> None:
        with pytest.raises(ValueError, match="no complete block design|no method has"):
            average_ranks({"a": {"d1": 1.0}, "b": {"d2": 1.0}}, missing="complete")

    def test_lower_is_better_flips_the_ordering(self) -> None:
        ranks = average_ranks(ORDERED_SCORES, higher_is_better=False)
        assert list(ranks) == ["poor", "middle", "good"]

    def test_ranks_average_to_the_midpoint(self) -> None:
        """Mean of all average ranks is always (k+1)/2."""
        ranks = average_ranks(ORDERED_SCORES)
        assert float(np.mean(list(ranks.values()))) == pytest.approx(2.0)


class TestNoImputation:
    """The defect this module was written to prevent recurring.

    A method applicable to only half the suite must not be ranked worst on the
    half it cannot run. The shape mirrors ``SO2EquivariantFeatureMap``, which
    accepts only two features and so has no result on the atlas's 4-feature
    datasets, yet ranks near the top on the datasets it does support.
    """

    #: "narrow" is the best method wherever it runs, and absent elsewhere.
    SCORES = {
        "narrow": {"d1": 0.99, "d2": 0.98},
        "a": {"d1": 0.90, "d2": 0.91, "d3": 0.92, "d4": 0.93},
        "b": {"d1": 0.80, "d2": 0.81, "d3": 0.82, "d4": 0.83},
        "c": {"d1": 0.70, "d2": 0.71, "d3": 0.72, "d4": 0.73},
        "d": {"d1": 0.60, "d2": 0.61, "d3": 0.62, "d4": 0.63},
    }

    def test_descriptive_rank_reflects_where_it_actually_ran(self) -> None:
        ranks = average_ranks(self.SCORES, missing="available")
        assert ranks["narrow"] == pytest.approx(1.0)

    def test_zero_imputation_would_have_ranked_it_last(self) -> None:
        """Pin the wrong answer, so the regression is unmistakable."""
        imputed = dict(self.SCORES)
        imputed["narrow"] = {"d1": 0.99, "d2": 0.98, "d3": 0.0, "d4": 0.0}
        assert average_ranks(imputed)["narrow"] == pytest.approx(3.0)
        assert average_ranks(self.SCORES, missing="available")["narrow"] < 3.0

    def test_incomplete_method_is_excluded_and_named(self) -> None:
        result = compare_over_datasets(self.SCORES)
        assert "narrow" not in result.methods
        assert result.excluded == {"narrow": ("d3", "d4")}

    def test_exclusion_appears_in_the_summary(self) -> None:
        """A dropped method must never be silently absent from the report."""
        assert "narrow" in compare_over_datasets(self.SCORES).summary()

    def test_excluding_one_method_does_not_disturb_the_others(self) -> None:
        """The bias was concentrated on the incomplete method; verify that."""
        without = {k: v for k, v in self.SCORES.items() if k != "narrow"}
        assert average_ranks(self.SCORES) == pytest.approx(average_ranks(without))


class TestFriedman:
    def test_matches_the_closed_form(self) -> None:
        """Independent oracle: the textbook formula, on untied data."""
        result = friedman_test(ORDERED_SCORES)
        ranks = list(average_ranks(ORDERED_SCORES).values())
        expected = _friedman_closed_form(ranks, result.n_datasets)
        assert result.chi_square == pytest.approx(expected, rel=1e-9)

    def test_perfect_separation_saturates_the_statistic(self) -> None:
        result = friedman_test(ORDERED_SCORES)
        assert result.chi_square == pytest.approx(
            result.n_datasets * (result.n_methods - 1)
        )
        assert math.isinf(result.f_statistic)
        assert result.rejected

    def test_identical_methods_do_not_reject(self) -> None:
        flat = {name: {"d1": 0.5, "d2": 0.5, "d3": 0.5} for name in "abc"}
        result = friedman_test(flat)
        assert result.chi_square == pytest.approx(0.0)
        assert not result.rejected

    def test_reports_both_forms(self) -> None:
        result = friedman_test(ORDERED_SCORES)
        assert result.chi_square_dof == result.n_methods - 1
        assert result.f_dof == (
            result.n_methods - 1,
            (result.n_methods - 1) * (result.n_datasets - 1),
        )

    def test_f_never_fails_to_reject_where_chi_square_rejects(self) -> None:
        """The sense in which Iman-Davenport is the less conservative test.

        Their p-values are *not* pointwise ordered — for small statistics the
        F p-value is often the larger of the two. What holds, and what makes
        the F form preferable, is that it never *loses* a rejection the
        chi-square form would have made.
        """
        rng = np.random.default_rng(0)
        checked = 0
        for k in (3, 4, 5, 8):
            for n_datasets in (3, 5, 8, 14):
                for _ in range(15):
                    scores = {
                        f"m{j}": {
                            f"d{i}": float(rng.normal(j * 0.2, 1.0))
                            for i in range(n_datasets)
                        }
                        for j in range(k)
                    }
                    result = friedman_test(scores)
                    if result.chi_square_rejected:
                        assert result.rejected
                        checked += 1
        assert checked > 0, "no trial rejected; the assertion was never exercised"

    def test_excludes_incomplete_methods(self) -> None:
        scores = dict(ORDERED_SCORES, partial={"d1": 0.99})
        assert friedman_test(scores).n_methods == 3

    def test_too_few_methods_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 3 methods"):
            friedman_test({"a": {"d1": 1.0, "d2": 1.0}, "b": {"d1": 0.0, "d2": 0.0}})

    def test_too_few_datasets_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 2 datasets"):
            friedman_test(
                {n: {"d1": v} for n, v in (("a", 1.0), ("b", 2.0), ("c", 3.0))}
            )

    @pytest.mark.parametrize("bad_alpha", [0.0, 1.0, -0.5])
    def test_bad_alpha_raises(self, bad_alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            friedman_test(ORDERED_SCORES, alpha=bad_alpha)

    def test_alpha_controls_the_rejection_flag(self) -> None:
        rng = np.random.default_rng(3)
        scores = {
            f"m{j}": {f"d{i}": float(rng.normal(j * 0.15, 1.0)) for i in range(5)}
            for j in range(4)
        }
        strict = friedman_test(scores, alpha=1e-9)
        assert not strict.rejected


class TestCompareOverDatasets:
    def test_end_to_end_on_a_separable_suite(self) -> None:
        result = compare_over_datasets(ORDERED_SCORES)
        assert result.best == "good"
        assert result.friedman.rejected
        assert result.posthoc is not None
        assert result.posthoc.test == "nemenyi"

    def test_no_posthoc_when_the_omnibus_does_not_reject(self) -> None:
        """The guard: a post-hoc without a rejecting omnibus is invalid."""
        flat = {name: {f"d{i}": 0.5 for i in range(4)} for name in "abc"}
        result = compare_over_datasets(flat)
        assert not result.friedman.rejected
        assert result.posthoc is None
        assert "does NOT reject" in result.summary()

    def test_posthoc_reports_the_rank_range_beside_the_cd(self) -> None:
        """So a caller can see when the suite simply cannot resolve much."""
        result = compare_over_datasets(ORDERED_SCORES)
        assert result.posthoc is not None
        assert result.posthoc.rank_range == pytest.approx(2.0)

    def test_significant_pairs_are_ordered_and_directed(self) -> None:
        result = compare_over_datasets(ORDERED_SCORES)
        assert result.posthoc is not None
        gaps = [gap for _, _, gap in result.posthoc.significant_pairs]
        assert gaps == sorted(gaps, reverse=True)
        for better, worse, _ in result.posthoc.significant_pairs:
            assert result.average_ranks[better] < result.average_ranks[worse]

    def test_separated_from_is_symmetric(self) -> None:
        result = compare_over_datasets(ORDERED_SCORES)
        assert result.posthoc is not None
        for better, worse, _ in result.posthoc.significant_pairs:
            assert worse in result.posthoc.separated_from(better)
            assert better in result.posthoc.separated_from(worse)

    def test_bonferroni_dunn_compares_only_against_the_control(self) -> None:
        result = compare_over_datasets(
            ORDERED_SCORES, test="bonferroni-dunn", control="good"
        )
        assert result.posthoc is not None
        assert result.posthoc.control == "good"
        for better, worse, _ in result.posthoc.significant_pairs:
            assert "good" in (better, worse)

    def test_bonferroni_dunn_without_control_raises(self) -> None:
        with pytest.raises(ValueError, match="control"):
            compare_over_datasets(ORDERED_SCORES, test="bonferroni-dunn")

    def test_unknown_control_raises(self) -> None:
        with pytest.raises(ValueError, match="not part of the complete"):
            compare_over_datasets(
                ORDERED_SCORES, test="bonferroni-dunn", control="nope"
            )

    def test_control_excluded_for_incompleteness_raises_helpfully(self) -> None:
        scores = dict(ORDERED_SCORES, partial={"d1": 0.99})
        with pytest.raises(ValueError, match="partial lacks"):
            compare_over_datasets(scores, test="bonferroni-dunn", control="partial")

    def test_unknown_test_raises(self) -> None:
        with pytest.raises(ValueError, match="nemenyi"):
            compare_over_datasets(ORDERED_SCORES, test="tukey")  # type: ignore[arg-type]

    def test_lower_is_better_is_honoured_end_to_end(self) -> None:
        result = compare_over_datasets(ORDERED_SCORES, higher_is_better=False)
        assert result.best == "poor"

    def test_result_is_json_serialisable(self) -> None:
        payload = compare_over_datasets(ORDERED_SCORES).to_dict()
        restored = json.loads(json.dumps(payload))
        assert restored["best"] == "good"
        assert restored["friedman"]["rejected"] is True

    def test_infinite_f_survives_json(self) -> None:
        """Perfect separation gives F = inf; json must not choke on the report."""
        payload = compare_over_datasets(ORDERED_SCORES).to_dict()
        assert math.isinf(payload["friedman"]["f_statistic"])
        assert json.loads(json.dumps(payload))["friedman"]["f_statistic"] == float(
            "inf"
        )

    def test_summary_is_plain_text(self) -> None:
        text = compare_over_datasets(ORDERED_SCORES).summary()
        assert "Friedman" in text and "Iman-Davenport" in text
        assert text == text.strip()

    def test_returns_the_documented_type(self) -> None:
        assert isinstance(compare_over_datasets(ORDERED_SCORES), ComparisonResult)


class TestWhyNotAllPairsWilcoxon:
    """Pin the arithmetic that motivates this module's existence."""

    def test_wilcoxon_cannot_reach_significance_at_atlas_scale(self) -> None:
        n_datasets, n_methods = 8, 15
        min_attainable_p = 2 / 2**n_datasets
        n_comparisons = n_methods * (n_methods - 1) // 2
        holm_threshold = 0.05 / n_comparisons
        assert n_comparisons == 105
        assert min_attainable_p == pytest.approx(0.0078125)
        assert holm_threshold == pytest.approx(0.05 / 105)
        assert min_attainable_p > holm_threshold

    def test_friedman_separates_what_wilcoxon_cannot(self) -> None:
        """Same data, same alpha: the omnibus route yields a verdict."""
        rng = np.random.default_rng(11)
        scores = {
            f"m{j}": {
                f"d{i}": float(0.9 - 0.05 * j + rng.normal(0, 0.005)) for i in range(8)
            }
            for j in range(15)
        }
        result = compare_over_datasets(scores)
        assert result.friedman.rejected
        assert result.posthoc is not None
        assert result.posthoc.n_significant > 0


class TestDescribeExclusions:
    def test_empty_reads_as_none(self) -> None:
        assert describe_exclusions({}) == "none"

    def test_names_method_and_datasets(self) -> None:
        text = describe_exclusions({"b": ("d2", "d3"), "a": ("d1",)})
        assert text.index("a lacks") < text.index("b lacks")
        assert "['d2', 'd3']" in text


@pytest.mark.skipif(not RAW_KERNEL.exists(), reason="raw stage6b output not available")
class TestAgainstTheRealAtlas:
    """Pins the published numbers when the raw pipeline output is present.

    Not packaged, so this skips in CI exactly as the atlas tests do; it is the
    check that runs on the machine where the figures are regenerated.
    """

    @staticmethod
    def _kernel_scores() -> dict[str, dict[str, float]]:
        raw = json.loads(RAW_KERNEL.read_text(encoding="utf-8"))
        scores: dict[str, dict[str, float]] = {}
        for entry in raw["results"]:
            name = entry["encoding_name"]
            if name.startswith("classical_"):
                continue
            row = scores.setdefault(name, {})
            for dataset, payload in (entry["result"].get("datasets") or {}).items():
                if payload.get("status") != "success" or dataset in row:
                    continue
                accuracy = payload.get("mean_test_accuracy")
                if accuracy is not None:
                    row[dataset] = float(accuracy)
        return scores

    def test_reproduces_the_published_omnibus(self) -> None:
        result = compare_over_datasets(self._kernel_scores())
        assert result.friedman.n_methods == 15
        assert result.friedman.n_datasets == 8
        assert result.friedman.chi_square == pytest.approx(79.093, abs=5e-3)
        assert result.friedman.f_statistic == pytest.approx(16.825, abs=5e-3)
        assert result.friedman.rejected

    def test_so2_is_excluded_for_the_documented_reason(self) -> None:
        result = compare_over_datasets(self._kernel_scores())
        assert result.excluded == {
            "so2_equivariant": ("breast_cancer", "digits_01", "iris", "wine")
        }

    def test_so2_descriptive_rank_is_top_third_not_bottom_half(self) -> None:
        ranks = average_ranks(self._kernel_scores(), missing="available")
        assert ranks["so2_equivariant"] == pytest.approx(4.38, abs=0.01)

    def test_only_five_encodings_separate_from_the_best(self) -> None:
        """The honest headline: 8 datasets buy very little resolution."""
        result = compare_over_datasets(self._kernel_scores())
        assert result.posthoc is not None
        assert result.best == "angle"
        assert len(result.posthoc.separated_from("angle")) == 5
        assert result.posthoc.critical_difference == pytest.approx(7.583, abs=1e-3)
