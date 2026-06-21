"""Tests for pystars.tests_continuous: t-test, Mann-Whitney, Wilcoxon, ANOVA, Kruskal-Wallis."""

import numpy as np
import pandas as pd
import pytest

from pystars.result import TestResult
from pystars.tests_continuous import (
    anova,
    anova_twoway,
    kruskal,
    mannwhitney,
    ttest,
    wilcoxon,
)


@pytest.fixture
def rng():
    return np.random.default_rng(123)


@pytest.fixture
def two_groups_different(rng):
    """Two groups with clearly different means (n=30 each)."""
    return pd.DataFrame(
        {
            "group": ["wt"] * 30 + ["mut"] * 30,
            "length": np.concatenate([rng.normal(10, 2, 30), rng.normal(15, 2, 30)]),
        }
    )


@pytest.fixture
def two_groups_same(rng):
    """Two groups with the same mean."""
    return pd.DataFrame(
        {
            "group": ["wt"] * 30 + ["mut"] * 30,
            "length": np.concatenate([rng.normal(10, 2, 30), rng.normal(10, 2, 30)]),
        }
    )


@pytest.fixture
def paired_data(rng):
    """Paired pre/post measurements with a treatment effect."""
    subjects = [f"s{i}" for i in range(25)]
    baseline = rng.normal(10, 2, 25)
    treatment = baseline + rng.normal(3, 1, 25)
    return pd.DataFrame(
        {
            "group": ["pre"] * 25 + ["post"] * 25,
            "subject": subjects * 2,
            "length": np.concatenate([baseline, treatment]),
        }
    )


@pytest.fixture
def three_groups_different(rng):
    """Three groups with different means."""
    return pd.DataFrame(
        {
            "group": ["ctrl"] * 20 + ["drug_low"] * 20 + ["drug_high"] * 20,
            "length": np.concatenate(
                [rng.normal(10, 2, 20), rng.normal(13, 2, 20), rng.normal(16, 2, 20)]
            ),
        }
    )


@pytest.fixture
def three_groups_unequal_var(rng):
    """Three groups with different means and very unequal variances."""
    return pd.DataFrame(
        {
            "group": ["ctrl"] * 30 + ["low"] * 30 + ["high"] * 30,
            "length": np.concatenate(
                [rng.normal(10, 1, 30), rng.normal(13, 5, 30), rng.normal(16, 10, 30)]
            ),
        }
    )


@pytest.fixture
def twoway_data(rng):
    """2x2 factorial design with an interaction effect."""
    rows = []
    for geno in ["wt", "ko"]:
        for time in ["d0", "d1"]:
            effect = 10
            if geno == "ko" and time == "d1":
                effect = 16  # interaction
            elif geno == "ko":
                effect = 11
            elif time == "d1":
                effect = 12
            vals = rng.normal(effect, 2, 20)
            for v in vals:
                rows.append({"genotype": geno, "time": time, "length": v})
    return pd.DataFrame(rows)


class TestTTest:
    def test_welch_different_means_significant(self, two_groups_different):
        result = ttest(two_groups_different, value="length", group="group")
        assert isinstance(result, TestResult)
        assert result.test_name == "Welch's t-test"
        assert result.p_value < 0.05

    def test_student_different_means_significant(self, two_groups_different):
        result = ttest(two_groups_different, value="length", group="group", welch=False)
        assert result.test_name == "Student's t-test"
        assert result.p_value < 0.05

    def test_same_means_not_significant(self, two_groups_same):
        result = ttest(two_groups_same, value="length", group="group")
        assert result.p_value > 0.05

    def test_paired_significant(self, paired_data):
        result = ttest(paired_data, value="length", group="group", subject="subject", paired=True)
        assert result.test_name == "Paired t-test"
        assert result.p_value < 0.05

    def test_paired_without_subject_raises(self, two_groups_different):
        with pytest.raises(ValueError, match="subject"):
            ttest(two_groups_different, value="length", group="group", paired=True)

    def test_more_than_two_groups_raises(self, three_groups_different):
        with pytest.raises(ValueError, match="2 groups"):
            ttest(three_groups_different, value="length", group="group")

    def test_effect_size_included(self, two_groups_different):
        result = ttest(two_groups_different, value="length", group="group")
        assert "cohen_d" in result.effect_size
        assert "CI95%" in result.effect_size

    def test_wide_format(self):
        df = pd.DataFrame({"wt": [1.0, 2.0, 3.0, 4.0], "mut": [5.0, 6.0, 7.0, 8.0]})
        result = ttest(df, format="wide", groups=["wt", "mut"])
        assert result.p_value < 0.05


class TestMannWhitney:
    def test_different_distributions_significant(self, two_groups_different):
        result = mannwhitney(two_groups_different, value="length", group="group")
        assert isinstance(result, TestResult)
        assert result.test_name == "Mann-Whitney U test"
        assert result.p_value < 0.05

    def test_same_distribution_not_significant(self, two_groups_same):
        result = mannwhitney(two_groups_same, value="length", group="group")
        assert result.p_value > 0.05

    def test_more_than_two_groups_raises(self, three_groups_different):
        with pytest.raises(ValueError, match="2 groups"):
            mannwhitney(three_groups_different, value="length", group="group")


class TestWilcoxon:
    def test_paired_significant(self, paired_data):
        result = wilcoxon(paired_data, value="length", group="group", subject="subject")
        assert isinstance(result, TestResult)
        assert result.test_name == "Wilcoxon signed-rank test"
        assert result.p_value < 0.05

    def test_without_subject_raises(self, two_groups_different):
        with pytest.raises(ValueError, match="subject"):
            wilcoxon(two_groups_different, value="length", group="group")


class TestAnova:
    def test_one_way_significant(self, three_groups_different):
        result = anova(three_groups_different, value="length", group="group")
        assert isinstance(result, TestResult)
        assert result.test_name == "One-way ANOVA"
        assert result.p_value < 0.05
        assert "np2" in result.effect_size

    def test_welch_significant(self, three_groups_unequal_var):
        result = anova(three_groups_unequal_var, value="length", group="group", welch=True)
        assert result.test_name == "Welch's ANOVA"
        assert result.p_value < 0.05

    def test_two_groups_works(self, two_groups_different):
        result = anova(two_groups_different, value="length", group="group")
        assert result.p_value < 0.05


class TestKruskal:
    def test_significant(self, three_groups_different):
        result = kruskal(three_groups_different, value="length", group="group")
        assert isinstance(result, TestResult)
        assert result.test_name == "Kruskal-Wallis test"
        assert result.p_value < 0.05

    def test_two_groups_works(self, two_groups_different):
        result = kruskal(two_groups_different, value="length", group="group")
        assert result.p_value < 0.05


class TestTwoWayAnova:
    def test_interaction_significant(self, twoway_data):
        result = anova_twoway(twoway_data, value="length", group=["genotype", "time"])
        assert isinstance(result, TestResult)
        assert "Two-way ANOVA" in result.test_name
        assert result.p_value < 0.05
        assert result.details is not None
        # Details table should contain all sources
        sources = result.details["Source"].tolist()
        assert "genotype" in sources
        assert "time" in sources

    def test_single_factor_raises(self, two_groups_different):
        with pytest.raises(ValueError, match="2 factor"):
            anova_twoway(two_groups_different, value="length", group="group")
