"""Tests for pystars.dispatcher: auto-selection of tests based on the flowchart."""

import numpy as np
import pandas as pd
import pytest

from pystars.dispatcher import test
from pystars.result import TestResult


@pytest.fixture
def rng():
    return np.random.default_rng(2024)


# --------------------------------------------------------------- 2-group data


@pytest.fixture
def two_normal_equal_var():
    """2 groups, normal, equal variance → Student's t-test.

    Uses a dedicated seed (2021) and n=80 so that both Shapiro-Wilk and
    Levene's tests pass reliably.
    """
    rng = np.random.default_rng(2021)
    return pd.DataFrame(
        {
            "group": ["wt"] * 80 + ["mut"] * 80,
            "value": np.concatenate([rng.normal(10, 2, 80), rng.normal(15, 2, 80)]),
        }
    )


@pytest.fixture
def two_normal_unequal_var(rng):
    """2 groups, normal, unequal variance → Welch's t-test."""
    return pd.DataFrame(
        {
            "group": ["wt"] * 50 + ["mut"] * 50,
            "value": np.concatenate([rng.normal(10, 1, 50), rng.normal(15, 10, 50)]),
        }
    )


@pytest.fixture
def two_nonnormal(rng):
    """2 groups, non-normal → Mann-Whitney U."""
    return pd.DataFrame(
        {
            "group": ["wt"] * 50 + ["mut"] * 50,
            "value": np.concatenate([rng.exponential(5, 50), rng.exponential(10, 50)]),
        }
    )


@pytest.fixture
def paired_normal(rng):
    """Paired, normal differences → Paired t-test."""
    subjects = [f"s{i}" for i in range(30)]
    baseline = rng.normal(10, 2, 30)
    treatment = baseline + rng.normal(3, 1, 30)
    return pd.DataFrame(
        {
            "group": ["pre"] * 30 + ["post"] * 30,
            "subject": subjects * 2,
            "value": np.concatenate([baseline, treatment]),
        }
    )


@pytest.fixture
def paired_small_n(rng):
    """Paired, very small n → differences non-normal → Wilcoxon."""
    subjects = [f"s{i}" for i in range(4)]
    baseline = np.array([10.0, 10.0, 10.0, 10.0])
    treatment = np.array([12.0, 11.0, 10.5, 25.0])  # one outlier → non-normal diffs
    return pd.DataFrame(
        {
            "group": ["pre"] * 4 + ["post"] * 4,
            "subject": subjects * 2,
            "value": np.concatenate([baseline, treatment]),
        }
    )


# --------------------------------------------------------------- >2-group data


@pytest.fixture
def three_normal_equal_var(rng):
    """3 groups, normal, equal variance, different means → One-way ANOVA + Tukey."""
    return pd.DataFrame(
        {
            "group": ["ctrl"] * 30 + ["low"] * 30 + ["high"] * 30,
            "value": np.concatenate(
                [rng.normal(10, 2, 30), rng.normal(14, 2, 30), rng.normal(18, 2, 30)]
            ),
        }
    )


@pytest.fixture
def three_normal_unequal_var(rng):
    """3 groups, normal, unequal variance, different means → Welch ANOVA + Games-Howell."""
    return pd.DataFrame(
        {
            "group": ["ctrl"] * 30 + ["low"] * 30 + ["high"] * 30,
            "value": np.concatenate(
                [rng.normal(10, 1, 30), rng.normal(14, 5, 30), rng.normal(18, 10, 30)]
            ),
        }
    )


@pytest.fixture
def three_nonnormal(rng):
    """3 groups, non-normal, different distributions → Kruskal-Wallis + Dunn."""
    return pd.DataFrame(
        {
            "group": ["ctrl"] * 50 + ["low"] * 50 + ["high"] * 50,
            "value": np.concatenate(
                [rng.exponential(5, 50), rng.exponential(10, 50), rng.exponential(20, 50)]
            ),
        }
    )


@pytest.fixture
def three_same_mean(rng):
    """3 groups, normal, equal variance, same mean → non-significant ANOVA."""
    return pd.DataFrame(
        {
            "group": ["ctrl"] * 30 + ["low"] * 30 + ["high"] * 30,
            "value": np.concatenate(
                [rng.normal(10, 2, 30), rng.normal(10, 2, 30), rng.normal(10, 2, 30)]
            ),
        }
    )


# --------------------------------------------------------------- factorial data


@pytest.fixture
def twoway_data(rng):
    """2×2 factorial design with an interaction."""
    rows = []
    for geno in ["wt", "ko"]:
        for time in ["d0", "d1"]:
            effect = 10
            if geno == "ko" and time == "d1":
                effect = 16
            elif geno == "ko":
                effect = 11
            elif time == "d1":
                effect = 12
            for v in rng.normal(effect, 2, 20):
                rows.append({"genotype": geno, "time": time, "value": v})
    return pd.DataFrame(rows)


# ================================================================ tests


class TestTwoGroupDispatch:
    def test_normal_equal_var_student_t(self, two_normal_equal_var):
        result = test(two_normal_equal_var, value="value", group="group")
        assert result.test_name == "Student's t-test"
        assert result.assumptions is not None
        assert "normality" in result.assumptions
        assert "equal_variance" in result.assumptions
        assert result.p_value < 0.05

    def test_normal_unequal_var_welch_t(self, two_normal_unequal_var):
        result = test(two_normal_unequal_var, value="value", group="group")
        assert result.test_name == "Welch's t-test"
        assert result.assumptions is not None
        assert result.p_value < 0.05

    def test_nonnormal_mann_whitney(self, two_nonnormal):
        result = test(two_nonnormal, value="value", group="group")
        assert result.test_name == "Mann-Whitney U test"
        assert result.assumptions is not None
        assert "normality" in result.assumptions

    def test_paired_normal_paired_t(self, paired_normal):
        result = test(paired_normal, value="value", group="group", subject="subject", paired=True)
        assert result.test_name == "Paired t-test"
        assert result.p_value < 0.05

    def test_paired_nonnormal_wilcoxon(self, paired_small_n):
        result = test(paired_small_n, value="value", group="group", subject="subject", paired=True)
        assert result.test_name == "Wilcoxon signed-rank test"

    def test_paired_no_equal_variance_check(self, paired_normal):
        """Paired tests should not check equal_variance (only normality of differences)."""
        result = test(paired_normal, value="value", group="group", subject="subject", paired=True)
        assert result.assumptions is not None
        assert "equal_variance" not in result.assumptions

    def test_small_n_treated_as_nonnormal(self, rng):
        """n < 3 per group should trigger the non-parametric branch."""
        df = pd.DataFrame({"group": ["a", "a", "b", "b"], "value": [1.0, 2.0, 5.0, 6.0]})
        result = test(df, value="value", group="group")
        assert result.test_name == "Mann-Whitney U test"


class TestManyGroupDispatch:
    def test_normal_equal_var_anova_tukey(self, three_normal_equal_var):
        result = test(three_normal_equal_var, value="value", group="group")
        assert result.test_name == "One-way ANOVA"
        assert result.p_value < 0.05
        assert result.posthoc is not None
        assert isinstance(result.posthoc, TestResult)
        assert result.posthoc.test_name == "Tukey HSD"

    def test_normal_unequal_var_welch_anova_gameshowell(self, three_normal_unequal_var):
        result = test(three_normal_unequal_var, value="value", group="group")
        assert result.test_name == "Welch's ANOVA"
        assert result.p_value < 0.05
        assert result.posthoc is not None
        assert isinstance(result.posthoc, TestResult)
        assert result.posthoc.test_name == "Games-Howell"

    def test_nonnormal_kruskal_dunn(self, three_nonnormal):
        result = test(three_nonnormal, value="value", group="group")
        assert result.test_name == "Kruskal-Wallis test"
        assert result.p_value < 0.05
        assert result.posthoc is not None
        assert isinstance(result.posthoc, TestResult)
        assert result.posthoc.test_name == "Dunn's test"

    def test_nonsig_anova_no_posthoc(self, three_same_mean):
        result = test(three_same_mean, value="value", group="group")
        assert result.test_name == "One-way ANOVA"
        assert result.p_value > 0.05
        assert result.posthoc is None

    def test_auto_posthoc_false(self, three_normal_equal_var):
        result = test(three_normal_equal_var, value="value", group="group", auto_posthoc=False)
        assert result.test_name == "One-way ANOVA"
        assert result.p_value < 0.05
        assert result.posthoc is None


class TestTwoWayDispatch:
    def test_twoway_anova(self, twoway_data):
        result = test(twoway_data, value="value", group=["genotype", "time"])
        assert "Two-way ANOVA" in result.test_name
        assert result.details is not None
        assert result.p_value < 0.05  # interaction should be significant


class TestAssumptionsAttached:
    def test_assumptions_dict_present(self, two_normal_equal_var):
        result = test(two_normal_equal_var, value="value", group="group")
        assert result.assumptions is not None
        assert result.assumptions["normality"]["method"] == "shapiro"
        assert result.assumptions["equal_variance"]["method"] == "levene"

    def test_assumptions_p_values_present(self, two_normal_equal_var):
        result = test(two_normal_equal_var, value="value", group="group")
        assert result.assumptions is not None
        assert "p" in result.assumptions["normality"]
        assert "p" in result.assumptions["equal_variance"]


class TestWideFormat:
    def test_wide_two_groups(self):
        df = pd.DataFrame({"wt": [1.0, 2.0, 3.0, 4.0, 5.0], "mut": [6.0, 7.0, 8.0, 9.0, 10.0]})
        result = test(df, format="wide", groups=["wt", "mut"])
        assert isinstance(result, TestResult)
        assert result.p_value < 0.05
