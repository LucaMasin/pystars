"""Tests for pystars.assumptions: normality and equal-variance checks."""

import numpy as np
import pandas as pd
import pytest

from pystars.assumptions import (
    _check_equal_variance,
    _check_normality,
    check_equal_variance,
    check_normality,
)
from pystars.data import normalize_data
from pystars.result import TestResult


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def normal_long(rng):
    """Long df with 2 normal groups, equal variance."""
    return pd.DataFrame(
        {
            "group": ["wt"] * 50 + ["mut"] * 50,
            "length": np.concatenate([rng.normal(10, 2, 50), rng.normal(12, 2, 50)]),
        }
    )


@pytest.fixture
def nonnormal_long(rng):
    """Long df with 2 groups drawn from exponential (non-normal)."""
    return pd.DataFrame(
        {
            "group": ["wt"] * 50 + ["mut"] * 50,
            "length": np.concatenate([rng.exponential(5, 50), rng.exponential(5, 50)]),
        }
    )


@pytest.fixture
def unequal_var_long(rng):
    """Long df with 2 normal groups, very different variances."""
    return pd.DataFrame(
        {
            "group": ["wt"] * 50 + ["mut"] * 50,
            "length": np.concatenate([rng.normal(10, 1, 50), rng.normal(10, 10, 50)]),
        }
    )


@pytest.fixture
def paired_long(rng):
    """Long df with paired measurements (same subject measured twice)."""
    subjects = [f"s{i}" for i in range(30)]
    baseline = rng.normal(10, 2, 30)
    treatment = baseline + rng.normal(2, 1, 30)
    return pd.DataFrame(
        {
            "group": ["pre"] * 30 + ["post"] * 30,
            "subject": subjects * 2,
            "length": np.concatenate([baseline, treatment]),
        }
    )


class TestNormalityInternal:
    def test_normal_data_not_rejected(self, normal_long):
        data = normalize_data(normal_long, value="length", group="group")
        result = _check_normality(data)
        assert result["method"] == "shapiro"
        assert result["p"] > 0.05
        assert result["passed"] is True
        assert "statistic" in result
        assert result["per_group"] is not None
        assert len(result["per_group"]) == 2

    def test_nonnormal_data_rejected(self, nonnormal_long):
        data = normalize_data(nonnormal_long, value="length", group="group")
        result = _check_normality(data)
        assert result["p"] < 0.05
        assert result["passed"] is False

    def test_paired_checks_differences(self, paired_long):
        data = normalize_data(paired_long, value="length", group="group", subject="subject")
        result = _check_normality(data, paired=True)
        assert result["per_group"] is None
        assert "statistic" in result
        assert "p" in result

    def test_paired_without_subject_raises(self, normal_long):
        data = normalize_data(normal_long, value="length", group="group")
        with pytest.raises(ValueError, match="subject"):
            _check_normality(data, paired=True)

    def test_small_n_treated_as_nonnormal(self, rng):
        """Very small samples (<3) should be treated as non-normal."""
        df = pd.DataFrame({"group": ["wt"] * 2 + ["mut"] * 2, "length": [1.0, 2.0, 3.0, 4.0]})
        data = normalize_data(df, value="length", group="group")
        result = _check_normality(data)
        assert result["passed"] is False


class TestEqualVarianceInternal:
    def test_equal_var_not_rejected(self, normal_long):
        data = normalize_data(normal_long, value="length", group="group")
        result = _check_equal_variance(data)
        assert result["method"] == "levene"
        assert result["p"] > 0.05
        assert result["passed"] is True

    def test_unequal_var_rejected(self, unequal_var_long):
        data = normalize_data(unequal_var_long, value="length", group="group")
        result = _check_equal_variance(data)
        assert result["p"] < 0.05
        assert result["passed"] is False


class TestPublicAPI:
    def test_check_normality_returns_test_result(self, normal_long):
        result = check_normality(normal_long, value="length", group="group")
        assert isinstance(result, TestResult)
        assert result.test_name == "Shapiro-Wilk normality test"
        assert result.p_value > 0.05

    def test_check_normality_nonnormal(self, nonnormal_long):
        result = check_normality(nonnormal_long, value="length", group="group")
        assert result.p_value < 0.05

    def test_check_equal_variance_returns_test_result(self, normal_long):
        result = check_equal_variance(normal_long, value="length", group="group")
        assert isinstance(result, TestResult)
        assert "Levene" in result.test_name or "levene" in result.test_name
        assert result.p_value > 0.05

    def test_check_equal_variance_unequal(self, unequal_var_long):
        result = check_equal_variance(unequal_var_long, value="length", group="group")
        assert result.p_value < 0.05

    def test_check_normality_alpha_parameter(self, normal_long):
        """With very low alpha, even normal data might be 'rejected'."""
        result = check_normality(normal_long, value="length", group="group", alpha=0.999)
        assert result.extra is not None
        # With alpha=0.999 almost everything is rejected
        assert result.extra["alpha"] == 0.999
