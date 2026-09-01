"""Assumption checks: normality (Shapiro-Wilk) and equal variance (Levene)."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from scipy import stats

from pystars.data import LongData, normalize_data, paired_differences
from pystars.result import TestResult

SMALL_N = 3  # Below this per group, normality is unreliable -> treat as non-normal.


def _check_normality(
    data: LongData, *, paired: bool = False, alpha: float = 0.05
) -> dict[str, Any]:
    """Run Shapiro-Wilk normality test.

    If ``paired`` is True, test the normality of paired differences (requires
    a subject column and exactly 2 groups). Otherwise test each group
    independently and report the worst-case (minimum) p-value.

    Samples smaller than ``SMALL_N`` are treated as non-normal.
    """
    if paired:
        diffs = paired_differences(data)
        if len(diffs) < SMALL_N:
            return {
                "method": "shapiro",
                "statistic": float("nan"),
                "p": 0.0,
                "passed": False,
                "per_group": None,
            }
        w, p = stats.shapiro(diffs)
        return {
            "method": "shapiro",
            "statistic": float(w),
            "p": float(p),
            "passed": bool(p > alpha),
            "per_group": None,
        }

    # Independent: test per group.
    sizes = data.df.groupby("group")["value"].size()
    if (sizes < SMALL_N).any():
        return {
            "method": "shapiro",
            "statistic": float("nan"),
            "p": 0.0,
            "passed": False,
            "per_group": _per_group_table(data),
        }

    rows = []
    for grp, sub in data.df.groupby("group"):
        w, p = stats.shapiro(sub["value"].to_numpy())
        rows.append({"group": grp, "W": float(w), "p": float(p), "normal": bool(p > alpha)})
    per_group = pd.DataFrame(rows)
    min_p = float(per_group["p"].min().item())
    min_w = float(per_group["W"].min().item())
    return {
        "method": "shapiro",
        "statistic": min_w,
        "p": min_p,
        "passed": bool(min_p > alpha),
        "per_group": per_group,
    }


def _check_equal_variance(data: LongData, *, alpha: float = 0.05) -> dict[str, Any]:
    """Run Levene's test for equal variance (median-centered, robust to non-normality)."""
    groups = [g["value"].to_numpy() for _, g in data.df.groupby("group")]
    if len(groups) < 2:
        raise ValueError("equal variance test requires at least 2 groups.")
    statistic, p = stats.levene(*groups, center="median")
    return {
        "method": "levene",
        "statistic": float(statistic),
        "p": float(p),
        "passed": bool(p > alpha),
    }


def _per_group_table(data: LongData) -> pd.DataFrame:
    rows = []
    for grp, _sub in data.df.groupby("group"):
        rows.append(
            {
                "group": grp,
                "W": float("nan"),
                "p": float("nan"),
                "normal": False,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- public


def check_normality(
    df: pd.DataFrame,
    *,
    value: str | None = None,
    group: str | list[str] | None = None,
    subject: str | None = None,
    paired: bool = False,
    alpha: float = 0.05,
    format: Literal["long", "wide"] = "long",
    groups: list[str] | None = None,
    subject_index: str | None = None,
) -> TestResult:
    """User-facing Shapiro-Wilk normality test.

    For independent data, Shapiro-Wilk is applied separately to every group
    and the smallest p-value is used for the returned result. For paired data,
    it is applied to the within-subject differences instead. Groups with fewer
    than :data:`SMALL_N` observations are treated as non-normal because the
    normality check is unreliable at that sample size.

    Parameters
    ----------
    df:
        Input observations as a pandas dataframe.
    value:
        For long format, the name of the numeric outcome column.
    group:
        For long format, the grouping column, or a list of factor columns.
    subject:
        For long format, the subject or matched-unit column. Required when
        ``paired=True``.
    paired:
        If ``True``, test normality of paired differences. The data must have
        a subject column and exactly two groups.
    alpha:
        Threshold used for the ``passed`` verdict stored in ``result.extra``.
        The default is ``0.05``.
    format:
        Input layout: ``"long"`` (default) or ``"wide"``.
    groups:
        For wide format, the value-column names representing the groups.
    subject_index:
        For wide format, an optional subject-id column. If omitted, row
        indices are used as subject labels.

    Returns
    -------
    TestResult
        A result named ``"Shapiro-Wilk normality test"``. Its ``p_value`` is
        the minimum per-group p-value for independent data, or the p-value of
        the paired differences. ``result.extra`` contains ``alpha``,
        ``passed``, and ``per_group``; independent results also expose the
        per-group table through ``result.details``.

    Raises
    ------
    ValueError
        If the input schema is invalid, or if paired data lacks a subject
        column or does not contain exactly two groups.

    Examples
    --------
    >>> result = check_normality(df, value="length", group="genotype")
    >>> 0 <= result.p_value <= 1
    True
    """
    data = normalize_data(
        df,
        value=value,
        group=group,
        subject=subject,
        format=format,
        groups=groups,
        subject_index=subject_index,
    )
    res = _check_normality(data, paired=paired, alpha=alpha)
    tr = TestResult(
        test_name="Shapiro-Wilk normality test",
        statistic=res["statistic"],
        p_value=res["p"],
        effect_size={},
        extra={"alpha": alpha, "per_group": res["per_group"], "passed": res["passed"]},
    )
    if res["per_group"] is not None:
        tr.details = res["per_group"]
    return tr


def check_equal_variance(
    df: pd.DataFrame,
    *,
    value: str | None = None,
    group: str | list[str] | None = None,
    alpha: float = 0.05,
    format: Literal["long", "wide"] = "long",
    groups: list[str] | None = None,
    subject_index: str | None = None,
) -> TestResult:
    """Run Levene's test for equal variance across independent groups.

    The test uses median-centered Levene's test, which is more robust to
    departures from normality than mean-centered Levene's test. The returned
    verdict is ``True`` when the p-value is greater than ``alpha``.

    Parameters
    ----------
    df:
        Input observations as a pandas dataframe.
    value:
        For long format, the name of the numeric outcome column.
    group:
        For long format, the grouping column, or a list of factor columns.
    alpha:
        Threshold used for the ``passed`` verdict stored in ``result.extra``.
        The default is ``0.05``.
    format:
        Input layout: ``"long"`` (default) or ``"wide"``.
    groups:
        For wide format, the value-column names representing the groups.
    subject_index:
        For wide format, an optional subject-id column. It is retained during
        normalization but is not used by this independent-groups test.

    Returns
    -------
    TestResult
        A result named ``"Levene's test for equal variance"`` with the Levene
        statistic and p-value. ``result.extra`` contains ``alpha`` and the
        boolean ``passed`` verdict.

    Raises
    ------
    ValueError
        If the input schema is invalid or fewer than two groups are present.

    Examples
    --------
    >>> result = check_equal_variance(df, value="length", group="genotype")
    >>> result.test_name
    "Levene's test for equal variance"
    """
    data = normalize_data(
        df,
        value=value,
        group=group,
        format=format,
        groups=groups,
        subject_index=subject_index,
    )
    res = _check_equal_variance(data, alpha=alpha)
    return TestResult(
        test_name="Levene's test for equal variance",
        statistic=res["statistic"],
        p_value=res["p"],
        effect_size={},
        extra={"alpha": alpha, "passed": res["passed"]},
    )
