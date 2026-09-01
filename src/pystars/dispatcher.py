"""Dispatcher: walks the statistical-test flowchart and auto-selects the appropriate test."""

from __future__ import annotations

from typing import Literal

import pandas as pd

from pystars.assumptions import _check_equal_variance, _check_normality
from pystars.data import LongData, normalize_data
from pystars.posthoc import posthoc_dunn, posthoc_games_howell, posthoc_tukey
from pystars.result import TestResult
from pystars.tests_continuous import (
    anova,
    anova_twoway,
    kruskal,
    mannwhitney,
    ttest,
    wilcoxon,
)


def test(
    df: pd.DataFrame,
    *,
    value: str | None = None,
    group: str | list[str] | None = None,
    subject: str | None = None,
    paired: bool = False,
    format: Literal["long", "wide"] = "long",
    groups: list[str] | None = None,
    subject_index: str | None = None,
    alpha: float = 0.05,
    auto_posthoc: bool = True,
) -> TestResult:
    """Walk the flowchart and perform the appropriate test automatically.

    This is the main entry point of PyStars. Based on the data type (continuous
    in Phase 1), number of groups, pairing, normality, and equal-variance
    assumptions, it selects and runs the correct statistical test.

    For two independent groups, normal data uses Student's or Welch's t-test
    according to Levene's test, while non-normal data uses Mann-Whitney U.
    For two paired groups, normal differences use a paired t-test and
    non-normal differences use Wilcoxon signed-rank. For more than two groups,
    the corresponding choices are one-way ANOVA, Welch's ANOVA, or
    Kruskal-Wallis, with an appropriate post-hoc test when enabled. A list of
    at least two factor columns always selects two-way ANOVA, regardless of
    ``paired``.

    Groups with fewer than :data:`pystars.assumptions.SMALL_N` observations are
    treated as non-normal. The selected result records the assumption checks in
    ``result.assumptions``. ``auto_posthoc`` controls post-hoc execution only
    for the more-than-two-group and multi-factor branches, and post-hoc tests
    are run only when the omnibus p-value is less than ``alpha``.

    Parameters
    ----------
    df:
        Input dataframe (long or wide format).
    value:
        (Long) Name of the outcome column.
    group:
        (Long) Grouping column, or list of factor columns for a factorial design.
    subject:
        (Long) Subject/id column (required for paired tests).
    paired:
        Whether the samples are paired/matched.
    format:
        ``"long"`` (default) or ``"wide"``.
    groups:
        (Wide) Column names representing each group.
    subject_index:
        (Wide) Optional subject-id column.
    alpha:
        Significance level for assumption verdicts and post-hoc gating. The
        default is ``0.05``.
    auto_posthoc:
        Whether to automatically run post-hoc tests after a significant
        omnibus test in branches that support post-hoc comparisons. The
        default is ``True``.

    Returns
    -------
    TestResult
        Result of the selected test, with assumptions attached and (optionally)
        post-hoc comparisons.

    Raises
    ------
    ValueError
        If the input format or column names are invalid, or if the selected
        test requires a subject column or a specific number of groups that the
        data does not provide.

    Examples
    --------
    >>> result = test(df, value="length", group="treatment")
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

    # Multi-factor design → two-way ANOVA.
    if len(data.group_cols) >= 2:
        return _dispatch_twoway(data, alpha=alpha, auto_posthoc=auto_posthoc)

    n_groups = data.df["group"].nunique()
    if n_groups == 2:
        return _dispatch_two_groups(data, paired=paired, alpha=alpha)
    return _dispatch_many_groups(data, alpha=alpha, auto_posthoc=auto_posthoc)


# --------------------------------------------------------------- helpers


def _assumption_dict(result: dict) -> dict:
    return {
        "method": result["method"],
        "statistic": result["statistic"],
        "p": result["p"],
    }


def _dispatch_two_groups(data: LongData, *, paired: bool, alpha: float) -> TestResult:
    normality = _check_normality(data, paired=paired, alpha=alpha)
    assumptions: dict = {"normality": _assumption_dict(normality)}
    subj = data.subject_col  # "subject" or None

    if paired:
        if normality["passed"]:
            result = ttest(data.df, value="value", group="group", subject=subj, paired=True)
        else:
            result = wilcoxon(data.df, value="value", group="group", subject=subj)
    else:
        equal_var = _check_equal_variance(data, alpha=alpha)
        assumptions["equal_variance"] = _assumption_dict(equal_var)
        if normality["passed"]:
            welch = not equal_var["passed"]
            result = ttest(data.df, value="value", group="group", welch=welch)
        else:
            result = mannwhitney(data.df, value="value", group="group")

    result.assumptions = assumptions
    return result


def _dispatch_many_groups(data: LongData, *, alpha: float, auto_posthoc: bool) -> TestResult:
    normality = _check_normality(data, alpha=alpha)
    equal_var = _check_equal_variance(data, alpha=alpha)
    assumptions = {
        "normality": _assumption_dict(normality),
        "equal_variance": _assumption_dict(equal_var),
    }

    if normality["passed"]:
        if equal_var["passed"]:
            result = anova(data.df, value="value", group="group", welch=False)
            posthoc_fn = posthoc_tukey
        else:
            result = anova(data.df, value="value", group="group", welch=True)
            posthoc_fn = posthoc_games_howell
    else:
        result = kruskal(data.df, value="value", group="group")
        posthoc_fn = posthoc_dunn

    result.assumptions = assumptions

    if auto_posthoc and result.p_value < alpha:
        result.posthoc = posthoc_fn(data.df, value="value", group="group")

    return result


def _dispatch_twoway(data: LongData, *, alpha: float, auto_posthoc: bool) -> TestResult:
    normality = _check_normality(data, alpha=alpha)
    equal_var = _check_equal_variance(data, alpha=alpha)
    assumptions = {
        "normality": _assumption_dict(normality),
        "equal_variance": _assumption_dict(equal_var),
    }

    result = anova_twoway(data.df, value="value", group=data.group_cols)
    result.assumptions = assumptions

    if auto_posthoc and result.p_value < alpha:
        result.posthoc = posthoc_tukey(data.df, value="value", group="group")

    return result


# Prevent pytest from collecting the `test` function as a test.
test.__test__ = False
