"""Multiple-comparison correction helpers."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pingouin as pg

from pystars.result import TestResult

_METHODS = {
    "b",
    "bonf",
    "bonferroni",
    "s",
    "sidak",
    "h",
    "holm",
    "fdr",
    "fdr_bh",
    "bh",
    "fdr_by",
    "by",
    "none",
}


def adjust_pvalues(
    p_values: Iterable[float | int | None],
    *,
    method: str = "holm",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Correct a collection of p-values for multiple comparisons.

    Supported methods mirror :func:`pingouin.multicomp`: ``"bonf"``,
    ``"sidak"``, ``"holm"``, ``"fdr_bh"``, ``"fdr_by"``, and aliases.
    Missing values are preserved and are never marked as rejected.
    """
    method = _validate_method(method)
    alpha = _validate_alpha(alpha)
    pvals = _validate_pvalues(p_values)

    reject, p_adjusted = pg.multicomp(pvals, method=method, alpha=alpha)
    reject = [bool(r) if not pd.isna(p) else False for r, p in zip(reject, p_adjusted, strict=True)]

    return pd.DataFrame(
        {
            "p_value": pvals,
            "p_adjusted": p_adjusted,
            "reject": pd.Series(reject, dtype=object),
            "p_adjust_method": method,
            "p_adjust_alpha": alpha,
        }
    )


def adjust_results(
    results: Iterable[TestResult],
    *,
    method: str = "holm",
    alpha: float = 0.05,
    inplace: bool = False,
) -> list[TestResult]:
    """Attach corrected p-values to a list of :class:`TestResult` objects."""
    result_list = results if isinstance(results, list) else list(results)
    if not result_list:
        return []

    target = result_list if inplace else deepcopy(result_list)
    adjusted = adjust_pvalues([result.p_value for result in target], method=method, alpha=alpha)

    for result, row in zip(target, adjusted.to_dict("records"), strict=True):
        result.p_adjusted = float(row["p_adjusted"]) if not pd.isna(row["p_adjusted"]) else np.nan
        result.reject = bool(row["reject"])
        result.p_adjust_method = str(row["p_adjust_method"])
        result.p_adjust_alpha = float(row["p_adjust_alpha"])

    return target


def adjust_pairwise(
    pairwise: pd.DataFrame,
    *,
    p_col: str = "p",
    method: str = "holm",
    alpha: float = 0.05,
    inplace: bool = False,
) -> pd.DataFrame:
    """Return a pairwise comparison table with corrected p-values attached."""
    if p_col not in pairwise.columns:
        raise ValueError(f"p_col must name an existing p-value column, got {p_col!r}.")

    target = pairwise if inplace else pairwise.copy()
    adjusted = adjust_pvalues(target[p_col].to_list(), method=method, alpha=alpha)
    target["p_adjusted"] = adjusted["p_adjusted"].to_numpy()
    target["reject"] = adjusted["reject"].to_list()
    target["p_adjust_method"] = adjusted["p_adjust_method"].to_list()
    target["p_adjust_alpha"] = adjusted["p_adjust_alpha"].to_numpy()
    return target


def _validate_method(method: str) -> str:
    if not isinstance(method, str):
        raise ValueError("method must be a string.")
    method = method.lower()
    if method not in _METHODS:
        raise ValueError(f"method must be one of {sorted(_METHODS)}, got {method!r}.")
    return method


def _validate_alpha(alpha: float) -> float:
    if not isinstance(alpha, int | float) or isinstance(alpha, bool):
        raise ValueError("alpha must be a number between 0 and 1.")
    alpha = float(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    return alpha


def _validate_pvalues(
    p_values: Iterable[float | int | None],
) -> np.ndarray[Any, np.dtype[np.float64]]:
    try:
        pvals = np.asarray(list(p_values), dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("p-values must be numeric or NaN.") from exc

    valid = np.isnan(pvals) | ((0 <= pvals) & (pvals <= 1))
    if not valid.all():
        raise ValueError("p-values must be between 0 and 1, or NaN.")
    return pvals
