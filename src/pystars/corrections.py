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

    Parameters
    ----------
    p_values:
        Iterable of numeric p-values. ``None`` and ``NaN`` are treated as
        missing, preserved in the output, and never rejected.
    method:
        Multiple-comparison correction method. Supported names are
        ``"bonf"``, ``"bonferroni"``, ``"sidak"``, ``"holm"``,
        ``"fdr_bh"``, ``"fdr_by"``, their short aliases, and ``"none"``.
        Names are case-insensitive. The default is ``"holm"``.
    alpha:
        Significance threshold used to populate the ``reject`` column. Must
        be strictly between 0 and 1. The default is ``0.05``.

    Returns
    -------
    pandas.DataFrame
        One row per input p-value, preserving input order. The columns are
        ``p_value``, ``p_adjusted``, ``reject``, ``p_adjust_method``, and
        ``p_adjust_alpha``.

    Raises
    ------
    ValueError
        If ``method`` is unsupported, ``alpha`` is not strictly between 0 and
        1, or a p-value is not numeric or lies outside ``[0, 1]``.

    Examples
    --------
    >>> adjusted = adjust_pvalues([0.01, 0.03, 0.20], method="holm")
    >>> adjusted["p_adjusted"].tolist()
    [0.03, 0.06, 0.2]
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
    """Attach corrected p-values to several :class:`TestResult` objects.

    By default, the input results and their nested data are deep-copied before
    correction. Use ``inplace=True`` to update the supplied list and result
    objects directly.

    Parameters
    ----------
    results:
        Iterable of :class:`TestResult` objects. Any iterable is consumed into
        a list; an empty iterable returns an empty list.
    method:
        Multiple-comparison correction method accepted by
        :func:`adjust_pvalues`. The default is ``"holm"``.
    alpha:
        Significance threshold used for each result's ``reject`` field. Must
        be strictly between 0 and 1. The default is ``0.05``.
    inplace:
        If ``False`` (default), return corrected deep copies. If ``True``,
        mutate the supplied result objects when the input is a list.

    Returns
    -------
    list[TestResult]
        Results in the original order, with ``p_adjusted``, ``reject``,
        ``p_adjust_method``, and ``p_adjust_alpha`` populated.

    Raises
    ------
    ValueError
        If a correction method, alpha, or p-value is invalid. The input must
        also contain :class:`TestResult` objects with numeric p-values.

    Examples
    --------
    >>> adjusted = adjust_results(results, method="fdr_bh")
    >>> corrected_p_values = [result.p_adjusted for result in adjusted]
    """
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
    """Add multiple-comparison corrections to a pairwise table.

    The input table is expected to contain one p-value per comparison. All
    existing columns are preserved and four correction columns are appended.

    Parameters
    ----------
    pairwise:
        Pairwise comparison table as a pandas dataframe.
    p_col:
        Name of the column containing the p-values to correct. The default is
        ``"p"``.
    method:
        Multiple-comparison correction method accepted by
        :func:`adjust_pvalues`. The default is ``"holm"``.
    alpha:
        Significance threshold used for the ``reject`` column. Must be
        strictly between 0 and 1. The default is ``0.05``.
    inplace:
        If ``False`` (default), return a copy and leave ``pairwise`` unchanged.
        If ``True``, add the correction columns to the supplied dataframe and
        return it.

    Returns
    -------
    pandas.DataFrame
        The pairwise table with ``p_adjusted``, ``reject``,
        ``p_adjust_method``, and ``p_adjust_alpha`` columns.

    Raises
    ------
    ValueError
        If ``p_col`` is not an existing column or the correction inputs are
        invalid.

    Examples
    --------
    >>> adjusted = adjust_pairwise(pairwise, p_col="p", method="holm")
    >>> corrected_p_values = adjusted["p_adjusted"]
    """
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
