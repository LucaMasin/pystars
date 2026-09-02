"""Plot annotations: add statistical significance annotations to an existing Matplotlib axes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Literal

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.text import Text

from pystars.result import TestResult

_ANNOTATION_GID = "pystars_annotation"
_VALID_MODES = ("stars", "pvalue", "value", "letters")
_VALID_BRACKETS = ("line", "square")

# Properties that move a bracket/label away from its computed geometry.
_RESERVED_LINE_KWS = {"xdata", "ydata", "data", "transform", "x", "y"}
_RESERVED_TEXT_KWS = {"x", "y", "s", "text", "transform", "data", "position", "gid"}
_FORWARDED_LINE_KWS = {"color", "linewidth", "linestyle", "alpha", "marker", "zorder"}
_FORWARDED_TEXT_KWS = {
    "color",
    "fontsize",
    "fontweight",
    "fontstyle",
    "alpha",
    "rotation",
    "ha",
    "va",
    "zorder",
}


# ============================================================== data records


@dataclass(frozen=True)
class _Comparison:
    left: object
    right: object
    p_value: float
    source: str
    ordinal: int


@dataclass(frozen=True)
class _ResolvedComparison:
    visual_left: object
    visual_right: object
    p_value: float
    source: str
    ordinal: int


@dataclass
class _GroupGeometry:
    x_center: float
    y_top: float  # display-space y of the highest rendered data artist


@dataclass(frozen=True)
class _ArtistGeometry:
    x_center: float
    y_top: float  # display-space y of the highest rendered extent
    color: tuple[float, float, float, float] | None
    kind: str


# ============================================================== public API


def annotate_significance(
    ax: Axes,
    result: TestResult,
    *,
    comparison_table: pd.DataFrame | None = None,
    comparisons: Literal["significant", "all"] | Sequence[tuple[object, object]] = ("significant"),
    groups: tuple[object, object] | None = None,
    label_map: Mapping[object, object] | None = None,
    mode: Literal["stars", "pvalue", "value", "letters"] = "stars",
    bracket: Literal["line", "square"] = "line",
    alpha: float = 0.05,
    p_column: str = "auto",
    p_decimals: int | None = None,
    color: str | None = "black",
    y_offset: float = 0.0,
    rc: Mapping[str, Any] | None = None,
    line_kws: Mapping[str, Any] | None = None,
    text_kws: Mapping[str, Any] | None = None,
) -> Axes:
    """Add significance annotations to an existing Matplotlib axes.

    This function is an annotation layer only: it reads the rendered bars,
    points, error bars, and legend on ``ax``, then draws significance labels on
    top of them. It does not plot raw data or recompute a statistical test.

    Comparison data is selected in this order:

    1. An explicitly supplied ``comparison_table``.
    2. ``result.pairwise``, followed recursively by ``result.posthoc``.
    3. A direct two-group comparison using the caller-supplied ``groups``.

    Pairwise tables must contain ``A`` and ``B`` columns and a p-value column.
    With ``p_column="auto"``, a non-null ``p_adjusted`` column is preferred
    over ``p``. Pair direction is ignored, so ``(A, B)`` and ``(B, A)`` are
    the same comparison. Duplicate pairs and cycles in nested post-hoc results
    are rejected.

    Parameters
    ----------
    ax:
        Existing two-dimensional Cartesian Matplotlib axes containing a
        vertical categorical plot. The same axes object is returned.
    result:
        :class:`~pystars.result.TestResult` containing a pairwise table,
        nested post-hoc results, or a direct two-group p-value.
    comparison_table:
        Optional explicit pairwise dataframe. It overrides every pairwise
        table attached to ``result`` and must contain ``A``, ``B``, and the
        selected p-value column.
    comparisons:
        Which comparisons to draw. ``"significant"`` (default) keeps rows
        with ``p <= alpha``; ``"all"`` draws every row; a sequence of
        two-tuples draws exactly the requested pairs.
    groups:
        Two source-group labels for a direct two-group ``TestResult`` that has
        no pairwise table. ``TestResult`` does not retain the source labels, so
        this argument is required in that case. The p-value uses a finite
        ``result.p_adjusted`` when available, otherwise ``result.p_value``.
    label_map:
        Optional mapping from statistical group labels to displayed x labels.
        Scalar values target one x tick. Two-item tuple values of the form
        ``(x_category, hue_category)`` target a particular dodged cell in a
        hue plot. Tuple mappings require a visible legend and mappings for all
        selected comparison endpoints.
    mode:
        Label format: ``"stars"`` (default), ``"pvalue"`` or ``"value"`` for
        the formatted p-value, or ``"letters"`` for a compact-letter display.
        Star labels use ``*`` for
        ``p <= 0.05``, ``**`` for ``p <= 0.01``, ``***`` for ``p <= 0.001``,
        and ``****`` for ``p <= 0.0001``; larger p-values use ``"ns"``.
    bracket:
        Bracket shape: ``"line"`` (default) draws a horizontal line, while
        ``"square"`` adds downward caps at both ends.
    alpha:
        Significance threshold used by ``comparisons="significant"`` and for
        compact-letter grouping. The default is ``0.05``.
    p_column:
        P-value column to read from pairwise tables. ``"auto"`` prefers
        ``p_adjusted`` when it has at least one non-null value, then falls back
        to ``p``. Pass a column name to select it explicitly.
    p_decimals:
        Optional fixed number of decimal places for ``"pvalue"`` and
        ``"value"`` labels. It must be an integer of at least 1. Values that
        would round to zero are shown with a ``<`` threshold instead. Ignored
        in ``"stars"`` and ``"letters"`` modes.
    color:
        Color for brackets and labels. The default is ``"black"``; ``None``
        leaves the artist default in place unless overridden in the style
        mappings.
    y_offset:
        Signed offset in display points applied uniformly to the complete
        annotation stack. Positive values move annotations upward and negative
        values move them downward, including on log-scaled or inverted axes.
    rc:
        Optional Matplotlib rc-parameter mapping scoped to this call. It is
        applied with :func:`matplotlib.rc_context` and does not mutate global
        ``rcParams``.
    line_kws:
        Optional Matplotlib ``Line2D`` properties for brackets. Geometry and
        transform properties are reserved and cannot be overridden.
    text_kws:
        Optional Matplotlib ``Text`` properties for labels. Position,
        transform, geometry, and annotation ``gid`` properties are reserved.

    Returns
    -------
    matplotlib.axes.Axes
        The same ``ax`` object, allowing calls to be chained with other
        Matplotlib operations.

    Raises
    ------
    ValueError
        If the axes, options, comparison schema, p-values, group labels, or
        pairwise graph are invalid; if a direct result lacks ``groups``; or if
        rendered plot geometry cannot resolve a requested category. This also
        includes a ``p_decimals`` value that is not an integer of at least 1.

    Notes
    -----
    Compact-letter mode requires ``comparisons="all"`` and a complete
    undirected pairwise graph. Artist geometry is resolved after drawing the
    canvas, and annotations are tagged so a later call does not mistake them
    for data artists.

    Examples
    --------
    >>> ax = df.plot.bar(x="group", y="length")
    >>> annotate_significance(ax, result, groups=("control", "treated"))
    """
    _validate_options(
        ax=ax,
        mode=mode,
        bracket=bracket,
        alpha=alpha,
        p_column=p_column,
        p_decimals=p_decimals,
        label_map=label_map,
        y_offset=y_offset,
        rc=rc,
        line_kws=line_kws,
        text_kws=text_kws,
    )
    if mode == "letters" and comparisons != "all":
        raise ValueError(
            "mode='letters' requires comparisons='all' because compact-letter "
            "display needs every pairwise relationship."
        )

    with mpl.rc_context(rc=rc):
        records = _gather_records(result, comparison_table=comparison_table, p_column=p_column)
        if records is None:
            if groups is None:
                raise ValueError(
                    "No pairwise comparison table is available on the result. "
                    "Direct two-group TestResult objects do not retain their "
                    "source-group names; provide groups=(left, right) or pass "
                    "an explicit comparison_table."
                )
            _validate_groups(groups)
            p_value = _direct_two_group_pvalue(result)
            records = [
                _Comparison(
                    left=groups[0],
                    right=groups[1],
                    p_value=p_value,
                    source="direct",
                    ordinal=0,
                )
            ]
        selected = _select_comparisons(records, comparisons=comparisons, alpha=alpha)
        if not selected:
            return ax
        resolved = _resolve_visual_labels(selected, label_map)

        ax.figure.canvas.draw()
        geometry = _resolve_positions(ax, resolved)
        y_offset_display = float(y_offset) * ax.figure.dpi / 72.0

        if mode == "letters":
            _draw_compact_letters(
                ax,
                resolved,
                geometry,
                alpha=alpha,
                color=color,
                y_offset_display=y_offset_display,
                text_kws=text_kws,
            )
        else:
            _draw_brackets(
                ax,
                resolved,
                geometry,
                mode=mode,
                bracket=bracket,
                color=color,
                y_offset_display=y_offset_display,
                p_decimals=p_decimals,
                line_kws=line_kws,
                text_kws=text_kws,
            )
    return ax


# ============================================================== validation


def _validate_options(
    *,
    ax: Any,
    mode: Any,
    bracket: Any,
    alpha: Any,
    p_column: Any,
    p_decimals: Any,
    label_map: Any,
    y_offset: Any,
    rc: Any,
    line_kws: Any,
    text_kws: Any,
) -> None:
    if not isinstance(ax, Axes):
        raise ValueError("ax must be a matplotlib.axes.Axes instance.")
    if getattr(ax, "name", "") in {"polar", "3d"}:
        raise ValueError(
            "annotate_significance only supports 2-D Cartesian axes "
            "(got a non-Cartesian projection)."
        )
    try:
        ax.transData.inverted()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            "annotate_significance only supports rectilinear 2-D Cartesian axes."
        ) from exc
    for container in ax.containers:
        if getattr(container, "orientation", "vertical") == "horizontal":
            raise ValueError(
                "Only vertical categorical plots are supported. Detected a "
                "horizontal BarContainer on the axes."
            )

    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES!r}, got {mode!r}.")
    if bracket not in _VALID_BRACKETS:
        raise ValueError(f"bracket must be one of {_VALID_BRACKETS!r}, got {bracket!r}.")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a finite real number strictly between 0 and 1.")
    alpha_f = float(alpha)
    if not (np.isfinite(alpha_f)) or not (0.0 < alpha_f < 1.0):
        raise ValueError("alpha must be a finite real number strictly between 0 and 1.")
    if not isinstance(p_column, str) or p_column == "":
        raise ValueError("p_column must be 'auto' or a non-empty string.")
    if p_decimals is not None and (
        isinstance(p_decimals, bool) or not isinstance(p_decimals, int) or p_decimals < 1
    ):
        raise ValueError("p_decimals must be None or an integer of at least 1.")
    if isinstance(y_offset, bool) or not isinstance(y_offset, Real):
        raise ValueError("y_offset must be a finite real number (display points).")
    y_offset_f = float(y_offset)
    if not np.isfinite(y_offset_f):
        raise ValueError("y_offset must be a finite real number (display points).")
    for name, value in (
        ("label_map", label_map),
        ("rc", rc),
        ("line_kws", line_kws),
        ("text_kws", text_kws),
    ):
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"{name} must be a mapping when provided.")


# ============================================================== comparison extraction


def _gather_records(
    result: TestResult,
    *,
    comparison_table: pd.DataFrame | None,
    p_column: str,
) -> list[_Comparison] | None:
    if comparison_table is not None:
        return _normalise_pairwise_table(
            comparison_table, p_column=p_column, source="comparison_table"
        )
    return _extract_from_result(result, p_column=p_column)


def _extract_from_result(
    result: TestResult,
    *,
    p_column: str,
) -> list[_Comparison] | None:
    records: list[_Comparison] = []
    seen: set[frozenset[object]] = set()
    has_table = False
    ordinal = 0
    visited: set[int] = set()

    def _add_from_table(table: pd.DataFrame, source: str) -> None:
        nonlocal ordinal, has_table
        new_records = _normalise_pairwise_table(table, p_column=p_column, source=source)
        has_table = True
        for rec in new_records:
            key = _pair_key(rec.left, rec.right)
            if key in seen:
                raise ValueError(
                    f"Duplicate unordered pair ({rec.left!r}, {rec.right!r}) "
                    "found across comparison sources. Pass comparison_table to "
                    "select a single post-hoc procedure."
                )
            seen.add(key)
            records.append(
                _Comparison(
                    left=rec.left,
                    right=rec.right,
                    p_value=rec.p_value,
                    source=rec.source,
                    ordinal=ordinal,
                )
            )
            ordinal += 1

    def _walk(node: Any, source: str) -> None:
        if id(node) in visited:
            raise ValueError("Cycle detected in TestResult.posthoc traversal.")
        if isinstance(node, TestResult):
            visited.add(id(node))
            if node.pairwise is not None and not isinstance(node.pairwise, pd.DataFrame):
                raise ValueError("TestResult.pairwise must be a pandas DataFrame or None.")
            if node.pairwise is not None:
                _add_from_table(node.pairwise, source=node.test_name or source)
            if node.posthoc is not None:
                _walk(node.posthoc, source=node.test_name or source)
        elif isinstance(node, list):
            for item in node:
                if not isinstance(item, TestResult):
                    raise ValueError(
                        "TestResult.posthoc must be a TestResult or a list of "
                        f"TestResult, got item of type {type(item).__name__}."
                    )
                _walk(item, source=source)
        else:
            raise ValueError(
                "TestResult.posthoc must be a TestResult or a list of "
                f"TestResult, got {type(node).__name__}."
            )

    if result.pairwise is not None and not isinstance(result.pairwise, pd.DataFrame):
        raise ValueError("TestResult.pairwise must be a pandas DataFrame or None.")
    if result.pairwise is not None:
        _add_from_table(result.pairwise, source=result.test_name or "primary")
    if result.posthoc is not None:
        _walk(result.posthoc, source=result.test_name or "primary")

    if not has_table:
        return None
    return records


def _normalise_pairwise_table(
    table: pd.DataFrame,
    *,
    p_column: str,
    source: str,
) -> list[_Comparison]:
    if not isinstance(table, pd.DataFrame):
        raise ValueError("comparison_table must be a pandas DataFrame.")
    if table.columns.duplicated().any():
        raise ValueError("comparison_table must have unique column names.")
    for col in ("A", "B"):
        if col not in table.columns:
            raise ValueError(f"comparison_table must contain column {col!r}.")
    if p_column == "auto":
        if "p_adjusted" in table.columns and table["p_adjusted"].notna().any():
            chosen = "p_adjusted"
        else:
            chosen = "p"
    else:
        if p_column not in table.columns:
            raise ValueError(f"Requested p_column={p_column!r} is missing from {source!r}.")
        chosen = p_column

    out: list[_Comparison] = []
    seen_pairs: set[frozenset[object]] = set()
    for idx, row in table.iterrows():
        a = row["A"]
        b = row["B"]
        if pd.isna(a) or pd.isna(b):
            raise ValueError(f"comparison_table row {idx} has null A/B label.")
        if not _is_hashable(a) or not _is_hashable(b):
            raise ValueError(f"comparison_table row {idx} has non-hashable A/B label.")
        if a == b:
            raise ValueError(f"comparison_table row {idx} compares a group with itself: {a!r}.")
        key = _pair_key(a, b)
        if key in seen_pairs:
            raise ValueError(
                f"comparison_table contains a duplicate unordered pair ({a!r}, {b!r}) at row {idx}."
            )
        seen_pairs.add(key)
        try:
            p = float(row[chosen])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"comparison_table row {idx} has non-numeric value in {chosen!r}."
            ) from exc
        if not np.isfinite(p):
            raise ValueError(f"comparison_table row {idx} has non-finite p-value in {chosen!r}.")
        if not (0.0 <= p <= 1.0):
            raise ValueError(
                f"comparison_table row {idx} has p-value {p} outside [0, 1] in {chosen!r}."
            )
        out.append(
            _Comparison(
                left=a,
                right=b,
                p_value=p,
                source=source,
                ordinal=len(out),
            )
        )
    return out


def _is_hashable(x: Any) -> bool:
    try:
        hash(x)
    except TypeError:
        return False
    return True


def _pair_key(left: object, right: object) -> frozenset[object]:
    """Normalize (a, b) and (b, a) to a single key."""
    return frozenset((left, right))


def _direct_two_group_pvalue(result: TestResult) -> float:
    for attr in ("p_adjusted", "p_value"):
        val = getattr(result, attr)
        if val is None:
            continue
        try:
            p = float(val)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"TestResult.{attr} is not numeric: {val!r}.") from exc
        if np.isfinite(p) and 0.0 <= p <= 1.0:
            return p
    raise ValueError(
        "Direct two-group result has no usable p_value or p_adjusted. "
        "Provide a finite value in [0, 1]."
    )


def _validate_groups(groups: tuple[object, object]) -> None:
    if not isinstance(groups, tuple) or len(groups) != 2:
        raise ValueError("groups must be a 2-tuple of two distinct, hashable labels.")
    if groups[0] == groups[1]:
        raise ValueError(f"groups must contain two distinct labels, got {groups[0]!r} twice.")
    for g in groups:
        if not _is_hashable(g):
            raise ValueError(f"groups entries must be hashable, got {g!r}.")


# ============================================================== selection


def _select_comparisons(
    records: list[_Comparison],
    *,
    comparisons: Any,
    alpha: float,
) -> list[_Comparison]:
    if comparisons == "significant":
        return [r for r in records if r.p_value <= alpha]
    if comparisons == "all":
        return list(records)
    if isinstance(comparisons, str):
        raise ValueError(
            f"comparisons must be 'significant', 'all', or a sequence of pairs; "
            f"got string {comparisons!r}."
        )
    if not hasattr(comparisons, "__iter__"):
        raise ValueError("comparisons must be 'significant', 'all', or a sequence of pairs.")
    requested = list(comparisons)
    available = {(r.left, r.right) for r in records} | {(r.right, r.left) for r in records}
    for pair in requested:
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise ValueError(f"Each comparison pair must be a 2-tuple, got {pair!r}.")
        if pair not in available and (pair[1], pair[0]) not in available:
            raise ValueError(
                f"Requested pair {pair!r} is not present in the available comparisons."
            )
    requested_set: set[tuple[object, object]] = set()
    for pair in requested:
        requested_set.add(pair)
        requested_set.add((pair[1], pair[0]))
    out: list[_Comparison] = []
    for r in records:
        if (r.left, r.right) in requested_set:
            out.append(r)
    return out


# ============================================================== label map


def _resolve_visual_labels(
    selected: list[_Comparison],
    label_map: Mapping[object, object] | None,
) -> list[_ResolvedComparison]:
    if not selected:
        return []
    mapping = dict(label_map) if label_map is not None else {}

    # If any label_map value is a tuple, every selected comparison must have
    # both ends in the label_map; otherwise the resolver cannot produce
    # consistent tuple targets for all brackets.
    has_tuple_mapping = any(_classify_label(v) == "tuple" for v in mapping.values())
    if has_tuple_mapping:
        missing: list[tuple[object, object]] = []
        for rec in selected:
            if rec.left not in mapping or rec.right not in mapping:
                missing.append((rec.left, rec.right))
        if missing:
            raise ValueError(
                "label_map contains tuple entries but some selected "
                "comparisons have no mapping. Either add the missing entries "
                "to label_map, restrict comparisons to the mapped pairs, or "
                "use only scalar label_map values. Missing: "
                f"{missing!r}."
            )

    resolved: list[_ResolvedComparison] = []
    seen_visuals: set[tuple[object, object]] = set()
    scalar_or_tuple: str | None = None
    for rec in selected:
        left = mapping.get(rec.left, rec.left)
        right = mapping.get(rec.right, rec.right)
        l_kind = _classify_label(left)
        r_kind = _classify_label(right)
        if l_kind != r_kind:
            raise ValueError(
                f"Inconsistent label_map shapes for comparison "
                f"({rec.left!r}, {rec.right!r}): mixed scalar and tuple targets."
            )
        if scalar_or_tuple is None:
            scalar_or_tuple = l_kind
        elif l_kind != scalar_or_tuple:
            raise ValueError(
                "All label_map targets must be consistently scalar or consistently two-item tuples."
            )
        if l_kind == "tuple":
            if len(left) != 2 or len(right) != 2:  # type: ignore[arg-type]
                raise ValueError(
                    f"Tuple label_map targets must be length 2; got {left!r} and {right!r}."
                )
            visual_left = tuple(left)  # type: ignore[arg-type]
            visual_right = tuple(right)  # type: ignore[arg-type]
        else:
            if left == right:
                raise ValueError(
                    f"label_map maps both {rec.left!r} and {rec.right!r} to "
                    f"the same visual label {left!r}."
                )
            visual_left = left
            visual_right = right
        key = (visual_left, visual_right)
        if key in seen_visuals:
            raise ValueError(
                f"Two statistical comparisons map to the same visual target "
                f"{key!r}; provide distinct label_map entries."
            )
        seen_visuals.add(key)
        resolved.append(
            _ResolvedComparison(
                visual_left=visual_left,
                visual_right=visual_right,
                p_value=rec.p_value,
                source=rec.source,
                ordinal=rec.ordinal,
            )
        )
    return resolved


def _classify_label(label: object) -> str:
    if isinstance(label, tuple) and not isinstance(label, str):
        return "tuple"
    return "scalar"


# ============================================================== position resolution


def _resolve_positions(
    ax: Axes,
    resolved: list[_ResolvedComparison],
) -> dict[object, _GroupGeometry]:
    tick_texts = [t.get_text() for t in ax.get_xticklabels()]
    if not tick_texts:
        raise ValueError(
            "Could not resolve categorical labels from x ticks. The axes must "
            "show visible category labels."
        )
    if len(set(tick_texts)) != len(tick_texts):
        raise ValueError("xticklabels are not unique; cannot resolve category labels.")
    tick_positions = list(ax.get_xticks())
    n = min(len(tick_positions), len(tick_texts))
    tick_positions = tick_positions[:n]
    tick_texts = tick_texts[:n]

    needs_hue = any(_classify_label(r.visual_left) == "tuple" for r in resolved)
    legend_lookup: dict[str, tuple[float, float, float, float]] | None = None
    if needs_hue:
        legend_lookup = _collect_legend_colors(ax)
        if not legend_lookup:
            raise ValueError("Hue resolution requires a visible legend on the axes.")

    bars = _collect_bar_geometry(ax)
    summaries = _collect_summary_geometry(ax)
    points = _collect_point_geometry(ax)
    errorbars = _collect_errorbar_geometry(ax)
    spacing = _category_spacing(bars, points, tick_positions)

    geom: dict[object, _GroupGeometry] = {}
    for rec in resolved:
        for visual in (rec.visual_left, rec.visual_right):
            if visual in geom:
                continue
            geom[visual] = _resolve_one(
                ax,
                visual,
                tick_texts,
                tick_positions,
                bars,
                summaries,
                points,
                errorbars,
                legend_lookup,
                spacing,
            )
    return geom


def _category_spacing(
    bars: list[_ArtistGeometry],
    points: list[_ArtistGeometry],
    tick_positions: list[float],
) -> float:
    """Tolerance (in data x) for assigning a bar/point to a category tick.

    Uses the local tick spacing when several tick positions are available so
    that dodged artists (which may be offset up to half a unit from a tick)
    are still matched to the correct category.
    """
    tick_set = sorted(set(round(t, 9) for t in tick_positions))
    if len(tick_set) >= 2:
        diffs = [tick_set[i + 1] - tick_set[i] for i in range(len(tick_set) - 1)]
        tick_spacing = min(d for d in diffs if d > 0) if any(d > 0 for d in diffs) else 1.0
    else:
        tick_spacing = 1.0
    # Use half the local tick spacing as the tolerance.
    return max(tick_spacing * 0.5, 0.5)


def _resolve_one(
    ax: Axes,
    target: object,
    tick_texts: list[str],
    tick_positions: list[float],
    bars: list[_ArtistGeometry],
    summaries: list[_ArtistGeometry],
    points: list[_ArtistGeometry],
    errorbars: list[_ArtistGeometry],
    legend_lookup: dict[str, tuple[float, float, float, float]] | None,
    spacing: float,
) -> _GroupGeometry:
    if _classify_label(target) == "tuple":
        assert isinstance(target, tuple)
        x_label, hue_label = target
        if x_label not in tick_texts:
            raise ValueError(
                f"Unknown x category {x_label!r}; displayed xticklabels are "
                f"{tick_texts!r}. Use label_map to map statistical identifiers "
                "to displayed categories."
            )
        if legend_lookup is None or hue_label not in legend_lookup:
            raise ValueError(
                f"Unknown hue label {hue_label!r}; available legend labels are "
                f"{list(legend_lookup or {})!r}. Use label_map of the form "
                "(x_category, hue_category) for dodged plots."
            )
        x_center_tick = tick_positions[tick_texts.index(x_label)]
        hue_color = legend_lookup[hue_label]
        x_center = _resolve_dodged_anchor(
            x_center_tick,
            bars,
            summaries,
            errorbars,
            points,
            hue_color,
            spacing,
        )
        y_top = _resolve_y_top(
            x_center_tick,
            x_center,
            bars,
            summaries,
            errorbars,
            points,
            hue_color,
            spacing,
        )
    else:
        if target not in tick_texts:
            raise ValueError(
                f"Unknown x category {target!r}; displayed xticklabels are "
                f"{tick_texts!r}. Use label_map to map statistical identifiers "
                "to displayed categories."
            )
        x_center_tick = tick_positions[tick_texts.index(target)]
        x_center = x_center_tick
        y_top = _resolve_y_top(
            x_center_tick,
            x_center,
            bars,
            summaries,
            errorbars,
            points,
            None,
            spacing,
        )
    return _GroupGeometry(x_center=x_center, y_top=y_top)


def _resolve_dodged_anchor(
    x_center_tick: float,
    bars: list[_ArtistGeometry],
    summaries: list[_ArtistGeometry],
    errorbars: list[_ArtistGeometry],
    points: list[_ArtistGeometry],
    hue_color: tuple[float, float, float, float],
    spacing: float,
) -> float:
    for kind, candidates in (
        ("bar", bars),
        ("summary", summaries),
        ("errorbar", errorbars),
        ("points", points),
    ):
        near = [artist for artist in candidates if abs(artist.x_center - x_center_tick) <= spacing]
        if not near:
            continue
        color_matches = [artist for artist in near if _color_close(artist.color, hue_color)]
        # If an artist has no usable color, its x position is the only
        # available discriminator. Never use a different known hue silently.
        matches = color_matches or [artist for artist in near if artist.color is None]
        if not matches:
            continue
        unique = {round(artist.x_center, 9) for artist in matches}
        if len(unique) > 1:
            kind_label = {
                "bar": "bar",
                "summary": "summary-marker",
                "errorbar": "error-bar",
                "points": "point-collection",
            }[kind]
            raise ValueError(
                f"Multiple {kind_label} candidates match the requested hue near "
                f"x={x_center_tick}; cannot resolve a unique dodged position."
            )
        return float(matches[0].x_center)
    raise ValueError(
        f"Could not find a rendered bar, summary marker, error bar, or point "
        f"near x={x_center_tick} for the requested hue. Pass a label_map entry "
        "that matches an existing dodged cell."
    )


def _resolve_y_top(
    x_center_tick: float,
    x_center: float,
    bars: list[_ArtistGeometry],
    summaries: list[_ArtistGeometry],
    errorbars: list[_ArtistGeometry],
    points: list[_ArtistGeometry],
    hue_color: tuple[float, float, float, float] | None,
    spacing: float,
) -> float:
    tops: list[float] = []
    for artist in (*bars, *summaries, *errorbars, *points):
        if abs(artist.x_center - x_center_tick) > spacing:
            continue
        if hue_color is None:
            matches = True
        else:
            matches = _color_close(artist.color, hue_color)
            # Some error-bar artists use a neutral color. Their exact x position
            # is sufficient to associate them with the selected dodged cell.
            if not matches and artist.kind == "errorbar":
                matches = bool(np.isclose(artist.x_center, x_center, rtol=0.0, atol=1e-9))
            if not matches and artist.color is None:
                matches = bool(np.isclose(artist.x_center, x_center, rtol=0.0, atol=1e-9))
        if matches and not np.isnan(artist.y_top):
            tops.append(artist.y_top)
    if not tops:
        return float("nan")
    return max(tops)


def _color_close(
    a: tuple[float, float, float, float] | None,
    b: tuple[float, float, float, float] | None,
) -> bool:
    if a is None or b is None:
        return False
    return all(abs(x - y) < 1e-3 for x, y in zip(a, b, strict=False))


def _collect_bar_geometry(ax: Axes) -> list[_ArtistGeometry]:
    """Collect centers and visual tops for vertical bars."""
    trans = ax.transData
    out: list[_ArtistGeometry] = []
    for container in ax.containers:
        if getattr(container, "orientation", "vertical") != "vertical":
            continue
        if getattr(container, "get_gid", lambda: None)() == _ANNOTATION_GID:
            continue
        for patch in container:
            if not isinstance(patch, Rectangle):
                continue
            if getattr(patch, "get_gid", lambda: None)() == _ANNOTATION_GID:
                continue
            x = patch.get_x()
            w = patch.get_width()
            y0 = patch.get_y()
            center = x + w / 2
            tops_disp = _max_visual_y(trans, [(0.0, y0), (0.0, y0 + patch.get_height())])
            face = patch.get_facecolor()
            edge = patch.get_edgecolor()
            out.append(
                _ArtistGeometry(
                    x_center=center,
                    y_top=tops_disp,
                    color=_preferred_color(face, edge),
                    kind="bar",
                )
            )
    return out


def _collect_summary_geometry(ax: Axes) -> list[_ArtistGeometry]:
    """Collect finite marker locations from point-estimate lines."""
    trans = ax.transData
    out: list[_ArtistGeometry] = []
    for line in ax.lines:
        if getattr(line, "get_gid", lambda: None)() == _ANNOTATION_GID:
            continue
        if not _is_summary_marker(line):
            continue
        try:
            xs = np.asarray(line.get_xdata(orig=False), dtype=float)
            ys = np.asarray(line.get_ydata(orig=False), dtype=float)
        except (TypeError, ValueError):
            continue
        if len(xs) == 0 or len(xs) != len(ys):
            continue
        color = _preferred_color(
            line.get_markerfacecolor(),
            line.get_markeredgecolor(),
            line.get_color(),
        )
        for x, y in zip(xs, ys, strict=False):
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            out.append(
                _ArtistGeometry(
                    x_center=float(x),
                    y_top=_max_visual_y(trans, [(float(x), float(y))]),
                    color=color,
                    kind="summary",
                )
            )
    return out


def _collect_point_geometry(ax: Axes) -> list[_ArtistGeometry]:
    """Collect centers and visual tops for raw point collections."""
    trans = ax.transData
    out: list[_ArtistGeometry] = []
    for coll in ax.collections:
        if not isinstance(coll, PathCollection):
            continue
        if getattr(coll, "get_gid", lambda: None)() == _ANNOTATION_GID:
            continue
        offsets = coll.get_offsets()
        if offsets is None or len(offsets) == 0:
            continue
        offsets_array = np.ma.asarray(offsets, dtype=float).filled(np.nan)
        if offsets_array.ndim != 2 or offsets_array.shape[1] != 2:
            continue
        finite = np.isfinite(offsets_array).all(axis=1)
        if not finite.any():
            continue
        facecolors = coll.get_facecolor()
        if facecolors is None or len(facecolors) == 0:
            continue
        first = tuple(facecolors[0])
        for fc in facecolors:
            if not _color_close(tuple(fc), first):
                raise ValueError(
                    "annotate_significance requires a uniform color per "
                    "PathCollection; found a multicolored point collection. "
                    "Pass a label_map of (x, hue) tuples to target a single "
                    "hue."
                )
        finite_offsets = offsets_array[finite]
        xs = finite_offsets[:, 0]
        rounded_xs = np.round(xs, 9)
        unique_xs, counts = np.unique(rounded_xs, return_counts=True)
        if counts.max() >= 2:
            mode_xs = unique_xs[counts == counts.max()]
            x_center = float(mode_xs[np.argmin(np.abs(mode_xs - np.median(xs)))])
        else:
            x_center = float(np.median(xs))
        tops = _max_visual_y(trans, [(float(x), float(y)) for x, y in finite_offsets])
        out.append(
            _ArtistGeometry(
                x_center=x_center,
                y_top=tops,
                color=_to_rgba(first),
                kind="points",
            )
        )
    return out


def _collect_errorbar_geometry(ax: Axes) -> list[_ArtistGeometry]:
    """Collect vertical error-bar stems."""
    trans = ax.transData
    out: list[_ArtistGeometry] = []
    for collection in ax.collections:
        if not isinstance(collection, LineCollection):
            continue
        if getattr(collection, "get_gid", lambda: None)() == _ANNOTATION_GID:
            continue
        colors = collection.get_colors()
        for segment_index, segment in enumerate(collection.get_segments()):
            arr = np.asarray(segment, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 2:
                continue
            if not np.isfinite(arr).all() or not np.allclose(arr[:, 0], arr[0, 0]):
                continue
            color = None
            if colors is not None and len(colors) > 0:
                color = _to_rgba(colors[min(segment_index, len(colors) - 1)])
            out.append(
                _ArtistGeometry(
                    x_center=float(arr[0, 0]),
                    y_top=_max_visual_y(trans, [tuple(p) for p in arr]),
                    color=color,
                    kind="errorbar",
                )
            )
    for line in ax.lines:
        if getattr(line, "get_gid", lambda: None)() == _ANNOTATION_GID:
            continue
        if _has_marker(line):
            continue
        try:
            xs = np.asarray(line.get_xdata(orig=False), dtype=float)
            ys = np.asarray(line.get_ydata(orig=False), dtype=float)
        except (TypeError, ValueError):
            continue
        if len(xs) < 2 or len(xs) != len(ys):
            continue
        finite = np.isfinite(xs) & np.isfinite(ys)
        start: int | None = None
        for index, is_finite in enumerate(np.r_[finite, False]):
            if is_finite and start is None:
                start = index
            elif not is_finite and start is not None:
                _append_errorbar_run(out, trans, xs[start:index], ys[start:index], line)
                start = None
    return out


def _append_errorbar_run(
    out: list[_ArtistGeometry],
    trans: Any,
    xs: np.ndarray,
    ys: np.ndarray,
    line: Line2D,
) -> None:
    if len(xs) < 2:
        return
    points = [(float(x), float(y)) for x, y in zip(xs, ys, strict=False)]
    color = _preferred_color(line.get_color())
    if np.allclose(xs, xs[0]):
        out.append(
            _ArtistGeometry(
                x_center=float(xs[0]),
                y_top=_max_visual_y(trans, points),
                color=color,
                kind="errorbar",
            )
        )


def _is_summary_marker(line: Line2D) -> bool:
    marker = line.get_marker()
    return marker not in (None, "", "None", "none", " ", "_", "|")


def _has_marker(line: Line2D) -> bool:
    marker = line.get_marker()
    return marker not in (None, "", "None", "none", " ")


def _max_visual_y(trans: Any, points: list[tuple[float, float]]) -> float:
    if not points:
        return float("nan")
    arr = np.asarray(points, dtype=float)
    disp = trans.transform(arr)
    return float(np.max(disp[:, 1]))


def _preferred_color(*colors: Any) -> tuple[float, float, float, float] | None:
    for color in colors:
        rgba = _to_rgba(color)
        if rgba is not None and rgba[3] > 0:
            return rgba
    return None


def _to_rgba(color: Any) -> tuple[float, float, float, float] | None:
    if color is None:
        return None
    try:
        rgba = mpl.colors.to_rgba(color)
    except (ValueError, TypeError):
        return None
    return tuple(rgba)  # type: ignore[return-value]


def _collect_legend_colors(ax: Axes) -> dict[str, tuple[float, float, float, float]]:
    legend = ax.get_legend()
    if legend is None:
        return {}
    out: dict[str, tuple[float, float, float, float]] = {}
    for handle, text in zip(legend.legend_handles, legend.get_texts(), strict=False):
        label = text.get_text()
        if not label:
            continue
        rgba = _legend_handle_color(handle)
        if rgba is None:
            continue
        if label in out and not _color_close(out[label], rgba):
            raise ValueError(
                f"Legend label {label!r} maps to multiple distinct colors; "
                "annotate_significance cannot disambiguate the hue."
            )
        out[label] = rgba
    return out


def _legend_handle_color(handle: Any) -> tuple[float, float, float, float] | None:
    if isinstance(handle, Patch):
        face = handle.get_facecolor()
        rgba = _to_rgba(face)
        if rgba is not None and rgba[3] > 0:
            return rgba
        edge = _to_rgba(handle.get_edgecolor())
        if edge is not None and edge[3] > 0:
            return edge
        return None
    if isinstance(handle, Line2D):
        c = handle.get_markerfacecolor()
        rgba = _to_rgba(c)
        if rgba is not None and rgba[3] > 0:
            return rgba
        c = handle.get_markeredgecolor()
        rgba = _to_rgba(c)
        if rgba is not None and rgba[3] > 0:
            return rgba
        c = handle.get_color()
        return _to_rgba(c)
    if isinstance(handle, LineCollection):
        colors = handle.get_colors()
        if colors is None or len(colors) == 0:
            return None
        return _to_rgba(colors[0])
    if isinstance(handle, PathCollection):
        facecolors = handle.get_facecolor()
        if facecolors is not None and len(facecolors) > 0:
            return _to_rgba(tuple(facecolors[0]))
        return None
    return None


# ============================================================== layout & drawing


def _draw_brackets(
    ax: Axes,
    resolved: list[_ResolvedComparison],
    geometry: dict[object, _GroupGeometry],
    *,
    mode: str,
    bracket: str,
    color: str | None,
    y_offset_display: float,
    p_decimals: int | None,
    line_kws: Mapping[str, Any] | None,
    text_kws: Mapping[str, Any] | None,
) -> None:
    # Convert each resolved comparison into a (left_x, right_x, p_value, key_left, key_right)
    # tuple in display order.
    ordered: list[tuple[float, object, float, float, object]] = []
    for rec in resolved:
        g_l = geometry[rec.visual_left]
        g_r = geometry[rec.visual_right]
        if g_l.x_center <= g_r.x_center:
            ordered.append(
                (g_l.x_center, rec.visual_left, rec.p_value, g_r.x_center, rec.visual_right)
            )
        else:
            ordered.append(
                (g_r.x_center, rec.visual_right, rec.p_value, g_l.x_center, rec.visual_left)
            )
    # Stable sort
    ordered.sort(key=lambda t: (t[0], t[3]))

    def to_data_y(disp_y: float) -> float:
        return ax.transData.inverted().transform((0.0, disp_y))[1]

    # For each comparison, compute the maximum visual y under its x interval.
    def max_top_under(left_x: float, right_x: float) -> float:
        best = float("-inf")
        for g in geometry.values():
            if left_x - 1e-9 <= g.x_center <= right_x + 1e-9:
                if not np.isnan(g.y_top) and g.y_top > best:
                    best = g.y_top
        if best == float("-inf"):
            y0, y1 = ax.get_ylim()
            top_data = max(y0, y1) if y0 < y1 else min(y0, y1)
            return ax.transData.transform((0.0, top_data))[1]
        return best

    # Greedy level assignment: closed intervals, touching at endpoints overlap.
    level_of_index: list[int] = []
    levels: list[list[tuple[float, float, float]]] = []  # (left_x, right_x, base_top_disp)
    for left_x, _, _, right_x, _ in ordered:
        placed = False
        for li, lvl in enumerate(levels):
            if all(not (left <= right_x and left_x <= right) for left, right, _ in lvl):
                base_top = max_top_under(left_x, right_x)
                lvl.append((left_x, right_x, base_top))
                level_of_index.append(li)
                placed = True
                break
        if not placed:
            base_top = max_top_under(left_x, right_x)
            levels.append([(left_x, right_x, base_top)])
            level_of_index.append(len(levels) - 1)

    line_style = _merged_style(
        line_kws,
        color=color,
        kind="line",
        valid=_FORWARDED_LINE_KWS,
        reserved=_RESERVED_LINE_KWS,
    )
    text_style = _merged_style(
        text_kws,
        color=color,
        kind="text",
        valid=_FORWARDED_TEXT_KWS,
        reserved=_RESERVED_TEXT_KWS,
    )

    # Layout parameters in display pixels.
    fs = _resolve_fontsize(text_kws, default=mpl.rcParams["font.size"])
    # Keep the first annotation close to the rendered data, but reserve a
    # complete text line before placing the next overlapping bracket.  A
    # fixed level increment is not enough here: it lets labels collide with
    # brackets at larger font sizes (and even at Matplotlib's default size).
    base_gap = 2.0
    cap_height = 6.0
    label_gap = 4.0
    text_height = fs / 72.0 * ax.figure.dpi
    level_step = text_height + 2.0

    text_y_disp_max = float("-inf")
    positions: list[tuple[float, float, float, float, float, float]] = []
    # Per-level y position: take the maximum base top within the level so
    # disjoint intervals on the same level share a common bracket height.
    level_y_offset: dict[int, float] = {}
    for li, lvl in enumerate(levels):
        level_y_offset[li] = max(t for _, _, t in lvl)
    for (left_x, _left_k, p, right_x, _right_k), li in zip(ordered, level_of_index, strict=False):
        base_top_disp = level_y_offset[li]
        bracket_top_disp = (
            base_top_disp + base_gap + y_offset_display + li * (cap_height + label_gap + level_step)
        )
        cap_bottom_disp = bracket_top_disp - cap_height
        text_y_disp = bracket_top_disp + label_gap
        text_y_disp_max = max(text_y_disp_max, text_y_disp)
        positions.append((left_x, right_x, p, bracket_top_disp, cap_bottom_disp, text_y_disp))

    # Expand the limits before converting display coordinates to data
    # coordinates. Changing the limits changes transData, so doing this after
    # drawing would move the annotations away from their requested positions.
    _expand_y_headroom(
        ax,
        top_disp_hint=text_y_disp_max,
        text_fontsize=fs,
    )

    for left_x, right_x, p, bracket_top_disp, cap_bottom_disp, text_y_disp in positions:
        if bracket == "square":
            xs_disp = [left_x, left_x, right_x, right_x]
            ys_disp = [cap_bottom_disp, bracket_top_disp, bracket_top_disp, cap_bottom_disp]
        else:
            xs_disp = [left_x, right_x]
            ys_disp = [bracket_top_disp, bracket_top_disp]

        xs_data = list(xs_disp)
        ys_data = [to_data_y(y) for y in ys_disp]
        line = Line2D(
            xs_data,
            ys_data,
            transform=ax.transData,
            clip_on=False,
            gid=_ANNOTATION_GID,
            **line_style,
        )
        line.set_zorder(max(line_style.get("zorder", 3.0), 3.0))
        ax.add_line(line)

        label_text = _format_label(p, mode=mode, p_decimals=p_decimals)
        text_x_data = (left_x + right_x) / 2.0
        text_y_data = to_data_y(text_y_disp)
        text = Text(
            text_x_data,
            text_y_data,
            label_text,
            transform=ax.transData,
            clip_on=False,
            ha="center",
            va="bottom",
            gid=_ANNOTATION_GID,
            **text_style,
        )
        text.set_zorder(max(text_style.get("zorder", 3.0), 3.0))
        ax.add_artist(text)


def _format_label(p: float, *, mode: str, p_decimals: int | None = None) -> str:
    if mode == "stars":
        if p <= 1e-4:
            return "****"
        if p <= 1e-3:
            return "***"
        if p <= 1e-2:
            return "**"
        if p <= 0.05:
            return "*"
        return "ns"
    fmt = _format_pvalue(p, decimals=p_decimals)
    return fmt


def _format_pvalue(p: float, *, decimals: int | None = None) -> str:
    if decimals is not None:
        if round(p, decimals) == 0.0:
            return f"<{10**-decimals:.{decimals}f}"
        return f"{p:.{decimals}f}"
    if p < 1e-4:
        return f"{p:.2e}"
    return f"{p:.4g}"


def _resolve_fontsize(mapping: Mapping[str, Any] | None, *, default: float) -> float:
    if mapping and "fontsize" in mapping:
        try:
            return float(mapping["fontsize"])
        except (TypeError, ValueError):
            return default
    return default


def _merged_style(
    user_kws: Mapping[str, Any] | None,
    *,
    color: str | None,
    kind: str,
    valid: set[str],
    reserved: set[str],
) -> dict[str, Any]:
    defaults: dict[str, Any] = (
        {"linewidth": 1.0, "linestyle": "-"}
        if kind == "line"
        else {"fontsize": mpl.rcParams["font.size"]}
    )
    if color is not None:
        defaults["color"] = color
    merged = dict(defaults)
    if user_kws:
        for key, val in user_kws.items():
            if key in reserved:
                raise ValueError(
                    f"{kind}_kws key {key!r} is reserved and cannot be set by the caller."
                )
            merged[key] = val
    return merged


def _expand_y_headroom(
    ax: Axes,
    *,
    top_disp_hint: float,
    text_fontsize: float,
) -> None:
    if top_disp_hint == float("-inf"):
        return
    y0, y1 = ax.get_ylim()
    fig_dpi = ax.figure.dpi
    text_height_disp = text_fontsize / 72.0 * fig_dpi + 4.0
    margin = 6.0
    # In get_ylim order, y1 is the visual top endpoint even when the axis is inverted.
    y_top_disp = ax.transData.transform((0.0, y1))[1]
    needed = top_disp_hint + text_height_disp + margin
    if needed > y_top_disp:
        new_y1 = ax.transData.inverted().transform((0.0, needed))[1]
        ax.set_ylim(y0, new_y1)


# ============================================================== compact letters


def _draw_compact_letters(
    ax: Axes,
    resolved: list[_ResolvedComparison],
    geometry: dict[object, _GroupGeometry],
    *,
    alpha: float,
    color: str | None,
    y_offset_display: float,
    text_kws: Mapping[str, Any] | None,
) -> None:
    # Build complete pairwise set among the unique groups involved.
    groups_set: set[object] = set()
    for rec in resolved:
        groups_set.add(rec.visual_left)
        groups_set.add(rec.visual_right)
    groups = sorted(groups_set, key=lambda k: geometry[k].x_center)
    n = len(groups)
    expected = n * (n - 1) // 2
    pair_lookup: dict[frozenset[object], float] = {}
    for rec in resolved:
        a, b = rec.visual_left, rec.visual_right
        pair_lookup[_pair_key(a, b)] = rec.p_value

    # Check completeness among all pairwise combinations of the groups.
    pair_set: set[frozenset[object]] = set(pair_lookup)
    if len(pair_set) != expected:
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = groups[i], groups[j]
                k = _pair_key(a, b)
                if k not in pair_set:
                    missing.append((a, b))
        raise ValueError(
            f"mode='letters' requires a complete undirected pairwise graph. "
            f"Expected {expected} pairs across {n} groups, found {len(pair_set)}. "
            f"Missing pairs: {missing!r}."
        )

    sig = {k for k, p in pair_lookup.items() if p <= alpha}

    # Insert-and-absorb letter assignment.
    columns: list[frozenset] = [frozenset(groups)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = groups[i], groups[j]
            k = _pair_key(a, b)
            if k in sig:
                new_columns: list[frozenset] = []
                for col in columns:
                    if a in col and b in col:
                        new_columns.append(col - {a})
                        new_columns.append(col - {b})
                    else:
                        new_columns.append(col)
                uniq: list[frozenset] = []
                for c in new_columns:
                    if not c:
                        continue
                    if any(c < d for d in uniq):
                        continue
                    if any(d < c for d in uniq):
                        uniq = [x for x in uniq if not (x < c)]
                    uniq.append(c)
                columns = uniq

    columns.sort(key=lambda c: tuple(sorted(repr(x) for x in c)))

    def letter_name(idx: int) -> str:
        if idx < 26:
            return chr(ord("a") + idx)
        first = (idx // 26) - 1
        second = idx % 26
        return chr(ord("a") + first) + chr(ord("a") + second)

    group_letters: dict[object, list[str]] = {g: [] for g in groups}
    for ci, col in enumerate(columns):
        name = letter_name(ci)
        for g in col:
            group_letters[g].append(name)

    text_style = _merged_style(
        text_kws,
        color=color,
        kind="text",
        valid=_FORWARDED_TEXT_KWS,
        reserved=_RESERVED_TEXT_KWS,
    )

    fs = _resolve_fontsize(text_kws, default=mpl.rcParams["font.size"])
    fig_dpi = ax.figure.dpi
    text_height_disp = fs / 72.0 * fig_dpi + 4.0

    max_letter_y_disp = float("-inf")
    positions: list[tuple[float, float, str]] = []
    for g in groups:
        letters = "".join(sorted(group_letters.get(g, [])))
        if not letters:
            continue
        geom_g = geometry[g]
        if np.isnan(geom_g.y_top):
            y0, y1 = ax.get_ylim()
            top_data = max(y0, y1) if y0 < y1 else min(y0, y1)
            top_disp = ax.transData.transform((0.0, top_data))[1]
        else:
            top_disp = geom_g.y_top
        text_y_disp = top_disp + text_height_disp + y_offset_display
        max_letter_y_disp = max(max_letter_y_disp, text_y_disp)
        positions.append((geom_g.x_center, text_y_disp, letters))

    _expand_y_headroom(
        ax,
        top_disp_hint=max_letter_y_disp,
        text_fontsize=fs,
    )

    for x_data, text_y_disp, letters in positions:
        text_y_data = ax.transData.inverted().transform((0.0, text_y_disp))[1]
        t = Text(
            x_data,
            text_y_data,
            letters,
            transform=ax.transData,
            clip_on=False,
            ha="center",
            va="bottom",
            gid=_ANNOTATION_GID,
            **text_style,
        )
        t.set_zorder(max(text_style.get("zorder", 3.0), 3.0))
        ax.add_artist(t)
