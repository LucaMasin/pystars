"""Tests for pystars.plotting: annotate_significance adds statistical annotations to an Axes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib import rcParams

import pystars as ps
from pystars.result import TestResult

# ----------------------------------------------------------------- fixtures


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any figures opened during a test to keep state clean."""
    yield
    plt.close("all")


@pytest.fixture
def rng():
    return np.random.default_rng(2024)


def _direct_two_group_result(p: float = 0.01) -> TestResult:
    """Return a bare 2-group TestResult with no pairwise table."""
    return TestResult(
        test_name="Welch's t-test",
        statistic=2.5,
        p_value=p,
        effect_size={},
    )


def _posthoc_pairwise_result(
    pairs: list[tuple[str, str, float]],
    *,
    use_adjusted: bool = True,
) -> TestResult:
    """Build a one-way ANOVA-style primary result with a Tukey post-hoc table.

    ``pairs`` is a list of (A, B, p_value) rows.
    """
    raw = pd.DataFrame(pairs, columns=["A", "B", "p"])
    if use_adjusted:
        raw["p_adjusted"] = raw["p"]
    posthoc = TestResult(
        test_name="Tukey HSD",
        statistic=float("nan"),
        p_value=float("nan"),
        effect_size={},
        pairwise=raw,
    )
    return TestResult(
        test_name="One-way ANOVA",
        statistic=5.2,
        p_value=0.001,
        effect_size={},
        posthoc=posthoc,
    )


def _draw_simple_bar(ax, heights=(2.0, 4.0), labels=("ctrl", "trt")):
    ax.bar(list(labels), list(heights))
    return ax


def _draw_simple_strip(ax, n: int = 8, labels=("ctrl", "trt"), rng=None):
    rng = rng or np.random.default_rng(0)
    for i, _lab in enumerate(labels):
        xs = rng.normal(i, 0.0, n)
        ys = rng.normal(0, 1, n) + (i + 1) * 2
        ax.scatter(xs, ys)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(list(labels))
    return ax


def _annotation_artists(ax):
    """Return Line2D/Text artists tagged with the plotting module gid."""
    lines = [line for line in ax.lines if line.get_gid() == "pystars_annotation"]
    texts = [t for t in ax.texts if getattr(t, "get_gid", lambda: None)() == "pystars_annotation"]
    return lines, texts


def _summary_marker_lines(ax):
    """Return non-empty marker-bearing lines, excluding annotation artists."""
    return [
        line
        for line in ax.lines
        if line.get_gid() != "pystars_annotation"
        and line.get_marker() not in (None, "", "None", "none", " ")
        and len(line.get_xdata()) > 0
        and len(line.get_xdata()) == len(line.get_ydata())
    ]


class _CollidingLabel:
    """Distinct labels that intentionally share a hash value."""

    def __init__(self, name: str):
        self.name = name

    def __hash__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _CollidingLabel) and self.name == other.name


# ================================================================ core tests


class TestCoreAnnotation:
    def test_returns_same_axes(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result()
        out = ps.annotate_significance(ax, result, groups=("ctrl", "trt"))
        assert out is ax

    def test_bracket_uses_category_centers_with_star_label(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=0.01)
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"))
        lines, texts = _annotation_artists(ax)
        assert len(lines) >= 1
        assert len(texts) == 1
        text = texts[0]
        assert text.get_text() == "**"
        # Default bracket is a two-endpoint line
        xs, _ys = lines[0].get_xdata(), lines[0].get_ydata()
        # First/last x equal the displayed category tick positions
        ticks = list(ax.get_xticks())
        assert xs[0] == pytest.approx(ticks[0])
        assert xs[-1] == pytest.approx(ticks[1])
        # Default color is black
        assert lines[0].get_color() == "black"
        assert text.get_color() == "black"

    def test_dispatcher_posthoc_pairwise_discovered(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 3.0, 5.0), labels=("a", "b", "c"))
        result = _posthoc_pairwise_result(
            [("a", "b", 0.01), ("a", "c", 0.4), ("b", "c", 0.001)],
        )
        ps.annotate_significance(ax, result)
        # Default selection = significant, alpha=0.05 → two significant pairs
        _, texts = _annotation_artists(ax)
        assert len(texts) == 2
        labels = sorted(t.get_text() for t in texts)
        assert labels == ["**", "***"]

    def test_comparison_table_overrides_result(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 3.0, 5.0), labels=("a", "b", "c"))
        result = _posthoc_pairwise_result(
            [("a", "b", 0.01), ("a", "c", 0.4), ("b", "c", 0.001)],
        )
        # Override p_adjusted values; p is non-significant in the result, so
        # default would not draw any bracket. With adjusted p < .05 we should
        # see a single bracket for the chosen pair.
        override = pd.DataFrame(
            {"A": ["a", "b"], "B": ["b", "c"], "p": [0.001, 0.7], "p_adjusted": [0.001, 0.7]}
        )
        ps.annotate_significance(ax, result, comparison_table=override)
        _, texts = _annotation_artists(ax)
        assert len(texts) == 1
        # Only the first pair meets adjusted p ≤ alpha
        assert texts[0].get_text() == "***"

    def test_label_map_remaps_statistics_to_x_ticks(self):
        fig, ax = plt.subplots()
        ax.bar(["control", "treated"], [1.0, 2.0])
        result = _direct_two_group_result(p=0.001)
        ps.annotate_significance(
            ax,
            result,
            groups=("control_st", "treated_st"),
            label_map={"control_st": "control", "treated_st": "treated"},
        )
        lines, _ = _annotation_artists(ax)
        assert len(lines) >= 1
        ticks = list(ax.get_xticks())
        xs = lines[0].get_xdata()
        assert xs[0] == pytest.approx(ticks[0])
        assert xs[-1] == pytest.approx(ticks[1])

    def test_missing_groups_raises_helpful_error(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result()
        with pytest.raises(ValueError, match="groups"):
            ps.annotate_significance(ax, result)


# ========================================================== artist resolver


class TestArtistResolver:
    def test_matplotlib_bar_scalar_label_resolves_to_tick(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        result = _posthoc_pairwise_result(
            [("a", "c", 0.01), ("a", "b", 0.4), ("b", "c", 0.4)],
        )
        ps.annotate_significance(ax, result)
        lines, _ = _annotation_artists(ax)
        assert len(lines) == 1
        ticks = list(ax.get_xticks())
        xs = lines[0].get_xdata()
        assert xs[0] == pytest.approx(ticks[0])
        assert xs[-1] == pytest.approx(ticks[2])

    def test_matplotlib_strip_scalar_label_resolves(self, rng):
        fig, ax = plt.subplots()
        _draw_simple_strip(ax, n=10, labels=("a", "b", "c"), rng=rng)
        result = _posthoc_pairwise_result(
            [("a", "b", 0.01), ("a", "c", 0.4), ("b", "c", 0.4)],
        )
        ps.annotate_significance(ax, result)
        lines, _ = _annotation_artists(ax)
        assert len(lines) == 1

    def test_seaborn_dodged_bar_with_three_hue_levels(self):
        sns = pytest.importorskip("seaborn")
        df = pd.DataFrame(
            {
                "x": (["a"] * 6 + ["b"] * 6 + ["c"] * 6),
                "h": (["h1", "h2", "h3"] * 6),
                "y": np.tile([1.0, 2.0, 3.0], 6) + np.random.default_rng(0).normal(0, 0.2, 18),
            }
        )
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="x", y="y", hue="h", ax=ax)
        result = _posthoc_pairwise_result(
            [("a::h1", "b::h1", 0.01), ("a::h1", "a::h2", 0.5), ("a::h2", "b::h2", 0.4)],
        )
        ps.annotate_significance(
            ax,
            result,
            label_map={
                "a::h1": ("a", "h1"),
                "b::h1": ("b", "h1"),
            },
        )
        lines, _ = _annotation_artists(ax)
        # Two endpoints should differ from each other and correspond to distinct
        # dodged bar centers on x = "a" and x = "b" with hue "h1".
        assert len(lines) == 1
        xs = list(lines[0].get_xdata())
        # Each dodged bar group is separated; assert endpoints are unequal.
        assert xs[0] != xs[-1]
        # Endpoints should fall in the [-0.5, 0.5] (for x="a") and [0.5, 1.5]
        # (for x="b") regions when hue is "h1".
        assert xs[0] < xs[-1]

    def test_seaborn_dodged_strip_with_hue(self):
        sns = pytest.importorskip("seaborn")
        df = pd.DataFrame(
            {
                "x": (["a"] * 6 + ["b"] * 6 + ["c"] * 6),
                "h": (["h1", "h2"] * 9),
                "y": np.random.default_rng(0).normal(0, 1, 18) + np.tile([0.0, 1.0, 2.0], 6),
            }
        )
        fig, ax = plt.subplots()
        sns.stripplot(data=df, x="x", y="y", hue="h", dodge=True, ax=ax)
        result = _posthoc_pairwise_result(
            [("a::h1", "b::h1", 0.01), ("a::h1", "a::h2", 0.5), ("a::h2", "b::h2", 0.4)],
        )
        ps.annotate_significance(
            ax,
            result,
            label_map={
                "a::h1": ("a", "h1"),
                "b::h1": ("b", "h1"),
            },
        )
        lines, _ = _annotation_artists(ax)
        assert len(lines) == 1
        xs = list(lines[0].get_xdata())
        assert xs[0] != xs[-1]

    def test_seaborn_pointplot_scalar_endpoints_use_ticks(self):
        sns = pytest.importorskip("seaborn")
        df = pd.DataFrame(
            {
                "group": ["a"] * 5 + ["b"] * 5,
                "value": [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 8.0, 9.0],
            }
        )
        fig, ax = plt.subplots()
        sns.pointplot(
            data=df,
            x="group",
            y="value",
            errorbar=("ci", 95),
            capsize=0.2,
            n_boot=100,
            seed=0,
            ax=ax,
        )

        ps.annotate_significance(ax, _direct_two_group_result(), groups=("a", "b"))
        lines, _ = _annotation_artists(ax)
        assert len(lines) == 1
        assert list(lines[0].get_xdata()) == list(ax.get_xticks())

    def test_seaborn_pointplot_hue_uses_dodged_summary_centers(self):
        sns = pytest.importorskip("seaborn")
        df = pd.DataFrame(
            {
                "x": (["a"] * 6 + ["b"] * 6),
                "h": (["h1", "h2"] * 6),
                "value": [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 8.0, 9.0, 10.0, 11.0],
            }
        )
        fig, ax = plt.subplots()
        sns.pointplot(
            data=df,
            x="x",
            y="value",
            hue="h",
            dodge=0.4,
            errorbar=("ci", 95),
            capsize=0.2,
            n_boot=100,
            seed=0,
            ax=ax,
        )
        h1_color = matplotlib.colors.to_rgba(ax.get_legend().legend_handles[0].get_color())
        expected_x = [
            float(x)
            for line in _summary_marker_lines(ax)
            if np.allclose(matplotlib.colors.to_rgba(line.get_color()), h1_color)
            for x in line.get_xdata()
        ]

        ps.annotate_significance(
            ax,
            _direct_two_group_result(),
            groups=("a::h1", "b::h1"),
            label_map={"a::h1": ("a", "h1"), "b::h1": ("b", "h1")},
        )
        lines, _ = _annotation_artists(ax)
        assert len(lines) == 1
        assert sorted(lines[0].get_xdata()) == pytest.approx(sorted(expected_x))

    def test_layered_swarm_and_pointplot_prefers_summary_centers(self):
        sns = pytest.importorskip("seaborn")
        base_values = [0.2, 0.201, 0.29561, 0.29664, 0.2, 0.201]
        rows = []
        for x in ("a", "b"):
            for hue in ("h1", "h2"):
                rows.extend({"x": x, "h": hue, "value": value} for value in base_values)
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots()
        sns.swarmplot(data=df, x="x", y="value", hue="h", dodge=True, size=5, ax=ax)
        sns.pointplot(
            data=df,
            x="x",
            y="value",
            hue="h",
            dodge=0.4,
            errorbar=("ci", 95),
            capsize=0.2,
            n_boot=100,
            seed=0,
            ax=ax,
        )
        h1_color = matplotlib.colors.to_rgba(ax.get_legend().legend_handles[0].get_color())
        summary_x = [
            float(x)
            for line in _summary_marker_lines(ax)
            if np.allclose(matplotlib.colors.to_rgba(line.get_color()), h1_color)
            for x in line.get_xdata()
        ]
        h1_swarms = [
            coll
            for coll in ax.collections
            if len(coll.get_offsets()) and np.allclose(coll.get_facecolor()[0], h1_color)
        ]
        swarm_medians = sorted(
            float(np.median(np.asarray(coll.get_offsets())[:, 0])) for coll in h1_swarms
        )
        assert any(
            abs(median - center) > 1e-3
            for median, center in zip(swarm_medians, summary_x, strict=True)
        )

        ps.annotate_significance(
            ax,
            _direct_two_group_result(),
            groups=("a::h1", "b::h1"),
            label_map={"a::h1": ("a", "h1"), "b::h1": ("b", "h1")},
        )
        lines, _ = _annotation_artists(ax)
        assert len(lines) == 1
        assert sorted(lines[0].get_xdata()) == pytest.approx(sorted(summary_x))

    def test_bracket_clears_capped_pointplot_confidence_intervals(self):
        sns = pytest.importorskip("seaborn")
        df = pd.DataFrame(
            {
                "group": ["a"] * 5 + ["b"] * 5,
                "value": [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 12.0],
            }
        )
        fig, ax = plt.subplots()
        sns.pointplot(
            data=df,
            x="group",
            y="value",
            errorbar=("ci", 95),
            capsize=0.2,
            n_boot=100,
            seed=0,
            ax=ax,
        )
        summary_y = np.concatenate(
            [np.asarray(line.get_ydata(), dtype=float) for line in _summary_marker_lines(ax)]
        )
        errorbar_y = np.concatenate(
            [
                np.asarray(line.get_ydata(), dtype=float)[
                    np.isfinite(np.asarray(line.get_ydata(), dtype=float))
                ]
                for line in ax.lines
                if line.get_marker() in (None, "", "None", "none", " ") and len(line.get_ydata())
            ]
        )
        assert errorbar_y.size
        assert np.max(errorbar_y) > np.max(summary_y)
        assert any(np.isnan(line.get_xdata()).any() for line in ax.lines if len(line.get_xdata()))
        ax.set_ylim(0.0, 9.0)

        ps.annotate_significance(ax, _direct_two_group_result(), groups=("a", "b"))
        fig.canvas.draw()
        lines, _ = _annotation_artists(ax)
        bracket_top = ax.transData.transform((0.0, lines[0].get_ydata()[0]))[1]
        errorbar_top = max(ax.transData.transform((0.0, float(value)))[1] for value in errorbar_y)
        assert bracket_top > errorbar_top

    def test_pure_dodged_swarm_uses_nominal_cell_centers(self):
        sns = pytest.importorskip("seaborn")
        base_values = [0.2, 0.201, 0.29561, 0.29664, 0.2, 0.201]
        rows = []
        for x in ("a", "b"):
            for hue in ("h1", "h2"):
                rows.extend({"x": x, "h": hue, "value": value} for value in base_values)
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots()
        sns.swarmplot(data=df, x="x", y="value", hue="h", dodge=True, size=5, ax=ax)
        h1_color = matplotlib.colors.to_rgba(ax.get_legend().legend_handles[0].get_color())
        h1_swarms = [
            coll
            for coll in ax.collections
            if len(coll.get_offsets()) and np.allclose(coll.get_facecolor()[0], h1_color)
        ]
        h1_swarms.sort(key=lambda coll: float(np.mean(np.asarray(coll.get_offsets())[:, 0])))
        assert len(h1_swarms) == 2
        swarm_maxima = [float(np.max(np.asarray(coll.get_offsets())[:, 0])) for coll in h1_swarms]
        assert any(
            abs(maximum - center) > 1e-3
            for maximum, center in zip(swarm_maxima, (-0.2, 0.8), strict=True)
        )

        ps.annotate_significance(
            ax,
            _direct_two_group_result(),
            groups=("a::h1", "b::h1"),
            label_map={"a::h1": ("a", "h1"), "b::h1": ("b", "h1")},
        )
        lines, _ = _annotation_artists(ax)
        assert len(lines) == 1
        assert list(lines[0].get_xdata()) == pytest.approx([-0.2, 0.8])

    def test_horizontal_reference_line_does_not_override_dodged_centers(self):
        sns = pytest.importorskip("seaborn")
        rows = []
        for x in ("a", "b"):
            for hue in ("h1", "h2"):
                rows.extend({"x": x, "h": hue, "value": value} for value in (1.0, 2.0, 3.0))
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots()
        sns.stripplot(data=df, x="x", y="value", hue="h", dodge=True, jitter=False, ax=ax)
        h1_color = ax.get_legend().legend_handles[0].get_color()
        ax.axhline(4.0, color=h1_color)

        ps.annotate_significance(
            ax,
            _direct_two_group_result(),
            groups=("a::h1", "b::h1"),
            label_map={"a::h1": ("a", "h1"), "b::h1": ("b", "h1")},
        )

        lines, _ = _annotation_artists(ax)
        assert len(lines) == 1
        assert list(lines[0].get_xdata()) == pytest.approx([-0.2, 0.8])

    def test_partial_tuple_label_map_raises(self):
        """When some selected comparisons lack tuple mappings, raise helpfully."""
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        result = _posthoc_pairwise_result(
            [("a::h1", "b::h1", 0.01), ("a::h1", "a::h2", 0.01)],
        )
        with pytest.raises(ValueError, match="(?i)missing|label_map"):
            ps.annotate_significance(
                ax,
                result,
                label_map={"a::h1": ("a", "h1")},
            )

    def test_missing_legend_raises(self):
        fig, ax = plt.subplots()
        ax.bar(["a", "b", "c"], [1.0, 2.0, 3.0])
        result = _posthoc_pairwise_result([("a::h1", "b::h1", 0.01)])
        with pytest.raises(ValueError, match=r"(?i)h1|hue|legend"):
            ps.annotate_significance(
                ax,
                result,
                label_map={"a::h1": ("a", "h1"), "b::h1": ("b", "h1")},
            )

    def test_unknown_x_category_raises(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, labels=("a", "b"))
        result = _posthoc_pairwise_result([("zzz", "b", 0.01)])
        with pytest.raises(ValueError, match="zzz"):
            ps.annotate_significance(ax, result)

    def test_duplicate_visual_mapping_raises(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        result = _posthoc_pairwise_result(
            [("a", "b", 0.01), ("a", "c", 0.02)],
        )
        # Map "b" and "c" both to the displayed label "a" — they then share
        # a visual target on the x axis.
        with pytest.raises(ValueError, match="(?i)visual|map|target"):
            ps.annotate_significance(
                ax,
                result,
                label_map={"b": "a", "c": "a"},
            )

    def test_earlier_annotations_ignored_as_data(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        result = _posthoc_pairwise_result([("a", "b", 0.01)])
        # First call adds a bracket to "a" vs "b"
        ps.annotate_significance(ax, result)
        first_lines, _ = _annotation_artists(ax)
        assert len(first_lines) == 1
        # Second call asks for a comparison that would otherwise look at all
        # rendered bars. The previous bracket must not be picked up as a bar
        # (it has no x center inside a tick spacing).
        ps.annotate_significance(ax, _posthoc_pairwise_result([("a", "c", 0.01)]))
        lines, _ = _annotation_artists(ax)
        # We have now two annotations (a vs b and a vs c), both well-defined.
        assert len(lines) == 2


# ============================================== selection, formatting, layout


class TestSelectionFormattingLayout:
    @pytest.mark.parametrize(
        "p,expected",
        [
            (1e-5, "****"),
            (0.0005, "***"),
            (0.005, "**"),
            (0.04, "*"),
        ],
    )
    def test_star_boundaries(self, p, expected):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=p)
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"))
        _, texts = _annotation_artists(ax)
        assert texts[0].get_text() == expected

    def test_non_significant_label_with_all(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=0.06)
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"), comparisons="all")
        _, texts = _annotation_artists(ax)
        assert texts[0].get_text() == "ns"

    def test_default_significant_only_filtering(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        result = _posthoc_pairwise_result(
            [("a", "b", 0.5), ("a", "c", 0.5), ("b", "c", 0.5)],
        )
        ps.annotate_significance(ax, result)
        _, texts = _annotation_artists(ax)
        assert texts == []

    def test_comparisons_all_includes_ns(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        result = _posthoc_pairwise_result(
            [("a", "b", 0.5), ("a", "c", 0.01), ("b", "c", 0.5)],
        )
        ps.annotate_significance(ax, result, comparisons="all")
        _, texts = _annotation_artists(ax)
        labels = [t.get_text() for t in texts]
        assert "ns" in labels
        assert "**" in labels

    def test_explicit_non_significant_pair(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        result = _posthoc_pairwise_result(
            [("a", "b", 0.5), ("a", "c", 0.01), ("b", "c", 0.5)],
        )
        ps.annotate_significance(ax, result, comparisons=[("a", "b")])
        _, texts = _annotation_artists(ax)
        assert len(texts) == 1
        assert texts[0].get_text() == "ns"

    def test_pvalue_mode_format(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=0.003)
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"), mode="pvalue")
        _, texts = _annotation_artists(ax)
        assert texts[0].get_text() == "0.003"

    def test_p_decimals_pvalue_mode_uses_threshold_when_value_rounds_to_zero(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=0.003)
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"), mode="pvalue", p_decimals=2)
        _, texts = _annotation_artists(ax)
        assert texts[0].get_text() == "<0.01"

    def test_p_decimals_value_mode(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=0.123456)
        ps.annotate_significance(
            ax,
            result,
            groups=("ctrl", "trt"),
            mode="value",
            p_decimals=3,
            comparisons="all",
        )
        _, texts = _annotation_artists(ax)
        assert texts[0].get_text() == "0.123"

    def test_p_decimals_threshold_for_tiny_p(self):
        result = _direct_two_group_result(p=1.2e-7)

        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"), mode="pvalue", p_decimals=3)
        _, texts = _annotation_artists(ax)
        assert texts[0].get_text() == "<0.001"

        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"), mode="value", p_decimals=3)
        _, texts = _annotation_artists(ax)
        assert texts[0].get_text() == "<0.001"

    def test_p_decimals_rounds_to_nonzero(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=0.00051)
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"), mode="value", p_decimals=3)
        _, texts = _annotation_artists(ax)
        assert texts[0].get_text() == "0.001"

    def test_p_decimals_default_unchanged(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=0.123456)
        ps.annotate_significance(
            ax, result, groups=("ctrl", "trt"), mode="value", comparisons="all"
        )
        _, texts = _annotation_artists(ax)
        assert texts[0].get_text() == "0.1235"

    def test_p_decimals_ignored_for_stars(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=0.03)
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"), mode="stars", p_decimals=2)
        _, texts = _annotation_artists(ax)
        assert texts[0].get_text() == "*"

    def test_p_decimals_threads_through_pairwise_tables(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 3.0, 5.0), labels=("a", "b", "c"))
        result = _posthoc_pairwise_result([("a", "b", 0.0034)])
        ps.annotate_significance(ax, result, mode="pvalue", p_decimals=2)
        _, texts = _annotation_artists(ax)
        assert texts[0].get_text() == "<0.01"

    def test_value_mode_format(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=0.003)
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"), mode="value")
        _, texts = _annotation_artists(ax)
        assert texts[0].get_text() == "0.003"

    def test_tiny_pvalue_uses_scientific_notation(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=1.2e-7)
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"), mode="value")
        _, texts = _annotation_artists(ax)
        text = texts[0].get_text()
        assert "e" in text.lower() or "E" in text

    def test_line_vs_square_bracket_shape(self):
        fig1, ax1 = plt.subplots()
        _draw_simple_bar(ax1)
        ps.annotate_significance(
            ax1, _direct_two_group_result(0.01), groups=("ctrl", "trt"), bracket="line"
        )
        line_lines, _ = _annotation_artists(ax1)
        assert len(line_lines[0].get_xdata()) == 2

        fig2, ax2 = plt.subplots()
        _draw_simple_bar(ax2)
        ps.annotate_significance(
            ax2, _direct_two_group_result(0.01), groups=("ctrl", "trt"), bracket="square"
        )
        square_lines, _ = _annotation_artists(ax2)
        assert len(square_lines[0].get_xdata()) == 4

    def test_default_bracket_is_line(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        ps.annotate_significance(ax, _direct_two_group_result(0.01), groups=("ctrl", "trt"))
        lines, _ = _annotation_artists(ax)
        assert len(lines[0].get_xdata()) == 2

    def test_default_color_is_black(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        ps.annotate_significance(ax, _direct_two_group_result(0.01), groups=("ctrl", "trt"))
        lines, texts = _annotation_artists(ax)
        assert lines[0].get_color() == "black"
        assert texts[0].get_color() == "black"

    def test_color_applies_to_both(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        ps.annotate_significance(
            ax, _direct_two_group_result(0.01), groups=("ctrl", "trt"), color="red"
        )
        lines, texts = _annotation_artists(ax)
        assert lines[0].get_color() == "red"
        assert texts[0].get_color() == "red"

    def test_color_overridden_in_text_kws(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        # Use a non-default color to verify the override mechanism; the
        # default is now "black" so a custom color must explicitly differ.
        ps.annotate_significance(
            ax,
            _direct_two_group_result(0.01),
            groups=("ctrl", "trt"),
            color="red",
            text_kws={"color": "blue"},
        )
        lines, texts = _annotation_artists(ax)
        assert lines[0].get_color() == "red"
        assert texts[0].get_color() == "blue"

    def test_supported_line_text_properties_forwarded(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        ps.annotate_significance(
            ax,
            _direct_two_group_result(0.01),
            groups=("ctrl", "trt"),
            line_kws={"linewidth": 2.0, "linestyle": "--"},
            text_kws={"fontsize": 14.0, "fontweight": "bold"},
        )
        lines, texts = _annotation_artists(ax)
        assert lines[0].get_linewidth() == 2.0
        assert lines[0].get_linestyle() == "--"
        assert texts[0].get_fontsize() == 14.0
        assert texts[0].get_fontweight() == "bold"

    def test_overlapping_pairs_distinct_levels(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0, 4.0), labels=("a", "b", "c", "d"))
        result = _posthoc_pairwise_result(
            [("a", "b", 0.01), ("a", "c", 0.01), ("a", "d", 0.01)],
        )
        ps.annotate_significance(ax, result)
        lines, texts = _annotation_artists(ax)
        # All three share a common left endpoint, so they cannot all share a
        # level. We expect at least two distinct y values among the brackets.
        y_values = [lines[i].get_ydata()[0] for i in range(len(lines))]
        assert len({round(y, 6) for y in y_values}) >= 2

    def test_overlapping_pairs_leave_room_for_labels(self):
        """Each lower-level label must clear the bracket immediately above it."""
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        result = _posthoc_pairwise_result(
            [("a", "b", 0.00001), ("a", "c", 0.00001), ("b", "c", 0.00001)],
        )

        ps.annotate_significance(ax, result)
        fig.canvas.draw()
        lines, texts = _annotation_artists(ax)

        bracket_ys = sorted(ax.transData.transform((0.0, line.get_ydata()[0]))[1] for line in lines)
        renderer = fig.canvas.get_renderer()
        label_tops = sorted(text.get_window_extent(renderer).ymax for text in texts)

        assert len(bracket_ys) == len(label_tops) == 3
        for label_top, next_bracket_y in zip(label_tops, bracket_ys[1:], strict=False):
            assert label_top < next_bracket_y

    def test_disjoint_pairs_share_a_level(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0, 4.0), labels=("a", "b", "c", "d"))
        result = _posthoc_pairwise_result(
            [("a", "b", 0.01), ("c", "d", 0.01)],
        )
        ps.annotate_significance(ax, result)
        lines, _ = _annotation_artists(ax)
        # Disjoint spans should share the same top level.
        y_a = lines[0].get_ydata()[0]
        y_c = lines[1].get_ydata()[0]
        assert y_a == pytest.approx(y_c, rel=1e-6)

    def test_log_scale_y_axis(self):
        fig, ax = plt.subplots()
        ax.bar(["a", "b"], [10.0, 100.0])
        ax.set_yscale("log")
        result = _direct_two_group_result(p=0.01)
        ps.annotate_significance(ax, result, groups=("a", "b"))
        _, texts = _annotation_artists(ax)
        assert len(texts) == 1
        # Annotation must remain within expanded limits.
        y0, y1 = ax.get_ylim()
        assert y0 < 10.0

    def test_inverted_y_axis(self):
        fig, ax = plt.subplots()
        ax.bar(["a", "b"], [10.0, 5.0])
        ax.invert_yaxis()
        original_ylim = ax.get_ylim()
        result = _direct_two_group_result(p=0.01)
        ps.annotate_significance(ax, result, groups=("a", "b"))
        new_ylim = ax.get_ylim()
        # Inverted axis: the larger value is at the bottom. Original (top) is
        # the second element; the new top should be smaller (more negative).
        assert new_ylim[1] < original_ylim[1]

    def test_bracket_clears_bar_error_bars(self):
        fig, ax = plt.subplots()
        ax.bar(["a", "b"], [2.0, 4.0], yerr=[1.0, 3.0], capsize=3)
        ps.annotate_significance(ax, _direct_two_group_result(0.01), groups=("a", "b"))
        lines, _ = _annotation_artists(ax)
        bracket_top = ax.transData.transform((0.0, lines[0].get_ydata()[0]))[1]
        errorbar_top = ax.transData.transform((0.0, 7.0))[1]
        assert bracket_top > errorbar_top

    def test_bracket_clears_seaborn_confidence_intervals(self):
        sns = pytest.importorskip("seaborn")
        df = pd.DataFrame(
            {
                "group": ["a"] * 5 + ["b"] * 5,
                "value": [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 9.0, 12.0],
            }
        )
        fig, ax = plt.subplots()
        sns.barplot(data=df, x="group", y="value", errorbar=("ci", 95), n_boot=100, seed=0, ax=ax)
        errorbar_top_values = [
            max(line.get_ydata())
            for line in ax.lines
            if len(line.get_xdata()) >= 2 and np.allclose(line.get_xdata(), line.get_xdata()[0])
        ]
        assert errorbar_top_values

        ps.annotate_significance(ax, _direct_two_group_result(0.01), groups=("a", "b"))
        lines, _ = _annotation_artists(ax)
        bracket_top = ax.transData.transform((0.0, lines[0].get_ydata()[0]))[1]
        errorbar_top = ax.transData.transform((0.0, max(errorbar_top_values)))[1]
        assert bracket_top > errorbar_top

    def test_existing_artists_unchanged(self):
        fig, ax = plt.subplots()
        bars = ax.bar(["a", "b", "c"], [1.0, 2.0, 3.0])
        result = _posthoc_pairwise_result(
            [("a", "b", 0.01), ("a", "c", 0.5), ("b", "c", 0.5)],
        )
        # Capture original bar heights and y limits
        before_heights = [b.get_height() for b in bars]
        ps.annotate_significance(ax, result)
        # Bar heights must remain unchanged
        after_heights = [b.get_height() for b in bars]
        assert before_heights == after_heights


# ========================================== compact letter display and validation


class TestCompactLettersAndValidation:
    def test_letter_mode_non_transitive_three_groups(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        # Non-transitive: a-b ns, b-c ns, a-c significant.
        result = _posthoc_pairwise_result(
            [("a", "b", 0.2), ("b", "c", 0.2), ("a", "c", 0.01)],
        )
        ps.annotate_significance(ax, result, mode="letters", comparisons="all")
        # No brackets, only text labels at each tick.
        lines, texts = _annotation_artists(ax)
        assert lines == []
        assert len(texts) == 3
        # a-c are different; a-b and b-c share letters; a-c shares none.
        text_by_x = {round(t.get_position()[0], 6): t.get_text() for t in texts}
        a_letters = set(text_by_x[round(ax.get_xticks()[0], 6)])
        b_letters = set(text_by_x[round(ax.get_xticks()[1], 6)])
        c_letters = set(text_by_x[round(ax.get_xticks()[2], 6)])
        assert a_letters.isdisjoint(c_letters)
        assert a_letters & b_letters
        assert b_letters & c_letters

    def test_letter_mode_rejects_significant(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        result = _posthoc_pairwise_result(
            [("a", "b", 0.2), ("b", "c", 0.2), ("a", "c", 0.01)],
        )
        with pytest.raises(ValueError, match="letters"):
            ps.annotate_significance(ax, result, mode="letters", comparisons="significant")

    def test_letter_mode_rejects_explicit_pairs(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        result = _posthoc_pairwise_result(
            [("a", "b", 0.2), ("b", "c", 0.2), ("a", "c", 0.01)],
        )
        with pytest.raises(ValueError, match="letters"):
            ps.annotate_significance(ax, result, mode="letters", comparisons=[("a", "b")])

    def test_letter_mode_rejects_incomplete_pairs(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        result = _posthoc_pairwise_result(
            [("a", "b", 0.2), ("b", "c", 0.2)],  # missing a-c
        )
        with pytest.raises(ValueError, match="a.*c|c.*a|missing"):
            ps.annotate_significance(ax, result, mode="letters", comparisons="all")

    @pytest.mark.parametrize(
        "bad_p",
        ["not_a_number", None],
    )
    def test_invalid_p_value_raises(self, bad_p):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        df = pd.DataFrame({"A": ["a", "b"], "B": ["b", "c"], "p": [bad_p, 0.5]})
        result = TestResult(
            test_name="Custom", statistic=1.0, p_value=0.5, effect_size={}, pairwise=df
        )
        with pytest.raises(ValueError, match="p"):
            ps.annotate_significance(ax, result, comparison_table=df)

    def test_nan_p_value_raises(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        df = pd.DataFrame({"A": ["a", "b"], "B": ["b", "c"], "p": [float("nan"), 0.5]})
        result = TestResult(
            test_name="Custom", statistic=1.0, p_value=0.5, effect_size={}, pairwise=df
        )
        with pytest.raises(ValueError, match="p"):
            ps.annotate_significance(ax, result, comparison_table=df)

    def test_p_out_of_range_raises(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        df = pd.DataFrame({"A": ["a", "b"], "B": ["b", "c"], "p": [1.5, 0.5]})
        result = TestResult(
            test_name="Custom", statistic=1.0, p_value=0.5, effect_size={}, pairwise=df
        )
        with pytest.raises(ValueError, match=r"\[0, 1\]|0.*1"):
            ps.annotate_significance(ax, result, comparison_table=df)

    def test_self_pair_raises(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        df = pd.DataFrame({"A": ["a", "b"], "B": ["a", "c"], "p": [0.1, 0.1]})
        result = TestResult(
            test_name="Custom", statistic=1.0, p_value=0.5, effect_size={}, pairwise=df
        )
        with pytest.raises(ValueError, match="self"):
            ps.annotate_significance(ax, result, comparison_table=df)

    def test_duplicate_pair_raises(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        df = pd.DataFrame({"A": ["a", "a"], "B": ["b", "b"], "p": [0.1, 0.2]})
        result = TestResult(
            test_name="Custom", statistic=1.0, p_value=0.5, effect_size={}, pairwise=df
        )
        with pytest.raises(ValueError, match="(?i)duplicate"):
            ps.annotate_significance(ax, result, comparison_table=df)

    @pytest.mark.parametrize("mode", ["star", "STAR", "", "value "])
    def test_invalid_mode_raises(self, mode):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=0.01)
        with pytest.raises(ValueError, match="mode"):
            ps.annotate_significance(ax, result, groups=("ctrl", "trt"), mode=mode)

    def test_invalid_bracket_raises(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=0.01)
        with pytest.raises(ValueError, match="bracket"):
            ps.annotate_significance(ax, result, groups=("ctrl", "trt"), bracket="circle")

    def test_invalid_alpha_raises(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=0.01)
        with pytest.raises(ValueError, match="alpha"):
            ps.annotate_significance(ax, result, groups=("ctrl", "trt"), alpha=0)
        with pytest.raises(ValueError, match="alpha"):
            ps.annotate_significance(ax, result, groups=("ctrl", "trt"), alpha=True)
        with pytest.raises(ValueError, match="alpha"):
            ps.annotate_significance(ax, result, groups=("ctrl", "trt"), alpha=1.0)

    def test_y_offset_default_matches_baseline(self):
        fig, ax = plt.subplots()
        ax.set_ylim(-5, 50)
        _draw_simple_bar(ax, heights=(2.0, 4.0), labels=("ctrl", "trt"))
        result = _direct_two_group_result(p=0.01)
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"))
        fig.canvas.draw()
        lines, _ = _annotation_artists(ax)
        baseline_disp = ax.transData.transform((0.0, lines[0].get_ydata()[0]))[1]

        fig2, ax2 = plt.subplots()
        ax2.set_ylim(-5, 50)
        _draw_simple_bar(ax2, heights=(2.0, 4.0), labels=("ctrl", "trt"))
        ps.annotate_significance(ax2, result, groups=("ctrl", "trt"), y_offset=0)
        fig2.canvas.draw()
        lines2, _ = _annotation_artists(ax2)
        default_disp = ax2.transData.transform((0.0, lines2[0].get_ydata()[0]))[1]
        assert baseline_disp == pytest.approx(default_disp, abs=1e-6)

    def test_y_offset_positive_shifts_bracket_stack_up(self):
        fig, ax = plt.subplots()
        ax.set_ylim(-5, 50)
        _draw_simple_bar(ax, heights=(2.0, 4.0), labels=("ctrl", "trt"))
        result = _direct_two_group_result(p=0.01)
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"))
        fig.canvas.draw()
        lines, texts = _annotation_artists(ax)
        baseline_bracket_disp = ax.transData.transform((0.0, lines[0].get_ydata()[0]))[1]
        renderer = fig.canvas.get_renderer()
        baseline_label_disp = texts[0].get_window_extent(renderer).ymin

        fig2, ax2 = plt.subplots()
        ax2.set_ylim(-5, 50)
        _draw_simple_bar(ax2, heights=(2.0, 4.0), labels=("ctrl", "trt"))
        ps.annotate_significance(ax2, result, groups=("ctrl", "trt"), y_offset=12)
        fig2.canvas.draw()
        lines2, texts2 = _annotation_artists(ax2)
        shifted_bracket_disp = ax2.transData.transform((0.0, lines2[0].get_ydata()[0]))[1]
        renderer2 = fig2.canvas.get_renderer()
        shifted_label_disp = texts2[0].get_window_extent(renderer2).ymin

        expected = -12.0 * fig2.dpi / 72.0
        assert (baseline_bracket_disp - shifted_bracket_disp) == pytest.approx(expected, abs=0.5)
        assert (baseline_label_disp - shifted_label_disp) == pytest.approx(expected, abs=0.5)

    def test_y_offset_negative_shifts_bracket_stack_down(self):
        fig, ax = plt.subplots()
        ax.set_ylim(-5, 50)
        _draw_simple_bar(ax, heights=(2.0, 4.0), labels=("ctrl", "trt"))
        result = _direct_two_group_result(p=0.01)
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"))
        fig.canvas.draw()
        lines, _ = _annotation_artists(ax)
        baseline_disp = ax.transData.transform((0.0, lines[0].get_ydata()[0]))[1]

        fig2, ax2 = plt.subplots()
        ax2.set_ylim(-5, 50)
        _draw_simple_bar(ax2, heights=(2.0, 4.0), labels=("ctrl", "trt"))
        ps.annotate_significance(ax2, result, groups=("ctrl", "trt"), y_offset=-10)
        fig2.canvas.draw()
        lines2, _ = _annotation_artists(ax2)
        shifted_disp = ax2.transData.transform((0.0, lines2[0].get_ydata()[0]))[1]
        expected = -10.0 * fig2.dpi / 72.0
        assert (shifted_disp - baseline_disp) == pytest.approx(expected, abs=0.5)

    def test_y_offset_preserves_display_distance_when_headroom_expands(self):
        result = _direct_two_group_result(p=0.01)

        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(2.0, 4.0))
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"))
        fig.canvas.draw()
        lines, _ = _annotation_artists(ax)
        baseline_disp = ax.transData.transform((0.0, lines[0].get_ydata()[0]))[1]

        fig2, ax2 = plt.subplots()
        _draw_simple_bar(ax2, heights=(2.0, 4.0))
        ps.annotate_significance(ax2, result, groups=("ctrl", "trt"), y_offset=18)
        fig2.canvas.draw()
        lines2, _ = _annotation_artists(ax2)
        shifted_disp = ax2.transData.transform((0.0, lines2[0].get_ydata()[0]))[1]

        expected = 18.0 * fig2.dpi / 72.0
        assert (shifted_disp - baseline_disp) == pytest.approx(expected, abs=0.5)

    def test_y_offset_preserves_level_spacing(self):
        fig, ax = plt.subplots()
        ax.set_ylim(-5, 50)
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0, 4.0), labels=("a", "b", "c", "d"))
        result = _posthoc_pairwise_result([("a", "b", 0.01), ("a", "c", 0.01), ("a", "d", 0.01)])
        ps.annotate_significance(ax, result)
        fig.canvas.draw()
        lines, _ = _annotation_artists(ax)
        baseline_levels = sorted(
            ax.transData.transform((0.0, line.get_ydata()[0]))[1] for line in lines
        )
        baseline_spacing = baseline_levels[1] - baseline_levels[0]

        fig2, ax2 = plt.subplots()
        ax2.set_ylim(-5, 50)
        _draw_simple_bar(ax2, heights=(1.0, 2.0, 3.0, 4.0), labels=("a", "b", "c", "d"))
        ps.annotate_significance(ax2, result, y_offset=8)
        fig2.canvas.draw()
        lines2, _ = _annotation_artists(ax2)
        shifted_levels = sorted(
            ax2.transData.transform((0.0, line.get_ydata()[0]))[1] for line in lines2
        )
        shifted_spacing = shifted_levels[1] - shifted_levels[0]
        assert shifted_spacing == pytest.approx(baseline_spacing, abs=0.5)

    def test_y_offset_applies_to_letter_mode(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        result = _posthoc_pairwise_result([("a", "b", 0.2), ("b", "c", 0.2), ("a", "c", 0.01)])
        ps.annotate_significance(ax, result, mode="letters", comparisons="all")
        fig.canvas.draw()
        _, baseline_texts = _annotation_artists(ax)
        renderer = fig.canvas.get_renderer()
        baseline_tops = sorted(t.get_window_extent(renderer).ymin for t in baseline_texts)

        fig2, ax2 = plt.subplots()
        _draw_simple_bar(ax2, heights=(1.0, 2.0, 3.0), labels=("a", "b", "c"))
        ps.annotate_significance(ax2, result, mode="letters", comparisons="all", y_offset=10)
        fig2.canvas.draw()
        _, shifted_texts = _annotation_artists(ax2)
        renderer2 = fig2.canvas.get_renderer()
        shifted_tops = sorted(t.get_window_extent(renderer2).ymin for t in shifted_texts)

        expected = -10.0 * fig2.dpi / 72.0
        for base, shifted in zip(baseline_tops, shifted_tops, strict=True):
            assert (base - shifted) == pytest.approx(expected, abs=0.5)

    def test_y_offset_works_on_inverted_axis(self):
        fig, ax = plt.subplots()
        ax.bar(["a", "b"], [10.0, 5.0])
        ax.invert_yaxis()
        result = _direct_two_group_result(p=0.01)
        ps.annotate_significance(ax, result, groups=("a", "b"))
        fig.canvas.draw()
        lines, _ = _annotation_artists(ax)
        baseline_disp = ax.transData.transform((0.0, lines[0].get_ydata()[0]))[1]

        fig2, ax2 = plt.subplots()
        ax2.bar(["a", "b"], [10.0, 5.0])
        ax2.invert_yaxis()
        original_ylim = ax2.get_ylim()
        ps.annotate_significance(ax2, result, groups=("a", "b"), y_offset=9)
        fig2.canvas.draw()
        lines2, _ = _annotation_artists(ax2)
        shifted_disp = ax2.transData.transform((0.0, lines2[0].get_ydata()[0]))[1]

        expected = 9.0 * fig2.dpi / 72.0
        assert (shifted_disp - baseline_disp) == pytest.approx(expected, abs=0.5)
        assert ax2.get_ylim()[1] < original_ylim[1]

    @pytest.mark.parametrize("offset", [np.float32(2), np.float64(2), np.int64(2)])
    def test_numpy_real_y_offset_is_accepted(self, offset):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=0.01)
        ps.annotate_significance(ax, result, groups=("ctrl", "trt"), y_offset=offset)
        assert _annotation_artists(ax)[0]

    @pytest.mark.parametrize("bad", [True, "10", float("nan"), float("inf")])
    def test_invalid_y_offset_raises(self, bad):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=0.01)
        with pytest.raises(ValueError, match="y_offset"):
            ps.annotate_significance(ax, result, groups=("ctrl", "trt"), y_offset=bad)

    @pytest.mark.parametrize("bad", [-1, 0, True, 2.5, "2"])
    def test_invalid_p_decimals_raises(self, bad):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = _direct_two_group_result(p=0.01)
        with pytest.raises(ValueError, match="p_decimals"):
            ps.annotate_significance(ax, result, groups=("ctrl", "trt"), p_decimals=bad)

    def test_invalid_groups_length_raises(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax, labels=("a", "b"))
        result = _direct_two_group_result()
        with pytest.raises(ValueError, match="groups"):
            ps.annotate_significance(ax, result, groups=("a", "b", "c"))
        with pytest.raises(ValueError, match="groups"):
            ps.annotate_significance(ax, result, groups=("a", "a"))

    def test_horizontal_axes_rejected(self):
        fig, ax = plt.subplots()
        ax.barh(["a", "b"], [1.0, 2.0])
        result = _direct_two_group_result(p=0.01)
        with pytest.raises(ValueError, match="(?i)vertical|horizontal"):
            ps.annotate_significance(ax, result, groups=("a", "b"))

    def test_unsupported_axes_raises(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="polar")
        result = _direct_two_group_result(p=0.01)
        with pytest.raises(ValueError, match="(?i)cartesian|2.?d|polar"):
            ps.annotate_significance(ax, result, groups=("a", "b"))

    def test_three_dimensional_axes_raises(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        result = _direct_two_group_result(p=0.01)
        with pytest.raises(ValueError, match="(?i)cartesian|2.?d|3.?d"):
            ps.annotate_significance(ax, result, groups=("a", "b"))

    def test_reverse_duplicate_pair_with_colliding_hashes_raises(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        a = _CollidingLabel("a")
        b = _CollidingLabel("b")
        table = pd.DataFrame({"A": [a, b], "B": [b, a], "p": [0.01, 0.02]})
        with pytest.raises(ValueError, match="(?i)duplicate"):
            ps.annotate_significance(
                ax,
                _direct_two_group_result(),
                comparison_table=table,
                label_map={a: "ctrl", b: "trt"},
            )

    @pytest.mark.parametrize("pairwise", ["not a table", ["not", "a", "table"]])
    def test_invalid_root_pairwise_raises(self, pairwise):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        result = TestResult(
            test_name="Custom", statistic=1.0, p_value=0.01, effect_size={}, pairwise=pairwise
        )
        with pytest.raises(ValueError, match="pairwise.*DataFrame"):
            ps.annotate_significance(ax, result, groups=("ctrl", "trt"))

    def test_invalid_posthoc_pairwise_raises(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        posthoc = TestResult(
            test_name="Posthoc", statistic=1.0, p_value=0.01, effect_size={}, pairwise="not a table"
        )
        result = TestResult(
            test_name="Primary", statistic=1.0, p_value=0.01, effect_size={}, posthoc=posthoc
        )
        with pytest.raises(ValueError, match="pairwise.*DataFrame"):
            ps.annotate_significance(ax, result, groups=("ctrl", "trt"))

    def test_rc_context_does_not_mutate_globals(self):
        fig, ax = plt.subplots()
        _draw_simple_bar(ax)
        # Capture a known rcParam
        before = rcParams["font.size"]
        ps.annotate_significance(
            ax,
            _direct_two_group_result(0.01),
            groups=("ctrl", "trt"),
            rc={"font.size": 24.0},
        )
        assert rcParams["font.size"] == before
        # The created text should use the larger size
        _, texts = _annotation_artists(ax)
        assert texts[0].get_fontsize() == 24.0
