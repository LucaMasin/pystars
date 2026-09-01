# Agent Guide

This is the reporsitory for a library that automates significance testing for common biological and life sciences data. It provides a simple interface to perform statistical tests on pandas dataframes.

## Working Rules

- Use `uv` for project commands: `uv run ...`, not bare `python`, `pip`, or global tools. Load the `uv` skill for details.
- After adding or changing a feature, check `AGENTS.md` and `README.md` and update them if the guidance or user-facing docs are stale.
- The README file should be focused on user-facing documentation, while AGENTS.md should be focused on internal guidance for contributors and agents. If you see something in the README that is more about internal implementation or contributor guidance, move it to AGENTS.md.
- Use test driven development (TDD) for new features. Add a test in `tests/` before implementing the feature. We do not need to test every single function, but we should have a test for each feature and edge case for the core logic. Chek the TDD section below for details.

## Architecture

Phase 1 covers continuous-data tests only. The package is organised into thin
modules, each with a single responsibility:

```
src/pystars/
  data.py              # long/wide normalization -> canonical LongData
  assumptions.py       # _check_normality / _check_equal_variance + public wrappers
  tests_continuous.py  # ttest, mannwhitney, wilcoxon, anova, kruskal, anova_twoway
  posthoc.py           # posthoc_tukey, posthoc_games_howell, posthoc_dunn
  corrections.py       # batch and pairwise multiple-comparison corrections
  result.py            # TestResult dataclass + to_dataframe/summary/__rich__
  dispatcher.py        # test() walks the flowchart
  plotting.py          # annotate_significance(ax, result, ...)
  __init__.py          # public re-exports
tests/
  test_data.py
  test_assumptions.py
  test_result.py
  test_tests_continuous.py
  test_posthoc.py
  test_dispatcher.py
  test_plotting.py
```

### Data flow

All entry points (dispatcher, direct test functions, assumption checks) call
`normalize_data(...)` to produce a canonical `LongData` with columns `value`,
`group` (a composite `"a::b"` label for multi-factor designs), optional
`subject`, and the original factor columns. Everything downstream operates on
this canonical form, so tests never re-parse user input.

### Backend

Stats are computed with `pingouin` (t-test, ANOVA, Welch ANOVA, Kruskal-Wallis,
Tukey, Games-Howell, effect sizes), `scipy.stats` (Shapiro-Wilk, Levene), and
`scikit-posthocs` (Dunn's test). `TestResult` wraps their outputs into a
uniform shape.

### Dispatcher rules (continuous branch)

```
n_groups = unique(group)
multi-factor (>=2 cols) -> two-way ANOVA (interaction reported; Tukey posthoc if sig)
n_groups == 2:
    paired?
        normality of differences -> Paired t-test : Wilcoxon
    independent?
        normal?  equal_var? -> Student t : Welch t
        non-normal / small n (<3) -> Mann-Whitney U
n_groups > 2:
    normal & equal_var -> One-way ANOVA  + (sig? Tukey)
    normal & unequal   -> Welch ANOVA    + (sig? Games-Howell)
    non-normal         -> Kruskal-Wallis + (sig? Dunn, default p_adjust="holm")
```

Post-hoc is gated on `p < alpha` and `auto_posthoc=True`.

### Small-n guard

`assumptions.SMALL_N = 3`. Any group with fewer than `SMALL_N` observations
makes normality unreliable, so `_check_normality` returns `passed=False` and
the dispatcher routes to the non-parametric branch.

### TestResult shape

`TestResult` carries `test_name`, `statistic`, `p_value`, `effect_size` (dict),
`assumptions` (dict or None), `posthoc` (TestResult or list or None),
`pairwise` (DataFrame for post-hoc tables), `details` (DataFrame for ANOVA
tables / per-group normality), `extra` (free-form metadata), and optional
multiple-comparison fields (`p_adjusted`, `p_adjust_method`, `p_adjust_alpha`,
`reject`).

`to_dataframe()` flattens nested dicts (effect sizes, assumptions) into
columns; list values like `CI95%` expand into `CI95%_0`, `CI95%_1`. The
module-level `to_dataframe()` concatenates several results with a column union.

### Pytest collection gotchas

- `TestResult` is a dataclass whose name starts with `Test` — it sets
  `__test__ = False` so pytest skips it.
- The dispatcher function is named `test` — it also sets `__test__ = False`.

### Plot annotation module (`plotting.py`)

`annotate_significance(ax, result, ...)` is an annotation layer for an
*existing* Matplotlib axes. It must not call a plotting function, inspect
raw data, or recompute a test. It only reads rendered artists, draws
bracket `Line2D` and label `Text` artists, and expands y limits to keep
them visible.

Key contracts:

- Comparison source priority: caller-supplied `comparison_table` (if any)
  overrides everything; otherwise walk `result.pairwise` and recurse into
  `result.posthoc` (a `TestResult` or list of `TestResult`s). Cycles are
  rejected.
- For tables, with `p_column="auto"` use `p_adjusted` when the column is
  present and non-null, else `p`. Treat `(A, B)` and `(B, A)` as the same
  unordered pair; duplicates within or across tables raise.
- Direct two-group fallback: when no pairwise table is present, the caller
  must pass `groups=(left, right)` — `TestResult` intentionally does not
  retain the source group names. The p-value comes from `p_adjusted` if
  finite, otherwise from `p_value`.
- `y_offset` is a signed display-point value added uniformly to the entire
  bracket stack (and to compact-letter labels). Positive values pad the
  annotation upward in display space, negative values pad it downward.
  Because the offset is in display space, the visual direction matches the
  user's intent on linear, log, and inverted axes. Bracket-to-bracket
  spacing is preserved across levels.
- Tuple `label_map` entries (`(x_category, hue_category)`) target a specific
  dodged cell on a Seaborn hue plot. Scalar `label_map` entries target one
  x tick.
- The resolver calls `ax.figure.canvas.draw()` before reading artist
  geometry, so swarm placement is finalized. Artists created by this module
  are tagged with `gid="pystars_annotation"` so a later call never treats
  them as data geometry.
- Layout uses display-space y geometry: brackets are placed in pixels
  above the rendered top of each comparison's x interval, with a greedy
  closed-interval level assignment. Y-limit expansion preserves the
  original direction (inverted y axes stay inverted).
- Letter mode requires `comparisons="all"` and a complete undirected
  pairwise graph; missing pairs raise an error naming them.
- Label formatting: default 4-significant-figure/scientific; `p_decimals=N`
  switches to fixed decimals with a `<` threshold when the value rounds to
  zero; ignored in stars/letters modes.
- Validation runs in `matplotlib.rc_context(rc=rc)`, so `rc` is scoped to
  the call and never mutates global `rcParams`.
- Reject horizontal `BarContainer`, polar/3-D axes, non-`Axes` objects,
  bad `mode`/`bracket`/`alpha`, bad table schemas, non-finite p-values,
  out-of-range p-values, self-pairs, and duplicate unordered pairs.
  When in doubt, raise a `ValueError` with a message that tells the caller
  what to do.

### Adding a new test

1. Add a function in the appropriate module (e.g. `tests_continuous.py` for a
   continuous test, or a new `tests_counts.py` for Phase 2) that takes a
   dataframe + column args, calls `normalize_data`, runs the test, and returns
   a `TestResult`.
2. Add a test in `tests/` first (TDD). Cover the typical case, the edge case
   (wrong number of groups, missing subject for paired, etc.), and the
   p-value direction.
3. If the test should be reachable from the flowchart, extend `dispatcher.py`
   with a branch and add a dispatcher test asserting the correct test is
   selected for data matching that branch.
4. Re-export the new function from `src/pystars/__init__.py` and add it to
   `__all__`.
5. Update `README.md` (user-facing) and, if the dispatcher rules changed, the
   table above in this file.
6. Run `uv run pytest`, `uv run ruff check .`, `uv run ruff format .`.

## Roadmap

- **Phase 1 (done):** continuous tests — t-test (Student/Welch/paired),
  Mann-Whitney, Wilcoxon, one-way ANOVA, Welch ANOVA, Kruskal-Wallis, two-way
  ANOVA, post-hoc (Tukey / Games-Howell / Dunn), assumption checks, long/wide
  input, dataframe export, rich printing, auto-dispatcher, batch and pairwise
  multiple-comparison correction.
- **Phase 2:** counts (Poisson / negative binomial regression with offset),
  proportions (Fisher's exact, logistic regression, beta-binomial).
- **Phase 3:** survival (lifelines: Kaplan-Meier + log-rank, Cox regression),
  mixed-effects models for nested cell/axon/image data.

## Test Driven Development (TDD)

When adding a new feature, first add a test in the `tests/` directory that defines the expected behavior. Then implement the feature to make the test pass. This ensures that features are well-tested and that edge cases are considered.

### Basic rules

- Add a test for each a new feature before implementing it.
- Tests should cover typical use cases and edge cases.
- Keep tests focused on core logic. Do not add pytest coverage for simple wiring or other thin integration glue.
- Use descriptive test names and comments to clarify the purpose of each test.
- Run tests frequently during development to catch issues early.
- Aim for a good balance of test coverage without over-testing trivial code. Focus on testing the behavior and outcomes of features rather than every single function.
- Run the full suite with `uv run pytest`.
- Use `uv run pytest -k <test_name>` to run specific tests during development.
