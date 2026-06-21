# Agent Guide

This is the reporsitory for a library that automates significance testing for common biological and life sciences data. It provides a simple interface to perform statistical tests on pandas dataframes.

## Working Rules

- Use `uv` for project commands: `uv run ...`, not bare `python`, `pip`, or global tools. Chek the `uv` section below for details.
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
  result.py            # TestResult dataclass + to_dataframe/summary/__rich__
  dispatcher.py        # test() walks the flowchart
  __init__.py          # public re-exports
tests/
  test_data.py
  test_assumptions.py
  test_result.py
  test_tests_continuous.py
  test_posthoc.py
  test_dispatcher.py
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
tables / per-group normality), and `extra` (free-form metadata).

`to_dataframe()` flattens nested dicts (effect sizes, assumptions) into
columns; list values like `CI95%` expand into `CI95%_0`, `CI95%_1`. The
module-level `to_dataframe()` concatenates several results with a column union.

### Pytest collection gotchas

- `TestResult` is a dataclass whose name starts with `Test` — it sets
  `__test__ = False` so pytest skips it.
- The dispatcher function is named `test` — it also sets `__test__ = False`.

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
  input, dataframe export, rich printing, auto-dispatcher.
- **Phase 2:** counts (Poisson / negative binomial regression with offset),
  proportions (Fisher's exact, logistic regression, beta-binomial).
- **Phase 3:** survival (lifelines: Kaplan-Meier + log-rank, Cox regression),
  mixed-effects models for nested cell/axon/image data, batch
  multiple-comparison correction (Holm / Benjamini-Hochberg) across many tests.

## Python dependency and tool usage with `uv`

Use `uv` for Python dependency management, virtual environments, and tool execution. Do not use `pip`, `pipx`, `poetry`, or manual virtualenv commands unless explicitly required.

### Project setup

```bash
uv sync
```

This creates/updates the project environment from `pyproject.toml` and `uv.lock`.

### Adding dependencies

Add runtime dependencies with:

```bash
uv add requests
uv add "fastapi[standard]"
uv add "pydantic>=2"
```

Add development dependencies with:

```bash
uv add --dev pytest
uv add --dev ruff
```

Add optional dependencies/extras with:

```bash
uv add --optional cli typer
```

Remove dependencies with:

```bash
uv remove requests
```

After changing dependencies, commit both:

```text
pyproject.toml
uv.lock
```

### Running code

Run commands inside the project environment with `uv run`:

```bash
uv run python main.py
uv run python -m my_package
uv run pytest
uv run ruff check .
uv run ruff format .
```

Avoid activating the virtual environment manually. Prefer `uv run ...` so commands use the locked project environment.

### Running scripts

For a normal project script:

```bash
uv run python scripts/example.py
```

For a standalone script with inline dependencies:

```bash
uv add --script scripts/example.py requests
uv run --script scripts/example.py
```

### Tool usage with `uvx`

Use `uvx` for one-off Python tools that should not become project dependencies.

Examples:

```bash
uvx ruff check .
uvx black .
```

Run a specific tool version:

```bash
uvx ruff@0.6.9 check .
uvx ruff@latest check .
```

Use `--from` when the executable name differs from the package name or when pinning the package explicitly:

```bash
uvx --from "ruff==0.6.9" ruff check .
```

Install a tool user-wide only when it is used frequently outside this project:

```bash
uv tool install ruff
uv tool list
uv tool uninstall ruff
```

### Common workflow

```bash
uv sync
uv add <package>
uv add --dev <dev-package>
uv run pytest
uv run ruff check .
uv run ruff format .
```

### Agent rules

- Use `uv add` instead of editing dependencies manually when possible.
- Use `uv run` for project commands.
- Use `uvx` for temporary, one-off tools.
- Do not use `pip install` directly.
- Do not rely on an activated shell environment.
- Keep `uv.lock` updated and committed.


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
