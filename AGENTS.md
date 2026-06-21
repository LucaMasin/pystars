# Agent Guide

This is the reporsitory for a library that automates significance testing for common biological and life sciences data. It provides a simple interface to perform statistical tests on pandas dataframes.

## Working Rules

- Use `uv` for project commands: `uv run ...`, not bare `python`, `pip`, or global tools. Chek the `uv` section below for details.
- After adding or changing a feature, check `AGENTS.md` and `README.md` and update them if the guidance or user-facing docs are stale.
- The README file should be focused on user-facing documentation, while AGENTS.md should be focused on internal guidance for contributors and agents. If you see something in the README that is more about internal implementation or contributor guidance, move it to AGENTS.md.
- Use test driven development (TDD) for new features. Add a test in `tests/` before implementing the feature. We do not need to test every single function, but we should have a test for each feature and edge case for the core logic. Chek the TDD section below for details.

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
uv add --dev ruff pyright
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
uvx pyright
uvx cookiecutter gh:some/template
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
