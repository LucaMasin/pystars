---
name: uv
description: Use uv for Python dependency management, virtual environments, and tool execution in the PyStars project.
---

# Python dependency and tool usage with `uv`

Use `uv` for Python dependency management, virtual environments, and tool execution. Do not use `pip`, `pipx`, `poetry`, or manual virtualenv commands unless explicitly required.

## Project setup

```bash
uv sync
```

This creates/updates the project environment from `pyproject.toml` and `uv.lock`.

## Adding dependencies

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

## Running code

Run commands inside the project environment with `uv run`:

```bash
uv run python main.py
uv run python -m my_package
uv run pytest
uv run ruff check .
uv run ruff format .
```

Avoid activating the virtual environment manually. Prefer `uv run ...` so commands use the locked project environment.

## Running scripts

For a normal project script:

```bash
uv run python scripts/example.py
```

For a standalone script with inline dependencies:

```bash
uv add --script scripts/example.py requests
uv run --script scripts/example.py
```

## Tool usage with `uvx`

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

## Common workflow

```bash
uv sync
uv add <package>
uv add --dev <dev-package>
uv run pytest
uv run ruff check .
uv run ruff format .
```

## Agent rules

- Use `uv add` instead of editing dependencies manually when possible.
- Use `uv run` for project commands.
- Use `uvx` for temporary, one-off tools.
- Do not use `pip install` directly.
- Do not rely on an activated shell environment.
- Keep `uv.lock` updated and committed.
