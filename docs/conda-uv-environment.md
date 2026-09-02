# Conda environment managed with uv

This setup creates a Conda environment named `pystars` while letting uv install
the exact dependency versions from `uv.lock`. PyStars itself is installed from
the local checkout in editable mode, so source-code changes are available
immediately without reinstalling the package.

## Prerequisites

- Conda (Miniconda or Anaconda)
- uv
- A local clone of this repository

Run all commands from the repository root.

## Initial setup

The project pins Python 3.11 in `.python-version`:

```powershell
conda create --name pystars python=3.11 --yes
conda activate pystars

# uv recognizes active virtual environments through VIRTUAL_ENV. Conda normally
# sets CONDA_PREFIX instead, so explicitly connect the two.
$env:VIRTUAL_ENV = $env:CONDA_PREFIX

# Copy mode is robust when the uv cache is on cloud-synced Windows storage,
# where hardlinks can fail with OS error 396.
$env:UV_LINK_MODE = "copy"

uv sync --active
```

For Bash or Zsh, use the equivalent environment-variable syntax after
activating Conda:

```bash
conda create --name pystars python=3.11 --yes
conda activate pystars
export VIRTUAL_ENV="$CONDA_PREFIX"
export UV_LINK_MODE=copy
uv sync --active
```

Setting `VIRTUAL_ENV` is important. Without it, `uv sync --active` may create a
separate `.venv` in the repository instead of installing into the Conda
environment.

## Verify the setup

With the `pystars` environment still active:

```powershell
python -c "import pathlib, sys, pystars; print(sys.executable); print(pathlib.Path(pystars.__file__).resolve())"
uv pip show pystars
```

The Python executable should be inside the Conda environment. The PyStars
module path should resolve into this checkout, and `uv pip show pystars` should
report an editable project location pointing to the repository.

## Refresh after pulling changes

Source-only changes need no reinstall because the package is editable. After a
pull that changes `pyproject.toml` or `uv.lock`, resync the environment:

```powershell
conda activate pystars
$env:VIRTUAL_ENV = $env:CONDA_PREFIX
$env:UV_LINK_MODE = "copy"
uv sync --active
```

Run project commands in the active environment as usual:

```powershell
uv run --active pytest
uv run --active ruff check .
```

## Recreate the environment

If the environment becomes inconsistent, recreate and sync it from the lockfile:

```powershell
conda deactivate
conda env remove --name pystars --yes
conda create --name pystars python=3.11 --yes
conda activate pystars
$env:VIRTUAL_ENV = $env:CONDA_PREFIX
$env:UV_LINK_MODE = "copy"
uv sync --active
```

## Prompt for an LLM on another machine

Copy and adapt this prompt:

> In this repository, create a Conda environment named `pystars` using the
> Python version pinned by the project. Activate it, set `VIRTUAL_ENV` to
> `CONDA_PREFIX`, and run `uv sync --active` so uv installs `uv.lock` into the
> Conda environment and installs the local project editably. On Windows or
> cloud-synced storage, set `UV_LINK_MODE=copy`. Verify that `sys.executable` is
> inside the Conda environment and that `pystars.__file__` resolves into this
> checkout. Do not create or use a repository `.venv`. Report the commands used
> and any platform-specific adjustments.
