#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/defaults.sh

[[ "$(uname -s)" == Linux ]] || { echo 'H200 setup requires Linux.' >&2; exit 1; }
source scripts/h200_environment.sh
command -v uv >/dev/null || { echo 'Install uv before running this script.' >&2; exit 1; }
command -v nvidia-smi >/dev/null || { echo 'nvidia-smi is missing; load/install the NVIDIA driver.' >&2; exit 1; }
uv --version
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv
# Do not silently download a different Python runtime or regenerate the dependency lock.
uv python find --no-python-downloads 3.11
UV_PROJECT_ENVIRONMENT=.venv uv sync --locked --python 3.11 --no-python-downloads
UV_PROJECT_ENVIRONMENT=.venv-colqwen uv sync --locked --python 3.11 --no-python-downloads \
  --no-default-groups --extra colqwen --group dev

PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -m scientific_tot.h200_check default "$@"
PATH="$PWD/.venv-colqwen/bin:$PATH" .venv-colqwen/bin/python -m scientific_tot.h200_check colqwen "$@"
