#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/defaults.sh
source scripts/environment.sh --
: "${TOT_GENERATION_REPO:?Set TOT_GENERATION_REPO to the dataset generation checkout}"
export PYTHONPATH="${TOT_GENERATION_REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec uv run --locked python -m scientific_tot.stage_dataset "$@"
