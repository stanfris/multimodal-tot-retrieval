#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/defaults.sh
source scripts/environment.sh
exec uv run --locked ${stage_uv_args[@]+"${stage_uv_args[@]}"} python -m scientific_tot.stage_cli evaluate "$@"
