#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/defaults.sh
exec uv run --locked python -m scientific_tot.stage_cli index "$@"
