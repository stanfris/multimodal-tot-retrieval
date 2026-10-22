#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

test_enabled=false
for argument in "$@"; do
    if [[ "$argument" == "--test" ]]; then
        test_enabled=true
    fi
done
if [[ "$test_enabled" != true ]]; then
    echo "Usage: scripts/00_test.sh --test [Hydra key=value ...]" >&2
    exit 2
fi

source scripts/defaults.sh
source scripts/environment.sh --
exec uv run --locked python -m scientific_tot.stage_cli test "$@"
