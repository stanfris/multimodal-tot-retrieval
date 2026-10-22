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

./scripts/02_index.sh "$@"
./scripts/03_retrieve.sh "$@"
./scripts/04_evaluate.sh "$@"
