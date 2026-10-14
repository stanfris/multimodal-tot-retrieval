#!/usr/bin/env bash
# Shared defaults for all four stages. Environment variables take precedence.
# Relative paths resolve from the repository root, even when launched elsewhere.
export TOT_DATA_ROOT="${TOT_DATA_ROOT:-./data}"
export TOT_CACHE_ROOT="${TOT_CACHE_ROOT:-./cache}"
export TOT_OUTPUT_ROOT="${TOT_OUTPUT_ROOT:-./results}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TOT_CACHE_ROOT}/uv}"
