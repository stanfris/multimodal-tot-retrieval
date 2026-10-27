#!/usr/bin/env bash
# ColQwen needs Transformers 5; retain the default textual environment.
# Explicit offline runs use only installed packages and cached model/data files.
if [[ "${HF_HUB_OFFLINE:-0}" == 1 || "${TRANSFORMERS_OFFLINE:-0}" == 1 || "${UV_OFFLINE:-0}" == 1 ]]; then
    export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 UV_OFFLINE=1 UV_NO_SYNC=1
fi
stage_uv_args=()
stage_model=""
stage_experiment=""
for argument in "$@"; do
    case "$argument" in
        model=*) stage_model="${argument#model=}" ;;
        experiment=*) stage_experiment="${argument#experiment=}" ;;
    esac
done
if [[ "$stage_model" == colqwen || ( -z "$stage_model" && "$stage_experiment" == colqwen_pages ) ]]; then
    export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-colqwen}"
    stage_uv_args=(--no-default-groups --extra colqwen --group dev)
fi

if [[ "$stage_model" == qwen3_text || ( -z "$stage_model" && "$stage_experiment" == qwen3_text ) ]]; then
    stage_uv_args=(--group h200)
fi
