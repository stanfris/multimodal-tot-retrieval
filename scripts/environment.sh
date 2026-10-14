#!/usr/bin/env bash
# ColQwen needs Transformers 5; retain the default textual environment.
stage_uv_args=()
for argument in "$@"; do
    if [[ "$argument" == "experiment=colqwen_pages" || "$argument" == "model=colqwen" ]]; then
        export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-.venv-colqwen}"
        stage_uv_args=(--no-default-groups --extra colqwen --group dev)
    fi
done
