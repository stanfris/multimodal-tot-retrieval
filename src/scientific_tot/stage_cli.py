"""Convenient single-stage runs with verified automatic artifact selection."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scientific_tot.config_artifacts import verify_index, verify_retrieval
from scientific_tot.configuration import CONFIG_DIR, VISUAL_DENSE_MODELS, resolve_config


def find_artifact(cfg, stage: str) -> Path | None:
    """Select exact indexes by current corpus identity; saved runs by verified settings."""
    if stage == "index" and cfg.model.name in {"colqwen"} | VISUAL_DENSE_MODELS:
        # Visual retrieval selects completed metadata; originals and rendering are
        # unavailable by contract. Multiple content versions require an explicit path.
        candidates = []
        for path in sorted(Path(cfg.launcher.output_root).glob("indexes/*/index-manifest.json")):
            try:
                verify_index(path.parent, cfg)
                candidates.append(path.parent)
            except (OSError, ValueError, KeyError, TypeError):
                continue
        if len(candidates) > 1:
            raise ValueError("Multiple compatible visual indexes; supply paths.index explicitly")
        return candidates[0] if candidates else None
    if stage == "index":
        from scientific_tot.artifact_identity import artifact_directory, resolve_identity
        from scientific_tot.hydra_run import verify_completed

        build = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        build.stage.name = "index"
        build.paths.index = None
        build.paths.run = None
        build = resolve_config(build, Path.cwd())
        identity = resolve_identity(build)
        location = artifact_directory(build, identity)
        if location.exists():
            build = resolve_config(build, Path.cwd(), location)
            verify_completed(location, build, identity)
            return location
        return None
    candidates = []
    for path in sorted(Path(cfg.launcher.output_root).glob("models/*/manifest.json")):
        try:
            verify_retrieval(path.parent, cfg)
            saved = OmegaConf.load(path.parent / ".hydra/config.yaml")
            if saved.retrieval.top_k == cfg.retrieval.top_k:
                candidates.append(path.parent)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    if len(candidates) > 1:
        raise ValueError("Multiple compatible saved runs; supply paths.run explicitly")
    return candidates[0] if candidates else None


def plan(stage: str, overrides: list[str]):
    """Compose ordinary Hydra overrides and supply only omitted artifact inputs."""
    if stage not in {"prepare", "index", "retrieve", "evaluate"}:
        raise ValueError(f"Unsupported script stage: {stage}")
    if any(arg.startswith("-") for arg in overrides):
        raise ValueError("Stage scripts accept single-run Hydra key=value overrides only")
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="config", overrides=[*overrides, f"stage={stage}"])
    key = {"retrieve": "index", "evaluate": "run"}.get(stage)
    automatic = key is not None and cfg.paths[key] is None
    if automatic:
        # Validation requires input paths even during configuration-only planning.
        cfg.paths[key] = str(Path.cwd() / ".artifact-selection-placeholder")
    resolved = resolve_config(cfg, Path.cwd())
    command = [*overrides, f"stage={stage}"]
    if automatic:
        if not resolved.dry_run:
            print(f"Selecting compatible {key} artifact...", flush=True)
        artifact = (
            Path(resolved.paths[key])
            if resolved.dry_run
            else find_artifact(resolved, "index" if stage == "retrieve" else "retrieve")
        )
        if artifact is None:
            preceding = "02_index.sh" if stage == "retrieve" else "03_retrieve.sh"
            raise ValueError(
                f"No compatible completed artifact for {resolved.run.name}. "
                f"Run scripts/{preceding} with the same experiment overrides first, "
                f"or supply paths.{key}=... explicitly."
            )
        # JSON quoting is accepted by Hydra and preserves spaces in filesystem paths.
        command.append(f"paths.{key}={json.dumps(str(artifact))}")
        if not resolved.dry_run:
            print(f"Using {key}: {artifact}", flush=True)
    return resolved, command


def main() -> None:
    args = sys.argv[1:]
    if not args or "--help" in args or "-h" in args:
        print(
            "Usage: scripts/0N_STAGE.sh [Hydra key=value ...]\n"
            "Defaults: BM25/fulltext, ACL, all corpus documents, test queries, CPU, top-100.\n"
            "Examples: scripts/02_index.sh experiment=dense_passages\n"
            "          scripts/03_retrieve.sh retrieval.top_k=200\n"
            "          scripts/04_evaluate.sh 'evaluation.metrics=[RR@10,R@100]'"
        )
        return
    stage, *overrides = args
    cfg, command = plan(stage, overrides)
    if not cfg.dry_run and stage in {"prepare", "index"} and cfg.dataset.source == "local":
        if not Path(cfg.dataset.root).is_dir():
            raise ValueError(
                f"Dataset root does not exist: {cfg.dataset.root}. "
                "Put restored data in ./data or set TOT_DATA_ROOT once in scripts/defaults.sh."
            )
    result = subprocess.run([sys.executable, "-m", "scientific_tot.hydra_entry", *command])
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
