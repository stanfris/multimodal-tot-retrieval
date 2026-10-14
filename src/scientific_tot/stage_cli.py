"""Convenient single-stage runs with verified automatic artifact selection."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scientific_tot.config_artifacts import verify_index, verify_retrieval
from scientific_tot.configuration import CONFIG_DIR, resolve_config


def find_artifact(cfg, stage: str) -> Path | None:
    """Choose the newest compatible completed attempt, including custom index locations."""
    candidates = []
    for path in Path(cfg.launcher.output_root).glob("**/manifest.json"):
        try:
            manifest = json.loads(path.read_text())
            if (
                manifest.get("stage") == stage
                and manifest.get("status") == "completed"
                and manifest.get("experiment_name") == cfg.run.name
            ):
                candidates.append(path)
        except (OSError, ValueError):
            continue
    for path in sorted(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p)), reverse=True):
        try:
            if stage == "index":
                location = json.loads(path.read_text())["index_location"]
                artifact = Path(location)
                verify_index(artifact, cfg)
            else:
                artifact = path.parent
                verify_retrieval(artifact, cfg)
                saved = OmegaConf.load(artifact / ".hydra/config.yaml")
                # Auto-selection follows requested depth; explicit paths retain the
                # evaluator's ability to use any saved depth covering its metrics.
                if saved.retrieval.top_k != cfg.retrieval.top_k:
                    continue
            return artifact
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def plan(stage: str, overrides: list[str], *, rebuild: bool = False):
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
        artifact = find_artifact(resolved, "index" if stage == "retrieve" else "retrieve")
        if artifact is None:
            preceding = "02_index.sh" if stage == "retrieve" else "03_retrieve.sh"
            raise ValueError(
                f"No compatible completed artifact for {resolved.run.name}. "
                f"Run scripts/{preceding} with the same experiment overrides first, "
                f"or supply paths.{key}=... explicitly."
            )
        # JSON quoting is accepted by Hydra and preserves spaces in filesystem paths.
        command.append(f"paths.{key}={json.dumps(str(artifact))}")
        print(f"Using {key}: {artifact}", flush=True)
    if stage == "index" and not rebuild and not resolved.dry_run and resolved.paths.index is None:
        artifact = find_artifact(resolved, "index")
        if artifact is not None:
            print(f"Reusing index: {artifact}\nPass --rebuild to build a fresh index.", flush=True)
            return resolved, None
    return resolved, command


def main() -> None:
    args = sys.argv[1:]
    if not args or "--help" in args or "-h" in args:
        print(
            "Usage: scripts/0N_STAGE.sh [--rebuild (index only)] [Hydra key=value ...]\n"
            "Defaults: BM25/fulltext, ACL, all corpus documents, test queries, CPU, top-100.\n"
            "Examples: scripts/02_index.sh experiment=dense_passages\n"
            "          scripts/03_retrieve.sh retrieval.top_k=200\n"
            "          scripts/04_evaluate.sh 'evaluation.metrics=[RR@10,R@100]'"
        )
        return
    stage, *overrides = args
    rebuild = "--rebuild" in overrides
    overrides = [arg for arg in overrides if arg != "--rebuild"]
    try:
        if rebuild and stage != "index":
            raise ValueError("--rebuild is only supported by the index script")
        cfg, command = plan(stage, overrides, rebuild=rebuild)
        if command is None:
            return
        if not cfg.dry_run and stage in {"prepare", "index"} and cfg.dataset.source == "local":
            if not Path(cfg.dataset.root).is_dir():
                raise ValueError(
                    f"Dataset root does not exist: {cfg.dataset.root}. "
                    "Put restored data in ./data or set TOT_DATA_ROOT once in scripts/defaults.sh."
                )
        result = subprocess.run([sys.executable, "-m", "scientific_tot.hydra_entry", *command])
        raise SystemExit(result.returncode)
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
