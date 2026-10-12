"""Stage boundary and provenance for Hydra runs; no backend imports or model loading."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

from scientific_tot.configuration import STAGES, absolute, resolve_config


class FreshOutput(Callback):
    """Reserve run/sweep roots before Hydra writes logging or multirun metadata."""

    @staticmethod
    def _reserve(path: str, config: DictConfig) -> None:
        target = absolute(path, Path.cwd())
        inputs = [config.dataset.root, *config.dataset.artifacts.values()]
        for value in inputs:
            source = absolute(value, Path.cwd())
            if target.is_relative_to(source) or source.is_relative_to(target):
                raise SystemExit(f"Hydra output {target} overlaps dataset input {source}")
        try:
            target.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            # Hydra catches Exception from callbacks; SystemExit ensures refusal is fatal.
            raise SystemExit(f"Refusing to overwrite existing Hydra output: {target}") from exc

    def on_run_start(self, config: DictConfig, **kwargs) -> None:
        try:
            task = OmegaConf.masked_copy(config, [key for key in config if key != "hydra"])
            resolve_config(task, Path.cwd(), absolute(config.hydra.run.dir, Path.cwd()))
            self._reserve(config.hydra.run.dir, config)
        except Exception as exc:
            # Callback exceptions are only warnings in Hydra. Guard failures must stop writes.
            raise SystemExit(f"Cannot validate Hydra output: {exc}") from exc

    def on_multirun_start(self, config: DictConfig, **kwargs) -> None:
        try:
            self._reserve(config.hydra.sweep.dir, config)
        except Exception as exc:
            raise SystemExit(f"Cannot validate Hydra sweep output: {exc}") from exc


def execute_stage(cfg: DictConfig) -> None:
    """Connect real adapters/backends here in later phases, one operation per stage."""
    name = cfg.stage.name
    if name not in STAGES:
        raise ValueError(f"Unsupported stage: {name}")
    raise NotImplementedError(
        f"stage={name} is not connected to execution yet. "
        "Use dry_run=true to validate configuration. "
        "The existing examples remain runnable independently."
    )


def git_commit(root: Path) -> str | None:
    """Installed wheels outside a source checkout have no live Git provenance."""
    if not (root / ".git").exists():
        return None
    try:
        return subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_stage(cfg: DictConfig, run_dir: Path, hydra_cfg: DictConfig, repo_root: Path) -> None:
    """Preserve resolved settings and failed attempts without claiming a successful stage."""
    # Hydra creates the directory before invoking us. Guard job subdirs as well.
    if any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty run directory: {run_dir}")
    metadata_dir = run_dir / ".hydra"
    metadata_dir.mkdir(exist_ok=False)
    OmegaConf.save(cfg, metadata_dir / "config.yaml", resolve=True)
    OmegaConf.save(hydra_cfg, metadata_dir / "hydra.yaml", resolve=False)
    OmegaConf.save(hydra_cfg.overrides.task, metadata_dir / "overrides.yaml")
    manifest = {
        "status": "started",
        "dataset_revision": cfg.dataset.revision,
        "local_revision_verified": False if cfg.dataset.source == "local" else None,
        "model_revision": cfg.model.get("revision"),
        "model_identifier": cfg.model.get("model_id", cfg.model.get("checkpoint")),
        "git_commit": git_commit(repo_root),
        "output_dir": str(run_dir),
        "index_location": cfg.paths.index
        or (str(run_dir / "index") if cfg.stage.name == "index" else None),
        "overrides": OmegaConf.to_container(hydra_cfg.overrides.task),
    }
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    try:
        execute_stage(cfg)
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        manifest["status"] = "completed"
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
