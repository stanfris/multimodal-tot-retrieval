"""Stage dispatch and provenance for Hydra runs; backends load only during execution."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

from scientific_tot.config_artifacts import digest, verify_index, verify_retrieval
from scientific_tot.configuration import STAGES, absolute, overlaps, resolve_config


class FreshOutput(Callback):
    """Reserve run/sweep roots before Hydra writes logging or multirun metadata."""

    @staticmethod
    def _protect(path: str, config: DictConfig) -> Path:
        target = absolute(path, Path.cwd())
        inputs = [
            config.dataset.root,
            *config.dataset.artifacts.values(),
            config.launcher.cache_root,
            config.dataset.cache_dir,
        ]
        if config.stage.name in {"retrieve", "search"} and config.paths.index is not None:
            inputs.append(config.paths.index)
        if config.paths.run is not None:
            inputs.append(config.paths.run)
        for value in inputs:
            source = absolute(value, Path.cwd())
            if target.is_relative_to(source) or source.is_relative_to(target):
                raise SystemExit(f"Hydra output {target} overlaps protected input/cache {source}")
        return target

    @classmethod
    def _reserve(cls, path: str, config: DictConfig) -> None:
        target = cls._protect(path, config)
        try:
            target.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            # Hydra catches Exception from callbacks; SystemExit ensures refusal is fatal.
            raise SystemExit(f"Refusing to overwrite existing Hydra output: {target}") from exc

    @staticmethod
    def _check_hydra(config: DictConfig) -> None:
        if config.hydra.output_subdir is not None:
            raise ValueError("hydra.output_subdir must remain null to preserve exclusive snapshots")
        if config.hydra.job_logging.get("handlers"):
            raise ValueError("Hydra job file logging must remain disabled")

    def on_job_start(self, config: DictConfig, **kwargs) -> None:
        from hydra.core.hydra_config import HydraConfig
        from hydra.utils import get_original_cwd

        try:
            task = OmegaConf.masked_copy(config, [key for key in config if key != "hydra"])
            resolve_config(
                task, Path(get_original_cwd()), Path(HydraConfig.get().runtime.output_dir)
            )
        except Exception as exc:
            raise SystemExit(f"Cannot validate Hydra job: {exc}") from exc

    def on_run_start(self, config: DictConfig, **kwargs) -> None:
        try:
            task = OmegaConf.masked_copy(config, [key for key in config if key != "hydra"])
            resolve_config(task, Path.cwd(), absolute(config.hydra.run.dir, Path.cwd()))
            self._check_hydra(config)
            self._reserve(config.hydra.run.dir, config)
        except Exception as exc:
            # Callback exceptions are only warnings in Hydra. Guard failures must stop writes.
            raise SystemExit(f"Cannot validate Hydra output: {exc}") from exc

    def on_multirun_start(self, config: DictConfig, **kwargs) -> None:
        try:
            self._check_hydra(config)
            subdir = OmegaConf.to_container(config.hydra.sweep, resolve=False)["subdir"]
            if subdir not in {"${hydra.job.num}", "${hydra.job.num}-${run.name}-${stage.name}"}:
                raise ValueError("hydra.sweep.subdir must use the supported numbered job layout")
            self._validate_sweep(config)
            self._reserve(config.hydra.sweep.dir, config)
        except Exception as exc:
            raise SystemExit(f"Cannot validate Hydra sweep output: {exc}") from exc

    def _validate_sweep(self, config: DictConfig) -> None:
        # Hydra 1.3's basic sweeper is pinned. Expand exactly its overrides before
        # it writes multirun.yaml, including params declared in the sweeper config.
        from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
        from hydra.core.global_hydra import GlobalHydra
        from hydra.core.override_parser.overrides_parser import OverridesParser

        if (
            config.hydra.sweeper._target_
            != "hydra._internal.core_plugins.basic_sweeper.BasicSweeper"
        ):
            raise ValueError("Only Hydra's basic sweeper has output preflight support")
        loader = GlobalHydra.instance().hydra.config_loader
        params = config.hydra.sweeper.params or {}
        overrides = [f"{k}={v}" for k, v in params.items()] + list(config.hydra.overrides.task)
        parsed = OverridesParser.create(config_loader=loader).parse_overrides(overrides)
        jobs = BasicSweeper.split_arguments(parsed, None)[0]
        sweep = absolute(config.hydra.sweep.dir, Path.cwd())
        index_destinations = set()
        for number, overrides in enumerate(jobs):
            job = loader.load_sweep_config(config, overrides)
            job.hydra.job.num = number
            self._check_hydra(job)
            task = OmegaConf.masked_copy(job, [k for k in job if k != "hydra"])
            directory = sweep / job.hydra.sweep.subdir
            resolved = resolve_config(task, Path.cwd(), directory)
            self._protect(str(sweep), job)
            if resolved.stage.name == "index":
                if any(overlaps(Path(resolved.paths.index), Path(p)) for p in index_destinations):
                    raise ValueError("Sweep jobs must use distinct paths.index destinations")
                index_destinations.add(resolved.paths.index)


def execute_stage(cfg: DictConfig, *, run_dir: Path | None = None) -> dict | None:
    """Dispatch one operation using the already resolved application settings."""
    name = cfg.stage.name
    if name not in STAGES:
        raise ValueError(f"Unsupported stage: {name}")
    if cfg.model.name in {"bm25", "dense"} and name in {"prepare", "index", "retrieve", "evaluate"}:
        from scientific_tot.stages import execute_text_stage

        return execute_text_stage(cfg, run_dir=run_dir)
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
    cfg = resolve_config(cfg, Path.cwd(), run_dir)
    # Exclusive lock also protects direct callers racing on the same empty directory.
    with (run_dir / ".run.lock").open("x"):
        pass
    metadata_dir = run_dir / ".hydra"
    metadata_dir.mkdir(exist_ok=False)
    OmegaConf.save(cfg, metadata_dir / "config.yaml", resolve=True)
    OmegaConf.save(hydra_cfg, metadata_dir / "hydra.yaml", resolve=False)
    OmegaConf.save(hydra_cfg.overrides.task, metadata_dir / "overrides.yaml")
    for snapshot in metadata_dir.iterdir():
        snapshot.chmod(0o444)
    configuration = OmegaConf.to_container(cfg, resolve=True)
    manifest = {
        "schema_version": 1,
        "stage": cfg.stage.name,
        "experiment_name": cfg.run.name,
        "configuration_sha256": digest(configuration),
        "status": "started",
        "dataset_revision": cfg.dataset.revision,
        "local_revision_verified": False if cfg.dataset.source == "local" else None,
        "model_revision": cfg.model.get("revision"),
        "model_identifier": cfg.model.get("model_id", cfg.model.get("checkpoint")),
        "git_commit": git_commit(repo_root),
        "git_dirty": git_dirty(repo_root),
        "output_dir": str(run_dir),
        "index_location": cfg.paths.index
        or (str(run_dir / "index") if cfg.stage.name == "index" else None),
        "overrides": OmegaConf.to_container(hydra_cfg.overrides.task),
    }
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    try:
        if cfg.stage.name == "index" and Path(cfg.paths.index).exists():
            raise FileExistsError(f"Refusing to overwrite existing index: {cfg.paths.index}")
        if cfg.stage.name in {"retrieve", "search"}:
            manifest["index_provenance"] = verify_index(Path(cfg.paths.index), cfg)
        if cfg.stage.name == "evaluate":
            manifest["retrieval_provenance"] = verify_retrieval(Path(cfg.paths.run), cfg)
        result = execute_stage(cfg, run_dir=run_dir)
        if result is not None:
            manifest["artifacts"] = result
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        manifest["status"] = "completed"
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifest_path.chmod(0o444)


def git_dirty(root: Path) -> bool | None:
    if git_commit(root) is None:
        return None
    try:
        return bool(
            subprocess.run(
                ["git", "-C", str(root), "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None
