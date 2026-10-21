"""Stage dispatch and provenance for Hydra runs; backends load only during execution."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf

from scientific_tot.config_artifacts import (
    artifact_reference,
    digest,
    verify_index,
    verify_retrieval,
)
from scientific_tot.configuration import VISUAL_DENSE_MODELS, absolute, overlaps, resolve_config


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
            if overlaps(target, source):
                raise SystemExit(f"Hydra output {target} overlaps protected input/cache {source}")
        return target

    @classmethod
    def _reserve(cls, path: str, config: DictConfig, *, reuse=False) -> None:
        target = cls._protect(path, config)
        try:
            target.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            if reuse:
                task = OmegaConf.masked_copy(config, [k for k in config if k != "hydra"])
                cfg = resolve_config(task, Path.cwd(), target)
                if not cfg.dry_run and cfg.stage.name == "index":
                    return
                if not cfg.dry_run:
                    from scientific_tot.artifact_identity import resolve_identity

                    verify_completed(target, cfg, resolve_identity(cfg))
                    return
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
            self._reserve(config.hydra.run.dir, config, reuse=True)
        except Exception as exc:
            # Callback exceptions are only warnings in Hydra. Guard failures must stop writes.
            raise SystemExit(f"Cannot validate Hydra output: {exc}") from exc

    def on_multirun_start(self, config: DictConfig, **kwargs) -> None:
        try:
            self._check_hydra(config)
            subdir = OmegaConf.to_container(config.hydra.sweep, resolve=False)["subdir"]
            if subdir != "${artifact_dir:}":
                raise ValueError("hydra.sweep.subdir must use the deterministic artifact layout")
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

        destinations = {}
        for number, overrides in enumerate(jobs):
            job = loader.load_sweep_config(config, overrides)
            job.hydra.job.num = number
            self._check_hydra(job)
            task = OmegaConf.masked_copy(job, [k for k in job if k != "hydra"])
            directory = sweep / job.hydra.sweep.subdir
            resolved = resolve_config(task, Path.cwd(), directory)
            if not resolved.dry_run:
                from scientific_tot.artifact_identity import resolve_identity

                identity = {} if resolved.stage.name == "index" else resolve_identity(resolved)
                for previous, previous_identity in destinations.items():
                    if overlaps(directory, previous) and (
                        directory != previous or identity != previous_identity
                    ):
                        raise ValueError(
                            "Conflicting sweep artifacts must use distinct destinations"
                        )
                destinations[directory] = identity
            self._protect(str(sweep), job)
            self._protect(str(directory), job)


def execute_stage(cfg: DictConfig, *, run_dir: Path | None = None) -> dict | None:
    """Dispatch one operation using the already resolved application settings."""
    name = cfg.stage.name
    if cfg.model.name in {"bm25", "dense", "colbert", "colqwen"} | VISUAL_DENSE_MODELS and name in {
        "prepare",
        "index",
        "retrieve",
        "evaluate",
    }:
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
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run_stage(cfg: DictConfig, run_dir: Path, hydra_cfg: DictConfig, repo_root: Path) -> None:
    if cfg.stage.name == "index" and cfg.model.name == "colqwen":
        import portalocker

        with portalocker.Lock(run_dir / ".run.lock", mode="a+", timeout=0):
            _run_stage(cfg, run_dir, hydra_cfg, repo_root)
    else:
        _run_stage(cfg, run_dir, hydra_cfg, repo_root)


def _run_stage(cfg: DictConfig, run_dir: Path, hydra_cfg: DictConfig, repo_root: Path) -> None:
    """Preserve resolved settings and failed attempts without claiming a successful stage."""
    # Hydra creates directories but automatic snapshots/logging are disabled.
    cfg = resolve_config(cfg, Path.cwd(), run_dir)
    print(f"Starting {cfg.stage.name}: {cfg.run.name}; output={run_dir}", flush=True)
    from scientific_tot.artifact_identity import INDEX_EXISTENCE

    contents = {path.name for path in run_dir.iterdir()}
    if cfg.stage.name == "index" and cfg.model.name == "colqwen":
        contents.discard(".run.lock")
    if cfg.stage.name == "index" and (
        INDEX_EXISTENCE.get(run_dir, False)
        or (Path(cfg.paths.index).exists() and Path(cfg.paths.index) != run_dir)
        or contents
    ):
        if cfg.model.name == "colqwen":
            location = Path(cfg.paths.index)
            saved = location / "manifest.json"
            if saved.is_file() and json.loads(saved.read_text()).get("status") in {
                "started",
                "failed",
            }:
                from scientific_tot.recover_colqwen import _recover_locked, recover_index

                if location == run_dir:
                    _recover_locked(location, requested=cfg)
                else:
                    recover_index(location, requested=cfg)
                return
        if cfg.model.name in {"colbert", "colqwen"} | VISUAL_DENSE_MODELS:
            from scientific_tot.artifact_identity import resolve_identity

            verify_completed(Path(cfg.paths.index), cfg, resolve_identity(cfg))
            print(f"Reusing verified index: {cfg.paths.index}", flush=True)
        else:
            print(f"Skipping existing index: {cfg.paths.index}", flush=True)
        return
    if contents:
        from scientific_tot.artifact_identity import resolve_identity

        # Fail unfinished reservations before attempting any expensive source reads.
        saved = run_dir / "manifest.json"
        if not saved.is_file() or json.loads(saved.read_text()).get("status") != "completed":
            raise FileExistsError(f"Refusing to overwrite incomplete artifact: {run_dir}")
        verify_completed(run_dir, cfg, resolve_identity(cfg))
        print(f"Reusing verified {cfg.stage.name} artifact: {run_dir}", flush=True)
        return
    # Exclusive lock also protects direct callers racing on the same empty directory.
    if not (cfg.stage.name == "index" and cfg.model.name == "colqwen"):
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
        "schema_version": 2,
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
        "index_location": cfg.paths.index or (str(run_dir) if cfg.stage.name == "index" else None),
        "overrides": OmegaConf.to_container(hydra_cfg.overrides.task),
    }
    manifest_path = run_dir / "manifest.json"
    with manifest_path.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    try:
        from scientific_tot.artifact_identity import PLANNED_IDENTITIES, resolve_identity

        print("Resolving and verifying input identity...", flush=True)
        identity = resolve_identity(cfg)
        if run_dir in PLANNED_IDENTITIES and PLANNED_IDENTITIES[run_dir] != identity:
            raise ValueError("Source inputs differ from the reserved artifact identity")
        manifest.update(artifact_identity=identity, artifact_sha256=digest(identity))
        if cfg.stage.name == "index":
            effective = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
            effective.dataset.revision = identity["index"]["dataset"]["revision"]
            if cfg.model.name == "dense":
                effective.model.revision = identity["index"]["model"]["revision"]
            OmegaConf.set_readonly(effective, True)
            cfg = effective
            manifest["resolved_dataset_revision"] = cfg.dataset.revision
            manifest["resolved_model_revision"] = cfg.model.get("revision")
        if cfg.stage.name in {"retrieve", "search"}:
            manifest["index_provenance"] = verify_index(Path(cfg.paths.index), cfg)
        if cfg.stage.name == "evaluate":
            manifest["retrieval_provenance"] = verify_retrieval(Path(cfg.paths.run), cfg)
        result = execute_stage(cfg, run_dir=run_dir)
        if result is not None:
            manifest["artifacts"] = result
        print("Verifying and sealing outputs...", flush=True)
        if resolve_identity(cfg) != identity:
            raise ValueError("Source inputs changed during artifact construction")
        if cfg.stage.name == "index":
            built = verify_index(Path(cfg.paths.index), cfg, allow_building=True)
            if built.get("artifact_identity") != identity:
                raise ValueError("Built index differs from reserved artifact identity")
        # Seal every stage output, including preparation sidecars and index bytes.
        manifest["output_files"] = {
            str(path.relative_to(run_dir)): artifact_reference(path)
            for path in sorted(run_dir.rglob("*"))
            if path.is_file() and path.name not in {"manifest.json", ".run.lock"}
        }
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    else:
        manifest["status"] = "completed"
        print(f"Completed {cfg.stage.name} artifact: {run_dir}", flush=True)
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        manifest_path.chmod(0o444)


def verify_completed(directory: Path, cfg: DictConfig, identity: dict) -> dict:
    """Reuse only sealed, identical artifacts; legacy/failed/in-progress paths fail closed."""
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest.get("status") != "completed"
            or manifest.get("schema_version") != 2
            or manifest.get("stage") != cfg.stage.name
            or manifest.get("artifact_identity") != identity
            or manifest.get("artifact_sha256") != digest(identity)
        ):
            raise ValueError("Artifact is incomplete, legacy, or has a different identity")
        snapshot = OmegaConf.to_container(
            OmegaConf.load(directory / ".hydra/config.yaml"), resolve=True
        )
        if digest(snapshot) != manifest["configuration_sha256"]:
            raise ValueError("Artifact configuration snapshot checksum differs")
        files = manifest.get("output_files")
        if not files:
            raise ValueError("Artifact has no sealed output inventory")
        actual = {
            str(path.relative_to(directory))
            for path in directory.rglob("*")
            if path.is_file() and path.name not in {"manifest.json", ".run.lock"}
        }
        if actual != set(files):
            raise ValueError("Artifact output inventory differs")
        for name, reference in files.items():
            if artifact_reference(directory / name) != reference:
                raise ValueError(f"Artifact output checksum differs: {name}")
        if cfg.stage.name == "index":
            index = verify_index(Path(cfg.paths.index), cfg)
            if index.get("artifact_identity") != identity or not index.get("index_files"):
                raise ValueError("Index lacks a completed content identity/file inventory")
        elif cfg.stage.name == "retrieve":
            verify_retrieval(directory)
        return manifest
    except (OSError, ValueError, KeyError) as exc:
        raise FileExistsError(
            f"Refusing to overwrite or reuse invalid artifact {directory}: {exc}"
        ) from exc


def git_dirty(root: Path) -> bool | None:
    if not (root / ".git").exists():
        return None
    return bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
