"""Hydra CLI: scientific-tot or python -m scientific_tot.hydra_entry."""

import sys
from pathlib import Path

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import scientific_tot.artifact_identity  # noqa: F401 - registers the Hydra path resolver
from scientific_tot.configuration import CONFIG_DIR, resolve_config
from scientific_tot.hydra_run import run_stage
from scientific_tot.pbs import is_worker


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="config")
def _main(cfg: DictConfig) -> None:
    runtime = HydraConfig.get()
    run_dir = Path(runtime.runtime.output_dir).resolve()
    resolved = resolve_config(cfg, Path(get_original_cwd()), run_dir)
    if resolved.dry_run:
        print(OmegaConf.to_yaml(resolved, resolve=True))
        return
    run_stage(resolved, run_dir, runtime, Path(__file__).resolve().parents[2])


def main() -> None:
    # Submit before Hydra's artifact callbacks reserve outputs or read the corpus
    # on the login node. CLI introspection retains Hydra's normal flag handling.
    if not is_worker() and not any(argument.startswith("-") for argument in sys.argv[1:]):
        with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
            cfg = compose(config_name="config", overrides=sys.argv[1:])
        if cfg.launcher.kind == "pbs":
            from scientific_tot.stage_cli import main as stage_main

            sys.argv = [sys.argv[0], cfg.stage.name, *sys.argv[1:]]
            stage_main()
            return
    _main()


if __name__ == "__main__":
    main()
