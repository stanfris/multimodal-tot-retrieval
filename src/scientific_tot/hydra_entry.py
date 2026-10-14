"""Hydra CLI: scientific-tot or python -m scientific_tot.hydra_entry."""

from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import scientific_tot.artifact_identity  # noqa: F401 - registers the Hydra path resolver
from scientific_tot.configuration import CONFIG_DIR, resolve_config
from scientific_tot.hydra_run import run_stage


@hydra.main(version_base="1.3", config_path=str(CONFIG_DIR), config_name="config")
def main(cfg: DictConfig) -> None:
    runtime = HydraConfig.get()
    run_dir = Path(runtime.runtime.output_dir).resolve()
    resolved = resolve_config(cfg, Path(get_original_cwd()), run_dir)
    if resolved.dry_run:
        print(OmegaConf.to_yaml(resolved, resolve=True))
        return
    run_stage(resolved, run_dir, runtime, Path(__file__).resolve().parents[2])


if __name__ == "__main__":
    main()
