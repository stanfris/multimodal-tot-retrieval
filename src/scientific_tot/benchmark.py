"""Sequential orchestration of the existing stage scripts with Hydra overrides."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf

from scientific_tot.configuration import CONFIG_DIR
from scientific_tot.data.schemas import DOMAINS
from scientific_tot.job_logging import run_logged
from scientific_tot.pbs import is_worker, submit
from scientific_tot.stage_cli import plan

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = {
    "prepare": "01_prepare.sh",
    "index": "02_index.sh",
    "retrieve": "03_retrieve.sh",
    "evaluate": "04_evaluate.sh",
}


def jobs(overrides: list[str]):
    """Validate every job before starting the first expensive stage; perform no data I/O."""
    parsed = OverridesParser.create().parse_overrides(overrides)
    orchestration, shared = [], []
    for argument, override in zip(overrides, parsed, strict=True):
        key = override.key_or_group
        if key in {"experiment", "dataset.domains", "stage", "stage.name"}:
            raise ValueError("Select benchmark.experiments/domains; the runner selects each stage")
        if key.startswith("benchmark.") or key == "dry_run":
            orchestration.append(argument)
        if not key.startswith("benchmark."):
            shared.append(argument)
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="benchmark", overrides=orchestration)
    experiments, domains = cfg.benchmark.experiments, cfg.benchmark.domains
    available = {path.stem for path in (CONFIG_DIR / "experiment").glob("*.yaml")}
    for values, choices, label in (
        (experiments, available, "experiments"),
        (domains, set(DOMAINS), "domains"),
    ):
        if not OmegaConf.is_list(values) or not values or any(v not in choices for v in values):
            raise ValueError(f"benchmark.{label} must select from {sorted(choices)}")
        if len(set(values)) != len(values):
            raise ValueError(f"benchmark.{label} contains duplicates")
    if type(cfg.benchmark.prepare) is not bool:
        raise ValueError("benchmark.prepare must be boolean")
    if type(cfg.dry_run) is not bool:
        raise ValueError("dry_run must be boolean")
    # Explicit CLI overrides retain priority over model defaults in every stage.
    shared = ["launcher=h200", *shared]
    stages = list(SCRIPTS) if cfg.benchmark.prepare else ["index", "retrieve", "evaluate"]
    commands = []
    for experiment in experiments:
        for domain in domains:
            arguments = [f"experiment={experiment}", f"dataset.domains=[{domain}]", *shared]
            for stage in stages:
                resolved, _ = plan(stage, [*arguments, "dry_run=true"])
                log_root = (
                    Path(resolved.launcher.log_root)
                    if resolved.launcher.kind == "pbs"
                    else Path(resolved.launcher.output_root) / "logs"
                )
                log = log_root / experiment / domain
                commands.append(
                    (log / f"{stage}.log", [str(ROOT / "scripts" / SCRIPTS[stage]), *arguments])
                )
    return cfg.dry_run, commands


def main() -> None:
    dry_run, commands = jobs(sys.argv[1:])
    # One allocation owns the complete sequential benchmark, including artifact
    # discovery after preceding stages complete. Do not queue independent stages.
    stage = next(key for key, script in SCRIPTS.items() if commands[0][1][0].endswith(script))
    cfg, _ = plan(stage, [*commands[0][1][1:], "dry_run=true"])
    if cfg.launcher.kind == "pbs" and not is_worker():
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        cfg.dry_run = dry_run
        submit(cfg, ["scientific_tot.benchmark", *sys.argv[1:]], name="tot-benchmark")
        return
    for log, command in commands:
        if dry_run:
            print(f"$ {shlex.join(command)}\nLog (append): {log}", flush=True)
            continue
        code = run_logged(command, log, label=log.stem, cwd=ROOT)
        if code:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
