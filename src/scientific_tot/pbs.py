"""PBS submission adapter following the dataset producer's qsub/worker contract."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from scientific_tot.job_logging import log_event, run_logged

ROOT = Path(__file__).resolve().parents[2]


def is_worker() -> bool:
    worker = os.environ.get("TOT_PBS_WORKER") == "1"
    if worker and not os.environ.get("PBS_JOBID"):
        raise RuntimeError("A PBS worker requires PBS_JOBID from the scheduler")
    return worker


def submit(cfg, arguments: list[str], *, name: str, colqwen: bool = False) -> None:
    """Submit an existing Python entry point; workers reuse the ordinary stage code."""
    launcher = cfg.launcher
    assert launcher.kind == "pbs"
    log_root = Path(launcher.log_root)
    if cfg.dry_run:
        directory = log_root / f"{name}-preview"
    else:
        if shutil.which("qsub") is None:
            raise RuntimeError(
                "launcher=h200 submits PBS jobs, but qsub is not on PATH. "
                "Run on the PBS submission host, or use launcher=local_gpu inside an allocation."
            )
        log_root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix=f"{name}-", dir=log_root))
    # Preserve venv executable symlinks: resolve() would select the base interpreter.
    python = Path(launcher.colqwen_python if colqwen else launcher.python).expanduser()
    if not python.is_absolute():
        python = ROOT / python
    command = [str(python), "-u", "-m", *arguments]
    stage = "benchmark" if arguments[0] == "scientific_tot.benchmark" else arguments[1]
    log_path = directory / f"{stage}.log"
    plan_path = directory / "plan.json"
    worker = [str(python), "-u", "-m", "scientific_tot.pbs", "--worker", str(plan_path)]
    environment = {
        key: os.environ[key]
        for key in (
            "PATH",
            "CUDA_HOME",
            "CUDA_PATH",
            "TORCH_CUDA_ARCH_LIST",
            "CXX",
            "CC",
            "LD_LIBRARY_PATH",
            "LIBRARY_PATH",
            "HF_HOME",
            "HF_HUB_CACHE",
            "HF_TOKEN_PATH",
            "PYTERRIER_HOME",
            "HF_HUB_OFFLINE",
            "TRANSFORMERS_OFFLINE",
            "UV_OFFLINE",
            "MAX_JOBS",
            "COLBERT_LOAD_TORCH_EXTENSION_VERBOSE",
        )
        if key in os.environ
    }
    environment.update(
        TOT_DATA_ROOT=launcher.dataset_root,
        TOT_CACHE_ROOT=launcher.cache_root,
        TOT_OUTPUT_ROOT=launcher.output_root,
        TOT_TEST_ROOT=cfg.test_root,
        TOT_DATA_REVISION=cfg.dataset.revision or "",
        TOT_LOG_ROOT=launcher.log_root,
        TOT_PYTHON=launcher.python,
        TOT_COLQWEN_PYTHON=launcher.colqwen_python,
        TOT_INDEXING_THREADS=str(launcher.threads.indexing),
        TOT_ENCODING_THREADS=str(launcher.threads.encoding),
        TOT_FAISS_THREADS=str(launcher.threads.faiss),
        VTT_PBS_PROJECT=launcher.project,
        VTT_PBS_QUEUE=launcher.queue,
        HYDRA_FULL_ERROR="1",
        UV_CACHE_DIR=str(Path(launcher.cache_root) / "uv"),
    )
    if cfg.dataset.revision is None:
        environment.pop("TOT_DATA_REVISION")
    environment.update(OmegaConf.to_container(launcher.environment, resolve=True))
    # Stage wrappers run the already installed default/ColQwen environments on
    # the compute node. Never carry a submitter's active venv into both models.
    body = "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            ': "${PBS_JOBID:?This script must run inside a PBS allocation}"',
            f"cd {shlex.quote(str(ROOT))}",
            f"export PYTHONPATH={shlex.quote(str(ROOT / 'src'))}${{PYTHONPATH:+:$PYTHONPATH}}",
            "unset UV_PROJECT_ENVIRONMENT TOT_DATA_REVISION",
            *[f"export {key}={shlex.quote(str(value))}" for key, value in environment.items()],
            "export TOT_PBS_WORKER=1 UV_NO_SYNC=1",
            "exec " + shlex.join(worker),
            "",
        ]
    )
    # rt_HC is the producer's whole-node queue; rt_HG requests explicit resources.
    select = (
        "select=1"
        if launcher.queue == "rt_HC"
        else (f"select=1:ncpus={launcher.cpus}:mem={launcher.memory}:ngpus={launcher.gpus}")
    )
    qsub = [
        "qsub",
        "-N",
        name[:63],
        "-P",
        launcher.project,
        "-q",
        launcher.queue,
        "-l",
        select,
        "-l",
        f"walltime={launcher.walltime}",
        "-o",
        str(directory / "stdout.log"),
        "-e",
        str(directory / "stderr.log"),
        str(directory / "job.sh"),
    ]
    print("$ " + shlex.join(qsub), flush=True)
    if cfg.dry_run:
        print(body, end="", flush=True)
        print("Task: " + shlex.join(command), flush=True)
        return
    (directory / "job.sh").write_text(body, encoding="utf-8")
    OmegaConf.save(cfg, directory / "config.yaml", resolve=True)
    # Publish the plan and visible stage log before qsub: a worker may start
    # before the submission command has returned its job ID.
    plan = {"command": command, "log": str(log_path), "name": name}
    (directory / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    log_event(log_path, f"{name} job submitting")
    result = subprocess.run(qsub, cwd=ROOT, capture_output=True, text=True)
    (directory / "submission.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        log_event(
            log_path, f"{name} submission failed ({result.returncode}): {result.stderr.strip()}"
        )
        raise RuntimeError(f"qsub failed ({result.returncode}): {result.stderr.strip()}")
    job_id = result.stdout.strip()
    if not job_id:
        log_event(log_path, f"{name} submission failed: qsub returned no job ID")
        raise RuntimeError(f"qsub returned no job ID; inspect {directory / 'submission.log'}")
    log_event(log_path, f"{name} job submitted: {job_id}")
    (directory / "submission.json").write_text(
        json.dumps({"job_id": job_id, "command": qsub, "worker": worker}, indent=2) + "\n"
    )
    print(
        f"Submitted PBS job {job_id}\nSubmission and logs: {directory}\nStage log: {log_path}",
        flush=True,
    )


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise ValueError("Use the stage scripts to submit PBS jobs")
    if not is_worker():
        raise RuntimeError("PBS workers must be started by the generated job script")
    plan = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
    raise SystemExit(run_logged(plan["command"], Path(plan["log"]), label=plan["name"], cwd=ROOT))


if __name__ == "__main__":
    main()
