"""PBS submission adapter following the dataset producer's qsub/worker contract."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

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
    environment = {
        key: os.environ[key]
        for key in (
            "PATH",
            "CUDA_HOME",
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
            "exec " + shlex.join(command),
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
        return
    (directory / "job.sh").write_text(body, encoding="utf-8")
    OmegaConf.save(cfg, directory / "config.yaml", resolve=True)
    result = subprocess.run(qsub, cwd=ROOT, capture_output=True, text=True)
    (directory / "submission.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"qsub failed ({result.returncode}): {result.stderr.strip()}")
    job_id = result.stdout.strip()
    if not job_id:
        raise RuntimeError(f"qsub returned no job ID; inspect {directory / 'submission.log'}")
    (directory / "submission.json").write_text(
        json.dumps({"job_id": job_id, "command": qsub, "worker": command}, indent=2) + "\n"
    )
    print(f"Submitted PBS job {job_id}\nSubmission and logs: {directory}", flush=True)
