"""Exercise real CLI submission with a fake qsub executable; never require a PBS server."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scientific_tot import benchmark, stage_cli
from scientific_tot.pbs import is_worker

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def submission(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.delenv("TOT_PBS_WORKER", raising=False)
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.setenv("TOT_DATA_ROOT", str(tmp_path / "not-mounted-on-login"))
    monkeypatch.setenv("TOT_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("TOT_OUTPUT_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("TOT_TEST_ROOT", str(tmp_path / "test"))
    monkeypatch.setenv("HF_TOKEN", "must-not-be-written-to-job-script")
    for key in ("VTT_PBS_PROJECT", "VTT_PBS_QUEUE", "TOT_LOG_ROOT", "VTT_LOG_ROOT"):
        monkeypatch.delenv(key, raising=False)
    executable = tmp_path / "bin/qsub"
    executable.parent.mkdir()
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(tmp_path / 'qsub-args.json')!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "print('12345.server')\n"
    )
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(executable.parent) + os.pathsep + os.environ["PATH"])
    return tmp_path


@pytest.mark.parametrize("stage", ["prepare", "index", "retrieve", "evaluate"])
def test_stage_submits_without_loading_sources_or_artifacts(submission, monkeypatch, stage):
    monkeypatch.setenv("CUDA_PATH", "/opt/cuda")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0")
    monkeypatch.setattr(stage_cli, "find_artifact", lambda *args: pytest.fail("Login-node I/O"))
    monkeypatch.setattr(sys, "argv", ["stage", stage, "launcher=h200", "experiment=dense_fulltext"])
    stage_cli.main()
    args = json.loads((submission / "qsub-args.json").read_text())
    assert args[:6] == ["-N", f"{stage}-dense-ACL", "-P", "gcb50357", "-q", "rt_HG"]
    assert "select=1:ncpus=16:mem=160gb:ngpus=1" in args
    assert "walltime=03:00:00" in args
    script = Path(args[-1])
    body = script.read_text()
    assert "export CUDA_PATH=/opt/cuda" in body
    assert "source scripts/h200_environment.sh" in body
    assert body.index("export CUDA_PATH=") < body.index("source scripts/h200_environment.sh")
    assert "export TORCH_CUDA_ARCH_LIST=9.0" in body
    plan = json.loads((script.parent / "plan.json").read_text())
    assert plan["command"][3:5] == ["scientific_tot.stage_cli", stage]
    log = Path(plan["log"])
    assert log.name == f"{stage}.log"
    assert "job submitted: 12345.server" in log.read_text()
    assert "-m scientific_tot.pbs --worker" in body
    assert "TOT_PBS_WORKER=1 UV_NO_SYNC=1" in body
    assert "must-not-be-written-to-job-script" not in body
    assert ".artifact-selection-placeholder" not in body
    assert json.loads((script.parent / "submission.json").read_text())["job_id"] == "12345.server"
    assert not (submission / "results/indexes").exists()
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_colqwen_uses_its_venv_without_resolving_python_symlink(submission, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["stage", "index", "launcher=h200", "experiment=colqwen_pages"]
    )
    stage_cli.main()
    args = json.loads((submission / "qsub-args.json").read_text())
    body = Path(args[-1]).read_text()
    assert str(ROOT / ".venv-colqwen/bin/python") in body


def test_qwen_batch_check_submits_without_reading_the_corpus(submission, monkeypatch):
    from scientific_tot import check_qwen_batch

    arguments = [
        "8912",
        "launcher=h200",
        "launcher.walltime=00:20:00",
        "experiment=qwen3_vl_multimodal",
        "split=test",
        "dataset.domains=[ACL]",
    ]
    monkeypatch.setattr(sys, "argv", ["check", *arguments])
    monkeypatch.setattr(
        check_qwen_batch, "check_batch", lambda *args: pytest.fail("Login-node encoding")
    )
    check_qwen_batch.main()
    args = json.loads((submission / "qsub-args.json").read_text())
    assert "walltime=00:20:00" in args
    plan = json.loads((Path(args[-1]).parent / "plan.json").read_text())
    assert plan["command"][3:] == ["scientific_tot.check_qwen_batch", *arguments]
    assert not (submission / "results/indexes").exists()


def test_benchmark_submits_one_worker_for_all_sequential_stages(submission, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "launcher=h200",
            "benchmark.experiments=[bm25_fulltext,dense_fulltext]",
            "benchmark.domains=[ACL,Physics]",
        ],
    )
    benchmark.main()
    args = json.loads((submission / "qsub-args.json").read_text())
    plan = json.loads((Path(args[-1]).parent / "plan.json").read_text())
    assert plan["command"][3] == "scientific_tot.benchmark"
    assert "benchmark.experiments=[bm25_fulltext,dense_fulltext]" in plan["command"]
    assert Path(plan["log"]).name == "benchmark.log"
    assert len(list((submission / "results/logs/pbs").glob("*/submission.json"))) == 1
    assert not (submission / "results/indexes").exists()


def test_test_chain_is_one_pbs_job(submission, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["stage", "test", "--test", "launcher=h200"])
    stage_cli.main()
    args = json.loads((submission / "qsub-args.json").read_text())
    plan = json.loads((Path(args[-1]).parent / "plan.json").read_text())
    assert plan["command"][3:6] == ["scientific_tot.stage_cli", "test", "--test"]
    assert Path(plan["log"]).name == "test.log"
    assert Path(args[-1]).is_relative_to(submission / "test/results/logs/pbs")


def test_worker_discovers_artifacts_and_does_not_submit_again(submission, monkeypatch):
    monkeypatch.setenv("TOT_PBS_WORKER", "1")
    monkeypatch.setenv("PBS_JOBID", "12345.server")
    index = submission / "completed-index"
    selected = []
    monkeypatch.setattr(stage_cli, "find_artifact", lambda *args: selected.append(args) or index)
    monkeypatch.setattr(
        stage_cli.subprocess,
        "run",
        lambda command: selected.append(command) or subprocess.CompletedProcess(command, 0),
    )
    monkeypatch.setattr(sys, "argv", ["stage", "retrieve", "launcher=h200"])
    with pytest.raises(SystemExit) as result:
        stage_cli.main()
    assert result.value.code == 0
    assert len(selected) == 2
    assert any("paths.index=" in part and str(index) in part for part in selected[1])
    assert not (submission / "qsub-args.json").exists()


def test_direct_hydra_entry_also_submits(submission):
    result = subprocess.run(
        [sys.executable, "-m", "scientific_tot.hydra_entry", "launcher=h200", "stage=retrieve"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Submitted PBS job 12345.server" in result.stdout
    assert not (submission / "results/models").exists()


def test_dry_run_prints_qsub_and_worker_without_writes(submission, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["stage", "index", "launcher=h200", "dry_run=true"])
    stage_cli.main()
    output = capsys.readouterr().out
    assert "$ qsub " in output and "-q rt_HG" in output
    assert "-m scientific_tot.stage_cli index" in output
    assert not (submission / "qsub-args.json").exists()
    assert not (submission / "results").exists()


def test_qsub_failure_is_fatal(submission, monkeypatch):
    executable = submission / "bin/qsub"
    executable.write_text("#!/bin/sh\necho 'invalid project' >&2\nexit 2\n")
    monkeypatch.setattr(sys, "argv", ["stage", "index", "launcher=h200"])
    with pytest.raises(RuntimeError, match=r"qsub failed \(2\): invalid project"):
        stage_cli.main()


def test_pbs_worker_runs_real_offline_bm25_chain(submission, monkeypatch):
    pt = pytest.importorskip("pyterrier")
    jars = Path(pt.io.pyterrier_home())
    if not (jars / "terrier-assemblies-5.11-jar-with-dependencies.jar").is_file() or not list(
        jars.glob("terrier-python-helper-*.jar")
    ):
        pytest.skip("Real offline worker test requires cached Terrier jars")
    monkeypatch.setenv("TOT_DATA_ROOT", str(ROOT / "tests/fixtures/adapter"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "launcher=h200",
            "launcher.setup_script=null",
            "benchmark.experiments=[bm25_fulltext]",
            "benchmark.domains=[ACL]",
            "launcher.threads.indexing=1",
            "launcher.threads.encoding=1",
            "launcher.threads.faiss=1",
        ],
    )
    benchmark.main()
    args = json.loads((submission / "qsub-args.json").read_text())
    script = Path(args[-1])
    original_submission = (submission / "qsub-args.json").read_text()
    result = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env=dict(os.environ, PBS_JOBID="12345.server", KMP_DUPLICATE_LIB_OK="TRUE"),
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Submitted PBS job" not in result.stdout
    assert (submission / "qsub-args.json").read_text() == original_submission
    job_log = (script.parent / "benchmark.log").read_text()
    assert "job submitted: 12345.server" in job_log
    assert "tot-benchmark worker started" in job_log
    assert "tot-benchmark worker completed (exit=0)" in job_log
    for directory in ("indexes", "models", "evaluations"):
        manifests = list((submission / "results" / directory).glob("*/manifest.json"))
        assert len(manifests) == 1
        assert json.loads(manifests[0].read_text())["status"] == "completed"
    for stage in ("index", "retrieve", "evaluate"):
        log = submission / "results/logs/pbs/bm25_fulltext/ACL" / f"{stage}.log"
        assert "worker started" in log.read_text()
        assert "worker completed (exit=0)" in log.read_text()


def test_standalone_worker_failure_is_in_stage_log(submission, monkeypatch):
    setup = submission / "custom setup.sh"
    setup.write_text("export CUDA_HOME=/custom/cuda\necho toolkit=$CUDA_HOME\n")
    monkeypatch.setattr(
        sys, "argv", ["stage", "index", "launcher=h200", f"launcher.setup_script={setup}"]
    )
    stage_cli.main()
    args = json.loads((submission / "qsub-args.json").read_text())
    script = Path(args[-1])
    result = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env=dict(os.environ, PBS_JOBID="12345.server"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "toolkit=/custom/cuda" in result.stdout
    log = (script.parent / "index.log").read_text()
    assert "worker started" in log
    assert "Traceback" in log and "Dataset root does not exist" in log
    assert "worker failed" in log


def test_custom_log_root_covers_submission_and_method_logs(submission, monkeypatch):
    root = submission / "custom-logs"
    monkeypatch.setenv("VTT_LOG_ROOT", str(root))
    overrides = [
        "launcher=h200",
        "benchmark.experiments=[bm25_fulltext]",
        "benchmark.domains=[ACL]",
    ]
    _, commands = benchmark.jobs(overrides)
    assert all(log.is_relative_to(root / "bm25_fulltext/ACL") for log, _ in commands)
    monkeypatch.setattr(sys, "argv", ["benchmark", *overrides])
    benchmark.main()
    args = json.loads((submission / "qsub-args.json").read_text())
    assert Path(args[-1]).is_relative_to(root)
    assert (Path(args[-1]).parent / "benchmark.log").is_file()


def test_worker_flag_requires_scheduler_allocation(monkeypatch):
    monkeypatch.setenv("TOT_PBS_WORKER", "1")
    monkeypatch.delenv("PBS_JOBID", raising=False)
    with pytest.raises(RuntimeError, match="PBS_JOBID"):
        is_worker()


@pytest.mark.parametrize(
    "override",
    [
        "launcher.cpus=0",
        "launcher.gpus=2",
        "launcher.memory=invalid",
        "launcher.walltime=bad",
        "launcher.queue='queue\\nextra'",
        "launcher.setup_script=false",
    ],
)
def test_invalid_scheduler_resources_fail_before_submission(override):
    with pytest.raises(ValueError, match="launcher|PBS"):
        stage_cli.plan("index", ["launcher=h200", override])
