"""H200 configuration, offline staging, and sequential script orchestration."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scientific_tot import benchmark, h200_check, stage_dataset
from scientific_tot.data.hub import DEFAULT_REPO, Dataset
from scientific_tot.stage_cli import plan
from scientific_tot.validate_dataset import validate

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = sorted(
    p.stem for p in (ROOT / "config/experiment").glob("*.yaml") if p.stem != "visual_pages"
)


@pytest.mark.parametrize("launcher", ["local_cpu", "local_gpu", "h200"])
@pytest.mark.parametrize("test_mode", [False, True])
@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_launchers_all_experiments(tmp_path, launcher, test_mode, experiment):
    overrides = [f"experiment={experiment}", f"launcher={launcher}", "dry_run=true"]
    if launcher == "local_cpu" and experiment != "bm25_fulltext":
        if experiment in {"colqwen_pages"}:
            with pytest.raises(ValueError, match="requires launcher.device=cuda"):
                plan("index", overrides)
            return
        if experiment.startswith("qwen"):
            overrides.append("model.precision=fp32")
    if test_mode:
        overrides.append("--test")
    for stage in ("prepare", "index", "retrieve", "evaluate"):
        cfg, _ = plan(stage, overrides)
        assert cfg.launcher.device == ("cpu" if launcher == "local_cpu" else "cuda")
        assert cfg.split.corpus == ("test" if test_mode else "all")
        assert cfg.split.queries == "test"
        assert cfg.test_mode == test_mode


def test_h200_roots_threads_and_dense_fulltext(tmp_path, monkeypatch):
    monkeypatch.setenv("TOT_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("TOT_CACHE_ROOT", str(tmp_path / "cache"))
    monkeypatch.setenv("TOT_OUTPUT_ROOT", str(tmp_path / "results"))
    monkeypatch.setenv("TOT_ENCODING_THREADS", "3")
    monkeypatch.setenv("TOT_DATA_REVISION", "a" * 40)
    cfg, _ = plan(
        "index",
        [
            "launcher=h200",
            "model=dense",
            "representation=fulltext",
            "model.batch_size=1",
            "dry_run=true",
        ],
    )
    assert cfg.dataset.root == str(tmp_path / "data")
    assert cfg.launcher.cache_root == str(tmp_path / "cache")
    assert cfg.launcher.output_root == str(tmp_path / "results")
    assert cfg.launcher.threads.encoding == 3
    assert cfg.model.batch_size == 1
    assert cfg.model.precision == "fp32"
    assert cfg.dataset.repo_id == DEFAULT_REPO
    assert cfg.dataset.revision == "a" * 40
    assert DEFAULT_REPO.endswith("multimodal-tip-of-the-tongue-retrieval-for-scientific-documents")


@pytest.fixture
def staged(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "tests/fixtures/adapter", tmp_path / "data")
    monkeypatch.chdir(tmp_path)
    return tmp_path / "data"


def test_staging_does_not_read_document_content(staged, monkeypatch):
    original = Dataset.get_file

    def metadata_only(self, name):
        assert not name.endswith((".md", ".pdf", ".png"))
        return original(self, name)

    monkeypatch.setattr(Dataset, "get_file", metadata_only)
    cfg, _ = plan("prepare", [])
    assert validate(cfg)["papers"] == 2
    assert validate(cfg)["queries"] == 1
    next(staged.glob("preprocessed/*/*/markdown.md")).unlink()
    with pytest.raises(FileNotFoundError, match="missing or empty"):
        validate(cfg)


def test_staging_rejects_missing_split_papers_and_query_positives(staged):
    cfg, _ = plan("prepare", [])
    split_path = staged / "splits/pdf_dataset_split.json"
    split = json.loads(split_path.read_text())
    split["train"].append("ACL/missing.pdf")
    split_path.write_text(json.dumps(split))
    with pytest.raises(ValueError, match="lack extractions"):
        validate(cfg)
    split["train"].pop()
    split_path.write_text(json.dumps(split))
    queries = next(staged.glob("query_collections/**/queries.jsonl"))
    query = json.loads(queries.read_text())
    query["relevant_ids"].append("absent-paper")
    queries.write_text(json.dumps(query) + "\n")
    with pytest.raises(ValueError, match="positives absent"):
        validate(cfg)


def test_page_and_figure_staging_require_selected_binaries(staged):
    cfg, _ = plan("prepare", ["experiment=openclip_pages"])
    with pytest.raises(FileNotFoundError, match="pdf_datasets"):
        validate(cfg)
    paper = next(staged.glob("preprocessed/*/*/paper.json"))
    row = json.loads(paper.read_text())
    row["figures"] = [{"figure_id": "figure-1", "image_relpath": "images/missing.png"}]
    paper.write_text(json.dumps(row))
    cfg, _ = plan("prepare", ["experiment=openclip_figures"])
    with pytest.raises(ValueError, match="missing image"):
        validate(cfg)


def test_benchmark_selection_shared_overrides_and_isolation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dry, commands = benchmark.jobs(
        [
            "benchmark.experiments=[qwen3_vl_text]",
            "benchmark.domains=[ACL,Physics]",
            "benchmark.prepare=true",
            "model.batch_size=1",
            "mode=test",
            "dry_run=true",
        ]
    )
    assert dry and len(commands) == 8
    for log, command in commands:
        assert "test/results/logs/pbs/qwen3_vl_text" in str(log)
        assert "model.batch_size=1" in command and "launcher=h200" in command
        assert command[0].endswith(tuple(benchmark.SCRIPTS.values()))
        assert not any(arg.startswith("benchmark.") for arg in command)
    assert not (tmp_path / "test").exists()


@pytest.mark.parametrize(
    "override",
    [
        "benchmark.experiments=[visual_pages]",
        "benchmark.domains=[Invalid]",
        "experiment=bm25_fulltext",
    ],
)
def test_benchmark_invalid_selection(override):
    with pytest.raises(ValueError):
        benchmark.jobs([override])


def test_benchmark_stops_on_failure_and_appends_logs(tmp_path, monkeypatch):
    log = tmp_path / "logs/index.log"
    log.parent.mkdir()
    log.write_text("previous attempt\n")
    failing = tmp_path / "02_index.sh"
    failing.write_text("#!/usr/bin/env bash\necho failure\nexit 7\n")
    failing.chmod(0o755)
    commands = [
        (log, [str(failing)]),
        (tmp_path / "later.log", ["bash", "-c", "echo must-not-run"]),
    ]
    monkeypatch.setattr(benchmark, "jobs", lambda args: (False, commands))
    monkeypatch.setattr(
        benchmark,
        "plan",
        lambda *args: (SimpleNamespace(launcher=SimpleNamespace(kind="local")), []),
    )
    with pytest.raises(SystemExit) as error:
        benchmark.main()
    assert error.value.code == 7
    assert log.read_text().startswith("previous attempt\n")
    assert "failure" in log.read_text()
    assert not (tmp_path / "later.log").exists()


@pytest.mark.parametrize(
    "arguments,environment",
    [
        (["experiment=colqwen_pages"], ".venv-colqwen"),
        (["experiment=colqwen_pages", "model=dense"], "default"),
        (["experiment=dense_pages", "model=colqwen"], ".venv-colqwen"),
    ],
)
def test_stage_environment_and_offline(arguments, environment):
    env = dict(os.environ, HF_HUB_OFFLINE="1")
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    result = subprocess.run(
        [
            "bash",
            "-c",
            "source scripts/environment.sh; "
            'printf "%s %s %s %s" "${UV_PROJECT_ENVIRONMENT:-default}" '
            '"$UV_OFFLINE" "$UV_NO_SYNC" "$TRANSFORMERS_OFFLINE"',
            "stage",
            *arguments,
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout == f"{environment} 1 1 1"


def test_preflight_rejects_unavailable_cuda(monkeypatch):
    import torch

    monkeypatch.setattr(h200_check.platform, "system", lambda: "Linux")
    monkeypatch.setattr(h200_check.sys, "version_info", (3, 11))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        h200_check.tomllib,
        "loads",
        lambda text: {
            "project": {
                "optional-dependencies": {
                    name: [] for name in ("dense", "colbert", "visual", "pyterrier")
                }
            }
        },
    )
    with pytest.raises(RuntimeError, match="driver/runtime compatibility"):
        h200_check.check("default", SimpleNamespace())


def test_staging_pins_download_and_delegates_restoration(staged, monkeypatch):
    transfers, restored = [], []

    def download(**kwargs):
        transfers.append(kwargs)
        snapshot = Path(kwargs["local_dir"])
        for component in ("preprocessed", "query_collections", "splits"):
            (snapshot / component).mkdir(parents=True, exist_ok=True)
        (snapshot / "metadata.parquet").write_bytes(b"metadata")

    def restore(source, destination, *, overwrite):
        assert not overwrite
        restored.append(destination.name)

    def copy(source, destination, **kwargs):
        assert kwargs == {"overwrite": False, "size": 8}
        destination.write_bytes(source.read())

    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setattr(stage_dataset, "snapshot_download", download)
    monkeypatch.setitem(sys.modules, "dataset_packaging", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "dataset_packaging.download_hf_dataset",
        SimpleNamespace(
            BUILD_ARTIFACTS=["*.manifest.json"],
            allow_patterns=lambda components: [f"{c}/**" for c in sorted(components)],
            copy_stream=copy,
            restore_archive_component=restore,
            restore_plain_component=restore,
            safe_destination=lambda root, relative: root / relative,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "dataset_packaging.restore_hf_dataset",
        SimpleNamespace(
            restore_dataset=lambda **kwargs: pytest.fail("Text staging must not restore raw PDFs")
        ),
    )
    sha = "a" * 40
    cfg, _ = plan(
        "prepare",
        [
            "dataset=hub",
            f"dataset.revision={sha}",
            "dataset.download.components=[metadata,queries,splits,preprocessed]",
        ],
    )
    stage_dataset.stage(cfg)
    stage_dataset.stage(cfg)
    assert len(transfers) == 2
    assert all(t["revision"] == sha and t["repo_id"] == DEFAULT_REPO for t in transfers)
    assert all("data/*/shard-*.tar" not in t["allow_patterns"] for t in transfers)
    assert restored == ["preprocessed", "query_collections", "splits"] * 2
    assert (staged / "metadata.parquet").read_bytes() == b"metadata"


def test_staging_requires_pin_and_refuses_offline(staged, monkeypatch):
    cfg, _ = plan("prepare", ["dataset=hub"])
    with pytest.raises(ValueError, match="full Hub commit SHA"):
        stage_dataset.stage(cfg)
    cfg, _ = plan("prepare", ["dataset=hub", f"dataset.revision={'b' * 40}"])
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ValueError, match="offline mode"):
        stage_dataset.stage(cfg)
