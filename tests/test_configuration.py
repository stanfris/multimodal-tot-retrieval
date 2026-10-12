"""Offline composition, validation, and real Hydra CLI tests."""

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scientific_tot.configuration import CONFIG_DIR, resolve_config
from scientific_tot.hydra_run import execute_stage, run_stage

ROOT = Path(__file__).resolve().parents[1]


def config(*overrides, hydra=False):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="config", overrides=list(overrides), return_hydra_config=hydra)


def cli(tmp_path, *overrides):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), HYDRA_FULL_ERROR="1")
    return subprocess.run(
        [sys.executable, "-m", "scientific_tot.hydra_entry", *overrides],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )


def test_default_composition(tmp_path):
    cfg = resolve_config(config(), tmp_path)
    assert cfg.dataset.source == "local"
    assert cfg.dataset.download.enabled is False
    assert cfg.model.name == "bm25"
    assert cfg.representation.name == "fulltext"
    assert cfg.split.corpus == "all"
    assert cfg.split.queries == "test"
    assert cfg.stage.name == "prepare"
    assert cfg.dataset.root == str(tmp_path / "data")
    assert OmegaConf.is_readonly(cfg)


@pytest.mark.parametrize(
    "model,rep",
    list(
        itertools.product(
            ["bm25", "dense", "colbert", "visual"],
            ["title_abstract", "fulltext", "passages", "figures", "pages"],
        )
    ),
)
def test_model_representation_matrix(tmp_path, model, rep):
    cfg = config(f"model={model}", f"representation={rep}")
    compatible = (model == "visual") == (rep in {"figures", "pages"})
    if compatible:
        assert resolve_config(cfg, tmp_path).representation.name == rep
    else:
        with pytest.raises(ValueError, match="incompatible"):
            resolve_config(cfg, tmp_path)


@pytest.mark.parametrize(
    "profile,model,rep",
    [
        ("bm25_fulltext", "bm25", "fulltext"),
        ("dense_passages", "dense", "passages"),
        ("visual_pages", "visual", "pages"),
    ],
)
def test_experiment_presets(tmp_path, profile, model, rep):
    cfg = resolve_config(config(f"experiment={profile}"), tmp_path)
    assert (cfg.model.name, cfg.representation.name, cfg.stage.name) == (model, rep, "index")
    # Explicit CLI selections take precedence over preset group defaults.
    override = resolve_config(
        config(
            f"experiment={profile}", "model=bm25", "representation=title_abstract", "stage=export"
        ),
        tmp_path,
    )
    assert override.model.name == "bm25"
    assert override.stage.name == "export"


@pytest.mark.parametrize("source", ["default", "hub"])
@pytest.mark.parametrize("launcher", ["local_cpu", "local_gpu"])
@pytest.mark.parametrize("split", ["train", "test", "train+test", "all"])
def test_dataset_launcher_split_overrides(tmp_path, source, launcher, split):
    cfg = resolve_config(
        config(
            f"dataset={source}",
            f"launcher={launcher}",
            f"split={split}",
            "dataset.domains=[ACL,NeurIPS]",
        ),
        tmp_path,
    )
    assert cfg.dataset.source == ("hub" if source == "hub" else "local")
    assert cfg.launcher.device == ("cuda" if launcher == "local_gpu" else "cpu")
    assert cfg.split.queries == split
    assert cfg.split.corpus == "all"
    assert list(cfg.dataset.domains) == ["ACL", "NeurIPS"]


@pytest.mark.parametrize("stage", ["prepare", "index", "evaluate", "search", "export"])
def test_all_stages_explicitly_unimplemented(tmp_path, stage):
    overrides = [f"stage={stage}", "paths.index=existing/index"]
    if stage == "search":
        overrides.append("stage.query=scientific query")
    cfg = resolve_config(config(*overrides), tmp_path)
    with pytest.raises(NotImplementedError, match=f"stage={stage}"):
        execute_stage(cfg)


@pytest.mark.parametrize(
    "override,message",
    [
        ("dataset.repo_id=null", "repo_id"),
        ("dataset.root=null", "dataset.root"),
        ("dataset.domains=[]", "domains"),
        ("dataset.domains=[../ACL]", "domain"),
        ("dataset.domains=[ACL,ACL]", "distinct"),
        ("dataset.revision=main", "immutable"),
        ("dataset.download.enabled=true", "source=hub"),
        ("dataset.download.components=[clues]", "components"),
        ("split.queries=other", "Unsupported split"),
        ("stage.name=generate", "Unsupported stage"),
        ("model.backend=unknown", "backend"),
        ("representation.source=query_clues", "original document"),
        ("evaluation.top_k=0", "top_k"),
        ("evaluation.metrics=[]", "metrics"),
        ("dataset.artifacts.pdfs=''", "paths"),
        ("launcher.cache_root=data/cache", "overlaps"),
        ("launcher.output_root=data/output", "overlaps"),
        ("launcher.output_root=.", "overlaps"),
        ("dry_run=yes", "boolean"),
    ],
)
def test_invalid_settings(tmp_path, override, message):
    with pytest.raises(ValueError, match=message):
        resolve_config(config(override), tmp_path)


@pytest.mark.parametrize(
    "overrides,message",
    [
        (["model=dense", "model.model_id=null"], "model_id"),
        (["model=dense", "model.model_id=???"], "model_id"),
        (["model=dense", "model.batch_size=0"], "batch_size"),
        (["model=dense", "model.indexing.factory=Unknown"], "FAISS"),
        (["model=colbert", "model.checkpoint=null"], "checkpoint"),
        (["model=colbert", "model.indexing.nbits=3"], "nbits"),
        (["model=visual", "representation=pages", "model.model_id=null"], "model_id"),
        (["model=visual", "representation=pages", "model.indexing.overwrite=true"], "overwrite"),
        (["representation=passages", "representation.chunking.chunk_overlap=1000"], "overlap"),
        (["stage=evaluate"], "paths.index"),
        (["stage=search", "paths.index=/tmp/prior-index"], "stage.query"),
    ],
)
def test_missing_and_backend_parameters(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        resolve_config(config(*overrides), tmp_path)


def test_environment_paths_and_serialization(tmp_path, monkeypatch):
    monkeypatch.setenv("TOT_DATA_ROOT", str(tmp_path / "dgx-data"))
    monkeypatch.setenv("TOT_CACHE_ROOT", str(tmp_path / "cached"))
    monkeypatch.setenv("TOT_OUTPUT_ROOT", str(tmp_path / "runs"))
    cfg = resolve_config(config("launcher=local_gpu", "model=dense"), tmp_path)
    assert cfg.dataset.artifacts.preprocessed == str(tmp_path / "dgx-data/preprocessed")
    assert cfg.dataset.cache_dir == str(tmp_path / "cached/datasets")
    assert cfg.launcher.output_root == str(tmp_path / "runs")
    serialized = OmegaConf.to_yaml(cfg, resolve=True)
    assert "${" not in serialized
    assert OmegaConf.to_container(OmegaConf.create(serialized)) == OmegaConf.to_container(cfg)


def test_symlink_overlap_and_run_index_separation(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "alias").symlink_to(tmp_path / "data", target_is_directory=True)
    with pytest.raises(ValueError, match="overlaps"):
        resolve_config(config("launcher.output_root=alias/runs"), tmp_path)
    with pytest.raises(ValueError, match="separate"):
        resolve_config(
            config("stage=evaluate", "paths.index=runs/index"), tmp_path, tmp_path / "runs"
        )


def test_dry_run_does_not_dispatch_or_write_provenance(tmp_path):
    result = cli(tmp_path, "dry_run=true", "experiment=dense_passages")
    assert result.returncode == 0, result.stderr
    cfg = OmegaConf.create(result.stdout)
    assert cfg.model.name == "dense"
    assert cfg.dataset.root == str(tmp_path / "data")
    assert not list(tmp_path.rglob("manifest.json"))
    assert not any(p.is_file() for p in tmp_path.rglob("*"))
    assert not (tmp_path / "cache").exists()


def test_cli_invalid_parameters_and_show_config(tmp_path):
    bad = cli(tmp_path, "dry_run=true", "model=dense", "model.model_id=null")
    assert bad.returncode != 0
    assert "model.model_id" in bad.stderr
    shown = cli(tmp_path, "--cfg", "job", "--resolve", "experiment=visual_pages")
    assert shown.returncode == 0, shown.stderr
    assert OmegaConf.create(shown.stdout).model.name == "visual"


def test_real_multirun(tmp_path):
    result = cli(
        tmp_path,
        "--multirun",
        "dry_run=true",
        "experiment=bm25_fulltext,dense_passages,visual_pages",
        "split=train,test",
        f"hydra.sweep.dir={tmp_path / 'sweep'}",
    )
    assert result.returncode == 0, result.stderr
    for i in range(6):
        assert (tmp_path / "sweep" / str(i)).is_dir()
    assert result.stdout.count("dry_run: true") == 6
    assert (tmp_path / "sweep/multirun.yaml").is_file()


def test_failed_execution_provenance_and_no_overwrite(tmp_path):
    output = tmp_path / "attempt"
    result = cli(tmp_path, "stage=index", f"hydra.run.dir={output}", "dataset.revision=" + "a" * 40)
    assert result.returncode != 0
    assert "NotImplementedError" in result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["dataset_revision"] == "a" * 40
    assert manifest["local_revision_verified"] is False
    assert manifest["git_commit"]
    assert manifest["index_location"] == str(output / "index")
    assert (output / ".hydra/hydra.yaml").is_file()
    cfg = OmegaConf.load(output / ".hydra/config.yaml")
    assert cfg.dataset.root == str(tmp_path / "data")
    overrides = OmegaConf.load(output / ".hydra/overrides.yaml")
    assert "stage=index" in overrides
    before = {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}
    repeat = cli(tmp_path, "dry_run=true", f"hydra.run.dir={output}")
    assert repeat.returncode != 0
    assert "Refusing to overwrite" in repeat.stderr
    assert before == {p: p.read_bytes() for p in output.rglob("*") if p.is_file()}


def test_existing_sweep_refused_before_hydra_writes(tmp_path):
    folder = tmp_path / "sweep"
    folder.mkdir()
    sentinel = folder / "multirun.yaml"
    sentinel.write_text("preserve me")
    result = cli(
        tmp_path, "--multirun", "dry_run=true", "split=train,test", f"hydra.sweep.dir={folder}"
    )
    assert result.returncode != 0
    assert sentinel.read_text() == "preserve me"


def test_run_stage_guard_and_manifest_boundary(tmp_path, monkeypatch):
    cfg = resolve_config(config(), tmp_path)
    hydra_cfg = config(hydra=True).hydra
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls = []
    monkeypatch.setattr("scientific_tot.hydra_run.execute_stage", lambda value: calls.append(value))
    run_stage(cfg, run_dir, hydra_cfg, ROOT)
    assert len(calls) == 1
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "completed"
    with pytest.raises(FileExistsError):
        run_stage(cfg, run_dir, hydra_cfg, ROOT)
    assert len(calls) == 1


def test_cli_chdir_keeps_original_path_base(tmp_path):
    result = cli(tmp_path, "dry_run=true", "hydra.job.chdir=true")
    assert result.returncode == 0, result.stderr
    cfg = OmegaConf.create(result.stdout)
    assert cfg.dataset.root == str(tmp_path / "data")
    assert cfg.dataset.artifacts.splits == str(tmp_path / "data/splits/pdf_dataset_split.json")


def test_cli_rejects_output_inside_data_before_creating_it(tmp_path):
    output = tmp_path / "data/runs/attempt"
    result = cli(tmp_path, "dry_run=true", f"hydra.run.dir={output}")
    assert result.returncode != 0
    assert "overlaps" in result.stderr
    assert not output.exists()


def test_missing_group_reports_hydra_error(tmp_path):
    result = cli(tmp_path, "dry_run=true", "model=nonexistent")
    assert result.returncode != 0
    assert "model/nonexistent" in result.stderr


@pytest.mark.parametrize(
    "override",
    [
        "dataset.root=null",
        "dataset.artifacts=null",
        "dataset.artifacts.metadata=null",
    ],
)
@pytest.mark.parametrize("existing", [False, True])
def test_invalid_sweep_paths_cannot_bypass_output_guard(tmp_path, override, existing):
    output = tmp_path / "sweep"
    sentinel = output / "multirun.yaml"
    if existing:
        output.mkdir()
        sentinel.write_text("previous experiment")
    result = cli(
        tmp_path,
        "--multirun",
        "dry_run=true",
        "split=train,test",
        f"hydra.sweep.dir={output}",
        override,
    )
    assert result.returncode != 0
    if existing:
        assert sentinel.read_text() == "previous experiment"
        assert list(output.iterdir()) == [sentinel]
    else:
        assert not output.exists()
