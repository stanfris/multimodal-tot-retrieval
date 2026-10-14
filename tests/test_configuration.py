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
from omegaconf.errors import MissingMandatoryValue

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
    assert list(cfg.evaluation.metrics) == ["RR@10", "R@1", "R@10", "nDCG@10"]
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
    if model == "colbert":
        compatible = rep == "fulltext"
    if compatible:
        assert resolve_config(cfg, tmp_path).representation.name == rep
    else:
        with pytest.raises(ValueError, match="incompatible|requires representation"):
            resolve_config(cfg, tmp_path)


@pytest.mark.parametrize(
    "profile,model,rep",
    [
        ("bm25_fulltext", "bm25", "fulltext"),
        ("dense_passages", "dense", "passages"),
        ("colbert_fulltext", "colbert", "fulltext"),
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
            "dataset.domains=[Physics]",
        ),
        tmp_path,
    )
    assert cfg.dataset.source == ("hub" if source == "hub" else "local")
    assert cfg.launcher.device == ("cuda" if launcher == "local_gpu" else "cpu")
    assert cfg.split.queries == split
    assert cfg.split.corpus == "all"
    assert list(cfg.dataset.domains) == ["Physics"]


@pytest.mark.parametrize("stage", ["search", "export"])
def test_remaining_stages_explicitly_unimplemented(tmp_path, stage):
    overrides = [f"stage={stage}"]
    if stage in {"index", "retrieve", "search"}:
        overrides.append("paths.index=existing/index")
    if stage == "evaluate":
        overrides.append("paths.run=existing/retrieval")
    if stage == "search":
        overrides.append("stage.query=scientific query")
    cfg = resolve_config(config(*overrides), tmp_path)
    with pytest.raises(NotImplementedError, match=f"stage={stage}"):
        execute_stage(cfg)


@pytest.mark.parametrize(
    "override,message",
    [
        ("dataset.repo_id=null", "repo_id"),
        ("dataset.root=null", "Filesystem paths"),
        ("dataset.domains=[]", "domains"),
        ("dataset.domains=[../ACL]", "domain"),
        ("dataset.domains=[ACL,ACL]", "exactly one"),
        ("dataset.domains=[ACL,Physics]", "exactly one"),
        ("dataset.revision=main", "immutable"),
        ("dataset.download.enabled=true", "source=hub"),
        ("dataset.download.components=[clues]", "components"),
        ("split.queries=other", "Unsupported split"),
        ("stage.name=generate", "Unsupported stage"),
        ("model.backend=unknown", "backend"),
        ("representation.source=query_clues", "original document"),
        ("retrieval.top_k=0", "top_k"),
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
        (["model=dense", "model.batch_size=0"], "batch_size"),
        (["model=dense", "model.indexing.factory=Unknown"], "FAISS"),
        (["model=colbert", "model.checkpoint=null"], "checkpoint"),
        (["model=colbert", "model.indexing.nbits=3"], "nbits"),
        (["model=visual", "representation=pages", "model.model_id=null"], "model_id"),
        (["model=visual", "representation=pages", "model.indexing.overwrite=true"], "overwrite"),
        (["representation=passages", "representation.chunking.chunk_overlap=1000"], "overlap"),
        (["stage=evaluate"], "paths.run"),
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
            config("stage=retrieve", "paths.index=runs/index"), tmp_path, tmp_path / "runs"
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
    assert len(list((tmp_path / "results/dry-runs").iterdir())) == 6
    assert result.stdout.count("dry_run: true") == 6
    assert (tmp_path / "sweep/multirun.yaml").is_file()


def test_failed_execution_provenance_and_no_overwrite(tmp_path):
    output = tmp_path / "attempt"
    result = cli(tmp_path, "stage=index", f"hydra.run.dir={output}", "dataset.revision=" + "a" * 40)
    assert result.returncode != 0
    assert "MissingComponentError" in result.stderr
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["dataset_revision"] == "a" * 40
    assert manifest["local_revision_verified"] is False
    assert manifest["git_commit"]
    assert manifest["index_location"] == str(output)
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
    cfg = resolve_config(config(f"dataset.root={ROOT / 'tests/fixtures/adapter'}"), tmp_path)
    hydra_cfg = config(hydra=True).hydra
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    calls = []
    monkeypatch.setattr(
        "scientific_tot.hydra_run.execute_stage", lambda value, **kwargs: calls.append(value)
    )
    run_stage(cfg, run_dir, hydra_cfg, ROOT)
    assert len(calls) == 1
    assert json.loads((run_dir / "manifest.json").read_text())["status"] == "completed"
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


@pytest.mark.parametrize(
    "overrides,message",
    [
        (["model=dense", "model.encoding.query_prompt=null"], "prompt"),
        (["model=dense", "model.indexing.metric=l2"], "inner_product"),
        (["model=bm25", "model.indexing.blocks=yes"], "boolean"),
        (["model=bm25", "model.indexing.meta.docno=0"], "docno"),
        (["model=bm25", "model.retrieval.controls={bm25.b:nan,bm25.k_1:1.2}"], "nonnegative"),
        (["model=colbert", "model.indexing.nbits=true"], "nbits"),
        (["representation=title_abstract", "representation.fields=[title]"], "fields"),
        (["representation=passages", "representation.chunking.separators=[]"], "separators"),
        (["model=visual", "representation=pages", "representation.rendering.scale=.inf"], "scale"),
        (["launcher.threads.encoding=true"], "positive"),
        (["launcher.cache_root=results/hydra/cache"], "cache overlap"),
        (["dataset.cache_dir=results/hydra"], "cache overlap"),
        (["run.name=../unsafe"], "run.name"),
        (["run.name='a/b'"], "run.name"),
        (["evaluation.metrics=[UNKNOWN@10]"], "Unsupported"),
        (["evaluation.metrics=[RR@0]"], "Unsupported"),
        (["evaluation.metrics=[RR@10,RR@10]"], "distinct"),
    ],
)
def test_new_invalid_settings(tmp_path, overrides, message):
    with pytest.raises(ValueError, match=message):
        resolve_config(config(*overrides), tmp_path)


def test_names_follow_effective_groups_and_splits(tmp_path):
    cfg = resolve_config(
        config("experiment=dense_passages", "representation=fulltext", "split.queries=train+test"),
        tmp_path,
    )
    assert cfg.run.name == "dense_fulltext_c-all_q-train+test"
    assert resolve_config(config("run.name=my-study"), tmp_path).run.name == "my-study"


@pytest.mark.parametrize(
    "override",
    ["launcher.cache_root=cache,results/hydra", "retrieval.top_k=100,0", "model=bm25,visual"],
)
def test_invalid_sweep_job_refused_before_metadata(tmp_path, override):
    result = cli(
        tmp_path, "--multirun", "dry_run=true", override, f"hydra.sweep.dir={tmp_path / 'sweep'}"
    )
    assert result.returncode != 0
    assert not (tmp_path / "sweep").exists()


@pytest.mark.parametrize("override", ["hydra.sweep.subdir=../escape", "hydra.sweep.subdir=same"])
def test_unsafe_sweep_layout(tmp_path, override):
    result = cli(
        tmp_path,
        "--multirun",
        "dry_run=true",
        "split=train,test",
        override,
        f"hydra.sweep.dir={tmp_path / 'sweep'}",
    )
    assert result.returncode != 0
    assert not (tmp_path / "sweep").exists()


def test_dry_sweep_shared_index_is_validated_without_writes(tmp_path):
    result = cli(
        tmp_path,
        "--multirun",
        "dry_run=true",
        "stage=index",
        "split=train,test",
        "paths.index=shared-index",
        f"hydra.sweep.dir={tmp_path / 'sweep'}",
    )
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "shared-index").exists()


def test_hydra_cannot_write_automatic_snapshots(tmp_path):
    result = cli(
        tmp_path, "dry_run=true", "hydra.output_subdir=.hydra", f"hydra.run.dir={tmp_path / 'run'}"
    )
    assert result.returncode != 0
    assert not (tmp_path / "run").exists()


def test_index_contract_and_integrity(tmp_path):
    from scientific_tot.config_artifacts import verify_index, write_index_manifest

    index = tmp_path / "index"
    index.mkdir()
    (index / "data.properties").write_text("offline index")
    built = resolve_config(config("stage=index"), tmp_path)
    provenance = {"dataset_revision": None, "model_revision": None, "corpus_sha256": "a" * 64}
    write_index_manifest(index, built, provenance=provenance)
    cfg = resolve_config(
        config(
            "stage=retrieve",
            "paths.index=index",
            "split=train",
            "retrieval.top_k=20",
            "model.retrieval.controls={bm25.b:0.5,bm25.k_1:1.2}",
        ),
        tmp_path,
    )
    assert verify_index(index, cfg)["provenance"] == provenance
    manifest_path = index / "index-manifest.json"
    current = json.loads(manifest_path.read_text())
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps({**current, "schema_version": 1}))
    with pytest.raises(ValueError, match="supported artifact"):
        verify_index(index, cfg)
    manifest_path.write_text(json.dumps(current))
    with pytest.raises(FileExistsError):
        write_index_manifest(index, built, provenance=provenance)
    for override in (
        "representation=title_abstract",
        "split.corpus=train",
        "model.indexing.blocks=true",
    ):
        bad = resolve_config(config("stage=retrieve", "paths.index=index", override), tmp_path)
        with pytest.raises(ValueError, match="differs"):
            verify_index(index, bad)
    path = index / "index-manifest.json"
    value = json.loads(path.read_text())
    value["configuration"]["model"]["indexing"]["blocks"] = True
    path.chmod(0o644)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checksum"):
        verify_index(index, cfg)


def test_consumer_preflight_prevents_dispatch(tmp_path, monkeypatch):
    cfg = resolve_config(config("stage=retrieve", "paths.index=legacy-index"), tmp_path)
    calls = []
    monkeypatch.setattr(
        "scientific_tot.hydra_run.execute_stage", lambda cfg, **kwargs: calls.append(cfg)
    )
    run = tmp_path / "attempt"
    run.mkdir()
    with pytest.raises(FileNotFoundError):
        run_stage(cfg, run, config(hydra=True).hydra, ROOT)
    assert not calls
    assert json.loads((run / "manifest.json").read_text())["status"] == "failed"
    assert (run / ".hydra/config.yaml").stat().st_mode & 0o222 == 0


def test_index_destination_and_cache_guard(tmp_path):
    cfg = resolve_config(config("stage=index"), tmp_path, tmp_path / "attempt")
    assert cfg.paths.index == str(tmp_path / "attempt")
    for override in ("paths.index=cache/index", "paths.index=data/index"):
        with pytest.raises(ValueError, match="overlap"):
            resolve_config(config("stage=index", override), tmp_path, tmp_path / "attempt")


def test_existing_index_skipped_before_dispatch(tmp_path, monkeypatch):
    (tmp_path / "index").mkdir()
    sentinel = tmp_path / "index/keep"
    sentinel.write_text("existing index")
    cfg = resolve_config(config("stage=index", "paths.index=index"), tmp_path)
    calls = []
    monkeypatch.setattr(
        "scientific_tot.hydra_run.execute_stage", lambda cfg, **kwargs: calls.append(cfg)
    )
    run = tmp_path / "attempt"
    run.mkdir()
    run_stage(cfg, run, config(hydra=True).hydra, ROOT)
    assert not list(run.iterdir())
    assert not calls
    assert sentinel.read_text() == "existing index"


def test_evaluation_consumes_only_completed_retrieval(tmp_path, monkeypatch):
    from scientific_tot.config_artifacts import (
        BATCH_FILES,
        artifact_reference,
        digest,
        verify_retrieval,
    )

    retrieval = tmp_path / "batch"
    (retrieval / ".hydra").mkdir(parents=True)
    snapshot = resolve_config(
        config("stage=retrieve", "paths.index=index", "retrieval.top_k=20"), tmp_path
    )
    OmegaConf.save(snapshot, retrieval / ".hydra/config.yaml")
    value = {
        "schema_version": 2,
        "artifact_identity": {"source": "test"},
        "artifact_sha256": digest({"source": "test"}),
        "status": "completed",
        "stage": "retrieve",
        "configuration_sha256": digest(OmegaConf.to_container(snapshot)),
    }
    for name in BATCH_FILES:
        (retrieval / name).touch()
    value["artifacts"] = {
        "outputs": {name: artifact_reference(retrieval / name) for name in BATCH_FILES}
    }
    (retrieval / "manifest.json").write_text(json.dumps(value))
    assert verify_retrieval(retrieval)["stage"] == "retrieve"
    (retrieval / "manifest.json").write_text(json.dumps({**value, "schema_version": 1}))
    with pytest.raises(ValueError, match="completed retrieve stage"):
        verify_retrieval(retrieval)
    (retrieval / "manifest.json").write_text(json.dumps(value))
    cfg = resolve_config(
        config("stage=evaluate", "paths.run=batch", "evaluation.metrics=[R@100]"), tmp_path
    )
    attempt = tmp_path / "evaluation"
    attempt.mkdir()
    calls = []
    monkeypatch.setattr(
        "scientific_tot.hydra_run.execute_stage", lambda cfg, **kwargs: calls.append(cfg)
    )
    with pytest.raises(ValueError, match="saved retrieval.top_k"):
        run_stage(cfg, attempt, config(hydra=True).hydra, ROOT)
    assert not calls
    (retrieval / "topics.jsonl").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        verify_retrieval(retrieval)


def test_configured_adapter_uses_existing_loader(tmp_path):
    from scientific_tot.config_arguments import (
        configured_adapter,
        corpus_arguments,
        query_arguments,
    )

    cfg = resolve_config(config(f"dataset.root={ROOT / 'tests/fixtures/adapter'}"), tmp_path)
    adapter = configured_adapter(cfg)
    corpus = adapter.load_corpus(**corpus_arguments(cfg))
    queries = adapter.load_queries(**query_arguments(cfg))
    assert len(corpus) == 2 and len(queries) == 1
    assert corpus.provenance["selection"]["split"] == "all"
    assert queries.provenance["selection"]["split"] == "test"
    moved = resolve_config(
        config("dataset.artifacts.metadata=elsewhere/metadata.parquet"), tmp_path
    )
    with pytest.raises(NotImplementedError, match="canonical"):
        configured_adapter(moved)


def test_argument_bindings(tmp_path):
    from scientific_tot.config_arguments import bm25_arguments, dense_arguments

    cfg = resolve_config(
        config(
            "model.indexing.blocks=true",
            "model.indexing.meta.docno=256",
            "model.retrieval.controls={bm25.b:0.75,bm25.k_1:2.5}",
            "retrieval.top_k=17",
            "launcher.threads.indexing=3",
        ),
        tmp_path,
    )
    indexing, retrieval = bm25_arguments(cfg, docno_bytes=140)
    assert indexing == {
        "type": "classic",
        "blocks": True,
        "meta": {"docno": 256},
        "text_attrs": ["text"],
        "threads": 3,
    }
    assert retrieval == {
        "wmodel": "BM25",
        "controls": {"bm25.b": 0.75, "bm25.k_1": 2.5},
        "num_results": 17,
        "threads": 3,
    }
    with pytest.raises(ValueError, match="truncate"):
        bm25_arguments(cfg, docno_bytes=257)
    dense = resolve_config(
        config(
            "model=dense",
            "model.batch_size=7",
            "launcher=local_gpu",
            "model.encoding.normalize_embeddings=false",
            "model.encoding.document_prompt=''",
            "launcher.threads.faiss=2",
        ),
        tmp_path,
    )
    args = dense_arguments(dense)
    assert args["similarity"] == "dot" and args["batch_size"] == 7
    assert args["device"] == "cuda" and args["document_prompt"] == ""
    assert args["faiss_threads"] == 2 and args["model_revision"] == dense.model.revision


@pytest.mark.parametrize(
    "profile", sorted(CONFIG_DIR.glob("*/*.yaml")), ids=lambda p: str(p.relative_to(CONFIG_DIR))
)
def test_every_yaml_profile_composes_and_validates(tmp_path, profile):
    group, name = profile.parent.name, profile.stem
    overrides = [f"{group}={name}"]
    if (group == "model" and name == "visual") or (
        group == "representation" and name in {"figures", "pages"}
    ):
        overrides += (
            ["model=visual", "representation=pages"] if group == "model" else ["model=visual"]
        )
    if group == "model" and name == "colqwen":
        overrides.append("representation=pages")
    if group == "stage" and name in {"retrieve", "search"}:
        overrides.append("paths.index=prior-index")
    if group == "stage" and name == "evaluate":
        overrides.append("paths.run=prior-retrieval")
    if group == "stage" and name == "search":
        overrides.append("stage.query=test")
    assert resolve_config(config(*overrides), tmp_path)


def test_index_cannot_contain_run_directory(tmp_path):
    with pytest.raises(ValueError, match="must not contain"):
        resolve_config(config("stage=index", "paths.index=runs"), tmp_path, tmp_path / "runs/job")


def test_dense_index_revision_and_query_prompt_contract(tmp_path):
    from scientific_tot.config_artifacts import verify_index, write_index_manifest

    index = tmp_path / "index"
    index.mkdir()
    (index / "index.faiss").write_text("offline index")
    cfg = resolve_config(config("model=dense", "stage=index"), tmp_path)
    provenance = {
        "dataset_revision": None,
        "model_revision": cfg.model.revision,
        "corpus_sha256": "a" * 64,
    }
    with pytest.raises(ValueError, match="differs"):
        write_index_manifest(index, cfg, provenance={**provenance, "model_revision": "b" * 40})
    assert not (index / "index-manifest.json").exists()
    write_index_manifest(index, cfg, provenance=provenance)
    query_change = resolve_config(
        config(
            "model=dense",
            "stage=retrieve",
            "paths.index=index",
            "model.encoding.query_prompt='Find: '",
        ),
        tmp_path,
    )
    assert verify_index(index, query_change)
    document_change = resolve_config(
        config(
            "model=dense",
            "stage=retrieve",
            "paths.index=index",
            "model.encoding.document_prompt='Doc: '",
        ),
        tmp_path,
    )
    with pytest.raises(ValueError, match="differs"):
        verify_index(index, document_change)


@pytest.mark.parametrize("revision", ["main", "null"])
def test_dense_index_consumers_need_immutable_model_revision(tmp_path, revision):
    with pytest.raises(ValueError, match="build commit"):
        resolve_config(
            config(
                "model=dense", "stage=retrieve", "paths.index=index", f"model.revision={revision}"
            ),
            tmp_path,
        )


def test_local_checkpoint_paths_share_original_base(tmp_path):
    cfg = resolve_config(config("model=colbert", "model.checkpoint=../checkpoints/model"), tmp_path)
    assert cfg.model.checkpoint == str((tmp_path / "../checkpoints/model").resolve())
    visual = resolve_config(
        config("model=visual", "representation=pages", "model.model_id=./checkpoints/visual"),
        tmp_path,
    )
    assert visual.model.model_id == str(tmp_path / "checkpoints/visual")


def test_index_contract_records_resolved_commits_not_mutable_refs(tmp_path):
    from scientific_tot.config_artifacts import verify_index, write_index_manifest

    index = tmp_path / "index"
    index.mkdir()
    (index / "index.faiss").write_text("offline index")
    cfg = resolve_config(
        config("dataset=hub", "model=dense", "stage=index", "model.revision=main"), tmp_path
    )
    provenance = {
        "dataset_revision": "a" * 40,
        "model_revision": "b" * 40,
        "corpus_sha256": "c" * 64,
    }
    write_index_manifest(index, cfg, provenance=provenance)
    consumer = resolve_config(
        config(
            "dataset=hub",
            "model=dense",
            "stage=retrieve",
            "paths.index=index",
            "dataset.revision=" + "a" * 40,
            "model.revision=" + "b" * 40,
        ),
        tmp_path,
    )
    manifest = verify_index(index, consumer)
    assert manifest["configuration"]["model"]["revision"] == "main"
    assert manifest["index_settings"]["model"]["revision"] == "b" * 40


@pytest.mark.parametrize(
    "overrides,key",
    [
        (["model=dense", "model.model_id=???"], "model.model_id"),
        (["stage=search", "paths.index=/tmp/prior-index"], "stage.query"),
    ],
)
def test_missing_value_uses_omegaconf_error(tmp_path, overrides, key):
    with pytest.raises(MissingMandatoryValue, match=key):
        resolve_config(config(*overrides), tmp_path)


@pytest.mark.parametrize("override", ["seed=42", "model.magic=1", "evaluation.top_k=100"])
def test_unknown_override_uses_hydra_error(tmp_path, override):
    result = cli(tmp_path, "dry_run=true", override)
    assert result.returncode != 0
    assert "Could not override" in result.stderr
    assert not any(tmp_path.iterdir())
