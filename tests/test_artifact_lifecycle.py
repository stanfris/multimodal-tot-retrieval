"""Content identities and actual offline Hydra artifact lifecycle, including multiruns."""

import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scientific_tot.artifact_identity import artifact_directory, artifact_name, resolve_identity
from scientific_tot.configuration import CONFIG_DIR, resolve_config
from scientific_tot.hydra_run import run_stage
from scientific_tot.records import records_digest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/adapter"


def configuration(base, *overrides):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        raw = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=[f"dataset.root={base / 'data'}", *overrides],
        )
    return resolve_config(
        OmegaConf.masked_copy(raw, [k for k in raw if k != "hydra"]), base
    ), raw.hydra


@pytest.fixture
def dataset(tmp_path):
    shutil.copytree(FIXTURE, tmp_path / "data")
    for path in (tmp_path / "data").glob("preprocessed/**/paper.json"):
        row = json.loads(path.read_text())
        row.update(title="Original title", abstract="Original abstract")
        path.write_text(json.dumps(row))
    return tmp_path / "data"


def identity(base, *overrides):
    cfg, _ = configuration(base, "stage=index", *overrides)
    value = resolve_identity(cfg)
    return value, artifact_directory(cfg, value)


def test_index_name_has_no_checksum(tmp_path, dataset):
    original, path = identity(tmp_path)
    assert path.parent == tmp_path / "results/indexes"
    assert path.name == "bm25_fulltext_d-ACL_c-all"
    for change in (
        "split.queries=train",
        "retrieval.top_k=5",
        "evaluation.metrics=[RR@5]",
        "model.retrieval.controls={bm25.b:0.4,bm25.k_1:2.0}",
        "run.name=custom",
    ):
        assert identity(tmp_path, change) == (original, path)
    for change in (
        "representation=title_abstract",
        "split.corpus=test",
    ):
        assert identity(tmp_path, change)[1] != path
    paper = next(dataset.glob("preprocessed/**/markdown.md"))
    paper.write_text(paper.read_text() + " changed original corpus")
    assert identity(tmp_path)[1] == path
    assert identity(tmp_path)[0] != original


@pytest.mark.parametrize("model", ["bm25", "dense", "colbert"])
@pytest.mark.parametrize("domain", ["ACL", "Biology", "Engineering", "Medicine", "Physics"])
def test_index_name_includes_domain(tmp_path, model, domain):
    cfg, _ = configuration(tmp_path, "stage=index", f"model={model}", f"dataset.domains=[{domain}]")
    assert artifact_name(cfg, {}) == f"{model}_fulltext_d-{domain}_c-all"


def test_dense_identity_pins_encoder_and_excludes_query_settings(tmp_path, dataset):
    original = identity(tmp_path, "model=dense")
    assert (
        identity(tmp_path, "model=dense", "model.encoding.query_prompt='New query: '") == original
    )
    for change in (
        "model.revision=" + "b" * 40,
        "model.encoding.document_prompt='New document: '",
        "model.encoding.normalize_embeddings=false",
    ):
        assert identity(tmp_path, "model=dense", change)[0] != original[0]
        assert identity(tmp_path, "model=dense", change)[1] == original[1]
    passages = identity(tmp_path, "model=dense", "representation=passages")
    assert (
        identity(
            tmp_path,
            "model=dense",
            "representation=passages",
            "representation.aggregation.method=mean",
        )
        == passages
    )
    assert (
        identity(
            tmp_path,
            "model=dense",
            "representation=passages",
            "representation.chunking.chunk_size=400",
        )
        != passages
    )


def test_index_identity_and_stage_do_not_read_queries(tmp_path, dataset, monkeypatch):
    from scientific_tot import stages
    from scientific_tot.config_artifacts import write_index_manifest
    from scientific_tot.records import records_digest

    shutil.rmtree(dataset / "query_collections")

    def blocked(*args, **kwargs):
        raise AssertionError("Index must not load queries/qrels")

    monkeypatch.setattr("scientific_tot.adapter.DatasetAdapter.load_queries", blocked)
    monkeypatch.setattr("scientific_tot.adapter.validate_qrels", blocked)
    monkeypatch.setattr(stages, "validate_qrels", blocked)
    calls = []

    def build(corpus, inputs, cfg, *, reserved):
        assert reserved
        calls.append(inputs.documents)
        (Path(cfg.paths.index) / "data.properties").write_text("offline index")
        provenance = {
            "dataset_revision": None,
            "model_revision": None,
            "corpus_sha256": records_digest(inputs.documents),
        }
        write_index_manifest(Path(cfg.paths.index), cfg, provenance=provenance)
        return provenance

    monkeypatch.setattr("scientific_tot.integrations.pyterrier.build_bm25_index", build)
    cfg, hydra = configuration(tmp_path, "stage=index")
    target = artifact_directory(cfg, resolve_identity(cfg))
    target.mkdir(parents=True)
    run_stage(cfg, target, hydra, ROOT)
    before = {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}
    changed, hydra = configuration(tmp_path, "stage=index", "split.queries=train")
    run_stage(changed, target, hydra, ROOT)
    assert len(calls) == 1
    assert before == {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}
    (target / "data.properties").write_text("corrupted")
    shutil.rmtree(dataset)
    monkeypatch.setattr("scientific_tot.hydra_run.artifact_reference", blocked)
    run_stage(cfg, target, hydra, ROOT)
    assert len(calls) == 1


def offline_cli(base, *overrides):
    script = """
import socket, sys
import pyterrier as pt
from scientific_tot.hydra_entry import main
pt.java.mavenresolver.offline()
def blocked(*args, **kwargs):
    raise AssertionError("Offline lifecycle must not access network")
socket.socket.connect = blocked
socket.create_connection = blocked
main()
"""
    return subprocess.run(
        [sys.executable, "-c", script, *overrides],
        cwd=base,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), HYDRA_FULL_ERROR="1"),
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.fixture
def cached_terrier():
    pt = pytest.importorskip("pyterrier")
    pytest.importorskip("jdk4py")
    cache = Path(pt.io.pyterrier_home())
    if not (cache / "terrier-assemblies-5.11-jar-with-dependencies.jar").is_file() or not list(
        cache.glob("terrier-python-helper-*.jar")
    ):
        pytest.skip("Offline lifecycle requires cached Terrier jars")


def check(result):
    assert result.returncode == 0, result.stdout + result.stderr


def test_actual_hydra_lifecycle_reuse_and_independent_evaluation(tmp_path, dataset, cached_terrier):
    common = ["retrieval.top_k=2", "evaluation.metrics=[RR@2]"]
    check(offline_cli(tmp_path, "stage=index", *common))
    index = next((tmp_path / "results/indexes").iterdir())
    assert (index / "data.properties").is_file()
    assert (index / ".hydra/config.yaml").is_file()
    assert not (index / "index").exists()
    snapshot = {p: p.read_bytes() for p in index.rglob("*") if p.is_file()}
    repeated = offline_cli(tmp_path, "stage=index", "split.queries=train", *common)
    check(repeated)
    assert "Skipping existing index" in repeated.stdout
    assert snapshot == {p: p.read_bytes() for p in index.rglob("*") if p.is_file()}
    check(offline_cli(tmp_path, "stage=retrieve", f"paths.index={index}", *common))
    run = next((tmp_path / "results/models").iterdir())
    before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    repeated = offline_cli(
        tmp_path,
        "stage=retrieve",
        f"paths.index={index}",
        "evaluation.metrics=[R@2]",
        "retrieval.top_k=2",
    )
    check(repeated)
    assert "Reusing verified" in repeated.stdout
    assert before == {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
    check(
        offline_cli(
            tmp_path,
            "stage=retrieve",
            f"paths.index={index}",
            "retrieval.top_k=1",
            "evaluation.metrics=[RR@1]",
        )
    )
    assert len(list((tmp_path / "results/models").iterdir())) == 2
    assert snapshot == {p: p.read_bytes() for p in index.rglob("*") if p.is_file()}
    shutil.rmtree(dataset)
    shutil.rmtree(index)
    for metric in ("RR@2", "R@2"):
        command = ["stage=evaluate", f"paths.run={run}", f"evaluation.metrics=[{metric}]"]
        check(offline_cli(tmp_path, *command))
        repeated = offline_cli(tmp_path, *command)
        check(repeated)
        assert "Reusing verified" in repeated.stdout
    evaluations = list((tmp_path / "results/evaluations").iterdir())
    assert len(evaluations) == 2
    assert all(
        (p / "metrics.csv").is_file() and (p / "per-query.csv").is_file() for p in evaluations
    )
    (run / "run.trec").chmod(0o644)
    (run / "run.trec").write_text("corrupt")
    invalid = offline_cli(tmp_path, "stage=evaluate", f"paths.run={run}", *common)
    assert invalid.returncode != 0
    assert "checksum differs" in invalid.stderr


def test_actual_multirun_skips_shared_index(tmp_path, dataset, cached_terrier):
    command = [
        "--multirun",
        "stage=index",
        "split.queries=train,test",
        "model.indexing.blocks=false,true",
    ]
    check(offline_cli(tmp_path, *command))
    indexes = list((tmp_path / "results/indexes").iterdir())
    assert len(indexes) == 1
    before = {p: p.read_bytes() for p in (tmp_path / "results").rglob("*") if p.is_file()}
    check(offline_cli(tmp_path, *command))
    assert before == {p: p.read_bytes() for p in (tmp_path / "results").rglob("*") if p.is_file()}
    conflict = offline_cli(
        tmp_path,
        "--multirun",
        "stage=index",
        "paths.index=shared",
        "model.indexing.blocks=false,true",
    )
    check(conflict)
    assert (tmp_path / "shared/data.properties").is_file()


def test_concurrent_cli_reservations_do_not_overwrite(tmp_path, dataset, cached_terrier):
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: offline_cli(tmp_path, "stage=index"), range(2)))
    assert any(result.returncode == 0 for result in results), [r.stderr for r in results]
    indexes = list((tmp_path / "results/indexes").iterdir())
    assert len(indexes) == 1
    manifest = json.loads((indexes[0] / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    check(offline_cli(tmp_path, "stage=index"))


def test_existing_index_skips_without_writes(tmp_path, dataset):
    cfg, hydra = configuration(tmp_path, "stage=index")
    target = artifact_directory(cfg, resolve_identity(cfg))
    target.mkdir(parents=True)
    (target / ".run.lock").touch()
    run_stage(cfg, target, hydra, ROOT)
    assert list(target.iterdir()) == [target / ".run.lock"]


def test_mutable_encoder_request_is_pinned_for_build(tmp_path, dataset, monkeypatch):
    from scientific_tot.config_artifacts import write_index_manifest

    revision = "b" * 40
    monkeypatch.setattr(
        "scientific_tot.integrations.dense.resolve_model_revision",
        lambda config: config.model_revision if config.model_revision != "main" else revision,
    )

    def build(corpus, inputs, cfg, *, reserved):
        assert reserved and cfg.model.revision == revision
        target = Path(cfg.paths.index)
        (target / "index.faiss").write_bytes(b"offline vectors")
        provenance = {
            "dataset_revision": None,
            "model_revision": revision,
            "corpus_sha256": records_digest(inputs.documents),
        }
        write_index_manifest(target, cfg, provenance=provenance)
        return provenance

    monkeypatch.setattr("scientific_tot.integrations.dense_batch.build_dense_index", build)
    cfg, hydra = configuration(tmp_path, "stage=index", "model=dense", "model.revision=main")
    target = artifact_directory(cfg, resolve_identity(cfg))
    target.mkdir(parents=True)
    run_stage(cfg, target, hydra, ROOT)
    assert OmegaConf.load(target / ".hydra/config.yaml").model.revision == "main"
    manifest = json.loads((target / "index-manifest.json").read_text())
    assert manifest["artifact_identity"]["index"]["model"]["revision"] == revision
    assert json.loads((target / "manifest.json").read_text())["resolved_model_revision"] == revision
    run_stage(cfg, target, hydra, ROOT)


def test_sources_changing_during_build_cannot_publish_reusable_index(
    tmp_path, dataset, monkeypatch
):
    from scientific_tot.config_artifacts import verify_index, write_index_manifest

    def build(corpus, inputs, cfg, *, reserved):
        target = Path(cfg.paths.index)
        (target / "data.properties").write_text("offline index")
        provenance = {
            "dataset_revision": None,
            "model_revision": None,
            "corpus_sha256": records_digest(inputs.documents),
        }
        write_index_manifest(target, cfg, provenance=provenance)
        next(dataset.glob("preprocessed/**/markdown.md")).write_text("changed during build")
        return provenance

    monkeypatch.setattr("scientific_tot.integrations.pyterrier.build_bm25_index", build)
    cfg, hydra = configuration(tmp_path, "stage=index")
    target = artifact_directory(cfg, resolve_identity(cfg))
    target.mkdir(parents=True)
    with pytest.raises(ValueError, match="Source inputs changed"):
        run_stage(cfg, target, hydra, ROOT)
    assert json.loads((target / "manifest.json").read_text())["status"] == "failed"
    consumer, _ = configuration(tmp_path, "stage=retrieve", f"paths.index={target}")
    with pytest.raises(ValueError, match="stage manifest is incomplete"):
        verify_index(target, consumer)


def test_reserved_builder_refuses_existing_backend_files(tmp_path, dataset):
    from types import SimpleNamespace

    from scientific_tot.config_arguments import configured_adapter, corpus_arguments
    from scientific_tot.integrations.pyterrier import build_bm25_index

    cfg, _ = configuration(tmp_path, "stage=index")
    target = artifact_directory(cfg, resolve_identity(cfg))
    cfg = resolve_config(cfg, tmp_path, target)
    target.mkdir(parents=True)
    (target / "data.properties").write_text("existing index")
    corpus = configured_adapter(cfg).load_corpus(**corpus_arguments(cfg))
    inputs = SimpleNamespace(
        documents=tuple({"docno": r["doc_id"], "text": r["text"]} for r in corpus)
    )
    with pytest.raises(FileExistsError, match="nonempty index"):
        build_bm25_index(corpus, inputs, cfg, reserved=True)
    assert (target / "data.properties").read_text() == "existing index"


@pytest.mark.parametrize("model", ["bm25", "dense"])
@pytest.mark.parametrize("empty", [False, True])
def test_cli_skips_existing_index_without_dataset(tmp_path, model, empty):
    target = tmp_path / "results/indexes" / f"{model}_fulltext_d-ACL_c-all"
    target.mkdir(parents=True)
    if not empty:
        (target / "index.bin").write_bytes(b"existing index")
    before = {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}
    result = subprocess.run(
        [sys.executable, "-m", "scientific_tot.hydra_entry", "stage=index", f"model={model}"],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), HYDRA_FULL_ERROR="1"),
        capture_output=True,
        text=True,
        timeout=30,
    )
    check(result)
    assert "Skipping existing index" in result.stdout
    assert before == {p: p.read_bytes() for p in target.rglob("*") if p.is_file()}
