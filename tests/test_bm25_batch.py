"""Offline batch contracts, independent evaluation, and the complete BM25 pipeline."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scientific_tot.config_arguments import configured_adapter, corpus_arguments
from scientific_tot.config_artifacts import artifact_reference, write_index_manifest
from scientific_tot.configuration import CONFIG_DIR, resolve_config
from scientific_tot.hydra_run import run_stage

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/adapter"
METRICS = "evaluation.metrics=[RR@1,R@1,nDCG@1]"
CONTROLS = "model.retrieval.controls={bm25.b:0.5,bm25.k_1:2.0}"


def settings(base, output, dataset, stage, *overrides):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        raw = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=[
                f"stage={stage}",
                f"dataset.root={dataset}",
                METRICS,
                CONTROLS,
                *overrides,
            ],
        )
    task = OmegaConf.masked_copy(raw, [key for key in raw if key != "hydra"])
    return resolve_config(task, base, output), raw.hydra


def execute(cfg, hydra, output):
    output.mkdir()
    run_stage(cfg, output, hydra, ROOT)
    return json.loads((output / "manifest.json").read_text())


def blocked(*args, **kwargs):
    raise AssertionError("This operation must not index, infer, evaluate, or load corpus content")


@pytest.fixture
def batch_setup(tmp_path, monkeypatch):
    """Mock only the Terrier index/retriever; serialization and metrics use real libraries."""
    pt = pytest.importorskip("pyterrier")
    dataset = tmp_path / "data"
    shutil.copytree(FIXTURE, dataset)
    query_file = next(dataset.glob("query_collections/**/queries.jsonl"))
    original = json.loads(query_file.read_text())
    rows = [
        {**original, "query_id": "001", "query": "synthetic"},
        {**original, "query_id": "NA", "query": "unfindablexyz"},
    ]
    query_file.write_text("".join(json.dumps(row) + "\n" for row in rows))
    index_dir = tmp_path / "index"
    cfg, _ = settings(tmp_path, tmp_path / "build", dataset, "index", f"paths.index={index_dir}")
    corpus = configured_adapter(cfg).load_corpus(**corpus_arguments(cfg))
    index_dir.mkdir()
    (index_dir / "data.properties").write_text("mock persistent Terrier properties")
    write_index_manifest(
        index_dir,
        cfg,
        provenance={
            "dataset_revision": None,
            "model_revision": None,
            "corpus_sha256": "a" * 64,
            "documents": len(corpus),
            "document_domains": {r["doc_id"]: r["domain"] for r in corpus},
            "corpus_provenance": corpus.provenance,
            "terrier_version": "5.11",
        },
    )
    ids = [row["doc_id"] for row in corpus]
    calls = []
    index = SimpleNamespace(
        getCollectionStatistics=lambda: SimpleNamespace(getNumberOfDocuments=lambda: len(ids)),
        getMetaIndex=lambda: SimpleNamespace(getItem=lambda key, i: ids[i]),
        close=lambda: calls.append("close"),
    )

    class Retriever:
        def __init__(self, selected, **kwargs):
            assert selected is index
            calls.append(kwargs)

        def transform(self, topics):
            assert topics.qid.tolist() == ["001", "NA"]
            calls.append("retrieve")
            return pd.DataFrame([{"qid": "001", "docno": ids[0], "rank": 0, "score": 2.0}])

    experiment = pt.Experiment
    monkeypatch.setattr(pt.java, "started", lambda: True)
    monkeypatch.setattr(pt.java, "init", blocked)
    monkeypatch.setattr(pt.terrier, "version", lambda: "5.11")
    monkeypatch.setattr(pt.terrier, "IndexFactory", SimpleNamespace(of=lambda path: index))
    monkeypatch.setattr(pt.terrier, "Retriever", Retriever)
    monkeypatch.setattr(pt.terrier, "IterDictIndexer", blocked)
    monkeypatch.setattr(pt, "Experiment", blocked)
    monkeypatch.setattr("scientific_tot.adapter.DatasetAdapter.load_corpus", blocked)
    return SimpleNamespace(
        dataset=dataset,
        index_dir=index_dir,
        calls=calls,
        pt=pt,
        experiment=experiment,
        query_file=query_file,
        index=index,
    )


@pytest.fixture
def saved_batch(tmp_path, monkeypatch, batch_setup):
    output = tmp_path / "retrieve"
    cfg, hydra = settings(
        tmp_path,
        output,
        batch_setup.dataset,
        "retrieve",
        f"paths.index={batch_setup.index_dir}",
        "retrieval.top_k=1",
    )
    manifest = execute(cfg, hydra, output)
    monkeypatch.setattr(batch_setup.pt, "Experiment", batch_setup.experiment)
    batch_setup.output = output
    batch_setup.manifest = manifest
    return batch_setup


def test_retrieve_forwards_yaml_and_records_exact_artifacts(saved_batch):
    batch = saved_batch
    assert batch.calls == [
        {
            "wmodel": "BM25",
            "controls": {"bm25.b": 0.5, "bm25.k_1": 2.0},
            "num_results": 1,
            "threads": 1,
            "verbose": True,
        },
        "retrieve",
        "close",
    ]
    tokens = (batch.output / "run.trec").read_text().split()
    assert tokens[:4] == ["001", "Q0", "ACL_2022.example-1.1", "1"]
    assert not (batch.output / "metrics.csv").exists()
    refs = batch.manifest["artifacts"]["inputs"]
    assert refs["index_manifest"] == artifact_reference(batch.index_dir / "index-manifest.json")
    assert any(ref["path"] == str(batch.query_file) for ref in refs["query_files"])
    for name, reference in batch.manifest["artifacts"]["outputs"].items():
        assert reference == artifact_reference(batch.output / name)


def test_repeat_evaluation_is_independent_of_dataset_index_and_inference(
    tmp_path, monkeypatch, saved_batch
):
    batch = saved_batch
    shutil.rmtree(batch.dataset)
    shutil.rmtree(batch.index_dir)
    monkeypatch.setattr(batch.pt.terrier, "Retriever", blocked)
    monkeypatch.setattr(batch.pt.terrier, "IndexFactory", SimpleNamespace(of=blocked))
    monkeypatch.setattr("scientific_tot.adapter.DatasetAdapter.load_queries", blocked)
    monkeypatch.setattr("scientific_tot.data.hub.Dataset.discover", blocked)
    before = {path.name: path.read_bytes() for path in batch.output.iterdir() if path.is_file()}
    for name in ("evaluate-first", "evaluate-again"):
        target = tmp_path / name
        cfg, hydra = settings(
            tmp_path, target, batch.dataset, "evaluate", f"paths.run={batch.output}"
        )
        manifest = execute(cfg, hydra, target)
        metrics = pd.read_csv(target / "metrics.csv")
        assert metrics["RR@1"].iloc[0] == pytest.approx(0.5)
        assert metrics["R@1"].iloc[0] == pytest.approx(0.5)
        assert metrics["nDCG@1"].iloc[0] == pytest.approx(0.5)
        perquery = pd.read_csv(target / "per-query.csv", dtype=str, keep_default_na=False)
        assert set(perquery.qid) == {"001", "NA"}
        assert len(perquery) == 6
        assert set(perquery.loc[perquery.qid == "NA", "value"]) == {"0.0"}
        assert manifest["artifacts"]["inputs"]["batch"] == batch.manifest["artifacts"]["outputs"]
        assert OmegaConf.to_container(OmegaConf.load(target / ".hydra/config.yaml")) == (
            OmegaConf.to_container(cfg)
        )
    assert (tmp_path / "evaluate-first/metrics.csv").read_bytes() == (
        tmp_path / "evaluate-again/metrics.csv"
    ).read_bytes()
    assert before == {
        path.name: path.read_bytes() for path in batch.output.iterdir() if path.is_file()
    }
    assert batch.calls.count("retrieve") == 1


@pytest.mark.parametrize(
    "change",
    [
        "representation=title_abstract",
        "split.corpus=test",
        "split.queries=train",
        "dataset.query_mode=visual-only",
        "model.retrieval.controls={bm25.b:0.8,bm25.k_1:2.0}",
        "evaluation.metrics=[R@2]",
    ],
)
def test_evaluation_rejects_incompatible_configuration(tmp_path, saved_batch, change):
    target = tmp_path / "evaluate"
    cfg, hydra = settings(
        tmp_path, target, saved_batch.dataset, "evaluate", f"paths.run={saved_batch.output}", change
    )
    with pytest.raises(ValueError, match="differs|cutoff"):
        execute(cfg, hydra, target)
    assert not (target / "metrics.csv").exists()


@pytest.mark.parametrize(
    "name", ["run.trec", "topics.jsonl", "qrels.tsv", "query-ids.jsonl", "corpus-ids.jsonl"]
)
@pytest.mark.parametrize("damage", ["missing", "modified"])
def test_evaluation_rejects_missing_or_changed_artifacts(tmp_path, saved_batch, name, damage):
    path = saved_batch.output / name
    path.chmod(0o644)
    if damage == "missing":
        path.unlink()
    else:
        path.write_text(path.read_text() + "\n")
    target = tmp_path / "evaluate"
    cfg, hydra = settings(
        tmp_path, target, saved_batch.dataset, "evaluate", f"paths.run={saved_batch.output}"
    )
    with pytest.raises(ValueError, match="Incomplete|checksum"):
        execute(cfg, hydra, target)
    assert not (target / "metrics.csv").exists()


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate",
        "unknown_doc",
        "unknown_query",
        "multi_run",
        "nonfinite",
        "fractional_rank",
        "missing_positive",
    ],
)
def test_evaluation_rejects_ambiguous_semantic_inputs(tmp_path, monkeypatch, saved_batch, damage):
    batch = saved_batch
    path = batch.output / ("qrels.tsv" if damage == "missing_positive" else "run.trec")
    content = path.read_text()
    if damage == "duplicate":
        content += content
    elif damage == "unknown_doc":
        content = content.replace("ACL_2022.example-1.1", "unknown")
    elif damage == "unknown_query":
        content = content.replace("001", "unknown")
    elif damage == "multi_run":
        content = content.rsplit(" ", 1)[0] + " other-experiment\n"
    elif damage == "nonfinite":
        content = content.replace("2.0", "inf")
    elif damage == "fractional_rank":
        content = content.replace(" 1 ", " 1.5 ")
    else:
        content = "\n".join(content.splitlines()[:2]) + "\n"
    path.chmod(0o644)
    path.write_text(content)
    # Re-sign the malformed file to exercise semantic checks beyond byte integrity.
    manifest_path = batch.output / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["outputs"][path.name] = artifact_reference(path)
    manifest_path.chmod(0o644)
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(batch.pt, "Experiment", blocked)
    target = tmp_path / "evaluate"
    cfg, hydra = settings(tmp_path, target, batch.dataset, "evaluate", f"paths.run={batch.output}")
    with pytest.raises(ValueError):
        execute(cfg, hydra, target)


@pytest.mark.parametrize(
    "change", ["representation=title_abstract", "split.corpus=test", "model.indexing.blocks=true"]
)
def test_retrieve_rejects_incompatible_index(tmp_path, batch_setup, change):
    target = tmp_path / "retrieve"
    cfg, hydra = settings(
        tmp_path,
        target,
        batch_setup.dataset,
        "retrieve",
        f"paths.index={batch_setup.index_dir}",
        change,
    )
    with pytest.raises(ValueError, match="differs"):
        execute(cfg, hydra, target)
    assert batch_setup.calls == []


def test_retrieve_requires_persisted_index(tmp_path, batch_setup):
    (batch_setup.index_dir / "data.properties").unlink()
    target = tmp_path / "retrieve"
    cfg, hydra = settings(
        tmp_path, target, batch_setup.dataset, "retrieve", f"paths.index={batch_setup.index_dir}"
    )
    with pytest.raises(FileNotFoundError, match="Incomplete"):
        execute(cfg, hydra, target)
    assert batch_setup.calls == []


def test_index_requires_document_domains(tmp_path, batch_setup):
    path = batch_setup.index_dir / "index-manifest.json"
    manifest = json.loads(path.read_text())
    del manifest["provenance"]["document_domains"]
    path.chmod(0o644)
    path.write_text(json.dumps(manifest))
    target = tmp_path / "retrieve"
    cfg, hydra = settings(
        tmp_path, target, batch_setup.dataset, "retrieve", f"paths.index={batch_setup.index_dir}"
    )
    with pytest.raises(ValueError, match="identities/domains"):
        execute(cfg, hydra, target)


def test_older_multidomain_index_is_ambiguous(tmp_path, batch_setup):
    from scientific_tot.integrations.bm25_batch import _indexed_corpus

    cfg, _ = settings(
        tmp_path,
        tmp_path / "retrieve",
        batch_setup.dataset,
        "retrieve",
        f"paths.index={batch_setup.index_dir}",
        "dataset.domains=[ACL,NeurIPS]",
    )
    manifest = json.loads((batch_setup.index_dir / "index-manifest.json").read_text())
    del manifest["provenance"]["document_domains"]
    with pytest.raises(ValueError, match="identities/domains"):
        _indexed_corpus(batch_setup.index, manifest, cfg)


def test_empty_rankings_evaluate_all_queries_as_zero(tmp_path, monkeypatch, batch_setup):
    batch = batch_setup
    monkeypatch.setattr(
        batch.pt.terrier,
        "Retriever",
        lambda *args, **kwargs: SimpleNamespace(
            transform=lambda topics: pd.DataFrame(columns=["qid", "docno", "rank", "score"])
        ),
    )
    output = tmp_path / "retrieve"
    cfg, hydra = settings(
        tmp_path,
        output,
        batch.dataset,
        "retrieve",
        f"paths.index={batch.index_dir}",
        "retrieval.top_k=1",
    )
    execute(cfg, hydra, output)
    assert (output / "run.trec").stat().st_size == 0
    monkeypatch.setattr(batch.pt, "Experiment", batch.experiment)
    target = tmp_path / "evaluate"
    cfg, hydra = settings(tmp_path, target, batch.dataset, "evaluate", f"paths.run={output}")
    execute(cfg, hydra, target)
    metrics = pd.read_csv(target / "metrics.csv")
    assert metrics["RR@1"].iloc[0] == 0
    perquery = pd.read_csv(target / "per-query.csv", dtype=str, keep_default_na=False)
    assert len(perquery) == 6
    assert set(perquery.value) == {"0.0"}


def test_colliding_collection_query_ids_have_reversible_trec_mapping(
    tmp_path, monkeypatch, batch_setup
):
    from urllib.parse import unquote

    batch = batch_setup
    train = batch.dataset / "query_collections/ACL/query_generation_train/visual_and_text"
    train.mkdir(parents=True)
    (train / "queries.jsonl").write_text(
        batch.query_file.read_text().replace('"split": "test"', '"split": "train"')
    )
    monkeypatch.setattr(
        batch.pt.terrier,
        "Retriever",
        lambda *args, **kwargs: SimpleNamespace(
            transform=lambda topics: pd.DataFrame(
                [
                    {"qid": qid, "docno": "ACL_2022.example-1.1", "rank": 0, "score": 2.0}
                    for qid in topics.qid
                ]
            )
        ),
    )
    output = tmp_path / "retrieve"
    cfg, hydra = settings(
        tmp_path,
        output,
        batch.dataset,
        "retrieve",
        f"paths.index={batch.index_dir}",
        "split.queries=train+test",
    )
    execute(cfg, hydra, output)
    mapping = [json.loads(line) for line in (output / "query-ids.jsonl").read_text().splitlines()]
    assert len({row["qid"] for row in mapping}) == 4
    assert {row["query_id"] for row in mapping} == {"001", "NA"}
    assert all(unquote(row["qid"]) == row["query_key"] for row in mapping)
    assert {line.split()[0] for line in (output / "run.trec").read_text().splitlines()} == {
        row["qid"] for row in mapping
    }


def test_offline_prepare_index_retrieve_evaluate(tmp_path):
    """Real Terrier retrieval followed by repeated evaluation in a fresh, Java-free process."""
    pt = pytest.importorskip("pyterrier")
    pytest.importorskip("jdk4py")
    cache = Path(pt.io.pyterrier_home())
    if not (cache / "terrier-assemblies-5.11-jar-with-dependencies.jar").is_file() or not list(
        cache.glob("terrier-python-helper-*.jar")
    ):
        pytest.skip("Offline end-to-end test requires cached Terrier 5.11 and helper jars")
    dataset = tmp_path / "data"
    shutil.copytree(FIXTURE, dataset)
    paper = dataset / "preprocessed/ACL/ACL_2022.example-1.1/markdown.md"
    paper.write_text("# Original synthetic paper\nZebra zebra scientific retrieval\n")
    paper2 = dataset / "preprocessed/ACL/ACL_2022.example-1.2"
    (paper2 / "markdown.md").write_text("Whale ocean biology\n")
    paper3 = dataset / "preprocessed/ACL/ACL_2022.example-1.3"
    shutil.copytree(paper2, paper3)
    (paper3 / "paper.json").write_text((paper3 / "paper.json").read_text().replace("1.2", "1.3"))
    (paper3 / "markdown.md").write_text("Computer dialogue language\n")
    query_file = next(dataset.glob("query_collections/**/queries.jsonl"))
    row = json.loads(query_file.read_text())
    query_file.write_text(
        "".join(
            json.dumps({**row, "query_id": qid, "query": text}) + "\n"
            for qid, text in [("001", "zebra"), ("NA", "unfindablexyz")]
        )
    )
    setup = f"""
import json, socket, shutil
from pathlib import Path
from types import SimpleNamespace
import pyterrier as pt
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scientific_tot.configuration import CONFIG_DIR, resolve_config
from scientific_tot.hydra_run import run_stage
from scientific_tot.adapter import DatasetAdapter
BASE=Path({str(tmp_path)!r})
DATASET=Path({str(dataset)!r})
ROOT=Path({str(ROOT)!r})
def blocked(*args, **kwargs):
    raise AssertionError("Offline stage crossed its operation boundary")
socket.socket.connect = blocked
socket.create_connection = blocked
pt.java.mavenresolver.offline()
def execute(stage, name=None, extra=()):
    output = BASE / (name or stage)
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        raw = compose(config_name="config", return_hydra_config=True, overrides=[
            f"stage={{stage}}", f"dataset.root={{DATASET}}", {METRICS!r}, {CONTROLS!r}, *extra])
    cfg = resolve_config(OmegaConf.masked_copy(raw, [k for k in raw if k != "hydra"]), BASE, output)
    output.mkdir()
    run_stage(cfg, output, raw.hydra, ROOT)
"""
    build_and_retrieve = (
        setup
        + """
execute("prepare")
execute("index")
DatasetAdapter.load_corpus = blocked
pt.terrier.IterDictIndexer = blocked
pt.Experiment = blocked
shutil.rmtree(DATASET / "preprocessed")
execute("retrieve", extra=[f"paths.index={BASE / 'index'}", "retrieval.top_k=1"])
assert not (BASE / "retrieve/metrics.csv").exists()
"""
    )
    evaluation = (
        setup
        + """
pt.java.init = blocked
pt.terrier.IterDictIndexer = blocked
pt.terrier.Retriever = blocked
pt.terrier.IndexFactory = SimpleNamespace(of=blocked)
DatasetAdapter.load_queries = blocked
DatasetAdapter.load_corpus = blocked
execute("evaluate", name="evaluate-first", extra=[f"paths.run={BASE / 'retrieve'}"])
execute("evaluate", name="evaluate-again", extra=[f"paths.run={BASE / 'retrieve'}"])
assert not pt.java.started()
assert (BASE / "evaluate-first/metrics.csv").read_bytes() == (
    BASE / "evaluate-again/metrics.csv"
).read_bytes()
"""
    )
    for script in (build_and_retrieve, evaluation):
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=tmp_path,
            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        if script == build_and_retrieve:
            shutil.rmtree(dataset)
            shutil.rmtree(tmp_path / "index")
    metrics = pd.read_csv(tmp_path / "evaluate-first/metrics.csv")
    assert metrics["RR@1"].iloc[0] == pytest.approx(0.5)
    perquery = pd.read_csv(
        tmp_path / "evaluate-first/per-query.csv", dtype=str, keep_default_na=False
    )
    assert set(perquery.query_id) == {"001", "NA"}
    assert len(perquery) == 6
    rankings = (tmp_path / "retrieve/run.trec").read_text().split()
    assert rankings[:4] == ["001", "Q0", "ACL_2022.example-1.1", "1"]
