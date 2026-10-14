"""Offline stage integration against the existing restored synthetic dataset."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scientific_tot.config_artifacts import verify_index
from scientific_tot.configuration import CONFIG_DIR, resolve_config
from scientific_tot.hydra_run import run_stage
from scientific_tot.records import IncompleteCorpusError, UnresolvedRelevanceError

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/adapter"


def settings(tmp_path, stage, *overrides, dataset=FIXTURE):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        raw = compose(
            config_name="config",
            overrides=[f"dataset.root={dataset}", f"stage={stage}", *overrides],
            return_hydra_config=True,
        )
    task = OmegaConf.masked_copy(raw, [key for key in raw if key != "hydra"])
    output = tmp_path / stage
    return resolve_config(task, tmp_path, output), raw.hydra, output


def execute(cfg, hydra, output):
    output.mkdir()
    run_stage(cfg, output, hydra, ROOT)
    return json.loads((output / "manifest.json").read_text())


def test_prepare_exports_validated_original_records(tmp_path):
    cfg, hydra, output = settings(tmp_path, "prepare", "dataset.revision=" + "a" * 40)
    manifest = execute(cfg, hydra, output)
    assert manifest["status"] == "completed"
    assert manifest["artifacts"]["documents"] == 2
    corpus = [json.loads(line) for line in (output / "corpus.jsonl").read_text().splitlines()]
    assert {row["doc_id"] for row in corpus} == {"ACL_2022.example-1.1", "ACL_2022.example-1.2"}
    assert [
        json.loads(line)["doc_id"] for line in (output / "qrels.jsonl").read_text().splitlines()
    ] == ["ACL_2022.example-1.1"]
    sidecar = json.loads((output / "corpus.jsonl.manifest.json").read_text())
    assert sidecar["provenance"]["revision"] == "a" * 40
    assert sidecar["provenance"]["local_revision_verified"] is False
    assert sidecar["sha256"] == hashlib.sha256((output / "corpus.jsonl").read_bytes()).hexdigest()
    assert not (output / "index").exists()
    assert not (output / "run.trec").exists()


@pytest.mark.parametrize("stage", ["prepare"])
@pytest.mark.parametrize("missing", ["error", "skip"])
def test_every_positive_must_resolve_before_writes(tmp_path, stage, missing):
    dataset = tmp_path / "data"
    shutil.copytree(FIXTURE, dataset)
    query = next(dataset.glob("query_collections/**/queries.jsonl"))
    row = json.loads(query.read_text())
    row["relevant_ids"].append("ACL_absent-positive")
    query.write_text(json.dumps(row) + "\n")
    cfg, hydra, output = settings(tmp_path, stage, f"dataset.missing={missing}", dataset=dataset)
    with pytest.raises(UnresolvedRelevanceError):
        execute(cfg, hydra, output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
    assert not (output / "corpus.jsonl").exists()
    assert not (output / "index").exists()


@pytest.mark.parametrize("component", ["queries", "splits", "preprocessed"])
def test_missing_required_artifact_fails_explicitly(tmp_path, component):
    dataset = tmp_path / "data"
    shutil.copytree(FIXTURE, dataset)
    path = {"queries": "query_collections", "splits": "splits", "preprocessed": "preprocessed"}
    shutil.rmtree(dataset / path[component])
    cfg, hydra, output = settings(tmp_path, "prepare", "split.corpus=test", dataset=dataset)
    with pytest.raises(FileNotFoundError):
        execute(cfg, hydra, output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"


def test_selection_representation_and_missing_policy(tmp_path):
    dataset = tmp_path / "data"
    shutil.copytree(FIXTURE, dataset)
    for path in dataset.glob("preprocessed/**/paper.json"):
        row = json.loads(path.read_text())
        row.update(title="Original title", abstract="Original abstract")
        path.write_text(json.dumps(row))
    cfg, hydra, output = settings(
        tmp_path, "prepare", "split.corpus=test", "representation=title_abstract", dataset=dataset
    )
    execute(cfg, hydra, output)
    corpus = [json.loads(line) for line in (output / "corpus.jsonl").read_text().splitlines()]
    assert len(corpus) == 1
    assert corpus[0]["doc_id"] == "ACL_2022.example-1.1"
    assert corpus[0]["text"] == "Original title\n\nOriginal abstract"

    shutil.rmtree(output)
    (dataset / "preprocessed/ACL/ACL_2022.example-1.2/markdown.md").unlink()
    cfg, hydra, output = settings(tmp_path, "prepare", dataset=dataset)
    with pytest.raises(IncompleteCorpusError):
        execute(cfg, hydra, output)
    shutil.rmtree(output)
    cfg, hydra, output = settings(tmp_path, "prepare", "dataset.missing=skip", dataset=dataset)
    manifest = execute(cfg, hydra, output)
    assert manifest["artifacts"]["documents"] == 1
    assert manifest["artifacts"]["corpus_report"]["skipped"] == 1


@pytest.mark.parametrize("corrupt_docnos", [False, True])
def test_index_stage_passes_yaml_and_saves_completion_contract(
    tmp_path, monkeypatch, corrupt_docnos
):
    cfg, hydra, output = settings(
        tmp_path,
        "index",
        "model.indexing.blocks=true",
        "model.indexing.meta.docno=256",
        "launcher.threads.indexing=3",
        f"paths.index={tmp_path / 'external-index'}",
    )
    calls = []
    documents = []

    class Indexer:
        def __init__(self, path, **kwargs):
            calls.append(kwargs)
            self.path = Path(path)

        def index(self, rows):
            documents.extend(rows)
            (self.path / "data.properties").write_text("synthetic mocked index")
            return self.path

    index = SimpleNamespace(
        getCollectionStatistics=lambda: SimpleNamespace(
            getNumberOfDocuments=lambda: len(documents)
        ),
        getMetaIndex=lambda: SimpleNamespace(
            getItem=lambda key, i: "truncated" if corrupt_docnos else documents[i]["docno"]
        ),
        close=lambda: calls.append("closed"),
    )
    classic = object()
    monkeypatch.setitem(
        sys.modules,
        "pyterrier",
        SimpleNamespace(
            tqdm=lambda rows, **kwargs: rows,
            java=SimpleNamespace(started=lambda: True),
            terrier=SimpleNamespace(
                version=lambda: "5.11",
                IterDictIndexer=Indexer,
                IndexFactory=SimpleNamespace(of=lambda ref: index),
                IndexingType=SimpleNamespace(CLASSIC=classic),
            ),
        ),
    )
    if corrupt_docnos:
        with pytest.raises(AssertionError, match="docnos differ"):
            execute(cfg, hydra, output)
        assert calls[-1] == "closed"
        assert not (Path(cfg.paths.index) / "index-manifest.json").exists()
        assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
        return
    manifest = execute(cfg, hydra, output)
    assert manifest["status"] == "completed"
    assert calls == [
        {
            "type": classic,
            "blocks": True,
            "meta": {"docno": 256},
            "text_attrs": ["text"],
            "threads": 3,
        },
        "closed",
    ]
    saved = verify_index(Path(cfg.paths.index), cfg)
    assert saved["provenance"]["documents"] == 2
    assert saved["provenance"]["versions"]["pyterrier"]
    assert saved["provenance"]["output_dir"] == cfg.paths.index
    assert saved["provenance"]["dataset_revision"] is None
    assert saved["configuration"] == OmegaConf.to_container(cfg, resolve=True)
    assert not (output / "run.trec").exists()
    assert not (output / "metrics.csv").exists()


def test_docno_capacity_fails_before_build(tmp_path):
    cfg, hydra, output = settings(tmp_path, "index", "model.indexing.meta.docno=5")
    with pytest.raises(ValueError, match="truncate"):
        execute(cfg, hydra, output)
    assert not (output / "index").exists()


def test_offline_persistent_terrier_index(tmp_path):
    """Real JVM integration; cached jars are required, and downloads are prohibited."""
    pt = pytest.importorskip("pyterrier")
    pytest.importorskip("jdk4py")
    cache = Path(pt.io.pyterrier_home())
    if not (cache / "terrier-assemblies-5.11-jar-with-dependencies.jar").is_file() or not list(
        cache.glob("terrier-python-helper-*.jar")
    ):
        pytest.skip("Offline JVM test requires cached Terrier 5.11 and helper jars")
    script = """
import socket
from pathlib import Path
import pyterrier as pt
from hydra import compose, initialize_config_dir
from scientific_tot.configuration import CONFIG_DIR, resolve_config
from scientific_tot.hydra_run import run_stage

def blocked(*args, **kwargs):
    raise AssertionError("Offline integration must not access the network or retrieve")
socket.socket.connect = blocked
socket.create_connection = blocked
pt.java.mavenresolver.offline()
pt.terrier.Retriever = blocked
pt.Experiment = blocked
with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
    raw = compose(config_name="config", overrides=OVERRIDES, return_hydra_config=True)
from omegaconf import OmegaConf
cfg = OmegaConf.masked_copy(raw, [k for k in raw if k != "hydra"])
cfg = resolve_config(cfg, BASE, OUTPUT)
OUTPUT.mkdir()
run_stage(cfg, OUTPUT, raw.hydra, ROOT)
index = pt.terrier.IndexFactory.of(str(OUTPUT / "data.properties"))
try:
    assert index.getCollectionStatistics().getNumberOfDocuments() == 2
    assert {str(index.getMetaIndex().getItem("docno", i)) for i in range(2)} == {
        "ACL_2022.example-1.1", "ACL_2022.example-1.2"
    }
finally:
    index.close()
"""
    output = tmp_path / "index-run"
    prefix = (
        f"from pathlib import Path\nBASE=Path({str(tmp_path)!r})\n"
        f"OUTPUT=Path({str(output)!r})\nROOT=Path({str(ROOT)!r})\n"
        f"OVERRIDES={['stage=index', f'dataset.root={FIXTURE}', 'model.indexing.blocks=true']!r}\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", prefix + script],
        cwd=tmp_path,
        env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    saved = json.loads((output / "index-manifest.json").read_text())
    assert saved["status"] == "completed"
    assert saved["provenance"]["doc_ids_verified_against_index"] is True
    assert saved["configuration"]["model"]["indexing"]["blocks"] is True
    assert not (output / "run.trec").exists()
