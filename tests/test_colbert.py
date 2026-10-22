"""Offline ColBERT stage contracts; inference/index/search are library stubs."""

import json
import shutil
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scientific_tot.artifact_identity import resolve_identity
from scientific_tot.config_artifacts import verify_index
from scientific_tot.configuration import CONFIG_DIR, resolve_config
from scientific_tot.hydra_run import run_stage
from scientific_tot.integrations import colbert

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/adapter"


def settings(base, stage, output, *overrides, dataset=None):
    dataset = dataset or FIXTURE
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        raw = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=[
                "experiment=colbert_fulltext",
                f"stage={stage}",
                f"dataset.root={dataset}",
                "retrieval.top_k=2",
                "evaluation.metrics=[RR@2,R@2]",
                *overrides,
            ],
        )
    cfg = resolve_config(OmegaConf.masked_copy(raw, [k for k in raw if k != "hydra"]), base, output)
    return cfg, raw.hydra


def execute(base, stage, name, *overrides, dataset=None):
    output = base / name
    cfg, hydra = settings(base, stage, output, *overrides, dataset=dataset)
    output.mkdir()
    run_stage(cfg, output, hydra, ROOT)
    return output, cfg, hydra


@pytest.fixture
def backend(monkeypatch, tmp_path):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls = []
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("pretrained ColBERT stub")
    monkeypatch.setattr(colbert, "checkpoint_path", lambda cfg: str(checkpoint))

    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Indexer:
        def __init__(self, **kwargs):
            self.config = kwargs["config"]
            calls.append(("indexer", kwargs))

        def index(self, *, name, collection, overwrite):
            assert name == "colbert" and overwrite is False
            calls.append(("index", collection))
            path = Path(self.config.index_path)
            path.mkdir()
            (path / "doclens.0.json").write_text(json.dumps([4] * len(collection)))
            (path / "metadata.json").write_text(json.dumps({"num_embeddings": 4 * len(collection)}))
            (path / "0.residuals.pt").write_bytes(b"compressed index stub")
            return str(path)

    class Searcher:
        def __init__(self, **kwargs):
            calls.append(("searcher", kwargs))

        def encode(self, texts):
            calls.append(("encode", texts))
            return texts

        def dense_search(self, vectors, k):
            assert len(vectors) == 1
            calls.append(("search", vectors[0], k))
            # Exercise the actual library's surplus-ranks behavior for k > corpus size.
            return [0, 1][:k], list(range(1, k + 1)), [2.0, 1.0][:k]

    monkeypatch.setitem(sys.modules, "colbert", SimpleNamespace(Indexer=Indexer, Searcher=Searcher))
    monkeypatch.setitem(
        sys.modules,
        "colbert.infra",
        SimpleNamespace(
            ColBERTConfig=Config,
            RunConfig=Config,
            Run=lambda: SimpleNamespace(context=lambda *a, **kw: nullcontext()),
        ),
    )
    return calls


def test_independent_lifecycle_and_parameter_propagation(tmp_path, backend, monkeypatch):
    dataset = tmp_path / "data"
    shutil.copytree(FIXTURE, dataset)
    train = dataset / "query_collections/ACL/query_generation_train/visual_and_text"
    shutil.copytree(dataset / "query_collections/ACL/query_generation_test/visual_and_text", train)
    query = json.loads((train / "queries.jsonl").read_text())
    query["query_id"] = "train_query"
    query["relevant_ids"] = ["ACL_2022.example-1.2"]
    query["metadata"].update(split="train", paper_id="ACL_2022.example-1.2")
    (train / "queries.jsonl").write_text(json.dumps(query) + "\n")
    (train / "metadata.json").write_text(
        json.dumps(
            {"example_count": 1, "mode": "visual-and-text", "config": {"split_name": "train"}}
        )
    )
    monkeypatch.setattr(sys.modules[__name__], "FIXTURE", dataset)
    execute(tmp_path, "prepare", "prepared")
    assert not backend
    index, cfg, hydra = execute(
        tmp_path, "index", "index", "model.indexing.index_bsize=7", "model.indexing.kmeans_niters=3"
    )
    arguments = backend[0][1]["config"].__dict__
    assert arguments == {
        "index_path": str(index / "colbert"),
        "index_bsize": 7,
        "nbits": 2,
        "kmeans_niters": 3,
        "doc_maxlen": 220,
    }
    assert len(backend[1][1]) == 2
    assert not (index / "run.trec").exists()
    mapping = json.loads((index / "ids.json").read_text())
    assert [row["doc_id"] for row in mapping] == ["ACL_2022.example-1.1", "ACL_2022.example-1.2"]
    manifest = verify_index(index, cfg)
    assert manifest["provenance"]["truncation"]["doc_maxlen"] == 220
    assert manifest["provenance"]["model_revision"] == cfg.model.revision
    identity = resolve_identity(cfg)
    changed, _ = settings(
        tmp_path,
        "index",
        index,
        "split.queries=train",
        "model.retrieval.query_maxlen=24",
        "retrieval.top_k=10",
        "model.indexing.index_bsize=7",
        "model.indexing.kmeans_niters=3",
    )
    assert resolve_identity(changed) == identity
    run_stage(changed, index, hydra, ROOT)
    assert len(backend) == 2
    with pytest.raises(FileExistsError, match="different identity"):
        wrong, _ = settings(tmp_path, "index", index, "model.indexing.nbits=4")
        run_stage(wrong, index, hydra, ROOT)

    def blocked(*args, **kwargs):
        raise AssertionError("Independent retrieval/evaluation must not load documents")

    monkeypatch.setattr("scientific_tot.adapter.DatasetAdapter.load_corpus", blocked)
    overrides = ["model.indexing.index_bsize=7", "model.indexing.kmeans_niters=3"]
    output, _, _ = execute(
        tmp_path,
        "retrieve",
        "ranked",
        f"paths.index={index}",
        *overrides,
        "model.retrieval.query_maxlen=24",
        "retrieval.top_k=10",
    )
    assert backend[2][1]["config"].query_maxlen == 24
    assert backend[3][0] == "encode"
    assert backend[4][2] == 10
    assert len((output / "run.trec").read_text().splitlines()) == 2
    assert (output / "run.trec").read_text().splitlines()[0].split()[2] == mapping[0]["doc_id"]
    # A different query selection and depth reuse the same index.
    execute(
        tmp_path, "retrieve", "train", f"paths.index={index}", *overrides, "split.queries=train"
    )
    monkeypatch.setattr("scientific_tot.adapter.DatasetAdapter.load_queries", blocked)
    monkeypatch.setattr(colbert, "checkpoint_path", blocked)
    shutil.rmtree(index)
    metrics, _, _ = execute(
        tmp_path,
        "evaluate",
        "metrics",
        f"paths.run={output}",
        *overrides,
        "model.retrieval.query_maxlen=24",
    )
    assert pd.read_csv(metrics / "metrics.csv")["RR@2"].iloc[0] == 1.0
    assert pd.read_csv(metrics / "per-query.csv").query_id.notna().all()
    (output / "run.trec").chmod(0o644)
    (output / "run.trec").write_text("corrupted")
    with pytest.raises(ValueError, match="checksum"):
        execute(
            tmp_path,
            "evaluate",
            "corrupt-run",
            f"paths.run={output}",
            *overrides,
            "model.retrieval.query_maxlen=24",
        )


@pytest.mark.parametrize(
    "override",
    [
        "representation=title_abstract",
        "representation=figures",
        "model.revision=main",
    ],
)
def test_unsupported_configuration(tmp_path, override):
    with pytest.raises(ValueError):
        settings(tmp_path, "index", tmp_path / "index", override)


def test_missing_and_incompatible_indexes_fail_before_search(tmp_path, backend):
    index, cfg, hydra = execute(tmp_path, "index", "index")
    with pytest.raises(ValueError, match="differs"):
        execute(
            tmp_path, "retrieve", "wrong", f"paths.index={index}", "model.indexing.doc_maxlen=128"
        )
    with pytest.raises(ValueError, match="differs"):
        execute(
            tmp_path,
            "retrieve",
            "wrong-model",
            f"paths.index={index}",
            "model.revision=" + "b" * 40,
        )
    path = index / "colbert/0.residuals.pt"
    path.write_bytes(b"corrupt")
    assert verify_index(index, cfg)  # Preflight deliberately does not checksum data.
    path.unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete persistent index"):
        execute(tmp_path, "retrieve", "missing", f"paths.index={index}")
    with pytest.raises(FileExistsError, match="invalid artifact"):
        run_stage(cfg, index, hydra, ROOT)
    assert len(backend) == 2


def test_incomplete_index_is_never_reused(tmp_path, backend):
    index = tmp_path / "index"
    index.mkdir()
    (index / "keep").write_text("incomplete")
    cfg, hydra = settings(tmp_path, "index", index)
    with pytest.raises(FileExistsError):
        run_stage(cfg, index, hydra, ROOT)
    assert not backend
    assert (index / "keep").read_text() == "incomplete"


@pytest.mark.colbert_live
def test_pretrained_colbertv2_tiny_corpus(tmp_path):
    """Real checkpoint, compression, native scoring, serialization and saved evaluation."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Real ColBERT indexing/retrieval requires an NVIDIA GPU")
    dataset = tmp_path / "data"
    shutil.copytree(FIXTURE, dataset)
    # Enough real tokens for the upstream automatic k-means partition heuristic.
    for number, path in enumerate(sorted(dataset.glob("preprocessed/**/markdown.md"))):
        path.write_text(
            ("neural language retrieval " if number == 0 else "marine ocean biology ") * 100
        )
    index, _, _ = execute(tmp_path, "index", "index", dataset=dataset)
    output, _, _ = execute(tmp_path, "retrieve", "ranked", f"paths.index={index}", dataset=dataset)
    assert (output / "run.trec").stat().st_size > 0
    assert set(pd.read_json(output / "corpus-ids.jsonl", lines=True).doc_id) == {
        "ACL_2022.example-1.1",
        "ACL_2022.example-1.2",
    }
    shutil.rmtree(dataset)
    shutil.rmtree(index)
    metrics, _, _ = execute(tmp_path, "evaluate", "metrics", f"paths.run={output}", dataset=dataset)
    assert (metrics / "metrics.csv").is_file()


def test_model_bytes_are_verified_before_search(tmp_path, backend):
    index, _, _ = execute(tmp_path, "index", "index")
    (tmp_path / "checkpoint/config.json").write_text("changed checkpoint")
    with pytest.raises(ValueError, match="checkpoint bytes differ"):
        execute(tmp_path, "retrieve", "changed-model", f"paths.index={index}")
    assert len(backend) == 2


def test_local_model_content_enters_identity(tmp_path, backend):
    checkpoint = tmp_path / "checkpoint"
    cfg, _ = settings(tmp_path, "index", tmp_path / "index", f"model.checkpoint={checkpoint}")
    before = resolve_identity(cfg)
    (checkpoint / "config.json").write_text("changed local checkpoint")
    assert resolve_identity(cfg) != before


def test_cuda_default_and_runtime_requirement(tmp_path, backend, monkeypatch):
    import torch

    cfg, _ = settings(tmp_path, "index", tmp_path / "index")
    assert cfg.launcher.device == "cuda"
    assert colbert.run_config(cfg).gpus == 1
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="NVIDIA GPU"):
        colbert.library_config(cfg, tmp_path / "index")
    cpu, _ = settings(tmp_path, "index", tmp_path / "index", "launcher=local_cpu")
    with pytest.raises(ValueError, match="require launcher.device=cuda"):
        colbert.library_config(cpu, tmp_path / "index")
