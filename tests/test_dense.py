"""Offline fixtures exercise real FAISS/evaluation; neural encoding is stubbed."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("faiss")
pytest.importorskip("pyterrier")

import numpy as np

from scientific_tot import Dataset, DatasetAdapter, Records
from scientific_tot.integrations.dense import (
    DEFAULT_MODEL_REVISION,
    EncodingConfig,
    dense_experiment,
    encoding_kwargs,
    resolve_model_revision,
)


@pytest.fixture
def data():
    adapter = DatasetAdapter(Dataset(local_dir=Path(__file__).parent / "fixtures/adapter"))
    return (
        adapter.load_corpus(domain="ACL", split="train+test", representation="fulltext"),
        adapter.load_queries(domain="ACL", split="test"),
    )


@pytest.fixture
def encoder(monkeypatch):
    monkeypatch.setattr("scientific_tot.integrations.dense.require_cuda", lambda: None)
    calls = []

    def stub(config, revision, model_cache, task, texts, kwargs):
        calls.append(("load", {"revision": revision}))
        calls.append(
            (
                task,
                {
                    **kwargs,
                    "batch_size": config.batch_size,
                    "normalize_embeddings": config.similarity == "cosine",
                },
            )
        )
        vectors = np.array([[1, 0], [0, 1]] if task == "document" else [[1, 0]], dtype=np.float32)
        return vectors, {"max_seq_length": 512, "dimension": 2}

    monkeypatch.setattr("scientific_tot.integrations.dense._encode_task", stub)
    return calls


def test_dense_requires_cuda_device():
    with pytest.raises(ValueError, match="requires a CUDA device"):
        EncodingConfig(device="cpu")


@pytest.mark.parametrize("cached", [False, True])
def test_dense_fails_without_gpu_before_cache_or_encoding(monkeypatch, tmp_path, cached):
    from scientific_tot.integrations.dense import cached_encoding

    cache = tmp_path / "cache"
    if cached:
        cache.mkdir()
        (cache / "manifest.json").write_text("must not be read without CUDA")
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    )
    with pytest.raises(RuntimeError, match="available NVIDIA GPU"):
        cached_encoding(
            EncodingConfig(),
            DEFAULT_MODEL_REVISION,
            cache,
            task="document",
            rows=[{"docno": "paper", "text": "text"}],
            dataset_revision=None,
        )
    assert cache.exists() == cached


def test_model_prompts_and_revision_are_explicit():
    config = EncodingConfig()
    assert config.device == "cuda"
    assert resolve_model_revision(config) == DEFAULT_MODEL_REVISION
    assert encoding_kwargs(config, "query") == {"prompt": "query: "}
    assert encoding_kwargs(config, "document") == {"prompt": "passage: "}
    custom = replace(config, model_id="org/custom-model", profile="model")
    assert encoding_kwargs(custom, "query") == {}  # saved model prompt/Router task
    assert encoding_kwargs(replace(custom, query_prompt="Task: "), "query") == {"prompt": "Task: "}
    assert encoding_kwargs(replace(custom, document_prompt_name="corpus"), "document") == {
        "prompt_name": "corpus"
    }
    with pytest.raises(ValueError, match="task-specific"):
        encoding_kwargs(
            replace(config, model_id="intfloat/multilingual-e5-large-instruct", profile="e5"),
            "query",
        )
    with pytest.raises(ValueError, match="not both"):
        EncodingConfig(query_prompt="x", query_prompt_name="query")


def test_model_ref_is_resolved_to_commit(monkeypatch):
    from scientific_tot.integrations import dense

    calls = []

    class API:
        def model_info(self, repo_id, revision):
            calls.append((repo_id, revision))
            return SimpleNamespace(sha="a" * 40)

    monkeypatch.setattr(dense, "HfApi", API)
    assert (
        resolve_model_revision(EncodingConfig(model_id="org/custom", model_revision="v1"))
        == "a" * 40
    )
    assert calls == [("org/custom", "v1")]


def test_faiss_search_standard_metrics_and_cache_reuse(data, encoder, tmp_path):
    docs, queries = data
    metrics = dense_experiment(docs, queries, tmp_path / "first", cache=tmp_path / "cache")
    assert metrics.iloc[0]["RR@10"] == pytest.approx(1)
    assert set(metrics.columns) == {"name", "RR@10", "R@1", "R@10", "R@100", "nDCG@10"}
    for measure in ["R@1", "R@10", "nDCG@10"]:
        assert metrics.iloc[0][measure] == pytest.approx(1)
    assert [c[0] for c in encoder] == ["load", "document", "load", "query"]
    assert encoder[0][1]["revision"] == DEFAULT_MODEL_REVISION
    assert encoder[1][1]["prompt"] == "passage: "
    assert encoder[3][1]["prompt"] == "query: "
    assert encoder[1][1]["batch_size"] == 16
    assert encoder[1][1]["normalize_embeddings"] is True
    encoder.clear()
    dense_experiment(docs, queries, tmp_path / "second", cache=tmp_path / "cache")
    assert encoder == []
    manifest = json.loads((tmp_path / "second/manifest.json").read_text())
    assert manifest["cache"]["document_hit"] and manifest["cache"]["query_hit"]
    assert manifest["doc_ids_verified_against_index"] is True
    assert manifest["dataset_revision_status"] == "local/unrecorded"
    assert (tmp_path / "first/run.trec").read_bytes() == (tmp_path / "second/run.trec").read_bytes()
    assert json.loads((tmp_path / "second/selected-documents.json").read_text()) == [
        r["doc_id"] for r in docs
    ]


def test_changed_queries_reuse_document_index(data, encoder, tmp_path):
    docs, queries = data
    dense_experiment(docs, queries, tmp_path / "first", cache=tmp_path / "cache")
    encoder.clear()
    row = queries[0]
    row["text"] = "A changed query"
    changed = Records("queries", [row], queries.report, queries.provenance)
    dense_experiment(docs, changed, tmp_path / "second", cache=tmp_path / "cache")
    assert [c[0] for c in encoder] == ["load", "query"]
    manifest = json.loads((tmp_path / "second/manifest.json").read_text())
    assert manifest["cache"]["document_hit"] and not manifest["cache"]["query_hit"]


def test_cache_detects_corruption_before_faiss_read(data, encoder, tmp_path):
    docs, queries = data
    dense_experiment(docs, queries, tmp_path / "first", cache=tmp_path / "cache")
    index = next((tmp_path / "cache").glob("*/index.faiss"))
    index.write_bytes(b"damaged index")
    with pytest.raises(ValueError, match="Corrupt embedding/index cache"):
        dense_experiment(docs, queries, tmp_path / "second", cache=tmp_path / "cache")


def test_baseline_identity_checked_before_encoding(data, encoder, tmp_path):
    docs, queries = data
    dense_experiment(docs, queries, tmp_path / "first", cache=tmp_path / "cache")
    encoder.clear()
    dense_experiment(
        docs, queries, tmp_path / "second", cache=tmp_path / "cache", baseline=tmp_path / "first"
    )
    assert encoder == []
    rows = list(docs)
    rows[0]["text"] += " changed corpus"
    changed = Records("corpus", rows, docs.report, docs.provenance)
    with pytest.raises(ValueError, match="BM25 baseline"):
        dense_experiment(
            changed,
            queries,
            tmp_path / "third",
            cache=tmp_path / "cache",
            baseline=tmp_path / "first",
        )
    assert encoder == []


@pytest.mark.dense_live
def test_pretrained_model_and_persistent_cache(data, tmp_path):
    docs, queries = data
    first = dense_experiment(docs, queries, tmp_path / "first", cache=tmp_path / "cache")
    second = dense_experiment(docs, queries, tmp_path / "second", cache=tmp_path / "cache")
    assert first.equals(second)
    assert (tmp_path / "first/run.trec").read_bytes() == (tmp_path / "second/run.trec").read_bytes()
    manifest = json.loads((tmp_path / "second/manifest.json").read_text())
    assert manifest["encoding"]["documents"]["cpu_threads"] == 1
    assert manifest["model"]["resolved_revision"] == DEFAULT_MODEL_REVISION
    assert manifest["cache"]["document_hit"] and manifest["cache"]["query_hit"]


@pytest.mark.parametrize("field", ["encoding_threads", "faiss_threads", "batch_size"])
def test_rejects_invalid_thread_and_batch_settings(field):
    with pytest.raises(ValueError, match="must be positive"):
        EncodingConfig(**{field: 0})


def test_hydra_settings_propagate_into_existing_dense_calls(data, encoder, tmp_path):
    from hydra import compose, initialize_config_dir

    from scientific_tot.configuration import CONFIG_DIR, resolve_config

    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=dense",
                f"dataset.root={Path(__file__).parent / 'fixtures/adapter'}",
                "split.corpus=train+test",
                "model.batch_size=3",
                "model.encoding.normalize_embeddings=false",
                "model.encoding.query_prompt=Find: ",
                "model.encoding.document_prompt=''",
                "retrieval.top_k=1",
                "evaluation.metrics=[nDCG@1,RR@1]",
                f"launcher.cache_root={tmp_path / 'cache'}",
            ],
        )
    cfg = resolve_config(cfg, tmp_path)
    metrics = dense_experiment(
        *data, tmp_path / "run", cache=tmp_path / "cache/embeddings", settings=cfg
    )
    assert encoder[1][1] == {"prompt": "", "batch_size": 3, "normalize_embeddings": False}
    assert encoder[3][1]["prompt"] == "Find: "
    assert metrics.iloc[0]["nDCG@1"] == pytest.approx(1)
    assert "R@100" not in metrics.columns
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["index"]["cutoff"] == 1
    assert manifest["model"]["model_revision"] == cfg.model.revision
    assert len((tmp_path / "run/run.trec").read_text().splitlines()) == 1
    with pytest.raises(ValueError, match="cache=launcher"):
        dense_experiment(*data, tmp_path / "second", cache=tmp_path / "other", settings=cfg)


def test_sentence_transformer_external_arguments(monkeypatch, tmp_path):
    import sys

    from scientific_tot.integrations.dense import _encode_task

    calls = []

    class Model:
        max_seq_length = 512
        prompts = {}
        default_prompt_name = None

        def __init__(self, identifier, **kwargs):
            calls.append((identifier, kwargs))

        def encode_document(self, texts, **kwargs):
            calls.append((texts, kwargs))
            return np.array([[1, 0]], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda n: calls.append(("threads", n)),
            get_num_threads=lambda: 2,
            float32="float32",
            bfloat16="bfloat16",
            float16="float16",
        ),
    )
    cfg = EncodingConfig(batch_size=7, encoding_threads=2, similarity="dot", device="cuda")
    _encode_task(cfg, "a" * 40, tmp_path, "document", ["paper"], {"prompt": "Doc: "})
    assert calls[0] == ("threads", 2)
    assert calls[1] == (
        cfg.model_id,
        {
            "revision": "a" * 40,
            "device": "cuda",
            "cache_folder": str(tmp_path),
            "trust_remote_code": False,
            "model_kwargs": {"torch_dtype": "float32"},
        },
    )
    assert calls[2] == (
        ["paper"],
        {
            "batch_size": 7,
            "normalize_embeddings": False,
            "precision": "float32",
            "convert_to_numpy": True,
            "show_progress_bar": True,
            "prompt": "Doc: ",
        },
    )


def test_changed_query_prompt_reuses_document_cache(data, encoder, tmp_path):
    config = EncodingConfig()
    dense_experiment(*data, tmp_path / "first", cache=tmp_path / "cache", config=config)
    encoder.clear()
    dense_experiment(
        *data,
        tmp_path / "second",
        cache=tmp_path / "cache",
        config=replace(config, query_prompt="Find: "),
    )
    assert [call[0] for call in encoder] == ["load", "query"]
    manifest = json.loads((tmp_path / "second/manifest.json").read_text())
    assert manifest["cache"]["document_hit"]
    assert not manifest["cache"]["query_hit"]


def test_baseline_preserves_na_identifiers(tmp_path):
    import pandas as pd

    from scientific_tot.integrations.dense import verify_baseline
    from scientific_tot.records import records_digest

    inputs = SimpleNamespace(
        documents=({"docno": "NA", "text": "paper"},),
        topics=pd.DataFrame([{"qid": "NA", "query": "query"}]),
        qrels=pd.DataFrame([{"qid": "NA", "docno": "NA", "label": 1}]),
        query_ids=pd.DataFrame([{"qid": "NA", "query_id": "NA", "query_key": "key"}]),
    )
    (tmp_path / "manifest.json").write_text(
        json.dumps({"corpus_sha256": records_digest(inputs.documents)})
    )
    (tmp_path / "selected-documents.json").write_text(json.dumps(["NA"]))
    inputs.topics.to_json(tmp_path / "topics.jsonl", orient="records", lines=True)
    inputs.query_ids.to_json(tmp_path / "query-ids.jsonl", orient="records", lines=True)
    inputs.qrels.to_csv(tmp_path / "qrels.tsv", sep="\t", index=False)
    assert verify_baseline(inputs, tmp_path)["directory"] == str(tmp_path.resolve())
