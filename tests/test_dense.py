"""Offline fixtures exercise real FAISS/evaluation; neural encoding is stubbed."""

import json
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
    calls = []

    def stub(config, revision, model_cache, tasks):
        calls.append(("load", {"revision": revision}))
        encoded = {}
        for task, request in tasks.items():
            kwargs = {
                **request["kwargs"],
                "batch_size": config.batch_size,
                "normalize_embeddings": config.similarity == "cosine",
            }
            calls.append((task, kwargs))
            vectors = np.array(
                [[1, 0], [0, 1]] if task == "document" else [[1, 0]], dtype=np.float32
            )
            encoded[task] = (vectors, {"max_seq_length": 512, "dimension": 2})
        return encoded

    monkeypatch.setattr("scientific_tot.integrations.dense._encode_tasks", stub)
    return calls


def test_model_prompts_and_revision_are_explicit():
    config = EncodingConfig()
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
    assert metrics.iloc[0]["R@100"] == pytest.approx(1)
    assert [c[0] for c in encoder] == ["load", "document", "query"]
    assert encoder[0][1]["revision"] == DEFAULT_MODEL_REVISION
    assert encoder[1][1]["prompt"] == "passage: "
    assert encoder[2][1]["prompt"] == "query: "
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
