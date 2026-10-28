"""Encoding configuration and real embedding caches with stubbed neural inference."""

import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("faiss")

import numpy as np

from scientific_tot.integrations.dense import (
    DEFAULT_MODEL_REVISION,
    EncodingConfig,
    cached_encoding,
    encoding_kwargs,
    resolve_model_revision,
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
        vectors = np.eye(2, dtype=np.float32)[np.arange(len(texts)) % 2]
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


@pytest.mark.parametrize("field", ["encoding_threads", "faiss_threads", "batch_size"])
def test_rejects_invalid_thread_and_batch_settings(field):
    with pytest.raises(ValueError, match="must be positive"):
        EncodingConfig(**{field: 0})


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


def test_cache_reuse_and_corruption_detection(encoder, tmp_path):
    args = dict(
        task="document",
        rows=[{"docno": "one", "text": "first"}, {"docno": "two", "text": "second"}],
        dataset_revision=None,
    )
    config = EncodingConfig()
    first, _, entry, hit = cached_encoding(config, DEFAULT_MODEL_REVISION, tmp_path, **args)
    assert not hit
    encoder.clear()
    second, _, _, hit = cached_encoding(config, DEFAULT_MODEL_REVISION, tmp_path, **args)
    assert hit and encoder == []
    np.testing.assert_array_equal(first, second)
    (entry / "index.faiss").write_bytes(b"damaged index")
    with pytest.raises(ValueError, match="Corrupt embedding/index cache"):
        cached_encoding(config, DEFAULT_MODEL_REVISION, tmp_path, **args)


def test_query_prompt_change_reuses_document_cache(encoder, tmp_path):
    config = EncodingConfig()
    documents = dict(task="document", rows=[{"docno": "one", "text": "paper"}])
    queries = dict(task="query", rows=[{"query": "question"}])
    for args in (documents, queries):
        cached_encoding(config, DEFAULT_MODEL_REVISION, tmp_path, dataset_revision=None, **args)
    encoder.clear()
    changed = replace(config, query_prompt="Find: ")
    for args, expected_hit in ((documents, True), (queries, False)):
        _, _, _, hit = cached_encoding(
            changed, DEFAULT_MODEL_REVISION, tmp_path, dataset_revision=None, **args
        )
        assert hit == expected_hit
    assert [call[0] for call in encoder] == ["load", "query"]


@pytest.mark.dense_live
def test_pretrained_model_and_persistent_cache(tmp_path):
    args = dict(
        task="document",
        rows=[{"docno": "one", "text": "Scientific paper about language retrieval."}],
        dataset_revision=None,
    )
    config = EncodingConfig()
    first, manifest, entry, hit = cached_encoding(config, DEFAULT_MODEL_REVISION, tmp_path, **args)
    assert not hit
    assert manifest["encoding"]["cpu_threads"] == 1
    second, _, reused, hit = cached_encoding(config, DEFAULT_MODEL_REVISION, tmp_path, **args)
    assert hit and reused == entry
    np.testing.assert_array_equal(first, second)
