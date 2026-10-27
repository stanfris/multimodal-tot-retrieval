"""Interrupted encoding reuses verified chunks and restores canonical vector order."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from scientific_tot.integrations.dense import EncodingConfig, cached_encoding


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("task,batch_size", [("document", 2), ("query", 3)])
def test_resume_encoding_and_model_options(
    tmp_path, monkeypatch, capsys, task, batch_size, corrupt
):
    monkeypatch.setattr("scientific_tot.integrations.dense.require_cuda", lambda: None)
    loaded, encoded = [], []
    state = {"interrupt": True}

    class Model:
        max_seq_length = 512
        prompts = {}
        default_prompt_name = None

        def __init__(self, model_id, **kwargs):
            loaded.append(kwargs)

        def encode_document(self, texts, **kwargs):
            assert kwargs["batch_size"] == batch_size
            assert kwargs["show_progress_bar"] is False
            if state["interrupt"] and encoded:
                state["interrupt"] = False
                raise RuntimeError("scheduler stopped job")
            encoded.extend(texts)
            return np.array([[len(text), 1] for text in texts], dtype=np.float32)

        encode_query = encode_document

    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=Model)
    )
    config = EncodingConfig(
        batch_size=2,
        query_batch_size=3,
        checkpoint_size=2,
        precision="bf16",
        max_seq_length=32768,
        attention="sdpa",
        padding_side="left",
        use_cache=False,
    )
    texts = ["a", "bbbb", "cc", "eeeee", "ddd"]
    rows = [
        {"docno": str(i), "text": text} if task == "document" else {"qid": str(i), "query": text}
        for i, text in enumerate(texts)
    ]
    kwargs = dict(task=task, rows=rows, dataset_revision=None)
    with pytest.raises(RuntimeError, match="scheduler stopped"):
        cached_encoding(config, "a" * 40, tmp_path, **kwargs)
    assert len(list(tmp_path.glob("*.partial/0/manifest.json"))) == 1
    if corrupt:
        next(tmp_path.glob("*.partial/0/embeddings.npy")).write_bytes(b"corrupt")
        with pytest.raises(ValueError, match="Corrupt embedding/index cache"):
            cached_encoding(config, "a" * 40, tmp_path, **kwargs)
        return
    vectors, manifest, entry, hit = cached_encoding(config, "a" * 40, tmp_path, **kwargs)
    assert not hit
    progress = capsys.readouterr().err
    assert f"Encoding {task} inputs" in progress
    assert "2/5" in progress and "5/5" in progress
    assert encoded == ["eeeee", "bbbb", "ddd", "cc", "a"]  # No repeated completed chunk.
    np.testing.assert_array_equal(vectors[:, 0], [len(text) for text in texts])
    assert manifest["encoding"]["batch_size"] == batch_size
    assert manifest["encoding"]["checkpoint_hits"] == 1
    assert loaded[0]["config_kwargs"]["use_cache"] is False
    assert loaded[0]["model_kwargs"]["attn_implementation"] == "sdpa"
    assert loaded[0]["processor_kwargs"] == {"padding_side": "left"}
    assert not list(tmp_path.glob("*.partial"))
    if task == "document":
        assert json.loads((entry / "ids.json").read_text()) == [str(i) for i in range(len(texts))]
    # Query batch tuning does not force document re-encoding.
    revised = replace(config, query_batch_size=7) if task == "document" else config
    cached_encoding(revised, "a" * 40, tmp_path, **kwargs)
    assert len(loaded) == 2  # A completed cache hit never reloads the model.
