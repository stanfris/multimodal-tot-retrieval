"""Fast offline dense stages: real FAISS, stubbed neural encoding."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scientific_tot.configuration import CONFIG_DIR, resolve_config
from scientific_tot.hydra_run import run_stage
from scientific_tot.integrations.dense import DEFAULT_MODEL_REVISION

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/adapter"


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    shutil.copytree(FIXTURE, root)
    (root / "preprocessed/ACL/ACL_2022.example-1.1/markdown.md").write_text(
        "zebra top\n\nnoise low"
    )
    (root / "preprocessed/ACL/ACL_2022.example-1.2/markdown.md").write_text("whale mid")
    query = next(root.glob("query_collections/**/queries.jsonl"))
    row = json.loads(query.read_text())
    row["query"] = "zebra"
    query.write_text(json.dumps(row) + "\n")
    return root


@pytest.fixture
def encoder(monkeypatch):
    monkeypatch.setattr("scientific_tot.integrations.dense.require_cuda", lambda: None)
    calls = []

    def stub(config, revision, cache, task, texts, kwargs, **options):
        assert revision == (
            "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
            if config.model_id == "Qwen/Qwen3-Embedding-0.6B"
            else DEFAULT_MODEL_REVISION
        )
        if options.get("checkpoint") is not None:
            options["checkpoint"].mkdir(parents=True, exist_ok=True)
        calls.append((config, revision, {task: {"texts": texts, "kwargs": kwargs}}))
        vectors = np.array(
            [
                [1.0, 0.0] if "zebra" in text else [0.75, 0.25] if "whale" in text else [0.0, 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )
        return vectors, {"dimension": 2, "explicit_prompt_arguments": kwargs}

    monkeypatch.setattr("scientific_tot.integrations.dense._encode_task", stub)
    return calls


def execute(base, dataset, stage, name, *overrides):
    output = base / name
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        raw = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=[
                "experiment=dense_fulltext",
                f"stage={stage}",
                f"dataset.root={dataset}",
                "model.encoding.normalize_embeddings=false",
                "model.batch_size=3",
                "model.encoding.query_prompt='Find: '",
                "model.encoding.document_prompt=''",
                "launcher.threads.encoding=2",
                "launcher.threads.faiss=2",
                "retrieval.top_k=2",
                "evaluation.metrics=[RR@2,R@2]",
                *overrides,
            ],
        )
    cfg = resolve_config(OmegaConf.masked_copy(raw, [k for k in raw if k != "hydra"]), base, output)
    output.mkdir(exist_ok=True)
    run_stage(cfg, output, raw.hydra, ROOT)
    return output, json.loads((output / "manifest.json").read_text()), cfg


def test_fulltext_document_and_query_encoding_are_independent_and_cached(
    tmp_path, dataset, encoder, monkeypatch
):
    execute(tmp_path, dataset, "prepare", "prepared")
    assert encoder == []
    index, built, _ = execute(tmp_path, dataset, "index", "built", "launcher=local_gpu")
    assert [list(call[2]) for call in encoder] == [["document"]]
    config, revision, tasks = encoder[0]
    assert (config.device, config.batch_size, config.faiss_threads, config.encoding_threads) == (
        "cuda",
        3,
        2,
        2,
    )
    assert config.similarity == "dot" and config.model_revision == revision
    assert tasks["document"]["kwargs"] == {"prompt": ""}
    assert tasks["document"]["texts"] == ["zebra top\n\nnoise low", "whale mid"]
    assert built["artifacts"]["index_provenance"]["vectors"] == 2
    assert built["artifacts"]["index_provenance"]["aggregation"] is None
    assert built["artifacts"]["index_provenance"]["index_device"] == "cpu"
    encoder.clear()
    _, reused, _ = execute(
        tmp_path,
        dataset,
        "index",
        "reused",
        "launcher=local_gpu",
        "model.encoding.query_prompt='Different query: '",
    )
    assert encoder == []
    assert reused["artifacts"]["index_provenance"]["cache"]["document_hit"] is True

    def blocked(*args, **kwargs):
        raise AssertionError("Retrieval must not load/re-encode documents")

    monkeypatch.setattr("scientific_tot.adapter.DatasetAdapter.load_corpus", blocked)
    output, retrieved, _ = execute(
        tmp_path,
        dataset,
        "retrieve",
        "ranked",
        "launcher=local_gpu",
        f"paths.index={index}",
    )
    assert [list(call[2]) for call in encoder] == [["query"]]
    assert encoder[0][2]["query"]["kwargs"] == {"prompt": "Find: "}
    assert retrieved["artifacts"]["cache"]["query_hit"] is False
    assert (output / "run.trec").read_text().splitlines()[0].split()[2] == "ACL_2022.example-1.1"
    encoder.clear()
    _, repeated, _ = execute(
        tmp_path,
        dataset,
        "retrieve",
        "ranked-again",
        "launcher=local_gpu",
        f"paths.index={index}",
    )
    assert encoder == []
    assert repeated["artifacts"]["cache"]["query_hit"] is True
    _, changed, _ = execute(
        tmp_path,
        dataset,
        "retrieve",
        "changed-query",
        "launcher=local_gpu",
        f"paths.index={index}",
        "model.encoding.query_prompt='New: '",
    )
    assert [list(call[2]) for call in encoder] == [["query"]]
    assert changed["artifacts"]["encoding"]["explicit_prompt_arguments"] == {"prompt": "New: "}
    encoder.clear()
    shutil.rmtree(dataset)
    shutil.rmtree(index)
    monkeypatch.setattr("scientific_tot.adapter.DatasetAdapter.load_queries", blocked)
    metrics, evaluated, _ = execute(
        tmp_path, dataset, "evaluate", "metrics", "launcher=local_gpu", f"paths.run={output}"
    )
    assert encoder == []
    assert pd.read_csv(metrics / "metrics.csv")["RR@2"].iloc[0] == 1.0
    assert evaluated["artifacts"]["ranking_level"] == "paper"


def test_corrupt_dense_index_and_changed_document_prompt_fail_before_inference(
    tmp_path, dataset, encoder
):
    index, _, _ = execute(tmp_path, dataset, "index", "built")
    encoder.clear()
    with pytest.raises(ValueError, match="differs"):
        execute(
            tmp_path,
            dataset,
            "retrieve",
            "wrong-prompt",
            f"paths.index={index}",
            "model.encoding.document_prompt='Changed: '",
        )
    artifact = index / "index.faiss"
    artifact.chmod(0o644)
    artifact.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt"):
        execute(tmp_path, dataset, "retrieve", "corrupt", f"paths.index={index}")
    assert encoder == []


def test_qwen3_text_index_retrieve_and_evaluate(tmp_path, dataset, encoder):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        preset = compose(config_name="config", overrides=["experiment=qwen3_text"])
    prompt = preset.model.encoding.query_prompt
    common = (
        "experiment=qwen3_text",
        "retrieval.search_device=cpu",
        f"model.encoding.query_prompt='{prompt}'",
    )
    index, built, cfg = execute(tmp_path, dataset, "index", "qwen-built", *common)
    config, revision, tasks = encoder[0]
    assert config.model_id == "Qwen/Qwen3-Embedding-0.6B"
    assert config.precision == "bf16" and config.max_seq_length == 32768
    assert tasks["document"]["kwargs"] == {"prompt": ""}
    assert built["artifact_identity"]["index"]["model"]["name"] == "qwen3_text"
    from scientific_tot.artifact_identity import artifact_name

    assert artifact_name(cfg, {}).startswith("qwen3_text_fulltext_")
    encoder.clear()
    output, retrieved, _ = execute(
        tmp_path, dataset, "retrieve", "qwen-ranked", *common, f"paths.index={index}"
    )
    assert [list(call[2]) for call in encoder] == [["query"]]
    assert encoder[0][2]["query"]["kwargs"] == {"prompt": prompt}
    assert retrieved["artifacts"]["ranking_level"] == "paper"
    metrics, _, _ = execute(
        tmp_path, dataset, "evaluate", "qwen-metrics", *common, f"paths.run={output}"
    )
    assert pd.read_csv(metrics / "metrics.csv")["RR@2"].iloc[0] == 1.0


def test_qwen3_text_retries_failed_index_and_preserves_attempt(
    tmp_path, dataset, encoder, monkeypatch
):
    from scientific_tot.integrations import dense

    encode = dense._encode_task
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("interrupted encoding")
        return encode(*args, **kwargs)

    monkeypatch.setattr(dense, "_encode_task", interrupted)
    with pytest.raises(RuntimeError, match="interrupted encoding"):
        execute(tmp_path, dataset, "index", "qwen-retry", "experiment=qwen3_text")
    output, manifest, _ = execute(tmp_path, dataset, "index", "qwen-retry", "experiment=qwen3_text")
    assert manifest["status"] == "completed"
    archived = list((tmp_path / ".failed/qwen-retry").glob("*/manifest.json"))
    assert len(archived) == 1
    assert json.loads(archived[0].read_text())["status"] == "failed"
    assert (output / "index.faiss").is_file()


def test_qwen_retry_rejects_changed_source(tmp_path, dataset, encoder, monkeypatch):
    def interrupted(*args, **kwargs):
        raise RuntimeError("interrupted encoding")

    monkeypatch.setattr("scientific_tot.integrations.dense._encode_task", interrupted)
    with pytest.raises(RuntimeError, match="interrupted encoding"):
        execute(tmp_path, dataset, "index", "qwen-changed", "experiment=qwen3_text")
    (dataset / "preprocessed/ACL/ACL_2022.example-1.1/markdown.md").write_text("changed paper")
    with pytest.raises(ValueError, match="source identity differs"):
        execute(tmp_path, dataset, "index", "qwen-changed", "experiment=qwen3_text")
    assert not (tmp_path / ".failed").exists()
