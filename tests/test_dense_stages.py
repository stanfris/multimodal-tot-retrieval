"""Fast offline dense stages: real FAISS/splitting/aggregation, stubbed neural encoding."""

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
    calls = []

    def stub(config, revision, cache, tasks):
        assert revision == DEFAULT_MODEL_REVISION
        calls.append((config, revision, tasks))
        result = {}
        for task, request in tasks.items():
            vectors = np.array(
                [
                    [1.0, 0.0]
                    if "zebra" in text
                    else [0.75, 0.25]
                    if "whale" in text
                    else [0.0, 1.0]
                    for text in request["texts"]
                ],
                dtype=np.float32,
            )
            result[task] = vectors, {"dimension": 2, "explicit_prompt_arguments": request["kwargs"]}
        return result

    monkeypatch.setattr("scientific_tot.integrations.dense._encode_tasks", stub)
    return calls


def execute(base, dataset, stage, name, *overrides):
    output = base / name
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        raw = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=[
                "model=dense",
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
    output.mkdir()
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


PASSAGES = [
    "representation=passages",
    "representation.chunking.chunk_size=14",
    "representation.chunking.chunk_overlap=0",
]


@pytest.mark.parametrize(
    "method,first,rr", [("max", "ACL_2022.example-1.1", 1.0), ("mean", "ACL_2022.example-1.2", 0.5)]
)
def test_passage_aggregation_ranks_and_evaluates_canonical_papers(
    tmp_path, dataset, encoder, method, first, rr
):
    preparation, _, _ = execute(tmp_path, dataset, "prepare", "prepared", *PASSAGES)
    passages = [
        json.loads(line) for line in (preparation / "passages.jsonl").read_text().splitlines()
    ]
    assert [row["text"] for row in passages] == ["zebra top", "noise low", "whale mid"]
    assert encoder == []
    index, _, _ = execute(tmp_path, dataset, "index", "built", *PASSAGES)
    assert [list(call[2]) for call in encoder] == [["document"]]
    encoder.clear()
    output, manifest, _ = execute(
        tmp_path,
        dataset,
        "retrieve",
        "ranked",
        *PASSAGES,
        f"representation.aggregation.method={method}",
        f"paths.index={index}",
    )
    rows = [line.split() for line in (output / "run.trec").read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0][2] == first
    assert {row[2] for row in rows} == {"ACL_2022.example-1.1", "ACL_2022.example-1.2"}
    assert manifest["artifacts"]["aggregation"]["method"] == method
    assert manifest["artifacts"]["vector_search_depth"] == 3
    assert manifest["artifacts"]["ranking_level"] == "paper"
    encoder.clear()
    evaluated, result, _ = execute(
        tmp_path,
        dataset,
        "evaluate",
        "metrics",
        *PASSAGES,
        f"representation.aggregation.method={method}",
        f"paths.run={output}",
    )
    assert encoder == []
    assert pd.read_csv(evaluated / "metrics.csv")["RR@2"].iloc[0] == rr
    assert result["artifacts"]["aggregation"]["method"] == method
    _, rebuilt, _ = execute(
        tmp_path, dataset, "index", "reused", *PASSAGES, "representation.aggregation.method=mean"
    )
    assert encoder == []
    assert rebuilt["artifacts"]["index_provenance"]["cache"]["document_hit"] is True


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
