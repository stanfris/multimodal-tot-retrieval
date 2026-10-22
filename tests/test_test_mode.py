"""Opt-in split isolation and a real FAISS retrieval/evaluation pipeline."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf

from scientific_tot.artifact_identity import artifact_directory, resolve_identity
from scientific_tot.config_arguments import configured_adapter, configured_corpus
from scientific_tot.hydra_run import run_stage
from scientific_tot.stage_cli import plan

ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    "experiment=bm25_fulltext",
    "model=dense",
    "experiment=dense_passages",
    "experiment=colbert_fulltext",
    "experiment=colqwen_pages",
    "experiment=openclip_pages",
    "experiment=openclip_figures",
    "experiment=qwen3_vl_page",
    "experiment=qwen3_vl_multimodal",
    "experiment=qwen3_vl_figures",
]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("stage", ["index", "retrieve", "evaluate"])
def test_flag_selects_test_split_and_isolates_every_method(tmp_path, monkeypatch, method, stage):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TOT_TEST_ROOT", raising=False)
    cfg, command = plan(stage, [method, "--test", "dry_run=true"])
    assert cfg.test_mode and cfg.split.corpus == cfg.split.queries == "test"
    assert Path(cfg.launcher.output_root) == tmp_path / "test/results"
    assert Path(cfg.launcher.cache_root) == tmp_path / "test/cache"
    assert Path(cfg.dataset.cache_dir) == tmp_path / "test/cache/datasets"
    assert "mode=test" in command and "--test" not in command


def test_flag_is_required_and_outside_artifacts_are_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    normal, _ = plan("index", ["dry_run=true"])
    assert not normal.test_mode and normal.split.corpus == "all"
    for overrides in (
        ["split.corpus=all"],
        ["split.queries=train"],
        ["launcher.output_root=results"],
        ["launcher.cache_root=cache"],
    ):
        with pytest.raises(ValueError, match="Test mode"):
            plan("index", ["--test", "dry_run=true", *overrides])
    for stage, key in (("retrieve", "index"), ("evaluate", "run")):
        with pytest.raises(ValueError, match="inside test_root"):
            plan(stage, ["--test", f"paths.{key}={tmp_path / 'normal'}"])


def test_official_test_split_has_no_fixed_size(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg, _ = plan("index", ["--test", f"dataset.root={ROOT / 'tests/fixtures/adapter'}"])
    corpus = configured_corpus(cfg, configured_adapter(cfg))
    assert len(corpus) == 1 and corpus.report.skipped == 0


@pytest.mark.parametrize(
    "method", ["model=dense", "experiment=openclip_figures", "experiment=qwen3_vl_figures"]
)
def test_110_document_pipeline_is_separate_and_excludes_train(tmp_path, monkeypatch, method):
    from PIL import Image

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TOT_TEST_ROOT", raising=False)
    dataset = tmp_path / "data"
    shutil.copytree(ROOT / "tests/fixtures/adapter", dataset)
    papers = dataset / "preprocessed/ACL"
    template = json.loads(next(papers.glob("*/paper.json")).read_text())
    shutil.rmtree(papers)
    test_count = 110
    sources = []
    for number in range(1, test_count + 2):
        doc_id = f"ACL_2022.example-1.{number}"
        source = f"ACL/2022.example-1.{number}.pdf"
        sources.append(source)
        target = papers / doc_id
        target.mkdir(parents=True)
        (target / "paper.json").write_text(
            json.dumps(
                {**template, "paper_id": doc_id, "source_pdf_relpath": source, "figures": []}
            )
        )
        (target / "markdown.md").write_text("target" if number == 1 else "distractor")
        if "figures" in method:
            # Reproduce a restored lean corpus with no per-paper/figure metadata.
            (target / "paper.json").unlink()
            (target / "images").mkdir()
            Image.new("RGB", (16 if number == 1 else 8, 8)).save(target / "images/figure.png")
    for number in range(10):
        unrelated = papers / f"ACL_2026.findings-acl.{number}" / "images"
        unrelated.mkdir(parents=True)
        (unrelated / "figure.png").write_bytes(b"unrelated image must never be read")
    (dataset / "splits/pdf_dataset_split.json").write_text(
        json.dumps({"train": sources[test_count:], "test": sources[:test_count]})
    )
    monkeypatch.setattr("scientific_tot.integrations.dense.require_cuda", lambda: None)

    def encode(config, revision, cache, task, texts, kwargs):
        values = [[1, 0] if task == "query" or text == "target" else [0, 1] for text in texts]
        return np.asarray(values, dtype=np.float32), {"dimension": 2}

    monkeypatch.setattr("scientific_tot.integrations.dense._encode_task", encode)

    class VisualEncoder:
        metadata = {}
        truncated = 0

        def __init__(self, cfg):
            self.dimension = cfg.model.dimension

        def encode(self, values, task):
            import torch

            vectors = torch.zeros(len(values), self.dimension)
            for number, value in enumerate(values):
                vectors[number, 0 if task == "query" or value["image"].width == 16 else 1] = 1
            return vectors

    monkeypatch.setattr("scientific_tot.integrations.visual_dense.load_model", VisualEncoder)
    arguments = ["--test", method, f"dataset.root={dataset}"]
    if "figures" in method:
        arguments.extend(["launcher=local_cpu", "model.precision=fp32"])
    outputs = {}
    for stage in ("index", "retrieve", "evaluate"):
        cfg, command = plan(stage, arguments)
        target = artifact_directory(cfg, resolve_identity(cfg))
        target.mkdir(parents=True)
        run_stage(cfg, target, OmegaConf.create({"overrides": {"task": command}}), ROOT)
        outputs[stage] = target
        assert target.is_relative_to(tmp_path / "test/results")
        assert json.loads((target / "manifest.json").read_text())["status"] == "completed"
    index = json.loads((outputs["index"] / "index-manifest.json").read_text())
    assert index["provenance"]["documents"] == index["provenance"]["vectors"] == test_count
    assert index["index_settings"]["test_mode"] is True
    assert Path(index["provenance"]["cache"]["documents"]).is_relative_to(tmp_path / "test/cache")
    corpus = pd.read_json(outputs["retrieve"] / "corpus-ids.jsonl", lines=True)
    assert len(corpus) == test_count
    assert f"ACL_2022.example-1.{test_count + 1}" not in set(corpus.doc_id)
    assert len((outputs["retrieve"] / "run.trec").read_text().splitlines()) == 100
    assert pd.read_csv(outputs["evaluate"] / "metrics.csv")["RR@10"].iloc[0] == 1.0
    assert not (tmp_path / "results").exists() and not (tmp_path / "cache").exists()

    # Even an explicitly supplied test index cannot be used without the flag.
    normal, _ = plan(
        "retrieve",
        [
            method,
            "split.corpus=test",
            f"dataset.root={dataset}",
            f"paths.index={outputs['index']}",
        ],
    )
    from scientific_tot.config_artifacts import verify_index

    with pytest.raises(ValueError, match="configuration differs"):
        verify_index(outputs["index"], normal)
