"""Stage scripts share deterministic index selection with the main Hydra CLI."""

import json
import shutil
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scientific_tot.artifact_identity import artifact_directory, records_digest, resolve_identity
from scientific_tot.config_artifacts import (
    BATCH_FILES,
    artifact_reference,
    digest,
    write_index_manifest,
)
from scientific_tot.configuration import resolve_config
from scientific_tot.hydra_run import run_stage
from scientific_tot.stage_cli import plan

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    shutil.copytree(ROOT / "tests/fixtures/adapter", tmp_path / "data")
    for name in ("TOT_DATA_ROOT", "TOT_CACHE_ROOT", "TOT_OUTPUT_ROOT"):
        monkeypatch.delenv(name, raising=False)


def completed_index(monkeypatch):
    cfg, _ = plan("index", [])
    target = artifact_directory(cfg, resolve_identity(cfg))
    target.mkdir(parents=True)

    def execute(cfg, **kwargs):
        from scientific_tot.config_arguments import configured_adapter, corpus_arguments

        corpus = configured_adapter(cfg).load_corpus(**corpus_arguments(cfg))
        (target / "data.properties").write_text("offline index")
        write_index_manifest(
            target,
            cfg,
            provenance={
                "dataset_revision": None,
                "model_revision": None,
                "corpus_sha256": records_digest(
                    {"docno": row["doc_id"], "text": row["text"]} for row in corpus
                ),
            },
        )

    monkeypatch.setattr("scientific_tot.hydra_run.execute_stage", execute)
    run_stage(cfg, target, OmegaConf.create({"overrides": {"task": []}}), ROOT)
    return target


def test_index_always_dispatches_for_verified_reuse(monkeypatch):
    index = completed_index(monkeypatch)
    assert plan("index", [])[1] == ["stage=index"]
    assert plan("index", [], rebuild=True)[1] == ["stage=index"]
    assert plan("index", ["dry_run=true"])[1] == ["dry_run=true", "stage=index"]
    _, command = plan("retrieve", ["retrieval.top_k=200", "split.queries=train"])
    assert command[-1] == f"paths.index={json.dumps(str(index))}"


def test_corrupt_and_changed_corpus_cannot_select_stale_index(monkeypatch):
    index = completed_index(monkeypatch)
    (index / "data.properties").write_text("corrupt")
    with pytest.raises(FileExistsError, match="checksum"):
        plan("retrieve", [])
    paper = next(Path("data").glob("preprocessed/**/markdown.md"))
    paper.write_text("changed corpus")
    with pytest.raises(ValueError, match="No compatible completed artifact"):
        plan("retrieve", [])


def test_explicit_inputs_and_labels(monkeypatch):
    index = completed_index(monkeypatch)
    assert plan("retrieve", ["paths.index=custom"])[1][-1] == "stage=retrieve"
    assert str(index) in plan("retrieve", ["run.name=another-study"])[1][-1]


def test_evaluation_selects_intact_matching_batch(monkeypatch):
    index = completed_index(monkeypatch)
    cfg, _ = plan("retrieve", [])
    cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    cfg.paths.index = str(index)
    cfg = resolve_config(cfg, Path.cwd())
    attempt = Path(cfg.launcher.output_root) / "models/batch"
    (attempt / ".hydra").mkdir(parents=True)
    OmegaConf.save(cfg, attempt / ".hydra/config.yaml")
    for name in BATCH_FILES:
        (attempt / name).write_text("")
    (attempt / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "stage": "retrieve",
                "status": "completed",
                "experiment_name": cfg.run.name,
                "artifact_identity": {"source": "test"},
                "artifact_sha256": digest({"source": "test"}),
                "configuration_sha256": digest(OmegaConf.to_container(cfg)),
                "artifacts": {
                    "outputs": {name: artifact_reference(attempt / name) for name in BATCH_FILES}
                },
            }
        )
    )
    assert str(attempt) in plan("evaluate", [])[1][-1]
    with pytest.raises(ValueError, match="No compatible completed artifact"):
        plan("evaluate", ["retrieval.top_k=200"])
    (attempt / "run.trec").write_text("changed")
    with pytest.raises(ValueError, match="No compatible completed artifact"):
        plan("evaluate", [])
