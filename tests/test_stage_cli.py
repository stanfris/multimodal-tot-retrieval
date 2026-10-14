"""Automatic stage selection respects completed artifacts and configuration identity."""

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scientific_tot.config_artifacts import (
    BATCH_FILES,
    artifact_reference,
    digest,
    write_index_manifest,
)
from scientific_tot.stage_cli import plan


def completed_index(label, overrides=()):
    cfg, _ = plan("index", list(overrides), rebuild=True)
    attempt = Path(cfg.launcher.output_root) / label
    index = attempt / "index"
    index.mkdir(parents=True)
    write_index_manifest(
        index,
        cfg,
        provenance={"dataset_revision": None, "model_revision": None, "corpus_sha256": "a" * 64},
    )
    (attempt / "manifest.json").write_text(
        json.dumps(
            {
                "stage": "index",
                "status": "completed",
                "experiment_name": cfg.run.name,
                "index_location": str(index),
            }
        )
    )
    return index


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("TOT_DATA_ROOT", "TOT_CACHE_ROOT", "TOT_OUTPUT_ROOT"):
        monkeypatch.delenv(name, raising=False)


def test_reuse_and_rebuild():
    index = completed_index("first")
    assert plan("index", [])[1] is None
    assert plan("index", [], rebuild=True)[1] == ["stage=index"]
    assert plan("index", ["dry_run=true"])[1] == ["dry_run=true", "stage=index"]
    _, command = plan("retrieve", ["retrieval.top_k=200"])
    assert command[-1] == f"paths.index={json.dumps(str(index))}"


def test_newest_compatible_and_failed_attempts():
    old = completed_index("old")
    completed_index("incompatible", ["representation=title_abstract"])
    newest = completed_index("newest")
    manifest = newest.parent / "manifest.json"
    data = json.loads(manifest.read_text())
    data["status"] = "failed"
    manifest.write_text(json.dumps(data))
    assert str(old) in plan("retrieve", [])[1][-1]
    # Corrupt a completed index: selection must fail rather than infer its identity.
    (old / "index-manifest.json").chmod(0o644)
    (old / "index-manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="No compatible completed artifact"):
        plan("retrieve", [])


def test_explicit_inputs_and_study_names():
    completed_index("existing")
    assert plan("retrieve", ["paths.index=custom"])[1][-1] == "stage=retrieve"
    with pytest.raises(ValueError, match="No compatible completed artifact"):
        plan("retrieve", ["run.name=another-study"])


def test_evaluation_selects_intact_matching_batch():
    index = completed_index("build with spaces")
    cfg, _ = plan("retrieve", [])
    cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    cfg.paths.index = str(index)
    attempt = Path(cfg.launcher.output_root) / "batch"
    (attempt / ".hydra").mkdir(parents=True)
    OmegaConf.save(cfg, attempt / ".hydra/config.yaml")
    for name in BATCH_FILES:
        (attempt / name).write_text("")
    (attempt / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stage": "retrieve",
                "status": "completed",
                "experiment_name": cfg.run.name,
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
