"""Immutable configuration contracts for indexes and retrieval outputs, without backends."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from scientific_tot.configuration import OPENCLIP_MODELS, VISUAL_DENSE_MODELS

BATCH_FILES = ("run.trec", "topics.jsonl", "qrels.tsv", "query-ids.jsonl", "corpus-ids.jsonl")


def artifact_reference(path: Path) -> dict:
    """An exact absolute input/output reference and its content fingerprint."""
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return {"path": str(path.resolve()), "sha256": checksum.hexdigest()}


def digest(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def save_json(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o444)


def index_settings(cfg: DictConfig, provenance: dict | None = None) -> dict:
    """Only corpus/index identity; query selection and retrieval controls may vary."""
    data = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    model = data["model"]
    model.pop("retrieval", None)
    if model["name"] == "dense":
        model["encoding"].pop("query_prompt")
    if model["name"] in VISUAL_DENSE_MODELS - OPENCLIP_MODELS:
        model.pop("query_instruction")
    representation = data["representation"]
    if representation["name"] == "passages":
        representation.pop("aggregation")
    dataset = data["dataset"]
    dataset = {
        k: dataset[k]
        for k in ("source", "repo_id", "revision", "root", "artifacts", "domains", "missing")
    }
    if provenance is not None:
        dataset["revision"] = provenance["dataset_revision"]
        if model["name"] == "dense":
            model["revision"] = provenance["model_revision"]
    return {
        "dataset": dataset,
        "corpus_split": data["split"]["corpus"],
        "representation": representation,
        "model": model,
    }


def require_index_reservation(target: Path) -> None:
    """Builders may share a stage reservation, never an existing backend index."""
    if {path.name for path in target.iterdir()} != {".run.lock", ".hydra", "manifest.json"}:
        raise FileExistsError(f"Refusing to overwrite nonempty index destination: {target}")
    manifest = json.loads((target / "manifest.json").read_text())
    if (
        manifest.get("stage") != "index"
        or manifest.get("status") != "started"
        or manifest.get("index_location") != str(target.resolve())
    ):
        raise FileExistsError(f"Index destination lacks a fresh stage reservation: {target}")


def write_index_manifest(index: Path, cfg: DictConfig, *, provenance: dict) -> None:
    """Called only after a builder completes; legacy indexes require explicit migration.

    provenance must record resolved dataset/model commits (or null for local/unrecorded),
    and a corpus content digest; requested revisions alone do not verify local bytes.
    """
    assert cfg.stage.name == "index", "Index manifests require a stage=index build configuration"
    _validate_provenance(cfg, provenance)
    configuration = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    from scientific_tot.artifact_identity import index_identity

    identity = {"stage": "index", "index": index_identity(cfg, provenance)}
    files = {
        str(path.relative_to(index)): artifact_reference(path)
        for path in sorted(index.rglob("*"))
        if path.is_file()
        and ".hydra" not in path.relative_to(index).parts
        and path.name not in {"manifest.json", ".run.lock", "index-manifest.json"}
    }
    save_json(
        index / "index-manifest.json",
        {
            "schema_version": 2,
            "status": "completed",
            "configuration": configuration,
            "configuration_sha256": digest(configuration),
            "index_settings": index_settings(cfg, provenance),
            "artifact_identity": identity,
            "artifact_sha256": digest(identity),
            "index_files": files,
            "provenance": provenance,
        },
    )


def verify_index(index: Path, cfg: DictConfig, *, allow_building: bool = False) -> dict:
    """Check completion, compatible metadata, and file presence without hashing index data."""
    manifest = json.loads((index / "index-manifest.json").read_text())
    if manifest.get("schema_version") != 2 or manifest.get("status") != "completed":
        raise ValueError("Index manifest is not a completed supported artifact")
    if (
        cfg.model.name == "dense"
        and cfg.stage.name != "index"
        and (
            not isinstance(cfg.model.revision, str)
            or not re.fullmatch(r"[0-9a-f]{40}", cfg.model.revision)
        )
    ):
        raise ValueError("Dense index consumers must pin model.revision to the build commit")
    configuration = manifest["configuration"]
    if digest(configuration) != manifest["configuration_sha256"]:
        raise ValueError("Index configuration snapshot checksum differs")
    provenance = manifest.get("provenance", {})
    built_config = OmegaConf.create(configuration)
    _validate_provenance(built_config, provenance)
    built = index_settings(built_config, provenance)
    if built != manifest["index_settings"] or built != index_settings(
        cfg, provenance if cfg.stage.name == "index" else None
    ):
        raise ValueError("Index configuration differs from requested corpus/model/representation")
    _validate_provenance(cfg, provenance)
    if not manifest.get("artifact_identity") or not manifest.get("index_files"):
        raise ValueError("Incomplete index identity/file inventory")
    stage_manifest = index / "manifest.json"
    if stage_manifest.exists() and not allow_building:
        stage = json.loads(stage_manifest.read_text())
        if (
            stage.get("status") != "completed"
            or stage.get("artifact_identity") != manifest["artifact_identity"]
        ):
            raise ValueError("Index stage manifest is incomplete or differs from index identity")
    from scientific_tot.artifact_identity import index_identity

    identity = {"stage": "index", "index": index_identity(built_config, provenance)}
    if identity != manifest["artifact_identity"] or digest(identity) != manifest.get(
        "artifact_sha256"
    ):
        raise ValueError("Index identity checksum differs")
    for name in manifest["index_files"]:
        if not (index / name).is_file():
            raise FileNotFoundError(f"Incomplete persistent index: {index / name}")
    return manifest


def verify_retrieval(directory: Path, cfg: DictConfig | None = None) -> dict:
    """Metrics consume a completed batch artifact; they never implicitly retrieve."""
    manifest = json.loads((directory / "manifest.json").read_text())
    if (
        manifest.get("schema_version") != 2
        or manifest.get("status") != "completed"
        or manifest.get("stage") != "retrieve"
    ):
        raise ValueError("paths.run must identify a completed retrieve stage")
    identity = manifest.get("artifact_identity")
    if not identity or digest(identity) != manifest.get("artifact_sha256"):
        raise ValueError("Retrieval identity checksum differs")
    snapshot = OmegaConf.to_container(
        OmegaConf.load(directory / ".hydra/config.yaml"), resolve=True
    )
    if digest(snapshot) != manifest.get("configuration_sha256"):
        raise ValueError("Retrieval configuration snapshot checksum differs")
    outputs = manifest.get("artifacts", {}).get("outputs", {})
    for name in BATCH_FILES:
        if not (directory / name).is_file():
            raise ValueError(f"Incomplete retrieval artifact: {name}")
        if outputs.get(name) != artifact_reference(directory / name):
            raise ValueError(f"Retrieval artifact reference/checksum differs: {name}")
    from scientific_tot.configuration import resolve_config

    built = resolve_config(OmegaConf.create(snapshot), directory.parent)
    if built.stage.name != "retrieve" or built.model.name not in VISUAL_DENSE_MODELS | {
        "bm25",
        "dense",
        "colbert",
        "colqwen",
    }:
        raise ValueError("Saved batch configuration must describe a supported retrieve stage")
    if cfg is not None:
        if (
            index_settings(built) != index_settings(cfg)
            or built.dataset.query_mode != cfg.dataset.query_mode
            or built.split.queries != cfg.split.queries
            or built.model.get("retrieval") != cfg.model.get("retrieval")
            or built.representation != cfg.representation
            or built.model.name == "dense"
            and built.model.encoding != cfg.model.encoding
            or built.model.name in VISUAL_DENSE_MODELS - OPENCLIP_MODELS
            and built.model.query_instruction != cfg.model.query_instruction
        ):
            raise ValueError("Retrieval experiment differs from requested evaluation configuration")
        if any(int(m.split("@")[1]) > built.retrieval.top_k for m in cfg.evaluation.metrics):
            raise ValueError("Metric cutoff exceeds the saved retrieval.top_k")
    return manifest


def _validate_provenance(cfg: DictConfig, provenance: dict) -> None:
    required = {"dataset_revision", "model_revision", "corpus_sha256"}
    if not required <= set(provenance) or not provenance["corpus_sha256"]:
        raise ValueError("Index provenance requires revisions and corpus_sha256")
    if not isinstance(provenance["corpus_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", provenance["corpus_sha256"]
    ):
        raise ValueError("corpus_sha256 must be a full SHA256 digest")
    for field in ("dataset_revision", "model_revision"):
        actual = provenance[field]
        if actual is not None and (
            not isinstance(actual, str) or not re.fullmatch(r"[0-9a-f]{40}", actual)
        ):
            raise ValueError(f"Resolved {field} must be an immutable commit SHA or null")
    if cfg.dataset.source == "hub" and provenance["dataset_revision"] is None:
        raise ValueError("Hub index requires a resolved dataset revision")
    if (
        cfg.model.name in {"dense", "colbert", "colqwen"} | VISUAL_DENSE_MODELS
        and provenance["model_revision"] is None
    ):
        raise ValueError(f"{cfg.model.name} index requires a resolved model revision")
    for field, requested in (
        ("dataset_revision", cfg.dataset.revision),
        ("model_revision", cfg.model.get("revision")),
    ):
        actual = provenance[field]
        if (
            requested is not None
            and re.fullmatch(r"[0-9a-f]{40}", requested)
            and requested != actual
        ):
            raise ValueError(f"Index resolved {field} differs from requested revision")
