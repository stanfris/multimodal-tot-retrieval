"""Content-derived filesystem identities, shared by Hydra and manifest publication."""

from __future__ import annotations

import json
import tempfile
import uuid
from importlib.metadata import version
from pathlib import Path

from omegaconf import OmegaConf

from scientific_tot.config_arguments import (
    configured_adapter,
    corpus_arguments,
    dense_arguments,
    query_arguments,
)
from scientific_tot.config_artifacts import (
    artifact_reference,
    digest,
    index_settings,
    verify_index,
    verify_retrieval,
)
from scientific_tot.configuration import resolve_config
from scientific_tot.records import records_digest

PLANNED_IDENTITIES: dict[Path, dict] = {}


def index_identity(cfg, provenance: dict) -> dict:
    settings = index_settings(cfg, provenance)
    # Locations are provenance, not content. Query artifact locations cannot affect indexes.
    settings["dataset"] = {
        k: v for k, v in settings["dataset"].items() if k not in {"root", "artifacts"}
    }
    settings["corpus_sha256"] = provenance["corpus_sha256"]
    if cfg.model.name == "bm25":
        settings["terrier_version"] = cfg.launcher.terrier_version
        settings["indexing_threads"] = cfg.launcher.threads.indexing
        settings["versions"] = {"pyterrier": version("pyterrier")}
    elif cfg.model.name == "dense":
        settings["encoding_threads"] = cfg.launcher.threads.encoding
        settings["faiss_threads"] = cfg.launcher.threads.faiss
        names = ["sentence-transformers", "torch", "faiss-cpu", "numpy"]
        if cfg.representation.name == "passages":
            names.append("langchain-text-splitters")
        settings["versions"] = {name: version(name) for name in names}
    return settings


def _resolve_identity(cfg) -> dict:
    stage = cfg.stage.name
    if stage == "evaluate":
        source = Path(cfg.paths.run)
        manifest = verify_retrieval(source, cfg)
        return {
            "stage": stage,
            "source_run": artifact_reference(source / "manifest.json"),
            "source_identity": manifest.get("artifact_identity"),
            "evaluation": OmegaConf.to_container(cfg.evaluation, resolve=True),
        }
    if stage == "retrieve":
        source = Path(cfg.paths.index)
        manifest = verify_index(source, cfg)
        queries = configured_adapter(cfg).load_queries(**query_arguments(cfg))
        return {
            "stage": stage,
            "source_index": artifact_reference(source / "index-manifest.json"),
            "source_identity": manifest.get("artifact_identity"),
            "model": OmegaConf.to_container(cfg.model, resolve=True),
            "representation": OmegaConf.to_container(cfg.representation, resolve=True),
            "queries": {
                "split": cfg.split.queries,
                "mode": cfg.dataset.query_mode,
                "revision": queries.provenance["revision"],
                "sha256": records_digest(queries),
            },
            "retrieval": OmegaConf.to_container(cfg.retrieval, resolve=True),
            "threads": OmegaConf.to_container(cfg.launcher.threads),
            "run_name": cfg.run.name,
        }
    adapter = configured_adapter(cfg)
    corpus = adapter.load_corpus(**corpus_arguments(cfg))
    if stage == "prepare":
        from scientific_tot.adapter import validate_qrels

        queries = adapter.load_queries(**query_arguments(cfg))
        qrels = validate_qrels(queries, corpus)
        return {
            "stage": stage,
            "corpus": records_digest(corpus),
            "corpus_provenance": corpus.provenance,
            "queries": records_digest(queries),
            "qrels": records_digest(qrels),
            "query_selection": {"split": cfg.split.queries, "mode": cfg.dataset.query_mode},
            "representation": OmegaConf.to_container(cfg.representation, resolve=True),
        }
    revision = None
    if cfg.model.name == "dense":
        from scientific_tot.integrations.dense import EncodingConfig, resolve_model_revision

        revision = resolve_model_revision(EncodingConfig(**dense_arguments(cfg)))
    provenance = {
        "dataset_revision": corpus.provenance["revision"],
        "model_revision": revision,
        "corpus_sha256": records_digest(
            {"docno": row["doc_id"], "text": row["text"]} for row in corpus
        ),
    }
    return {"stage": stage, "index": index_identity(cfg, provenance)}


def resolve_identity(cfg) -> dict:
    """Resolve selected inputs into the same canonical JSON values stored in manifests."""
    return json.loads(json.dumps(_resolve_identity(cfg), allow_nan=False))


def artifact_name(cfg, identity: dict) -> str:
    stem = f"{cfg.model.name}_{cfg.representation.name}_c-{cfg.split.corpus}"
    if cfg.stage.name != "index":
        stem += f"_q-{cfg.split.queries}"
    return f"{stem}_{digest(identity)[:16]}"


def artifact_directory(cfg, identity: dict) -> Path:
    if cfg.stage.name == "index" and cfg.paths.index is not None:
        return Path(cfg.paths.index)
    group = {
        "index": "indexes",
        "retrieve": "models",
        "evaluate": "evaluations",
        "prepare": "prepared",
    }[cfg.stage.name]
    return Path(cfg.launcher.output_root) / group / artifact_name(cfg, identity)


def hydra_directory(*, _root_) -> str:
    task = OmegaConf.masked_copy(_root_, [key for key in _root_ if key != "hydra"])
    cfg = resolve_config(task, Path.cwd())
    if cfg.dry_run:
        # Dry runs validate without touching dataset/model sources or pretending to resolve content.
        identity = {"dry_run": OmegaConf.to_container(cfg, resolve=True)}
        return str(Path(cfg.launcher.output_root) / "dry-runs" / artifact_name(cfg, identity))
    print(f"Planning {cfg.stage.name}: resolving selected inputs...", flush=True)
    identity = resolve_identity(cfg)
    directory = artifact_directory(cfg, identity)
    if directory in PLANNED_IDENTITIES and PLANNED_IDENTITIES[directory] != identity:
        raise ValueError(
            "Source inputs changed or conflicting identities require distinct destinations"
        )
    PLANNED_IDENTITIES[directory] = identity
    return str(directory)


OmegaConf.register_new_resolver("artifact_dir", hydra_directory, replace=True)


def sweep_directory() -> str:
    """Hydra's mutable orchestration metadata is temporary, outside artifact trees."""
    return str(Path(tempfile.gettempdir()) / "scientific-tot-sweeps" / uuid.uuid4().hex)


OmegaConf.register_new_resolver("sweep_dir", sweep_directory, replace=True, use_cache=True)
