"""Sentence Transformers encoding, FAISS search, and shared standard evaluation."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import HfApi

from scientific_tot.config_artifacts import artifact_reference
from scientific_tot.integrations.evaluation import evaluate_run
from scientific_tot.integrations.pyterrier import to_pyterrier
from scientific_tot.records import Records, records_digest

DEFAULT_MODEL = "intfloat/e5-small-v2"
DEFAULT_MODEL_REVISION = "ffb93f3bd4047442299a41ebb6fa998a38507c52"
E5_MODELS = {
    f"intfloat/{family}-{size}{suffix}"
    for family, suffixes in [("e5", ("", "-v2")), ("multilingual-e5", ("",))]
    for size in ("small", "base", "large")
    for suffix in suffixes
}


@dataclass(frozen=True)
class EncodingConfig:
    model_id: str = DEFAULT_MODEL
    model_revision: str | None = None
    device: str = "cpu"
    batch_size: int = 16
    max_seq_length: int | None = None
    profile: str = "auto"
    query_prompt: str | None = None
    document_prompt: str | None = None
    query_prompt_name: str | None = None
    document_prompt_name: str | None = None
    similarity: str = "cosine"
    faiss_threads: int = 1
    encoding_threads: int = 1

    def __post_init__(self):
        if self.batch_size < 1 or self.faiss_threads < 1 or self.encoding_threads < 1:
            raise ValueError("Batch size and thread counts must be positive")
        if self.max_seq_length is not None and self.max_seq_length < 1:
            raise ValueError("max_seq_length must be positive")
        if self.profile not in {"auto", "model", "e5"} or self.similarity not in {"cosine", "dot"}:
            raise ValueError("Unknown encoding profile or similarity")
        for task in ("query", "document"):
            if getattr(self, f"{task}_prompt") is not None and getattr(self, f"{task}_prompt_name"):
                raise ValueError("Choose a prompt string or a prompt name, not both")


def encoding_kwargs(config: EncodingConfig, task: str) -> dict:
    """Pass instructions through Sentence Transformers' prompt API exactly once."""
    prompt = getattr(config, f"{task}_prompt")
    prompt_name = getattr(config, f"{task}_prompt_name")
    if prompt is not None:
        return {"prompt": prompt}
    if prompt_name is not None:
        return {"prompt_name": prompt_name}
    if (
        "e5" in config.model_id.lower()
        and "instruct" in config.model_id.lower()
        and task == "query"
    ):
        raise ValueError("E5 instruct models need an explicit task-specific query prompt")
    e5 = config.profile == "e5" or config.profile == "auto" and config.model_id in E5_MODELS
    if e5:
        return {"prompt": "query: " if task == "query" else "passage: "}
    # encode_query/encode_document select the saved model prompts and Router task.
    return {}


def resolve_model_revision(config: EncodingConfig) -> str:
    revision = config.model_revision
    if revision is None and config.model_id == DEFAULT_MODEL:
        return DEFAULT_MODEL_REVISION
    if revision and re.fullmatch(r"[0-9a-f]{40}", revision):
        return revision
    return HfApi().model_info(config.model_id, revision=revision).sha


def verify_baseline(inputs, directory: str | Path) -> dict:
    """Refuse a comparison unless corpus text, IDs, topics and qrels match BM25."""
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text())
    if records_digest(inputs.documents) != manifest["corpus_sha256"]:
        raise ValueError("Corpus text/order/IDs differ from the BM25 baseline")
    if [d["docno"] for d in inputs.documents] != json.loads(
        (root / "selected-documents.json").read_text()
    ):
        raise ValueError("Document IDs differ from the BM25 baseline")
    topics = pd.read_json(root / "topics.jsonl", lines=True, dtype={"qid": str, "query": str})
    mapping = pd.read_json(
        root / "query-ids.jsonl",
        lines=True,
        dtype={"qid": str, "query_id": str, "query_key": str},
    )
    qrels = pd.read_csv(
        root / "qrels.tsv", sep="\t", dtype={"qid": str, "docno": str}, keep_default_na=False
    )
    for actual, expected in [
        (inputs.topics, topics),
        (inputs.qrels, qrels),
        (inputs.query_ids, mapping),
    ]:
        if actual.to_dict("records") != expected.to_dict("records"):
            raise ValueError("Queries, qrels or query mappings differ from the BM25 baseline")
    return {
        "directory": str(root.resolve()),
        "manifest_sha256": artifact_reference(root / "manifest.json")["sha256"],
    }


def _read_cache(path: Path, expected: dict):
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["configuration"] != expected:
        raise ValueError("Embedding cache configuration does not match")
    for name, digest in manifest["files"].items():
        if artifact_reference(path / name)["sha256"] != digest:
            raise ValueError(f"Corrupt embedding/index cache: {name}")
    vectors = np.load(path / "embeddings.npy", mmap_mode="r", allow_pickle=False)
    if vectors.ndim != 2 or vectors.dtype != np.float32 or not np.isfinite(vectors).all():
        raise ValueError("Invalid cached embeddings")
    return vectors, manifest


def _write_cache(path: Path, configuration: dict, vectors, metadata: dict, *, index=None, ids=None):
    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temp:
        staging = Path(temp) / "entry"
        staging.mkdir()
        np.save(staging / "embeddings.npy", vectors, allow_pickle=False)
        if index is not None:
            faiss.write_index(index, str(staging / "index.faiss"))
            (staging / "ids.json").write_text(json.dumps(ids) + "\n")
        manifest = {
            "configuration": configuration,
            "encoding": metadata,
            "files": {p.name: artifact_reference(p)["sha256"] for p in staging.iterdir()},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # A completed entry becomes visible only after all library files are written.
        try:
            os.rename(staging, path)
        except FileExistsError:
            _read_cache(path, configuration)


def _encode_tasks(config: EncodingConfig, revision: str, model_cache: Path, tasks: dict) -> dict:
    import torch
    from sentence_transformers import SentenceTransformer

    # Explicit CPU threading avoids oversubscription and the macOS ARM
    # FAISS/PyTorch OpenMP crash with the libraries' default thread pools.
    torch.set_num_threads(config.encoding_threads)
    model = SentenceTransformer(
        config.model_id,
        revision=revision,
        device=config.device,
        cache_folder=str(model_cache),
        trust_remote_code=False,
    )
    if config.max_seq_length is not None:
        model.max_seq_length = config.max_seq_length
    encoded = {}
    for task, request in tasks.items():
        vectors = getattr(model, f"encode_{task}")(
            request["texts"],
            batch_size=config.batch_size,
            normalize_embeddings=config.similarity == "cosine",
            precision="float32",
            convert_to_numpy=True,
            show_progress_bar=True,
            **request["kwargs"],
        )
        vectors = np.asarray(vectors, dtype=np.float32, order="C")
        if (
            vectors.ndim != 2
            or len(vectors) != len(request["texts"])
            or vectors.shape[1] < 1
            or not np.isfinite(vectors).all()
        ):
            raise ValueError("Model returned invalid embeddings")
        metadata = {
            "max_seq_length": model.max_seq_length,
            "cpu_threads": torch.get_num_threads(),
            "saved_prompts": model.prompts,
            "default_prompt_name": model.default_prompt_name,
            "task": task,
            "explicit_prompt_arguments": request["kwargs"],
            "dimension": vectors.shape[1],
            "long_text_policy": "model_tokenizer_truncation_no_chunking",
        }
        encoded[task] = vectors, metadata
    return encoded


def cached_encoding(
    config: EncodingConfig,
    revision: str,
    cache: Path,
    *,
    task: str,
    rows: list[dict],
    dataset_revision: str | None,
    representation: dict | None = None,
) -> tuple:
    """Independent task caches using the existing encoder, atomic cache and FAISS I/O.

    A document cache key excludes query prompts; query keys exclude document prompts.
    Only a cache miss loads Sentence Transformers, and each call encodes one task.
    """
    import faiss

    if task not in {"document", "query"} or not rows:
        raise ValueError("Choose a nonempty document or query encoding task")
    kwargs = encoding_kwargs(config, task)
    model = asdict(config)
    other = "query" if task == "document" else "document"
    for field in (f"{other}_prompt", f"{other}_prompt_name", "faiss_threads"):
        model.pop(field)
    model.update(model_revision=revision, resolved_revision=revision)
    configuration = {
        "schema_version": 2,
        "task": task,
        "model": model,
        "prompt_arguments": kwargs,
        "content_sha256": records_digest(rows),
        "dataset_revision": dataset_revision,
        "precision": "float32",
        "versions": {
            name: version(name)
            for name in ("sentence-transformers", "transformers", "torch", "numpy", "faiss-cpu")
        },
    }
    if task == "document":
        configuration.update(representation=representation, index="FAISS IndexFlatIP")
    entry = cache / records_digest([configuration])
    hit = entry.exists()
    print(
        f"{task.capitalize()} embeddings: {len(rows):,} inputs; "
        f"cache={'hit' if hit else 'miss'}, model={config.model_id}@{revision}",
        flush=True,
    )
    faiss.omp_set_num_threads(config.faiss_threads)
    if not hit:
        text_key = "text" if task == "document" else "query"
        vectors, metadata = _encode_tasks(
            config,
            revision,
            cache / "models",
            {task: {"texts": [row[text_key] for row in rows], "kwargs": kwargs}},
        )[task]
        index = None
        ids = None
        if task == "document":
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            ids = [row["docno"] for row in rows]
        _write_cache(entry, configuration, vectors, metadata, index=index, ids=ids)
    vectors, manifest = _read_cache(entry, configuration)
    if len(vectors) != len(rows):
        raise ValueError("Cached embedding count differs from the configured input")
    return vectors, manifest, entry, hit


def dense_experiment(
    corpus: Records,
    queries: Records,
    output: str | Path,
    *,
    cache: str | Path,
    config: EncodingConfig | None = None,
    baseline: str | Path | None = None,
    settings=None,
) -> pd.DataFrame:
    """Encode in library batches, search a persistent FAISS IndexFlatIP, evaluate.

    NumPy .npy files (memory-mapped on reuse) and FAISS's own index I/O persist
    embeddings and the index. Fully cached runs neither load a model nor start Java.
    """
    import faiss

    if settings is not None:
        from scientific_tot.config_arguments import dense_arguments, require_corpus_settings
        from scientific_tot.configuration import resolve_config

        if config is not None:
            raise ValueError("Pass Hydra settings or EncodingConfig, not both")
        settings = resolve_config(settings, Path.cwd(), Path(output).resolve())
        if settings.representation.name == "passages":
            raise NotImplementedError("Passage execution requires the separate Hydra stages")
        require_corpus_settings(settings, corpus, queries)
        if Path(cache).resolve() != Path(settings.launcher.cache_root) / "embeddings":
            raise ValueError("Hydra settings require cache=launcher.cache_root/embeddings")
        config = EncodingConfig(**dense_arguments(settings))
    config = config or EncodingConfig()
    top_k = settings.retrieval.top_k if settings is not None else 100
    inputs = to_pyterrier(corpus, queries)
    baseline_info = verify_baseline(inputs, baseline) if baseline is not None else None
    representations = {r["representation"] for r in corpus}
    if len(representations) != 1:
        raise ValueError("Use one corpus representation per experiment")
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(f"Choose a fresh experiment directory: {target}")
    resolved_revision = resolve_model_revision(config)
    cache_root = Path(cache).resolve()
    corpus_sha = records_digest(inputs.documents)
    doc_vectors, doc_manifest, doc_cache, doc_hit = cached_encoding(
        config,
        resolved_revision,
        cache_root,
        task="document",
        rows=inputs.documents,
        dataset_revision=corpus.provenance.get("revision"),
        representation={"name": next(iter(representations))},
    )
    ids = [d["docno"] for d in inputs.documents]
    index = faiss.read_index(str(doc_cache / "index.faiss"))
    cached_ids = json.loads((doc_cache / "ids.json").read_text())
    if (
        cached_ids != ids
        or len(doc_vectors) != len(ids)
        or index.ntotal != len(ids)
        or index.d != doc_vectors.shape[1]
    ):
        raise ValueError("Cached FAISS index/ID mapping does not match canonical corpus")
    query_vectors, query_manifest, query_cache, query_hit = cached_encoding(
        config,
        resolved_revision,
        cache_root,
        task="query",
        rows=inputs.topics.to_dict("records"),
        dataset_revision=queries.provenance.get("revision"),
    )
    if len(query_vectors) != len(inputs.topics) or query_vectors.shape[1] != index.d:
        raise ValueError("Cached query embeddings do not match topics/index")
    print(
        f"FAISS IndexFlatIP: {index.ntotal:,} documents, dimension={index.d}, "
        f"normalized={config.similarity == 'cosine'}, precision=float32, device=cpu; "
        f"domains={sorted({row['domain'] for row in corpus})}, "
        f"corpus_split={corpus.provenance.get('selection', {}).get('split')}, "
        f"representation={next(iter(representations))}, "
        f"query_split={queries.provenance.get('selection', {}).get('split')}",
        flush=True,
    )
    print(f"Searching {len(query_vectors):,} queries; top_k={top_k}...", flush=True)
    # Search and ranking are wholly supplied by FAISS; only row numbers are mapped to IDs.
    scores, positions = index.search(np.asarray(query_vectors), min(top_k, len(ids)))
    print("FAISS search complete; saving and evaluating rankings...", flush=True)
    rows = [
        {"qid": qid, "docno": cached_ids[int(position)], "rank": rank, "score": float(score)}
        for qid, matches, values in zip(inputs.topics.qid, positions, scores, strict=True)
        for rank, (position, score) in enumerate(zip(matches, values, strict=True))
        if position >= 0
    ]
    run = pd.DataFrame(rows, columns=["qid", "docno", "rank", "score"])
    target.mkdir(parents=True)
    metrics = evaluate_run(
        run,
        inputs.topics,
        inputs.qrels,
        inputs.query_ids,
        target,
        name="Dense",
        evaluation=settings.evaluation if settings is not None else None,
    )
    manifest = {
        "model": {**asdict(config), "resolved_revision": resolved_revision},
        "dataset_revision": corpus.provenance.get("revision"),
        "dataset_revision_status": "recorded"
        if corpus.provenance.get("revision")
        else "local/unrecorded",
        "representation": next(iter(representations)),
        "corpus_sha256": corpus_sha,
        "documents": len(ids),
        "queries": len(inputs.topics),
        "qrels": len(inputs.qrels),
        "index": {
            "library": "faiss",
            "class": "IndexFlatIP",
            "metric": "inner_product",
            "normalized_embeddings": config.similarity == "cosine",
            "dimension": index.d,
            "precision": "float32",
            "device": "cpu",
            "threads": config.faiss_threads,
            "cutoff": top_k,
        },
        "encoding": {"documents": doc_manifest["encoding"], "queries": query_manifest["encoding"]},
        "cache": {
            "documents": str(doc_cache),
            "queries": str(query_cache),
            "document_hit": doc_hit,
            "query_hit": query_hit,
        },
        "doc_ids_verified_against_index": True,
        "baseline": baseline_info,
        "versions": {
            **doc_manifest["configuration"]["versions"],
            **{name: version(name) for name in ("pyterrier", "ir-measures")},
        },
        "corpus_provenance": corpus.provenance,
        "corpus_report": corpus.report.to_dict(),
        "query_provenance": queries.provenance,
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (target / "selected-documents.json").write_text(json.dumps(ids, indent=2) + "\n")
    return metrics
