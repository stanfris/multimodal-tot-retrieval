"""Original PDF pages, official ColQwen encoding, and Qdrant exact multi-vector search."""

import json
import pickle
import shutil
import sqlite3
import tempfile
from contextlib import closing
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from scientific_tot.adapter import validate_qrels
from scientific_tot.config_arguments import configured_adapter, corpus_arguments, query_arguments
from scientific_tot.config_artifacts import (
    artifact_reference,
    require_index_reservation,
    save_json,
    verify_index,
    write_index_manifest,
)
from scientific_tot.data.validation import DatasetFormatError
from scientific_tot.integrations.bm25_batch import save_batch
from scientific_tot.integrations.pyterrier import to_pyterrier_queries
from scientific_tot.records import IncompleteCorpusError, Issue, Records, Report, records_digest


def dependency_versions() -> dict:
    return {
        name: version(name)
        for name in (
            "sentence-transformers",
            "transformers",
            "peft",
            "torch",
            "torchvision",
            "pillow",
            "pypdfium2",
            "qdrant-client",
            "numpy",
            "tokenizers",
            "huggingface-hub",
            "accelerate",
        )
    }


def page_id(doc_id: str, number: int) -> str:
    """Escape the PyTerrier passage delimiter; canonical IDs never become paths."""
    assert number > 0
    return f"{quote(doc_id, safe='')}%p{number}"


def render_page(pdf, number: int, cfg):
    """PDFium owns rasterization; PIL images are passed unchanged to the processor."""
    with closing(pdf[number - 1]) as page:
        with closing(page.render(scale=cfg.representation.rendering.scale)) as bitmap:
            return bitmap.to_pil().copy()


def load_pdf_corpus(adapter, cfg) -> Records:
    """Resolve original PDFs, fingerprint their bytes, and read page counts."""
    import pypdfium2 as pdfium
    from tqdm.auto import tqdm

    selected = adapter.load_corpus(**corpus_arguments(cfg))
    report = Report(
        selected=selected.report.selected,
        skipped=selected.report.skipped,
        issues=list(selected.report.issues),
    )
    rows = []
    for row in tqdm(selected, desc="Reading original PDFs", unit="paper"):
        try:
            path = adapter.resolve_pdf(row)
            reference = artifact_reference(path)
            with pdfium.PdfDocument(path) as pdf:
                if not len(pdf):
                    raise ValueError("PDF contains no pages")
                count = len(pdf)
        except DatasetFormatError:
            raise
        except (OSError, pdfium.PdfiumError, ValueError) as exc:
            report.skipped += 1
            report.issues.append(
                Issue("unusable_pdf", row["source_pdf_relpath"] or "", str(exc), row["doc_id"])
            )
            continue
        rows.append(
            {
                **row,
                "pdf_path": reference["path"],
                "pdf_sha256": reference["sha256"],
                "page_count": count,
            }
        )
    report.emitted = len(rows)
    if report.skipped and cfg.dataset.missing == "error":
        raise IncompleteCorpusError(report)
    if not rows:
        raise ValueError("No usable original PDFs selected")
    return Records("corpus", rows, report, selected.provenance)


def pdf_corpus_digest(corpus) -> str:
    return records_digest(
        {
            key: row[key]
            for key in ("doc_id", "domain", "source_pdf_relpath", "pdf_sha256", "page_count")
        }
        for row in corpus
    )


def page_mapping(corpus) -> list[dict]:
    return [
        {
            "page_id": page_id(row["doc_id"], number),
            "doc_id": row["doc_id"],
            "domain": row["domain"],
            "page_number": number,
        }
        for row in corpus
        for number in range(1, row["page_count"] + 1)
    ]


def load_model(cfg):
    """Load the official adapter/processor and pin its separately published base weights.

    ST/Transformers discard the adapter revision when loading its base. A temporary
    local adapter view points to an immutable base snapshot; no preprocessing or
    learned weights are edited. ST owns all module construction and inference.
    """
    import torch
    from huggingface_hub import snapshot_download

    torch.set_num_threads(cfg.launcher.threads.encoding)
    if not torch.cuda.is_available():
        raise RuntimeError("ColQwen CUDA execution requires an available NVIDIA GPU")
    from sentence_transformers import MultiVectorEncoder

    cache = Path(cfg.launcher.cache_root) / "models"
    cache.mkdir(parents=True, exist_ok=True)
    adapter = Path(
        snapshot_download(cfg.model.model_id, revision=cfg.model.revision, cache_dir=str(cache))
    )
    metadata = json.loads((adapter / "config_sentence_transformers.json").read_text())
    if metadata["model_type"] != "MultiVectorEncoder" or not (adapter / "modules.json").is_file():
        raise ValueError("Checkpoint lacks its official MultiVectorEncoder conversion")
    base = snapshot_download(
        cfg.model.base_model_id, revision=cfg.model.base_revision, cache_dir=str(cache)
    )
    with tempfile.TemporaryDirectory(dir=cache, prefix="colqwen-") as folder:
        view = Path(folder)
        for source in adapter.rglob("*"):
            relative = source.relative_to(adapter)
            if source.is_dir():
                (view / relative).mkdir(exist_ok=True)
            elif relative.as_posix() != "adapter_config.json":
                (view / relative).symlink_to(source.resolve())
        config = json.loads((adapter / "adapter_config.json").read_text())
        if config["base_model_name_or_path"] != cfg.model.base_model_id:
            raise ValueError("Checkpoint adapter base differs from configured base")
        config["base_model_name_or_path"] = base
        (view / "adapter_config.json").write_text(json.dumps(config))
        assert cfg.model.processor == "checkpoint"
        model = MultiVectorEncoder(str(view), device=cfg.model.device, local_files_only=True)
    if model.similarity_fn_name != "maxsim":
        raise ValueError("Checkpoint similarity must match Qdrant MAX_SIM")
    processor = model[0].processor
    effective = {
        "module_config": model[0].get_config_dict(),
        "chat_template": processor.chat_template,
        "processor": processor.to_dict(),
        "image_processor": processor.image_processor.to_dict(),
        "model_config": model[0].model.config.to_dict(),
        "prompts": model.prompts,
        "similarity": model.similarity_fn_name,
        "checkpoint_files": {
            str(path.relative_to(adapter)): artifact_reference(path)["sha256"]
            for path in adapter.rglob("*")
            if path.is_file()
        },
        "base_revision": cfg.model.base_revision,
    }
    return model, effective


def validate_points(
    database: Path, mapping: list[dict], dimension: int, *, allow_partial=False
) -> int:
    """Read one persisted point at a time; never instantiate an in-memory Qdrant index."""
    from qdrant_client.local.persistence import CollectionPersistence
    from tqdm.auto import tqdm

    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        count = db.execute("SELECT COUNT(*) FROM points").fetchone()[0]
        print(f"Stored pages: {count:,}; expected pages: {len(mapping):,}", flush=True)
        if count > len(mapping) or (not allow_partial and count != len(mapping)):
            raise ValueError("Stored page count differs from corpus; cannot finalize this index")
        seen = bytearray(count)
        for key, blob in tqdm(
            db.execute("SELECT id, point FROM points"), total=count, desc="Validating stored pages"
        ):
            point = pickle.loads(blob)
            position = point.id
            if type(position) is not int or not 0 <= position < count or seen[position]:
                raise ValueError(f"Invalid or duplicate page position: {position}")
            if (
                key != CollectionPersistence.encode_key(position)
                or point.payload != mapping[position]
            ):
                raise ValueError(f"Stored page identity/payload differs at position {position}")
            vectors = np.asarray(point.vector, dtype=np.float32)
            if (
                vectors.ndim != 2
                or vectors.shape[0] == 0
                or vectors.shape[1] != dimension
                or not np.isfinite(vectors).all()
            ):
                raise ValueError(f"Invalid page vectors at position {position}")
            seen[position] = 1
        return count


def validate_collection(index: Path, dimension: int) -> None:
    metadata = json.loads((index / "qdrant/meta.json").read_text())
    vectors = metadata["collections"]["pages"]["vectors"]
    if (
        vectors["size"] != dimension
        or vectors["distance"] != "Dot"
        or vectors["multivector_config"]["comparator"] != "max_sim"
        or vectors["hnsw_config"]["m"] != 0
    ):
        raise ValueError("Stored collection differs from the expected exact DOT/MAX_SIM index")


def build_index(corpus, cfg, *, reserved=False, existing=None) -> dict:
    """Persist pages incrementally; existing='resume' appends to a validated prefix."""
    import portalocker
    from qdrant_client import QdrantClient, models

    assert existing in {None, "resume", "finalize"}
    target = Path(cfg.paths.index)
    if existing is None:
        if reserved:
            require_index_reservation(target)
        else:
            target.mkdir(parents=True, exist_ok=False)
    print("Loading pinned ColQwen model...", flush=True)
    model, effective = load_model(cfg)
    dimension = model.get_embedding_dimension()
    mapping = page_mapping(corpus)
    if not (target / "qdrant").exists():
        if existing == "finalize":
            raise ValueError("No persisted Qdrant collection to finalize")
        # Create metadata for an empty collection; never reopen a populated local client.
        with closing(QdrantClient(path=str(target / "qdrant"))) as client:
            client.create_collection(
                "pages",
                vectors_config=models.VectorParams(
                    size=dimension,
                    distance=models.Distance(cfg.model.indexing.distance.title()),
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator(cfg.model.indexing.comparator)
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                ),
            )
    with portalocker.Lock(target / "qdrant/.lock", mode="r+", timeout=0):
        validate_collection(target, dimension)
        position = (
            validate_points(
                target / "qdrant/collection/pages/storage.sqlite",
                mapping,
                dimension,
                allow_partial=existing == "resume",
            )
            if existing is not None
            else 0
        )
        if existing == "resume":
            print(f"Resuming index: {position:,}/{len(mapping):,} pages already stored", flush=True)
        if position < len(mapping):
            persist_pages(corpus, cfg, model, mapping, position)
        return publish_index_metadata(corpus, cfg, effective)


def persist_pages(corpus, cfg, model, mapping, resume_from: int) -> None:
    """Append new PointStructs through Qdrant's persistence API without a full RAM index."""
    import pypdfium2 as pdfium
    from qdrant_client import models
    from qdrant_client.local.persistence import CollectionPersistence
    from tqdm.auto import tqdm

    target = Path(cfg.paths.index)
    with closing(CollectionPersistence(str(target / "qdrant/collection/pages"))) as store:
        position = 0
        with tqdm(
            total=len(mapping), initial=resume_from, desc="Encoding PDF pages", unit="page"
        ) as bar:
            for row in corpus:
                end = position + row["page_count"]
                if end <= resume_from:
                    position = end
                    continue
                with pdfium.PdfDocument(row["pdf_path"]) as pdf:
                    first = max(1, resume_from - position + 1)
                    for start in range(first, row["page_count"] + 1, cfg.model.batch_size):
                        images = [
                            render_page(pdf, number, cfg)
                            for number in range(
                                start, min(start + cfg.model.batch_size, row["page_count"] + 1)
                            )
                        ]
                        try:
                            vectors = model.encode_document(
                                images, batch_size=cfg.model.batch_size, convert_to_numpy=True
                            )
                        finally:
                            for image in images:
                                image.close()
                        assert len(vectors) == len(images)
                        for i, vector in enumerate(vectors):
                            offset = position + start - 1 + i
                            store.persist(
                                models.PointStruct(
                                    id=offset, vector=vector.tolist(), payload=mapping[offset]
                                )
                            )
                            bar.update(1)
                position = end
        count = store.storage.execute("SELECT COUNT(*) FROM points").fetchone()[0]
        if count != len(mapping):
            raise ValueError("Persistent page count differs from mapping")


def publish_index_metadata(corpus, cfg, effective) -> dict:
    """Publish metadata after either building or validating the persisted pages."""
    print("Writing page metadata and checksumming persistent index files...", flush=True)
    target = Path(cfg.paths.index)
    mapping = page_mapping(corpus)
    Records("page_mapping", mapping, Report(emitted=len(mapping)), corpus.provenance).write_jsonl(
        target / "page-mapping.jsonl"
    )
    provenance = {
        "dataset_revision": corpus.provenance["revision"],
        "model_revision": cfg.model.revision,
        "corpus_sha256": pdf_corpus_digest(corpus),
        "corpus_provenance": corpus.provenance,
        "local_revision_verified": corpus.provenance["local_revision_verified"],
        "documents": len(corpus),
        "pages": len(mapping),
        "document_domains": {row["doc_id"]: row["domain"] for row in corpus},
        "corpus_report": corpus.report.to_dict(),
        "rendering": OmegaConf.to_container(cfg.representation.rendering, resolve=True),
        "effective_model": effective,
        "versions": dependency_versions(),
    }
    write_index_manifest(target, cfg, provenance=provenance)
    return provenance


def aggregate_pages(pages: pd.DataFrame, cutoff: int) -> pd.DataFrame:
    """Benchmark-specific maximum page score; library aggregation then stable paper ties."""
    import pyterrier as pt

    result = pt.text.max_passage().transform(pages)
    result["docno"] = result.docno.map(unquote)
    result = result.sort_values(["score", "docno"], ascending=[False, True]).head(cutoff).copy()
    result["rank"] = range(1, len(result) + 1)
    return result


def retrieve(cfg, target: Path) -> dict:
    from qdrant_client import QdrantClient, models
    from tqdm.auto import tqdm

    location = Path(cfg.paths.index)
    manifest = verify_index(location, cfg)
    provenance = manifest["provenance"]
    mapping = [
        json.loads(line) for line in (location / "page-mapping.jsonl").read_text().splitlines()
    ]
    domains = provenance["document_domains"]
    if (
        len(mapping) != provenance["pages"]
        or len(domains) != provenance["documents"]
        or any(
            row["page_id"] != page_id(row["doc_id"], row["page_number"])
            or domains.get(row["doc_id"]) != row["domain"]
            for row in mapping
        )
        or len({row["page_id"] for row in mapping}) != len(mapping)
    ):
        raise ValueError("Invalid canonical page-to-paper mapping")
    corpus = Records(
        "corpus",
        [{"doc_id": key, "domain": value} for key, value in domains.items()],
        Report(emitted=len(domains)),
        provenance["corpus_provenance"],
    )
    adapter = configured_adapter(cfg)
    queries = adapter.load_queries(**query_arguments(cfg))
    inputs = to_pyterrier_queries(corpus, queries)
    model, effective = load_model(cfg)
    # Temporary model paths differ; learned/processor bytes must match the pinned checkpoint.
    if effective["checkpoint_files"] != provenance["effective_model"]["checkpoint_files"]:
        raise ValueError("Checkpoint bytes differ from indexed model")
    batches = []
    page_score_count = 0
    # Local Qdrant opens writable SQLite/lock files. Search a disposable copy to
    # keep the published artifact immutable and allow independent concurrent runs.
    with (
        tempfile.TemporaryDirectory(prefix="colqwen-search-") as folder,
        (target / "page-scores.jsonl").open("x", encoding="utf-8") as page_scores,
    ):
        work = Path(folder) / "qdrant"
        shutil.copytree(location / "qdrant", work)
        for path in work.rglob("*"):
            if path.is_file():
                path.chmod(0o600)
        client = QdrantClient(path=str(work))
        try:
            if client.count("pages", exact=True).count != len(mapping):
                raise ValueError("Qdrant page count differs from saved mapping")
            size = cfg.model.retrieval.batch_size
            topics = list(inputs.topics.itertuples())
            for start in tqdm(range(0, len(topics), size), desc="Retrieving queries", unit="batch"):
                chunk = topics[start : start + size]
                vectors = model.encode_query(
                    [topic.query for topic in chunk], batch_size=size, convert_to_numpy=True
                )
                assert len(vectors) == len(chunk)
                for topic, vector in zip(chunk, vectors, strict=True):
                    hits = client.query_points(
                        "pages",
                        query=vector.tolist(),
                        limit=len(mapping),
                        search_params=models.SearchParams(exact=cfg.model.indexing.exact),
                        with_payload=True,
                    ).points
                    if len(hits) != len(mapping) or len({hit.id for hit in hits}) != len(mapping):
                        raise ValueError("Exact page search did not exhaust the indexed collection")
                    rows = []
                    for hit in hits:
                        if (
                            type(hit.id) is not int
                            or not 0 <= hit.id < len(mapping)
                            or hit.payload != mapping[hit.id]
                        ):
                            raise ValueError("Qdrant result differs from canonical mapping")
                        rows.append(
                            {"qid": topic.qid, "docno": hit.payload["page_id"], "score": hit.score}
                        )
                        page_scores.write(
                            json.dumps(
                                {"qid": topic.qid, **hit.payload, "score": hit.score},
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                            + "\n"
                        )
                        page_score_count += 1
                    batches.append(aggregate_pages(pd.DataFrame(rows), cfg.retrieval.top_k))
        finally:
            client.close()
    run = pd.concat(batches, ignore_index=True)
    query_mapping = inputs.query_ids.copy()
    query_mapping["domain"] = query_mapping.query_key.map(
        {row["query_key"]: row["domain"] for row in queries}
    )
    outputs = save_batch(run, inputs, query_mapping, pd.DataFrame(list(corpus)), cfg, target)
    save_json(
        target / "page-scores.jsonl.manifest.json",
        {
            "schema_version": 1,
            "kind": "page_scores",
            "count": page_score_count,
            "artifact": artifact_reference(target / "page-scores.jsonl"),
            "provenance": queries.provenance,
        },
    )
    outputs["page-scores.jsonl"] = artifact_reference(target / "page-scores.jsonl")
    return {
        "inputs": {
            "index_manifest": artifact_reference(location / "index-manifest.json"),
            "query_files": [
                {"dataset_path": name, **artifact_reference(adapter.dataset.get_file(name))}
                for name in queries.provenance["selection"]["files"]
            ],
        },
        "outputs": outputs,
        "queries": len(queries),
        "qrels": len(inputs.qrels),
        "rankings": len(run),
        "query_provenance": queries.provenance,
        "retrieval_cutoff": cfg.retrieval.top_k,
        "page_search_depth": len(mapping),
        "ranking_level": "paper",
        "aggregation": {
            "method": "max",
            "library": "pyterrier.text.max_passage",
            "scope": "all_indexed_pages",
            "exact": True,
            "tie_break": "canonical_doc_id_ascending",
        },
        "versions": dependency_versions(),
    }


def execute_visual_stage(cfg, run_dir) -> dict:
    if cfg.stage.name == "retrieve":
        return retrieve(cfg, run_dir)
    adapter = configured_adapter(cfg)
    corpus = load_pdf_corpus(adapter, cfg)
    if cfg.stage.name == "index":
        return {
            "index_provenance": build_index(corpus, cfg, reserved=Path(cfg.paths.index) == run_dir)
        }
    assert cfg.stage.name == "prepare"
    queries = adapter.load_queries(**query_arguments(cfg))
    qrels = validate_qrels(queries, corpus)
    for records in (corpus, queries, qrels):
        records.write_jsonl(run_dir / f"{records.kind}.jsonl")
    mapping = page_mapping(corpus)
    Records("page_mapping", mapping, Report(emitted=len(mapping)), corpus.provenance).write_jsonl(
        run_dir / "page-mapping.jsonl"
    )
    return {
        "documents": len(corpus),
        "pages": len(mapping),
        "queries": len(queries),
        "qrels": len(qrels),
        "corpus_provenance": corpus.provenance,
        "query_provenance": queries.provenance,
        "corpus_report": corpus.report.to_dict(),
        "rendering": OmegaConf.to_container(cfg.representation.rendering, resolve=True),
    }
