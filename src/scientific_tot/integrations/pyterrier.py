"""PyTerrier data conversion and persistent BM25 indexing."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from scientific_tot.adapter import validate_qrels
from scientific_tot.records import Records, records_digest


@dataclass(frozen=True)
class TerrierInputs:
    documents: tuple[dict[str, str], ...]
    topics: pd.DataFrame
    qrels: pd.DataFrame
    query_ids: pd.DataFrame


def _start_terrier(pt, terrier_version: str) -> None:
    if not pt.java.started():
        from jdk4py import JAVA_HOME

        if os.environ.get("HF_HUB_OFFLINE") == "1":
            pt.java.mavenresolver.offline()
        pt.java.set_java_home(str(JAVA_HOME))
        pt.terrier.set_version(terrier_version)
        pt.java.init()
    elif pt.terrier.version() != terrier_version:
        raise ValueError("JVM already uses a different Terrier version; start a fresh process")


def _verify_document_ids(index, documents: tuple[dict[str, str], ...]) -> None:
    assert index.getCollectionStatistics().getNumberOfDocuments() == len(documents), (
        "Terrier document count differs from canonical corpus"
    )
    meta = index.getMetaIndex()
    indexed_ids = {str(meta.getItem("docno", i)) for i in range(len(documents))}
    assert indexed_ids == {row["docno"] for row in documents}, (
        "Terrier indexed docnos differ from canonical corpus IDs"
    )


def build_bm25_index(corpus: Records, inputs: TerrierInputs, cfg, *, reserved=False) -> dict:
    """Persist a fresh classic Terrier index without constructing a retrieval model."""
    from scientific_tot.config_arguments import bm25_arguments, require_corpus_settings
    from scientific_tot.config_artifacts import write_index_manifest

    assert cfg.stage.name == "index", "BM25 index builds require stage=index"
    require_corpus_settings(cfg, corpus)
    indexing_args, _ = bm25_arguments(
        cfg, docno_bytes=max(len(row["docno"].encode("utf-8")) for row in inputs.documents)
    )
    target = Path(cfg.paths.index)
    # Exclusive reservation also protects concurrent builders of an external destination.
    if reserved:
        from scientific_tot.config_artifacts import require_index_reservation

        require_index_reservation(target)
    else:
        target.mkdir(parents=True, exist_ok=False)
    import pyterrier as pt

    _start_terrier(pt, cfg.launcher.terrier_version)
    print(f"Building Terrier inverted index: {len(inputs.documents):,} documents", flush=True)
    indexing_args["type"] = pt.terrier.IndexingType.CLASSIC
    indexref = pt.terrier.IterDictIndexer(str(target), **indexing_args).index(
        pt.tqdm(inputs.documents, desc="Indexing documents", unit="doc")
    )
    index = pt.terrier.IndexFactory.of(indexref)
    try:
        _verify_document_ids(index, inputs.documents)
    finally:
        index.close()
    if not (target / "data.properties").is_file():
        raise FileNotFoundError(f"Terrier did not persist index properties: {target}")
    provenance = {
        "dataset_revision": corpus.provenance["revision"],
        "model_revision": None,
        "local_revision_verified": corpus.provenance["local_revision_verified"],
        "corpus_sha256": records_digest(inputs.documents),
        "corpus_provenance": corpus.provenance,
        "corpus_report": corpus.report.to_dict(),
        "documents": len(inputs.documents),
        "document_domains": {row["doc_id"]: row["domain"] for row in corpus},
        "doc_ids_verified_against_index": True,
        "versions": {name: version(name) for name in ("pyterrier", "pandas")},
        "terrier_version": pt.terrier.version(),
        "python": platform.python_version(),
        "output_dir": str(target),
        "index_location": str(target / "data.properties"),
    }
    write_index_manifest(target, cfg, provenance=provenance)
    print(
        f"Index ready: Terrier {pt.terrier.version()}; {len(inputs.documents):,} documents",
        flush=True,
    )
    return provenance


def to_pyterrier(corpus: Records, queries: Records, qrels: Records | None = None) -> TerrierInputs:
    """Convert canonical data; docno is exactly the corpus/qrel paper ID.

    qid preserves canonical query_id when globally unique. Colliding collections
    use percent-encoded query_key, recorded with original IDs in query_ids.
    Optional independently loaded qrels must match validated relevance targets.
    No tokenisation, ranking, or relevance calculation takes place here.
    """
    canonical_qrels = validate_qrels(queries, corpus)
    if qrels is not None:
        expected = {(r["query_key"], r["doc_id"], r["relevance"]) for r in canonical_qrels}
        supplied = [(r["query_key"], r["doc_id"], r["relevance"]) for r in qrels]
        if qrels.kind != "qrels" or len(supplied) != len(expected) or set(supplied) != expected:
            raise ValueError("Supplied canonical qrels disagree with validated queries/corpus")
        canonical_qrels = qrels
    documents = tuple({"docno": row["doc_id"], "text": row["text"]} for row in corpus)
    if not documents or not queries:
        raise ValueError("BM25 needs a nonempty corpus and query set")
    for doc in documents:
        if not doc["docno"] or any(c.isspace() for c in doc["docno"]):
            raise ValueError("Canonical doc_id must be a whitespace-free TREC identifier")
        if not doc["text"].strip():
            raise ValueError(f"Empty corpus text: {doc['docno']}")
    if len({d["docno"] for d in documents}) != len(documents):
        raise ValueError("Duplicate canonical document IDs")
    return _terrier_inputs(documents, queries, canonical_qrels)


def to_pyterrier_queries(corpus: Records, queries: Records) -> TerrierInputs:
    """Convert independent queries, validating positives against indexed identities."""
    if not corpus or not queries:
        raise ValueError("BM25 needs a nonempty corpus and query set")
    return _terrier_inputs((), queries, validate_qrels(queries, corpus))


def _terrier_inputs(documents, queries: Records, canonical_qrels: Records) -> TerrierInputs:
    unique = len({r["query_id"] for r in queries}) == len(queries)
    mapping = []
    topics = []
    for row in queries:
        qid = row["query_id"] if unique else quote(row["query_key"], safe="")
        if not qid or any(c.isspace() for c in qid):
            raise ValueError("Canonical query_id must be a whitespace-free TREC identifier")
        mapping.append({"qid": qid, "query_id": row["query_id"], "query_key": row["query_key"]})
        topics.append({"qid": qid, "query": row["text"]})
    by_key = {row["query_key"]: row["qid"] for row in mapping}
    if len(by_key) != len(mapping):
        raise ValueError("Duplicate canonical query keys")
    qrels = [
        {"qid": by_key[r["query_key"]], "docno": r["doc_id"], "label": r["relevance"]}
        for r in canonical_qrels
    ]
    return TerrierInputs(
        documents,
        pd.DataFrame(topics, columns=["qid", "query"]),
        pd.DataFrame(qrels, columns=["qid", "docno", "label"]),
        pd.DataFrame(mapping, columns=["qid", "query_id", "query_key"]),
    )
