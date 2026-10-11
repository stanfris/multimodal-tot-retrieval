"""Thin PyTerrier data conversion and a standard Terrier BM25 experiment."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from scientific_tot.adapter import validate_qrels
from scientific_tot.records import Records, Report


@dataclass(frozen=True)
class TerrierInputs:
    documents: tuple[dict[str, str], ...]
    topics: pd.DataFrame
    qrels: pd.DataFrame
    query_ids: pd.DataFrame


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


def bm25_experiment(
    corpus: Records,
    queries: Records,
    output: str | Path,
    *,
    threads: int = 1,
    terrier_version: str = "5.11",
) -> pd.DataFrame:
    """Build a fresh Terrier index, retrieve top 100, evaluate, and save TREC output.

    Uses IterDictIndexer, Retriever(wmodel='BM25'), Experiment, and ir_measures
    metric objects. Existing output directories are refused to prevent stale
    indexes/runs. Java starts only when this function is explicitly called.
    """
    import pyterrier as pt

    if threads < 1:
        raise ValueError("threads must be positive")
    inputs = to_pyterrier(corpus, queries)
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(f"Choose a fresh experiment directory: {target}")
    target.mkdir(parents=True)
    if not pt.java.started():
        pt.terrier.set_version(terrier_version)
        pt.java.init()
    elif pt.terrier.version() != terrier_version:
        raise ValueError("JVM already uses a different Terrier version; start a fresh process")
    # Terrier's metadata storage truncates at its configured capacity; compute
    # enough space for the longest actual ID and then verify every indexed ID.
    meta_length = max(len(row["docno"].encode("utf-8")) for row in inputs.documents)
    indexer = pt.terrier.IterDictIndexer(
        str(target / "index"), meta={"docno": meta_length}, text_attrs=["text"], threads=threads
    )
    indexref = indexer.index(iter(inputs.documents))
    index = pt.terrier.IndexFactory.of(indexref)
    try:
        if index.getCollectionStatistics().getNumberOfDocuments() != len(inputs.documents):
            raise RuntimeError("Terrier document count differs from canonical corpus")
        meta = index.getMetaIndex()
        indexed_ids = {str(meta.getItem("docno", i)) for i in range(len(inputs.documents))}
        if indexed_ids != {row["docno"] for row in inputs.documents}:
            raise RuntimeError("Terrier indexed docnos differ from canonical corpus IDs")
        # The standard Terrier tokeniser handles natural-language query syntax.
        retriever = pt.terrier.Retriever(index, wmodel="BM25", num_results=100, threads=threads)
        run = retriever.transform(inputs.topics)
        from scientific_tot.integrations.evaluation import evaluate_run

        metrics = evaluate_run(
            run, inputs.topics, inputs.qrels, inputs.query_ids, target, name="BM25"
        )
        digest = hashlib.sha256()
        for doc in inputs.documents:
            digest.update(json.dumps(doc, ensure_ascii=False, sort_keys=True).encode())
            digest.update(b"\n")
        manifest = {
            "documents": len(inputs.documents),
            "queries": len(inputs.topics),
            "qrels": len(inputs.qrels),
            "corpus_sha256": digest.hexdigest(),
            "doc_ids_verified_against_index": True,
            "model": "Terrier BM25 defaults",
            "cutoff": 100,
            "tokeniser": "Terrier English",
            "stemmer": "PorterStemmer",
            "stopwords": "Stopwords",
            "metrics": ["RR@10 (MRR@10)", "R@10", "R@100"],
            "versions": {name: version(name) for name in ["pyterrier", "pandas", "ir-measures"]},
            "terrier_version": pt.terrier.version(),
            "python": platform.python_version(),
            "corpus_provenance": corpus.provenance,
            "corpus_report": corpus.report.to_dict(),
            "query_provenance": queries.provenance,
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    finally:
        index.close()
    return metrics


def select_demo_subset(
    corpus: Records, queries: Records, documents: int | None, query_limit: int | None
) -> tuple[Records, Records]:
    """Deterministic demo selection retaining every selected query's positive targets."""
    if documents is not None and documents < 1 or query_limit is not None and query_limit < 1:
        raise ValueError("Subset limits must be positive")
    selected_queries = list(queries)[:query_limit]
    positives = {doc for row in selected_queries for doc in row["relevant_ids"]}
    if documents is not None and len(positives) > documents:
        raise ValueError("--documents is smaller than the selected queries' positive corpus")
    ids = set(positives)
    if documents is None:
        ids.update(row["doc_id"] for row in corpus)
    else:
        for row in sorted(corpus, key=lambda row: row["doc_id"]):
            if len(ids) >= documents:
                break
            ids.add(row["doc_id"])
    corpus_rows = [row for row in corpus if row["doc_id"] in ids]
    return (
        Records(
            "corpus",
            corpus_rows,
            Report(selected=len(corpus_rows), emitted=len(corpus_rows)),
            {
                **corpus.provenance,
                "parent_report": corpus.report.to_dict(),
                "subset": {"documents_requested": documents, "policy": "positives_then_sorted_ids"},
            },
        ),
        Records(
            "queries",
            selected_queries,
            Report(selected=len(selected_queries), emitted=len(selected_queries)),
            {
                **queries.provenance,
                "subset": {
                    "queries_requested": query_limit,
                    "policy": "canonical_collection_order",
                },
            },
        ),
    )
