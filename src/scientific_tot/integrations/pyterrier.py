"""Thin PyTerrier data conversion and a standard Terrier BM25 experiment."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote

import pandas as pd

from scientific_tot.adapter import validate_qrels
from scientific_tot.records import Records, Report, records_digest


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


def bm25_experiment(
    corpus: Records,
    queries: Records,
    output: str | Path,
    *,
    threads: int = 1,
    terrier_version: str = "5.11",
    settings=None,
) -> pd.DataFrame:
    """Build a fresh Terrier index, retrieve, evaluate, and save TREC output.

    Resolved Hydra settings bind model/depth/metric arguments when supplied.
    Otherwise the existing top-100 tutorial defaults remain unchanged.
    Uses IterDictIndexer, Retriever(wmodel='BM25'), Experiment, and ir_measures
    metric objects. Existing output directories are refused to prevent stale
    indexes/runs. Java starts only when this function is explicitly called.
    """
    import pyterrier as pt

    if threads < 1:
        raise ValueError("threads must be positive")
    if settings is not None:
        from scientific_tot.config_arguments import bm25_arguments, require_corpus_settings
        from scientific_tot.configuration import resolve_config

        settings = resolve_config(settings, Path.cwd(), Path(output).resolve())
        require_corpus_settings(settings, corpus, queries)
        terrier_version = settings.launcher.terrier_version
        indexing_args, retrieval_args = bm25_arguments(
            settings, docno_bytes=max(len(r["doc_id"].encode("utf-8")) for r in corpus)
        )
    else:
        indexing_args = retrieval_args = None
    inputs = to_pyterrier(corpus, queries)
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(f"Choose a fresh experiment directory: {target}")
    _start_terrier(pt, terrier_version)
    target.mkdir(parents=True)
    # Terrier's metadata storage truncates at its configured capacity; compute
    # enough space for the longest actual ID and then verify every indexed ID.
    meta_length = max(len(row["docno"].encode("utf-8")) for row in inputs.documents)
    if indexing_args is None:
        indexing_args = {"meta": {"docno": meta_length}, "text_attrs": ["text"], "threads": threads}
    else:
        indexing_args["type"] = pt.terrier.IndexingType.CLASSIC
    indexer = pt.terrier.IterDictIndexer(str(target / "index"), **indexing_args)
    print(
        f"BM25 / Terrier inverted index: {len(inputs.documents):,} documents, "
        f"{len(inputs.topics):,} queries; "
        f"domains={sorted({row['domain'] for row in corpus})}, "
        f"corpus_split={corpus.provenance.get('selection', {}).get('split')}, "
        f"representation={corpus[0]['representation']}; output={target}",
        flush=True,
    )
    indexref = indexer.index(pt.tqdm(inputs.documents, desc="Indexing documents", unit="doc"))
    index = pt.terrier.IndexFactory.of(indexref)
    try:
        _verify_document_ids(index, inputs.documents)
        retrieval_args = retrieval_args or {
            "wmodel": "BM25",
            "num_results": 100,
            "threads": threads,
        }
        retriever = pt.terrier.Retriever(index, verbose=True, **retrieval_args)
        # Explicit tokenisation prevents literal '#1' text from selecting MatchOp parsing.
        topics = pt.rewrite.tokenise().transform(inputs.topics)
        run = retriever.transform(topics)
        from scientific_tot.integrations.evaluation import evaluate_run

        metrics = evaluate_run(
            run,
            inputs.topics,
            inputs.qrels,
            inputs.query_ids,
            target,
            name="BM25",
            evaluation=settings.evaluation if settings is not None else None,
        )
        manifest = {
            "documents": len(inputs.documents),
            "queries": len(inputs.topics),
            "qrels": len(inputs.qrels),
            "corpus_sha256": records_digest(inputs.documents),
            "doc_ids_verified_against_index": True,
            "model": "Terrier BM25 defaults" if settings is None else dict(retrieval_args),
            "cutoff": retrieval_args["num_results"],
            "tokeniser": "Terrier English",
            "stemmer": "PorterStemmer",
            "stopwords": "Stopwords",
            "metrics": [column for column in metrics.columns if column != "name"],
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
