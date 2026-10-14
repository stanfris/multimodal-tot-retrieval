"""Hydra dense stages using existing Sentence Transformers caches and FAISS search."""

import json
import shutil
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from urllib.parse import quote, unquote

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from scientific_tot.config_arguments import (
    configured_adapter,
    dense_arguments,
    query_arguments,
    require_corpus_settings,
)
from scientific_tot.config_artifacts import artifact_reference, verify_index, write_index_manifest
from scientific_tot.integrations.bm25_batch import _read_jsonl, save_batch
from scientific_tot.integrations.dense import (
    EncodingConfig,
    cached_encoding,
    record_digest,
    resolve_model_revision,
)
from scientific_tot.integrations.pyterrier import to_pyterrier_queries
from scientific_tot.records import Records, Report


def project_passages(corpus: Records, cfg) -> Records:
    """Use the YAML-selected LangChain splitter; canonical paper IDs remain separate."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        **OmegaConf.to_container(cfg.representation.chunking, resolve=True),
        strip_whitespace=cfg.representation.preprocessing.strip_whitespace,
        add_start_index=True,
    )
    rows = []
    for paper in corpus:
        chunks = splitter.create_documents([paper["text"]])
        if not chunks:
            raise ValueError(f"Splitter emitted no passages for selected paper: {paper['doc_id']}")
        for number, chunk in enumerate(chunks):
            rows.append(
                {
                    "passage_id": f"{quote(paper['doc_id'], safe='')}%p{number}",
                    "doc_id": paper["doc_id"],
                    "domain": paper["domain"],
                    "text": chunk.page_content,
                    "start_index": chunk.metadata["start_index"],
                }
            )
    return Records(
        "passages",
        rows,
        Report(selected=len(rows), emitted=len(rows)),
        {
            **corpus.provenance,
            "representation": OmegaConf.to_container(cfg.representation, resolve=True),
        },
    )


def build_dense_index(corpus: Records, inputs, cfg, *, reserved=False) -> dict:
    """Encode documents only, reuse verified cached vectors/index, and publish a fresh index."""
    import faiss

    require_corpus_settings(cfg, corpus)
    representation = OmegaConf.to_container(cfg.representation, resolve=True)
    representation.pop("aggregation", None)  # Aggregation affects ranking, not document embeddings.
    if cfg.representation.name == "passages":
        passages = project_passages(corpus, cfg)
        documents = [{"docno": row["passage_id"], "text": row["text"]} for row in passages]
        mapping = [
            {"vector_id": row["passage_id"], "doc_id": row["doc_id"], "domain": row["domain"]}
            for row in passages
        ]
    else:
        documents = list(inputs.documents)
        mapping = [
            {"vector_id": row["doc_id"], "doc_id": row["doc_id"], "domain": row["domain"]}
            for row in corpus
        ]
    config = EncodingConfig(**dense_arguments(cfg))
    revision = resolve_model_revision(config)
    target = Path(cfg.paths.index)
    if reserved:
        from scientific_tot.config_artifacts import require_index_reservation

        require_index_reservation(target)
    else:
        target.mkdir(parents=True, exist_ok=False)
    vectors, cached, entry, hit = cached_encoding(
        config,
        revision,
        Path(cfg.launcher.cache_root) / "embeddings",
        task="document",
        rows=documents,
        dataset_revision=corpus.provenance["revision"],
        representation=representation,
    )
    index = faiss.read_index(str(entry / "index.faiss"))
    ids = json.loads((entry / "ids.json").read_text())
    if (
        ids != [row["docno"] for row in documents]
        or index.ntotal != len(mapping)
        or index.d != vectors.shape[1]
    ):
        raise ValueError("Cached FAISS index/ID mapping does not match selected corpus")
    for name in ("index.faiss", "ids.json"):
        shutil.copyfile(entry / name, target / name)
    Records(
        "vector_mapping",
        mapping,
        Report(selected=len(mapping), emitted=len(mapping)),
        corpus.provenance,
    ).write_jsonl(target / "vectors.jsonl")
    files = {
        name: artifact_reference(target / name)
        for name in ("index.faiss", "ids.json", "vectors.jsonl", "vectors.jsonl.manifest.json")
    }
    for name in files:
        (target / name).chmod(0o444)
    provenance = {
        "dataset_revision": corpus.provenance["revision"],
        "model_revision": revision,
        "local_revision_verified": corpus.provenance["local_revision_verified"],
        "corpus_sha256": record_digest(inputs.documents),
        "vector_input_sha256": record_digest(documents),
        "corpus_provenance": corpus.provenance,
        "documents": len(corpus),
        "vectors": len(mapping),
        "dimension": index.d,
        "document_domains": {row["doc_id"]: row["domain"] for row in corpus},
        "index_files": files,
        "index_location": str(target / "index.faiss"),
        "output_dir": str(target),
        "index_device": "cpu",
        "encoding_config": {**asdict(config), "resolved_revision": revision},
        "encoding": cached["encoding"],
        "versions": cached["configuration"]["versions"],
        "cache": {"documents": str(entry), "document_hit": hit},
        "aggregation": OmegaConf.to_container(cfg.representation.get("aggregation"), resolve=True)
        if cfg.representation.name == "passages"
        else None,
    }
    if cfg.representation.name == "passages":
        provenance["versions"]["langchain-text-splitters"] = version("langchain-text-splitters")
    write_index_manifest(target, cfg, provenance=provenance)
    return provenance


def retrieve_dense(cfg, target: Path) -> dict:
    """Encode queries independently and search the saved FAISS index at paper level."""
    import faiss

    location = Path(cfg.paths.index)
    manifest = verify_index(location, cfg)
    provenance = manifest["provenance"]
    names = ("index.faiss", "ids.json", "vectors.jsonl", "vectors.jsonl.manifest.json")
    for name in names:
        if provenance.get("index_files", {}).get(name) != artifact_reference(location / name):
            raise ValueError(f"Incomplete/corrupt persistent dense index: {name}")
    mapping = _read_jsonl(location / "vectors.jsonl")
    if (
        set(mapping.columns) != {"vector_id", "doc_id", "domain"}
        or mapping.vector_id.duplicated().any()
    ):
        raise ValueError("Ambiguous persistent vector mapping")
    ids = json.loads((location / "ids.json").read_text())
    if ids != mapping.vector_id.tolist():
        raise ValueError("Persistent FAISS IDs disagree with vector mapping")
    index = faiss.read_index(str(location / "index.faiss"))
    if index.ntotal != len(mapping) or index.d != provenance["dimension"]:
        raise ValueError("Persistent FAISS dimension/count differs from index manifest")
    domains = provenance["document_domains"]
    if (
        set(mapping.doc_id) != set(domains)
        or len(domains) != provenance["documents"]
        or any(row.domain != domains[row.doc_id] for row in mapping.itertuples())
    ):
        raise ValueError("Vector mapping differs from canonical paper identities/domains")
    corpus = Records(
        "corpus",
        [{"doc_id": doc_id, "domain": domain} for doc_id, domain in domains.items()],
        Report(emitted=len(domains)),
        provenance["corpus_provenance"],
    )
    adapter = configured_adapter(cfg)
    queries = adapter.load_queries(**query_arguments(cfg))
    inputs = to_pyterrier_queries(corpus, queries)
    config = EncodingConfig(**dense_arguments(cfg))
    vectors, cached, entry, hit = cached_encoding(
        config,
        provenance["model_revision"],
        Path(cfg.launcher.cache_root) / "embeddings",
        task="query",
        rows=inputs.topics.to_dict("records"),
        dataset_revision=queries.provenance["revision"],
    )
    if vectors.shape[1] != index.d:
        raise ValueError("Query embedding dimension differs from the saved FAISS index")
    passage_mode = cfg.representation.name == "passages"
    depth = len(ids) if passage_mode else min(cfg.retrieval.top_k, len(ids))
    batches = []
    for qid, vector in zip(inputs.topics.qid, vectors, strict=True):
        # Scoring/ranking belong to FAISS. All passages are considered before aggregation,
        # so top_k always counts papers and mean/max have their full-corpus semantics.
        scores, positions = index.search(np.asarray([vector]), depth)
        rows = [
            {"qid": qid, "docno": ids[int(position)], "rank": rank, "score": float(score)}
            for rank, (position, score) in enumerate(zip(positions[0], scores[0], strict=True))
            if position >= 0
        ]
        result = pd.DataFrame(rows, columns=["qid", "docno", "rank", "score"])
        if passage_mode:
            import pyterrier as pt

            method = cfg.representation.aggregation.method
            aggregator = pt.text.max_passage() if method == "max" else pt.text.mean_passage()
            result = aggregator.transform(result)
            result["docno"] = result.docno.map(unquote)
            result = result[result["rank"] < cfg.retrieval.top_k]
        batches.append(result)
    run = pd.concat(batches, ignore_index=True)
    run["rank"] += 1
    query_mapping = inputs.query_ids.copy()
    query_mapping["domain"] = query_mapping.query_key.map(
        {r["query_key"]: r["domain"] for r in queries}
    )
    identities = pd.DataFrame(list(corpus), columns=["doc_id", "domain"])
    outputs = save_batch(run, inputs, query_mapping, identities, cfg, target)
    return {
        "inputs": {
            "index_manifest": artifact_reference(location / "index-manifest.json"),
            "index_files": provenance["index_files"],
            "query_files": [
                {"dataset_path": name, **artifact_reference(adapter.dataset.get_file(name))}
                for name in queries.provenance["selection"]["files"]
            ],
            "query_cache_manifest": artifact_reference(entry / "manifest.json"),
        },
        "outputs": outputs,
        "queries": len(inputs.topics),
        "qrels": len(inputs.qrels),
        "rankings": len(run),
        "query_provenance": queries.provenance,
        "model_revision": provenance["model_revision"],
        "dataset_revision": provenance["dataset_revision"],
        "encoding_config": asdict(config),
        "encoding": cached["encoding"],
        "cache": {"queries": str(entry), "query_hit": hit},
        "retrieval_cutoff": cfg.retrieval.top_k,
        "vector_search_depth": depth,
        "aggregation": {
            "method": cfg.representation.aggregation.method,
            "library": f"pyterrier.text.{cfg.representation.aggregation.method}_passage",
            "scope": "all_indexed_passages",
        }
        if passage_mode
        else None,
        "ranking_level": "paper",
        "versions": cached["configuration"]["versions"],
    }
