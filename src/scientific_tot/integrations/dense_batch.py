"""Hydra dense stages using existing Sentence Transformers caches and FAISS search."""

import json
import shutil
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import pandas as pd
from omegaconf import OmegaConf

from scientific_tot.config_arguments import (
    configured_adapter,
    dense_arguments,
    query_arguments,
    require_corpus_settings,
)
from scientific_tot.config_artifacts import artifact_reference, verify_index, write_index_manifest
from scientific_tot.configuration import OPENCLIP_MODELS, VISUAL_DENSE_MODELS
from scientific_tot.integrations.bm25_batch import _read_jsonl, save_batch
from scientific_tot.integrations.dense import (
    EncodingConfig,
    cached_encoding,
    resolve_model_revision,
)
from scientific_tot.integrations.faiss_search import FaissSearch
from scientific_tot.integrations.pyterrier import to_pyterrier_queries
from scientific_tot.records import Records, Report, records_digest
from scientific_tot.reporting import phase


def load_page_text_corpus(adapter, cfg) -> Records:
    """Extract text within original PDF page boundaries using the existing PDF loader."""
    import pypdfium2 as pdfium

    from scientific_tot.integrations.colqwen import load_pdf_corpus, page_id
    from scientific_tot.reporting import tqdm

    corpus = load_pdf_corpus(adapter, cfg)

    def papers():
        for paper in tqdm(corpus, desc="Extracting PDF page text", unit="paper"):
            pages = []
            with pdfium.PdfDocument(paper["pdf_path"]) as pdf:
                for number in range(1, len(pdf) + 1):
                    with closing(pdf[number - 1]) as page, closing(page.get_textpage()) as textpage:
                        text = textpage.get_text_bounded().strip()
                    if not text:
                        raise ValueError(f"{paper['doc_id']}: PDF page {number} contains no text")
                    pages.append(
                        {
                            "page_id": page_id(paper["doc_id"], number),
                            "doc_id": paper["doc_id"],
                            "domain": paper["domain"],
                            "page_number": number,
                            "text": text,
                        }
                    )
            yield {**paper, "text": "\n\n".join(p["text"] for p in pages), "pages": pages}

    return Records("corpus", papers(), corpus.report, corpus.provenance)


def build_dense_index(corpus: Records, inputs, cfg, *, reserved=False) -> dict:
    """Encode documents only, reuse verified cached vectors/index, and publish a fresh index."""
    import faiss

    require_corpus_settings(cfg, corpus)
    representation = OmegaConf.to_container(cfg.representation, resolve=True)
    representation.pop("aggregation", None)  # Aggregation affects ranking, not document embeddings.
    visual = cfg.model.name in VISUAL_DENSE_MODELS
    if visual:
        from scientific_tot.integrations.visual_dense import document_mapping

        documents = document_mapping(corpus, cfg)
        mapping = [
            {"vector_id": row["docno"], "doc_id": row["doc_id"], "domain": row["domain"]}
            for row in documents
        ]
    elif cfg.representation.name == "page_text":
        pages = [page for paper in corpus for page in paper["pages"]]
        documents = [{"docno": row["page_id"], "text": row["text"]} for row in pages]
        mapping = [
            {"vector_id": row["page_id"], "doc_id": row["doc_id"], "domain": row["domain"]}
            for row in pages
        ]
    else:
        documents = list(inputs.documents)
        mapping = [
            {"vector_id": row["doc_id"], "doc_id": row["doc_id"], "domain": row["domain"]}
            for row in corpus
        ]
    config = None if visual else EncodingConfig(**dense_arguments(cfg))
    revision = cfg.model.revision if visual else resolve_model_revision(config)
    print(
        f"Building FAISS IndexFlatIP: {len(corpus):,} papers, {len(documents):,} vectors; "
        f"model={cfg.model.model_id}@{revision}, "
        f"encoding_device={cfg.model.device}, index_device=cpu",
        flush=True,
    )
    target = Path(cfg.paths.index)
    if reserved:
        from scientific_tot.config_artifacts import require_index_reservation

        require_index_reservation(target)
    else:
        target.mkdir(parents=True, exist_ok=False)
    if visual:
        from scientific_tot.integrations.visual_dense import cached_encoding as visual_encoding

        vectors, cached, entry, hit = visual_encoding(
            cfg,
            task="document",
            rows=documents,
            dataset_revision=corpus.provenance["revision"],
        )
    else:
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
    if visual:
        from scientific_tot.integrations.visual_dense import visual_corpus_digest

        corpus_sha256 = visual_corpus_digest(corpus, cfg)
        Records(
            "unit_mapping", documents, Report(emitted=len(documents)), corpus.provenance
        ).write_jsonl(target / "unit-mapping.jsonl")
        for name in ("unit-mapping.jsonl", "unit-mapping.jsonl.manifest.json"):
            files[name] = artifact_reference(target / name)
            (target / name).chmod(0o444)
    elif cfg.representation.name == "page_text":
        from scientific_tot.integrations.colqwen import pdf_corpus_digest

        corpus_sha256 = pdf_corpus_digest(corpus)
    else:
        corpus_sha256 = records_digest(inputs.documents)
    provenance = {
        "dataset_revision": corpus.provenance["revision"],
        "model_revision": revision,
        "local_revision_verified": corpus.provenance["local_revision_verified"],
        "corpus_sha256": corpus_sha256,
        "vector_input_sha256": records_digest(documents),
        "corpus_provenance": corpus.provenance,
        "corpus_report": corpus.report.to_dict(),
        "documents": len(corpus),
        "vectors": len(mapping),
        "dimension": index.d,
        "document_domains": {row["doc_id"]: row["domain"] for row in corpus},
        "index_files": files,
        "index_location": str(target / "index.faiss"),
        "output_dir": str(target),
        "index_device": "cpu",
        "encoding_config": OmegaConf.to_container(cfg.model, resolve=True)
        if visual
        else {**asdict(config), "resolved_revision": revision},
        "encoding": cached["encoding"],
        "versions": cached["configuration"]["versions"],
        "cache": {"documents": str(entry), "document_hit": hit},
        "aggregation": {"method": "max", "scope": "all_indexed_units"}
        if cfg.representation.name == "page_text"
        else None,
    }
    if cfg.representation.name == "page_text":
        from scientific_tot.reporting import package_version

        provenance["versions"]["pypdfium2"] = package_version("pypdfium2")
    provenance["efficiency"] = {
        "indexed_units": index.ntotal,
        "retrievable_documents": len({row["doc_id"] for row in mapping}),
        "documents_without_units": len(corpus) - len({row["doc_id"] for row in mapping}),
        "faiss_index_bytes": (target / "index.faiss").stat().st_size,
        "cached_embedding_generation_seconds": cached["encoding"].get("generation_seconds"),
    }
    write_index_manifest(target, cfg, provenance=provenance)
    print(
        f"Index ready: {index.ntotal:,} vectors, dimension={index.d}, "
        f"precision=float32; document_cache_hit={hit}",
        flush=True,
    )
    return provenance


def retrieve_dense(cfg, target: Path) -> dict:
    """Encode queries independently and search the saved FAISS index at paper level."""
    import faiss

    from scientific_tot.reporting import tqdm

    location = Path(cfg.paths.index)
    print(f"Verifying FAISS index: {location}", flush=True)
    manifest = verify_index(location, cfg)
    provenance = manifest["provenance"]
    figures = cfg.model.name == "openclip_figures"
    names = ("index.faiss", "ids.json", "vectors.jsonl", "vectors.jsonl.manifest.json")
    for name in names:
        if provenance.get("index_files", {}).get(name) != artifact_reference(location / name):
            raise ValueError(f"Incomplete/corrupt persistent dense index: {name}")
    if figures and provenance["vectors"] == 0 and (location / "vectors.jsonl").stat().st_size == 0:
        mapping = pd.DataFrame(columns=["vector_id", "doc_id", "domain"])
    else:
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
        (set(mapping.doc_id) - set(domains))
        or (not figures and set(mapping.doc_id) != set(domains))
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
    print(
        f"Dense retrieval: {len(corpus):,} papers, {index.ntotal:,} indexed vectors, "
        f"dimension={index.d}, {len(inputs.topics):,} queries; "
        f"query_split={cfg.split.queries}, top_k={cfg.retrieval.top_k}",
        flush=True,
    )
    visual = cfg.model.name in VISUAL_DENSE_MODELS
    config = None if visual else EncodingConfig(**dense_arguments(cfg))
    if visual:
        from scientific_tot.integrations.visual_dense import cached_encoding as visual_encoding

        vectors, cached, entry, hit = visual_encoding(
            cfg,
            task="query",
            rows=inputs.topics.to_dict("records"),
            dataset_revision=queries.provenance["revision"],
        )
    else:
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
    page_mode = cfg.representation.name == "page_text" or cfg.model.name in OPENCLIP_MODELS
    depth = min(cfg.retrieval.top_k, len(ids))
    target_documents = min(cfg.retrieval.top_k, mapping.doc_id.nunique())
    max_depth = depth
    started = perf_counter()
    batches = []
    search = FaissSearch(
        index, device=cfg.retrieval.search_device, threads=cfg.launcher.threads.faiss, depth=depth
    )
    with (
        phase("Search exact vectors and rank papers"),
        tqdm(
            total=len(inputs.topics), desc=f"Retrieving queries ({search.device})", unit="query"
        ) as progress,
    ):
        for start, initial_scores, initial_positions in search.initial_batches(
            vectors, depth, cfg.retrieval.batch_size
        ):
            for offset in range(len(initial_scores)):
                number = start + offset
                qid = inputs.topics.qid.iloc[number]
                search_depth = depth
                while True:
                    rows = []
                    if search_depth:
                        if search_depth == depth:
                            scores, positions = initial_scores[offset], initial_positions[offset]
                        else:
                            values, matches = search.search(
                                vectors[number : number + 1], search_depth
                            )
                            scores, positions = values[0], matches[0]
                        rows = [
                            {
                                "qid": qid,
                                "docno": ids[int(position)],
                                "rank": rank,
                                "score": float(score),
                            }
                            for rank, (position, score) in enumerate(
                                zip(positions, scores, strict=True)
                            )
                            if position >= 0
                        ]
                    result = pd.DataFrame(rows, columns=["qid", "docno", "rank", "score"])
                    if not page_mode or result.empty:
                        break
                    from scientific_tot.integrations.colqwen import aggregate_pages

                    result = aggregate_pages(result, cfg.retrieval.top_k)
                    result["rank"] -= 1  # The shared writer converts zero-based ranks to TREC.
                    # Increase exact FAISS depth until enough unique papers are seen. Include
                    # the whole score tie at the cutoff before applying canonical-ID tie breaks.
                    if search_depth == len(ids) or (
                        len(result) >= target_documents and result.score.iloc[-1] > scores[-1]
                    ):
                        break
                    search_depth = min(2 * search_depth, len(ids))
                max_depth = max(max_depth, search_depth)
                batches.append(result)
                progress.update(1)
    run = pd.concat(batches, ignore_index=True)
    search_seconds = perf_counter() - started
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
        "encoding_config": OmegaConf.to_container(cfg.model, resolve=True)
        if visual
        else asdict(config),
        "encoding": cached["encoding"],
        "cache": {"queries": str(entry), "query_hit": hit},
        "retrieval_cutoff": cfg.retrieval.top_k,
        "vector_search_depth": max_depth,
        "search": search.report(),
        "efficiency": {
            "search_seconds": search_seconds,
            "seconds_per_query": search_seconds / len(inputs.topics),
            "timing_scope": (
                "index transfer, vector search and paper aggregation; excludes query encoding"
            ),
        },
        "aggregation": {"method": "max", "scope": "all_indexed_units"}
        if page_mode
        else None,
        "ranking_level": "paper",
        "versions": cached["configuration"]["versions"],
    }
