"""Executable stages composed from the canonical adapter and backend integrations."""

from pathlib import Path

from omegaconf import DictConfig

from scientific_tot.adapter import validate_qrels
from scientific_tot.config_arguments import (
    configured_adapter,
    configured_corpus,
    query_arguments,
)
from scientific_tot.configuration import VISUAL_DENSE_MODELS


def execute_text_stage(cfg: DictConfig, *, run_dir: Path | None) -> dict:
    """Dispatch independent operations; settings are already resolved by Hydra."""
    assert cfg.stage.name == "index" or run_dir is not None, (
        f"stage={cfg.stage.name} requires a fresh run directory"
    )
    if cfg.model.name == "colqwen" and cfg.stage.name != "evaluate":
        from scientific_tot.integrations.colqwen import execute_visual_stage

        return execute_visual_stage(cfg, run_dir)
    if cfg.stage.name == "retrieve":
        if cfg.model.name == "colbert":
            from scientific_tot.integrations.colbert import retrieve_colbert

            return retrieve_colbert(cfg, run_dir)
        if cfg.model.name in {"dense"} | VISUAL_DENSE_MODELS:
            from scientific_tot.integrations.dense_batch import retrieve_dense

            return retrieve_dense(cfg, run_dir)
        from scientific_tot.integrations.bm25_batch import retrieve_bm25

        return retrieve_bm25(cfg, run_dir)
    if cfg.stage.name == "evaluate":
        from scientific_tot.integrations.bm25_batch import evaluate_saved_run

        return evaluate_saved_run(cfg, run_dir)
    if cfg.stage.name == "index":
        assert cfg.paths.index is not None, "stage=index requires a resolved index destination"
        if Path(cfg.paths.index).exists() and Path(cfg.paths.index) != run_dir:
            print(f"Skipping existing index: {cfg.paths.index}", flush=True)
            return {"index_location": cfg.paths.index}
    adapter = configured_adapter(cfg)
    if cfg.dataset.download.enabled:
        components = [
            c
            for c in cfg.dataset.download.components
            if cfg.stage.name != "index" or c in {"metadata", "splits", "preprocessed"}
        ]
        if components:
            adapter.download_components(
                domain=cfg.dataset.domains[0],
                components=components,
                split=cfg.split.corpus if cfg.stage.name == "index" else cfg.split.queries,
                mode=cfg.dataset.query_mode,
            )
    print(
        f"Corpus selection: domains={','.join(cfg.dataset.domains)}, split={cfg.split.corpus}, "
        f"representation={cfg.representation.name}",
        flush=True,
    )
    corpus = configured_corpus(cfg, adapter)
    print(f"Selected corpus: {len(corpus):,} documents", flush=True)
    if cfg.stage.name == "index":
        from types import SimpleNamespace

        print(
            f"Index build: model={cfg.model.get('model_id', cfg.model.name)}, "
            f"backend={cfg.model.backend}, dataset_revision={corpus.provenance['revision']}, "
            f"output={cfg.paths.index}",
            flush=True,
        )
        inputs = None
        if cfg.model.name not in VISUAL_DENSE_MODELS:
            documents = tuple({"docno": row["doc_id"], "text": row["text"]} for row in corpus)
            if not documents or any(
                not row["text"].strip() or any(c.isspace() for c in row["docno"])
                for row in documents
            ):
                raise ValueError("Index requires nonempty text and canonical document IDs")
            inputs = SimpleNamespace(documents=documents)
        if cfg.model.name == "colbert":
            from scientific_tot.integrations.colbert import build_colbert_index

            provenance = build_colbert_index(
                corpus, inputs, cfg, reserved=Path(cfg.paths.index) == run_dir
            )
        elif cfg.model.name in {"dense"} | VISUAL_DENSE_MODELS:
            from scientific_tot.integrations.dense_batch import build_dense_index

            provenance = build_dense_index(
                corpus, inputs, cfg, reserved=Path(cfg.paths.index) == run_dir
            )
        else:
            from scientific_tot.integrations.pyterrier import build_bm25_index

            provenance = build_bm25_index(
                corpus, inputs, cfg, reserved=Path(cfg.paths.index) == run_dir
            )
        return {"documents": len(corpus), "index_provenance": provenance}
    print(f"Loading queries: split={cfg.split.queries}, mode={cfg.dataset.query_mode}", flush=True)
    queries = adapter.load_queries(**query_arguments(cfg))
    print(f"Selected queries: {len(queries):,}", flush=True)
    qrels = validate_qrels(queries, corpus)

    result = {
        "documents": len(corpus),
        "queries": len(queries),
        "qrels": len(qrels),
        "corpus_provenance": corpus.provenance,
        "corpus_report": corpus.report.to_dict(),
        "query_provenance": queries.provenance,
    }
    assert cfg.stage.name == "prepare"
    for records in (corpus, queries, qrels):
        records.write_jsonl(run_dir / f"{records.kind}.jsonl")
    result["output_dir"] = str(run_dir)
    if cfg.model.name in VISUAL_DENSE_MODELS:
        from scientific_tot.integrations.visual_dense import document_mapping
        from scientific_tot.records import Records, Report

        units = document_mapping(corpus, cfg)
        Records("unit_mapping", units, Report(emitted=len(units)), corpus.provenance).write_jsonl(
            run_dir / "unit-mapping.jsonl"
        )
        result["units"] = len(units)
    if cfg.representation.name == "page_text":
        from scientific_tot.records import Records, Report

        pages = [page for paper in corpus for page in paper["pages"]]
        Records("pages", pages, Report(emitted=len(pages)), corpus.provenance).write_jsonl(
            run_dir / "pages.jsonl"
        )
        result["pages"] = len(pages)
    return result
