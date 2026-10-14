"""Executable stages composed from the canonical adapter and backend integrations."""

from pathlib import Path

from omegaconf import DictConfig

from scientific_tot.adapter import validate_qrels
from scientific_tot.config_arguments import (
    configured_adapter,
    corpus_arguments,
    query_arguments,
    require_corpus_settings,
)


def execute_text_stage(cfg: DictConfig, *, run_dir: Path | None) -> dict:
    """Dispatch independent operations; settings are already resolved by Hydra."""
    if cfg.stage.name != "index" and run_dir is None:
        raise ValueError(f"stage={cfg.stage.name} requires a fresh run directory")
    if cfg.stage.name == "retrieve":
        if cfg.model.name == "dense":
            from scientific_tot.integrations.dense_batch import retrieve_dense

            return retrieve_dense(cfg, run_dir)
        from scientific_tot.integrations.bm25_batch import retrieve_bm25

        return retrieve_bm25(cfg, run_dir)
    if cfg.stage.name == "evaluate":
        from scientific_tot.integrations.bm25_batch import evaluate_saved_run

        return evaluate_saved_run(cfg, run_dir)
    if cfg.stage.name == "index":
        if cfg.paths.index is None:
            raise ValueError("stage=index requires a resolved paths.index destination")
        if Path(cfg.paths.index).exists() and Path(cfg.paths.index) != run_dir:
            raise FileExistsError(f"Refusing to overwrite existing index: {cfg.paths.index}")
    adapter = configured_adapter(cfg)
    if cfg.dataset.download.enabled:
        components = [
            c
            for c in cfg.dataset.download.components
            if cfg.stage.name != "index" or c in {"metadata", "splits", "preprocessed"}
        ]
        if components:
            adapter.download_components(
                domain=list(cfg.dataset.domains),
                components=components,
                split=cfg.split.corpus if cfg.stage.name == "index" else cfg.split.queries,
                mode=cfg.dataset.query_mode,
            )
    corpus = adapter.load_corpus(**corpus_arguments(cfg))
    if cfg.stage.name == "index":
        from types import SimpleNamespace

        require_corpus_settings(cfg, corpus)
        documents = tuple({"docno": row["doc_id"], "text": row["text"]} for row in corpus)
        if not documents or any(
            not row["text"].strip() or any(c.isspace() for c in row["docno"]) for row in documents
        ):
            raise ValueError("Index requires nonempty text and canonical document IDs")
        inputs = SimpleNamespace(documents=documents)
        if cfg.model.name == "dense":
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
    queries = adapter.load_queries(**query_arguments(cfg))
    require_corpus_settings(cfg, corpus, queries)
    qrels = validate_qrels(queries, corpus)

    from scientific_tot.integrations.pyterrier import to_pyterrier

    inputs = to_pyterrier(corpus, queries, qrels)
    result = {
        "documents": len(corpus),
        "queries": len(queries),
        "qrels": len(qrels),
        "corpus_provenance": corpus.provenance,
        "corpus_report": corpus.report.to_dict(),
        "query_provenance": queries.provenance,
    }
    if cfg.stage.name == "prepare":
        for records in (corpus, queries, qrels):
            records.write_jsonl(run_dir / f"{records.kind}.jsonl")
        result["output_dir"] = str(run_dir)
        if cfg.representation.name == "passages":
            from scientific_tot.integrations.dense_batch import project_passages

            passages = project_passages(corpus, cfg)
            passages.write_jsonl(run_dir / "passages.jsonl")
            result["passages"] = len(passages)
    return result
