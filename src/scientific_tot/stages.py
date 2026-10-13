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
        if Path(cfg.paths.index).exists():
            raise FileExistsError(f"Refusing to overwrite existing index: {cfg.paths.index}")
    adapter = configured_adapter(cfg)
    if cfg.dataset.download.enabled:
        adapter.download_components(
            domain=list(cfg.dataset.domains),
            components=list(cfg.dataset.download.components),
            split=cfg.split.queries,
            mode=cfg.dataset.query_mode,
        )
    corpus = adapter.load_corpus(**corpus_arguments(cfg))
    queries = adapter.load_queries(**query_arguments(cfg))
    require_corpus_settings(cfg, corpus, queries)
    qrels = validate_qrels(queries, corpus)

    from scientific_tot.integrations.pyterrier import build_bm25_index, to_pyterrier

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
    else:
        if cfg.model.name == "dense":
            from scientific_tot.integrations.dense_batch import build_dense_index

            result["index_provenance"] = build_dense_index(corpus, inputs, cfg)
        else:
            result["index_provenance"] = build_bm25_index(corpus, inputs, cfg)
    return result
