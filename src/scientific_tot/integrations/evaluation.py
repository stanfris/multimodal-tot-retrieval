"""Shared standard PyTerrier evaluation/export; no Java or neural imports."""

from pathlib import Path

import pandas as pd

from scientific_tot.reporting import phase, tqdm


def evaluate_run(
    run, topics, qrels, query_ids, target: Path, *, name: str, evaluation=None
) -> pd.DataFrame:
    """Evaluate every selected query using established metric implementations."""
    metrics, per_query = evaluate_metrics(run, topics, qrels, name=name, evaluation=evaluation)
    import pyterrier as pt

    pt.io.write_results(run, str(target / "run.trec"), format="trec", run_name=name)
    metrics.to_csv(target / "metrics.csv", index=False)
    per_query.to_csv(target / "per-query.csv", index=False)
    topics.to_json(target / "topics.jsonl", orient="records", lines=True)
    qrels.to_csv(target / "qrels.tsv", sep="\t", index=False)
    query_ids.to_json(target / "query-ids.jsonl", orient="records", lines=True)
    return metrics


def evaluate_metrics(run, topics, qrels, *, name: str, evaluation=None) -> tuple:
    """Evaluate saved DataFrames only; never construct an index or a retriever."""
    import ir_measures
    import pyterrier as pt
    from omegaconf import OmegaConf

    from scientific_tot.configuration import CONFIG_DIR

    if evaluation is None:
        evaluation = OmegaConf.load(CONFIG_DIR / "evaluation/default.yaml")
    measures = [ir_measures.parse_measure(m) for m in evaluation.metrics]
    print(
        f"Evaluating {len(topics):,} queries, {len(run):,} rankings; "
        f"metrics={','.join(str(m) for m in measures)}",
        flush=True,
    )
    if not run.empty:
        pt.validate.result_frame(run)
    judgments = pt.model.to_ir_measures(qrels)
    results = [] if run.empty else pt.model.to_ir_measures(run)
    aggregators = {measure: measure.aggregator() for measure in measures}
    rows = []
    with (
        phase("Calculate aggregate and per-query retrieval metrics"),
        tqdm(
            ir_measures.iter_calc(measures, judgments, results),
            total=qrels.qid.nunique() * len(measures),
            desc="Evaluating query metrics",
            unit="query-metric",
        ) as progress,
    ):
        for metric in progress:
            aggregators[metric.measure].add(metric.value)
            rows.append(
                {
                    "name": name,
                    "qid": metric.query_id,
                    "measure": str(metric.measure),
                    "value": metric.value,
                }
            )
    metrics = pd.DataFrame(
        [{"name": name, **{str(m): aggregator.result() for m, aggregator in aggregators.items()}}]
    )
    per_query = pd.DataFrame(rows, columns=["name", "qid", "measure", "value"])
    return metrics, per_query
