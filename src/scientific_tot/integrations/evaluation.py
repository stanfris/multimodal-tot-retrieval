"""Shared standard PyTerrier evaluation/export; no Java or neural imports."""

from pathlib import Path

import pandas as pd


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
    import pyterrier as pt
    from pyterrier.measures import RR, R

    if evaluation is None:
        measures = [RR @ 10, R @ 10, R @ 100]  # Preserve published tutorial defaults.
    else:
        import ir_measures

        measures = [ir_measures.parse_measure(m) for m in evaluation.metrics]
    if run.empty:
        # Experiment rejects an entirely empty DataFrame. ir_measures still
        # evaluates every judged query correctly without inventing dummy hits.
        import ir_measures

        judgments = pt.model.to_ir_measures(qrels)
        aggregate = ir_measures.calc_aggregate(measures, judgments, [])
        metrics = pd.DataFrame(
            [{"name": name, **{str(m): value for m, value in aggregate.items()}}]
        )
        per_query = pd.DataFrame(
            [
                {"name": name, "qid": row.query_id, "measure": str(row.measure), "value": row.value}
                for row in ir_measures.iter_calc(measures, judgments, [])
            ],
            columns=["name", "qid", "measure", "value"],
        )
        return metrics, per_query
    metrics = pt.Experiment(
        [run],
        topics,
        qrels,
        eval_metrics=measures,
        names=[name],
        validate="error",
        filter_by_qrels=False,
    )
    # ir_measures owns both aggregate and per-query metric calculations.
    per_query = pt.Experiment(
        [run],
        topics,
        qrels,
        eval_metrics=measures,
        names=[name],
        perquery=True,
        validate="error",
        filter_by_qrels=False,
    )
    return metrics, per_query
