"""Shared standard PyTerrier evaluation/export; no Java or neural imports."""

from pathlib import Path

import pandas as pd


def evaluate_run(run, topics, qrels, query_ids, target: Path, *, name: str) -> pd.DataFrame:
    """Evaluate every selected query using established metric implementations."""
    import pyterrier as pt
    from pyterrier.measures import RR, R

    metrics = pt.Experiment(
        [run],
        topics,
        qrels,
        eval_metrics=[RR @ 10, R @ 10, R @ 100],
        names=[name],
        validate="error",
        filter_by_qrels=False,
    )
    pt.io.write_results(run, str(target / "run.trec"), format="trec", run_name=name)
    # ir_measures owns both aggregate and per-query metric calculations.
    per_query = pt.Experiment(
        [run],
        topics,
        qrels,
        eval_metrics=[RR @ 10, R @ 10, R @ 100],
        names=[name],
        perquery=True,
        validate="error",
        filter_by_qrels=False,
    )
    metrics.to_csv(target / "metrics.csv", index=False)
    per_query.to_csv(target / "per-query.csv", index=False)
    topics.to_json(target / "topics.jsonl", orient="records", lines=True)
    qrels.to_csv(target / "qrels.tsv", sep="\t", index=False)
    query_ids.to_json(target / "query-ids.jsonl", orient="records", lines=True)
    return metrics
