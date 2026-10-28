"""Shared retrieval metrics; no Java or neural imports."""

import pandas as pd

from scientific_tot.reporting import phase, tqdm


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
