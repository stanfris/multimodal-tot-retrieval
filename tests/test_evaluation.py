"""Shared evaluation defaults and cutoff behavior without retrieval backends."""

import math

import pandas as pd
import pytest

from scientific_tot.integrations.evaluation import evaluate_metrics


@pytest.mark.parametrize("empty", [False, True])
def test_default_metrics_and_cutoffs(empty):
    topics = pd.DataFrame([{"qid": "q1", "query": "example"}])
    qrels = pd.DataFrame(
        [
            {"qid": "q1", "docno": "d2", "label": 1},
            {"qid": "q1", "docno": "d11", "label": 1},
        ]
    )
    run = pd.DataFrame(
        [
            {"qid": "q1", "docno": f"d{i}", "rank": i - 1, "score": float(12 - i)}
            for i in range(1, 12)
        ]
    )
    if empty:
        run = run.iloc[:0]
    metrics, per_query = evaluate_metrics(run, topics, qrels, name="example")
    expected = {
        "RR@10": 0.5,
        "R@1": 0.0,
        "R@10": 0.5,
        "R@100": 1.0,
        "nDCG@10": (1 / math.log2(3)) / (1 + 1 / math.log2(3)),
    }
    assert set(metrics.columns) == {"name", *expected}
    assert set(per_query.measure) == set(expected)
    for measure, value in expected.items():
        value = 0.0 if empty else value
        assert metrics.iloc[0][measure] == pytest.approx(value)
        assert per_query.loc[per_query.measure == measure, "value"].item() == pytest.approx(value)
