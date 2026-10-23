"""Table aggregation uses datasets as the unit and preserves missing measurements."""

import json

import pytest
from omegaconf import OmegaConf

from scientific_tot.retrieval_table import collect_results, main, render_table


def artifact(root, domain, *, stage="evaluate", name=None, metrics=None, performance=None):
    folder = "evaluations" if stage == "evaluate" else "indexes"
    directory = root / folder / (name or domain)
    (directory / ".hydra").mkdir(parents=True)
    config = {
        "model": {"name": "bm25"},
        "representation": {"name": "fulltext"},
        "split": {"corpus": "all", "queries": "test"},
        "dataset": {"domains": [domain], "query_mode": "visual-and-text"},
    }
    OmegaConf.save(config, directory / ".hydra/config.yaml")
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "status": "completed",
                "performance": performance or {},
            }
        )
    )
    if metrics is not None:
        (directory / "metrics.csv").write_text(metrics)
    return directory


def test_partial_macro_average_and_independent_index_measurements(tmp_path):
    artifact(tmp_path, "ACL", metrics="name,RR@10,R@1,R@10\nBM25,0.2,0,0.4\n")
    artifact(tmp_path, "Biology", metrics="name,RR@10,R@10\nBM25,0.8,\n")
    artifact(tmp_path, "ACL", stage="index", performance={"index_bytes": 2e9, "wall_seconds": 10})
    artifact(tmp_path, "Physics", stage="index", performance={"index_bytes": 4e9})
    values, notes = collect_results(tmp_path)
    table = render_table(values)
    assert not notes
    assert "BM25 & Text & 0.5000 & 0.0000 & 0.4000 & - & 3.0000 & 10.0" in table
    assert "E5 & Text & - & - & - & - & - & -" in table
    assert "ColQwen" not in table
    assert sum(line.count(" & ") == 7 for line in table.splitlines()) == 10
    assert r"\begin{tabular}{@{}llcccccc@{}}" in table
    assert all(line.count(" & ") == 7 for line in table.splitlines() if " & " in line)


def test_missing_metrics_and_failed_or_test_runs(tmp_path):
    artifact(tmp_path, "ACL")
    failed = artifact(tmp_path, "Biology")
    (failed / "manifest.json").write_text('{"status": "failed"}')
    test = artifact(tmp_path, "Physics", metrics="name,RR@10\nBM25,1\n")
    config = OmegaConf.load(test / ".hydra/config.yaml")
    config.test_mode = True
    OmegaConf.save(config, test / ".hydra/config.yaml")
    values, notes = collect_results(tmp_path)
    assert not values
    assert len(notes) == 3


def test_repeated_dataset_is_ambiguous(tmp_path):
    for name in ("first", "second"):
        artifact(tmp_path, "ACL", name=name, metrics="name,RR@10\nBM25,0.5\n")
    with pytest.raises(ValueError, match="Multiple RR@10 results"):
        collect_results(tmp_path)


@pytest.mark.parametrize("value", ["nan", "inf", "-0.1", "1.1"])
def test_invalid_external_measurements_fail(tmp_path, value):
    artifact(tmp_path, "ACL", metrics=f"name,RR@10\nBM25,{value}\n")
    with pytest.raises(ValueError, match="Invalid RR@10"):
        collect_results(tmp_path)


def test_cli_keeps_coverage_out_of_latex(tmp_path, monkeypatch, capsys):
    artifact(tmp_path, "ACL", metrics="name,RR@10\nBM25,0.5\n")
    monkeypatch.setattr("sys.argv", ["retrieval_table", str(tmp_path)])
    main()
    captured = capsys.readouterr()
    assert captured.out.startswith(r"\begin{table}")
    assert "Coverage" not in captured.out
    assert "RR@10: 1/5; missing: Biology, Engineering, Medicine, Physics" in captured.err
    assert "Source:" in captured.err
