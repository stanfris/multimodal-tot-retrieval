"""Table aggregation uses datasets as the unit and preserves missing measurements."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scientific_tot.retrieval_table import (
    collect_results,
    main,
    render_table,
    retrieval_index_bytes,
)


def artifact(root, domain, *, stage="evaluate", name=None, metrics=None, performance=None):
    folder = "evaluations" if stage == "evaluate" else "indexes"
    directory = root / folder / (name or domain)
    directory.mkdir(parents=True)
    settings = {
        "model": {"name": "bm25"},
        "representation": {"name": "fulltext"},
        "corpus_split": "all",
        "dataset": {"domains": [domain]},
    }
    identity = {"stage": "index", "index": settings}
    if stage == "evaluate":
        identity = {
            "stage": "evaluate",
            "source_identity": {
                "stage": "retrieve",
                "model": settings["model"],
                "representation": settings["representation"],
                "queries": {"split": "test", "mode": "visual-and-text"},
                "source_identity": identity,
            },
        }
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "status": "completed",
                "performance": performance or {},
                "artifact_identity": identity,
            }
        )
    )
    if metrics is not None:
        (directory / "metrics.csv").write_text(metrics)
    if stage == "index":
        with (directory / "data.properties").open("wb") as stream:
            stream.truncate(int(performance["index_bytes"]))
    return directory


def test_partial_macro_average_and_independent_index_measurements(tmp_path):
    artifact(tmp_path, "ACL", metrics="name,RR@10,R@1,R@10\nBM25,0.2,0,0.4\n")
    artifact(tmp_path, "Biology", metrics="name,RR@10,R@10\nBM25,0.8,\n")
    artifact(tmp_path, "ACL", stage="index", performance={"index_bytes": 2e9, "wall_seconds": 5400})
    artifact(tmp_path, "Physics", stage="index", performance={"index_bytes": 4e9})
    values, notes = collect_results(tmp_path)
    table = render_table(values)
    assert not notes
    assert "BM25 & Text & 0.500 & 0.000 & 0.400 & - & 3.000 & 1.500" in table
    assert r"\textbf{Index Time (h)}" in table
    assert "E5 & Text & - & - & - & - & - & -" in table
    assert "ColQwen2.5 & Page & - & - & - & - & - & -" in table
    assert sum(line.count(" & ") == 7 for line in table.splitlines()) == 12
    assert r"\begin{tabular}{@{}llcccccc@{}}" in table
    assert all(line.count(" & ") == 7 for line in table.splitlines() if " & " in line)


def test_missing_metrics_and_failed_or_test_runs(tmp_path):
    artifact(tmp_path, "ACL")
    failed = artifact(tmp_path, "Biology")
    (failed / "manifest.json").write_text('{"stage": "evaluate", "status": "failed"}')
    test = artifact(tmp_path, "Physics", metrics="name,RR@10\nBM25,1\n")
    path = test / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["artifact_identity"]["source_identity"]["source_identity"]["index"]["test_mode"] = True
    path.write_text(json.dumps(manifest))
    values, notes = collect_results(tmp_path)
    assert not values
    assert len(notes) == 3


@pytest.mark.parametrize(
    "model,names",
    [
        ("bm25", ["data.properties", "data.inverted.bf"]),
        ("colbert", ["colbert/0.codes.pt", "colbert/metadata.json"]),
        ("colqwen", ["qdrant/collection/pages/storage.sqlite"]),
        ("dense", ["index.faiss"]),
        ("qwen3_vl_multimodal", ["index.faiss"]),
    ],
)
def test_retrieval_size_excludes_saved_inputs_and_provenance(tmp_path, model, names):
    excluded = [
        "unit-mapping.jsonl",
        "index-manifest.json",
        "manifest.json",
        "ids.json",
        "vectors.jsonl",
        "page-mapping.jsonl",
        "qdrant/meta.json",
        "data.direct.bf",
        "colbert/plan.json",
        "colbert/0.metadata.json",
    ]
    for name in [*names, *excluded]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"12345")
    assert retrieval_index_bytes(tmp_path, model) == 5 * len(names)


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


def test_direct_script_without_installed_package_or_dependencies(tmp_path):
    artifact(tmp_path / "nested/domain", "ACL", metrics="name,RR@10\nBM25,0.5\n")
    script = Path(__file__).resolve().parents[1] / "src/scientific_tot/retrieval_table.py"
    result = subprocess.run(
        [sys.executable, "-I", "-S", str(script), str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "BM25 & Text & 0.500" in result.stdout
    assert "RR@10: 1/5" in result.stderr


@pytest.mark.parametrize(
    "model_id,accepted", [("Qwen/Qwen3-Embedding-0.6B", True), ("Qwen/Qwen3-Embedding-4B", False)]
)
def test_qwen_text_table_requires_the_configured_model(tmp_path, model_id, accepted):
    directory = artifact(tmp_path, "ACL", metrics="name,RR@10\nQwen,0.5\n")
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    identity = manifest["artifact_identity"]["source_identity"]["source_identity"]
    identity["index"]["model"] = {"name": "qwen3_text", "model_id": model_id}
    path.write_text(json.dumps(manifest))
    values, _ = collect_results(tmp_path)
    assert ("Qwen3-Embedding-0.6B & Text & 0.500" in render_table(values)) == accepted
