"""Regressions for the documented commands, shared selection, and notebook defaults."""

import json
import runpy
import shutil
import sys
from pathlib import Path

import pytest

from scientific_tot import Dataset, DatasetAdapter
from scientific_tot.integrations.selection import load_experiment_data

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/adapter"


@pytest.fixture
def adapter():
    return DatasetAdapter(Dataset(local_dir=FIXTURE))


def test_fulltext_retrieval_needs_no_bibliography_or_pdf_metadata(tmp_path):
    root = tmp_path / "data"
    shutil.copytree(FIXTURE, root)
    # A missing/unusable bibliography must not affect full-document retrieval.
    (root / "metadata.parquet").write_bytes(b"not a readable metadata index")
    corpora, queries = load_experiment_data(DatasetAdapter(Dataset(local_dir=root)))
    assert set(corpora) == {"fulltext"}
    assert len(corpora["fulltext"]) == 2
    assert len(queries) == 1
    for row in corpora["fulltext"]:
        assert row["title"] is row["abstract"] is None
        assert (
            row["text"] == (root / "preprocessed/ACL" / row["doc_id"] / "markdown.md").read_text()
        )


def test_readme_bm25_cli_runs_only_fulltext(adapter, monkeypatch, tmp_path):
    import pandas as pd

    seen = []

    def experiment(corpus, queries, output, **kwargs):
        seen.append((corpus[0]["representation"], len(corpus), len(queries)))
        output.mkdir(parents=True)
        return pd.DataFrame([{"RR@10": 1.0}])

    monkeypatch.setattr("scientific_tot.integrations.pyterrier.bm25_experiment", experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pyterrier_bm25.py",
            "--local-dir",
            str(FIXTURE),
            "--domain",
            "ACL",
            "--output",
            str(tmp_path / "output"),
        ],
    )
    runpy.run_path(str(ROOT / "examples/pyterrier_bm25.py"), run_name="__main__")
    assert seen == [("fulltext", 2, 1)]


@pytest.mark.parametrize("name", ["pyterrier_bm25", "dense_retrieval"])
def test_notebook_defaults_use_full_collection(name, monkeypatch, full_collection):
    import nbformat

    monkeypatch.chdir(ROOT / "examples")
    for key in ("TOT_DATA_ROOT", "TOT_OUTPUT_DIR", "TOT_DENSE_OUTPUT", "TOT_BM25_BASELINE"):
        monkeypatch.delenv(key, raising=False)
    notebook = nbformat.read(ROOT / f"examples/{name}.ipynb", as_version=4)
    nbformat.validate(notebook)
    assert notebook.metadata.kernelspec.name == "scientific-tot"
    namespace = {}
    exec(notebook.cells[1].source, namespace)
    assert namespace["DATA"] == ROOT.parent / "multimodal-tip-of-tongue-for-scientific-papers/data"
    assert namespace["DOCUMENTS"] is namespace["QUERIES"] is None
    assert namespace["CORPUS_SPLIT"] == "all"
    assert not namespace["OUTPUT"].exists()
    # The unchanged default selection must include unassigned distractors and
    # exceed both old demo limits. Only the data location is overridden.
    monkeypatch.setenv("TOT_DATA_ROOT", str(full_collection))
    for cell in notebook.cells[1:3]:
        exec(cell.source, namespace)
    assert len(namespace["corpora"]["fulltext"]) == 262
    assert len(namespace["queries"]) == 12


@pytest.mark.parametrize(
    "name,output_env",
    [
        pytest.param("pyterrier_bm25", "TOT_OUTPUT_DIR", marks=pytest.mark.terrier),
        pytest.param("dense_retrieval", "TOT_DENSE_OUTPUT", marks=pytest.mark.dense_live),
    ],
)
def test_notebook_execute_all_in_shared_environment(name, output_env, monkeypatch, tmp_path):
    import nbformat
    from nbclient import NotebookClient

    monkeypatch.setenv("TOT_DATA_ROOT", str(FIXTURE))
    monkeypatch.setenv(output_env, str(tmp_path / "output"))
    monkeypatch.setenv("TOT_DENSE_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("TOT_BM25_BASELINE", raising=False)
    notebook = nbformat.read(ROOT / f"examples/{name}.ipynb", as_version=4)
    NotebookClient(
        notebook,
        timeout=600,
        kernel_name="scientific-tot",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    assert (tmp_path / "output/fulltext/metrics.csv").exists()
    if name == "pyterrier_bm25":
        assert not (tmp_path / "output/title_abstract").exists()
    else:
        manifest = json.loads((tmp_path / "output/fulltext-cached/manifest.json").read_text())
        assert manifest["cache"]["document_hit"] and manifest["cache"]["query_hit"]


@pytest.fixture
def full_collection(tmp_path):
    root = tmp_path / "data"
    shutil.copytree(FIXTURE, root)
    for index in range(260):
        paper_id = f"ACL_distractor_{index:04d}"
        folder = root / "preprocessed/ACL" / paper_id
        folder.mkdir()
        (folder / "paper.json").write_text(
            json.dumps(
                {
                    "paper_id": paper_id,
                    "source_pdf_relpath": f"ACL/distractor_{index:04d}.pdf",
                    "source_paper_dataset": "ACL",
                }
            )
        )
        (folder / "markdown.md").write_text("Original scientific distractor text.")
    path = root / "query_collections/ACL/query_generation_test/visual_and_text/queries.jsonl"
    template = json.loads(path.read_text())
    rows = [{**template, "query_id": f"test_visual_and_text_q{i:05d}"} for i in range(12)]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    metadata = path.with_name("metadata.json")
    info = json.loads(metadata.read_text())
    info["example_count"] = 12
    metadata.write_text(json.dumps(info))
    return root


@pytest.mark.parametrize("name", ["pyterrier_bm25", "dense_retrieval", "convert_dataset"])
def test_cli_defaults_include_full_collection(name, full_collection, monkeypatch, tmp_path):
    import pandas as pd

    seen = []

    def experiment(corpus, queries, output, **kwargs):
        seen.append((len(corpus), len(queries), corpus[0]["representation"]))
        output.mkdir(parents=True)
        return pd.DataFrame([{"RR@10": 1.0}])

    module, func = (
        ("pyterrier", "bm25_experiment")
        if name == "pyterrier_bm25"
        else ("dense", "dense_experiment")
    )
    monkeypatch.setattr(f"scientific_tot.integrations.{module}.{func}", experiment)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            name,
            "--local-dir",
            str(full_collection),
            "--domain",
            "ACL",
            "--output",
            str(output),
        ],
    )
    runpy.run_path(str(ROOT / f"examples/{name}.py"), run_name="__main__")
    if name == "convert_dataset":
        manifest = json.loads((output / "corpus.jsonl.manifest.json").read_text())
        assert manifest["count"] == 262
    else:
        assert seen == [(262, 12, "fulltext")]


@pytest.mark.parametrize("name", ["pyterrier_bm25", "dense_retrieval"])
def test_cli_explicit_demo_limits_still_apply(name, full_collection, monkeypatch, tmp_path):
    import pandas as pd

    seen = []

    def experiment(corpus, queries, output, **kwargs):
        seen.append((len(corpus), len(queries)))
        output.mkdir(parents=True)
        return pd.DataFrame([{"RR@10": 1.0}])

    module, func = (
        ("pyterrier", "bm25_experiment")
        if name == "pyterrier_bm25"
        else ("dense", "dense_experiment")
    )
    monkeypatch.setattr(f"scientific_tot.integrations.{module}.{func}", experiment)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            name,
            "--local-dir",
            str(full_collection),
            "--domain",
            "ACL",
            "--corpus-split",
            "train+test",
            "--documents",
            "2",
            "--queries",
            "1",
            "--output",
            str(tmp_path / "output"),
        ],
    )
    runpy.run_path(str(ROOT / f"examples/{name}.py"), run_name="__main__")
    assert seen == [(2, 1)]
