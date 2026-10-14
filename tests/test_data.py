import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from huggingface_hub import RepoFile, RepoFolder

from scientific_tot import Dataset
from scientific_tot.data.hub import MissingComponentError
from scientific_tot.data.identifiers import document_id_for_paper
from scientific_tot.data.validation import DatasetFormatError, relative_path, validate_query

PAPER = "preprocessed/ACL/ACL_2022.example-1.1"


def test_local_reads_preserve_metadata_and_ids(local_dataset):
    ds = Dataset(local_dir=local_dataset)
    inventory = ds.discover()
    assert inventory.revision is None
    assert inventory.components == ("preprocessed", "query_collections", "splits")
    (collection,) = inventory.collections
    assert collection.key == ("ACL", "query_generation_test", "visual_and_text")
    (row,) = ds.iter_queries(collection)
    assert row["query_id"] == "test_visual_and_text_q00000"
    assert "selected_components" in row["metadata"]
    paper = ds.load_paper(PAPER)
    assert row["relevant_ids"] == [paper.paper_id]
    assert "synthetic generated clue" not in paper.markdown
    assert paper.metadata["figures"][0]["figure_id"] == "authoritative-figure-id"
    assert ds.load_split()["metadata"]["seed"] == 42


def test_missing_components_and_lean_paper(local_dataset):
    ds = Dataset(local_dir=local_dataset)
    with pytest.raises(MissingComponentError, match="Inspect discover"):
        ds.read_json("missing.json")
    (local_dataset / PAPER / "paper.json").unlink()
    with pytest.raises(MissingComponentError):
        Dataset(local_dir=local_dataset).load_paper(PAPER)
    with pytest.raises(MissingComponentError, match="does not exist"):
        Dataset(local_dir=local_dataset / "absent").discover()


@pytest.mark.parametrize("content", ["", "   "])
def test_incomplete_paper(local_dataset, content):
    (local_dataset / PAPER / "markdown.md").write_text(content)
    with pytest.raises(DatasetFormatError, match="incomplete"):
        Dataset(local_dir=local_dataset).load_paper(PAPER)


@pytest.mark.parametrize("path", ["../a", "/a", "a//b", "a/./b", "a\\b", "C:/a", ""])
def test_unsafe_paths(path):
    with pytest.raises(DatasetFormatError):
        relative_path(path)


@pytest.mark.parametrize(
    "change",
    [
        {"query_id": ""},
        {"query": 3},
        {"metadata": []},
        {"relevant_ids": "paper"},
        {"negative_ids": ["ACL_2022.example-1.1"]},
        {"relevant_ids": ["other"]},
    ],
)
def test_malformed_query(local_dataset, change):
    ds = Dataset(local_dir=local_dataset)
    (collection,) = ds.discover().collections
    row = next(ds.iter_queries(collection))
    row.update(change)
    with pytest.raises(DatasetFormatError):
        validate_query(row, "fixture")


@pytest.mark.parametrize("content", ["{bad json}", "[]", '{"train": []}'])
def test_malformed_split(local_dataset, content):
    (local_dataset / "splits/pdf_dataset_split.json").write_text(content)
    with pytest.raises(DatasetFormatError):
        Dataset(local_dir=local_dataset).load_split()


@pytest.mark.parametrize(
    "train,test",
    [
        (["ACL/a.pdf"], ["ACL/a.pdf"]),
        (["ACL/a.pdf", "ACL/a.pdf"], []),
        (["ACL/../a.pdf"], []),
        (["ACL/a.txt"], []),
    ],
)
def test_invalid_split_membership(local_dataset, train, test):
    (local_dataset / "splits/pdf_dataset_split.json").write_text(
        json.dumps({"train": train, "test": test})
    )
    with pytest.raises(DatasetFormatError):
        Dataset(local_dir=local_dataset).load_split()


def test_duplicate_queries_and_bad_json_have_line_numbers(local_dataset):
    ds = Dataset(local_dir=local_dataset)
    (collection,) = ds.discover().collections
    path = local_dataset / collection.queries_path
    line = path.read_text()
    path.write_text(line * 2)
    with pytest.raises(DatasetFormatError, match="queries.jsonl:2: duplicate"):
        list(ds.iter_queries(collection))
    path.write_text(line + "{broken}")
    with pytest.raises(DatasetFormatError, match="queries.jsonl:2: invalid JSON"):
        list(ds.iter_queries(collection))


def test_collection_validation_and_size_guard(local_dataset):
    ds = Dataset(local_dir=local_dataset)
    (c,) = ds.discover().collections
    with pytest.raises(ValueError, match="Select a collection"):
        list(ds.iter_queries(replace(c, domain="Physics")))
    with pytest.raises(ValueError, match="max_bytes"):
        ds.read_json(c.metadata_path, max_bytes=1)
    with pytest.raises(ValueError, match=".json"):
        ds.read_json("data/ACL/shard-00000.tar")
    row = next(ds.iter_queries(c))
    row["metadata"]["mode"] = "visual-only"
    (local_dataset / c.queries_path).write_text(json.dumps(row))
    with pytest.raises(DatasetFormatError, match="mode disagrees"):
        list(ds.iter_queries(c))


def test_authoritative_join_handles_colliding_normalized_names():
    rows = [
        {"source": "ACL", "original_relative_path": "a/b.pdf", "document_id": "pdf-one"},
        {"source": "ACL", "original_relative_path": "a_b.pdf", "document_id": "pdf-two"},
    ]
    assert document_id_for_paper({"source_pdf_relpath": "ACL/a/b.pdf"}, rows) == "pdf-one"
    assert document_id_for_paper({"source_pdf_relpath": "ACL/a_b.pdf"}, rows) == "pdf-two"
    with pytest.raises(DatasetFormatError, match="Ambiguous"):
        document_id_for_paper({"source_pdf_relpath": "ACL/a_b.pdf"}, rows * 2)
    with pytest.raises(LookupError):
        document_id_for_paper({"source_pdf_relpath": "ACL/missing.pdf"}, rows)
    with pytest.raises(DatasetFormatError):
        document_id_for_paper({"paper_id": "do-not-guess"}, rows)


def test_hub_pins_revision_and_only_downloads_requested_file(monkeypatch, local_dataset):
    from scientific_tot.data import hub

    sha = "a" * 40
    filename = "splits/pdf_dataset_split.json"
    calls = []

    class Api:
        def dataset_info(self, repo_id):
            calls.append(("info", repo_id))
            return SimpleNamespace(sha=sha)

        def list_repo_tree(self, repo_id, **kwargs):
            assert kwargs["revision"] == sha
            return [
                RepoFile(path=filename, size=100, oid=sha),
                RepoFile(path="data/ACL/shard-00000.tar", size=10**9, oid=sha),
                RepoFolder(path="data/ACL", oid=sha),
            ]

    def download(repo_id, name, **kwargs):
        calls.append(("download", name, kwargs))
        return str(local_dataset / name)

    monkeypatch.setattr(hub, "HfApi", Api)
    monkeypatch.setattr(hub, "hf_hub_download", download)
    ds = Dataset()
    assert ds.discover().revision == sha
    assert not any(call[0] == "download" for call in calls)
    ds.load_split()
    assert calls[-1][1] == filename
    assert calls[-1][2]["revision"] == sha
    calls.clear()
    Dataset(revision=sha).discover()
    assert not calls
    with pytest.raises(ValueError, match="immutable"):
        Dataset(revision="main")


def test_parquet_preserves_columns(local_dataset):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.Table.from_pylist([{"document_id": "authoritative", "unknown_column": "keep"}])
    pq.write_table(table, local_dataset / "metadata.parquet")
    assert Dataset(local_dir=local_dataset).load_pdf_metadata().equals(table)


def test_split_label_consistency(local_dataset):
    ds = Dataset(local_dir=local_dataset)
    (collection,) = ds.discover().collections
    row = next(ds.iter_queries(collection))
    row["metadata"]["split"] = "train"
    (local_dataset / collection.queries_path).write_text(json.dumps(row))
    with pytest.raises(DatasetFormatError, match="split disagrees"):
        list(ds.iter_queries(collection))


def test_missing_markdown(local_dataset):
    (local_dataset / PAPER / "markdown.md").unlink()
    with pytest.raises(MissingComponentError, match="markdown.md"):
        Dataset(local_dir=local_dataset).load_paper(PAPER)
