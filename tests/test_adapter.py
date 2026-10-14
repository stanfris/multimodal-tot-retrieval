import hashlib
import io
import json
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.hf_api import RepoFile

from scientific_tot import (
    Dataset,
    DatasetAdapter,
    IncompleteCorpusError,
    UnresolvedRelevanceError,
    load_corpus,
    load_metadata,
    load_qrels,
    load_queries,
    validate_qrels,
)
from scientific_tot.data.hub import MissingComponentError
from scientific_tot.data.identifiers import extraction_path_candidates
from scientific_tot.data.validation import DatasetFormatError

PAPER = "preprocessed/ACL/ACL_2022.example-1.1"
TRAIN = "preprocessed/ACL/ACL_2022.example-1.2"
QPATH = "query_collections/ACL/query_generation_test/visual_and_text/queries.jsonl"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def complete(local_dataset):
    metadata = json.loads((local_dataset / PAPER / "paper.json").read_text())
    metadata.update(
        paper_id="ACL_2022.example-1.2",
        source_pdf_relpath="ACL/2022.example-1.2.pdf",
        figures=[],
        split="train",
    )
    write_json(local_dataset / TRAIN / "paper.json", metadata)
    (local_dataset / TRAIN / "markdown.md").write_text("# Train document\n")
    return local_dataset


def test_public_api_and_export(complete, tmp_path):
    ds = Dataset(local_dir=complete)
    corpus = load_corpus(dataset=ds, domain="ACL")
    queries = load_queries(dataset=ds, domain="ACL")
    qrels = load_qrels(dataset=ds, domain="ACL", corpus=corpus)
    assert [r["doc_id"] for r in corpus] == ["ACL_2022.example-1.1", "ACL_2022.example-1.2"]
    assert len(qrels) == len(queries) == 1
    assert qrels[0]["doc_id"] == corpus[0]["doc_id"]
    assert queries[0]["query_id"] == "test_visual_and_text_q00000"
    assert qrels[0]["query_key"] == queries[0]["query_key"]
    assert qrels[0]["relevance"] == 1
    assert "synthetic generated clue" not in corpus[0]["text"]
    assert "selected_components" in queries[0]["metadata"]
    for result in (corpus, queries, qrels):
        path = result.write_jsonl(tmp_path / "out" / f"{result.kind}.jsonl")
        assert [json.loads(line) for line in path.read_text().splitlines()] == list(result)
        manifest = json.loads(path.with_name(path.name + ".manifest.json").read_text())
        assert manifest["count"] == len(result)
        assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # Readers cannot mutate a validated corpus's internal record IDs.
    corpus[0]["doc_id"] = "changed"
    assert validate_qrels(queries, corpus)[0]["doc_id"] == "ACL_2022.example-1.1"


def test_default_strict_missing_extraction(local_dataset):
    with pytest.raises(IncompleteCorpusError) as error:
        load_corpus(local_dir=local_dataset, domain="ACL")
    assert error.value.report.skipped == 1
    assert error.value.report.issues[-1].code == "missing_extraction"
    corpus = load_corpus(local_dir=local_dataset, domain="ACL", missing="skip")
    assert len(corpus) == 1
    assert corpus.report.selected == 2
    assert corpus.report.skipped == 1


@pytest.mark.parametrize("content,code", [(None, "missing_fulltext"), ("  ", "empty_fulltext")])
def test_missing_content_policy(complete, content, code):
    path = complete / PAPER / "markdown.md"
    path.unlink() if content is None else path.write_text(content)
    with pytest.raises(IncompleteCorpusError) as error:
        load_corpus(local_dir=complete, domain="ACL")
    assert code in {i.code for i in error.value.report.issues}
    ds = Dataset(local_dir=complete)
    corpus = load_corpus(dataset=ds, domain="ACL", missing="skip")
    assert len(corpus) == 1
    with pytest.raises(UnresolvedRelevanceError) as unresolved:
        load_qrels(dataset=ds, domain="ACL", corpus=corpus)
    assert unresolved.value.report.issues[0].doc_id == "ACL_2022.example-1.1"


def test_checksum_failure(complete):
    path = complete / PAPER / "paper.json"
    paper = json.loads(path.read_text())
    paper["markdown_sha256"] = "0" * 64
    write_json(path, paper)
    with pytest.raises(IncompleteCorpusError) as error:
        load_corpus(local_dir=complete, domain="ACL")
    assert error.value.report.issues[0].code == "markdown_checksum_mismatch"


def test_split_all_includes_distractors(complete):
    folder = complete / "preprocessed/ACL/ACL_distractor"
    write_json(
        folder / "paper.json",
        {"paper_id": "ACL_distractor", "source_pdf_relpath": "ACL/distractor.pdf", "figures": []},
    )
    (folder / "markdown.md").write_text("Distractor content")
    assert len(load_corpus(local_dir=complete, domain="ACL", split="all")) == 3
    assert len(load_corpus(local_dir=complete, domain="ACL", split="test")) == 1
    assert len(load_corpus(local_dir=complete, domain="ACL", split="train+test")) == 2


def test_title_metadata_and_licensing_without_pdf(complete, monkeypatch):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = [
        {
            "document_id": f"packaged-{n}",
            "source": "ACL",
            "original_relative_path": f"2022.example-1.{n}.pdf",
            "title": f"Original title {n}",
            "license": "CC-BY-4.0",
            "doi": "10.0000/synthetic",
            "sha256": "original-hash",
        }
        for n in (1, 2)
    ]
    pq.write_table(pa.Table.from_pylist(rows), complete / "metadata.parquet")
    ds = Dataset(local_dir=complete)
    original = ds.get_file
    opened = []

    def get(name):
        opened.append(name)
        assert not name.endswith((".pdf", "markdown.md"))
        return original(name)

    monkeypatch.setattr(ds, "get_file", get)
    corpus = load_corpus(dataset=ds, domain="ACL", representation="title_abstract")
    assert corpus[0]["abstract"] is None
    assert corpus[0]["text"] == "Original title 1"
    assert corpus[0]["pdf_document_id"] == "packaged-1"
    assert corpus[0]["metadata"]["pdf"]["license"] == "CC-BY-4.0"
    assert "partial_title_abstract" in {i.code for i in corpus.report.issues}
    assert len(load_metadata(dataset=ds, domain="ACL")) == 2
    assert not any(p.endswith("markdown.md") for p in opened)


def test_missing_title_does_not_generate_abstract(complete):
    with pytest.raises(IncompleteCorpusError):
        load_corpus(local_dir=complete, domain="ACL", representation="title_abstract")
    corpus = load_corpus(
        local_dir=complete, domain="ACL", representation="title_abstract", missing="skip"
    )
    assert len(corpus) == 0
    assert corpus.report.skipped == 2


def test_explicit_original_abstract_preserved(complete):
    # Optional source metadata, not generated or extracted from markdown by this adapter.
    path = complete / PAPER / "paper.json"
    paper = json.loads(path.read_text())
    paper.update(title="Original title", abstract="Original abstract", license="CC0")
    write_json(path, paper)
    corpus = load_corpus(
        local_dir=complete, domain="ACL", split="test", representation="title_abstract"
    )
    assert corpus[0]["text"] == "Original title\n\nOriginal abstract"
    assert corpus[0]["metadata"]["paper"]["license"] == "CC0"


def test_lean_identity_and_images_never_read(complete, monkeypatch):
    (complete / PAPER / "paper.json").unlink()
    image = complete / PAPER / "images/figure.png"
    image.parent.mkdir()
    image.write_bytes(b"not-decoded")
    original = Path.open

    def guarded(self, *args, **kwargs):
        assert self.suffix != ".png", "Images must not be opened"
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    corpus = load_corpus(local_dir=complete, domain="ACL", split="test")
    assert corpus[0]["doc_id"] == "ACL_2022.example-1.1"
    assert corpus[0]["source_pdf_relpath"] == "ACL/2022.example-1.1.pdf"
    assert corpus[0]["figures"][0]["figure_id"] is None
    assert corpus[0]["figures"][0]["image"]["available"] is True
    assert corpus[0]["provenance"]["identity_source"] == "extraction_directory"


def test_lean_collision_lookup_requires_unambiguous_source(tmp_path):
    a, b = "ACL/a/b.pdf", "ACL/a_b.pdf"
    write_json(tmp_path / "splits/pdf_dataset_split.json", {"train": [a, b], "test": []})
    base = extraction_path_candidates(a)[0]
    assert base == extraction_path_candidates(b)[0]
    folder = tmp_path / f"preprocessed/ACL/{base}"
    folder.mkdir(parents=True)
    (folder / "markdown.md").write_text("Original")
    with pytest.raises(DatasetFormatError, match="ambiguous"):
        load_corpus(local_dir=tmp_path, domain="ACL", split="train")
    folder.rename(folder.with_name(extraction_path_candidates(a)[1]))
    other = tmp_path / "preprocessed/ACL" / extraction_path_candidates(b)[1]
    other.mkdir()
    (other / "markdown.md").write_text("Another original")
    corpus = load_corpus(local_dir=tmp_path, domain="ACL", split="train")
    assert {r["doc_id"] for r in corpus} == {
        extraction_path_candidates(a)[1],
        extraction_path_candidates(b)[1],
    }


def test_figure_metadata_and_missing_images(complete):
    corpus = load_corpus(local_dir=complete, domain="ACL", split="test")
    figure = corpus[0]["figures"][0]
    assert figure["figure_id"] == "authoritative-figure-id"
    assert figure["page_idx"] == 0
    assert not figure["image"]["available"]
    assert "missing_figure_image" in {i.code for i in corpus.report.issues}
    assert load_corpus(local_dir=complete, domain="ACL", include_figures=False)[0]["figures"] == []


def make_tar(path, members, *, link=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w") as tar:
        for name, data in members.items():
            item = tarfile.TarInfo(name)
            item.size = len(data)
            tar.addfile(item, io.BytesIO(data))
        if link:
            item = tarfile.TarInfo("link")
            item.type = tarfile.SYMTYPE
            item.linkname = "outside"
            tar.addfile(item)


def archive_extractions(root):
    base = root / "preprocessed/ACL"
    members = {
        p.relative_to(base).as_posix(): p.read_bytes() for p in base.rglob("*") if p.is_file()
    }
    shutil.rmtree(base)
    make_tar(base / "shard-00000.tar", members)


def test_archive_matches_restored_content(complete):
    before = load_corpus(local_dir=complete, domain="ACL")
    archive_extractions(complete)
    after = load_corpus(local_dir=complete, domain="ACL")
    assert [(r["doc_id"], r["text"]) for r in before] == [(r["doc_id"], r["text"]) for r in after]
    assert after[0]["provenance"]["content"]["member"].endswith("markdown.md")
    assert not (complete / PAPER).exists()  # no restoration, no extraction to disk
    with pytest.raises(MissingComponentError, match="archive reads disabled"):
        load_corpus(local_dir=complete, domain="ACL", archives=False)


@pytest.mark.parametrize("member", ["../escape", "/absolute", "paper/../escape", "paper\\escape"])
def test_archive_rejects_unsafe_members(tmp_path, member):
    make_tar(tmp_path / "preprocessed/ACL/shard-00000.tar", {member: b"data"})
    with pytest.raises(DatasetFormatError):
        load_corpus(local_dir=tmp_path, domain="ACL", split="all")


def test_archive_rejects_links(tmp_path):
    make_tar(tmp_path / "preprocessed/ACL/shard-00000.tar", {}, link=True)
    with pytest.raises(DatasetFormatError, match="unsupported"):
        load_corpus(local_dir=tmp_path, domain="ACL", split="all")


def test_selective_hub_downloads_pinned_and_reusable(complete, monkeypatch):
    from scientific_tot.data import hub

    archive_extractions(complete)
    sha = "a" * 40
    inventory = [
        RepoFile(path=p.relative_to(complete).as_posix(), size=p.stat().st_size, oid=sha)
        for p in complete.rglob("*")
        if p.is_file()
    ]
    inventory += [
        RepoFile(path="preprocessed/Physics/shard-00000.tar", size=10**9, oid=sha),
        RepoFile(path="data/ACL/shard-00000.tar", size=10**9, oid=sha),
    ]
    calls = []
    transfers = set()

    class Api:
        def dataset_info(self, repo_id):
            return SimpleNamespace(sha=sha)

        def list_repo_tree(self, repo_id, **kwargs):
            assert kwargs["revision"] == sha
            return inventory

    def download(repo_id, filename, **kwargs):
        assert kwargs["revision"] == sha
        calls.append(filename)
        transfers.add(filename)  # emulate the Hub's cached repeated file download
        return str(complete / filename)

    monkeypatch.setattr(hub, "HfApi", Api)
    monkeypatch.setattr(hub, "hf_hub_download", download)
    adapter = DatasetAdapter(Dataset())
    corpus = adapter.load_corpus(domain="ACL", split="test")
    assert len(corpus) == 1
    assert set(calls) == {"splits/pdf_dataset_split.json", "preprocessed/ACL/shard-00000.tar"}
    adapter.load_corpus(domain="ACL", split="train")
    assert len(transfers) == 2
    calls.clear()
    adapter.download_components(domain="ACL", components=["queries"], split="test")
    assert set(calls) == {QPATH, QPATH.replace("queries.jsonl", "metadata.json")}
    assert not any(name.endswith(".tar") for name in calls)
    calls.clear()
    with pytest.raises(MissingComponentError):
        adapter.download_components(domain="ACL", components=["queries", "metadata"], split="test")
    assert not calls  # all component selection is validated before downloading


def test_query_ids_scoped_across_domains(complete):
    row = json.loads((complete / QPATH).read_text())
    row["metadata"].update(paper_id="Physics_one", source_paper_dataset="Physics")
    row["relevant_ids"] = ["Physics_one"]
    target = complete / QPATH.replace("/ACL/", "/Physics/")
    write_json(target, row)
    queries = load_queries(local_dir=complete, domain=["ACL", "Physics"])
    assert queries[0]["query_id"] == queries[1]["query_id"]
    assert queries[0]["query_key"] != queries[1]["query_key"]
    corpus = load_corpus(local_dir=complete, domain="ACL")
    with pytest.raises(UnresolvedRelevanceError):
        validate_qrels(queries, corpus)


def test_source_revision_mismatch(complete):
    corpus = load_corpus(local_dir=complete, revision="a" * 40, domain="ACL")
    queries = load_queries(local_dir=complete, revision="b" * 40, domain="ACL")
    with pytest.raises(ValueError, match="revision differs"):
        validate_qrels(queries, corpus)


def test_duplicate_and_malformed_papers(complete):
    folder = complete / "preprocessed/ACL/duplicate"
    shutil.copytree(complete / PAPER, folder)
    with pytest.raises(DatasetFormatError, match="duplicate"):
        load_corpus(local_dir=complete, domain="ACL")
    shutil.rmtree(folder)
    (complete / PAPER / "paper.json").write_text("{broken}")
    with pytest.raises(DatasetFormatError, match="invalid JSON"):
        load_corpus(local_dir=complete, domain="ACL", missing="skip")


def test_query_only_publication_no_fabricated_corpus(tmp_path):
    row = json.loads((Path(__file__).parent / "fixtures/query.json").read_text())
    write_json(tmp_path / QPATH, row)
    assert len(load_queries(local_dir=tmp_path, domain="ACL")) == 1
    with pytest.raises(MissingComponentError):
        load_corpus(local_dir=tmp_path, domain="ACL")
    with pytest.raises(MissingComponentError, match="No preprocessed"):
        load_corpus(local_dir=tmp_path, domain="ACL", split="all")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"domain": []},
        {"domain": "../ACL"},
        {"domain": "ACL", "split": "validation"},
        {"domain": "ACL", "representation": "invented"},
        {"domain": "ACL", "missing": "guess"},
    ],
)
def test_invalid_selections(complete, kwargs):
    with pytest.raises(ValueError):
        load_corpus(local_dir=complete, **kwargs)


def test_absent_selected_domain_is_reported(complete):
    with pytest.raises(IncompleteCorpusError) as error:
        load_corpus(local_dir=complete, domain=["ACL", "Physics"], split="all")
    assert "missing_domain" in {i.code for i in error.value.report.issues}


def test_split_and_query_domain_mismatch(complete):
    path = complete / PAPER / "paper.json"
    paper = json.loads(path.read_text())
    paper["split"] = "train"
    write_json(path, paper)
    with pytest.raises(DatasetFormatError, match="split disagrees"):
        load_corpus(local_dir=complete, domain="ACL")


def test_archive_images_not_read(complete, monkeypatch):
    image = complete / PAPER / "images/diagram.png"
    image.parent.mkdir()
    image.write_bytes(b"image-bytes")
    archive_extractions(complete)
    from scientific_tot.data.artifacts import Artifacts

    original = Artifacts.read_text

    def guarded(self, name):
        assert not name.endswith(".png")
        return original(self, name)

    monkeypatch.setattr(Artifacts, "read_text", guarded)
    corpus = load_corpus(local_dir=complete, domain="ACL")
    locator = corpus[0]["figures"][0]["image"]
    assert locator["available"] is True
    assert locator["archive"].endswith("shard-00000.tar")
    assert locator["member"].endswith("images/diagram.png")


def test_archive_shards_can_split_a_single_paper(complete):
    base = complete / "preprocessed/ACL"
    metadata, content = {}, {}
    for p in base.rglob("*"):
        if p.is_file():
            target = metadata if p.name == "paper.json" else content
            target[p.relative_to(base).as_posix()] = p.read_bytes()
    shutil.rmtree(base)
    make_tar(base / "shard-00000.tar", metadata)
    make_tar(base / "shard-00001.tar", content)
    assert len(load_corpus(local_dir=complete, domain="ACL")) == 2


def test_example_runs_offline(tmp_path):
    import os
    import subprocess
    import sys

    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "examples/convert_dataset.py"),
            "--local-dir",
            str(repo / "tests/fixtures/adapter"),
            "--domain",
            "ACL",
            "--output",
            str(tmp_path / "converted"),
        ],
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "HF_HUB_OFFLINE": "1"},
    )
    counts = json.loads(result.stdout)
    assert (counts["documents"], counts["queries"], counts["qrels"]) == (2, 1, 1)
