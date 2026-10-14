"""Real PDFium/Qdrant/aggregation lifecycle with offline mocked neural inference."""

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scientific_tot.artifact_identity import artifact_directory, resolve_identity
from scientific_tot.config_arguments import configured_adapter
from scientific_tot.config_artifacts import artifact_reference, verify_index
from scientific_tot.configuration import CONFIG_DIR, resolve_config
from scientific_tot.hydra_run import run_stage
from scientific_tot.integrations import colqwen
from scientific_tot.records import IncompleteCorpusError, UnresolvedRelevanceError

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/adapter"


def settings(base, dataset, stage, output=None, *overrides):
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        raw = compose(
            config_name="config",
            return_hydra_config=True,
            overrides=[
                "experiment=colqwen_pages",
                f"stage={stage}",
                f"dataset.root={dataset}",
                "retrieval.top_k=2",
                "evaluation.metrics=[RR@2,R@2]",
                *overrides,
            ],
        )
    cfg = resolve_config(OmegaConf.masked_copy(raw, [k for k in raw if k != "hydra"]), base, output)
    return cfg, raw.hydra


def execute(base, dataset, stage, name, *overrides):
    output = base / name
    cfg, hydra = settings(base, dataset, stage, output, *overrides)
    output.mkdir()
    run_stage(cfg, output, hydra, ROOT)
    return output, cfg, hydra


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data"
    shutil.copytree(FIXTURE, root)
    for ordinal, path in enumerate(sorted(root.glob("preprocessed/ACL/*/paper.json"))):
        row = json.loads(path.read_text())
        target = root / "pdf_datasets" / row["source_pdf_relpath"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with pdfium.PdfDocument.new() as pdf:
            # First three highest-ranked pages belong to the same paper.
            for number in range(3 if ordinal == 0 else 1):
                pdf.new_page(100 + number if ordinal == 0 else 50, 80).close()
            pdf.save(target)
        # Page selection must work without extracted Markdown.
        (path.parent / "markdown.md").unlink()
    return root


@pytest.fixture
def encoder(monkeypatch):
    calls = []

    class Model:
        def get_embedding_dimension(self):
            return 2

        def encode_document(self, images, **kwargs):
            calls.append(("document", [image.size for image in images], kwargs))
            return [np.array([[image.width / 100, 0]], dtype=np.float32) for image in images]

        def encode_query(self, queries, **kwargs):
            calls.append(("query", queries, kwargs))
            return [np.array([[1, 0]], dtype=np.float32) for _ in queries]

    monkeypatch.setattr(
        colqwen, "load_model", lambda cfg: (Model(), {"checkpoint_files": {"model": "stub"}})
    )
    monkeypatch.setattr(colqwen, "dependency_versions", lambda: {"neural_inference": "mocked"})
    return calls


def test_pdf_preparation_and_canonical_mapping(tmp_path, dataset, encoder):
    cfg, _ = settings(tmp_path, dataset, "prepare")
    adapter = configured_adapter(cfg)
    corpus = colqwen.load_pdf_corpus(adapter, cfg)
    assert [row["page_count"] for row in corpus] == [3, 1]
    assert adapter.resolve_pdf(corpus[0]) == Path(corpus[0]["pdf_path"])
    mapping = colqwen.page_mapping(corpus)
    assert [r["page_number"] for r in mapping] == [1, 2, 3, 1]
    assert len({r["page_id"] for r in mapping}) == 4
    assert colqwen.page_id("a/%p/日本語:paper", 12) == colqwen.page_id("a/%p/日本語:paper", 12)
    assert colqwen.page_id("a/b", 1) != colqwen.page_id("a%2Fb", 1)
    with pdfium.PdfDocument(corpus[0]["pdf_path"]) as pdf:
        images = [colqwen.render_page(pdf, number, cfg) for number in range(1, 4)]
        assert [image.size for image in images] == [(200, 160), (202, 160), (204, 160)]
        for image in images:
            image.close()
    output, _, _ = execute(tmp_path, dataset, "prepare", "prepared")
    assert encoder == []
    assert len((output / "page-mapping.jsonl").read_text().splitlines()) == 4


@pytest.mark.parametrize("failure", ["missing", "corrupt"])
def test_missing_policy_never_hides_positive(tmp_path, dataset, failure):
    path = next(dataset.glob("pdf_datasets/ACL/*1.1.pdf"))
    if failure == "missing":
        path.unlink()
    else:
        path.write_bytes(b"corrupt PDF")
    cfg, _ = settings(tmp_path, dataset, "prepare")
    with pytest.raises(IncompleteCorpusError) as error:
        colqwen.load_pdf_corpus(configured_adapter(cfg), cfg)
    assert error.value.report.skipped == 1
    cfg, _ = settings(tmp_path, dataset, "prepare", None, "dataset.missing=skip")
    corpus = colqwen.load_pdf_corpus(configured_adapter(cfg), cfg)
    assert len(corpus) == 1
    assert corpus.report.issues[-1].code == "unusable_pdf"
    with pytest.raises(UnresolvedRelevanceError):
        execute(tmp_path, dataset, "prepare", "excluded", "dataset.missing=skip")


def test_exact_paper_cutoff_max_and_ties():
    pages = pd.DataFrame(
        [
            {"qid": "q", "docno": colqwen.page_id(doc, number), "score": score}
            for doc, number, score in [
                ("a/%p", 1, 9),
                ("a/%p", 2, 8),
                ("a/%p", 3, 7),
                ("b", 1, 6),
                ("c", 1, 6),
            ]
        ]
    )
    run = colqwen.aggregate_pages(pages, 2)
    assert run.docno.tolist() == ["a/%p", "b"]
    assert run.score.tolist() == [9, 6]
    assert run["rank"].tolist() == [1, 2]
    assert colqwen.aggregate_pages(pages, 100).docno.tolist() == ["a/%p", "b", "c"]


def test_independent_lifecycle_identity_reuse_and_integrity(
    tmp_path, dataset, encoder, monkeypatch
):
    index, cfg, hydra = execute(tmp_path, dataset, "index", "index")
    assert all(call[0] == "document" for call in encoder)
    assert [len(call[1]) for call in encoder] == [2, 1, 1]
    assert encoder[0][2] == {"batch_size": 2, "convert_to_numpy": True}
    manifest = verify_index(index, cfg)
    assert manifest["provenance"]["pages"] == 4
    assert manifest["provenance"]["documents"] == 2
    assert manifest["provenance"]["effective_model"]["checkpoint_files"] == {"model": "stub"}
    identity = resolve_identity(cfg)
    changed, _ = settings(
        tmp_path,
        dataset,
        "index",
        index,
        "split.queries=train",
        "retrieval.top_k=10",
        "model.retrieval.batch_size=7",
    )
    assert resolve_identity(changed) == identity
    default, _ = settings(tmp_path, dataset, "index")
    assert artifact_directory(default, identity).name.startswith("colqwen_pages_d-ACL_c-all_")
    assert "q-test" not in artifact_directory(default, identity).name
    before = artifact_reference(index / "index-manifest.json")
    run_stage(changed, index, hydra, ROOT)
    assert artifact_reference(index / "index-manifest.json") == before
    changed, _ = settings(tmp_path, dataset, "index", index, "representation.rendering.scale=1")
    assert resolve_identity(changed) != identity
    with pytest.raises(FileExistsError, match="different identity"):
        run_stage(changed, index, hydra, ROOT)

    def blocked(*args, **kwargs):
        raise AssertionError("Independent stages must not access originals")

    monkeypatch.setattr(colqwen, "load_pdf_corpus", blocked)
    monkeypatch.setattr(colqwen, "render_page", blocked)
    monkeypatch.setattr("scientific_tot.adapter.DatasetAdapter.load_corpus", blocked)
    encoder.clear()
    run, _, _ = execute(tmp_path, dataset, "retrieve", "ranked", f"paths.index={index}")
    assert [call[0] for call in encoder] == ["query"]
    lines = (run / "run.trec").read_text().splitlines()
    assert len(lines) == 2
    assert [line.split()[2] for line in lines] == ["ACL_2022.example-1.1", "ACL_2022.example-1.2"]
    assert len((run / "page-scores.jsonl").read_text().splitlines()) == 4
    verify_index(index, cfg)  # Searching did not change the sealed SQLite index.
    monkeypatch.setattr(colqwen, "load_model", blocked)
    shutil.rmtree(index)
    shutil.rmtree(dataset)
    output, _, _ = execute(tmp_path, dataset, "evaluate", "evaluated", f"paths.run={run}")
    assert (output / "metrics.csv").is_file()
    assert (output / "per-query.csv").is_file()


def test_pdf_bytes_and_corruption_change_identity(tmp_path, dataset, encoder):
    index, cfg, _ = execute(tmp_path, dataset, "index", "index")
    identity = resolve_identity(cfg)
    path = next(dataset.glob("pdf_datasets/ACL/*.pdf"))
    with path.open("ab") as stream:
        stream.write(b"\n% content identity changed\n")
    assert resolve_identity(cfg) != identity
    manifest = index / "page-mapping.jsonl"
    manifest.chmod(0o644)
    manifest.write_text("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        verify_index(index, cfg)


@pytest.mark.parametrize(
    "override",
    [
        "representation=fulltext",
        "representation=figures",
        "model.indexing.exact=false",
        "model.revision=main",
        "model.batch_size=0",
        "model.processor=custom",
    ],
)
def test_invalid_configuration(tmp_path, dataset, override):
    with pytest.raises(ValueError):
        settings(tmp_path, dataset, "prepare", None, override)


@pytest.mark.colqwen_live
def test_real_checkpoint(tmp_path, dataset):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("Real checkpoint test requires CUDA and enough memory for the 3.8B model")
    cfg, _ = settings(tmp_path, dataset, "index", tmp_path / "live", "launcher=local_gpu")
    model, _ = colqwen.load_model(cfg)
    with pdfium.PdfDocument(next(dataset.glob("pdf_datasets/ACL/*.pdf"))) as pdf:
        image = colqwen.render_page(pdf, 1, cfg)
        try:
            documents = model.encode_document([image], convert_to_numpy=True)
        finally:
            image.close()
    queries = model.encode_query(["What is shown on this page?"], convert_to_numpy=True)
    scores = model.similarity(queries, documents)
    assert tuple(scores.shape) == (1, 1)
    assert torch.isfinite(scores).all()
    # Verify Qdrant's DOT/MAX_SIM matches the official score on actual embeddings.
    from qdrant_client import QdrantClient, models

    with QdrantClient(path=str(tmp_path / "qdrant-live")) as client:
        client.create_collection(
            "pages",
            vectors_config=models.VectorParams(
                size=documents[0].shape[1],
                distance=models.Distance.DOT,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        )
        client.upsert("pages", points=[models.PointStruct(id=0, vector=documents[0].tolist())])
        hit = client.query_points(
            "pages", query=queries[0].tolist(), search_params=models.SearchParams(exact=True)
        ).points[0]
        assert hit.score == pytest.approx(scores.item(), rel=1e-4)


def test_canonical_ids_are_not_filesystem_paths(tmp_path, dataset, encoder):
    canonical = "paper/%p:日本語"
    path = next(dataset.glob("preprocessed/ACL/*1.1/paper.json"))
    row = json.loads(path.read_text())
    original = row["paper_id"]
    row["paper_id"] = canonical
    path.write_text(json.dumps(row))
    path = next(dataset.glob("query_collections/**/queries.jsonl"))
    query = json.loads(path.read_text())
    query["relevant_ids"] = [canonical if doc == original else doc for doc in query["relevant_ids"]]
    query["metadata"]["paper_id"] = canonical
    path.write_text(json.dumps(query) + "\n")
    index, _, _ = execute(tmp_path, dataset, "index", "index")
    output, _, _ = execute(tmp_path, dataset, "retrieve", "run", f"paths.index={index}")
    assert (output / "run.trec").read_text().splitlines()[0].split()[2] == canonical


def test_pdf_path_safety_is_not_a_skip_policy(tmp_path, dataset):
    path = next(dataset.glob("pdf_datasets/ACL/*.pdf"))
    external = tmp_path / "outside.pdf"
    shutil.copyfile(path, external)
    path.unlink()
    path.symlink_to(external)
    cfg, _ = settings(tmp_path, dataset, "prepare", None, "dataset.missing=skip")
    with pytest.raises(ValueError, match="escapes dataset root"):
        colqwen.load_pdf_corpus(configured_adapter(cfg), cfg)


def test_failure_on_later_page_excludes_entire_paper(tmp_path, dataset, monkeypatch):
    render = colqwen.render_page

    def fail(pdf, number, cfg):
        if number == 2:
            raise pdfium.PdfiumError("Unreadable second page")
        return render(pdf, number, cfg)

    monkeypatch.setattr(colqwen, "render_page", fail)
    cfg, _ = settings(tmp_path, dataset, "prepare", None, "dataset.missing=skip")
    corpus = colqwen.load_pdf_corpus(configured_adapter(cfg), cfg)
    assert len(corpus) == 1
    assert len(colqwen.page_mapping(corpus)) == 1
    assert "second page" in corpus.report.issues[-1].detail


def test_model_loader_pins_base_and_preserves_official_processor(tmp_path, dataset, monkeypatch):
    import sys
    from types import SimpleNamespace

    adapter = tmp_path / "adapter"
    base = tmp_path / "base"
    adapter.mkdir()
    base.mkdir()
    config = {"base_model_name_or_path": "vidore/colqwen2.5-base", "revision": None}
    (adapter / "adapter_config.json").write_text(json.dumps(config))
    (adapter / "config_sentence_transformers.json").write_text(
        '{"model_type":"MultiVectorEncoder"}'
    )
    (adapter / "modules.json").write_text("[]")
    (adapter / "preprocessor_config.json").write_text('{"official":true}')
    calls = []

    def snapshot(name, **kwargs):
        calls.append((name, kwargs))
        return str(adapter if name.endswith("v0.2") else base)

    class Model:
        def __init__(self, folder, **kwargs):
            calls.append(("encoder", kwargs))
            resolved = json.loads((Path(folder) / "adapter_config.json").read_text())
            assert resolved == {**config, "base_model_name_or_path": str(base)}
            assert (Path(folder) / "preprocessor_config.json").read_text() == '{"official":true}'
            self.prompts = {"query": "", "document": ""}
            self.similarity_fn_name = "maxsim"

        def __getitem__(self, index):
            return SimpleNamespace(
                processor=SimpleNamespace(
                    chat_template="official",
                    to_dict=lambda: {"official": True},
                    image_processor=SimpleNamespace(to_dict=lambda: {"official": True}),
                ),
                get_config_dict=lambda: {"processing_kwargs": "official"},
                model=SimpleNamespace(config=SimpleNamespace(to_dict=lambda: {"pinned": True})),
            )

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    monkeypatch.setitem(
        sys.modules, "sentence_transformers", SimpleNamespace(MultiVectorEncoder=Model)
    )
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    cfg, _ = settings(tmp_path, dataset, "prepare")
    _, effective = colqwen.load_model(cfg)
    assert calls[0][1]["revision"] == cfg.model.revision
    assert calls[1][1]["revision"] == cfg.model.base_revision
    assert calls[2] == ("encoder", {"device": "cuda", "local_files_only": True})
    assert effective["processor"] == {"official": True}
    assert json.loads((adapter / "adapter_config.json").read_text()) == config
    (adapter / "modules.json").unlink()
    with pytest.raises(ValueError, match="official MultiVectorEncoder conversion"):
        colqwen.load_model(cfg)


def test_incomplete_index_cannot_be_reused(tmp_path, dataset, encoder):
    output = tmp_path / "unfinished"
    output.mkdir()
    (output / "manifest.json").write_text('{"status":"started"}')
    cfg, hydra = settings(tmp_path, dataset, "index", output)
    with pytest.raises(FileExistsError, match="incomplete"):
        run_stage(cfg, output, hydra, ROOT)
    assert encoder == []


def test_automatic_index_selection_needs_no_pdfs(tmp_path, dataset, encoder, monkeypatch):
    from scientific_tot.stage_cli import find_artifact

    directory = tmp_path / "results/indexes"
    directory.mkdir(parents=True)
    index, _, _ = execute(tmp_path, dataset, "index", "results/indexes/visual")

    def blocked(*args, **kwargs):
        raise AssertionError("Retrieval artifact selection must not access PDFs or models")

    monkeypatch.setattr(colqwen, "load_pdf_corpus", blocked)
    monkeypatch.setattr(colqwen, "load_model", blocked)
    shutil.rmtree(dataset)
    cfg, _ = settings(tmp_path, dataset, "retrieve", None, "paths.index=placeholder")
    assert find_artifact(cfg, "index") == index


@pytest.mark.parametrize(
    "override", ["launcher=local_cpu", "launcher.device=cpu", "launcher.device=mps"]
)
def test_colqwen_requires_cuda_configuration(tmp_path, dataset, override):
    with pytest.raises(ValueError, match="requires launcher.device=cuda"):
        settings(tmp_path, dataset, "index", None, override)


def test_colqwen_defaults_to_cuda_and_fails_without_gpu(tmp_path, dataset, monkeypatch):
    cfg, _ = settings(tmp_path, dataset, "prepare")
    assert cfg.model.device == cfg.launcher.device == "cuda"
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    def blocked(*args, **kwargs):
        raise AssertionError("Unavailable CUDA must fail before downloading weights")

    monkeypatch.setattr("huggingface_hub.snapshot_download", blocked)
    with pytest.raises(RuntimeError, match="available NVIDIA GPU"):
        colqwen.load_model(cfg)
