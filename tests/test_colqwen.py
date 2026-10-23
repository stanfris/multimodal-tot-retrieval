"""Real PDFium/Qdrant/aggregation lifecycle with offline mocked neural inference."""

import json
import pickle
import shutil
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pypdfium2 as pdfium
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scientific_tot.artifact_identity import artifact_directory, artifact_name, resolve_identity
from scientific_tot.config_arguments import configured_adapter
from scientific_tot.config_artifacts import artifact_reference, verify_index
from scientific_tot.configuration import CONFIG_DIR, resolve_config
from scientific_tot.hydra_run import run_stage, verify_completed
from scientific_tot.integrations import colqwen
from scientific_tot.records import IncompleteCorpusError, UnresolvedRelevanceError
from scientific_tot.recover_colqwen import recover_index

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
            vectors = [np.array([[image.width / 100, 1]], dtype=np.float32) for image in images]
            return [v / np.linalg.norm(v, axis=-1, keepdims=True) for v in vectors]

        def encode_query(self, queries, **kwargs):
            calls.append(("query", queries, kwargs))
            return [np.array([[1, 0]], dtype=np.float32) for _ in queries]

    monkeypatch.setattr(
        colqwen, "load_model", lambda cfg: (Model(), {"checkpoint_files": {"model": "stub"}})
    )
    monkeypatch.setattr(colqwen, "dependency_versions", lambda: {"neural_inference": "mocked"})

    def score(queries, pages, *, device, chunk_elements):
        from qdrant_client import models
        from qdrant_client.local.multi_distances import calculate_multi_distance

        assert device == "cuda"
        assert chunk_elements > 0
        return np.array(
            [
                calculate_multi_distance(query.astype(np.float64), pages, models.Distance.DOT)
                for query in queries
            ],
            dtype=np.float32,
        )

    monkeypatch.setattr(colqwen, "score_pages", score)
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
    issue = error.value.report.issues[-1]
    assert issue.path in str(error.value)
    assert issue.detail in str(error.value)
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
    assert [len(call[1]) for call in encoder] == [2, 2]
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
        "model.retrieval.scoring_batch_size=32",
        "model.retrieval.page_batch_size=64",
        "model.retrieval.scoring_chunk_elements=8000000",
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
    with pytest.raises(FileExistsError, match="checksum"):
        verify_completed(index, cfg, identity)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("scoring_batch_size", [1, 2, 16])
@pytest.mark.parametrize("page_batch_size", [1, 3])
@pytest.mark.parametrize("pool_factor", [1, 3])
@pytest.mark.parametrize("dimension", [2, 128])
def test_streamed_search_matches_local_qdrant(
    tmp_path,
    dataset,
    encoder,
    monkeypatch,
    batch_size,
    scoring_batch_size,
    page_batch_size,
    dimension,
    pool_factor,
):
    from contextlib import closing

    from qdrant_client import QdrantClient, models

    query_file = next(dataset.glob("query_collections/**/queries.jsonl"))
    query = json.loads(query_file.read_text())
    query_file.write_text(
        "".join(
            json.dumps({**query, "query_id": f"q{i}", "query": f"query {i}"}) + "\n"
            for i in range(3)
        )
    )
    documents = {
        200: [[-1.5, -0.2], [-2.0, -0.8]],
        202: [[-1.0, -0.5]],
        204: [[-3.0, -0.9], [-2.0, -0.1], [-1.0, -1.0]],
        100: [[-1.0, -0.5]],  # Tied paper maxima for query 0; all scores negative.
    }
    queries = [np.array(v, dtype=np.float32) for v in ([[1, 0]], [[0, 1], [1, 0]], [[-1, 1]])]
    if dimension == 128:
        rng = np.random.default_rng(17)
        documents = {
            width: rng.standard_normal((len(vectors) * 11, dimension)).astype(np.float32)
            for width, vectors in documents.items()
        }
        queries = [
            rng.standard_normal((len(v) * 17, dimension)).astype(np.float32) for v in queries
        ]
    encoding_batches = []

    class Model:
        def get_embedding_dimension(self):
            return dimension

        def encode_document(self, images, **kwargs):
            return [np.array(documents[image.width], dtype=np.float32) for image in images]

        def encode_query(self, texts, **kwargs):
            encoding_batches.append(list(texts))
            return [queries[int(text.split()[-1])] for text in texts]

    monkeypatch.setattr(colqwen, "load_model", lambda cfg: (Model(), {"checkpoint_files": {}}))
    index, cfg, _ = execute(
        tmp_path, dataset, "index", "index", f"model.indexing.pooling.factor={pool_factor}"
    )
    work = tmp_path / "reference-qdrant"
    shutil.copytree(index / "qdrant", work)
    for path in work.rglob("*"):
        if path.is_file():
            path.chmod(0o600)
    expected_scores = {}
    expected_rankings = {}
    with closing(QdrantClient(path=str(work))) as client:
        for i, vector in enumerate(queries):
            hits = client.query_points(
                "pages",
                query=vector.tolist(),
                limit=4,
                search_params=models.SearchParams(exact=True),
            ).points
            rows = []
            for hit in hits:
                expected_scores[(f"q{i}", hit.payload["page_id"])] = hit.score
                rows.append({"qid": f"q{i}", "docno": hit.payload["page_id"], "score": hit.score})
            ranking = colqwen.aggregate_pages(pd.DataFrame(rows), 2)
            expected_rankings[f"q{i}"] = list(zip(ranking.docno, ranking.score, strict=True))

    def blocked(*args, **kwargs):
        raise AssertionError("Retrieval must not load or copy the full Qdrant collection")

    monkeypatch.setattr("qdrant_client.QdrantClient", blocked)
    monkeypatch.setattr(shutil, "copytree", blocked)
    read_points = colqwen.iter_points
    scans = []

    def counted_scan(*args, **kwargs):
        scans.append(args[0])
        yield from read_points(*args, **kwargs)

    monkeypatch.setattr(colqwen, "iter_points", counted_scan)
    score_pages = colqwen.score_pages
    score_batches = []

    def counted_scores(queries, pages, **kwargs):
        score_batches.append((len(queries), len(pages)))
        assert all(vector.dtype == np.float32 for vector in [*queries, *pages])
        return score_pages(queries, pages, **kwargs)

    monkeypatch.setattr(colqwen, "score_pages", counted_scores)
    output, _, _ = execute(
        tmp_path,
        dataset,
        "retrieve",
        "run",
        f"paths.index={index}",
        f"model.retrieval.batch_size={batch_size}",
        f"model.retrieval.scoring_batch_size={scoring_batch_size}",
        f"model.retrieval.page_batch_size={page_batch_size}",
        f"model.indexing.pooling.factor={pool_factor}",
    )
    assert len(scans) == 1
    assert [text for batch in encoding_batches for text in batch] == [
        f"query {i}" for i in range(3)
    ]
    assert all(len(batch) <= batch_size for batch in encoding_batches)
    assert all(q <= scoring_batch_size and p <= page_batch_size for q, p in score_batches)
    assert sum(q * p for q, p in score_batches) == len(queries) * 4
    saved = json.loads((output / "manifest.json").read_text())
    assert saved["artifacts"]["scoring"] == {
        "device": "cuda",
        "dtype": "float32",
        "matmul_precision": "highest",
        "library": "sentence_transformers.util.maxsim",
    }
    scores = [json.loads(line) for line in (output / "page-scores.jsonl").read_text().splitlines()]
    assert len(scores) == len(expected_scores)
    assert {(row["qid"], row["page_id"]): row["score"] for row in scores} == pytest.approx(
        expected_scores, rel=2e-6, abs=2e-5
    )
    run = pd.read_csv(output / "run.trec", sep=r"\s+", header=None, float_precision="round_trip")
    for qid, group in run.groupby(0, sort=False):
        assert group[2].tolist() == [doc for doc, _ in expected_rankings[qid]]
        assert group[4].tolist() == pytest.approx(
            [score for _, score in expected_rankings[qid]], rel=2e-6, abs=2e-5
        )
    verify_index(index, cfg)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("chunk_elements", [1, 1000000])
@pytest.mark.parametrize("dimension", [2, 128])
def test_official_maxsim_device_budget_and_parity(monkeypatch, device, chunk_elements, dimension):
    import torch
    from qdrant_client import models
    from qdrant_client.local.multi_distances import calculate_multi_distance

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable; run on the GPU host to verify CUDA scoring")
    pytest.importorskip("sentence_transformers", minversion="6.1.0")
    queries = [
        np.array(v, dtype=np.float32)
        for v in (
            [[1, 0]],
            [[1, 0], [0, 1]],
            [[0, 0], [-1, 1], [1, 1]],
        )
    ]
    pages = [
        np.array(v, dtype=np.float32)
        for v in (
            [[-1, -2], [-2, -1], [-3, -3]],
            [[-0.5, -0.5]],
            [[0, 0], [-2, -2]],
        )
    ]
    if dimension == 128:
        rng = np.random.default_rng(17)
        queries = [
            rng.standard_normal((len(v) * 17, dimension)).astype(np.float32) for v in queries
        ]
        pages = [rng.standard_normal((len(v) * 13, dimension)).astype(np.float32) for v in pages]
    expected = np.array(
        [
            calculate_multi_distance(
                query.astype(np.float64),
                [page.astype(np.float64) for page in pages],
                models.Distance.DOT,
            )
            for query in queries
        ]
    )
    einsum = torch.einsum
    calls = []

    def inspect_einsum(equation, queries, pages):
        assert queries.device.type == pages.device.type == device
        assert queries.dtype == pages.dtype == torch.float32
        assert not torch.is_grad_enabled()
        calls.append(pages.shape[0])
        return einsum(equation, queries, pages)

    monkeypatch.setattr(torch, "einsum", inspect_einsum)
    scores = colqwen.score_pages(queries, pages, device=device, chunk_elements=chunk_elements)
    np.testing.assert_allclose(scores, expected, rtol=2e-6, atol=2e-5)
    assert scores.dtype == np.float32
    assert calls == ([1] * len(pages) if chunk_elements == 1 else [len(pages)])


def test_index_closes_images_when_rendering_fails(tmp_path, dataset, encoder, monkeypatch):
    render = colqwen.render_page
    images = []

    def fail_second_page(pdf, number, cfg):
        if number == 2:
            raise RuntimeError("Cannot render page")
        image = render(pdf, number, cfg)
        images.append(image)
        return image

    monkeypatch.setattr(colqwen, "render_page", fail_second_page)
    with pytest.raises(RuntimeError, match="Cannot render page"):
        execute(tmp_path, dataset, "index", "index")
    assert len(images) == 1
    with pytest.raises(ValueError, match="closed image"):
        images[0].getpixel((0, 0))


@pytest.mark.parametrize(
    "override",
    [
        "representation=fulltext",
        "representation=figures",
        "model.indexing.exact=false",
        "model.revision=main",
        "model.batch_size=0",
        "model.retrieval.scoring_batch_size=0",
        "model.retrieval.page_batch_size=0",
        "model.retrieval.scoring_chunk_elements=0",
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


def test_preparation_and_identity_do_not_render_pages(tmp_path, dataset, encoder, monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Pages must only be rendered for document encoding")

    monkeypatch.setattr(colqwen, "render_page", blocked)
    output, _, _ = execute(tmp_path, dataset, "prepare", "prepared")
    assert len((output / "page-mapping.jsonl").read_text().splitlines()) == 4
    cfg, _ = settings(tmp_path, dataset, "index")
    assert resolve_identity(cfg)["index"]["corpus_sha256"]
    assert encoder == []


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


def test_incomplete_index_without_snapshot_cannot_be_resumed(tmp_path, dataset, encoder):
    output = tmp_path / "unfinished"
    output.mkdir()
    (output / "manifest.json").write_text('{"status":"started"}')
    cfg, hydra = settings(tmp_path, dataset, "index", output)
    with pytest.raises(FileNotFoundError, match="config.yaml"):
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


def test_lean_pages_use_original_filename_format(tmp_path, dataset, encoder):
    # Reproduce the real restored tree: no paper.json, no metadata.parquet,
    # all-corpus selection, and canonical ACL_<original filename stem> IDs.
    for path in dataset.glob("preprocessed/ACL/*/paper.json"):
        (path.parent / "_SUCCESS").touch()
        path.unlink()
    cfg, _ = settings(tmp_path, dataset, "prepare")
    corpus = colqwen.load_pdf_corpus(configured_adapter(cfg), cfg)
    assert [row["source_pdf_relpath"] for row in corpus] == [
        "ACL/2022.example-1.1.pdf",
        "ACL/2022.example-1.2.pdf",
    ]
    assert [row["doc_id"] for row in corpus] == ["ACL_2022.example-1.1", "ACL_2022.example-1.2"]
    index, _, _ = execute(tmp_path, dataset, "index", "lean-index")
    output, _, _ = execute(tmp_path, dataset, "retrieve", "lean-run", f"paths.index={index}")
    assert len((output / "run.trec").read_text().splitlines()) == 2


@pytest.fixture
def interrupted(tmp_path, dataset, encoder, monkeypatch):
    cfg, _ = settings(tmp_path, dataset, "index")
    name = artifact_name(cfg, resolve_identity(cfg))

    def stop(*args):
        raise SystemExit("Interrupted before metadata publication")

    with monkeypatch.context() as patch:
        patch.setattr(colqwen, "publish_index_metadata", stop)
        with pytest.raises(SystemExit, match="Interrupted"):
            execute(tmp_path, dataset, "index", name)
    return tmp_path / name


def test_recover_completed_points(interrupted, encoder):
    database = interrupted / "qdrant/collection/pages/storage.sqlite"
    before = artifact_reference(database)
    calls = list(encoder)
    recover_index(interrupted)
    assert artifact_reference(database) == before
    assert encoder == calls  # No document or query encoding during recovery.
    manifest = json.loads((interrupted / "manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["recovery"]["original_manifest"]["status"] == "started"
    cfg = OmegaConf.load(interrupted / ".hydra/config.yaml")
    assert verify_index(interrupted, cfg)["provenance"]["pages"] == 4
    verify_completed(interrupted, cfg, resolve_identity(cfg))


@pytest.mark.parametrize("damage", ["missing", "payload", "vector"])
def test_recovery_rejects_invalid_points(interrupted, damage):
    database = interrupted / "qdrant/collection/pages/storage.sqlite"
    with sqlite3.connect(database) as db:
        key, blob = db.execute("SELECT id, point FROM points LIMIT 1").fetchone()
        if damage == "missing":
            db.execute("DELETE FROM points WHERE id = ?", (key,))
        else:
            point = pickle.loads(blob)
            if damage == "payload":
                point.payload["doc_id"] = "wrong-paper"
            else:
                point.vector = [[float("nan"), 0.0]]
            db.execute("UPDATE points SET point = ? WHERE id = ?", (pickle.dumps(point), key))
    with pytest.raises(ValueError, match="count differs|payload differs|Invalid page vectors"):
        recover_index(interrupted)
    assert not (interrupted / "index-manifest.json").exists()
    assert json.loads((interrupted / "manifest.json").read_text())["status"] == "started"


def test_recovery_rejects_changed_pdf(interrupted, dataset):
    pdf = next((dataset / "pdf_datasets").rglob("*.pdf"))
    with pdf.open("ab") as stream:
        stream.write(b"\n% changed after indexing\n")
    with pytest.raises(ValueError, match="original index hash"):
        recover_index(interrupted)
    assert not (interrupted / "index-manifest.json").exists()


def test_recovery_rejects_active_qdrant(interrupted):
    import portalocker

    with (interrupted / "qdrant/.lock").open("r+") as lock:
        portalocker.lock(lock, portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING)
        with pytest.raises(portalocker.exceptions.LockException):
            recover_index(interrupted)
    assert not (interrupted / "index-manifest.json").exists()


@pytest.mark.parametrize("committed", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("failure", [SystemExit, RuntimeError])
def test_index_command_resumes_committed_pages(
    tmp_path, dataset, encoder, monkeypatch, committed, failure
):
    from qdrant_client.local.persistence import CollectionPersistence

    cfg, _ = settings(tmp_path, dataset, "index")
    name = artifact_name(cfg, resolve_identity(cfg))
    original_persist = CollectionPersistence.persist

    def interrupt(store, point):
        if committed == 0:
            raise failure("Interrupted during encoding")
        original_persist(store, point)
        if point.id + 1 == committed:
            raise failure("Interrupted during encoding")

    with monkeypatch.context() as patch:
        patch.setattr(CollectionPersistence, "persist", interrupt)
        with pytest.raises(failure, match="Interrupted"):
            execute(tmp_path, dataset, "index", name)

    output = tmp_path / name
    database = output / "qdrant/collection/pages/storage.sqlite"
    with sqlite3.connect(database) as db:
        before = dict(db.execute("SELECT id, point FROM points"))
    assert len(before) == committed
    cfg, hydra = settings(tmp_path, dataset, "index", output)
    encoder.clear()

    def no_local_client(*args, **kwargs):
        raise AssertionError("Resume must not load the existing collection into RAM")

    with monkeypatch.context() as patch:
        patch.setattr("qdrant_client.QdrantClient", no_local_client)
        run_stage(cfg, output, hydra, ROOT)
        # A second invocation must reuse the completed index without encoding.
        calls = list(encoder)
        run_stage(cfg, output, hydra, ROOT)
        assert encoder == calls

    assert sum(len(call[1]) for call in encoder) == 4 - committed
    with sqlite3.connect(database) as db:
        after = dict(db.execute("SELECT id, point FROM points"))
    assert len(after) == 4
    assert all(after[key] == blob for key, blob in before.items())
    verify_completed(output, cfg, resolve_identity(cfg))
    run, _, _ = execute(tmp_path, dataset, "retrieve", "resumed-run", f"paths.index={output}")
    assert len((run / "run.trec").read_text().splitlines()) == 2


def test_index_command_rejects_gap_in_saved_pages(interrupted, dataset, tmp_path, encoder):
    from qdrant_client.local.persistence import CollectionPersistence

    database = interrupted / "qdrant/collection/pages/storage.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("DELETE FROM points WHERE id = ?", (CollectionPersistence.encode_key(1),))
    before = artifact_reference(database)
    cfg, hydra = settings(tmp_path, dataset, "index", interrupted)
    encoder.clear()
    with pytest.raises(ValueError, match="Invalid or duplicate page position"):
        run_stage(cfg, interrupted, hydra, ROOT)
    assert encoder == []
    assert artifact_reference(database) == before


def test_index_command_finishes_interrupted_sealing(
    interrupted, dataset, tmp_path, encoder, monkeypatch
):
    cfg, hydra = settings(tmp_path, dataset, "index", interrupted)

    def stop(*args, **kwargs):
        raise SystemExit("Interrupted during output verification")

    with monkeypatch.context() as patch:
        patch.setattr("scientific_tot.recover_colqwen.verify_index", stop)
        with pytest.raises(SystemExit, match="output verification"):
            run_stage(cfg, interrupted, hydra, ROOT)
    assert (interrupted / "index-manifest.json").is_file()
    encoder.clear()

    def no_model(*args, **kwargs):
        raise AssertionError("Completed backend metadata needs no model reload")

    monkeypatch.setattr(colqwen, "load_model", no_model)
    run_stage(cfg, interrupted, hydra, ROOT)
    assert encoder == []
    verify_completed(interrupted, cfg, resolve_identity(cfg))


def test_index_command_rejects_active_run(interrupted, dataset, tmp_path, encoder):
    import portalocker

    cfg, hydra = settings(tmp_path, dataset, "index", interrupted)
    with portalocker.Lock(interrupted / ".run.lock", mode="a+", timeout=0):
        with pytest.raises(portalocker.exceptions.LockException):
            run_stage(cfg, interrupted, hydra, ROOT)


def test_index_command_resumes_before_collection_creation(tmp_path, dataset, encoder, monkeypatch):
    cfg, _ = settings(tmp_path, dataset, "index")
    name = artifact_name(cfg, resolve_identity(cfg))

    def stop(*args, **kwargs):
        raise SystemExit("Interrupted before collection creation")

    with monkeypatch.context() as patch:
        patch.setattr(colqwen, "load_model", stop)
        with pytest.raises(SystemExit, match="before collection creation"):
            execute(tmp_path, dataset, "index", name)
    output = tmp_path / name
    assert not (output / "qdrant").exists()
    cfg, hydra = settings(tmp_path, dataset, "index", output)
    run_stage(cfg, output, hydra, ROOT)
    assert sum(len(call[1]) for call in encoder) == 4
    verify_completed(output, cfg, resolve_identity(cfg))


def test_index_command_rejects_changed_settings(interrupted, dataset, tmp_path, encoder):
    cfg, hydra = settings(tmp_path, dataset, "index", interrupted, "model.batch_size=3")
    encoder.clear()
    with pytest.raises(ValueError, match="settings differ"):
        run_stage(cfg, interrupted, hydra, ROOT)
    assert encoder == []


@pytest.mark.parametrize("factor", [1, 3])
def test_hierarchical_pooling(tmp_path, dataset, encoder, monkeypatch, factor):
    # Two unit directions, each repeated three times: pooling should preserve both.
    vectors = np.array([[0.6, 0.8]] * 3 + [[-0.8, 0.6]] * 3, dtype=np.float32)

    class Model:
        def get_embedding_dimension(self):
            return 2

        def encode_document(self, images, **kwargs):
            return [vectors.copy() for _ in images]

    monkeypatch.setattr(colqwen, "load_model", lambda cfg: (Model(), {"checkpoint_files": {}}))
    index, cfg, _ = execute(
        tmp_path, dataset, "index", "pooled", f"model.indexing.pooling.factor={factor}"
    )
    mapping = [json.loads(line) for line in (index / "page-mapping.jsonl").read_text().splitlines()]
    pages = list(colqwen.iter_points(index / "qdrant/collection/pages/storage.sqlite", mapping, 2))
    assert len(pages) == 4
    for _, page in pages:
        assert page.dtype == np.float32
        assert page.shape == (6 // factor, 2)
        if factor == 1:
            np.testing.assert_array_equal(page, vectors)
        else:
            np.testing.assert_allclose(
                page[np.argsort(page[:, 0])], [[-0.8, 0.6], [0.6, 0.8]], atol=1e-7
            )
            np.testing.assert_allclose(np.linalg.norm(page, axis=-1), 1, atol=1e-7)
    provenance = verify_index(index, cfg)["provenance"]
    assert provenance["compression"]["factor"] == factor


@pytest.mark.parametrize(
    "override",
    [
        "model.indexing.pooling.factor=0",
        "model.indexing.pooling.factor=1.5",
        "model.indexing.pooling.factor=true",
        "model.indexing.pooling.method=mean",
    ],
)
def test_invalid_pooling_settings(tmp_path, dataset, override):
    with pytest.raises(ValueError):
        settings(tmp_path, dataset, "index", None, override)


def test_pooling_changes_index_identity(tmp_path, dataset, encoder):
    index, cfg, _ = execute(tmp_path, dataset, "index", "index")
    changed, _ = settings(tmp_path, dataset, "index", None, "model.indexing.pooling.factor=1")
    assert resolve_identity(changed) != resolve_identity(cfg)
    with pytest.raises(ValueError, match="configuration differs"):
        verify_index(index, changed)
