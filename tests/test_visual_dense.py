"""Real PDFium/FAISS/stage lifecycle; pretrained inference is opt-in."""

import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from test_colqwen import dataset as _pdf_dataset
from test_colqwen import execute, settings

from scientific_tot.artifact_identity import resolve_identity
from scientific_tot.config_arguments import configured_adapter
from scientific_tot.config_artifacts import verify_retrieval
from scientific_tot.configuration import OPENCLIP_MODELS
from scientific_tot.integrations import colqwen, visual_dense
from scientific_tot.integrations.visual_dense import OpenCLIPEncoder, QwenEmbeddingEncoder
from scientific_tot.stage_cli import find_artifact

pdf_dataset = _pdf_dataset
QWEN_MODELS = ["qwen3_vl_text", "qwen3_vl_multimodal", "qwen3_vl_pages", "qwen3_vl_figures"]
MODELS = ["openclip", "openclip_figures", *QWEN_MODELS]


def overrides(model):
    experiment = "openclip_pages" if model == "openclip" else model
    return (f"experiment={experiment}", "launcher=local_cpu", "model.precision=fp32")


@pytest.mark.parametrize(
    "model,change_source",
    [("openclip_figures", False), ("qwen3_vl_pages", False), ("qwen3_vl_pages", True)],
)
def test_actual_hydra_reuses_startup_inputs_and_rechecks_sources(
    tmp_path, visual_dataset, model, change_source
):
    script = """
import json, os, socket
from pathlib import Path
import torch
from scientific_tot.data.hub import Dataset
from scientific_tot.integrations import visual_dense
from scientific_tot.hydra_entry import main

counts = {"inventories": 0, "corpus_loads": 0, "encoded_units": 0}
discover = Dataset.discover
load_corpus = visual_dense.load_visual_corpus
def counted_discover(self):
    if self._inventory is None:
        counts["inventories"] += 1
    return discover(self)
def counted_corpus(*args, **kwargs):
    counts["corpus_loads"] += 1
    return load_corpus(*args, **kwargs)
class Encoder:
    metadata = {}
    def __init__(self, cfg):
        self.dimension = cfg.model.dimension
    def encode(self, values, task):
        assert task == "document"
        counts["encoded_units"] += len(values)
        if os.environ["TOT_REGRESSION_MUTATE"] == "1":
            path = next(Path(os.environ["TOT_REGRESSION_DATASET"]).glob("pdf_datasets/ACL/*.pdf"))
            with path.open("ab") as stream:
                stream.write(b"\\n% changed PDF input\\n")
        return torch.ones(len(values), self.dimension)
def blocked(*args, **kwargs):
    raise AssertionError("Startup reuse regression must not access the network")
Dataset.discover = counted_discover
visual_dense.load_visual_corpus = counted_corpus
visual_dense.load_model = Encoder
socket.socket.connect = blocked
socket.create_connection = blocked
try:
    main()
finally:
    Path("input-counts.json").write_text(json.dumps(counts))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            *overrides(model),
            "stage=index",
            f"dataset.root={visual_dataset}",
            f"launcher.output_root={tmp_path / 'results'}",
            f"launcher.cache_root={tmp_path / 'cache'}",
        ],
        cwd=tmp_path,
        env=dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
            TOT_REGRESSION_DATASET=str(visual_dataset),
            TOT_REGRESSION_MUTATE=str(int(change_source)),
            HYDRA_FULL_ERROR="1",
        ),
        capture_output=True,
        text=True,
        timeout=60,
    )
    counts = json.loads((tmp_path / "input-counts.json").read_text())
    assert counts == {
        "inventories": 2,
        "corpus_loads": 2,
        "encoded_units": 4 if model in OPENCLIP_MODELS else 2,
    }
    assert result.stdout.count("▶ Startup: identify") == 1
    manifest_path = next((tmp_path / "results/indexes").glob("*/manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    if change_source:
        assert result.returncode != 0
        assert "Source inputs changed during artifact construction" in result.stderr
        assert manifest["status"] == "failed"
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert manifest["status"] == "completed"


@pytest.fixture
def visual_dataset(pdf_dataset):
    for ordinal, path in enumerate(sorted(pdf_dataset.glob("preprocessed/ACL/*/paper.json"))):
        paper = json.loads(path.read_text())
        figures = []
        count = 3 if ordinal == 0 else 1
        for number in range(count):
            relative = f"images/visual-{number}.png"
            target = path.parent / relative
            target.parent.mkdir(exist_ok=True)
            Image.new("RGB", (200 + number if ordinal == 0 else 100, 160)).save(target)
            figures.append(
                {
                    "figure_id": f"{paper['paper_id']}.fig.{number}",
                    "image_relpath": relative,
                    "page_idx": number,
                    "type": "table" if number == 1 else "image",
                    "caption": ["Never an explicit model input in figures mode"],
                }
            )
        paper["figures"] = figures
        (path.parent / "markdown.md").write_text(
            f"Full original paper {ordinal}.\nUnmodified text."
        )
        path.write_text(json.dumps(paper))
    return pdf_dataset


@pytest.fixture
def encoder(monkeypatch):
    calls = []

    class Encoder:
        metadata = {"inference": "mocked"}
        truncated = 0

        def __init__(self, cfg):
            self.dimension = cfg.model.dimension

        def encode(self, values, task):
            import torch

            calls.append(
                (
                    task,
                    values
                    if task == "query"
                    else [
                        {
                            **{k: v for k, v in value.items() if k not in {"image", "images"}},
                            **({"image": value["image"].size} if "image" in value else {}),
                            **(
                                {"images": [image.size for image in value["images"]]}
                                if "images" in value
                                else {}
                            ),
                        }
                        for value in values
                    ],
                )
            )
            result = torch.zeros(len(values), self.dimension)
            result[:, 0] = 1
            if task == "document":
                result[:, 1] = torch.tensor(
                    [
                        1 / value["image"].width if "image" in value else 1 / (i + 2)
                        for i, value in enumerate(values)
                    ]
                )
            return result

    monkeypatch.setattr(visual_dense, "load_model", Encoder)
    return calls


@pytest.mark.parametrize("model", MODELS)
def test_visual_stages(tmp_path, visual_dataset, monkeypatch, encoder, model):
    args = overrides(model)
    prepared, _, _ = execute(tmp_path, visual_dataset, "prepare", "prepared", *args)
    units = [
        json.loads(line) for line in (prepared / "unit-mapping.jsonl").read_text().splitlines()
    ]
    count = 4 if model in OPENCLIP_MODELS else 2
    assert len(units) == count and not encoder
    if model in {"qwen3_vl_text", "qwen3_vl_multimodal"}:
        assert [unit["text"] for unit in units] == [
            f"Full original paper {i}.\nUnmodified text." for i in range(2)
        ]
    if model in QWEN_MODELS:
        assert [unit["docno"] for unit in units] == [unit["doc_id"] for unit in units]
        assert [unit["page_count"] for unit in units] == (
            [3, 1] if model == "qwen3_vl_pages" else [0, 0]
        )
    (tmp_path / "results/indexes").mkdir(parents=True)
    index, _, _ = execute(tmp_path, visual_dataset, "index", "results/indexes/first", *args)
    provenance = json.loads((index / "index-manifest.json").read_text())["provenance"]
    assert provenance["vectors"] == count and provenance["documents"] == 2
    if model in QWEN_MODELS:
        assert provenance["encoding"]["rendered_pages"] == (4 if model == "qwen3_vl_pages" else 0)
        assert provenance["encoding"]["supplied_images"] == (
            4 if model in {"qwen3_vl_figures", "qwen3_vl_multimodal"} else 0
        )
    assert all(task == "document" for task, _ in encoder)
    values = [value for _, batch in encoder for value in batch]
    assert all(
        ("text" in value) == (model in {"qwen3_vl_text", "qwen3_vl_multimodal"}) for value in values
    )
    if model in OPENCLIP_MODELS:
        widths = [200, 201, 202, 100] if model.endswith("figures") else [200, 202, 204, 100]
        assert [value["image"] for value in values] == [(width, 160) for width in widths]
    else:
        assert [value["images"] for value in values] == (
            [[], []]
            if model == "qwen3_vl_text"
            else [[(200, 160), (202, 160), (204, 160)], [(100, 160)]]
            if model == "qwen3_vl_pages"
            else [[(200, 160), (201, 160), (202, 160)], [(100, 160)]]
        )
    vectors = np.load(Path(provenance["cache"]["documents"]) / "embeddings.npy")
    assert vectors.shape == (count, 512 if model in OPENCLIP_MODELS else 2048)
    assert np.isfinite(vectors).all() and np.allclose(np.linalg.norm(vectors, axis=1), 1)
    encoder.clear()
    execute(tmp_path, visual_dataset, "index", "results/indexes/second", *args)
    assert not encoder

    def blocked(*args):
        raise AssertionError("Retrieval/evaluation must not reopen corpus assets")

    monkeypatch.setattr(visual_dense, "load_visual_corpus", blocked)
    monkeypatch.setattr(colqwen, "render_page", blocked)
    run, _, _ = execute(
        tmp_path,
        visual_dataset,
        "retrieve",
        "run",
        *args,
        f"paths.index={index}",
        "retrieval.top_k=100",
    )
    assert [task for task, _ in encoder] == ["query"]
    assert encoder[0][1] == ["I remember a scientific paper with a blue diagram."]
    ranks = [line.split() for line in (run / "run.trec").read_text().splitlines()]
    assert len(ranks) == len({row[2] for row in ranks}) == 2
    assert [int(row[3]) for row in ranks] == [1, 2]
    report = json.loads((run / "manifest.json").read_text())["artifacts"]
    assert report["vector_search_depth"] == count
    assert report["aggregation"] == (
        {"method": "max", "scope": "all_indexed_units"} if model in OPENCLIP_MODELS else None
    )
    assert report["efficiency"]["search_seconds"] >= 0
    smaller_run, _, _ = execute(
        tmp_path, visual_dataset, "retrieve", "smaller-run", *args, f"paths.index={index}"
    )
    smaller = json.loads((smaller_run / "manifest.json").read_text())["artifacts"]
    assert smaller["retrieval_cutoff"] == 2 and smaller["rankings"] == 2
    assert smaller["vector_search_depth"] == count
    assert smaller["cache"]["query_hit"] is True
    cfg, _ = settings(tmp_path, visual_dataset, "retrieve", None, *args, f"paths.index={index}")
    with pytest.raises(ValueError, match="Multiple compatible"):
        find_artifact(cfg, "index")
    evaluated, _, _ = execute(
        tmp_path,
        visual_dataset,
        "evaluate",
        "evaluation",
        *args,
        f"paths.run={run}",
        "retrieval.top_k=100",
        "evaluation.metrics=[RR@10,R@1,R@10,R@100]",
    )
    assert (evaluated / "metrics.csv").is_file()


@pytest.mark.parametrize("model", ["openclip_figures"])
@pytest.mark.parametrize("empty_all", [False, True])
def test_papers_without_figures_stay_in_evaluation(
    tmp_path, visual_dataset, encoder, empty_all, model
):
    paths = sorted(visual_dataset.glob("preprocessed/ACL/*/paper.json"))
    for path in paths if empty_all else paths[:1]:
        paper = json.loads(path.read_text())
        paper["figures"] = []
        path.write_text(json.dumps(paper))
        for image in (path.parent / "images").iterdir():
            image.unlink()
    args = overrides(model)
    index, _, _ = execute(tmp_path, visual_dataset, "index", "index", *args)
    run, _, _ = execute(tmp_path, visual_dataset, "retrieve", "run", *args, f"paths.index={index}")
    assert len((run / "corpus-ids.jsonl").read_text().splitlines()) == 2
    assert len((run / "run.trec").read_text().splitlines()) == (0 if empty_all else 1)
    evaluated, _, _ = execute(
        tmp_path, visual_dataset, "evaluate", "evaluation", *args, f"paths.run={run}"
    )
    assert "0.0" in (evaluated / "metrics.csv").read_text()


@pytest.mark.parametrize("models", [["openclip", "openclip_figures"], QWEN_MODELS])
def test_queries_are_shared_but_document_indexes_differ(tmp_path, visual_dataset, encoder, models):
    queries = []
    identities = []
    for model in models:
        args = overrides(model)
        index, _, _ = execute(tmp_path, visual_dataset, "index", f"index-{model}", *args)
        identities.append(
            json.loads((index / "index-manifest.json").read_text())["artifact_sha256"]
        )
        run, _, _ = execute(
            tmp_path, visual_dataset, "retrieve", f"run-{model}", *args, f"paths.index={index}"
        )
        queries.append(
            json.loads((run / "manifest.json").read_text())["artifacts"]["cache"]["queries"]
        )
    assert len(set(identities)) == len(models)
    assert len(set(queries)) == 1
    assert len([task for task, _ in encoder if task == "query"]) == 1
    if models == QWEN_MODELS:
        cfg, _ = settings(
            tmp_path,
            visual_dataset,
            "evaluate",
            None,
            *overrides("qwen3_vl_pages"),
            f"paths.run={run}",
            "model.query_instruction='Different query instruction.'",
        )
        with pytest.raises(ValueError, match="Retrieval experiment differs"):
            verify_retrieval(run, cfg)


@pytest.mark.parametrize("model", QWEN_MODELS)
def test_qwen_needs_only_selected_document_inputs(tmp_path, visual_dataset, model):
    import shutil

    # Each profile must work without any assets belonging to the other profiles.
    if model != "qwen3_vl_pages":
        shutil.rmtree(visual_dataset / "pdf_datasets")
    for path in visual_dataset.glob("preprocessed/ACL/*/paper.json"):
        paper = json.loads(path.read_text())
        paper.pop("structured_outputs", None)
        path.write_text(json.dumps(paper))
        if model in {"qwen3_vl_text", "qwen3_vl_pages"}:
            shutil.rmtree(path.parent / "images")
        if model in {"qwen3_vl_pages", "qwen3_vl_figures"}:
            (path.parent / "markdown.md").unlink()
    cfg, _ = settings(tmp_path, visual_dataset, "index", None, *overrides(model))
    corpus = visual_dense.load_visual_corpus(configured_adapter(cfg), cfg)
    assert len(visual_dense.document_mapping(corpus, cfg)) == len(corpus) == 2


@pytest.mark.parametrize("model", ["qwen3_vl_pages"])
def test_qwen_pdf_identity_and_missing_pdf(tmp_path, visual_dataset, model):
    from scientific_tot.records import IncompleteCorpusError

    cfg, _ = settings(tmp_path, visual_dataset, "index", None, *overrides(model))
    before = resolve_identity(cfg)
    pdf = next(visual_dataset.glob("pdf_datasets/ACL/*.pdf"))
    with pdf.open("ab") as stream:
        stream.write(b"\n% changed PDF input\n")
    assert resolve_identity(cfg) != before
    pdf.unlink()
    with pytest.raises(IncompleteCorpusError, match="unusable_pdf"):
        resolve_identity(cfg)


@pytest.mark.parametrize("model", ["openclip_figures", "qwen3_vl_figures", "qwen3_vl_multimodal"])
def test_figure_archive_and_content_identity(tmp_path, visual_dataset, model):
    args = overrides(model)
    cfg, _ = settings(tmp_path, visual_dataset, "index", None, *args)
    before = resolve_identity(cfg)
    image = next(visual_dataset.glob("preprocessed/ACL/*/images/*.png"))
    Image.new("RGB", (8, 8), "red").save(image)
    assert resolve_identity(cfg) != before
    base = visual_dataset / "preprocessed/ACL"
    paths = list(base.glob("*/*")) + list(base.glob("*/images/*"))
    with tarfile.open(base / "shard.tar", "w") as archive:
        for path in paths:
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(base)))
                path.unlink()
    corpus = visual_dense.load_visual_corpus(configured_adapter(cfg), cfg)
    assert len(visual_dense.document_mapping(corpus, cfg)) == (
        4 if model == "openclip_figures" else 2
    )


@pytest.mark.parametrize("model", ["qwen3_vl_pages"])
def test_qwen_page_rendering_is_part_of_identity(tmp_path, visual_dataset, model):
    cfg, _ = settings(tmp_path, visual_dataset, "index", None, *overrides(model))
    resized, _ = settings(
        tmp_path,
        visual_dataset,
        "index",
        None,
        *overrides(model),
        "representation.rendering.scale=1.0",
    )
    assert resolve_identity(cfg) != resolve_identity(resized)


@pytest.mark.parametrize("model", ["qwen3_vl_figures", "qwen3_vl_multimodal"])
def test_qwen_paper_without_figures(tmp_path, visual_dataset, model):
    import shutil

    path = next(visual_dataset.glob("preprocessed/ACL/*/paper.json"))
    paper = json.loads(path.read_text())
    paper["figures"] = []
    path.write_text(json.dumps(paper))
    shutil.rmtree(path.parent / "images")
    cfg, _ = settings(tmp_path, visual_dataset, "index", None, *overrides(model))
    if model == "qwen3_vl_figures":
        with pytest.raises(ValueError, match="figure-only input requires supplied figures"):
            resolve_identity(cfg)
    else:
        corpus = visual_dense.load_visual_corpus(configured_adapter(cfg), cfg)
        mapping = visual_dense.document_mapping(corpus, cfg)
        row = next(row for row in mapping if row["doc_id"] == paper["paper_id"])
        assert row["images"] == [] and row["text"]
        assert row["page_count"] == 0


def test_openclip_standard_long_query_tokenization():
    import open_clip

    encoder = OpenCLIPEncoder.__new__(OpenCLIPEncoder)
    encoder.cfg = SimpleNamespace(model=SimpleNamespace(device="cpu"))
    encoder.tokenizer = open_clip.get_tokenizer("ViT-B-32")
    encoder.truncated = 0
    captured = []
    encoder.model = SimpleNamespace(context_length=77, encode_text=lambda x: captured.append(x))
    encoder.encode(["short query", "scientific document " * 200], "query")
    assert captured[0].shape == (2, 77)
    assert captured[0][1, -1] == encoder.tokenizer.eot_token_id
    assert encoder.truncated == 1


def test_qwen_full_inputs_and_processor_truncation():
    import torch

    encoder = QwenEmbeddingEncoder.__new__(QwenEmbeddingEncoder)
    encoder.cfg = SimpleNamespace(
        model=SimpleNamespace(query_instruction="Retrieve relevant images.")
    )
    seen = []
    limits = []
    forwarded = []

    def format_input(image=None, text=None, instruction=None):
        content = []
        if image is not None:
            content.append({"type": "image", "image": image})
        if text is not None:
            content.append({"type": "text", "text": text})
        return [
            {"role": "system", "content": instruction},
            {"role": "user", "content": content},
        ]

    def preprocess(rows):
        seen.extend(rows)
        limits.append(encoder.model.max_length)
        count = sum(item["type"] == "image" for row in rows for item in row[1]["content"])
        return {
            "attention_mask": torch.ones(len(rows), min(9000, encoder.model.max_length)),
            **(
                {"pixel_values": torch.ones(count), "image_grid_thw": torch.ones(count, 3)}
                if count
                else {}
            ),
        }

    encoder.model = SimpleNamespace(
        max_length=32768,
        format_model_input=format_input,
        _preprocess_inputs=preprocess,
        model=SimpleNamespace(device="cpu"),
        forward=lambda values: (
            forwarded.append(values)
            or {
                "last_hidden_state": torch.ones(len(values["attention_mask"]), 2, 2),
                "attention_mask": values["attention_mask"],
            }
        ),
        _pooling_last=lambda hidden, mask: hidden[:, -1],
    )
    query = "original ToT text " * 1000
    encoder.encode([query], "query")
    assert seen[-1] == format_input(text=query, instruction="Retrieve relevant images.")
    images = [Image.new("RGB", (8, 8)) for _ in range(3)]
    for text in (None, "Full original paper text"):
        row = {"images": images, "doc_id": "paper", **({"text": text} if text else {})}
        result = encoder.encode([row], "document")
        assert result.shape == (1, 2)
        assert seen[-1][1]["content"] == [
            *[{"type": "image", "image": image} for image in images],
            *([{"type": "text", "text": text}] if text else []),
        ]
    encoder.encode([{"images": [], "text": "Full original paper text"}], "document")
    assert seen[-1][1]["content"] == [{"type": "text", "text": "Full original paper text"}]
    encoder.model.max_length = 8192
    for values, task in (
        ([query], "query"),
        ([{"images": [], "text": "Long text", "doc_id": "paper"}], "document"),
        ([{"images": images, "text": "Long text", "doc_id": "paper"}], "document"),
    ):
        result = encoder.encode(values, task)
        assert result.shape == (1, 2)
        assert forwarded[-1]["attention_mask"].shape == (1, 8192)
        assert limits[-1] == encoder.model.max_length == 8192
    encoder.model.max_length = 32768
    encoder.model._preprocess_inputs = lambda rows: {
        "attention_mask": torch.ones(len(rows), 2),
        "pixel_values": torch.ones(1),
        "image_grid_thw": torch.ones(1, 3),
    }
    with pytest.raises(ValueError, match="every image"):
        encoder.encode([{"images": images, "text": "Full text"}], "document")
    for image in images:
        image.close()


@pytest.mark.visual_live
def test_real_qwen_prefix_truncation(tmp_path, visual_dataset):
    import torch

    cfg, _ = settings(
        tmp_path,
        visual_dataset,
        "index",
        None,
        *overrides("qwen3_vl_multimodal"),
        "model.max_length=256",
    )
    encoder = visual_dense.load_model(cfg)
    captured = []
    forward = encoder.model.forward

    def capture(values):
        captured.append(values)
        return forward(values)

    encoder.model.forward = capture
    text = "PREFIXRETAINED " + "scientific document " * 1000 + " SUFFIXREMOVED"
    from scientific_tot.config_arguments import configured_corpus

    corpus = configured_corpus(cfg, configured_adapter(cfg))
    row = visual_dense.document_mapping(corpus, cfg)[0]
    from scientific_tot.data.artifacts import Artifacts

    store = Artifacts(configured_adapter(cfg).dataset, tuple(cfg.dataset.domains), archives=True)
    value = visual_dense._load_batch([row], cfg, "document", store)[0]
    images = value["images"]
    assert [image.size for image in images] == [(200, 160), (201, 160), (202, 160)]
    try:
        with torch.inference_mode():
            vectors = encoder.encode(
                [
                    {"images": images},
                    {"images": [], "text": text},
                    {"images": images, "text": text},
                ],
                "document",
            )
            assert vectors.shape == (3, 2048) and torch.isfinite(vectors).all()
            assert torch.allclose(vectors.norm(dim=1), torch.ones(3), atol=1e-5)
            assert captured[-1]["attention_mask"].sum(dim=1).tolist()[1:] == [256, 256]
            assert len(captured[-1]["image_grid_thw"]) == 6
            decoded = encoder.model.processor.tokenizer.batch_decode(captured[-1]["input_ids"])
            for value in decoded[1:]:
                assert "PREFIXRETAINED" in value and "SUFFIXREMOVED" not in value
            encoder.encode([text], "query")
            assert captured[-1]["attention_mask"].sum().item() == 256
            assert encoder.model.processor.tokenizer.truncation_side == "right"
    finally:
        for image in images:
            image.close()


@pytest.mark.visual_live
@pytest.mark.parametrize("model", MODELS)
def test_real_checkpoint_stages(tmp_path, visual_dataset, model):
    """Two papers, original ToT query, persistent index and shared evaluation."""
    query_file = next(visual_dataset.glob("query_collections/**/queries.jsonl"))
    query = json.loads(query_file.read_text())
    query["query"] += " I remember a scientific diagram with several colored panels." * 20
    query_file.write_text(json.dumps(query) + "\n")
    args = (
        *overrides(model),
        "model.batch_size=2",
        "retrieval.top_k=100",
        "evaluation.metrics=[RR@10,R@1,R@10,R@100]",
    )
    index, _, _ = execute(tmp_path, visual_dataset, "index", "index", *args)
    report = json.loads((index / "index-manifest.json").read_text())["provenance"]
    assert report["vectors"] == (4 if model in OPENCLIP_MODELS else 2)
    if model in QWEN_MODELS:
        assert report["encoding"]["rendered_pages"] == (4 if model == "qwen3_vl_pages" else 0)
        assert report["encoding"]["supplied_images"] == (
            4 if model in {"qwen3_vl_figures", "qwen3_vl_multimodal"} else 0
        )
    run, _, _ = execute(tmp_path, visual_dataset, "retrieve", "run", *args, f"paths.index={index}")
    evaluated, _, _ = execute(
        tmp_path, visual_dataset, "evaluate", "evaluation", *args, f"paths.run={run}"
    )
    rankings = [row.split() for row in (run / "run.trec").read_text().splitlines()]
    assert len(rankings) == len({row[2] for row in rankings}) == 2
    assert json.loads((run / "topics.jsonl").read_text())["query"] == query["query"]
    if model in OPENCLIP_MODELS:
        report = json.loads((run / "manifest.json").read_text())["artifacts"]
        assert report["encoding"]["truncation_proportion"] == 1.0
    assert (evaluated / "metrics.csv").is_file()
