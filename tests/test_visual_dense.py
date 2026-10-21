"""Real PDFium/FAISS/stage lifecycle; pretrained inference is opt-in."""

import json
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
from scientific_tot.data.validation import DatasetFormatError
from scientific_tot.integrations import colqwen, visual_dense
from scientific_tot.integrations.visual_dense import OpenCLIPEncoder, QwenEmbeddingEncoder
from scientific_tot.stage_cli import find_artifact

pdf_dataset = _pdf_dataset
QWEN_MODELS = ["qwen3_vl_page", "qwen3_vl_multimodal", "qwen3_vl_figures"]
MODELS = ["openclip", "openclip_figures", *QWEN_MODELS]


def overrides(model):
    experiment = "openclip_pages" if model == "openclip" else model
    return (f"experiment={experiment}", "launcher=local_cpu", "model.precision=fp32")


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
        paper["structured_outputs"] = {"paper_content_list.json": "mineru/paper_content_list.json"}
        target = path.parent / "mineru/paper_content_list.json"
        target.parent.mkdir()
        target.write_text(
            json.dumps(
                [
                    {
                        "type": "text",
                        "page_idx": number,
                        "text": f"Original page {number + 1}.\nUnmodified.",
                    }
                    for number in range(count - 1)  # Last page intentionally has no extracted text.
                ]
            )
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
                    else [{**value, "image": value["image"].size} for value in values],
                )
            )
            result = torch.zeros(len(values), self.dimension)
            result[:, 0] = 1
            if task == "document":
                result[:, 1] = torch.tensor([1 / value["image"].width for value in values])
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
    assert len(units) == 4 and not encoder
    if model == "qwen3_vl_multimodal":
        assert [unit["text"] for unit in units] == [
            "Original page 1.\nUnmodified.",
            "Original page 2.\nUnmodified.",
            "",
            "",
        ]
    (tmp_path / "results/indexes").mkdir(parents=True)
    index, _, _ = execute(tmp_path, visual_dataset, "index", "results/indexes/first", *args)
    provenance = json.loads((index / "index-manifest.json").read_text())["provenance"]
    assert provenance["vectors"] == 4 and provenance["documents"] == 2
    assert all(task == "document" for task, _ in encoder)
    values = [value for _, batch in encoder for value in batch]
    assert all(("text" in value) == (model == "qwen3_vl_multimodal") for value in values)
    widths = [200, 201, 202, 100] if model.endswith("figures") else [200, 202, 204, 100]
    assert [value["image"] for value in values] == [(width, 160) for width in widths]
    vectors = np.load(Path(provenance["cache"]["documents"]) / "embeddings.npy")
    assert vectors.shape == (4, 512 if model in OPENCLIP_MODELS else 2048)
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
    assert report["vector_search_depth"] == 4
    assert report["aggregation"]["method"] == "max"
    assert report["efficiency"]["search_seconds"] >= 0
    smaller_run, _, _ = execute(
        tmp_path, visual_dataset, "retrieve", "smaller-run", *args, f"paths.index={index}"
    )
    smaller = json.loads((smaller_run / "manifest.json").read_text())["artifacts"]
    assert smaller["retrieval_cutoff"] == 2 and smaller["rankings"] == 2
    assert smaller["vector_search_depth"] == 4
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


@pytest.mark.parametrize("model", ["openclip_figures", "qwen3_vl_figures"])
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
            *overrides("qwen3_vl_figures"),
            f"paths.run={run}",
            "model.query_instruction='Different query instruction.'",
        )
        with pytest.raises(ValueError, match="Retrieval experiment differs"):
            verify_retrieval(run, cfg)


def test_mineru_alignment_requires_explicit_pages(tmp_path, visual_dataset):
    cfg, _ = settings(tmp_path, visual_dataset, "index", None, *overrides("qwen3_vl_multimodal"))
    source = next(visual_dataset.glob("preprocessed/ACL/*/mineru/*content_list.json"))
    source.write_text(json.dumps([{"type": "text", "text": "Unaligned text"}]))
    with pytest.raises(DatasetFormatError, match="page_idx"):
        resolve_identity(cfg)
    source.unlink()
    with pytest.raises(FileNotFoundError, match="Missing extracted artifact"):
        resolve_identity(cfg)


def test_mineru_v2_alignment_and_lean_limitation(tmp_path, visual_dataset):
    cfg, _ = settings(tmp_path, visual_dataset, "index", None, *overrides("qwen3_vl_multimodal"))
    paper_path = sorted(visual_dataset.glob("preprocessed/ACL/*/paper.json"))[0]
    paper = json.loads(paper_path.read_text())
    source = paper_path.parent / "mineru/paper_content_list.json"
    source.write_text(
        json.dumps(
            [
                [
                    {
                        "type": "paragraph",
                        "content": {
                            "paragraph_content": [
                                {"type": "text", "content": "First "},
                                {"type": "text", "content": "page"},
                            ]
                        },
                    }
                ],
                [
                    {
                        "type": "table",
                        "content": {
                            "html": "<table>Second page</table>",
                            "image_source": {"path": "not text"},
                        },
                    }
                ],
                [],
            ]
        )
    )
    corpus = visual_dense.load_visual_corpus(configured_adapter(cfg), cfg)
    assert corpus[0]["page_texts"] == ["First page", "<table>Second page</table>", ""]
    paper.pop("structured_outputs")
    paper_path.write_text(json.dumps(paper))
    source.unlink()
    with pytest.raises(ValueError, match="alignment unavailable"):
        resolve_identity(cfg)


def test_figure_archive_and_content_identity(tmp_path, visual_dataset):
    args = overrides("qwen3_vl_figures")
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
    assert len(visual_dense.document_mapping(corpus, cfg)) == 4


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


def test_qwen_full_query_and_combined_context():
    import torch

    encoder = QwenEmbeddingEncoder.__new__(QwenEmbeddingEncoder)
    encoder.cfg = SimpleNamespace(
        model=SimpleNamespace(query_instruction="Retrieve relevant images.")
    )
    seen = []
    encoder.model = SimpleNamespace(
        max_length=32768,
        format_model_input=lambda **row: seen.append(row) or row,
        _preprocess_inputs=lambda rows: {
            "attention_mask": torch.ones(len(rows), 9000),
            "pixel_values": torch.ones(1),
        },
        model=SimpleNamespace(device="cpu"),
        forward=lambda values: {
            "last_hidden_state": torch.ones(len(values["attention_mask"]), 2, 2),
            "attention_mask": values["attention_mask"],
        },
        _pooling_last=lambda hidden, mask: hidden[:, -1],
    )
    query = "original ToT text " * 1000
    encoder.encode([query], "query")
    assert seen == [{"text": query, "instruction": "Retrieve relevant images."}]
    image = Image.new("RGB", (8, 8))
    encoder.encode([{"image": image, "text": "Exact page text"}], "document")
    assert seen[-1] == {"image": image, "text": "Exact page text"}
    encoder.model.max_length = 8192
    with pytest.raises(ValueError, match="refusing truncation"):
        encoder.encode([query], "query")
    assert encoder.model.max_length == 8192
    with pytest.raises(ValueError, match="shared context"):
        encoder.encode([{"image": image, "text": "Too much combined input"}], "document")
    encoder.model.max_length = 32768
    encoder.model._preprocess_inputs = lambda rows: {"attention_mask": torch.ones(len(rows), 2)}
    with pytest.raises(ValueError, match="image modality"):
        encoder.encode([{"image": image, "text": "Page text"}], "document")


@pytest.mark.visual_live
@pytest.mark.parametrize("model", MODELS)
def test_real_checkpoint_stages(tmp_path, visual_dataset, model):
    """Four small indexed units, original ToT query, persistent index and shared evaluation."""
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
