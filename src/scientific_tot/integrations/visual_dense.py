"""Single-vector supplied-image encoders using official OpenCLIP and Qwen APIs."""

import hashlib
import importlib.util
import sys
import tempfile
from contextlib import ExitStack, closing
from importlib.metadata import version
from io import BytesIO
from pathlib import Path
from time import perf_counter
from urllib.parse import quote

import numpy as np
from omegaconf import OmegaConf

from scientific_tot.config_arguments import configured_adapter, corpus_arguments
from scientific_tot.configuration import OPENCLIP_MODELS
from scientific_tot.data.artifacts import Artifacts
from scientific_tot.data.validation import DatasetFormatError
from scientific_tot.integrations.dense import _read_cache, _write_cache
from scientific_tot.records import Issue, Records, Report, records_digest


def document_mapping(corpus, cfg) -> list[dict]:
    """Preserve page/figure identity; lean images use paths as internal vector keys."""
    if cfg.model.document_input == "figures":
        return [
            {
                "docno": f"{quote(paper['doc_id'], safe='')}%p"
                f"{quote(figure['figure_id'] or figure['image']['path'], safe='')}",
                "doc_id": paper["doc_id"],
                "domain": paper["domain"],
                "figure_id": figure["figure_id"],
                "image_path": figure["image"]["path"],
                "image_sha256": figure["image_sha256"],
            }
            for paper in corpus
            for figure in paper["figures"]
        ]
    from scientific_tot.integrations.colqwen import page_mapping

    papers = {row["doc_id"]: row for row in corpus}
    return [
        {
            **page,
            "docno": page["page_id"],
            "pdf_path": papers[page["doc_id"]]["pdf_path"],
            "pdf_sha256": papers[page["doc_id"]]["pdf_sha256"],
            **(
                {"text": papers[page["doc_id"]]["page_texts"][page["page_number"] - 1]}
                if cfg.model.document_input == "multimodal"
                else {}
            ),
        }
        for page in page_mapping(corpus)
    ]


def visual_corpus_digest(corpus, cfg) -> str:
    # Retain the full document collection, including papers with zero visual elements.
    return records_digest(
        [
            {"documents": [{"doc_id": row["doc_id"], "domain": row["domain"]} for row in corpus]},
            {
                "units": [
                    {k: v for k, v in row.items() if k != "pdf_path"}
                    for row in document_mapping(corpus, cfg)
                ]
            },
        ]
    )


def load_visual_corpus(adapter, cfg) -> Records:
    """Use original PDFs or supplied images, with explicit MinerU alignment for mixed input."""
    from PIL import Image
    from tqdm.auto import tqdm

    strategy = cfg.model.document_input
    if strategy != "figures":
        from scientific_tot.integrations.colqwen import load_pdf_corpus

        selected = load_pdf_corpus(adapter, cfg)
        if strategy == "pages":
            return selected
        from scientific_tot.integrations.mineru import page_texts

        store = Artifacts(adapter.dataset, tuple(cfg.dataset.domains), archives=True)
        rows = []
        for paper in tqdm(selected, desc="Aligning MinerU page text", unit="paper"):
            paper["page_texts"], paper["page_text_source"] = page_texts(store, paper)
            rows.append(paper)
        missing = sum(row["page_text_source"]["pages_without_text"] for row in rows)
        print(f"Multimodal pages without extracted text: {missing:,}", flush=True)
        return Records("corpus", rows, selected.report, selected.provenance)

    selected = adapter.load_corpus(**corpus_arguments(cfg))
    store = Artifacts(adapter.dataset, tuple(cfg.dataset.domains), archives=True)
    report = Report(
        selected=selected.report.selected,
        skipped=selected.report.skipped,
        issues=list(selected.report.issues),
    )
    rows = []
    for paper in tqdm(selected, desc="Reading supplied images", unit="paper"):
        figures = []
        for figure in paper["figures"]:
            try:
                data = store.read_bytes(figure["image"]["path"])
                with Image.open(BytesIO(data)) as image:
                    image.verify()
                figure["image_sha256"] = hashlib.sha256(data).hexdigest()
            except DatasetFormatError:
                raise
            except (OSError, ValueError) as exc:
                if cfg.dataset.missing == "error":
                    raise ValueError(f"Unusable image for {paper['doc_id']}: {exc}") from exc
                report.issues.append(
                    Issue(
                        "unusable_image", figure["image"]["path"] or "", str(exc), paper["doc_id"]
                    )
                )
                continue
            figures.append(figure)
        paper["figures"] = figures
        if not figures:
            report.issues.append(
                Issue(
                    "no_visual_elements",
                    paper["doc_id"],
                    "Retained for evaluation; not retrievable by the figures baseline",
                    paper["doc_id"],
                )
            )
        rows.append(paper)
    report.emitted = len(rows)
    return Records("corpus", rows, report, selected.provenance)


def dependency_versions(cfg) -> dict:
    names = ["torch", "torchvision", "pillow", "pypdfium2", "numpy", "faiss-cpu"]
    names += (
        ["open-clip-torch"]
        if cfg.model.name in OPENCLIP_MODELS
        else ["transformers", "qwen-vl-utils"]
    )
    return {name: version(name) for name in names}


class OpenCLIPEncoder:
    def __init__(self, cfg, cache):
        import open_clip
        from huggingface_hub import hf_hub_download

        self.cfg = cfg
        self.truncated = 0
        weights = hf_hub_download(
            cfg.model.model_id,
            "open_clip_model.safetensors",
            revision=cfg.model.revision,
            cache_dir=str(cache),
        )
        # Loading a local pinned file must retain the named pretrained transform settings.
        preprocessing = open_clip.get_pretrained_cfg(cfg.model.architecture, cfg.model.pretrained)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            cfg.model.architecture,
            pretrained=weights,
            device=cfg.model.device,
            precision=cfg.model.precision,
            image_mean=preprocessing["mean"],
            image_std=preprocessing["std"],
            image_interpolation=preprocessing["interpolation"],
            image_resize_mode=preprocessing["resize_mode"],
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(cfg.model.architecture)
        self.metadata = {
            "context_length": self.model.context_length,
            "preprocessing": preprocessing,
        }
        assert self.model.context_length == 77

    def encode(self, values, task):
        import torch

        if task == "document":
            images = torch.stack([self.preprocess(value["image"]) for value in values])
            return self.model.encode_image(
                images.to(device=self.cfg.model.device, dtype=self.model.visual.conv1.weight.dtype)
            )
        self.truncated += sum(
            len(self.tokenizer.encode(text)) + 2 > self.model.context_length for text in values
        )
        return self.model.encode_text(self.tokenizer(values).to(self.cfg.model.device))


class QwenEmbeddingEncoder:
    def __init__(self, cfg, cache):
        import torch
        from huggingface_hub import snapshot_download

        self.cfg = cfg
        snapshot = Path(
            snapshot_download(cfg.model.model_id, revision=cfg.model.revision, cache_dir=str(cache))
        )
        # Execute the checkpoint's pinned official implementation without copying its model code.
        name = "scientific_tot_qwen3_vl_embedding"
        spec = importlib.util.spec_from_file_location(
            name, snapshot / "scripts/qwen3_vl_embedding.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        self.model = module.Qwen3VLEmbedder(
            model_name_or_path=str(snapshot),
            max_length=cfg.model.max_length,
            torch_dtype=dtype[cfg.model.precision],
            attn_implementation="sdpa",
        )
        self.model.model.to(cfg.model.device)
        self.metadata = {
            "implementation": "checkpoint/scripts/qwen3_vl_embedding.py:Qwen3VLEmbedder.process",
            "max_length": self.model.max_length,
            "min_pixels": self.model.min_pixels,
            "max_pixels": self.model.max_pixels,
            "document_instruction": self.model.default_instruction,
            "query_instruction": cfg.model.query_instruction,
            "attention": "sdpa",
        }

    def encode(self, values, task):
        if task == "document":
            inputs = values
        else:
            inputs = [
                {"text": text, "instruction": self.cfg.model.query_instruction} for text in values
            ]
        # Probe the official processor with one extra token to detect overflow before
        # process() can silently truncate. This counts expanded image tokens as well
        # as text/template tokens, without moving the preview tensors to the GPU.
        conversations = [self.model.format_model_input(**row) for row in inputs]
        limit = self.model.max_length
        try:
            self.model.max_length = limit + 1
            preview = self.model._preprocess_inputs(conversations)
        finally:
            self.model.max_length = limit
        if (preview["attention_mask"].sum(dim=1) > limit).any():
            raise ValueError("Qwen input exceeds the shared context budget; refusing truncation")
        if task == "document" and "pixel_values" not in preview:
            raise ValueError("Official Qwen preprocessing did not retain the image modality")
        del preview
        return self.model.process(inputs, normalize=True)


def load_model(cfg):
    import torch

    torch.set_num_threads(cfg.launcher.threads.encoding)
    if cfg.model.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Visual encoding requires an available NVIDIA GPU for device=cuda")
    cache = Path(cfg.launcher.cache_root) / "models"
    encoder = OpenCLIPEncoder if cfg.model.name in OPENCLIP_MODELS else QwenEmbeddingEncoder
    return encoder(cfg, cache)


def cached_encoding(cfg, *, task, rows, dataset_revision):
    """Stream bounded image batches into the existing NumPy/FAISS embedding cache."""
    import faiss
    import torch
    from PIL import Image
    from tqdm.auto import tqdm

    assert task in {"document", "query"} and (rows or task == "document")
    model_settings = OmegaConf.to_container(cfg.model, resolve=True)
    if task == "document":
        model_settings.pop("query_instruction", None)
    else:
        model_settings.pop("document_input")
        model_settings["name"] = (
            "openclip" if cfg.model.name in OPENCLIP_MODELS else "qwen3_vl_embedding"
        )
    configuration = {
        "schema_version": 2,
        "task": task,
        "model": model_settings,
        "content_sha256": records_digest(rows),
        "dataset_revision": dataset_revision,
        "versions": dependency_versions(cfg),
        "encoding_threads": cfg.launcher.threads.encoding,
    }
    if task == "document":
        configuration.update(
            representation=OmegaConf.to_container(cfg.representation, resolve=True),
            index="FAISS IndexFlatIP",
        )
    cache = Path(cfg.launcher.cache_root) / "embeddings"
    entry = cache / records_digest([configuration])
    hit = entry.exists()
    print(
        f"{task.capitalize()} embeddings: {len(rows):,} inputs; cache={'hit' if hit else 'miss'}",
        flush=True,
    )
    faiss.omp_set_num_threads(cfg.launcher.threads.faiss)
    if not hit:
        started = perf_counter()
        encoder = load_model(cfg) if rows else None
        cache.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=cache) as folder, ExitStack() as resources:
            vectors = np.lib.format.open_memmap(
                Path(folder) / "embeddings.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(rows), cfg.model.dimension),
            )
            index = faiss.IndexFlatIP(cfg.model.dimension) if task == "document" else None
            store = (
                Artifacts(
                    configured_adapter(cfg).dataset, tuple(cfg.dataset.domains), archives=True
                )
                if task == "document" and cfg.model.document_input == "figures"
                else None
            )
            pdf = None
            document = None
            size = cfg.model.batch_size
            for start in tqdm(range(0, len(rows), size), desc=f"Encoding {task}", unit="batch"):
                batch = rows[start : start + size]
                with ExitStack() as images:
                    if task == "document":
                        values = []
                        for row in batch:
                            if cfg.model.document_input == "figures":
                                data = store.read_bytes(row["image_path"])
                                if hashlib.sha256(data).hexdigest() != row["image_sha256"]:
                                    raise ValueError(
                                        f"Image changed during encoding: {row['image_path']}"
                                    )
                                source = images.enter_context(Image.open(BytesIO(data)))
                                image = images.enter_context(source.convert("RGB"))
                            else:
                                import pypdfium2 as pdfium

                                from scientific_tot.integrations.colqwen import render_page

                                if row["doc_id"] != document:
                                    resources.close()
                                    document = row["doc_id"]
                                    pdf = resources.enter_context(
                                        pdfium.PdfDocument(row["pdf_path"])
                                    )
                                image = images.enter_context(
                                    closing(render_page(pdf, row["page_number"], cfg))
                                )
                            values.append(
                                {"image": image, **({"text": row["text"]} if "text" in row else {})}
                            )
                    else:
                        values = [row["query"] for row in batch]
                    with torch.inference_mode():
                        encoded = encoder.encode(values, task).float()
                        assert encoded.shape == (len(batch), cfg.model.dimension)
                        assert torch.isfinite(encoded).all()
                        assert (encoded.norm(dim=1) > 0).all()
                        encoded = torch.nn.functional.normalize(encoded, dim=1)
                        chunk = encoded.cpu().numpy()
                    assert np.allclose(np.linalg.norm(chunk, axis=1), 1, atol=1e-5)
                    vectors[start : start + len(batch)] = chunk
                    if index is not None:
                        index.add(chunk)
            metadata = {
                **(encoder.metadata if encoder is not None else {}),
                "dimension": cfg.model.dimension,
                "normalized": True,
                "generation_seconds": perf_counter() - started,
                "units": len(rows),
            }
            if cfg.model.name in OPENCLIP_MODELS and task == "query":
                metadata.update(
                    truncated_queries=encoder.truncated,
                    queries=len(rows),
                    truncation_proportion=encoder.truncated / len(rows),
                )
            _write_cache(
                entry,
                configuration,
                vectors,
                metadata,
                index=index,
                ids=[row["docno"] for row in rows] if index is not None else None,
            )
            del vectors
    vectors, manifest = _read_cache(entry, configuration)
    if vectors.shape != (len(rows), cfg.model.dimension):
        raise ValueError("Cached embedding dimensions/count differ from configured inputs")
    if task == "query" and cfg.model.name in OPENCLIP_MODELS:
        meta = manifest["encoding"]
        print(
            f"OpenCLIP truncated queries: {meta['truncated_queries']}/{len(rows)} "
            f"({meta['truncation_proportion']:.1%})",
            flush=True,
        )
    return vectors, manifest, entry, hit
