"""Single-vector page/text and supplied-image encoders using official OpenCLIP and Qwen APIs."""

import hashlib
import importlib.util
import json
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, closing
from io import BytesIO
from pathlib import Path
from time import perf_counter
from urllib.parse import quote

import numpy as np
from omegaconf import OmegaConf

from scientific_tot.config_arguments import configured_adapter, corpus_arguments
from scientific_tot.configuration import OPENCLIP_MODELS, QWEN_MODELS
from scientific_tot.data.artifacts import Artifacts
from scientific_tot.data.validation import DatasetFormatError
from scientific_tot.integrations.dense import _read_cache, _write_cache
from scientific_tot.records import Issue, Records, Report, records_digest
from scientific_tot.reporting import package_version, phase

QWEN_PAGE_LIMIT = 18


def document_mapping(corpus, cfg) -> list[dict]:
    """Qwen indexes papers; OpenCLIP indexes individual pages or supplied images."""
    return _document_mapping(corpus, cfg.model.name, cfg.model.document_input)


def _document_mapping(corpus, model, strategy):
    if model in QWEN_MODELS:
        return [
            {
                "docno": paper["doc_id"],
                "doc_id": paper["doc_id"],
                "domain": paper["domain"],
                **(
                    {
                        "page_count": min(paper["page_count"], QWEN_PAGE_LIMIT),
                        "pdf_path": paper["pdf_path"],
                        "pdf_sha256": paper["pdf_sha256"],
                    }
                    if strategy == "pages"
                    else {
                        "page_count": 0,
                        "images": [
                            {
                                "figure_id": figure["figure_id"],
                                "image_path": figure["image"]["path"],
                                "image_sha256": figure["image_sha256"],
                            }
                            for figure in paper["figures"]
                        ],
                    }
                ),
                **({"text": paper["text"]} if strategy in {"text", "multimodal"} else {}),
            }
            for paper in corpus
            if strategy != "figures" or paper["figures"]
        ]
    if strategy == "figures":
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
    from scientific_tot.integrations.colqwen import iter_page_mapping

    result = []
    for paper in corpus:
        for page in iter_page_mapping((paper,)):
            page.update(
                docno=page["page_id"],
                pdf_path=paper["pdf_path"],
                pdf_sha256=paper["pdf_sha256"],
            )
            result.append(page)
    return result


def visual_corpus_digest(corpus, cfg) -> str:
    # Retain the full document collection, including papers with zero visual elements.
    documents = []
    chunks = []
    model, strategy = cfg.model.name, cfg.model.document_input
    encoder = json.JSONEncoder(ensure_ascii=False, sort_keys=True, allow_nan=False)
    for paper in corpus:
        documents.append({"doc_id": paper["doc_id"], "domain": paper["domain"]})
        units = _document_mapping((paper,), model, strategy)
        for unit in units:
            unit.pop("pdf_path", None)
        if units:
            chunks.append(encoder.encode(units)[1:-1].encode())
    # Match records_digest's two JSON rows exactly, without retaining every unit
    # dictionary or allocating a corpus-sized JSON string and UTF-8 copy.
    checksum = hashlib.sha256()
    checksum.update(encoder.encode({"documents": documents}).encode())
    checksum.update(b'\n{"units": [')
    for number, chunk in enumerate(chunks):
        if number:
            checksum.update(b", ")
        checksum.update(chunk)
    checksum.update(b"]}\n")
    return checksum.hexdigest()


def load_visual_corpus(adapter, cfg) -> Records:
    """Reuse original PDF loading for page inputs and the text adapter for Markdown."""
    from scientific_tot.reporting import tqdm

    strategy = cfg.model.document_input
    if strategy == "pages":
        from scientific_tot.integrations.colqwen import load_pdf_corpus

        return load_pdf_corpus(adapter, cfg)
    if strategy == "text":
        return adapter.load_corpus(**corpus_arguments(cfg))

    print(
        f"Loading figure metadata: domains={','.join(cfg.dataset.domains)}, "
        f"split={cfg.split.corpus}; dataset={cfg.dataset.root}",
        flush=True,
    )
    selected = adapter.load_corpus(**corpus_arguments(cfg))
    store = adapter.artifacts
    assert store is not None
    report = Report(
        selected=selected.report.selected,
        skipped=selected.report.skipped,
        issues=list(selected.report.issues),
    )

    def papers():
        with store.reader() as read_bytes:
            for paper in tqdm(
                selected, desc="Fingerprinting figure inputs for cache identity", unit="paper"
            ):
                figures = []
                for figure in paper["figures"]:
                    try:
                        data = read_bytes(figure["image"]["path"])
                        figure["image_sha256"] = hashlib.sha256(data).hexdigest()
                    except DatasetFormatError:
                        raise
                    except (OSError, ValueError) as exc:
                        if cfg.dataset.missing == "error" or cfg.model.name in QWEN_MODELS:
                            raise ValueError(
                                f"Unusable image for {paper['doc_id']}: {exc}"
                            ) from exc
                        report.issues.append(
                            Issue(
                                "unusable_image",
                                figure["image"]["path"] or "",
                                str(exc),
                                paper["doc_id"],
                            )
                        )
                        continue
                    figures.append(figure)
                paper["figures"] = figures
                if not figures and strategy == "figures":
                    report.issues.append(
                        Issue(
                            "no_visual_elements",
                            paper["doc_id"],
                            "Retained for evaluation; not retrievable by the figures baseline",
                            paper["doc_id"],
                        )
                    )
                yield paper

    corpus = Records("corpus", papers(), report, selected.provenance)
    report.emitted = len(corpus)
    return corpus


def dependency_versions(cfg) -> dict:
    names = ["torch", "torchvision", "pillow", "pypdfium2", "numpy", "faiss"]
    names += (
        ["open-clip-torch"]
        if cfg.model.name in OPENCLIP_MODELS
        else ["transformers", "qwen-vl-utils"]
    )
    return {name: package_version(name) for name in names}


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
        self.process_vision_info = module.process_vision_info
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        self.model = module.Qwen3VLEmbedder(
            model_name_or_path=str(snapshot),
            max_length=cfg.model.max_length,
            torch_dtype=dtype[cfg.model.precision],
            attn_implementation="sdpa",
        )
        self.model.model.to(cfg.model.device)
        self.model.processor.tokenizer.truncation_side = cfg.model.truncation_side
        self.metadata = {
            "implementation": "official Qwen3VLEmbedder preprocessing, forward and _pooling_last",
            "max_length": self.model.max_length,
            "truncation_side": cfg.model.truncation_side,
            "min_pixels": self.model.min_pixels,
            "max_pixels": self.model.max_pixels,
            "document_instruction": self.model.default_instruction,
            "query_instruction": cfg.model.query_instruction,
            "attention": "sdpa",
        }

    def _preprocess(self, conversations, *, truncation=True):
        # Paper/query text can quote Qwen control tokens. Escape them before the
        # template adds real modality markers, or the processor consumes extra images.
        reserved = [*self.model.processor.tokenizer.all_special_tokens, "<|placeholder|>"]
        for conversation in conversations:
            for message in conversation:
                for block in message["content"]:
                    if block["type"] == "text":
                        for token in reserved:
                            block["text"] = block["text"].replace(
                                token, token.replace("<|", "< |", 1)
                            )
        # Same pinned official preprocessing arguments, with errors allowed to
        # propagate instead of its catch-all replacement with a NULL input.
        text = self.model.processor.apply_chat_template(
            conversations, add_generation_prompt=True, tokenize=False
        )
        images, videos, kwargs = self.process_vision_info(
            conversations,
            image_patch_size=16,
            return_video_metadata=True,
            return_video_kwargs=True,
        )
        assert videos is None  # This integration supplies images and text only.
        return self.model.processor(
            text=text,
            images=images,
            videos=None,
            video_metadata=None,
            truncation=truncation,
            max_length=self.model.max_length,
            padding=True,
            do_resize=False,
            return_tensors="pt",
            **kwargs,
        )

    def _limit_pages(self, processed, inputs):
        """Remove whole trailing image blocks and their matching pixel patches on CPU."""
        import torch

        self.retained_indices = list(range(len(inputs)))
        if processed["attention_mask"].sum(dim=1).max().item() <= self.model.max_length:
            return processed
        processor = self.model.processor
        grids = processed["image_grid_thw"]
        patch_counts = grids.prod(dim=1).tolist()
        image_offset = patch_offset = 0
        tokens = []
        retained_grids = []
        retained_pixels = []
        self.retained_indices = []
        skipped = self.metadata.setdefault("skipped_pages", {})
        for row_index, row in enumerate(inputs):
            active = processed["attention_mask"][row_index].bool()
            ids = processed["input_ids"][row_index][active]
            starts = (ids == processor.vision_start_token_id).nonzero().flatten().tolist()
            ends = (ids == processor.vision_end_token_id).nonzero().flatten().tolist()
            assert len(starts) == len(ends) == len(row["images"])
            lengths = [end - start + 1 for start, end in zip(starts, ends, strict=True)]
            size = len(ids) - sum(lengths)  # Reserve the instruction and complete chat template.
            retained = 0
            for length in lengths:
                if size + length > self.model.max_length:
                    break
                size += length
                retained += 1
            if retained < len(lengths):
                skipped[row["doc_id"]] = list(range(retained + 1, len(lengths) + 1))
                print(
                    f"Qwen pages {row['doc_id']}: retained {retained}/{len(lengths)} pages; "
                    f"skipped trailing pages exceeding {self.model.max_length} tokens",
                    flush=True,
                )
            if retained:
                keep = torch.ones(len(ids), dtype=torch.bool)
                for start, end in zip(starts[retained:], ends[retained:], strict=True):
                    keep[start : end + 1] = False
                tokens.append(
                    {
                        key: value[row_index][active][keep].tolist()
                        for key, value in processed.items()
                        if key not in {"pixel_values", "image_grid_thw"}
                    }
                )
                self.retained_indices.append(row_index)
                retained_grids.append(grids[image_offset : image_offset + retained])
                retained_patches = sum(patch_counts[image_offset : image_offset + retained])
                retained_pixels.append(
                    processed["pixel_values"][patch_offset : patch_offset + retained_patches]
                )
            patch_offset += sum(patch_counts[image_offset : image_offset + len(lengths)])
            image_offset += len(lengths)
        assert image_offset == len(grids)
        assert patch_offset == len(processed["pixel_values"])
        if not tokens:
            return None
        result = dict(processor.tokenizer.pad(tokens, padding=True, return_tensors="pt"))
        result.update(
            image_grid_thw=torch.cat(retained_grids), pixel_values=torch.cat(retained_pixels)
        )
        assert result["input_ids"].shape[1] <= self.model.max_length
        return result

    def encode(self, values, task):
        import torch

        ids = (
            [row.get("doc_id", "unknown paper") for row in values]
            if task == "document"
            else [f"query batch ({len(values)} queries)"]
        )
        try:
            return self._encode(values, task)
        except torch.cuda.OutOfMemoryError as exc:
            raise torch.cuda.OutOfMemoryError(
                f"Qwen CUDA out of memory for {ids}: {exc}. Reduce model.batch_size "
                "(try 1); a single paper may still exceed GPU memory."
            ) from exc
        except (ValueError, RuntimeError, IndexError) as exc:
            raise type(exc)(f"Qwen input {ids}: {exc}") from exc

    def _encode(self, values, task):
        import torch

        if task == "document":
            inputs = values
        else:
            inputs = [
                {"text": text, "instruction": self.cfg.model.query_instruction} for text in values
            ]
        conversations = []
        for row in inputs:
            images = row.get("images", [])
            conversation = self.model.format_model_input(
                image=images[0] if images else None,
                text=row.get("text"),
                instruction=row.get("instruction"),
            )
            # The official formatter handles one image. Extend its user content
            # with identically formatted image blocks before the full text block.
            conversation[1]["content"][1:1] = [
                self.model.format_model_input(image=image)[1]["content"][0] for image in images[1:]
            ]
            conversations.append(conversation)
        pages = task == "document" and self.cfg.model.document_input == "pages"
        # Only the first QWEN_PAGE_LIMIT pages reach preprocessing. Shorten at image
        # boundaries if they still exceed the configured context length.
        preview = (
            self._preprocess(conversations, truncation=False)
            if pages
            else self._preprocess(conversations)
        )
        image_count = sum(len(row.get("images", [])) for row in inputs)
        if image_count and (
            "pixel_values" not in preview
            or "image_grid_thw" not in preview
            or len(preview["image_grid_thw"]) != image_count
        ):
            raise ValueError(
                "Official Qwen preprocessing did not retain every image modality input"
            )
        if pages:
            preview = self._limit_pages(preview, inputs)
            if preview is None:
                return torch.empty((0, self.cfg.model.dimension))
        # The pinned processor itself rejects truncated image-token sequences.
        # Reuse the processed CPU tensors with the official forward/pooling methods.
        # Calling process(inputs) here would repeat image processing and tokenization.
        processed = {key: value.to(self.model.model.device) for key, value in preview.items()}
        output = self.model.forward(processed)
        embeddings = self.model._pooling_last(output["last_hidden_state"], output["attention_mask"])
        return torch.nn.functional.normalize(embeddings, p=2, dim=-1)


def load_model(cfg):
    import torch

    torch.set_num_threads(cfg.launcher.threads.encoding)
    if cfg.model.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Visual encoding requires an available NVIDIA GPU for device=cuda")
    cache = Path(cfg.launcher.cache_root) / "models"
    encoder = OpenCLIPEncoder if cfg.model.name in OPENCLIP_MODELS else QwenEmbeddingEncoder
    return encoder(cfg, cache)


def _load_batch(rows, cfg, task, store):
    """Decode one batch on a single worker while the previous batch runs on the GPU."""
    from PIL import Image

    if task == "query":
        return [row["query"] for row in rows]
    values = []
    try:
        with ExitStack() as resources:
            read_bytes = resources.enter_context(store.reader()) if store is not None else None
            document = None
            pdf = None
            for row in rows:
                if cfg.model.name in QWEN_MODELS:
                    value = {"doc_id": row["doc_id"], "images": []}
                    # Own decoded images even if a later image fails to decode.
                    values.append(value)
                    if row["page_count"]:
                        import pypdfium2 as pdfium

                        from scientific_tot.integrations.colqwen import render_page

                        with pdfium.PdfDocument(row["pdf_path"]) as paper_pdf:
                            for number in range(1, row["page_count"] + 1):
                                value["images"].append(render_page(paper_pdf, number, cfg))
                    else:
                        for image in row["images"]:
                            data = read_bytes(image["image_path"])
                            with Image.open(BytesIO(data)) as source:
                                value["images"].append(source.convert("RGB"))
                    if "text" in row:
                        value["text"] = row["text"]
                    continue
                if cfg.model.get("document_input") == "figures":
                    data = read_bytes(row["image_path"])
                    with Image.open(BytesIO(data)) as source:
                        image = source.convert("RGB")
                else:
                    import pypdfium2 as pdfium

                    from scientific_tot.integrations.colqwen import render_page

                    if row["doc_id"] != document:
                        resources.close()
                        document = row["doc_id"]
                        pdf = resources.enter_context(pdfium.PdfDocument(row["pdf_path"]))
                    with closing(render_page(pdf, row["page_number"], cfg)) as rendered:
                        image = rendered.copy()
                values.append({"image": image})
        return values
    except BaseException:
        _close_images(values)
        raise


def _close_images(values):
    for value in values:
        for image in value.get("images", [value["image"]] if "image" in value else []):
            image.close()


def prefetched_batches(rows, cfg, task="document", store=None):
    """Keep two owned batches at most; close both on success, cancellation or failure."""
    size = cfg.model.batch_size
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="visual-input") as worker:
        pending = worker.submit(_load_batch, rows[:size], cfg, task, store) if rows else None
        try:
            for start in range(0, len(rows), size):
                values = pending.result()
                pending = None
                if start + size < len(rows):
                    pending = worker.submit(
                        _load_batch, rows[start + size : start + 2 * size], cfg, task, store
                    )
                try:
                    yield start, values
                finally:
                    if task == "document":
                        _close_images(values)
        finally:
            if pending is not None and not pending.cancel():
                values = pending.result()
                if task == "document":
                    _close_images(values)


def cached_encoding(cfg, *, task, rows, dataset_revision):
    """Stream bounded image batches into the existing NumPy/FAISS embedding cache."""
    import faiss
    import torch

    from scientific_tot.reporting import tqdm

    assert task in {"document", "query"} and (rows or task == "document")
    model_settings = OmegaConf.to_container(cfg.model, resolve=True)
    if cfg.model.name in QWEN_MODELS:
        model_settings["text_special_tokens"] = "escape"
    if task == "document":
        model_settings.pop("query_instruction", None)
        if cfg.model.name == "qwen3_vl_pages":
            model_settings["max_pages"] = QWEN_PAGE_LIMIT
    else:
        model_settings.pop("document_input")
        model_settings.pop("page_overflow", None)
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
        with tempfile.TemporaryDirectory(dir=cache) as folder:
            vectors = np.lib.format.open_memmap(
                Path(folder) / "embeddings.npy",
                mode="w+",
                dtype=np.float32,
                shape=(len(rows), cfg.model.dimension),
            )
            index = faiss.IndexFlatIP(cfg.model.dimension) if task == "document" else None
            written = 0
            encoded_rows = []
            store = None
            if task == "document" and cfg.model.document_input in {"figures", "multimodal"}:
                adapter = configured_adapter(cfg)
                store = adapter.artifacts
                if store is None:
                    # Standalone encoding has no preceding corpus selection.
                    store = Artifacts(adapter.dataset, tuple(cfg.dataset.domains), archives=True)
            if task == "document" and cfg.model.name in QWEN_MODELS:
                print(
                    f"Qwen document input={cfg.model.document_input}: {len(rows):,} papers, "
                    f"{sum(row['page_count'] for row in rows):,} rendered PDF pages, "
                    f"{sum(len(row.get('images', [])) for row in rows):,} supplied figures, "
                    f"{sum('text' in row for row in rows):,} full texts; one embedding per paper",
                    flush=True,
                )
            with phase(f"Encode {task} inputs on {cfg.model.device} ({cfg.model.precision})"):
                with closing(prefetched_batches(rows, cfg, task, store)) as batches:
                    for start, values in tqdm(
                        batches,
                        total=(len(rows) + cfg.model.batch_size - 1) // cfg.model.batch_size,
                        desc=f"Encoding {task} ({cfg.model.device})",
                        unit="batch",
                    ):
                        with torch.inference_mode():
                            encoded = encoder.encode(values, task).float()
                            retained = (
                                encoder.retained_indices
                                if task == "document" and cfg.model.name == "qwen3_vl_pages"
                                else range(len(values))
                            )
                            assert encoded.shape == (len(retained), cfg.model.dimension)
                            assert torch.isfinite(encoded).all()
                            assert (encoded.norm(dim=1) > 0).all()
                            encoded = torch.nn.functional.normalize(encoded, dim=1)
                            chunk = encoded.cpu().numpy()
                        assert np.allclose(np.linalg.norm(chunk, axis=1), 1, atol=1e-5)
                        vectors[written : written + len(retained)] = chunk
                        written += len(retained)
                        if index is not None:
                            encoded_rows.extend(rows[start + offset] for offset in retained)
                            index.add(chunk)
                        del encoded, chunk
            metadata = {
                **(encoder.metadata if encoder is not None else {}),
                "dimension": cfg.model.dimension,
                "normalized": True,
                "generation_seconds": perf_counter() - started,
                "units": written,
            }
            if cfg.model.name in QWEN_MODELS and task == "document":
                metadata.update(
                    embedding_level="paper",
                    document_input=cfg.model.document_input,
                    rendered_pages=sum(row["page_count"] for row in rows),
                    supplied_images=sum(len(row.get("images", [])) for row in rows),
                    full_texts=sum("text" in row for row in rows),
                )
            if cfg.model.name in OPENCLIP_MODELS and task == "query":
                metadata.update(
                    truncated_queries=encoder.truncated,
                    queries=len(rows),
                    truncation_proportion=encoder.truncated / len(rows),
                )
            _write_cache(
                entry,
                configuration,
                vectors[:written],
                metadata,
                index=index,
                ids=[row["docno"] for row in encoded_rows] if index is not None else None,
            )
            del vectors
    vectors, manifest = _read_cache(entry, configuration)
    pages = task == "document" and cfg.model.name == "qwen3_vl_pages"
    if vectors.shape[1] != cfg.model.dimension or (
        vectors.shape[0] > len(rows) if pages else vectors.shape[0] != len(rows)
    ):
        raise ValueError("Cached embedding dimensions/count differ from configured inputs")
    if task == "query" and cfg.model.name in OPENCLIP_MODELS:
        meta = manifest["encoding"]
        print(
            f"OpenCLIP truncated queries: {meta['truncated_queries']}/{len(rows)} "
            f"({meta['truncation_proportion']:.1%})",
            flush=True,
        )
    return vectors, manifest, entry, hit
