"""Resolve and validate Hydra settings without importing retrieval backends."""

from __future__ import annotations

import math
import re
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from scientific_tot.data.schemas import DOMAINS

# The wheel contains the same YAML tree as the checkout, with no duplicated source.
CONFIG_DIR = Path(__file__).parent / "config"
if not CONFIG_DIR.is_dir():
    CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

STAGES = {"prepare", "index", "retrieve", "evaluate", "search", "export"}
SPLITS = {"train", "test", "train+test", "all"}
OPENCLIP_MODELS = {"openclip", "openclip_figures"}
QWEN_MODELS = {"qwen3_vl_pages", "qwen3_vl_figures", "qwen3_vl_text", "qwen3_vl_multimodal"}
VISUAL_DENSE_MODELS = OPENCLIP_MODELS | QWEN_MODELS
BACKENDS = {
    "bm25": "pyterrier",
    "dense": "sentence_transformers_faiss",
    "colbert": "colbert",
    "visual": "byaldi",
    "colqwen": "sentence_transformers_qdrant",
    "openclip": "visual_faiss",
    "openclip_figures": "visual_faiss",
    "qwen3_vl_text": "visual_faiss",
    "qwen3_vl_multimodal": "visual_faiss",
    "qwen3_vl_pages": "visual_faiss",
    "qwen3_vl_figures": "visual_faiss",
}
REPRESENTATIONS = {
    "title_abstract": ("text", "original_metadata"),
    "fulltext": ("text", "original_extracted_text"),
    "figures": ("image", "original_extracted_figures"),
    "pages": ("image", "original_pdf_pages"),
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _positive(value: object) -> bool:
    return type(value) is int and value > 0


def absolute(value: str, base: Path) -> Path:
    """Resolve relative paths once, against the invocation directory."""
    _require(_text(value) and "\x00" not in value, "Filesystem paths must be nonempty strings")
    path = Path(value).expanduser()
    return (base / path).resolve()


def resolve_config(cfg: DictConfig, base_dir: Path, run_dir: Path | None = None) -> DictConfig:
    """Return a resolved copy, validate values, and normalize all managed paths.

    Does not check data/index existence: dry runs may describe another machine.
    Executing stages must check the actual artifacts before consuming them.
    """
    data = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    _require(isinstance(data, dict), "Configuration must be a mapping")
    resolved = OmegaConf.create(data)
    _validate(resolved)
    base_dir = base_dir.resolve()
    run_dir = run_dir.resolve() if run_dir is not None else None
    if resolved.get("test_mode", False):
        resolved.test_root = str(absolute(resolved.test_root, base_dir))
    for key in ("dataset_root", "cache_root", "output_root"):
        resolved.launcher[key] = str(absolute(resolved.launcher[key], base_dir))
    if resolved.launcher.kind == "pbs":
        resolved.launcher.log_root = str(absolute(resolved.launcher.log_root, base_dir))
    for key in ("root", "cache_dir"):
        resolved.dataset[key] = str(absolute(resolved.dataset[key], base_dir))
    # Explicit local checkpoint references share the original invocation base.
    model_key = "checkpoint" if resolved.model.name == "colbert" else "model_id"
    reference = resolved.model.get(model_key)
    if isinstance(reference, str) and reference.startswith(("/", "./", "../", "~")):
        resolved.model[model_key] = str(absolute(reference, base_dir))
    for key, value in resolved.dataset.artifacts.items():
        resolved.dataset.artifacts[key] = str(absolute(value, base_dir))
    for key in ("index", "run"):
        if resolved.paths[key] is not None:
            resolved.paths[key] = str(absolute(resolved.paths[key], base_dir))
    if resolved.get("test_mode", False):
        root = Path(resolved.test_root)
        for value in (
            resolved.launcher.output_root,
            resolved.launcher.cache_root,
            resolved.launcher.get("log_root"),
            resolved.dataset.cache_dir,
            resolved.paths.index,
            resolved.paths.run,
            str(run_dir) if run_dir is not None else None,
        ):
            _require(
                value is None or Path(value).is_relative_to(root),
                "Test mode requires outputs, caches and artifact inputs inside test_root",
            )
    if run_dir is not None and resolved.stage.name == "index" and resolved.paths.index is None:
        resolved.paths.index = str(run_dir)
    inputs = [
        Path(resolved.dataset.root),
        *(Path(p) for p in resolved.dataset.artifacts.values()),
    ]
    # Output/cache writes must not land in data, nor contain it (including symlinks).
    writable = [
        Path(resolved.launcher.output_root),
        Path(resolved.dataset.cache_dir),
        Path(resolved.launcher.cache_root),
    ]
    if resolved.launcher.kind == "pbs":
        writable.append(Path(resolved.launcher.log_root))
    if run_dir is not None:
        writable.append(run_dir)
    for output in writable:
        for source in inputs:
            _require(
                not overlaps(output, source),
                f"Writable path {output} overlaps dataset input {source}",
            )
    caches = [Path(resolved.launcher.cache_root), Path(resolved.dataset.cache_dir)]
    outputs = [Path(resolved.launcher.output_root)]
    if resolved.launcher.kind == "pbs":
        outputs.append(Path(resolved.launcher.log_root))
    if run_dir is not None:
        outputs.append(run_dir)
    if resolved.stage.name == "index" and resolved.paths.index is not None:
        outputs.append(Path(resolved.paths.index))
    for output in outputs:
        for cache in caches:
            _require(not overlaps(output, cache), "Experiment output and reusable cache overlap")
    if resolved.paths.run is not None and run_dir is not None:
        _require(
            not overlaps(Path(resolved.paths.run), run_dir),
            "Existing paths.run must be separate from the new run directory",
        )
    if run_dir is not None and resolved.paths.index is not None and resolved.stage.name != "index":
        _require(
            not overlaps(Path(resolved.paths.index), run_dir),
            "Existing paths.index must be separate from the new run directory",
        )
    if resolved.stage.name == "index" and resolved.paths.index is not None:
        index = Path(resolved.paths.index)
        if run_dir is not None:
            _require(
                run_dir == index or not run_dir.is_relative_to(index),
                "Index destination must not contain the run directory",
            )
        for source in inputs:
            _require(
                not overlaps(index, source),
                "paths.index overlaps dataset input",
            )
    OmegaConf.set_readonly(resolved, True)
    return resolved


def _validate(cfg: DictConfig) -> None:
    _require(type(cfg.dry_run) is bool, "dry_run must be a boolean")
    _require(type(cfg.get("test_mode", False)) is bool, "test_mode must be a boolean")
    if cfg.get("test_mode", False):
        _require(_text(cfg.test_root), "test_root must be a nonempty path")
        _require(
            cfg.split.corpus == cfg.split.queries == "test",
            "Test mode requires split.corpus=test and split.queries=test",
        )
    _require(
        _text(cfg.run.name) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+-]*", cfg.run.name),
        "run.name must be a plain filesystem-safe experiment name",
    )
    _require(cfg.stage.name in STAGES, f"Unsupported stage: {cfg.stage.name}")
    for key in ("name", "corpus", "queries"):
        _require(cfg.split[key] in SPLITS, f"Unsupported split.{key}: {cfg.split[key]}")
    dataset = cfg.dataset
    _require(dataset.source in {"local", "hub"}, "dataset.source must be local or hub")
    _require(
        _text(dataset.repo_id) and re.fullmatch(r"[^/\s]+/[^/\s]+", dataset.repo_id) is not None,
        "dataset.repo_id must be a Hugging Face owner/repository ID",
    )
    _require(
        dataset.revision is None
        or isinstance(dataset.revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", dataset.revision) is not None,
        "dataset.revision must be an immutable full 40-character commit SHA or null",
    )
    domains = dataset.domains
    _require(
        OmegaConf.is_list(domains) and len(domains) == 1,
        "dataset.domains must select exactly one domain",
    )
    _require(domains[0] in DOMAINS, f"dataset.domains must select one of {DOMAINS}")
    _require(dataset.missing in {"error", "skip"}, "dataset.missing must be error or skip")
    _require(
        dataset.query_mode in {"visual-and-text", "visual-only"}, "Unsupported dataset.query_mode"
    )
    _require(type(dataset.download.enabled) is bool, "dataset.download.enabled must be boolean")
    _require(
        not dataset.download.enabled or dataset.source == "hub",
        "dataset.download.enabled requires dataset.source=hub; local data is read in place",
    )
    components = dataset.download.components
    _require(
        OmegaConf.is_list(components)
        and bool(components)
        and all(c in {"metadata", "splits", "queries", "preprocessed", "pdfs"} for c in components),
        "Unsupported or empty dataset.download.components",
    )
    _require(
        set(dataset.artifacts) == {"preprocessed", "metadata", "queries", "splits", "pdfs"},
        "dataset.artifacts requires preprocessed, metadata, queries, splits, and pdfs paths",
    )
    for key in ("indexing", "encoding", "faiss"):
        _require(_positive(cfg.launcher.threads[key]), f"launcher.threads.{key} must be positive")
    _require(_text(cfg.launcher.terrier_version), "launcher.terrier_version must be text")
    _require(cfg.launcher.kind in {"local", "pbs"}, "launcher.kind must be local or pbs")
    if cfg.launcher.kind == "pbs":
        launcher = cfg.launcher
        for key in ("project", "queue"):
            _require(
                isinstance(launcher[key], str) and re.fullmatch(r"[A-Za-z0-9_.@-]+", launcher[key]),
                f"launcher.{key} must be a PBS identifier",
            )
        _require(_positive(launcher.cpus), "launcher.cpus must be positive")
        _require(type(launcher.gpus) is int and launcher.gpus == 1, "PBS requires one GPU")
        _require(
            isinstance(launcher.memory, str)
            and re.fullmatch(r"[1-9][0-9]*(kb|mb|gb|tb)", launcher.memory),
            "launcher.memory must be a positive PBS size such as 160gb",
        )
        _require(
            isinstance(launcher.walltime, str)
            and re.fullmatch(r"[0-9]+:[0-5][0-9]:[0-5][0-9]", launcher.walltime),
            "launcher.walltime must be HH:MM:SS",
        )
        for key in ("log_root", "python", "colqwen_python"):
            _require(_text(launcher[key]), f"launcher.{key} must be a nonempty path")
        _require(OmegaConf.is_dict(launcher.environment), "launcher.environment must be a mapping")
        for key, value in launcher.environment.items():
            _require(
                isinstance(key, str)
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key)
                and type(value) in {str, int, float, bool}
                and "\x00" not in str(value),
                "launcher.environment requires valid variable names and scalar values",
            )
    _require(cfg.launcher.device in {"cpu", "cuda", "mps"}, "Unsupported launcher.device")
    rep, model = cfg.representation, cfg.model
    _require(rep.name in REPRESENTATIONS, f"Unsupported representation: {rep.name}")
    _require(
        (rep.modality, rep.source) == REPRESENTATIONS[rep.name],
        "representation must use original document content with its declared modality",
    )
    _require(model.name in BACKENDS, f"Unsupported model: {model.name}")
    _require(model.backend == BACKENDS[model.name], f"Incorrect backend for model {model.name}")
    expected = (
        "image"
        if model.name
        in {"visual", "colqwen", "qwen3_vl_pages", "qwen3_vl_figures"} | OPENCLIP_MODELS
        else "text"
    )
    _require(
        rep.modality == expected,
        f"model={model.name} is incompatible with representation={rep.name}; needs {expected}",
    )
    if rep.name == "title_abstract":
        _require(
            list(rep.fields) == ["title", "abstract"],
            "Existing adapter requires representation.fields=[title,abstract]",
        )
    if rep.name == "figures":
        _require(type(rep.include_captions) is bool, "include_captions must be boolean")
    if rep.name == "pages":
        _require(rep.rendering.library == "pypdfium2", "Unsupported page rendering library")
        _require(
            type(rep.rendering.scale) in {int, float}
            and math.isfinite(rep.rendering.scale)
            and rep.rendering.scale > 0,
            "rendering.scale must be positive",
        )
    if model.name in {"dense", "visual", "colqwen"} | VISUAL_DENSE_MODELS:
        _require(_text(model.model_id), f"model.model_id is required for {model.name}")
        _require(model.device == cfg.launcher.device, "model.device must match launcher.device")
    if model.name in VISUAL_DENSE_MODELS:
        strategies = {
            "openclip": "pages",
            "openclip_figures": "figures",
            "qwen3_vl_text": "text",
            "qwen3_vl_multimodal": "multimodal",
            "qwen3_vl_pages": "pages",
            "qwen3_vl_figures": "figures",
        }
        _require(
            model.document_input == strategies[model.name], "Incorrect document input strategy"
        )
        expected_rep = {
            "pages": "pages",
            "figures": "figures",
            "text": "fulltext",
            "multimodal": "fulltext",
        }[model.document_input]
        _require(rep.name == expected_rep, f"This strategy requires representation={expected_rep}")
        if rep.name == "figures":
            _require(not rep.include_captions, "Figure encoding must not include captions")
        _require(model.device in {"cpu", "cuda"}, "Visual FAISS supports CPU or CUDA")
        _require(_positive(model.batch_size), "model.batch_size must be positive")
        _require(model.precision in {"fp32", "fp16", "bf16"}, "Unsupported model.precision")
        _require(
            model.device != "cpu" or model.precision == "fp32",
            "CPU inference requires model.precision=fp32",
        )
        _require(
            isinstance(model.revision, str) and re.fullmatch(r"[0-9a-f]{40}", model.revision),
            "Visual FAISS requires model.revision pinned to an immutable commit",
        )
        _require(
            model.indexing.factory == "Flat" and model.indexing.metric == "inner_product",
            "Visual FAISS requires IndexFlatIP",
        )
        if model.name in OPENCLIP_MODELS:
            _require(model.architecture == "ViT-B-32", "OpenCLIP requires ViT-B-32")
            _require(model.pretrained == "laion2b_s34b_b79k", "Incorrect OpenCLIP weights")
            _require(model.dimension == 512, "OpenCLIP ViT-B-32 has dimension 512")
        else:
            _require(model.model_id == "Qwen/Qwen3-VL-Embedding-2B", "Incorrect Qwen checkpoint")
            _require(model.dimension == 2048, "Use Qwen's default 2048 dimensions")
            _require(model.truncation_side == "right", "Qwen requires right-side truncation")
            _require(
                _positive(model.max_length) and model.max_length <= 32768,
                "Qwen max_length must be positive and at most 32768",
            )
            _require(_text(model.query_instruction), "Qwen query instruction is required")
    elif model.name == "dense":
        _require(
            model.revision is None or _text(model.revision), "model.revision must be null or text"
        )
        if cfg.stage.name in {"retrieve", "search"}:
            _require(
                isinstance(model.revision, str)
                and re.fullmatch(r"[0-9a-f]{40}", model.revision) is not None,
                "Dense index consumers require model.revision pinned to the build commit",
            )
        _require(_positive(model.batch_size), "model.batch_size must be positive")
        _require(
            model.indexing.factory == "Flat", "Only the verified FAISS Flat index is supported"
        )
        _require(
            model.indexing.metric == "inner_product",
            "Existing FAISS implementation supports inner_product only",
        )
        _require(
            type(model.encoding.normalize_embeddings) is bool,
            "model.encoding.normalize_embeddings must be boolean",
        )
        for task in ("query", "document"):
            _require(
                isinstance(model.encoding[f"{task}_prompt"], str),
                f"model.encoding.{task}_prompt must be a string (empty is allowed)",
            )
    elif model.name == "colqwen":
        _require(model.device == "cuda", "ColQwen encoding requires launcher.device=cuda")
        _require(rep.name == "pages", "ColQwen requires representation=pages")
        _require(model.model_id == "vidore/colqwen2.5-v0.2", "Only ColQwen2.5 v0.2 is supported")
        _require(model.base_model_id == "vidore/colqwen2.5-base", "Incorrect ColQwen base model")
        for key in ("revision", "base_revision"):
            _require(
                isinstance(model[key], str) and re.fullmatch(r"[0-9a-f]{40}", model[key]),
                f"model.{key} requires an immutable commit",
            )
        _require(_positive(model.batch_size), "model.batch_size must be positive")
        _require(
            model.retrieval.matmul_precision in {"highest", "high", "medium"},
            "ColQwen matmul_precision must be highest, high or medium",
        )
        _require(_positive(model.retrieval.batch_size), "query batch_size must be positive")
        for key in ("scoring_batch_size", "page_batch_size", "scoring_chunk_elements"):
            _require(_positive(model.retrieval[key]), f"model.retrieval.{key} must be positive")
        _require(model.indexing.distance == "dot", "ColQwen index requires dot distance")
        _require(model.indexing.comparator == "max_sim", "ColQwen index requires max_sim")
        _require(model.indexing.exact is True, "ColQwen requires exact page search")
        _require(model.processor == "checkpoint", "Use the official checkpoint processor")
    elif model.name == "colbert":
        _require(rep.name == "fulltext", "ColBERT requires representation=fulltext")
        _require(cfg.launcher.device in {"cpu", "cuda"}, "ColBERT supports CPU or CUDA")
        _require(
            isinstance(model.revision, str) and re.fullmatch(r"[0-9a-f]{40}", model.revision),
            "ColBERT requires model.revision pinned to an immutable commit",
        )
        _require(_text(model.checkpoint), "model.checkpoint is required for colbert")
        for key in ("index_bsize", "kmeans_niters", "doc_maxlen"):
            _require(_positive(model.indexing[key]), f"model.indexing.{key} must be positive")
        _require(
            type(model.indexing.nbits) is int and model.indexing.nbits in {1, 2, 4},
            "ColBERT nbits must be 1, 2, or 4",
        )
        _require(_positive(model.retrieval.query_maxlen), "query_maxlen must be positive")
    elif model.name == "visual":
        _require(
            type(model.indexing.store_collection_with_index) is bool,
            "store_collection_with_index must be boolean",
        )
        _require(model.indexing.overwrite is False, "model.indexing.overwrite must remain false")
        for key in ("max_image_width", "max_image_height"):
            _require(
                model.indexing[key] is None or _positive(model.indexing[key]),
                f"model.indexing.{key} must be null or positive",
            )
    elif model.name == "bm25":
        _require(type(model.indexing.blocks) is bool, "model.indexing.blocks must be boolean")
        _require(
            set(model.indexing.meta) == {"docno"} and _positive(model.indexing.meta.docno),
            "model.indexing.meta requires a positive docno capacity",
        )
        _require(model.retrieval.wmodel == "BM25", "BM25 profile requires wmodel=BM25")
        _require(
            model.indexing.type == "classic", "BM25 profile requires a classic persistent index"
        )
        for key in ("bm25.b", "bm25.k_1"):
            value = model.retrieval.controls[key]
            _require(
                type(value) in {int, float} and math.isfinite(value) and value >= 0,
                f"model.retrieval.controls.{key} must be nonnegative",
            )
        _require(model.retrieval.controls["bm25.b"] <= 1, "BM25 b must be at most 1")
    _require(cfg.evaluation.backend == "ir_measures", "Unsupported evaluation backend")
    _require(_positive(cfg.retrieval.batch_size), "retrieval.batch_size must be positive")
    _require(
        cfg.retrieval.search_device in {"auto", "cpu", "cuda"},
        "retrieval.search_device must be auto, cpu or cuda",
    )
    if model.name == "dense":
        _require(model.precision in {"fp32", "fp16", "bf16"}, "Unsupported model.precision")
    _require(_positive(cfg.retrieval.top_k), "retrieval.top_k must be positive")
    _require(
        OmegaConf.is_list(cfg.evaluation.metrics)
        and bool(cfg.evaluation.metrics)
        and all(_text(m) for m in cfg.evaluation.metrics),
        "evaluation.metrics must be a nonempty list of ir_measures expressions",
    )
    metrics = list(cfg.evaluation.metrics)
    _require(len(set(metrics)) == len(metrics), "evaluation.metrics must be distinct")
    # A documented subset of ir_measures, validated without optional imports.
    _require(
        all(re.fullmatch(r"(nDCG|RR|R|P|AP)@[1-9][0-9]*", m) for m in metrics),
        "Unsupported evaluation.metrics; use nDCG, RR, R, P, or AP with positive @cutoffs",
    )
    if cfg.stage.name in {"retrieve", "search"}:
        if dataset.source == "hub":
            _require(
                dataset.revision is not None,
                "Hub index consumers require dataset.revision pinned to the build commit",
            )
        _require(_text(cfg.paths.index), f"stage={cfg.stage.name} requires an existing paths.index")
    if cfg.stage.name == "search":
        _require(_text(cfg.stage.query), "stage.query is required for search")

    if cfg.stage.name == "evaluate":
        _require(_text(cfg.paths.run), "stage=evaluate requires a completed retrieval paths.run")
    else:
        _require(cfg.paths.run is None, "paths.run is only consumed by stage=evaluate")
    if cfg.stage.name not in {"index", "retrieve", "search"}:
        _require(cfg.paths.index is None, "paths.index is only consumed by index/retrieve/search")


def overlaps(first: Path, second: Path) -> bool:
    return first.is_relative_to(second) or second.is_relative_to(first)
