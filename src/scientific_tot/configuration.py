"""Resolve and validate Hydra settings without importing retrieval backends."""

from __future__ import annotations

import re
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

# The wheel contains the same YAML tree as the checkout, with no duplicated source.
CONFIG_DIR = Path(__file__).parent / "config"
if not CONFIG_DIR.is_dir():
    CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

STAGES = {"prepare", "index", "evaluate", "search", "export"}
SPLITS = {"train", "test", "train+test", "all"}
BACKENDS = {
    "bm25": "pyterrier",
    "dense": "sentence_transformers_faiss",
    "colbert": "colbert",
    "visual": "byaldi",
}
REPRESENTATIONS = {
    "title_abstract": ("text", "original_metadata"),
    "fulltext": ("text", "original_extracted_text"),
    "passages": ("text", "original_extracted_text"),
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
    return (path if path.is_absolute() else base / path).resolve()


def resolve_config(cfg: DictConfig, base_dir: Path, run_dir: Path | None = None) -> DictConfig:
    """Return a resolved copy, validate values, and normalize all managed paths.

    Does not check data/index existence: dry runs may describe another machine.
    Executing stages must check the actual artifacts before consuming them.
    """
    try:
        data = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    except OmegaConfBaseException as exc:
        raise ValueError(f"Cannot resolve configuration: {exc}") from exc
    _require(isinstance(data, dict), "Configuration must be a mapping")
    resolved = OmegaConf.create(data)
    try:
        _validate(resolved)
        base_dir = base_dir.resolve()
        for key in ("dataset_root", "cache_root", "output_root"):
            resolved.launcher[key] = str(absolute(resolved.launcher[key], base_dir))
        for key in ("root", "cache_dir"):
            resolved.dataset[key] = str(absolute(resolved.dataset[key], base_dir))
        for key, value in resolved.dataset.artifacts.items():
            resolved.dataset.artifacts[key] = str(absolute(value, base_dir))
        if resolved.paths.index is not None:
            resolved.paths.index = str(absolute(resolved.paths.index, base_dir))
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
        if run_dir is not None:
            writable.append(run_dir.resolve())
        for output in writable:
            for source in inputs:
                _require(
                    not output.is_relative_to(source) and not source.is_relative_to(output),
                    f"Writable path {output} overlaps dataset input {source}",
                )
        if run_dir is not None and resolved.paths.index is not None:
            index, output = Path(resolved.paths.index), run_dir.resolve()
            if resolved.stage.name != "index":
                _require(
                    not index.is_relative_to(output) and not output.is_relative_to(index),
                    "Existing paths.index must be separate from the new run directory",
                )
        if resolved.stage.name == "index" and resolved.paths.index is not None:
            index = Path(resolved.paths.index)
            for source in inputs:
                _require(
                    not index.is_relative_to(source) and not source.is_relative_to(index),
                    "paths.index overlaps dataset input",
                )
    except (OmegaConfBaseException, TypeError, AttributeError, KeyError) as exc:
        raise ValueError(f"Invalid configuration structure: {exc}") from exc
    OmegaConf.set_readonly(resolved, True)
    return resolved


def _validate(cfg: DictConfig) -> None:
    _require(type(cfg.dry_run) is bool, "dry_run must be a boolean")
    _require(type(cfg.seed) is int and cfg.seed >= 0, "seed must be a nonnegative integer")
    _require(cfg.stage.name in STAGES, f"Unsupported stage: {cfg.stage.name}")
    for key in ("name", "corpus", "queries"):
        _require(cfg.split[key] in SPLITS, f"Unsupported split.{key}: {cfg.split[key]}")
    dataset = cfg.dataset
    _require(dataset.source in {"local", "hub"}, "dataset.source must be local or hub")
    _require(
        _text(dataset.repo_id) and re.fullmatch(r"[^/\s]+/[^/\s]+", dataset.repo_id) is not None,
        "dataset.repo_id must be a Hugging Face owner/repository ID",
    )
    _require(_text(dataset.root), "dataset.root is required")
    _require(
        dataset.revision is None
        or isinstance(dataset.revision, str)
        and re.fullmatch(r"[0-9a-f]{40}", dataset.revision) is not None,
        "dataset.revision must be an immutable full 40-character commit SHA or null",
    )
    domains = dataset.domains
    _require(
        OmegaConf.is_list(domains) and len(domains) > 0, "dataset.domains must be a nonempty list"
    )
    _require(
        all(_text(d) and re.fullmatch(r"[A-Za-z0-9_-]+", d) for d in domains),
        "dataset.domains must contain plain domain names",
    )
    _require(len(set(domains)) == len(domains), "dataset.domains must be distinct")
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
    _require(cfg.launcher.kind == "local", "Only local launcher profiles are implemented")
    _require(cfg.launcher.device in {"cpu", "cuda", "mps"}, "Unsupported launcher.device")
    rep, model = cfg.representation, cfg.model
    _require(rep.name in REPRESENTATIONS, f"Unsupported representation: {rep.name}")
    _require(
        (rep.modality, rep.source) == REPRESENTATIONS[rep.name],
        "representation must use original document content with its declared modality",
    )
    _require(model.name in BACKENDS, f"Unsupported model: {model.name}")
    _require(model.backend == BACKENDS[model.name], f"Incorrect backend for model {model.name}")
    expected = "image" if model.name == "visual" else "text"
    _require(
        rep.modality == expected,
        f"model={model.name} is incompatible with representation={rep.name}; needs {expected}",
    )
    if rep.name == "passages":
        _require(
            rep.preprocessing.library == "langchain_text_splitters.RecursiveCharacterTextSplitter",
            "Unsupported passage splitter",
        )
        _require(_positive(rep.chunking.chunk_size), "chunking.chunk_size must be positive")
        _require(
            type(rep.chunking.chunk_overlap) is int
            and 0 <= rep.chunking.chunk_overlap < rep.chunking.chunk_size,
            "chunking.chunk_overlap must be nonnegative and smaller than chunk_size",
        )
    if rep.name == "pages":
        _require(rep.rendering.library == "pypdfium2", "Unsupported page rendering library")
        _require(
            type(rep.rendering.scale) in {int, float} and rep.rendering.scale > 0,
            "rendering.scale must be positive",
        )
    if model.name in {"dense", "visual"}:
        _require(_text(model.model_id), f"model.model_id is required for {model.name}")
        _require(model.device == cfg.launcher.device, "model.device must match launcher.device")
    if model.name == "dense":
        _require(
            model.revision is None or _text(model.revision), "model.revision must be null or text"
        )
        _require(_positive(model.batch_size), "model.batch_size must be positive")
        _require(
            model.indexing.factory == "Flat", "Only the verified FAISS Flat index is supported"
        )
        _require(model.indexing.metric in {"inner_product", "l2"}, "Unsupported FAISS metric")
        _require(
            type(model.encoding.normalize_embeddings) is bool,
            "model.encoding.normalize_embeddings must be boolean",
        )
    elif model.name == "colbert":
        _require(_text(model.checkpoint), "model.checkpoint is required for colbert")
        for key in ("index_bsize", "kmeans_niters", "doc_maxlen"):
            _require(_positive(model.indexing[key]), f"model.indexing.{key} must be positive")
        _require(model.indexing.nbits in {1, 2, 4}, "ColBERT nbits must be 1, 2, or 4")
        _require(_positive(model.retrieval.query_maxlen), "query_maxlen must be positive")
    elif model.name == "visual":
        _require(model.indexing.overwrite is False, "model.indexing.overwrite must remain false")
        for key in ("max_image_width", "max_image_height"):
            _require(
                model.indexing[key] is None or _positive(model.indexing[key]),
                f"model.indexing.{key} must be null or positive",
            )
    elif model.name == "bm25":
        _require(model.retrieval.wmodel == "BM25", "BM25 profile requires wmodel=BM25")
        _require(
            model.indexing.type == "classic", "BM25 profile requires a classic persistent index"
        )
        for key in ("bm25.b", "bm25.k_1"):
            value = model.retrieval.controls[key]
            _require(
                type(value) in {int, float} and value >= 0,
                f"model.retrieval.controls.{key} must be nonnegative",
            )
        _require(model.retrieval.controls["bm25.b"] <= 1, "BM25 b must be at most 1")
    _require(cfg.evaluation.backend == "ir_measures", "Unsupported evaluation backend")
    _require(_positive(cfg.evaluation.top_k), "evaluation.top_k must be positive")
    _require(
        OmegaConf.is_list(cfg.evaluation.metrics)
        and bool(cfg.evaluation.metrics)
        and all(_text(m) for m in cfg.evaluation.metrics),
        "evaluation.metrics must be a nonempty list of ir_measures expressions",
    )
    if cfg.stage.name in {"evaluate", "search"}:
        _require(_text(cfg.paths.index), f"stage={cfg.stage.name} requires an existing paths.index")
    if cfg.stage.name == "search":
        _require(_text(cfg.stage.query), "stage.query is required for search")
