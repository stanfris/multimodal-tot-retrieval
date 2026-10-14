"""Explicit parameter bindings for existing implementations, without loading backends."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def bm25_arguments(cfg: DictConfig, *, docno_bytes: int) -> tuple[dict, dict]:
    """The caller converts classic to PyTerrier's IndexingType.CLASSIC enum."""
    assert cfg.model.name == "bm25"
    indexing = OmegaConf.to_container(cfg.model.indexing, resolve=True)
    if indexing["meta"]["docno"] < docno_bytes:
        raise ValueError("model.indexing.meta.docno would truncate corpus identifiers")
    indexing.update(text_attrs=["text"], threads=cfg.launcher.threads.indexing)
    retrieval = OmegaConf.to_container(cfg.model.retrieval, resolve=True)
    retrieval.update(num_results=cfg.retrieval.top_k, threads=cfg.launcher.threads.indexing)
    return indexing, retrieval


def dense_arguments(cfg: DictConfig) -> dict:
    """Map to EncodingConfig, whose fields feed ST, PyTorch, and FAISS calls."""
    assert cfg.model.name == "dense"
    if cfg.model.model_id.startswith(("/", "./", "../", "~")):
        raise NotImplementedError(
            "Existing dense model revision resolution requires a Hub model ID"
        )
    return {
        "model_id": cfg.model.model_id,
        "model_revision": cfg.model.revision,
        "device": cfg.model.device,
        "batch_size": cfg.model.batch_size,
        "profile": "model",  # YAML prompts are authoritative, including explicit empty strings.
        "query_prompt": cfg.model.encoding.query_prompt,
        "document_prompt": cfg.model.encoding.document_prompt,
        "similarity": "cosine" if cfg.model.encoding.normalize_embeddings else "dot",
        "faiss_threads": cfg.launcher.threads.faiss,
        "encoding_threads": cfg.launcher.threads.encoding,
    }


def require_corpus_settings(cfg: DictConfig, corpus, queries=None) -> None:
    """Do not apply settings to differently represented or selected library inputs."""
    expected_representation = corpus_arguments(cfg)["representation"]
    if {r["representation"] for r in corpus} != {expected_representation}:
        raise ValueError("Corpus representation differs from Hydra configuration")
    selection = corpus.provenance.get("selection", {})
    for key, expected in {"split": cfg.split.corpus, "missing": cfg.dataset.missing}.items():
        if selection.get(key) != expected:
            raise ValueError(f"Corpus {key} differs from Hydra configuration")

    configured_adapter(cfg)  # Validate the supported path/source contract, without I/O.
    for records in (corpus,) if queries is None else (corpus, queries):
        provenance = records.provenance
        if provenance.get("repo_id") != cfg.dataset.repo_id:
            raise ValueError("Record dataset repository differs from Hydra configuration")
        if cfg.dataset.revision is not None and provenance.get("revision") != cfg.dataset.revision:
            raise ValueError("Record dataset revision differs from Hydra configuration")
        local = provenance.get("local_dir")
        if cfg.dataset.source == "local":
            if local is None or Path(local).resolve() != Path(cfg.dataset.root):
                raise ValueError("Record dataset root differs from Hydra configuration")
        elif local is not None:
            raise ValueError("Hub configuration cannot consume local record provenance")
        selection = provenance.get("selection", {})
        if sorted(selection.get("domain", [])) != sorted(cfg.dataset.domains):
            raise ValueError("Record domains differ from Hydra configuration")
        if provenance.get("subset"):
            raise ValueError("Hydra profiles select complete splits; tutorial subsets are separate")
    if queries is not None:
        selection = queries.provenance.get("selection", {})
        if (
            selection.get("split") != cfg.split.queries
            or selection.get("mode") != cfg.dataset.query_mode
        ):
            raise ValueError("Query selection differs from Hydra configuration")


def configured_adapter(cfg: DictConfig):
    """Use the existing adapter directly; unsupported artifact relocation fails closed."""
    from scientific_tot import Dataset, DatasetAdapter

    canonical = {
        "preprocessed": "preprocessed",
        "metadata": "metadata.parquet",
        "queries": "query_collections",
        "splits": "splits/pdf_dataset_split.json",
        "pdfs": "pdf_datasets",
    }
    for key, suffix in canonical.items():
        if Path(cfg.dataset.artifacts[key]) != (Path(cfg.dataset.root) / suffix).resolve():
            raise NotImplementedError("The existing adapter requires canonical dataset.artifacts")
    if cfg.dataset.source == "hub" and not cfg.dataset.download.enabled:
        raise ValueError("Hub adapter reads require dataset.download.enabled=true")
    return DatasetAdapter(
        Dataset(
            repo_id=cfg.dataset.repo_id,
            revision=cfg.dataset.revision,
            local_dir=cfg.dataset.root if cfg.dataset.source == "local" else None,
            cache_dir=cfg.dataset.cache_dir,
        )
    )


def corpus_arguments(cfg: DictConfig) -> dict:
    representation = cfg.representation.name
    if representation == "passages" and cfg.model.name == "dense":
        representation = "fulltext"
    if representation not in {"fulltext", "title_abstract", "pages"}:
        raise NotImplementedError(f"representation={cfg.representation.name} is not implemented")
    return {
        "domain": cfg.dataset.domains[0],
        "split": cfg.split.corpus,
        "representation": representation,
        "missing": cfg.dataset.missing,
        "include_figures": False,
        "include_metadata": cfg.representation.name in {"title_abstract", "pages"},
    }


def query_arguments(cfg: DictConfig) -> dict:
    return {
        "domain": cfg.dataset.domains[0],
        "split": cfg.split.queries,
        "mode": cfg.dataset.query_mode,
    }
