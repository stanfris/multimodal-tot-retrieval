"""Stanford ColBERTv2 indexing/search with the shared canonical artifact contracts."""

import json
import os
from importlib.metadata import version
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from scientific_tot.config_arguments import (
    configured_adapter,
    query_arguments,
    require_corpus_settings,
)
from scientific_tot.config_artifacts import (
    artifact_reference,
    digest,
    require_index_reservation,
    verify_index,
    write_index_manifest,
)
from scientific_tot.integrations.bm25_batch import save_batch
from scientific_tot.integrations.pyterrier import to_pyterrier_queries
from scientific_tot.records import Records, Report, records_digest


def local_checkpoint_digest(path: Path) -> str:
    """Local checkpoint labels alone do not certify model bytes."""
    if not path.is_dir():
        raise ValueError("ColBERT local checkpoint must be a pretrained model directory")
    files = {
        str(file.relative_to(path)): artifact_reference(file)["sha256"]
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }
    if not files:
        raise ValueError("ColBERT checkpoint directory is empty")
    return digest(files)


def checkpoint_path(cfg) -> str:
    """ColBERT accepts paths, but has no revision argument: resolve an exact snapshot."""
    if Path(cfg.model.checkpoint).is_absolute():
        local_checkpoint_digest(Path(cfg.model.checkpoint))
        return cfg.model.checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(
        cfg.model.checkpoint,
        revision=cfg.model.revision,
        cache_dir=str(Path(cfg.launcher.cache_root) / "models"),
    )


def library_config(cfg, target: Path, *, retrieval=False):
    """Bind YAML parameters to the verified colbert-ai 0.2.22 configuration API."""
    import faiss
    import torch
    from colbert.infra import ColBERTConfig

    cuda = torch.cuda.is_available()
    if cfg.launcher.device == "cuda" and not cuda:
        raise RuntimeError("ColBERT CUDA execution requires an available NVIDIA GPU")
    if cfg.launcher.device == "cpu" and torch.cuda.device_count():
        raise RuntimeError("ColBERT CPU execution requires CUDA_VISIBLE_DEVICES='' before startup")
    torch.set_num_threads(cfg.launcher.threads.encoding)
    faiss.omp_set_num_threads(cfg.launcher.threads.faiss)
    # Keep native-extension compilation in reusable caches, outside sealed artifacts.
    os.environ["TORCH_EXTENSIONS_DIR"] = str(Path(cfg.launcher.cache_root) / "torch_extensions")
    arguments = OmegaConf.to_container(cfg.model.indexing, resolve=True)
    if retrieval:
        arguments.update(OmegaConf.to_container(cfg.model.retrieval, resolve=True))
    return ColBERTConfig(index_path=str(target / "colbert"), **arguments)


def run_config(cfg):
    from colbert.infra import RunConfig

    return RunConfig(
        root=cfg.launcher.cache_root,
        experiment="colbert",
        name="baseline",
        nranks=1,
        gpus=1 if cfg.launcher.device == "cuda" else 0,
        avoid_fork_if_possible=True,
    )


def build_colbert_index(corpus: Records, inputs, cfg, *, reserved=False) -> dict:
    """Index one truncated fulltext per paper; neural and compressed index logic is external."""
    require_corpus_settings(cfg, corpus)
    target = Path(cfg.paths.index)
    if reserved:
        require_index_reservation(target)
    else:
        target.mkdir(parents=True, exist_ok=False)
    config = library_config(cfg, target)
    checkpoint = checkpoint_path(cfg)
    from colbert import Indexer
    from colbert.infra import Run

    # In-memory collection positions are library PIDs; retain their exact canonical order.
    mapping = [{"doc_id": row["doc_id"], "domain": row["domain"]} for row in corpus]
    with Run().context(run_config(cfg), inherit_config=False):
        Indexer(checkpoint=checkpoint, config=config).index(
            name="colbert", collection=[row["text"] for row in inputs.documents], overwrite=False
        )
    metadata = json.loads((target / "colbert/metadata.json").read_text())
    lengths = [
        length
        for path in sorted((target / "colbert").glob("doclens.*.json"))
        for length in json.loads(path.read_text())
    ]
    if len(lengths) != len(mapping) or metadata["num_embeddings"] != sum(lengths):
        raise ValueError("ColBERT index document/embedding count differs from selected corpus")
    (target / "ids.json").write_text(json.dumps(mapping, ensure_ascii=False) + "\n")
    provenance = {
        "dataset_revision": corpus.provenance["revision"],
        "model_revision": cfg.model.revision,
        "model_identifier": cfg.model.checkpoint,
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": local_checkpoint_digest(Path(checkpoint)),
        "corpus_sha256": records_digest(inputs.documents),
        "corpus_provenance": corpus.provenance,
        "local_revision_verified": corpus.provenance["local_revision_verified"],
        "documents": len(mapping),
        "document_domains": {row["doc_id"]: row["domain"] for row in mapping},
        "truncation": {
            "doc_maxlen": cfg.model.indexing.doc_maxlen,
            "unit": "WordPiece tokens including CLS, D marker and SEP",
            "policy": "prefix truncation by ColBERT DocTokenizer; no passage splitting",
        },
        "versions": {
            name: version(name)
            for name in ("colbert-ai", "transformers", "torch", "faiss-cpu", "numpy")
        },
    }
    write_index_manifest(target, cfg, provenance=provenance)
    return provenance


def retrieve_colbert(cfg, target: Path) -> dict:
    """Search a verified saved index without loading or encoding the original corpus."""
    location = Path(cfg.paths.index)
    manifest = verify_index(location, cfg)
    provenance = manifest["provenance"]
    mapping = json.loads((location / "ids.json").read_text())
    domains = provenance["document_domains"]
    if (
        len(mapping) != provenance["documents"]
        or len(domains) != len(mapping)
        or {row["doc_id"]: row["domain"] for row in mapping} != domains
        or any(domain not in cfg.dataset.domains for domain in domains.values())
    ):
        raise ValueError("ColBERT PID mapping differs from canonical paper identities/domains")
    corpus = Records(
        "corpus", mapping, Report(emitted=len(mapping)), provenance["corpus_provenance"]
    )
    adapter = configured_adapter(cfg)
    queries = adapter.load_queries(**query_arguments(cfg))
    inputs = to_pyterrier_queries(corpus, queries)
    config = library_config(cfg, location, retrieval=True)
    checkpoint = checkpoint_path(cfg)
    if local_checkpoint_digest(Path(checkpoint)) != provenance["checkpoint_sha256"]:
        raise ValueError("ColBERT checkpoint bytes differ from the indexed model")
    from colbert import Searcher
    from colbert.infra import Run

    rows = []
    with Run().context(run_config(cfg), inherit_config=False):
        # Supply dummy text to avoid reloading/storing the original collection in search.
        searcher = Searcher(
            index=str(location / "colbert"),
            checkpoint=checkpoint,
            collection=[""] * len(mapping),
            config=config,
        )
        for topic in inputs.topics.itertuples():
            pids, ranks, scores = searcher.search(topic.query, k=cfg.retrieval.top_k)
            # Searcher returns k rank positions even when fewer than k PIDs are found.
            if len(pids) != len(scores) or len(ranks) < len(pids):
                raise ValueError("ColBERT returned inconsistent ranking lengths")
            for pid, rank, score in zip(pids, ranks[: len(pids)], scores, strict=True):
                if type(pid) is not int or not 0 <= pid < len(mapping):
                    raise ValueError("ColBERT returned an unknown PID")
                rows.append(
                    {
                        "qid": topic.qid,
                        "docno": mapping[pid]["doc_id"],
                        "rank": rank,
                        "score": score,
                    }
                )
    run = pd.DataFrame(rows, columns=["qid", "docno", "rank", "score"])
    query_mapping = inputs.query_ids.copy()
    query_mapping["domain"] = query_mapping.query_key.map(
        {row["query_key"]: row["domain"] for row in queries}
    )
    outputs = save_batch(run, inputs, query_mapping, pd.DataFrame(mapping), cfg, target)
    return {
        "inputs": {
            "index_manifest": artifact_reference(location / "index-manifest.json"),
            "query_files": [
                {"dataset_path": name, **artifact_reference(adapter.dataset.get_file(name))}
                for name in queries.provenance["selection"]["files"]
            ],
        },
        "outputs": outputs,
        "queries": len(inputs.topics),
        "qrels": len(inputs.qrels),
        "rankings": len(run),
        "query_provenance": queries.provenance,
        "model_revision": provenance["model_revision"],
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "retrieval_cutoff": cfg.retrieval.top_k,
        "ranking_level": "paper",
        "versions": provenance["versions"],
    }
