"""Stanford ColBERTv2 indexing/search with the shared canonical artifact contracts."""

import json
import os
import re
import shutil
import subprocess
import sys
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
from scientific_tot.reporting import package_version, phase, tqdm


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


def prepare_cuda(cfg) -> None:
    """Resolve the native toolchain and load ColBERT kernels before reading the corpus."""
    import torch

    if cfg.launcher.device != "cuda":
        raise ValueError("ColBERT indexing and retrieval require launcher.device=cuda")
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("ColBERT requires an available NVIDIA GPU; CUDA is unavailable")
    # Direct venv Python invocations do not put its Ninja executable on PATH.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    from torch.utils import cpp_extension

    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if not cuda_home:
        nvcc = shutil.which("nvcc")
        cuda_home = str(Path(nvcc).resolve().parent.parent) if nvcc else cpp_extension.CUDA_HOME
    if not cuda_home or not (Path(cuda_home) / "bin/nvcc").is_file():
        raise RuntimeError(
            "ColBERT requires a CUDA compiler toolkit, not just the PyTorch runtime. "
            f"Load a toolkit compatible with torch.version.cuda={torch.version.cuda} and "
            "set CUDA_HOME (or CUDA_PATH) to its root containing bin/nvcc before submission. "
            "Alternatively pass +launcher.environment.CUDA_HOME=/path/to/cuda with launcher=h200."
        )
    cuda_home = str(Path(cuda_home).expanduser().resolve())
    os.environ["CUDA_HOME"] = cuda_home
    # PyTorch caches this at import time; another library may have imported it
    # before the toolkit environment was configured.
    cpp_extension.CUDA_HOME = cuda_home
    for executable in (os.environ.get("CXX", "c++"), "ninja"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"ColBERT CUDA extension compilation requires {executable} on PATH")
    result = subprocess.run(
        [str(Path(cuda_home) / "bin/nvcc"), "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    release = re.search(r"\brelease (\d+\.\d+)", result.stdout)
    if release is None:
        raise RuntimeError(f"Cannot determine CUDA toolkit version from nvcc: {result.stdout}")
    if release.group(1).split(".")[0] != torch.version.cuda.split(".")[0]:
        raise RuntimeError(
            f"CUDA toolkit {release.group(1)} at {cuda_home} is incompatible with "
            f"torch.version.cuda={torch.version.cuda}; load a matching toolkit."
        )
    print(
        f"ColBERT CUDA_HOME={cuda_home}; toolkit={release.group(1)}; "
        f"Torch CUDA={torch.version.cuda}",
        flush=True,
    )
    os.environ["TORCH_EXTENSIONS_DIR"] = str(Path(cfg.launcher.cache_root) / "torch_extensions")
    os.environ.setdefault("MAX_JOBS", "4")
    from colbert.indexing.codecs.residual import ResidualCodec

    ResidualCodec.try_load_torch_extensions(True)


def library_config(cfg, target: Path, *, retrieval=False):
    """Bind YAML parameters to the verified colbert-ai 0.2.22 configuration API."""
    import faiss
    import torch
    from colbert.infra import ColBERTConfig

    torch.set_num_threads(cfg.launcher.threads.encoding)
    faiss.omp_set_num_threads(cfg.launcher.threads.faiss)
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
        gpus=1,
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
        "corpus_report": corpus.report.to_dict(),
        "local_revision_verified": corpus.provenance["local_revision_verified"],
        "documents": len(mapping),
        "document_domains": {row["doc_id"]: row["domain"] for row in mapping},
        "truncation": {
            "doc_maxlen": cfg.model.indexing.doc_maxlen,
            "unit": "WordPiece tokens including CLS, D marker and SEP",
            "policy": "prefix truncation by ColBERT DocTokenizer; no passage splitting",
        },
        "versions": {
            name: package_version(name)
            for name in ("colbert-ai", "transformers", "torch", "faiss", "numpy")
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
        topics = list(inputs.topics.itertuples())
        size = cfg.retrieval.batch_size
        with (
            phase("Encode queries in batches and score ColBERT candidates"),
            tqdm(
                total=len(topics), desc="Retrieving queries (CUDA ColBERT)", unit="query"
            ) as progress,
        ):
            for start in range(0, len(topics), size):
                chunk = topics[start : start + size]
                encoded = searcher.encode([topic.query for topic in chunk])
                for offset, topic in enumerate(chunk):
                    pids, ranks, scores = searcher.dense_search(
                        encoded[offset : offset + 1], k=cfg.retrieval.top_k
                    )
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
                    progress.update(1)
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
