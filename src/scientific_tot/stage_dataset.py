"""Pin a selective Hub snapshot and delegate canonical restoration to the producer."""

from __future__ import annotations

import os
import sys
from pathlib import Path, PurePosixPath

from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

from scientific_tot.stage_cli import plan


def stage(cfg) -> Path:
    if cfg.dataset.source != "hub" or not cfg.dataset.download.enabled:
        raise ValueError("Staging requires dataset=hub with dataset.download.enabled=true")
    if cfg.dataset.revision is None:
        raise ValueError("Pass dataset.revision=<full Hub commit SHA> for reproducible staging")
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        raise ValueError("Dataset staging downloads files; unset offline mode on the staging host")
    try:
        from dataset_packaging.download_hf_dataset import (
            BUILD_ARTIFACTS,
            allow_patterns,
            copy_stream,
            restore_archive_component,
            restore_plain_component,
            safe_destination,
        )
        from dataset_packaging.restore_hf_dataset import restore_dataset
    except ImportError as exc:
        raise ImportError(
            "Set TOT_GENERATION_REPO to the dataset generation checkout; "
            "only staging needs its packaging modules"
        ) from exc

    components = set(cfg.dataset.download.components)
    names = {
        "queries": "query_collections",
        "splits": "splits",
        "preprocessed": "preprocessed",
        "pdfs": "pdfs",
    }
    selected = {names[c] for c in components - {"metadata"}}
    patterns = allow_patterns(selected)
    if "metadata" in components and "metadata.parquet" not in patterns:
        patterns.append("metadata.parquet")
    root = Path(cfg.dataset.root)
    snapshot = (
        root
        / ".hf_dataset_download"
        / cfg.dataset.repo_id.replace("/", "--")
        / cfg.dataset.revision
    )
    print(
        f"Stage {cfg.dataset.repo_id}@{cfg.dataset.revision}: {sorted(components)} -> {root}",
        flush=True,
    )
    snapshot_download(
        repo_id=cfg.dataset.repo_id,
        repo_type="dataset",
        revision=cfg.dataset.revision,
        local_dir=snapshot,
        allow_patterns=patterns,
        ignore_patterns=BUILD_ARTIFACTS,
    )
    # Helpers compare existing files, skip matching bytes, reject changes, validate
    # TAR paths and publish restored files atomically. No --overwrite is implied.
    if {"metadata", "pdfs"} & components:
        metadata = snapshot / "metadata.parquet"
        with metadata.open("rb") as stream:
            copy_stream(
                stream,
                safe_destination(root, PurePosixPath("metadata.parquet")),
                overwrite=False,
                size=metadata.stat().st_size,
            )
    if "pdfs" in components:
        restore_dataset(
            dataset_dir=snapshot,
            target_dir=root / "pdf_datasets",
            resume=True,
            workers=cfg.launcher.threads.indexing,
        )
    for component in sorted(selected - {"pdfs"}):
        source = snapshot / component
        if not source.is_dir():
            raise FileNotFoundError(f"Missing downloaded component: {source}")
        restore = (
            restore_archive_component if component == "preprocessed" else restore_plain_component
        )
        restore(source, root / component, overwrite=False)
    return snapshot


def main() -> None:
    cfg, _ = plan("prepare", ["dataset=hub", *sys.argv[1:]])
    if cfg.dry_run:
        print(OmegaConf.to_yaml(cfg))
    else:
        stage(cfg)


if __name__ == "__main__":
    main()
