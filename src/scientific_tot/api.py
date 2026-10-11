"""Convenience functions; use DatasetAdapter for a shared pinned source."""

from pathlib import Path

from scientific_tot.adapter import (
    DatasetAdapter,
    Domain,
    MissingPolicy,
    Representation,
    Split,
)
from scientific_tot.data.hub import Dataset
from scientific_tot.records import Records


def _adapter(
    dataset: Dataset | None, local_dir: str | Path | None, revision: str | None
) -> DatasetAdapter:
    if dataset is not None and (local_dir is not None or revision is not None):
        raise ValueError("Pass dataset or local_dir/revision, not both")
    return DatasetAdapter(dataset or Dataset(local_dir=local_dir, revision=revision))


def load_corpus(
    *,
    domain: Domain,
    split: Split = "train+test",
    representation: Representation = "fulltext",
    missing: MissingPolicy = "error",
    include_figures: bool = True,
    include_metadata: bool = True,
    archives: bool = True,
    dataset: Dataset | None = None,
    local_dir: str | Path | None = None,
    revision: str | None = None,
) -> Records:
    """Load original-content IR documents; never silently omit incomplete papers."""
    return _adapter(dataset, local_dir, revision).load_corpus(
        domain=domain,
        split=split,
        representation=representation,
        missing=missing,
        include_figures=include_figures,
        include_metadata=include_metadata,
        archives=archives,
    )


def load_queries(
    *,
    domain: Domain,
    split: Split = "test",
    mode: str = "visual-and-text",
    dataset: Dataset | None = None,
    local_dir: str | Path | None = None,
    revision: str | None = None,
) -> Records:
    """Load standardized queries independently of document availability."""
    return _adapter(dataset, local_dir, revision).load_queries(
        domain=domain, split=split, mode=mode
    )


def load_qrels(
    *,
    corpus: Records,
    domain: Domain,
    split: Split = "test",
    mode: str = "visual-and-text",
    dataset: Dataset | None = None,
    local_dir: str | Path | None = None,
    revision: str | None = None,
) -> Records:
    """Load positive qrels, requiring all target IDs to exist in corpus.

    Pass the same dataset/revision used for corpus to avoid mixing publications.
    No corpus is downloaded implicitly by this function.
    """
    return _adapter(dataset, local_dir, revision).load_qrels(
        corpus=corpus,
        domain=domain,
        split=split,
        mode=mode,
    )


def load_metadata(
    *,
    domain: Domain,
    dataset: Dataset | None = None,
    local_dir: str | Path | None = None,
    revision: str | None = None,
) -> Records:
    """Load PDF-index metadata only; requires the optional parquet extra."""
    return _adapter(dataset, local_dir, revision).load_metadata(domain=domain)
