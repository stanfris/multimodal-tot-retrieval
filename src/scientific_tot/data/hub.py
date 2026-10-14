"""Selective Hub access and fully offline local reads."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.hf_api import RepoFile

from scientific_tot.data.schemas import FileInfo, Inventory, Paper, QueryCollection
from scientific_tot.data.validation import (
    DatasetFormatError,
    identifier,
    object_record,
    relative_path,
    validate_query,
    validate_split,
)

DEFAULT_REPO = "kasys/open-source-scientific-documents"


class MissingComponentError(FileNotFoundError):
    """Requested artifact is absent from the selected revision or local directory."""


class Dataset:
    """Use a Hub revision or an existing snapshot/restored data directory.

    Construction performs no I/O. A supplied revision must be a full commit SHA.
    Online discovery resolves main once, then pins every read to that commit.
    local_dir reads never access the network; inventory lists only local files.
    A revision with local_dir is a provenance assertion supplied by the caller,
    not an independent verification of local contents.
    """

    def __init__(
        self,
        *,
        repo_id: str = DEFAULT_REPO,
        revision: str | None = None,
        local_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if revision is not None and not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("revision must be an immutable full 40-character commit SHA")
        self.repo_id = repo_id
        self.revision = revision
        self.local_dir = Path(local_dir).expanduser() if local_dir is not None else None
        self.cache_dir = cache_dir
        self._inventory: Inventory | None = None
        self._paths: frozenset[str] = frozenset()

    def discover(self) -> Inventory:
        """List filenames and sizes only; never download artifact contents."""
        if self._inventory is not None:
            return self._inventory
        if self.local_dir is not None:
            if not self.local_dir.is_dir():
                raise MissingComponentError(
                    f"Local dataset directory does not exist: {self.local_dir}"
                )
            files = []
            # Prune hidden caches before traversal, rather than visiting every
            # cached PDF/image and then discarding it from the local inventory.
            for root, directories, filenames in os.walk(self.local_dir):
                directories[:] = [name for name in directories if not name.startswith(".")]
                for name in filenames:
                    path = Path(root) / name
                    if not name.startswith(".") and path.is_file():
                        files.append(
                            FileInfo(
                                path.relative_to(self.local_dir).as_posix(), path.stat().st_size
                            )
                        )
        else:
            api = HfApi()
            self.revision = self.revision or api.dataset_info(self.repo_id).sha
            files = [
                FileInfo(entry.path, entry.size)
                for entry in api.list_repo_tree(
                    self.repo_id, repo_type="dataset", revision=self.revision, recursive=True
                )
                if isinstance(entry, RepoFile)
            ]
        files.sort(key=lambda item: item.path)
        paths = self._paths = frozenset(item.path for item in files)
        collections = []
        for path in sorted(paths):
            parts = path.split("/")
            if len(parts) == 5 and parts[0] == "query_collections" and parts[-1] == "queries.jsonl":
                meta = path.removesuffix("queries.jsonl") + "metadata.json"
                collections.append(
                    QueryCollection(*parts[1:4], path, meta if meta in paths else None)
                )
        components = tuple(
            sorted(
                {
                    p.split("/")[0]
                    for p in paths
                    if p.split("/")[0]
                    in {
                        "query_collections",
                        "splits",
                        "preprocessed",
                        "clues",
                        "data",
                        "pdf_datasets",
                    }
                }
                | ({"metadata"} if "metadata.parquet" in paths else set())
            )
        )
        self._inventory = Inventory(tuple(files), tuple(collections), components, self.revision)
        return self._inventory

    def get_file(self, name: str) -> Path:
        """Select one exact artifact, using the Hub's revision cache incrementally.

        No other file is downloaded. Local mode returns the existing file only.
        """
        relative_path(name)
        self.discover()
        if name not in self._paths:
            raise MissingComponentError(
                f"Missing {name!r} in {self.local_dir or self.repo_id} "
                f"(revision {self.revision or 'local/unrecorded'}). "
                "Inspect discover(); select a revision or local directory "
                "containing this component."
            )
        if self.local_dir is not None:
            return self.local_dir / name
        return Path(
            hf_hub_download(
                self.repo_id,
                name,
                repo_type="dataset",
                revision=self.revision,
                cache_dir=self.cache_dir,
            )
        )

    def read_json(self, name: str, *, max_bytes: int = 2_000_000) -> dict[str, Any]:
        """Read explicitly selected JSON metadata (2 MB default size limit)."""
        relative_path(name)
        if not name.endswith(".json"):
            raise ValueError("read_json accepts only .json metadata files")
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        info = next((f for f in self.discover().files if f.path == name), None)
        if info is not None and (info.size is None or info.size > max_bytes):
            raise ValueError(f"{name}: metadata exceeds max_bytes or size is unknown")
        path = self.get_file(name)
        try:
            return object_record(json.loads(path.read_text(encoding="utf-8")), name)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise DatasetFormatError(f"{name}: invalid JSON: {exc}") from exc

    def load_split(self, name: str = "splits/pdf_dataset_split.json") -> dict[str, Any]:
        """Load the source-verified PDF split schema, preserving metadata."""
        return validate_split(self.read_json(name), name)

    def iter_queries(self, collection: QueryCollection) -> Iterator[dict[str, Any]]:
        """Download one selected JSONL if needed, then validate rows lazily.

        Duplicate query IDs are rejected within this file. Fully exhaust the
        iterator for complete validation. Records include untouched clue metadata.
        """
        if collection not in self.discover().collections:
            raise ValueError("Select a collection from this dataset's discover().collections")
        seen = set()
        with self.get_file(collection.queries_path).open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                source = f"{collection.queries_path}:{number}"
                try:
                    row = validate_query(json.loads(line), source)
                except json.JSONDecodeError as exc:
                    raise DatasetFormatError(f"{source}: invalid JSON: {exc}") from exc
                if row["query_id"] in seen:
                    raise DatasetFormatError(f"{source}: duplicate query_id {row['query_id']!r}")
                seen.add(row["query_id"])
                metadata = row["metadata"]
                if metadata["mode"].replace("-", "_") != collection.mode_directory:
                    raise DatasetFormatError(f"{source}: mode disagrees with collection directory")
                if metadata.get("source_paper_dataset") != collection.domain:
                    raise DatasetFormatError(
                        f"{source}: domain disagrees with collection directory"
                    )
                if collection.collection in {"query_generation_train", "query_generation_test"}:
                    expected_split = collection.collection.removeprefix("query_generation_")
                    if metadata.get("split") != expected_split:
                        raise DatasetFormatError(f"{source}: split disagrees with collection")
                yield row

    def load_pdf_metadata(self) -> Any:
        """Explicitly load metadata.parquet as an Arrow table, preserving every column."""
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "PDF metadata requires: uv sync --locked (or install the parquet package extra)"
            ) from exc
        return pq.read_table(self.get_file("metadata.parquet"))

    def load_paper(self, directory: str) -> Paper:
        """Read a full, already-restored local extraction; never use generated clues.

        Lean directories lacking paper.json need an authoritative ID mapping and
        are deliberately unsupported here. Missing markdown fails explicitly.
        """
        relative_path(directory)
        if self.local_dir is None or not directory.startswith("preprocessed/"):
            raise ValueError("load_paper requires local_dir and a preprocessed/ paper directory")
        metadata = self.read_json(f"{directory}/paper.json")
        paper_id = identifier(metadata.get("paper_id"), f"{directory}: paper_id")
        markdown_name = relative_path(metadata.get("markdown_relpath", "markdown.md"))
        markdown = self.get_file(f"{directory}/{markdown_name}").read_text(encoding="utf-8")
        if not markdown.strip():
            raise DatasetFormatError(f"{directory}: empty markdown; extraction is incomplete")
        return Paper(paper_id, markdown, metadata)
