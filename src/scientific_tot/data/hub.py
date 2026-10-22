"""Selective Hub access and fully offline local reads."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
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

DEFAULT_REPO = "kasys/multimodal-tip-of-the-tongue-retrieval-for-scientific-documents"


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
        self._inventories: dict[tuple[str, bool], Inventory] = {}
        self._files: dict[str, FileInfo] = {}

    def _local_path(self, name: str) -> Path | None:
        assert self.local_dir is not None
        if not self.local_dir.is_dir():
            raise MissingComponentError(f"Local dataset directory does not exist: {self.local_dir}")
        parts = name.split("/") if name else []
        if any(part.startswith(".") for part in parts):
            return None
        path = self.local_dir
        for part in parts[:-1]:
            path = path / part
            if path.is_symlink():
                return None
        return self.local_dir / name

    def _pin_revision(self) -> None:
        if self.revision is None:
            self.revision = HfApi().dataset_info(self.repo_id).sha

    def discover(self, prefix: str = "", *, include_images: bool = True) -> Inventory:
        """List a subtree without downloading contents; the default lists all files.

        Text/page selections skip images for known extraction directories and
        retain at most one file per remaining images tree to detect image-only papers.
        """
        if prefix:
            relative_path(prefix)
        key = (prefix, include_images)
        if key in self._inventories:
            return self._inventories[key]
        if self.local_dir is not None:
            root = self._local_path(prefix)
            files = []
            pending = (
                [(root, prefix + "/" if prefix else "")]
                if root is not None and (not prefix or not root.is_symlink()) and root.is_dir()
                else []
            )
            images = set()
            while pending:
                directory, parent = pending.pop()
                image_root = parent.partition("/images/")[0] if "/images/" in parent else None
                if not include_images and image_root in images:
                    continue
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if entry.name.startswith("."):
                            continue
                        name = parent + entry.name
                        if entry.is_dir():
                            if not entry.is_symlink():
                                pending.append((entry.path, name + "/"))
                        elif entry.is_file():
                            files.append(FileInfo(name, entry.stat().st_size))
                            if not include_images and entry.name in {
                                "paper.json",
                                "markdown.md",
                                "_SUCCESS",
                            }:
                                images.add(parent.rstrip("/"))
                            if not include_images and image_root is not None:
                                images.add(image_root)
                                break
        else:
            self._pin_revision()
            try:
                files = [
                    FileInfo(entry.path, entry.size)
                    for entry in HfApi().list_repo_tree(
                        self.repo_id,
                        path_in_repo=prefix or None,
                        repo_type="dataset",
                        revision=self.revision,
                        recursive=True,
                    )
                    if isinstance(entry, RepoFile)
                    and (not prefix or entry.path.startswith(prefix + "/"))
                ]
            except EntryNotFoundError:
                if not prefix:
                    raise
                files = []
        files.sort(key=lambda item: item.path)
        paths = {item.path: item for item in files}
        self._files.update(paths)
        collections = []
        for path in paths:
            parts = path.split("/")
            if len(parts) == 5 and parts[0] == "query_collections" and parts[-1] == "queries.jsonl":
                meta = path.removesuffix("queries.jsonl") + "metadata.json"
                collections.append(
                    QueryCollection(*parts[1:4], path, meta if meta in paths else None)
                )
        components = tuple(
            sorted(
                {p.partition("/")[0] for p in paths}
                & {
                    "query_collections",
                    "splits",
                    "preprocessed",
                    "clues",
                    "data",
                    "pdf_datasets",
                }
                | ({"metadata"} if "metadata.parquet" in paths else set())
            )
        )
        inventory = Inventory(tuple(files), tuple(collections), components, self.revision)
        self._inventories[key] = inventory
        return inventory

    def file_info(self, name: str) -> FileInfo | None:
        """Inspect one exact file without discovering unrelated dataset contents."""
        relative_path(name)
        if name not in self._files:
            if self.local_dir is not None:
                path = self._local_path(name)
                if path is not None and path.is_file():
                    self._files[name] = FileInfo(name, path.stat().st_size)
            else:
                self._pin_revision()
                for entry in HfApi().get_paths_info(
                    self.repo_id, paths=[name], repo_type="dataset", revision=self.revision
                ):
                    if isinstance(entry, RepoFile):
                        self._files[entry.path] = FileInfo(entry.path, entry.size)
        return self._files.get(name)

    def get_file(self, name: str) -> Path:
        """Select one exact artifact, using the Hub's revision cache incrementally.

        No other file is downloaded. Local mode returns the existing file only.
        """
        if self.file_info(name) is None:
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
        info = self.file_info(name)
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
        prefix = collection.queries_path.rpartition("/")[0]
        if collection not in self.discover(prefix).collections:
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
