"""Validate structural invariants without rewriting published records."""

from pathlib import PurePosixPath
from typing import Any


class DatasetFormatError(ValueError):
    """An artifact is malformed or internally inconsistent."""


def relative_path(value: str) -> str:
    """Require a portable, non-traversing artifact path."""
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or "\x00" in value
        or ":" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise DatasetFormatError(f"Invalid relative path: {value!r}")
    return value


def object_record(value: Any, source: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetFormatError(f"{source}: expected a JSON object")
    return value


def identifier(value: Any, source: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise DatasetFormatError(
            f"{source}: expected a nonempty identifier without edge whitespace"
        )
    return value


def validate_query(value: Any, source: str) -> dict[str, Any]:
    row = object_record(value, source)
    identifier(row.get("query_id"), f"{source}: query_id")
    if not isinstance(row.get("query"), str) or not row["query"].strip():
        raise DatasetFormatError(f"{source}: query must be nonempty text")
    for field in ("relevant_ids", "negative_ids"):
        ids = row.get(field)
        if not isinstance(ids, list) or (field == "relevant_ids" and not ids):
            raise DatasetFormatError(f"{source}: invalid {field}")
        for item in ids:
            identifier(item, f"{source}: {field}")
        if len(ids) != len(set(ids)):
            raise DatasetFormatError(f"{source}: duplicate {field}")
    if set(row["relevant_ids"]) & set(row["negative_ids"]):
        raise DatasetFormatError(f"{source}: relevant_ids and negative_ids overlap")
    metadata = object_record(row.get("metadata"), f"{source}: metadata")
    paper_id = identifier(metadata.get("paper_id"), f"{source}: metadata.paper_id")
    if paper_id not in row["relevant_ids"]:
        raise DatasetFormatError(f"{source}: metadata.paper_id absent from relevant_ids")
    if metadata.get("mode") not in {"visual-only", "visual-and-text"}:
        raise DatasetFormatError(f"{source}: unsupported query mode {metadata.get('mode')!r}")
    return row


def validate_split(value: Any, source: str) -> dict[str, Any]:
    row = object_record(value, source)
    for name in ("train", "test"):
        entries = row.get(name)
        if not isinstance(entries, list):
            raise DatasetFormatError(f"{source}: {name} must be a list of PDF paths")
        for entry in entries:
            path = PurePosixPath(relative_path(entry))
            if len(path.parts) < 2 or path.suffix.lower() != ".pdf":
                raise DatasetFormatError(f"{source}: invalid domain/PDF split entry {entry!r}")
        if len(entries) != len(set(entries)):
            raise DatasetFormatError(f"{source}: duplicate {name} paths")
    if set(row["train"]) & set(row["test"]):
        raise DatasetFormatError(f"{source}: train/test overlap")
    return row
