"""Small access-layer records, retaining original JSON separately."""

from dataclasses import dataclass
from typing import Any

DOMAINS = ("ACL", "Biology", "Engineering", "Medicine", "Physics")


@dataclass(frozen=True)
class FileInfo:
    path: str
    size: int | None


@dataclass(frozen=True)
class QueryCollection:
    domain: str
    collection: str
    mode_directory: str
    queries_path: str
    metadata_path: str | None

    @property
    def key(self) -> tuple[str, str, str]:
        """Scope within which upstream query IDs are unique."""
        return self.domain, self.collection, self.mode_directory


@dataclass(frozen=True)
class Inventory:
    files: tuple[FileInfo, ...]
    collections: tuple[QueryCollection, ...]
    components: tuple[str, ...]
    revision: str | None


@dataclass(frozen=True)
class Paper:
    paper_id: str
    markdown: str
    metadata: dict[str, Any]
