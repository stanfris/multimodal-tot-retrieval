"""Validated record collections and machine-readable coverage reports."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, overload


@dataclass(frozen=True)
class Issue:
    code: str
    path: str
    detail: str
    doc_id: str | None = None


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    selected: int = 0
    emitted: int = 0
    skipped: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IncompleteCorpusError(ValueError):
    """Requested records cannot be emitted under the strict content policy."""

    def __init__(self, report: Report) -> None:
        self.report = report
        codes = sorted({issue.code for issue in report.issues})
        super().__init__(
            f"Corpus has {report.skipped} missing/incomplete records ({', '.join(codes)}). "
            "Inspect exception.report; use missing='skip' only to explicitly exclude them."
        )


class UnresolvedRelevanceError(ValueError):
    """Positive relevance IDs do not resolve to the selected corpus."""

    def __init__(self, report: Report) -> None:
        self.report = report
        super().__init__(
            f"{len(report.issues)} relevance judgments do not resolve to corpus records. "
            "Inspect exception.report and select/restore a complete corpus; qrels were not emitted."
        )


class Records(Sequence[dict[str, Any]]):
    """Eagerly validated records; iteration returns independent JSON-compatible dictionaries.

    Raw payloads are stored as JSON to prevent caller mutations from invalidating
    the collection used for relevance validation and reproducible exports.
    """

    def __init__(
        self, kind: str, rows: Sequence[dict[str, Any]], report: Report, provenance: dict[str, Any]
    ) -> None:
        self.kind = kind
        self._rows = tuple(json.dumps(row, ensure_ascii=False, allow_nan=False) for row in rows)
        self.report = report
        self.provenance = provenance

    def __len__(self) -> int:
        return len(self._rows)

    @overload
    def __getitem__(self, index: int) -> dict[str, Any]: ...

    @overload
    def __getitem__(self, index: slice) -> list[dict[str, Any]]: ...

    def __getitem__(self, index: int | slice) -> dict[str, Any] | list[dict[str, Any]]:
        if isinstance(index, slice):
            return [json.loads(row) for row in self._rows[index]]
        return json.loads(self._rows[index])

    def __iter__(self) -> Iterator[dict[str, Any]]:
        return (json.loads(row) for row in self._rows)

    def write_jsonl(self, path: str | Path) -> Path:
        """Atomically replace JSONL, then write a report/provenance sidecar.

        The two files are individually atomic, not a multi-file transaction.
        The manifest's output hash detects mismatched files after interruption.
        """
        target = Path(path)
        content = "".join(row + "\n" for row in self._rows).encode("utf-8")
        manifest = {
            "schema_version": 1,
            "kind": self.kind,
            "count": len(self),
            "sha256": hashlib.sha256(content).hexdigest(),
            "provenance": self.provenance,
            "report": self.report.to_dict(),
        }
        _atomic_write(target, content)
        _atomic_write(
            target.with_name(target.name + ".manifest.json"),
            (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode(),
        )
        return target


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
