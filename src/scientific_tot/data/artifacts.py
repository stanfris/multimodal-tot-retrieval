"""Read restored files or TAR members without extracting or decoding images."""

from __future__ import annotations

import json
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scientific_tot.data.hub import Dataset, MissingComponentError
from scientific_tot.data.validation import DatasetFormatError, object_record, relative_path


@dataclass(frozen=True)
class Artifact:
    path: str
    file: str
    member: str | None = None
    offset: int = 0
    size: int = 0


class Artifacts:
    """A virtual preprocessed tree; a TAR shard is the smallest download unit."""

    def __init__(self, dataset: Dataset, domains: tuple[str, ...], *, archives: bool) -> None:
        self.dataset = dataset
        self.files: dict[str, Artifact] = {}
        self.images: dict[str, list[str]] = {}
        self.used: dict[str, Path] = {}
        self._archive_paths: dict[str, str] = {}
        for item in dataset.discover().files:
            name = item.path
            parts = name.split("/")
            if len(parts) < 3 or parts[0] != "preprocessed" or parts[1] not in domains:
                continue
            if name.endswith(".tar"):
                if not archives:
                    raise MissingComponentError(
                        f"{name}: archive reads disabled. "
                        "Use restored local files or archives=True."
                    )
                self._index_archive(name, parts[1])
            elif not name.endswith(".manifest.json"):
                self._add(Artifact(name, name))

    def _add(self, artifact: Artifact) -> None:
        relative_path(artifact.path)
        if artifact.path in self.files:
            raise DatasetFormatError(f"Duplicate extracted artifact: {artifact.path}")
        self.files[artifact.path] = artifact
        if "/images/" in artifact.path:
            directory = artifact.path.split("/images/", 1)[0]
            self.images.setdefault(directory, []).append(artifact.path)

    def _local(self, name: str) -> Path:
        if name not in self.used:
            self.used[name] = self.dataset.get_file(name)
        return self.used[name]

    def _index_archive(self, name: str, domain: str) -> None:
        local = self._local(name)
        try:
            with tarfile.open(local, "r:") as archive:
                seen = set()
                for member in archive:
                    relative = relative_path(
                        member.name.rstrip("/") if member.isdir() else member.name
                    )
                    if relative in seen or not (member.isfile() or member.isdir()):
                        raise DatasetFormatError(
                            f"{name}: unsupported/duplicate member {member.name}"
                        )
                    seen.add(relative)
                    if member.isfile():
                        self._add(
                            Artifact(
                                f"preprocessed/{domain}/{relative}",
                                name,
                                member.name,
                                member.offset_data,
                                member.size,
                            )
                        )
        except tarfile.TarError as exc:
            raise DatasetFormatError(f"{name}: invalid uncompressed TAR: {exc}") from exc

    def read_bytes(self, name: str) -> bytes:
        artifact = self.files.get(name)
        if artifact is None:
            raise MissingComponentError(f"Missing extracted artifact: {name}")
        local = self._local(artifact.file)
        if artifact.member is None:
            return local.read_bytes()
        with local.open("rb") as stream:
            stream.seek(artifact.offset)
            data = stream.read(artifact.size)
            if len(data) != artifact.size:
                raise DatasetFormatError(f"{name}: truncated TAR member")
            return data

    def read_text(self, name: str) -> str:
        try:
            return self.read_bytes(name).decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DatasetFormatError(f"{name}: invalid UTF-8") from exc

    def read_json(self, name: str) -> dict[str, Any]:
        try:
            return object_record(json.loads(self.read_text(name)), name)
        except json.JSONDecodeError as exc:
            raise DatasetFormatError(f"{name}: invalid JSON: {exc}") from exc

    def locator(self, name: str) -> dict[str, Any]:
        """Describe an artifact; never download a separate image just to locate it."""
        artifact = self.files.get(name)
        result: dict[str, Any] = {"path": name, "available": artifact is not None}
        if artifact is not None:
            result["archive"] = artifact.file if artifact.member else None
            result["member"] = artifact.member
            local = self.used.get(artifact.file)
            if local is None and self.dataset.local_dir is not None:
                local = self.dataset.local_dir / artifact.file
            if local is not None and artifact.member is not None:
                if artifact.file not in self._archive_paths:
                    self._archive_paths[artifact.file] = str(local.resolve())
                result["local_path"] = self._archive_paths[artifact.file]
            else:
                result["local_path"] = str(local.resolve()) if local is not None else None
        return result
