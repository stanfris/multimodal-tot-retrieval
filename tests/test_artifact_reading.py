"""Scoped TAR readers retain validation, bounded handles, and independent cursors."""

import io
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scientific_tot.data.artifacts import Artifacts
from scientific_tot.data.hub import Dataset, MissingComponentError
from scientific_tot.data.validation import DatasetFormatError


def archive(root, name, members, *, format=tarfile.PAX_FORMAT):
    path = root / f"preprocessed/ACL/{name}.tar"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w", format=format) as output:
        for name, content in members.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            output.addfile(member, io.BytesIO(content))
    return path


@pytest.mark.parametrize("format", [tarfile.PAX_FORMAT, tarfile.GNU_FORMAT])
def test_header_scan_releases_tarinfo_and_preserves_extended_names(tmp_path, monkeypatch, format):
    members = {f"paper/images/{'long-name-' * 20}{n}.png": str(n).encode() for n in range(80)}
    archive(tmp_path, "shard", members, format=format)
    next_member = tarfile.TarFile.next
    retained = []

    def counted_next(self):
        member = next_member(self)
        retained.append(len(self.members))
        return member

    monkeypatch.setattr(tarfile.TarFile, "next", counted_next)
    store = Artifacts(Dataset(local_dir=tmp_path), ("ACL",), archives=True)
    assert max(retained) <= 1
    with store.reader() as read:
        for name, content in members.items():
            assert read(f"preprocessed/ACL/{name}") == content


def test_scoped_reader_reuses_and_closes_archive_on_failure(tmp_path, monkeypatch):
    path = archive(tmp_path, "shard", {"paper/images/a.png": b"a", "paper/images/b.png": b"b"})
    store = Artifacts(Dataset(local_dir=tmp_path), ("ACL",), archives=True)
    opened = []
    original = Path.open

    def tracked(self, *args, **kwargs):
        stream = original(self, *args, **kwargs)
        if self == path:
            opened.append(stream)
        return stream

    monkeypatch.setattr(Path, "open", tracked)
    with pytest.raises(MissingComponentError):
        with store.reader() as read:
            for _ in range(20):
                assert read("preprocessed/ACL/paper/images/a.png") == b"a"
                assert read("preprocessed/ACL/paper/images/b.png") == b"b"
            assert len(opened) == 1
            read("missing")
    assert opened[0].closed
    # A new scope observes replacement content, even with an unchanged member size.
    archive(tmp_path, "shard", {"paper/images/a.png": b"z", "paper/images/b.png": b"b"})
    assert store.read_bytes("preprocessed/ACL/paper/images/a.png") == b"z"
    assert all(stream.closed for stream in opened)


def test_reader_closes_previous_shard_and_recovers_after_open_failure(tmp_path, monkeypatch):
    first = archive(tmp_path, "a", {"paper/a": b"a"})
    second = archive(tmp_path, "b", {"paper/b": b"b"})
    store = Artifacts(Dataset(local_dir=tmp_path), ("ACL",), archives=True)
    original = Path.open
    opened = []

    def tracked(self, *args, **kwargs):
        if self in {first, second}:
            assert all(stream.closed for stream in opened)
            stream = original(self, *args, **kwargs)
            opened.append(stream)
            return stream
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", tracked)
    with store.reader() as read:
        assert read("preprocessed/ACL/paper/a") == b"a"
        second.unlink()
        with pytest.raises(FileNotFoundError):
            read("preprocessed/ACL/paper/b")
        assert read("preprocessed/ACL/paper/a") == b"a"
    assert all(stream.closed for stream in opened)


def test_readers_have_independent_thread_cursors_and_detect_truncation(tmp_path):
    path = archive(tmp_path, "shard", {"paper/a": b"a" * 1000, "paper/b": b"b" * 1000})
    store = Artifacts(Dataset(local_dir=tmp_path), ("ACL",), archives=True)

    def read_worker(number):
        with store.reader() as read:
            for index in range(100):
                name = "a" if (number + index) % 2 else "b"
                assert read(f"preprocessed/ACL/paper/{name}") == name.encode() * 1000

    with ThreadPoolExecutor(max_workers=4) as workers:
        list(workers.map(read_worker, range(4)))
    member = store.files["preprocessed/ACL/paper/b"]
    with path.open("r+b") as stream:
        stream.truncate(member.offset + member.size - 1)
    with pytest.raises(DatasetFormatError, match="truncated TAR member"):
        with store.reader() as read:
            read(member.path)


def test_fresh_inventory_rejects_changed_headers_and_duplicate_members(tmp_path):
    path = archive(tmp_path, "shard", {"paper/markdown.md": b"original"})
    store = Artifacts(Dataset(local_dir=tmp_path), ("ACL",), archives=True)
    assert store.read_text("preprocessed/ACL/paper/markdown.md") == "original"
    with path.open("r+b") as stream:
        stream.write(b"X")  # Change a header byte without updating its TAR checksum.
    with pytest.raises(DatasetFormatError, match="invalid uncompressed TAR"):
        Artifacts(Dataset(local_dir=tmp_path), ("ACL",), archives=True)
    # Releasing TarInfo objects must not release duplicate-member validation.
    archive(tmp_path, "shard", {"paper/markdown.md": b"original"})
    with tarfile.open(path, "a") as output:
        output.addfile(tarfile.TarInfo("paper/markdown.md"), io.BytesIO(b""))
    with pytest.raises(DatasetFormatError, match="unsupported/duplicate member"):
        Artifacts(Dataset(local_dir=tmp_path), ("ACL",), archives=True)
