"""Canonical corpus, query, and positive-relevance adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from scientific_tot.data.artifacts import Artifacts
from scientific_tot.data.hub import Dataset, MissingComponentError
from scientific_tot.data.identifiers import extraction_path_candidates
from scientific_tot.data.schemas import DOMAINS, QueryCollection
from scientific_tot.data.validation import DatasetFormatError, identifier, relative_path
from scientific_tot.records import (
    IncompleteCorpusError,
    Issue,
    Records,
    Report,
    UnresolvedRelevanceError,
)

Domain = str
Split = Literal["train", "test", "train+test", "all"]
Representation = Literal["fulltext", "title_abstract", "pages"]
MissingPolicy = Literal["error", "skip"]
MODES = {"visual-and-text": "visual_and_text", "visual-only": "visual_only"}


def _domains(domain: Domain) -> tuple[str, ...]:
    if not isinstance(domain, str) or domain not in DOMAINS:
        raise ValueError(f"domain must be one of {DOMAINS}")
    return (domain,)


def _splits(split: Split) -> tuple[str, ...]:
    if split not in {"train", "test", "train+test", "all"}:
        raise ValueError("split must be train, test, train+test, or all")
    return ("train", "test") if split in {"train+test", "all"} else (split,)


class DatasetAdapter:
    """Reuse one Dataset to pin corpus, query, and qrel reads to the same revision."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def _provenance(self, **selection: Any) -> dict[str, Any]:
        return {
            "repo_id": self.dataset.repo_id,
            "revision": self.dataset.discover().revision,
            "local_dir": str(self.dataset.local_dir.resolve()) if self.dataset.local_dir else None,
            "local_revision_verified": False if self.dataset.local_dir else None,
            "selection": selection,
        }

    def download_components(
        self,
        *,
        domain: Domain,
        components: Sequence[str],
        split: Split = "train+test",
        mode: str = "visual-and-text",
    ) -> tuple[Path, ...]:
        """Download only explicitly selected files, reusing the Hub cache on reruns.

        Allowed components: metadata, splits, queries, preprocessed, clues, pdfs.
        Archives select whole domain shards: split selection filters query files
        only. This method never restores files or runs generation utilities.
        """
        domains = _domains(domain)
        splits = _splits(split)
        allowed = {"metadata", "splits", "queries", "preprocessed", "clues", "pdfs"}
        if not components or set(components) - allowed:
            raise ValueError(f"Choose components from {sorted(allowed)}")
        if mode not in MODES:
            raise ValueError(f"Choose mode from {sorted(MODES)}")
        paths = {f.path for f in self.dataset.discover().files}
        selected: set[str] = set()
        for component in components:
            if component in {"metadata", "splits"}:
                name = (
                    "metadata.parquet"
                    if component == "metadata"
                    else "splits/pdf_dataset_split.json"
                )
                if name not in paths:
                    raise MissingComponentError(f"Missing requested component: {name}")
                selected.add(name)
            elif component == "queries":
                for collection in self._collections(domains, splits, mode):
                    selected.add(collection.queries_path)
                    if collection.metadata_path:
                        selected.add(collection.metadata_path)
                    root_meta = str(
                        PurePosixPath(collection.queries_path).parent.parent / "metadata.json"
                    )
                    if root_meta in paths:
                        selected.add(root_meta)
            else:
                prefix = "data" if component == "pdfs" else component
                for item in domains:
                    matches = {
                        p
                        for p in paths
                        if p.startswith(f"{prefix}/{item}/") and not p.endswith(".manifest.json")
                    }
                    if not matches:
                        raise MissingComponentError(f"Missing requested component: {prefix}/{item}")
                    selected.update(matches)
                if component == "pdfs":
                    if "metadata.parquet" not in paths:
                        raise MissingComponentError("PDF restoration requires metadata.parquet")
                    selected.add("metadata.parquet")
        return tuple(self.dataset.get_file(name) for name in sorted(selected))

    def resolve_pdf(self, row: dict) -> Path:
        """Resolve an original restored PDF using the adapter's canonical source join.

        Packaged TAR PDFs must first be restored by the upstream restoration tool.
        """
        source = row["source_pdf_relpath"]
        if source is None:
            raise MissingComponentError(f"No original PDF source for {row['doc_id']}")
        name = f"pdf_datasets/{relative_path(source)}"
        path = self.dataset.get_file(name)
        if self.dataset.local_dir is not None and not path.resolve().is_relative_to(
            self.dataset.local_dir.resolve()
        ):
            raise DatasetFormatError(f"PDF escapes dataset root: {name}")
        return path

    def load_metadata(self, *, domain: Domain) -> Records:
        """Read domain-filtered PDF metadata without fetching any PDF or extraction archive.

        document_id here is the packaged PDF identifier, not an IR corpus doc_id.
        No abstract or extraction identity is inferred from the PDF index.
        """
        domains = _domains(domain)
        rows = [
            r for r in self.dataset.load_pdf_metadata().to_pylist() if r.get("source") in domains
        ]
        _pdf_index(rows)
        absent = set(domains) - {row["source"] for row in rows}
        if absent:
            raise MissingComponentError(f"No PDF metadata for domains: {sorted(absent)}")
        return Records(
            "metadata",
            rows,
            Report(selected=len(rows), emitted=len(rows)),
            self._provenance(domain=domains, files=["metadata.parquet"]),
        )

    def _collections(
        self, domains: tuple[str, ...], splits: tuple[str, ...], mode: str
    ) -> list[QueryCollection]:
        if mode not in MODES:
            raise ValueError(f"Choose the published mode spelling from {sorted(MODES)}")
        found = []
        available = self.dataset.discover().collections
        for domain in domains:
            for split in splits:
                matches = [
                    c
                    for c in available
                    if c.domain == domain
                    and c.collection == f"query_generation_{split}"
                    and c.mode_directory == MODES[mode]
                ]
                if not matches:
                    raise MissingComponentError(
                        f"Missing query collection: {domain}/{split}/{mode}"
                    )
                found.extend(matches)
        return found

    def load_queries(
        self, *, domain: Domain, split: Split = "test", mode: str = "visual-and-text"
    ) -> Records:
        domains, splits = _domains(domain), _splits(split)
        rows = []
        files = []
        collection_metadata = {}
        for collection in self._collections(domains, splits, mode):
            files.append(collection.queries_path)
            for name in (
                collection.metadata_path,
                str(PurePosixPath(collection.queries_path).parent.parent / "metadata.json"),
            ):
                if name and name in {f.path for f in self.dataset.discover().files}:
                    collection_metadata[name] = self.dataset.read_json(name)
                    files.append(name)
            for raw in self.dataset.iter_queries(collection):
                # Explicit compound key for frameworks requiring globally unique IDs.
                # The canonical query_id remains unchanged in its own field.
                key = json.dumps([*collection.key, raw["query_id"]], separators=(",", ":"))
                rows.append(
                    {
                        "query_id": raw["query_id"],
                        "query_key": key,
                        "text": raw["query"],
                        "domain": collection.domain,
                        "split": raw["metadata"]["split"],
                        "mode": raw["metadata"]["mode"],
                        "collection": collection.collection,
                        "relevant_ids": raw["relevant_ids"],
                        "negative_ids": raw["negative_ids"],
                        "metadata": raw["metadata"],
                        "extra_fields": {
                            k: v
                            for k, v in raw.items()
                            if k
                            not in {"query_id", "query", "relevant_ids", "negative_ids", "metadata"}
                        },
                    }
                )
        return Records(
            "queries",
            rows,
            Report(selected=len(rows), emitted=len(rows)),
            self._provenance(
                domain=domains,
                split=split,
                mode=mode,
                files=files,
                collection_metadata=collection_metadata,
            ),
        )

    def load_qrels(
        self,
        *,
        corpus: Records,
        domain: Domain,
        split: Split = "test",
        mode: str = "visual-and-text",
    ) -> Records:
        """Emit binary positives only after every relevant ID resolves to corpus."""
        queries = self.load_queries(domain=domain, split=split, mode=mode)
        return validate_qrels(queries, corpus)

    def load_corpus(
        self,
        *,
        domain: Domain,
        split: Split = "train+test",
        representation: Representation = "fulltext",
        missing: MissingPolicy = "error",
        include_figures: bool = True,
        include_metadata: bool = True,
        archives: bool = True,
    ) -> Records:
        """Build a corpus from full or lean original extractions.

        split='all' selects the available corpus, including distractors; train+test
        selects only the union of the two authoritative PDF split lists.
        missing='skip' returns an explicit coverage report, never placeholder text.
        """
        domains, splits = _domains(domain), _splits(split)
        if representation not in {"fulltext", "title_abstract", "pages"}:
            raise ValueError("representation must be fulltext, title_abstract, or pages")
        if missing not in {"error", "skip"}:
            raise ValueError("missing must be error or skip")
        inventory = self.dataset.discover()
        paths = {f.path for f in inventory.files}
        # Fail before any large transfer when mandatory selection information is absent.
        expected: set[str] | None = None
        split_map: dict[str, str] = {}
        files: list[str] = []
        if split != "all":
            index = self.dataset.load_split()
            files.append("splits/pdf_dataset_split.json")
            split_map = {p: name for name in ("train", "test") for p in index[name]}
            expected = {p for name in splits for p in index[name] if p.split("/")[0] in domains}
            if not expected:
                raise MissingComponentError(f"No PDF split entries for {domains}/{split}")
        pdf_rows = []
        if include_metadata and "metadata.parquet" in paths:
            pdf_rows = self.dataset.load_pdf_metadata().to_pylist()
            files.append("metadata.parquet")
        pdf_by_path = _pdf_index(pdf_rows)
        coverage_targets = (
            expected
            if expected is not None
            else {p for p in pdf_by_path if p.split("/")[0] in domains}
        )
        candidates: dict[str, set[str]] = {}
        for path in set(split_map) | set(pdf_by_path):
            for candidate in extraction_path_candidates(path):
                candidates.setdefault(candidate, set()).add(path)
        store = Artifacts(self.dataset, domains, archives=archives)
        if not store.files:
            raise MissingComponentError(
                f"No preprocessed artifacts for {domains} at {inventory.revision or 'local root'}. "
                "Supply restored data or a published revision containing original extractions."
            )
        # Include image-only directories in coverage reports, without opening image data.
        directories = {
            str(PurePosixPath(p).parent)
            for p in store.files
            if p.endswith(("/paper.json", "/markdown.md", "/_SUCCESS"))
        }
        directories.update(p.split("/images/", 1)[0] for p in store.files if "/images/" in p)
        rows = []
        report = Report()
        seen_ids: set[str] = set()
        seen_sources: set[str] = set()
        for directory in sorted(directories):
            meta_path = f"{directory}/paper.json"
            full = meta_path in store.files
            paper = store.read_json(meta_path) if full else {}
            paper_id = identifier(
                paper.get("paper_id") if full else PurePosixPath(directory).name,
                f"{directory}: paper_id",
            )
            source_path = paper.get("source_pdf_relpath")
            if source_path is not None:
                relative_path(source_path)
                if (
                    len(source_path.split("/")) < 2
                    or PurePosixPath(source_path).suffix.lower() != ".pdf"
                ):
                    raise DatasetFormatError(f"{directory}: invalid source PDF path")
            else:
                matches = candidates.get(paper_id, set())
                if len(matches) > 1:
                    raise DatasetFormatError(
                        f"{directory}: ambiguous source mapping for {paper_id}"
                    )
                source_path = next(iter(matches), None)
            domain_name = directory.split("/")[1]
            if paper.get("source_paper_dataset", domain_name) != domain_name:
                raise DatasetFormatError(f"{directory}: source domain disagrees with artifact path")
            if source_path is not None and source_path.split("/")[0] != domain_name:
                raise DatasetFormatError(f"{directory}: source_pdf_relpath disagrees with domain")
            if expected is not None and source_path is None:
                report.issues.append(
                    Issue(
                        "unresolved_source_path",
                        directory,
                        "Cannot establish split membership",
                        paper_id,
                    )
                )
                report.selected += 1
                report.skipped += 1
                continue
            if expected is not None and source_path not in expected:
                continue
            report.selected += 1
            if paper_id in seen_ids or (source_path is not None and source_path in seen_sources):
                raise DatasetFormatError(
                    f"{directory}: duplicate paper ID or source path: {paper_id}"
                )
            seen_ids.add(paper_id)
            if source_path:
                seen_sources.add(source_path)
            if source_path in split_map and paper.get("split") not in {
                None,
                split_map[source_path],
            }:
                raise DatasetFormatError(f"{directory}: paper split disagrees with split index")
            pdf = pdf_by_path.get(source_path, {})
            # The verified PDF index has titles but no abstract column; retain any
            # explicitly supplied original metadata without inferring absent values.
            title = _text_field(pdf, paper, "title")
            abstract = _text_field(pdf, paper, "abstract")
            markdown = f"{directory}/{relative_path(paper.get('markdown_relpath', 'markdown.md'))}"
            if representation == "fulltext":
                if markdown not in store.files:
                    _skip(report, "missing_fulltext", markdown, paper_id)
                    continue
                text = store.read_text(markdown)
                if not text.strip():
                    _skip(report, "empty_fulltext", markdown, paper_id)
                    continue
                checksum = paper.get("markdown_sha256")
                if checksum is not None and hashlib.sha256(text.encode()).hexdigest() != checksum:
                    _skip(report, "markdown_checksum_mismatch", markdown, paper_id)
                    continue
            elif representation == "pages":
                text = ""
            else:
                text = "\n\n".join(value for value in (title, abstract) if value)
                if not text:
                    _skip(report, "missing_title_abstract", directory, paper_id)
                    continue
                if title is None or abstract is None:
                    report.issues.append(
                        Issue(
                            "partial_title_abstract",
                            directory,
                            "Only available original fields were used",
                            paper_id,
                        )
                    )
            figures = _figures(store, directory, paper, paper_id, report) if include_figures else []
            if include_metadata and not pdf:
                report.issues.append(
                    Issue(
                        "missing_pdf_metadata",
                        directory,
                        "Bibliographic/license enrichment unavailable",
                        paper_id,
                    )
                )
            rows.append(
                {
                    "doc_id": paper_id,
                    "pdf_document_id": pdf.get("document_id"),
                    "domain": domain_name,
                    "split": split_map.get(source_path, paper.get("split")),
                    "source_pdf_relpath": source_path,
                    "representation": representation,
                    "title": title,
                    "abstract": abstract,
                    "license": pdf.get("license") or paper.get("license"),
                    "doi": pdf.get("doi") or paper.get("doi"),
                    "year": pdf.get("year") if pdf.get("year") is not None else paper.get("year"),
                    "text": text,
                    "figures": figures,
                    "metadata": {"paper": paper, "pdf": pdf},
                    "provenance": {
                        "paper_metadata": store.locator(meta_path) if full else None,
                        "content": store.locator(markdown)
                        if representation == "fulltext"
                        else {"source_pdf_relpath": source_path}
                        if representation == "pages"
                        else {
                            "fields": [
                                k
                                for k, v in (("title", title), ("abstract", abstract))
                                if v is not None
                            ]
                        },
                        "identity_source": "paper.json" if full else "extraction_directory",
                    },
                }
            )
        for path in sorted(coverage_targets - seen_sources):
            _skip(report, "missing_extraction", path, None)
            report.selected += 1
        report.emitted = len(rows)
        if split == "all":
            available_domains = {p.split("/")[1] for p in store.files}
            for domain_name in sorted(set(domains) - available_domains):
                report.selected += 1
                _skip(report, "missing_domain", f"preprocessed/{domain_name}", None)
        if report.skipped and missing == "error":
            raise IncompleteCorpusError(report)
        return Records(
            "corpus",
            rows,
            report,
            self._provenance(
                domain=domains,
                split=split,
                representation=representation,
                missing=missing,
                include_figures=include_figures,
                include_metadata=include_metadata,
                files=sorted(set(files) | set(store.used)),
            ),
        )


def _skip(report: Report, code: str, path: str, doc_id: str | None) -> None:
    report.skipped += 1
    report.issues.append(
        Issue(code, path, "Record excluded; no replacement content generated", doc_id)
    )


def _text_field(pdf: dict[str, Any], paper: dict[str, Any], key: str) -> str | None:
    for record in (pdf, paper):
        value = record.get(key)
        if value is not None:
            if not isinstance(value, str):
                raise DatasetFormatError(f"{key}: expected original text or null")
            if value.strip():
                return value
    return None


def _pdf_index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    ids = set()
    for row in rows:
        source = relative_path(row.get("source"))
        if "/" in source:
            raise DatasetFormatError("PDF metadata source must be a domain name")
        path = source + "/" + relative_path(row.get("original_relative_path"))
        doc_id = identifier(row.get("document_id"), "metadata.document_id")
        if path in result or doc_id in ids:
            raise DatasetFormatError(f"Duplicate PDF metadata ID or source path: {path}")
        ids.add(doc_id)
        result[path] = row
    return result


def _figures(
    store: Artifacts, directory: str, paper: dict[str, Any], paper_id: str, report: Report
) -> list[dict[str, Any]]:
    original = paper.get("figures", [])
    if not isinstance(original, list):
        raise DatasetFormatError(f"{directory}: figures must be a list")
    result = []
    seen = set()
    for figure in original:
        if not isinstance(figure, dict):
            raise DatasetFormatError(f"{directory}: malformed figure")
        fid = identifier(figure.get("figure_id"), f"{directory}: figure_id")
        if fid in seen or figure.get("paper_id", paper_id) != paper_id:
            raise DatasetFormatError(f"{directory}: duplicate figure ID or wrong paper ID")
        seen.add(fid)
        page = figure.get("page_idx")
        if page is not None and (type(page) is not int or page < 0):
            raise DatasetFormatError(f"{directory}: page_idx must be a nonnegative integer or null")
        rel = figure.get("image_relpath")
        # Historical absolute image paths are provenance, never opened implicitly.
        if isinstance(rel, str) and not PurePosixPath(rel).is_absolute():
            locator = store.locator(f"{directory}/{relative_path(rel)}")
        else:
            locator = {"path": None, "available": False}
        if not locator["available"]:
            report.issues.append(
                Issue(
                    "missing_figure_image",
                    directory,
                    f"Image unavailable for figure {fid}",
                    paper_id,
                )
            )
        result.append(
            {
                "figure_id": fid,
                "page_idx": figure.get("page_idx"),
                "image": locator,
                "metadata": figure,
            }
        )
    if not original:
        images = sorted(p for p in store.files if p.startswith(f"{directory}/images/"))
        if images:
            report.issues.append(
                Issue(
                    "missing_figure_metadata",
                    directory,
                    "Image paths retained without fabricated figure IDs",
                    paper_id,
                )
            )
        result.extend(
            {"figure_id": None, "page_idx": None, "image": store.locator(p), "metadata": {}}
            for p in images
        )
    return result


def validate_qrels(queries: Records, corpus: Records) -> Records:
    """Validate positive judgments against an already loaded corpus independently."""
    if queries.kind != "queries" or corpus.kind != "corpus":
        raise TypeError("validate_qrels requires query and corpus Records")
    for key in ("repo_id", "revision"):
        left, right = queries.provenance.get(key), corpus.provenance.get(key)
        if left is not None and right is not None and left != right:
            raise ValueError(f"Corpus/query {key} differs; use a shared Dataset/revision")
    ids = {row["doc_id"]: row["domain"] for row in corpus}
    rows = []
    report = Report()
    for query in queries:
        for doc_id in query["relevant_ids"]:
            report.selected += 1
            if ids.get(doc_id) != query["domain"]:
                report.issues.append(
                    Issue(
                        "unresolved_relevant_id",
                        query["query_key"],
                        "Relevant paper absent from selected corpus/domain",
                        doc_id,
                    )
                )
            else:
                rows.append(
                    {
                        "query_id": query["query_id"],
                        "query_key": query["query_key"],
                        "doc_id": doc_id,
                        "relevance": 1,
                        "domain": query["domain"],
                        "split": query["split"],
                        "mode": query["mode"],
                    }
                )
    if report.issues:
        raise UnresolvedRelevanceError(report)
    report.emitted = len(rows)
    return Records(
        "qrels",
        rows,
        report,
        {
            **queries.provenance,
            "corpus": corpus.provenance,
            "corpus_doc_ids_sha256": hashlib.sha256(json.dumps(sorted(ids)).encode()).hexdigest(),
            "relevance_policy": "relevant_ids mapped to 1; no inferred negatives",
        },
    )
