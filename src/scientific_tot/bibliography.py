"""Explicit projection of literal bibliography sections from original extracted text."""

from __future__ import annotations

import re
from typing import Any

from scientific_tot.records import IncompleteCorpusError, Issue, Records, Report

HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)


def title_abstract_from_markdown(corpus: Records, *, missing: str = "error") -> Records:
    """Select an existing H1 title and a delimited Abstract section, never summarize.

    This is an opt-in text projection, not regenerated bibliographic metadata.
    Recognized heading spelling is exactly Abstract (case-insensitive). Source
    character spans are recorded. Existing original title/abstract fields win.
    Unrecognized or undelimited sections remain null; no heuristic prose guessing.
    """
    if corpus.kind != "corpus" or missing not in {"error", "skip"}:
        raise ValueError("Requires corpus Records and missing='error' or 'skip'")
    rows = []
    report = Report(selected=len(corpus))
    for row in corpus:
        if row["representation"] != "fulltext":
            raise ValueError("The input must contain original fulltext")
        text = row["text"]
        headings = list(HEADING.finditer(text))
        title, abstract = row["title"], row["abstract"]
        spans: dict[str, Any] = {}
        if title is None and headings:
            first = headings[0]
            if first.group(1) == "#" and not text[: first.start()].strip():
                title = first.group(2)
                spans["title"] = [first.start(2), first.end(2)]
        if abstract is None:
            matches = [i for i, h in enumerate(headings) if h.group(2).casefold() == "abstract"]
            if len(matches) == 1:
                i = matches[0]
                if i + 1 < len(headings):
                    start, end = headings[i].end(), headings[i + 1].start()
                    if text[start:end].strip():
                        abstract = text[start:end].strip()
                        spans["abstract"] = [start, end]
        projected = "\n\n".join(value for value in (title, abstract) if value)
        if not projected:
            report.skipped += 1
            report.issues.append(
                Issue(
                    "missing_title_abstract",
                    row["doc_id"],
                    "No original fields or recognized Markdown sections",
                    row["doc_id"],
                )
            )
            continue
        if title is None or abstract is None:
            report.issues.append(
                Issue(
                    "partial_title_abstract",
                    row["doc_id"],
                    "Only available original fields/sections selected",
                    row["doc_id"],
                )
            )
        row.update(title=title, abstract=abstract, text=projected, representation="title_abstract")
        row["provenance"] = {
            **row["provenance"],
            "bibliography_projection": {
                "method": "literal_markdown_sections",
                "source": row["provenance"]["content"],
                "character_spans": spans,
            },
        }
        rows.append(row)
    report.emitted = len(rows)
    if report.skipped and missing == "error":
        raise IncompleteCorpusError(report)
    return Records(
        "corpus",
        rows,
        report,
        {
            **corpus.provenance,
            "representation": "title_abstract",
            "bibliography_projection": "literal_markdown_sections",
            "parent_report": corpus.report.to_dict(),
        },
    )
