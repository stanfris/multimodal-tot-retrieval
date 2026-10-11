"""Join authoritative IDs via source paths; never manufacture document IDs."""

from collections.abc import Iterable, Mapping
from typing import Any

from scientific_tot.data.validation import DatasetFormatError, identifier, relative_path


def document_id_for_paper(
    paper: Mapping[str, Any], metadata_rows: Iterable[Mapping[str, Any]]
) -> str:
    """Resolve an extracted paper through the PDF index's exact source/path pair.

    Lean artifacts without source_pdf_relpath cannot be joined by this function.
    Matching normalized paper names is deliberately not a fallback.
    """
    source_path = relative_path(paper.get("source_pdf_relpath"))
    source, separator, original = source_path.partition("/")
    if not separator:
        raise DatasetFormatError("source_pdf_relpath must include a domain")
    if paper.get("source_paper_dataset", source) != source:
        raise DatasetFormatError("source_paper_dataset disagrees with source_pdf_relpath")
    matches = [
        row
        for row in metadata_rows
        if row.get("source") == source and row.get("original_relative_path") == original
    ]
    if not matches:
        raise LookupError(f"No PDF metadata entry for {source_path}; obtain the matching index")
    if len(matches) != 1:
        raise DatasetFormatError(f"Ambiguous PDF metadata entries for {source_path}")
    return identifier(matches[0].get("document_id"), "metadata.document_id")
