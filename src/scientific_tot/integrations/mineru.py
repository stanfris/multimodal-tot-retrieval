"""Deterministic page text alignment from the existing MinerU content-list artifacts."""

import hashlib
import json
from pathlib import PurePosixPath

from scientific_tot.data.validation import DatasetFormatError, relative_path

# Text-bearing fields in MinerU content-list v1/v2. Asset paths and geometry
# are deliberately excluded; strings are retained verbatim in source order.
TEXT_FIELDS = {
    "text",
    "content",
    "children",
    "paragraph_content",
    "title_content",
    "math_content",
    "list_items",
    "item_content",
    "code_content",
    "code_body",
    "table_body",
    "html",
    "image_caption",
    "image_footnote",
    "table_caption",
    "table_footnote",
    "chart_caption",
    "chart_footnote",
    "code_caption",
    "code_footnote",
    "page_header_content",
    "page_footer_content",
    "page_number_content",
    "page_footnote_content",
    "page_aside_text_content",
}


def _text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        # Inline spans may split a word at a formatting boundary; concatenate them
        # verbatim. Separate list items/captions retain explicit line boundaries.
        spans = all(
            isinstance(item, dict) and "type" in item and "content" in item for item in value
        )
        return ("" if spans else "\n").join(part for item in value if (part := _text(item)))
    if isinstance(value, dict):
        return "\n".join(
            part for key, item in value.items() if key in TEXT_FIELDS and (part := _text(item))
        )
    if value is None:
        return ""
    raise DatasetFormatError("Unsupported MinerU text field; expected string, list, or object")


def page_texts(store, paper) -> tuple[list[str], dict]:
    """Map zero-based page_idx (v1) or explicit outer pages (v2) to PDF pages.

    Missing pages in an otherwise aligned extraction have empty text. A missing
    content list or a block with no unambiguous page index fails explicitly.
    """
    metadata = paper["provenance"]["paper_metadata"]
    directory = (
        str(PurePosixPath(metadata["path"]).parent)
        if metadata is not None
        else f"preprocessed/{paper['domain']}/{paper['doc_id']}"
    )
    outputs = paper["metadata"]["paper"].get("structured_outputs", {})
    if not isinstance(outputs, dict):
        raise DatasetFormatError("paper.structured_outputs must be a mapping")
    candidates = [
        f"{directory}/{relative_path(path)}"
        for name, path in outputs.items()
        if "content_list" in name
    ]
    if not candidates:
        candidates = [
            name
            for name in store.files
            if name.startswith(directory + "/")
            and "content_list" in PurePosixPath(name).name
            and name.endswith(".json")
        ]
    if not candidates:
        raise ValueError(
            f"{paper['doc_id']}: page-level text alignment unavailable; restore MinerU "
            "content_list JSON (lean markdown/images staging omits page associations)"
        )
    # Prefer the simpler flat v1 schema when both exports describe the same paper.
    preferred = [name for name in candidates if "content_list_v2" not in name] or candidates
    if len(preferred) != 1:
        raise DatasetFormatError(f"Ambiguous MinerU content lists for {paper['doc_id']}")
    name = preferred[0]
    payload = store.read_bytes(name)
    data = json.loads(payload)
    if not isinstance(data, list):
        raise DatasetFormatError(f"{name}: MinerU content list must be an array")
    count = paper["page_count"]
    nested = bool(data) and isinstance(data[0], list)
    if nested and len(data) > count:
        raise DatasetFormatError(f"{name}: more extraction pages than PDF pages")
    pages = [[] for _ in range(count)]
    groups = enumerate(data) if nested else [(None, data)]
    for page_index, blocks in groups:
        if not isinstance(blocks, list):
            raise DatasetFormatError(f"{name}: mixed page/block content-list schema")
        for block in blocks:
            if not isinstance(block, dict):
                raise DatasetFormatError(f"{name}: MinerU block must be an object")
            index = block.get("page_idx", page_index)
            if type(index) is not int or not 0 <= index < count:
                raise DatasetFormatError(f"{name}: invalid/missing zero-based page_idx: {index}")
            if nested and index != page_index:
                raise DatasetFormatError(f"{name}: page_idx disagrees with containing page")
            text = _text(block)
            if (
                not text
                and block.get("content")
                and block.get("type")
                in {"paragraph", "title", "text", "list", "index", "equation_interline", "code"}
            ):
                raise DatasetFormatError(f"{name}: unsupported text-bearing block schema")
            if text:
                pages[index].append(text)
    result = ["\n\n".join(blocks) for blocks in pages]
    return result, {
        "path": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "pages_without_text": sum(not text.strip() for text in result),
    }
