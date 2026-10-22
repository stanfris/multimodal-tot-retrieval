"""Offline staging checks using canonical adapter identities and streaming queries.

Only filenames, split lists and paper metadata are retained, never document text,
PDFs or decoded images. Encoding stages still verify content bytes and checksums.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scientific_tot.config_arguments import configured_adapter
from scientific_tot.data.validation import relative_path
from scientific_tot.stage_cli import plan


def validate(cfg) -> dict:
    if cfg.dataset.source != "local":
        raise ValueError(
            "Staging validation requires dataset.source=local (restored canonical paths)"
        )
    adapter = configured_adapter(cfg)
    dataset = adapter.dataset
    inventory = dataset.discover()
    sizes = {file.path: file.size for file in inventory.files}
    split = dataset.load_split()
    domain = cfg.dataset.domains[0]
    figures = cfg.representation.name == "figures" or cfg.model.name == "qwen3_vl_multimodal"
    # Pages is the adapter's identity-only view: no Markdown, PDF or image bytes.
    corpus = adapter.load_corpus(
        domain=domain,
        split=cfg.split.corpus,
        representation="pages",
        include_metadata=False,
        include_figures=figures,
        archives=False,
    )
    if not corpus:
        raise ValueError(f"No restored papers for {domain}")
    ids = {row["doc_id"] for row in corpus}
    sources = {row["source_pdf_relpath"] for row in corpus}
    selected_splits = (
        ("train", "test") if cfg.split.corpus in {"all", "train+test"} else (cfg.split.corpus,)
    )
    expected = {
        path for name in selected_splits for path in split[name] if path.startswith(f"{domain}/")
    }
    absent = expected - sources
    if absent:
        raise ValueError(
            f"{domain}: {len(absent)} split papers lack extractions; first={min(absent)}"
        )
    for row in corpus:
        metadata = row["provenance"]["paper_metadata"]
        directory = (
            metadata["path"].rsplit("/", 1)[0]
            if metadata
            else (f"preprocessed/{domain}/{row['doc_id']}")
        )
        required = []
        if cfg.representation.name in {"fulltext", "passages"}:
            markdown = row["metadata"]["paper"].get("markdown_relpath", "markdown.md")
            required.append(f"{directory}/{relative_path(markdown)}")
        if cfg.representation.name == "pages":
            required.append(f"pdf_datasets/{row['source_pdf_relpath']}")
        if figures:
            if (
                not row["figures"]
                and cfg.representation.name == "figures"
                and cfg.model.name.startswith("qwen3")
            ):
                raise ValueError(f"{row['doc_id']}: figure-only input requires supplied figures")
            for figure in row["figures"]:
                if not figure["image"]["available"]:
                    raise ValueError(f"{row['doc_id']}: missing image for {figure['figure_id']}")
                required.append(figure["image"]["path"])
        for path in required:
            if not sizes.get(path):
                raise FileNotFoundError(f"{row['doc_id']}: missing or empty {path}")
    splits = (
        ("train", "test") if cfg.split.queries in {"all", "train+test"} else (cfg.split.queries,)
    )
    collections = adapter._collections((domain,), splits, cfg.dataset.query_mode)
    count = 0
    for collection in collections:
        for query in dataset.iter_queries(collection):
            absent = set(query["relevant_ids"]) - ids
            if absent:
                raise ValueError(
                    f"{query['query_id']}: positives absent from corpus: {sorted(absent)}"
                )
            count += 1
    if not count:
        raise ValueError(f"No selected queries for {domain}")
    # Validate the actual packaged PDF schema without reading the whole Parquet table.
    if "metadata.parquet" in sizes:
        import pyarrow.parquet as pq

        table = pq.ParquetFile(Path(cfg.dataset.root) / "metadata.parquet")
        columns = {
            "document_id",
            "source",
            "original_relative_path",
            "shard",
            "member_path",
            "size_bytes",
            "sha256",
        }
        if not columns <= set(table.schema_arrow.names):
            raise ValueError(
                f"metadata.parquet missing columns: {columns - set(table.schema_arrow.names)}"
            )
        # Decode batches to expose corrupt row groups without retaining the full index.
        for batch in table.iter_batches(batch_size=4096, columns=sorted(columns)):
            batch.validate(full=True)
    return {
        "repo_id": dataset.repo_id,
        "revision_assertion": dataset.revision,
        "root": str(dataset.local_dir),
        "domain": domain,
        "papers": len(corpus),
        "queries": count,
        "official_split_papers": len(expected),
        "validation": "local identities, split coverage, queries and required file sizes",
    }


def main() -> None:
    cfg, _ = plan("prepare", [*sys.argv[1:], "dry_run=true"])
    print(json.dumps(validate(cfg), indent=2))


if __name__ == "__main__":
    main()
