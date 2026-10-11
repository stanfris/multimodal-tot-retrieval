# Dataset contract and inspection evidence

## Evidence boundary

Inspected on 2026-10-11:

- Live Hub API recursive inventory at commit
  `3b764a4c6855925ec802a2560712c4f5cc96f980`:
  [pinned file tree](https://huggingface.co/datasets/kasys/open-source-scientific-documents/tree/3b764a4c6855925ec802a2560712c4f5cc96f980).
  The machine-readable inventory is [hub-inventory.json](hub-inventory.json).
- All ten per-mode collection metadata JSON files and the ACL test collection
  root metadata. Counts below come from metadata, not a complete query audit.
- The complete ACL test `queries.jsonl` (6,939,521 bytes, 110 parsed rows).
  Other query files were inventoried but not downloaded.
- The exact linked generation repository, shallow-cloned at commit
  `2c208cf49d1d455be29bea12fa52e2cc8d53e1bb`, matching the available local checkout.
  [Pinned source](https://github.com/stanfris/multimodal-tip-of-tongue-retrieval-for-scientific-papers/tree/2c208cf49d1d455be29bea12fa52e2cc8d53e1bb).
- A local legacy `data/preprocessed/2023.acl-long.1/markdown.md` output confirmed
  extracted headings, body, captions, and relative `images/<hash>.jpg` links.
  This local sample is not evidence of current Hub publication.

The Hub inventory has 31 files: `.gitattributes` and 30 query collection files.
It has **no README/dataset card, PDF metadata, split index, preprocessed or clue
archives, or PDFs** at this commit. The web dataset viewer reports a schema cast
error; it is not a supported loading contract. The generation repository's
`src/dataset_packaging/dataset_card.md` is a template, not a published README.
No PDF, extraction archive, model, or complete dataset was downloaded.

## Published query layout

`query_collections/<DOMAIN>/query_generation_<train|test>/` contains root
`metadata.json`, plus `visual_and_text/metadata.json` and
`visual_and_text/queries.jsonl`. Domains are ACL, Biology, Engineering, Medicine,
and Physics. Only the `visual_and_text` directory is published at this revision.

| Domain | Train metadata count | Test metadata count |
| --- | ---: | ---: |
| ACL | 2199 | 110 |
| Biology | 2193 | 110 |
| Engineering | 2196 | 110 |
| Medicine | 2181 | 109 |
| Physics | 2186 | 109 |

Root metadata contains `collection_id`, `config`, `created_at_utc`, `modes`,
and `prompt_sha256`. Mode metadata contains `config`, `created_at_utc`,
`example_count`, `mode`, and `prompt_sha256`. Raw metadata is retained, including
configuration and historical absolute paths; those paths are provenance, not
portable locations to open on the current machine.

Verified query JSONL fields:

- `query_id`: string, e.g. `test_visual_and_text_q00000`.
- `query`: the generated natural-language text (also in visual modes).
- `relevant_ids`: list of extraction paper IDs, e.g. `ACL_2022.coling-1.30`.
- `negative_ids`: list; empty in the inspected first example.
- `metadata`: includes `paper_id`, `source_paper_dataset`, `mode`, `split`,
  `split_index`, `split_paper_index`, model/prompt provenance, component counts,
  budgets, `selected_components`, and the full generation `prompt`.

A component has `kind`, `record_id`, and `text`. Components and prompts are
LLM query-construction material, **not original paper content**. `metadata.mode`
uses hyphens (`visual-and-text`); the directory uses underscores. Source also
supports `visual-only`, but no such collection was present in the live listing.
Do not interpret either mode as an image binary query: the published `query`
field is text describing remembered content.

The source `_query_id` in `src/queries/query_generation.py` combines split,
normalized mode, and a five-digit paper index. It does not include the domain.
IDs must therefore be scoped by `(domain, collection, mode_directory)`; do not
concatenate collections and assume global uniqueness. This package retains the
original IDs and exposes a collection key rather than silently rewriting them.

Relevance is the `relevant_ids` list, not a TREC qrels file or graded judgments.
Source generation sets it to `[paper_id]`. The canonical adapter maps
these positives to grade 1 and records that convention in export provenance. Optional
`query_judgement` provenance in source is a query-quality assessment, not a
retrieval relevance grade. Unlisted papers must not silently become judged
negatives.

## Source-verified formats, not currently published

The following contracts are verified against the pinned generation implementation.
Availability, extraction completeness, and actual archive contents remain
unverified on the Hub.

### PDF index and source paths

`src/dataset_packaging/hf_dataset_packaging.py` defines `metadata.parquet` columns:
`document_id`, `source`, `original_filename`, `original_relative_path`, `shard`,
`member_path`, `size_bytes`, `sha256`, `license`, `title`, `year`, `doi`.
`original_relative_path` is relative to a source domain. A full source path is
`<source>/<original_relative_path>`. PDF TARs use
`data/<SOURCE>/shard-00000.tar`, with `<document_id>.pdf` and `.json` sidecars.
Exact duplicate bytes are tracked separately in `duplicates.parquet`; duplicate
contents do not authorize merging authoritative document IDs.

Packaged `document_id` is a different namespace from extraction `paper_id`:
`<normalized lowercase source>_<slugified natural ID>_<digest>`.
Natural ID precedence is anthology ID, arXiv ID, PMC ID, then filename stem.
The digest is the first 12 SHA-1 hex characters of `source + NUL + relative_path`.
Slugification is ASCII alphanumeric, lowercased, underscore-separated, truncated
to 80 characters. These details explain existing IDs; this package does not
regenerate them. Join via `paper.source_pdf_relpath` to the PDF index's exact
`source` and `original_relative_path`, retaining both authoritative IDs.

### Split index

`src/pdf_corpus/build_pdf_dataset_split.py` writes
`splits/pdf_dataset_split.json` with `train` and `test` lists of relative
`<DOMAIN>/<filename>.pdf` paths, plus `metadata`. Defaults are seed 42,
2200 training and 110 test PDFs per domain. These are **source defaults**, not
verified published membership or actual query counts. Query metadata's split
label cannot reconstruct the complete split. The shared retrieval corpus also
contains documents outside query-linked train/test subsets.

`src/common/validation.py` rejects unsafe paths, duplicates, and overlap.
Legacy extraction split indexes in `src/document_splits/` use other conventions;
the canonical adapter intentionally only accepts the PDF split contract above.

### Extraction identifiers, figures, and pages

`src/extraction/mineru_extraction.py::build_pdf_inputs` removes the `.pdf`
suffix from the relative source path, replaces characters other than Unicode
alphanumeric or `._-` with `_`, then strips leading/trailing `._` (fallback
`unknown`). When sanitized base IDs collide within the extraction input set,
**each colliding entry** gets `.` plus the first 12 SHA-256 hex characters of
its original relative PDF path. Never infer uniqueness from a basename alone.

Full `paper.json` includes `paper_id`, `source_pdf`, `source_pdf_relpath`,
`source_paper_dataset`, `split`, `markdown_relpath`, `markdown_sha256`,
`mineru_version`, backend/effort/parse method, `num_pages`, `figures`,
`structured_outputs`, `mineru_raw_relpath`, and `created_at_utc`.
`markdown.md` contains extracted original text and relative image references.
Successful full extraction uses `_SUCCESS`, `paper.json`, and `markdown.md`.

Figure records include authoritative `figure_id`, `paper_id`, source fields,
`type`, `sub_type`, `page_idx`, `bbox`, `caption`, `footnote`, `image_relpath`,
`image_path`, and dimensions. Source figure IDs have
`<paper_id>.fig.<block_index:04d>.<12-character SHA-256 suffix>` derived from
`paper_id|page_idx|block_index|img_path`. Preserve them; do not regenerate from
captions or filenames. `page_idx` is zero-based when available and may be null.
No independent canonical page ID is published. A future page adapter should
retain `(paper_id, page_idx)` rather than invent an upstream ID.

`src/preprocessing/stage_preprocessed.py` also supports lean staging with only
`markdown.md` and `images/`, deliberately omitting `paper.json` and success
markers. Source readers can derive fallback image IDs from image paths; these
are not evidence that an original authoritative figure ID survived. The Phase 2
adapter preserves existing lean directory paper IDs and resolves source paths
only when normal/collision candidates uniquely match actual split/index paths.
It never manufactures figure IDs: missing IDs stay null. The canonical adapter
checks a supplied markdown checksum; the low-level Phase 1 `load_paper` reader
does not. Neither reader claims multimodal completeness.

Additional archives preserve paths relative to `preprocessed/<DOMAIN>/` and
`clues/<DOMAIN>/`. Layout variants in source include domain/papers nesting;
`load_paper` takes an explicit directory instead of guessing. Clues live in
per-paper `base/textual_clues.jsonl` and `images/<figure_id>.jsonl` layouts.

## Retrieval inputs and incomplete data policy

Text retrieval needs original extracted markdown, authoritative paper identity,
query files, and the relevant split/index mappings. PDF metadata is needed when
linking extraction IDs to packaged IDs. Images and figure metadata are additional
inputs for figure retrieval; source PDFs and an explicit page mapping are needed
for page rendering. Clues and query-construction prompts are excluded from
standard corpus content. Neither PDFs nor images are required to inspect queries.

Malformed rows, ambiguous joins, duplicate query IDs within a collection,
missing requested files, empty markdown, and split overlap fail explicitly.
No missing document is silently dropped or replaced with clue text. The Phase 2
adapter reports coverage against selected PDF splits or the available PDF index,
with strict or explicit skip policies. Every emitted positive qrel must resolve
to the selected corpus. Lean identities, missing images, and metadata gaps are
reported explicitly. See [the API contract](api.md) for exact policy semantics.

## Restoration and test evidence

Inspected `src/dataset_packaging/download_hf_dataset.py` (component filters,
resume/copy behavior, archive path/link checks) and `restore_hf_dataset.py`
(PDF size/SHA-256 checks, original-path restoration, overwrite policy). These
are standalone tools in the generation repository; this package neither copies
nor imports their runtime. If archived artifacts become available, use those
tools separately and pass the restored root to `Dataset(local_dir=...)`.
The upstream downloader defaults to all components and has no revision argument;
do not use its default invocation for a lightweight or pinned experiment.
The Phase 2 adapter reads pinned, selected domain TARs directly without extracting
them. It indexes safe regular members and preserves archive/member locators,
so no new restoration implementation or runtime pipeline dependency is needed.

Inspected source tests for packaging, restoration/download, extraction collision
handling, preprocessing layouts, split validation, and query formats. The upstream
pipeline test suite was not run and is not a dependency. This project's fixtures
are synthetic representations of the verified query and source extraction/split
contracts, not copied corpus records. See [verification.md](verification.md) for
checks actually executed on this package.

## Phase 2 publication check

The Hub API was checked again during adapter implementation on 2026-10-11.
The head revision and query-only 31-file inventory remain unchanged. Archive,
PDF-index, and full split support therefore remain source-verified and
synthetically tested; they are not live-corpus verification claims.
