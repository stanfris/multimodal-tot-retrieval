# Architecture

The initial release is an access layer. `Dataset` orchestrates Hub discovery and
explicit file reads. `schemas.py` defines small inventory records and a local
paper result. `validation.py` enforces structural invariants without deleting
unknown fields. `identifiers.py` performs an exact source-path join between
existing identifiers. The generation pipeline is an inspection reference only.

## Dependencies and boundaries

The core uses `huggingface_hub` for all network file access. JSON, paths, and
iteration use the standard library. Arrow is an optional `parquet` extra backed
by an implemented reader; `datasets` is deferred until tabular conversion needs
it. Development tools are a uv dependency group. There are no retrieval extras,
empty export modules, configuration frameworks, or database dependencies.

No ranking, embeddings, evaluation metrics, PDF extraction, query generation,
or archive extraction runs in this repository. Restoration remains an explicit
external preparation step using the upstream utilities. This avoids duplicating
checksum/security code or bringing its pandas/generation dependency tree into
the package. An extraction-only reusable upstream API is a future integration
requirement, not a current hidden dependency.

## Reproducibility and selection

Discovery downloads no artifact content and records all listed filenames and
sizes. Default remote use resolves main to a commit once; all later calls use
that SHA. Supplied revisions must already be immutable SHAs. Local mode never
uses the network or falls back to remote content, and it reports local coverage
only. Passing a revision with local data is a caller assertion. Local integrity
and origin are not inferred from directory names.

Reading collection metadata and query files requires an explicit selection.
JSON metadata has a default size limit. JSONL loading downloads the selected
file in full, then parses it lazily; it is not remote row streaming. Training
files can be large. The inventory is cached per instance; callers create a new
instance after adding local files. No global singleton or mutable config exists.

## Identity and failure policy

Keep extraction paper IDs, packaged PDF IDs, and original query IDs separate.
Join only through authoritative source paths. Reject ambiguity rather than
choosing the first normalized-name match. Query IDs are collection-scoped;
future exports must record their namespacing policy and reverse mapping.
Figure IDs and zero-based page coordinates remain metadata; no new IDs are
assigned. Lean artifact identity restoration is explicitly deferred.

Missing components raise `MissingComponentError`; malformed content raises
`DatasetFormatError` with filename/line context where applicable. Missing joins
raise `LookupError`; ambiguous joins are format errors. Iterators may yield valid
records before a later error; exhaust them to validate an entire collection.
This is not a transactional corpus export API. Consumers must not publish a
partially consumed export as a successfully validated dataset.

Raw query metadata is retained for provenance, but only `query` is query text.
Only extracted markdown becomes local `Paper.markdown`. Prompts, interpretations,
and generated clues never enter original content implicitly. The paper reader
requires full metadata and nonempty text; it does not certify images, hashes,
or full-corpus extraction success. No missing item is automatically excluded.
