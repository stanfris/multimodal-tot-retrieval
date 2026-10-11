# Architecture

`Dataset` owns discovery and exact-file retrieval. `DatasetAdapter` selects
components and converts corpus, query, and relevance records. The public
`load_corpus`, `load_queries`, `load_qrels`, and `load_metadata` functions are
thin convenience wrappers. Shared Dataset instances pin all reads to one Hub
commit; local roots never fall back to a remote source.

`data/artifacts.py` presents restored files or uncompressed TAR members as one
logical extraction tree. It indexes TAR headers once, then seeks selected
metadata/text members. A paper may span multiple domain shards. It never
extracts files, decodes images, or implements PDF restoration/checksums. This
avoids duplicating upstream restoration utilities; their restored output is
supported unchanged. Physical restoration remains an external preparation step.

`data/identifiers.py` joins authoritative source paths and computes only lookup
candidates for existing lean directory IDs. It does not assign new canonical
IDs. Ambiguity is fatal. `data/validation.py` checks published query and PDF split
invariants. The source generation repository remains an inspection reference,
not a runtime dependency.

## Record contract

`Records` is an eagerly validated sequence of ordinary JSON-compatible records,
stored internally as JSON to isolate them from caller edits. It exposes a
structured `Report` and provenance. Separate corpus, query, qrel, and metadata
collections can be loaded/exported independently. Eager validation avoids
publishing a result whose iterator fails halfway through consumption; it also
means memory use scales with the selected corpus. Large streaming exports are
future work, not an undocumented promise.

JSONL is the Phase 2 export format. Each export has a checksum/provenance/report
manifest. Data and manifest writes are individually atomic, not a two-file
transaction. An output hash detects interrupted/stale pairs. Arrow is optional
for reading the published PDF index; no `datasets` or retrieval framework is
needed for the adapter. No Parquet corpus exporter or TREC adapter is implied.

## Identity and provenance

Extraction paper IDs are corpus `doc_id`; packaged PDF IDs stay in
`pdf_document_id`. Full metadata provides authoritative source paths. Lean
extractions retain their existing directory ID, using normal/collision path
candidates only to establish a unique source join against the split/PDF index.
Absent or ambiguous mappings never lead to invented identifiers. Figure IDs are
preserved; lean image paths without metadata get null figure IDs. Page indices
remain zero-based when present; no page IDs are generated.

Canonical query IDs are scoped by collection. An additional, explicit JSON
compound `query_key` records domain/collection/mode/original ID. Queries and
qrels expose both fields. Relevance is a declared binary mapping of source
`relevant_ids` to grade 1, with no inferred negatives or quality-judgment grades.
Every positive must resolve to a corpus record in the correct domain. Different
known revisions/repositories are rejected; local provenance remains a caller
assertion, not a certificate of origin.

Corpus content comes only from original markdown or available original
bibliographic fields. Generated clues, prompts, and interpretations remain query
provenance. Raw paper/PDF metadata, licenses, and unknown query fields survive
conversion. Collection metadata is preserved in query provenance. No title or
abstract is inferred from a filename, prompt, or LLM.

## Selection and failure policy

Exact-file downloads use `huggingface_hub` and its immutable-revision cache.
Discovery does not download contents. Queries select by domain/split/mode;
metadata selects the global index then filters locally. Archive selection can
only narrow domains without an upstream per-document shard manifest; split
membership filters emitted corpus records after indexing all selected-domain
shards. No full-corpus/PDF transfer happens just to inspect metadata or queries.

`train+test` means the union of published PDF split lists. `all` means the corpus
including distractors. Missing/incomplete documents are strict errors unless
`missing='skip'` is chosen; exclusions remain in the report. Missing figures or
bibliographic fields are nonfatal for usable text. Malformed records, unsafe
paths, duplicate identities, and ambiguity always fail. Missing entire components
raise actionable `MissingComponentError`. Unresolved relevance targets always
raise `UnresolvedRelevanceError`, even with a partial corpus.

Fulltext hashes are checked when supplied, but `_SUCCESS` is not mandatory for
restored/lean data. Coverage can only be certified against actual split lists or
a PDF index; `all` without an index describes discovered artifacts, not every
source PDF. The current pinned Hub release has queries only. Source-verified
archive/index contracts are covered by synthetic fixtures but remain unverified
against a live full-corpus publication.
