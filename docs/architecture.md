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

## Optional PyTerrier boundary

`integrations/pyterrier.py` is an optional conversion/example layer; the public
core imports none of its dependencies. It provides ordinary document dictionaries
and pandas tables, and calls existing Terrier indexing/BM25 and Experiment
metrics directly. There is no additional retrieval orchestration abstraction.
The optional TREC run export uses PyTerrier's writer. All canonical positive
IDs are validated before indexing, and indexed IDs/counts are checked afterwards.

`bibliography.py` provides a separately invoked literal Markdown projection for
extractions with recognizable original title/Abstract headings. It records source
character spans, keeps raw metadata intact, and never generates replacement
content. It does not alter the core loader's bibliographic field policy.


## Optional dense tutorial boundary

`integrations/dense.py` calls Sentence Transformers' query/document encoding
methods and FAISS's index/search/I/O APIs. It does not implement neural models,
similarity search or fine-tuning. `integrations/evaluation.py` is now shared by
BM25 and dense examples; canonical identity conversion and subset selection
remain shared too. Model snapshots use the library cache, embeddings use NumPy
persistence, and FAISS owns the binary index format. Cache manifests bind those
files to inputs, model SHA and configuration and detect corruption before reads.
The core adapter imports none of the optional neural dependencies.

## Shared repository environment

`uv sync --locked` installs the default `dev` and `examples` groups into `.venv`.
All CLI examples and both notebooks use that environment and the `scientific-tot`
kernel. Package extras remain available for library consumers, but repository
workflows never switch extra sets. BM25 selects the pinned jdk4py runtime before
Java initialization; it does not depend on a system JDK. Linux uses CPU Torch.

`integrations/selection.py` is the common fulltext example-loading path.
Retrieval never requests title/abstract representations or PDF metadata. All
positives must resolve before indexing or encoding. The example defaults use
`corpus_split="all"`, unlimited documents/test queries, and original fulltext.
Notebook inputs default to restored data, with fresh output paths per execution;
fixture demos require an explicit data-root override.


## Experiment configuration boundary

Hydra's independent groups in `config/` are the experiment-facing authority.
`configuration.py` validates YAML-owned keys, values, compatibility and managed
paths without importing optional backends. Plain functions in `config_arguments.py`
bind existing Dataset/adapter, Terrier, and EncodingConfig arguments; the adapter
is unchanged. Existing tutorial functions accept resolved `settings` without
changing defaults for callers that omit them.

`hydra_run.py` reserves fresh outputs, preflights every basic-sweeper job, stores
read-only configuration/provenance and validates stage artifacts. `config_artifacts.py`
provides index build-setting identity and retrieval snapshot verification.
`stages.py` executes BM25/dense/ColBERT preparation, indexing, retrieval and saved-run
evaluation through the existing adapter and backend integrations. Preparation exports validated canonical records;
indexing builds persistent Terrier/FAISS indexes and saves completion manifests.
Dense task caches reuse existing encoders/cache I/O; LangChain splits passages and
PyTerrier aggregates their FAISS scores to canonical papers. Retrieval saves
checksummed batches; evaluation consumes them without dataset/index/model access.
Remaining stage/model combinations are unimplemented. Combined tutorial functions
are never used as stage implementations. Reusable caches must be outside experiment results.
See the [configuration guide](configuration.md) and [complete profile audit](configuration-audit.md).

## ColBERT textual boundary

`integrations/colbert.py` binds the existing four stages to Stanford colbert-ai
Indexer/Searcher. An immutable model snapshot supplies tokenization/inference;
the library owns residual compression and late-interaction search. Ordered PIDs
map to canonical papers through a checksummed index mapping. Shared identity,
manifest, batch serialization and saved-run evaluation mechanisms are reused.
Fulltext is prefix-truncated to `doc_maxlen` tokens without passage splitting.
See [the ColBERT guide](colbert-retrieval.md).
