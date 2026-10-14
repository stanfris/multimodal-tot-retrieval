# Retrieval pipeline implementation report

Completed on 13 October 2026. This report covers the executable BM25 pipeline and
the subsequent Sentence Transformers/FAISS Hydra integration. Existing adapter,
configuration, output reservation, and provenance contracts were reused. Existing
standalone scripts and notebooks retain their defaults.

## Executable stages

| Backend / representation | prepare | index | retrieve | evaluate |
| --- | --- | --- | --- | --- |
| BM25 fulltext / title_abstract | Canonical records and coverage exports | Persistent classic Terrier index | Independent configured query collections | Saved-run metrics |
| Dense fulltext / title_abstract | Canonical records and coverage exports | Document encoding and persistent FAISS index | Independent query encoding and FAISS search | Saved-run metrics |
| Dense passages | Canonical paper records plus passage export | Library splitting, document encoding, persistent vector-to-paper mapping | FAISS passage scores, library aggregation, paper cutoff | Paper-level saved-run metrics |

`search`, `export`, visual/page retrieval, and local dense model-path
loading remain unsupported. ColBERT is now covered by the separate
[ColBERT implementation report](colbert-implementation-report.md). Unsupported operations fail explicitly.

## BM25 and canonical data

- Preparation and indexing use the configured `DatasetAdapter`. Dataset source,
  domains, revision, corpus/query splits, query mode, representation, missing-data
  policy, canonical artifact paths and configured Hub component downloads are
  respected. Restored local trees and existing read-only TAR support are retained.
- Every positive judgment must resolve to the selected corpus and domain. An
  explicit missing-data skip policy cannot silently remove positive targets.
- Preparation writes `corpus.jsonl`, `queries.jsonl`, and `qrels.jsonl` using the
  existing record exporter, including content hashes, coverage and provenance.
- Indexing uses `IterDictIndexer`, the classic persistent indexing type, YAML
  metadata capacity and blocks, configured threads, and Terrier version. It checks
  document counts and exact stored paper IDs. No retrieval/evaluation runs here.
- Index manifests contain resolved configuration, its checksum, corpus content
  identity, dataset/model revision provenance, document domains, versions and
  output locations. Local revisions remain caller assertions, explicitly recorded.
- Retrieval verifies the existing index contract, opens the saved index, loads
  queries independently, validates positives against stored corpus identities and
  uses the configured `Retriever` controls/cutoff. Original document text is not
  loaded. Manifests without explicit document-domain identities require rebuilding.

## Batch outputs and independent evaluation

Retrieval saves five immutable, checksummed artifacts:

| Artifact | Contents |
| --- | --- |
| `run.trec` | Standard rankings, canonical paper IDs, one-based ranks, rows in rank order |
| `topics.jsonl` | Selected topics, preserving string identifiers |
| `qrels.tsv` | Validated binary positive judgments |
| `query-ids.jsonl` | Original IDs, scoped keys, reversible framework IDs and domains |
| `corpus-ids.jsonl` | Canonical paper/domain identities needed for independent validation |

Unique query IDs remain unchanged. Collisions across collections use the existing
percent-encoded query-key convention, with the original IDs saved in the mapping.
Numeric-looking IDs and identifiers such as `NA` are preserved as strings.

Evaluation consumes only a completed saved batch. It checks configuration identity,
exact artifact paths/checksums, query/document mapping consistency, positive domain
membership, duplicate/unknown identifiers, finite scores, valid ranks, a single
run identity, and metric cutoffs against the saved retrieval depth. It does not
open the dataset or index, encode queries, instantiate a retriever, or build an
index. Rerunning evaluation writes a separate fresh result directory.

YAML metric expressions are parsed by ir_measures. PyTerrier `Experiment` handles
aggregate and per-query evaluation for saved nonempty rankings. Entirely empty
rankings use ir_measures directly because `Experiment` rejects an empty DataFrame;
no dummy hits or manually computed metrics are introduced. Evaluation writes
`metrics.csv` and `per-query.csv`, with canonical query mappings on per-query rows.
Its manifest references the exact saved run manifest, configuration and batch
files. The resolved evaluation configuration is saved separately in `.hydra/`.

## Dense encoding, caching and persistence

Dense settings map through the existing `dense_arguments` into `EncodingConfig`:
model ID/revision, device, batch size, normalization, explicit query/document
prompts, and encoding/FAISS thread counts. Explicit empty prompts are preserved.
Builds resolve model refs to immutable commits; index consumers must pin the
actual build commit.

The new task-level wrapper reuses `_encode_tasks`, `_read_cache`, `_write_cache`,
`encoding_kwargs`, and `record_digest`. It encodes documents and queries
independently. Cache entries use existing atomic publication and content
verification, with task-specific configuration and library version fingerprints.
Document keys exclude query prompts; query keys exclude document prompts.
Passage aggregation is excluded from the document cache/build identity because
it does not change the encoded text. Unchanged document embeddings and cached
FAISS indexes are reused when publishing a new, fresh index destination.

Persistent dense indexes contain `index.faiss`, ordered `ids.json`, and
`vectors.jsonl` with its sidecar. Every vector maps to a canonical paper/domain.
Index manifests record checksums and locations for those files, encoding metadata,
dimension/count, resolved revisions and cache hits. Retrieval validates them before
query inference and checks query/index dimensions. It does not require the original
document tree or document embedding cache once the index has been published.

CPU execution uses the existing libraries. Configured CUDA execution forwards the
device to Sentence Transformers where supported. FAISS remains the existing CPU
`IndexFlatIP`; no custom or GPU FAISS index implementation was added.

## Passage retrieval

`RecursiveCharacterTextSplitter` receives the YAML chunk size, overlap, separators
and whitespace policy. Its character offsets and chunk text are retained in the
preparation export. Internal passage IDs escape parent identifiers reversibly;
original scientific paper IDs are stored separately.

`representation.aggregation.method` supports `max` and `mean`, using PyTerrier's
`max_passage` and `mean_passage`. FAISS scores all indexed passages for each query
before aggregation; the configured cutoff then counts papers. Both methods have
their full-corpus semantics, and a paper contributes at most one ranking row.
Configuration and output manifests record the method, library, search depth and
paper ranking level. Evaluation always consumes canonical paper-level rankings.

The splitter dependency was added to the dense extra and default example group,
and `uv.lock` was updated and installed. Standalone combined dense examples remain
full-document examples; passage execution is routed through the separate stages.

## Changed implementation areas

| Files | Change |
| --- | --- |
| `src/scientific_tot/hydra_run.py`, `stages.py` | Stage dispatch, saved-artifact preflight, completion/failure manifests |
| `src/scientific_tot/config_arguments.py`, `configuration.py` | Parameter binding, canonical fulltext source for dense passages, aggregation validation, domain-order handling |
| `src/scientific_tot/config_artifacts.py` | Exact artifact references/checksums, batch completeness and experiment verification, aggregation-aware index identity |
| `src/scientific_tot/integrations/pyterrier.py` | Persistent BM25 builder, shared JVM/ID checks, independent canonical query conversion, document-domain provenance |
| `src/scientific_tot/integrations/bm25_batch.py` | BM25 retrieval and shared paper-level batch serialization/saved-run evaluation |
| `src/scientific_tot/integrations/evaluation.py` | Reusable metric-only operation, library-based empty-run handling, preserved tutorial exporter |
| `src/scientific_tot/integrations/dense.py`, `dense_batch.py` | Independent cached encoding, persistent FAISS stages, library passage projection and aggregation |
| `config/representation/passages.yaml` | Explicit max/mean aggregation setting |
| `pyproject.toml`, `uv.lock` | Locked LangChain splitter dependency |
| `tests/test_stages.py`, `test_bm25_batch.py`, `test_dense_stages.py` | Offline stage, integrity, cache, aggregation, and end-to-end coverage |
| `tests/test_configuration.py` | Executable stage expectations and strengthened artifact contract fixtures |
| `README.md`, configuration/architecture/audit documentation | Updated support boundaries, commands and behavior |

Pre-existing workspace changes were preserved. This work does not introduce custom
embedding models, indexing algorithms, similarity functions or evaluation metrics.

## Verification and practical limits

- The BM25-era full offline suite passed: **301 passed, 5 deselected**.
- After dense integration, the focused regression suite passed: **232 passed,
  2 deselected**, including BM25/dense stage tests, configuration contracts,
  existing dense examples and PyTerrier conversion.
- The real offline BM25 end-to-end test completed prepare → index → retrieve →
  evaluate. Repeated evaluation ran in a fresh process without Java startup and
  after the dataset and index were removed. Network access was blocked.
- Dense tests stub only neural encoding. They use real FAISS persistence/search,
  LangChain splitting, PyTerrier aggregation and established metric libraries on
  tiny synthetic data. They verify document/query isolation, cache reuse, prompt,
  revision, device/batch/thread propagation, corrupt artifact rejection, and
  correct max/mean paper rankings.
- Final dense/standalone-example checks passed: **24 passed, 3 deselected** in
  6.81 seconds after the last compatibility guard.
- Ruff and diff whitespace checks passed; the lockfile was checked offline.

No pretrained models were downloaded and no long scientific evaluations were run.
Actual GPU inference was not exercised locally; device propagation was tested.
The exact all-passage aggregation can be expensive on large corpora. Existing eager
record loading also scales with the selected corpus. Large-scale performance,
approximate indexes, streaming preparation, and live corpus completeness remain
outside this change. Offline JVM integration tests explicitly skip when required
Terrier jars are not already cached.
