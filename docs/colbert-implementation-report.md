# ColBERTv2 implementation report

Implemented 14 October 2026. This adds one textual ColBERTv2 baseline to the
existing four-stage pipeline and preserves pre-existing workspace changes.

## Files and integration

- Added `src/scientific_tot/integrations/colbert.py`: pinned checkpoint snapshot,
  library config binding, independent compressed indexing, canonical PID mapping,
  independent search, checkpoint byte verification and shared batch serialization.
- Added `config/experiment/colbert_fulltext.yaml`; updated the existing model YAML
  with the immutable checkpoint revision while preserving all five parameters.
- Extended `stages.py` and `hydra_run.py` dispatch and verified ColBERT index reuse;
  existing BM25/dense skip behavior is retained.
- Extended `artifact_identity.py`, `configuration.py` and `config_artifacts.py`
  for ColBERT dependency fingerprints, fulltext/CUDA compatibility, immutable
  model revisions and acceptance of standard ColBERT retrieval artifacts.
- Updated `pyproject.toml` and `uv.lock` for Stanford ColBERT and Transformers;
  changed Hub entry-class imports in `data/hub.py`, `test_data.py` and
  `test_adapter.py` to support the locked Hub version.
- Added `tests/test_colbert.py`; updated the network opt-in marker in
  `tests/conftest.py` and the configuration matrix/preset tests.
- Updated README, architecture, configuration, configuration audit and the prior
  implementation report; added the ColBERT experiment guide and this report.

Stanford `colbert-ai==0.2.22` owns tokenization, inference, k-means/residual
compression, candidate generation and late-interaction scoring. `Indexer.index`
receives ordered original text; `Searcher.search` receives independent selected
queries and the requested cutoff. Returned PIDs map to canonical paper IDs.
`save_batch` owns serialization; the existing saved-run evaluator is unchanged.
No custom neural model, MaxSim, search algorithm, visual retrieval, passages,
fine-tuning or additional metrics were implemented.

## Versions and provenance

The inspected and installed wheel is `colbert-ai 0.2.22`. Transformers is pinned
at `4.57.6`, with Hub `0.36.2`, Torch `2.14.1`, FAISS CPU `1.15.1`, and NumPy `2.4.6`,
all selected in `uv.lock`. Python is `3.11.14`. The obsolete upstream Torch
extra is not installed. Transformers 4's Hub constraint required lowering the
core minimum to 0.34; public `huggingface_hub.hf_api` entry-class imports work
across these Hub versions without a compatibility fallback.

The default checkpoint is `colbert-ir/colbertv2.0` at
`c1e84128e85ef755c096a95bdb06b47793b13acf`, resolved against the Hub. The library
receives a local immutable snapshot path because its checkpoint API lacks a
revision argument. Manifests store both the SHA and checkpoint byte digest,
corpus content and dataset provenance, exact PID ordering/domains, dependency
versions, full configuration, and checksummed compressed index files. Explicit
local checkpoint bytes enter index identity. Completed ColBERT indexes are
validated and reused; inconsistent and incomplete artifacts are rejected.

The index identity excludes query selection, query token limit, ranking depth
and metrics. Shared retrieval artifacts preserve canonical identifiers and
one-based TREC ranks. Saved-run evaluation requires neither original corpus
nor model weights nor index files.

## Commands and truncation

```sh
uv sync --locked
export TOT_DATA_ROOT=/your/restored/data
./scripts/02_index.sh experiment=colbert_fulltext
./scripts/03_retrieve.sh experiment=colbert_fulltext
./scripts/04_evaluate.sh experiment=colbert_fulltext
```

Optional preparation uses `01_prepare.sh` with the same preset. The model
profile selects CUDA by default; indexing/retrieval require an NVIDIA GPU.
Saved-run evaluation requires no GPU. Fulltext is
original extracted Markdown, prefix-truncated by the library to at most 220
WordPiece tokens including CLS, D marker and SEP. There is one PID per paper;
no passage splitting/aggregation. The policy is explicit in index provenance.

## Verification

- Final complete offline suite: **341 passed, 6 deselected** in 68.43 seconds.
- Focused final ColBERT offline suite: 9 passed, 1 deselected.
- Ruff lint and formatting checks and Git diff whitespace check passed.
- `uv lock --check --offline` passed; `uv sync --locked` installed the environment.
- Real pretrained ColBERTv2 CPU smoke test: **1 passed, 9 deselected** in
  121.33 seconds. It downloaded the pinned checkpoint, compiled the upstream
  native extensions, encoded two documents, trained 256 centroids, built the
  two-bit residual-compressed index, saved two canonical rankings, and evaluated
  the saved run after deleting the dataset and index. No inference was mocked.
- Index/retrieve/evaluate stage-script configuration planning passed in dry-run mode.

The real test was invoked with `.venv/bin` on PATH for Ninja,
`TOT_CACHE_ROOT=/tmp/tot-colbert-live-cache` for a stable reusable model/native
cache, and the process-local OpenMP workaround described below. CUDA was not
available and GPU execution remains unverified. Upstream emitted deprecation
and CPU AMP warnings; they did not prevent completion.

The offline suite includes existing BM25/dense tests. ColBERT tests mock expensive
library indexing/search but use real adapter loading, identity/manifest checks,
query conversion, serialization and metric libraries. Tests cover all YAML
parameters, canonical IDs, independent operations, reuse across query settings,
corrupt/incompatible indexes and runs, and checkpoint byte changes.

## Practical limits

ColBERT indexing/retrieval require CUDA, a C++ compiler, Ninja and a CUDA toolkit.
CPU/MPS execution is rejected. The earlier CPU smoke result above records the
initial implementation; the current opt-in test requires a GPU and skips without
one. CUDA default/requirement and unavailable-GPU errors are covered offline.
The upstream automatic k-means partition heuristic can fail on very small corpora
of short documents; the live test uses two long synthetic documents and does not
change the algorithm. Full scientific datasets and GPU performance have not been
measured. Existing eager corpus loading and the upstream in-memory collection
scale with corpus size.

This macOS host has duplicate OpenMP runtimes in Torch/FAISS: an unmodified test
process aborts. Local verification uses `KMP_DUPLICATE_LIB_OK=TRUE` only in the
shell environment, which is an unsupported host workaround and is not injected
by application code. This does not certify production macOS execution; a Linux
CUDA environment is the deployment verification target.

## CUDA default correction

The ColBERT model now overrides the launcher to `local_gpu`, matching dense.
Indexing and retrieval reject CPU configurations and unavailable CUDA; there is
no CPU fallback. RunConfig always selects one GPU. Saved-run evaluation still
requires no CUDA runtime. Offline tests simulate GPU availability for mocked
inference and verify the CUDA default, one-GPU binding and both rejection paths.

Validation after this correction: 170 focused ColBERT/configuration tests passed
(excluding the unrelated in-progress ColQwen profiles); 45 BM25/dense/stage-CLI
regression tests passed. Ruff and formatting checks for changed Python files and
Git whitespace checks passed. The real ColBERT test now skips on this host because
no NVIDIA GPU is available. The unfiltered configuration run had one failure in
an existing in-progress ColQwen YAML profile test, unrelated to this change.

A failed CPU attempt leaves an incomplete reserved index. Preserve it by renaming
`results/indexes/colbert_fulltext_d-ACL_c-all` before retrying the normal three commands;
incomplete artifacts are never silently overwritten.
