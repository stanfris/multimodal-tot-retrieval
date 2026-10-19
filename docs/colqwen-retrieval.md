# ColQwen2.5 visual PDF baseline

`experiment=colqwen_pages` executes prepare, index, retrieve, and evaluate using
original PDF pages. Scientific papers remain the retrieval and evaluation unit.
No figure extraction, OCR, fine-tuning, or reconstructed Markdown images are used.

## Model and environment

The official `vidore/colqwen2.5-v0.2` checkpoint is pinned to
`dcbe8d9cede518bce830488364ba0e40c873645b`. Its separately published
`vidore/colqwen2.5-base` weights are pinned to
`92908120384b7a2110c5beda3ab29cbdb2c08e49`.

The locked visual environment uses Sentence Transformers **6.1.0**, Transformers
**5.15.0**, PEFT **0.19.1**, PDFium/pypdfium2 **5.9.0**, Qdrant client **1.17.1**,
Pillow **12.3.0**, Torch **2.14.1**, and torchvision **0.29.1**. See `uv.lock` for
platform wheels and transitive versions. The checkpoint explicitly requires
Transformers >=5.15: earlier releases ignore its adapter key mapping and can
silently initialize incorrect weights. The installed v6 API was inspected:
`encode_document(PIL_images, batch_size=..., convert_to_numpy=True)` and
`encode_query(texts, batch_size=..., convert_to_numpy=True)` return ragged arrays.
The library slices its scoring mask, so padding is not indexed.

The [official ColPali migration recommendation](https://github.com/illuin-tech/colpali)
and [Sentence Transformers API](https://sbert.net/docs/package_reference/multi_vector_encoder/model.html)
recommend `MultiVectorEncoder`. This implementation uses that interface and the
checkpoint's own module definitions, learned projection, normalization, chat
template, tokenizer, and image processor. It supplies no custom prompts or image
transformations. Transformers drops an adapter's revision while resolving base
weights. A temporary local view of the official adapter changes only its base
location to the pinned base snapshot; pretrained weights and processor files
remain unchanged, and the original Hub cache is never edited. ST constructs the
model from this view with `local_files_only=True` after the two pinned downloads.
Manifests record effective module/model/image/processor configurations, template,
prompts, checkpoint file hashes, and dependency versions.

Transformers 5 conflicts with the established ColBERT Transformers 4 environment.
The four stage scripts select `.venv-colqwen` for `experiment=colqwen_pages` or
`model=colqwen`, using the mutually exclusive locked `colqwen` extra. Existing
BM25/dense/ColBERT commands retain `.venv`. To run visual Python commands directly:

```sh
UV_PROJECT_ENVIRONMENT=.venv-colqwen uv sync --locked --no-default-groups --extra colqwen --group dev
UV_PROJECT_ENVIRONMENT=.venv-colqwen uv run --locked --no-default-groups --extra colqwen --group dev python -m scientific_tot.hydra_entry dry_run=true experiment=colqwen_pages
```

## Four stages

Restore source PDFs with the upstream dataset restoration tool into
`DATA_ROOT/pdf_datasets/<source_pdf_relpath>`, alongside the canonical
`preprocessed/` extraction identity directories and query collections. Packaged PDF TAR shards
are not original PDF paths: restore them first. The adapter's existing canonical
paper/source join and split/domain selection are reused. Page selection uses existing extraction IDs and does not require title/abstract
text. When page source metadata is absent, format the canonical ID directly: remove
the leading domain prefix and append `.pdf` within that domain directory. For
example, `ACL_2026.findings-acl.999` resolves to
`pdf_datasets/ACL/2026.findings-acl.999.pdf`. Explicit source metadata takes
precedence. No PDF filename matching or inventory search is used.
A Hub revision can be used only if it contains restored original PDF paths;
this integration does not implement another PDF restoration algorithm.

```sh
export TOT_DATA_ROOT=/your/restored/data
./scripts/01_prepare.sh experiment=colqwen_pages
./scripts/02_index.sh experiment=colqwen_pages
./scripts/03_retrieve.sh experiment=colqwen_pages
./scripts/04_evaluate.sh experiment=colqwen_pages
```

ColQwen selects `launcher=local_gpu` by default and requires `device=cuda`.
CPU and MPS overrides are rejected. Indexing and query encoding fail explicitly
before downloading weights if CUDA is unavailable; there is no CPU fallback.
Preparation and saved-run evaluation do not initialize the model or check GPU
availability, so they can still run on machines without GPUs. PDFium rendering
and Qdrant's local exact search remain CPU operations; model inference always
runs on CUDA. Pretrained CUDA inference has not been verified on this development
machine.

Preparation resolves and opens the selected original PDFs and reads their page counts,
exports canonical corpus/query/qrel records and the page mapping, and initializes
no model and performs no renderability precheck. During indexing, PDFium renders every page in original order at `rendering.scale=2`
(144 pixels/inch for ordinary 72-point PDF units). PDFium's defaults remain in
force: white opaque background, no crop or extra rotation, color rendering,
annotations enabled, normal smoothing, and forms drawn when a form environment
exists. PDFium converts its native bitmap to PIL; the returned image is copied
before releasing bitmap/page resources. Its official model processor performs
resizing, rescaling, RGB conversion, normalization, patching, and tokenization.
PDF user-unit handling follows PDFium; unusual user-unit PDFs may differ from
physical DPI expectations.

Missing, corrupt, empty, and unreadable PDFs follow `dataset.missing=error|skip`.
Reports retain selected papers, emitted usable PDFs, skipped count, and reasons.
Filesystem escapes and malformed identity data always fail. Rendering errors encountered during indexing fail the build. Preparation and retrieval validate
all selected query positives: skipping a relevant paper raises an unresolved
relevance error. Independent indexing does not load queries to judge relevance.

Indexing rasterizes and encodes at most `model.batch_size` pages at once, then
commits each page through Qdrant's persistence API. It creates the empty local
collection with Qdrant, then writes points without retaining a full in-memory
collection. Qdrant
owns vector persistence and MaxSim scoring using DOT distance and MAX_SIM;
HNSW is disabled. The official model supplies normalized token vectors. The
output is `results/indexes/colqwen_pages_d-ACL_c-all_<hash>/`, with `.hydra/`, both
completion manifests, `page-mapping.jsonl` and its report sidecar, and `qdrant/`
SQLite/configuration files. Each ordered Qdrant integer position maps explicitly
to an escaped deterministic page ID, canonical `doc_id`, domain, and one-based
page number. Canonical IDs are never interpreted as filesystem paths.

Shared identity includes canonical IDs, original PDF SHA256/page counts/source
joins, dataset source/repository/revision, pinned adapter/base revisions,
checkpoint-owned processor selection, model/device/document batch settings,
rendering scale/library, index settings, encoding threads, and relevant versions.
Effective default preprocessing is determined by those pinned checkpoint files
and library versions. Query split/mode/batch size, paper depth, and metrics are
excluded. Completed identical indexes are checksummed and reused without writes.
Interrupted indexes can resume; incompatible or changed artifacts are rejected.

### Resuming an interrupted index

Rerun the original command with the same environment and index settings:

```bash
./scripts/02_index.sh experiment=colqwen_pages
```

For a saved `started` or `failed` ColQwen index, the command verifies the saved
configuration and original directory content hash against the current PDFs and
dependency versions. It streams the existing points from SQLite and validates
their IDs, page-to-paper payloads, vector dimensions and finite values. Points
must form a contiguous prefix of the expected page order. It then reports:

```text
Resuming index: 282,258/353,849 pages already stored
```

Only the remaining pages are rendered and encoded, including when the previous
run stopped partway through a PDF or batch. Each point is committed separately,
so another interruption can resume from the last committed page. Existing
points are not rewritten. Artifact and Qdrant locks prevent concurrent writers.
Indexing and resumption do not load the stored collection into RAM. Resumption
does scan all saved point blobs first, which can take time for a large database.

Keep the original index directory, saved configuration and PDFs intact. Use only
your own trusted index, since Qdrant persists Python pickle objects. Changed
inputs, corrupt records or gaps in stored positions fail instead of silently
rebuilding or skipping pages. Once all pages are stored, normal metadata and
completion manifests are written. A verified backend manifest left by an
interruption during final sealing is reused without loading the model.

### Finalizing an interrupted index

If indexing stopped after persisting all pages but before writing metadata, use
the recovery command from the repository root, in the original environment:

```bash
.venv-colqwen/bin/python -m scientific_tot.recover_colqwen \
  results/indexes/colqwen_pages_d-ACL_c-all_<hash>
```

Use only your own trusted index: Qdrant persists Python pickle objects. Stop any
indexing process first. Recovery holds Qdrant's lock and opens SQLite read-only;
it streams one point at a time rather than loading the collection into memory.
It checks the original directory's content identity against the current PDFs,
saved configuration and dependency versions, then checks every page position,
payload and vector shape/finite value. Missing pages or changed inputs fail
without publishing completion. This command finalizes complete data; it does
not resume a partial encoding run.

Recovery loads the pinned model on CUDA to reconstruct its effective processor
and checkpoint metadata, but does not render or encode pages. That metadata is
reconstructed, since the interrupted run did not save it; recovery cannot prove
that each stored vector was produced by those weights. After validation, it
writes the normal mapping and index manifest and atomically publishes the
completed stage manifest, retaining the original manifest in its recovery field.
Existing valid index completion metadata is verified and reused. Full
validation and repeated file checksums can take substantial time for large
databases. Leave the original PDFs and index files unchanged throughout.

Indexing and recovery avoid loading the stored collection into RAM. Retrieval
still uses local Qdrant, which loads persisted points into memory and requires enough RAM
for the collection, and makes a temporary disk copy of the database.

Retrieval reads only a completed index and selected queries. Automatic selection
uses saved index metadata without reopening the PDF collection. If several
compatible indexes exist, provide `paths.index=...` explicitly. This is useful
when the same configuration has been indexed against different PDF contents.
It encodes text in query batches with the official query encoder. It never
renders or encodes document pages. Local Qdrant requires writable SQLite/lock
files, so retrieval opens a disposable copy of the sealed Qdrant directory;
this preserves the published artifact and permits concurrent independent runs.

## Ranking and evaluation

The [ColPali paper](https://arxiv.org/abs/2407.01449) motivates direct visual
page representations and late interaction. [M3DocRAG](https://arxiv.org/abs/2411.04952)
demonstrates visual page retrieval across multi-page, multi-document collections.
[Multipage Document Retrieval by Textual and Visual Representations](https://marcalr.github.io/pdfs/ICPR12.pdf)
discusses combining page evidence, including CombMAX. Here the benchmark's qrels
judge papers, so we apply `s(q,paper) = max_page s(q,page)` through
`pyterrier.text.max_passage`. This is our benchmark-specific aggregation choice,
not part of the ColPali architecture, and does not combine evidence across pages.

Persistent alternatives were considered: `MultiVectorEncoder.similarity` is a
scorer, not a persistent index; FAISS's existing single-vector index cannot
supply exact late interaction; FastPlaid/PLAID uses approximate candidate
retrieval. [Qdrant multi-vectors](https://qdrant.tech/documentation/concepts/vectors/#multivectors)
provide library-owned persistence and MaxSim with exact local search. This first
baseline exhausts the page collection for each query before aggregating. Thus
`retrieval.top_k` counts papers even if the highest-ranked pages all belong to
one paper. Ties are ordered by ascending canonical paper ID; fewer available
papers yield fewer results. No approximate retrieval quality claim is made.
Synthetic-vector parity with ST's official MaxSim, including negative scores,
was checked; real pretrained score parity remains an opt-in test.

The existing shared writer produces `run.trec`, topics, qrels, query-ID and
corpus-ID mappings under `results/models/colqwen_pages_c-all_q-test_<hash>/`.
Additional `page-scores.jsonl` preserves every page score and canonical mapping;
these rows are streamed to disk. Run manifests record exact search depth, maximum
aggregation, paper cutoff, and tie-breaking. Evaluation reads only the saved batch
and writes existing aggregate/per-query metrics; it can run after PDFs, model
cache, and index are unavailable. No new metrics or model imports are introduced.

## Resources and verification

This is a correctness baseline for collections that fit in local Qdrant's RAM.
The 3.8B base model alone needs roughly 15 GB for float32 weights or 7.6 GB for
16-bit weights, excluding activations and framework overhead. Loading uses
library defaults; no precision/quantization or custom GPU-memory controls are
introduced. Budget additional RAM/VRAM for image tokens and page batches.
At 128 dimensions, float32 token arrays need 512 bytes/token before SQLite and
payload overhead; actual page token counts depend on the processor and page
shape. Local Qdrant loads the entire vector collection into memory and warns
above 20,000 points. Exact search scales with all page/token vectors per query.
A temporary index copy needs additional disk space approximately equal to the
persistent Qdrant directory. Exhaustive page-score exports grow as queries ×
pages. Input identity checks read PDF hashes and page counts without rendering pages;
page images are rendered only during document encoding. No full-corpus timing, GPU compatibility,
or scalability measurements have been performed.

Default tests use tiny synthetic PDFs, real PDFium rendering, Qdrant persistence
and search, PyTerrier aggregation, shared serialization/evaluation, and mocked
neural inference; they block network access and download no weights. Separately:

```sh
UV_PROJECT_ENVIRONMENT=.venv-colqwen uv run --locked --no-default-groups --extra colqwen --group dev pytest -m colqwen_live tests/test_colqwen.py
```

The real checkpoint test requires CUDA, loads actual weights, exercises image and
text encoding, and compares Qdrant scores to the model's official similarity.
See the [implementation report](colqwen-implementation-report.md) for checks
actually executed and their limitations.
