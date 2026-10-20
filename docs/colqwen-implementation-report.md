# ColQwen2.5 implementation report

Implemented on 14 October 2026. The independent Hydra prepare/index/retrieve/
evaluate stages now support original PDF pages through
`experiment=colqwen_pages`. BM25, dense, and textual ColBERT behavior is preserved.
The [visual guide](colqwen-retrieval.md) provides the complete commands, ownership,
scientific references, persistence semantics, resource estimates, and limitations.

## Files added and modified

Added:

- `src/scientific_tot/integrations/colqwen.py`: PDF boundary validation and
  rasterization, official model loading, Qdrant build/search, paper aggregation,
  page-score export, and stage implementation.
- `config/model/colqwen.yaml`, `config/experiment/colqwen_pages.yaml`: executable
  model profile and composition-only preset.
- `scripts/environment.sh`: select the isolated locked visual environment.
- `tests/test_colqwen.py`: offline lifecycle and selectable checkpoint tests.
- `docs/colqwen-retrieval.md`, this report: usage and implementation details.

Modified:

- `adapter.py`, `config_arguments.py`: reuse canonical selection with a pages
  representation that does not require extracted text; resolve original PDFs
  under the dataset root with filesystem safety checks.
- `configuration.py`: validate model/representation compatibility, pinned
  adapter/base revisions, official processor, batches, exact DOT/MAX_SIM settings.
- `artifact_identity.py`: PDF content identity, dependency fingerprint and
  content-addressed visual index names; exclude query configuration from builds.
- `config_artifacts.py`, `hydra_run.py`, `stages.py`: reuse manifest validation,
  index completion/reuse, dispatch and independent saved-run evaluation.
- `stage_cli.py`: automatic visual index selection from saved metadata without
  opening original PDFs; require an explicit path for ambiguous versions.
- `config/representation/pages.yaml`: retain original page scale/library, remove
  the obsolete configuration-only comment.
- `scripts/01_prepare.sh` through `04_evaluate.sh`: use shared environment selection.
- `pyproject.toml`, `uv.lock`, `.gitignore`: exact visual extra, mutually exclusive
  environments, lightweight offline PDF/index test dependencies, test marker,
  and environment exclusion.
- `tests/conftest.py`, `tests/test_configuration.py`: network policy for opt-in
  inference and composition coverage for the new image model.
- README, architecture/configuration/audit docs, previous implementation report:
  update execution boundaries and point to the visual guide.

## External libraries and reproducibility

Model: `vidore/colqwen2.5-v0.2` at
`dcbe8d9cede518bce830488364ba0e40c873645b`, with base
`vidore/colqwen2.5-base` at `92908120384b7a2110c5beda3ab29cbdb2c08e49`.
Pinned direct versions: Sentence Transformers 6.1.0, Transformers 5.15.0,
PEFT 0.19.1, pypdfium2 5.9.0, Qdrant client 1.17.1, Pillow 12.3.0.
The installed visual lock resolves Torch 2.14.1, torchvision 0.29.1,
NumPy 2.4.6, tokenizers 0.22.2, huggingface-hub 1.33.0, accelerate 1.15.0.
Textual dependencies retain their established ST 5/Transformers 4 versions.

PDFium renders all original pages in order at scale 2, with its established
color/background/annotation defaults. The official checkpoint's image processor,
query/chat template, projection, normalization and scoring mask are preserved.
ST performs query/image encoding in configurable batches. A temporary adapter
view points only the base location at its immutable snapshot, because the
underlying loader discards the adapter revision when fetching base weights.
No model weights, prompts, preprocessing, or similarity implementation is edited.
Effective model/module/processor/template settings and checkpoint hashes are saved.

Qdrant supplies persistent SQLite storage and exact DOT/MAX_SIM scoring. Its local
mode was selected after considering ST similarity (no persistent index), the
existing single-vector FAISS backend (no equivalent MaxSim index), and approximate
PLAID-style candidate indexing. Qdrant has no HNSW graph in this baseline. Search
streams read-only SQLite points once for all queries without opening a local
client or making a temporary index copy. Batched NumPy DOT/MaxSim is checked
against the pinned Qdrant scorer.

Shared identities include canonical papers, PDF SHA256/source joins/page counts,
dataset provenance, pinned adapter/base, rendering, model/index configuration,
threads and relevant versions. Processor defaults are fixed transitively by
checkpoint revision and library versions. Query split/mode/batch size, paper depth
and evaluation settings do not change the index identity. Existing identical
completed builds are checksummed and reused; failed/corrupt/incompatible builds fail.

Page mappings explicitly retain canonical IDs, domains and one-based page numbers.
Qdrant owns page scores. PyTerrier `max_passage` computes each paper's maximum page
score, then ascending canonical ID resolves score ties. Exhausting all pages before
the paper cutoff avoids the top-k-pages error. This maximum aggregation is our
benchmark's paper-level choice, not part of ColPali's architecture. All page scores
are streamed to an additional checksummed JSONL; the shared batch writer saves
paper-level TREC and canonical query/corpus/qrel artifacts. Evaluation uses the
existing saved-run implementation without model, PDF or index access.

## Verification actually executed

- Full default offline suite: **362 passed, 7 deselected**, including existing
  BM25/dense/ColBERT regressions and visual lifecycle tests. Command:
  `KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/pytest -q`. An initial invocation without
  this environment workaround aborted in FAISS/OpenMP initialization on macOS;
  no repository-wide threading workaround was added.
- Focused visual/configuration suite: **179 passed, 1 deselected**.
- Focused visual/stage-CLI suite: **25 passed, 1 deselected**.
- Final adapter/configuration/stage-CLI/visual checks after metadata/provenance
  review: **220 passed, 1 deselected**.
- Visual tests in the actual ST 6 locked environment: **18 passed, 1 deselected**.
  Neural inference in these tests is mocked; PDFium, Qdrant, PyTerrier aggregation,
  serialization and saved-run metrics are real.
- Optional actual checkpoint test: **1 skipped, 18 deselected**; the available macOS environment has no CUDA GPU. No pretrained model
  weights were downloaded and no actual ColQwen inference is claimed.
- Actual official processor-only loading: downloaded small config/tokenizer/template
  files for the pinned revision, then loaded `Qwen2_5_VLProcessor` and
  `Qwen2VLImageProcessor` under the locked dependencies. No model weights involved.
- Actual ST 6 library-owned MaxSim versus Qdrant DOT/MAX_SIM: matching scores on
  synthetic ragged arrays, including negative similarities. This checks scorer
  compatibility and is not real-model inference.
- All four requested shell entrypoints passed `experiment=colqwen_pages dry_run=true`
  with the isolated environment. These validate configuration/dispatch planning;
  they do not imply a real scientific index was built.
- Ruff lint and formatting checks, `git diff --check`, and offline
  `uv lock --check` passed. The UV cache directory was explicitly set to an allowed
  temporary directory for the offline lock check.

Offline tests cover missing/corrupt PDF failures and rendering-free preparation/identity checks, path safety, canonical
IDs with slash/percent/Unicode, rendering scale and order, model/base pin propagation,
independent index creation/reuse/corruption, query-only retrieval, maximum aggregation,
deduplication, paper cutoff when the top three pages share a paper, stable ties,
exhaustion below the cutoff, and evaluation after dataset/index removal. Default
tests block network connections. Every new YAML parameter binds a runtime API
argument, selection, or supported-value constraint; the experiment copies no settings.

## Practical limits

Original PDFs must already be restored; packaged PDF TAR restoration is delegated
to the existing upstream tool. Retrieval scans the stored vectors once for all
queries, retaining one page's vectors and compact query-by-paper maxima. Exact scoring
still visits all pages for every query. Preparation and content-identity checks
read hashes and page counts without rendering pages. Pages are rendered only for
document encoding; there is no page-image cache. Page-score exports scale with
queries × pages.
Canonical TREC identifiers retain the shared existing no-whitespace restriction.
No real pretrained inference, GPU compatibility, full-corpus performance, numerical
parity across devices, or large-index scalability was measured. The selectable
real-model test remains the required next verification on a suitable CUDA machine.

ColQwen now defaults to `launcher=local_gpu` and rejects CPU/MPS model devices.
Model loading checks CUDA availability before checkpoint downloads; preparation
and saved-run evaluation still require no GPU. PDFium and local Qdrant search
remain CPU operations. CUDA defaults, CPU/MPS rejection, unavailable-GPU failure,
and CUDA device forwarding are covered by offline tests.

After enforcing CUDA, the full offline suite passed: **366 passed, 7 deselected**.
The focused visual/configuration suite passed: **184 passed, 1 deselected**.
Ruff lint/format, offline lockfile and whitespace checks also passed.

A real lean-corpus failure exposed a path-formatting omission: with all-corpus
selection and no metadata index/source fields, every `source_pdf_relpath` was
unset despite restored originals. For page records without explicit source paths,
format the original filename directly: `ACL_2026.findings-acl.999` becomes
`ACL/2026.findings-acl.999.pdf`. Remove only the leading domain prefix and append
`.pdf`; no filename matching or PDF inventory search is added. Explicit metadata
paths and canonical IDs are retained. The lean-layout regression exercises real
PDF rendering, indexing and retrieval with mocked neural inference. Textual
source-selection behavior remains unchanged.

Streaming retrieval verification (2026-10-20): offline tests compare every page
score and paper ranking against a real local Qdrant client on ragged synthetic
vectors, including negative scores, ties, multiple query batches and a final
partial batch. Retrieval tests prohibit opening or copying the full collection
and verify the source index remains sealed. A rendering-failure regression checks
that previously rendered batch images are closed. The visual suite passed in
both environments: **47 passed, 1 deselected** in each; neural inference is mocked.

A small macOS loading-only benchmark used 128 persisted pages, each containing
256 tokens of 128 float32 components, in separate fresh processes. Copying and
opening the local client took 0.206 s before the first page was available and
increased peak RSS by 202.6 MiB. Reading the first streamed page took 0.006 s;
validating the whole stream took 0.097 s and increased peak RSS by 4.3 MiB.
These figures exclude model loading, scoring, shared identity checks and output
writing; they do not establish full-corpus retrieval throughput. This initial
streaming version re-read stored points per query batch and is superseded below.

Single-pass retrieval now encodes queries in GPU batches of two, retains CPU
query embeddings, and scans each stored page once for all queries. CPU scoring
concatenates the token matrices of up to `model.retrieval.scoring_batch_size`
queries (default 16), computes one DOT product, and applies the usual per-query
MaxSim reduction. Float64 query-by-paper maxima use 8 × queries × papers bytes;
the buffer size is reported before scoring. Query batch controls are excluded
from index identity, so existing indexes remain compatible. The page progress bar
now says `Scoring PDF pages`, including validation, scoring and JSON output.

Offline parity tests cover 2- and 128-dimensional ragged vectors, negative scores,
ties, GPU encoding batch sizes 1/2, and CPU scoring group sizes 1/2/16. They
require exactly one database scan and matching paper rankings, with page scores
matching Qdrant at 1e-12 relative/absolute tolerance. Floating-point batching can
change final rounding; this remains exhaustive MaxSim rather than approximation.

A synthetic single-BLAS-thread comparison used 110 ragged queries (16–19 tokens),
32 pages (256 tokens each), and 128 dimensions. Repeated decoding and individual
Qdrant scoring took 1.256 s; single-pass decoding and batched scoring took 0.037 s
(33.6×), with zero score difference in that sample. This excludes SQLite I/O,
GPU encoding and JSON output, and does not predict DGX03 end-to-end throughput.
