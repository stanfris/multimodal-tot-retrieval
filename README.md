# scientific-tot-retrieval

Canonical dataset adapters for the
[scientific tip-of-the-tongue dataset](https://huggingface.co/datasets/kasys/open-source-scientific-documents).
Load selected original-content corpus records, queries, and validated relevance
judgments, then export standard JSONL for existing retrieval frameworks.

The published dataset is the source of truth. The
[generation repository](https://github.com/stanfris/multimodal-tip-of-tongue-retrieval-for-scientific-papers)
creates the artifacts; this package does not depend on that pipeline at runtime
and delegates ranking, pretrained encoding, and evaluation to PyTerrier,
Sentence Transformers, FAISS, Stanford ColBERT, Qdrant, and ir_measures.

**Publication status rechecked 2026-10-11:** revision
`3b764a4c6855925ec802a2560712c4f5cc96f980` contains query collections only.
Corpus archives, `metadata.parquet`, and `splits/` are absent. Queries can be
loaded from the Hub now; full corpus/qrel conversion requires appropriate local
artifacts or a future publication containing them. Missing data is reported,
never fabricated. [Evidence and data contract](docs/dataset-schema.md).

This is the reference for the 11 core methods. All methods use the original
textual ToT query strings and return paper rankings. No fine-tuning is performed.
Configuration values below come from `config/model/`, `config/representation/`,
and the integrations in `src/scientific_tot/integrations/`.

## Core benchmark

Environment **default** means the locked `.venv`; **ColQwen** means the separate
locked `.venv-colqwen`. CUDA requirements in this table apply to indexing and
retrieval. Preparation and saved-run evaluation require no GPU.

| Method | Checkpoint/model | Document input | Indexed unit and paper-level aggregation | Configuration selector | CPU/GPU and environment |
| --- | --- | --- | --- | --- | --- |
| BM25 fulltext | PyTerrier/Terrier BM25; no neural checkpoint | Full original Markdown text | One text document per paper; ranks papers directly | `experiment=bm25_fulltext` | CPU, bundled Java; default environment |
| E5 fulltext | `intfloat/e5-small-v2` | Full original Markdown text | One 384-dimensional vector per paper; ranks papers directly | `model=dense representation=fulltext` | CUDA encoding required; FAISS search as below; default environment |
| E5 passages | `intfloat/e5-small-v2` | Overlapping passages of original Markdown | One vector per passage; maximum score across all passages per paper by default | `experiment=dense_passages` | CUDA encoding required; FAISS search as below; default environment |
| ColBERT fulltext | `colbert-ir/colbertv2.0` | Original Markdown, token-truncated | One paper PID with token multi-vectors; ColBERT late interaction ranks papers directly | `experiment=colbert_fulltext` | CUDA required; CUDA toolkit, compiler and Ninja; default environment |
| ColQwen pages | `vidore/colqwen2.5-v0.2` with `vidore/colqwen2.5-base` | Individually rendered original PDF pages | One multi-vector set per page; exact MaxSim per page, then maximum page score per paper | `experiment=colqwen_pages` | CUDA encoding and scoring required; CPU PDF rendering/storage; ColQwen environment |
| OpenCLIP pages | `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` (`ViT-B-32`, `laion2b_s34b_b79k`) | Individually rendered original PDF pages | One 512-dimensional vector per page; maximum page score per paper | `experiment=openclip_pages` | CUDA by default; explicit CPU/FP32 supported; FAISS search as below; default environment |
| OpenCLIP figures | Same OpenCLIP checkpoint | Individual extracted figure/table images, without added captions | One 512-dimensional vector per image; maximum image score per paper | `experiment=openclip_figures` | CUDA by default; explicit CPU/FP32 supported; FAISS search as below; default environment |
| Qwen text | `Qwen/Qwen3-VL-Embedding-2B` | full original Markdown text | ONE 2048-dimensional embedding per paper; ranks papers directly | `experiment=qwen3_vl_text` | CUDA/BF16 by default; explicit CPU/FP32 supported; FAISS search as below; default environment |
| Qwen figures | Same Qwen checkpoint | all extracted figure/table images together | ONE 2048-dimensional embedding per paper; ranks papers directly | `experiment=qwen3_vl_figures` | CUDA/BF16 by default; explicit CPU/FP32 supported; FAISS search as below; default environment |
| Qwen pages | Same Qwen checkpoint | all rendered PDF pages together, in page order | ONE 2048-dimensional embedding per paper; ranks papers directly | `experiment=qwen3_vl_pages` | CUDA/BF16 by default; explicit CPU/FP32 supported; FAISS search as below; default environment |
| Qwen multimodal | Same Qwen checkpoint | all extracted figure/table images, followed by full original Markdown text | ONE 2048-dimensional embedding per paper; ranks papers directly | `experiment=qwen3_vl_multimodal` | CUDA/BF16 by default; explicit CPU/FP32 supported; FAISS search as below; default environment |

Every Qwen configuration produces **ONE embedding per paper** and ranks papers
directly, without page/image score aggregation or embedding averaging. There is
no Qwen implementation named “images,” and no configuration combining rendered
pages with text. The full input definitions above describe what is supplied to
the encoder; token limits still apply.

The profiles pin these immutable model revisions:

| Model | Revision |
| --- | --- |
| E5 | `ffb93f3bd4047442299a41ebb6fa998a38507c52` |
| ColBERT | `c1e84128e85ef755c096a95bdb06b47793b13acf` |
| ColQwen adapter | `dcbe8d9cede518bce830488364ba0e40c873645b` |
| ColQwen base | `92908120384b7a2110c5beda3ab29cbdb2c08e49` |
| OpenCLIP | `1a25a446712ba5ee05982a381eed697ef9b435cf` |
| Qwen | `9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` |

## Additional supported configurations and unimplemented profiles

These are separate from the core benchmark. Runnable variants use the same
prepare/index/retrieve/evaluate scripts and the same checkpoints and environments
as their parent methods.

| Configuration | Document input | Indexed unit / ranking | Selector or override | Requirements / status |
| --- | --- | --- | --- | --- |
| BM25 title and abstract | Available original `title` and `abstract`, joined with a blank line | One paper document; direct BM25 ranking | `model=bm25 representation=title_abstract` | CPU/Java, default environment; implemented |
| E5 title and abstract | Same original metadata fields | One vector per paper; direct ranking | `model=dense representation=title_abstract` | CUDA encoding, default environment; implemented |
| E5 passages with mean aggregation | Same passages and E5 checkpoint as the core passage method | Mean score across **all** indexed passages per paper | `experiment=dense_passages representation.aggregation.method=mean` | CUDA encoding, default environment; implemented |
| Alternate dense encoders | Selected text representation and explicit model-specific prompts | Same single-vector FAISS pipeline | `model=dense model.model_id=... model.revision=...` with appropriate `model.encoding` prompts | CUDA; supported Sentence Transformers Hub models only; local checkpoint paths unsupported; not a separate benchmark method |
| BM25 passages | Declared passage representation | No BM25 passage execution path | `model=bm25 representation=passages` | **Configuration can compose, execution unimplemented**; passage execution is supported only for dense retrieval |
| Legacy Byaldi/ColPali profile | Declared pages or figures | No executable index/retrieval implementation | `model=visual` or `experiment=visual_pages` | **Configuration-only, not runnable**; declared `vidore/colpali-v1.3` is not an implemented baseline |
| `stage=search`, `stage=export` | Configuration contracts | No executable stage | Corresponding stage selector | **Unimplemented**; use `03_retrieve.sh` for batch search and `01_prepare.sh` for canonical JSONL export |

Title/abstract loading uses only supplied original metadata from `paper.json`
and, when present, `metadata.parquet`. It reports partial fields and rejects a
paper when both are absent under the default strict missing policy. It does not
derive abstracts from Markdown. Device, precision, batch size, split, and metric
overrides are execution settings rather than additional benchmark methods.

## Token limits and document processing

| Method | Existing limit and truncation behavior |
| --- | --- |
| BM25 | No application-level neural context limit or prefix truncation. Full supplied text goes to Terrier's indexing/tokenization pipeline. |
| E5 | Sentence Transformers uses the checkpoint's `max_seq_length`: **512 tokens** for the pinned E5, including prompts and special tokens. Long documents/queries are tokenizer-truncated; fulltext does not imply complete-document encoding. YAML supplies `passage: ` and `query: ` prompts. The effective length is recorded in encoding metadata. |
| E5 passages | LangChain `RecursiveCharacterTextSplitter`: **1000 characters**, **200-character overlap**, with whitespace stripping and configured separators. These are character counts, not tokens. Each passage and each query still obeys E5's token limit. FAISS scores every passage before max/mean paper aggregation. |
| ColBERT | `model.indexing.doc_maxlen=220` WordPiece tokens including CLS, document marker and SEP (about 217 content tokens); retains the text prefix, without passage splitting. `model.retrieval.query_maxlen=32` controls query tokenization. |
| ColQwen | Every original PDF page is rendered at scale 2.0 and encoded independently. The pinned checkpoint's official processor/module owns image resizing, query tokenization and any model length limits; this integration adds no shared Qwen3-VL `model.max_length` setting, document-text truncation or page-count cap. Effective processor, module and model configs are saved in index provenance. |
| OpenCLIP | The code asserts the checkpoint's **77-token** query context, including special tokens. Long queries are truncated by the OpenCLIP tokenizer; truncation counts/proportions are printed and saved. Images use the standard evaluation resize/center-crop to 224 pixels. All pages/images are indexed. |
| Qwen, all four modes | Shared **32768-token** context by default, including image tokens, text, instructions and chat template. The official processor truncates excess text **from the end** at `model.max_length`; `model.truncation_side=right` retains the prefix. Image blocks precede text and **image tokens must fit within the context**; truncation through image tokens is rejected. There is no automatic image/page dropping, paper chunking, summarization or multi-vector aggregation. Queries share the same context limit. |

E5's limit is loaded rather than hardcoded by the application; see its
[pinned checkpoint config](https://huggingface.co/intfloat/e5-small-v2/blob/ffb93f3bd4047442299a41ebb6fa998a38507c52/sentence_bert_config.json).
The standalone dense example offers `--max-seq-length`; the Hydra dense profile
does not expose that override. Qwen allows a positive `model.max_length` up to
32768, with right truncation required. Its official image processor uses
4096–1843200 pixels per image. Supplying every page/image in a single paper input
can exceed context or GPU memory even with a small batch size.

## Install environments

Run commands from the repository root with Python 3.11 and
[uv](https://docs.astral.sh/uv/). Install both locked environments:

```sh
# BM25, E5, ColBERT, OpenCLIP, Qwen, evaluation and test tools.
uv sync --locked

# Separate Transformers 5 / Sentence Transformers 6 environment for ColQwen.
UV_PROJECT_ENVIRONMENT=.venv-colqwen uv sync --locked \
  --no-default-groups --extra colqwen --group dev
```

The default environment pins Transformers 4.57.6 for ColBERT and the visual
single-vector methods. ColQwen pins Transformers 5.15.0, Sentence Transformers
6.1.0 and PEFT 0.19.1. These environments cannot be combined. The stage scripts
automatically select `.venv-colqwen` when passed `experiment=colqwen_pages` or
`model=colqwen`; leave `UV_PROJECT_ENVIRONMENT` unset for ordinary script use.

BM25 uses the bundled `jdk4py` Java runtime; its first run downloads Terrier
JARs. Neural methods download pinned pretrained weights on first use. Linux with
an NVIDIA GPU and a compatible CUDA driver is the intended benchmark setup.
ColBERT additionally needs a CUDA toolkit, compiler and Ninja for native
extensions. Verify PyTorch sees CUDA in each neural environment you use:

```sh
uv run --locked python -c 'import torch; print(torch.cuda.is_available())'
UV_PROJECT_ENVIRONMENT=.venv-colqwen uv run --locked \
  --no-default-groups --extra colqwen --group dev \
  python -c 'import torch; print(torch.cuda.is_available())'
```

## Dataset layout and required files

Set `TOT_DATA_ROOT` to an existing restored dataset root, with canonical paths:

```text
DATA_ROOT/
  preprocessed/<DOMAIN>/<PAPER_ID>/
    paper.json                         # Full extraction identity/source/figure metadata
    markdown.md                        # Original extracted text
    images/<image-file>                 # Extracted figure/table assets
  pdf_datasets/<DOMAIN>/<original-relative-path>.pdf
  metadata.parquet                     # Optional original bibliographic/PDF index
  splits/pdf_dataset_split.json         # Authoritative train/test PDF membership
  query_collections/<DOMAIN>/query_generation_test/visual_and_text/
    queries.jsonl
    metadata.json                      # Retained collection provenance when supplied
```

Use `query_generation_train/` as well when selecting training queries. The
default query mode is `visual-and-text` (directory `visual_and_text`), whose
`query` field is text. Each query supplies `relevant_ids`; qrels are validated
against canonical extraction paper IDs and positives receive grade 1. No separate
input TREC qrels file is required. Clue files and query-generation prompts are
not document inputs.

| Methods | Required document inputs for indexing |
| --- | --- |
| BM25 fulltext, E5 fulltext/passages, ColBERT fulltext, Qwen text | Canonical extraction directories and nonempty original `markdown.md`; no PDFs or image binaries required |
| ColQwen pages, OpenCLIP pages, Qwen pages | Canonical extraction identity directories plus restored original PDFs under `pdf_datasets/`; no Markdown/figure assets required |
| OpenCLIP figures, Qwen figures | Canonical extraction identity directories and extracted figure/table image assets; no PDFs or Markdown required |
| Qwen multimodal | Canonical extraction identity directories, nonempty original Markdown and all referenced extracted figure/table images; no PDFs required |
| Additional BM25/E5 title/abstract | Original title/abstract in `paper.json` and/or `metadata.parquet`; no Markdown/PDF/image binaries required |

Full `paper.json` supplies authoritative IDs, source paths and ordered figure
records with `image_relpath`. Lean staging without `paper.json` is supported:
directory names retain paper IDs, `images/` supplies extracted images in sorted
path order, and source mappings must be resolvable. For pages, absent explicit
source metadata, the adapter removes the domain prefix from the paper ID and
appends `.pdf` within the domain; for example `ACL_2026.findings-acl.999` maps
to `pdf_datasets/ACL/2026.findings-acl.999.pdf`. Explicit source metadata wins.
Page-only directories need a recognized identity marker (`paper.json`,
`markdown.md`, `_SUCCESS`, or an `images/` member); an empty directory is insufficient.

`metadata.parquet` is optional for fulltext/figure methods and page methods with
resolvable source paths. The split JSON is required for split-selected corpora,
including isolated test mode; `split.corpus=all` can index available extractions
without it. Test-query loading uses the corresponding query collection; index
stages do not read queries or qrels. Retrieval uses the saved index and query
files; explicit text index selection avoids rereading document inputs. Evaluation
reads only saved retrieval artifacts and does not reopen data, weights or indexes.

Qwen figures requires at least one supplied figure per paper. Qwen multimodal
encodes text alone for papers without figures; any referenced missing image is
an error for either Qwen image mode. OpenCLIP figures retains papers without
images in the evaluation corpus but gives them no retrievable vector. Missing
data defaults to `dataset.missing=error`; explicit `skip` cannot leave an
unresolved positive judgment. Test mode rejects skipped papers.

The adapter also supports selected domain extraction TARs without unpacking;
original PDF TAR shards must first be restored to the paths above. Restoration
is external to these scripts. See the [dataset contract](docs/dataset-schema.md) for
archive schemas and dated publication evidence. That evidence concerns a
query-only Hub revision, so it is not proof that full corpus inputs are available.

## Roots and one domain per run

```sh
export TOT_DATA_ROOT=/your/restored/data
export TOT_CACHE_ROOT=./cache
export TOT_OUTPUT_ROOT=./results/ACL
export TOT_TEST_ROOT=./test
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `TOT_DATA_ROOT` | `./data` | Read-only dataset inputs |
| `TOT_CACHE_ROOT` | `./cache` | Normal-run dataset/model/embedding caches; script uv cache defaults to `${TOT_CACHE_ROOT}/uv` |
| `TOT_OUTPUT_ROOT` | `./results` | Normal-run prepared/index/retrieval/evaluation artifacts |
| `TOT_TEST_ROOT` | `./test` | Isolated test mode's `cache/` and `results/`; overrides normal application cache/output roots |

The shell scripts change to the repository root, so their relative paths resolve
there. Direct Hydra invocations resolve relative paths from their invocation
directory. Keep data, cache and output roots separate and non-overlapping.

Each run selects **exactly one** of `ACL`, `Biology`, `Engineering`, `Medicine`,
or `Physics`, using the same `dataset.domains=[DOMAIN]` override for every stage.
The commands below select ACL explicitly. For another domain replace `[ACL]`
on all three commands and use an appropriate output root. Repeat runs per domain;
a multi-domain list is rejected.

Normal `split=test` selects **test queries and all available corpus documents**,
including unassigned distractors. There are no document/query count limits in
these stage scripts. It differs from `--test`, described below.

## Separate index, retrieval and evaluation commands for all core methods

Run each selected method's three commands in order after setting the roots above.
Repeat any model/representation/encoding overrides consistently at all stages.
The scripts discover compatible saved artifacts and print the paths selected.

```sh
# 1. BM25 fulltext
./scripts/02_index.sh experiment=bm25_fulltext split=test 'dataset.domains=[ACL]'
./scripts/03_retrieve.sh experiment=bm25_fulltext split=test 'dataset.domains=[ACL]'
./scripts/04_evaluate.sh experiment=bm25_fulltext split=test 'dataset.domains=[ACL]'

# 2. E5 fulltext
./scripts/02_index.sh model=dense representation=fulltext split=test 'dataset.domains=[ACL]'
./scripts/03_retrieve.sh model=dense representation=fulltext split=test 'dataset.domains=[ACL]'
./scripts/04_evaluate.sh model=dense representation=fulltext split=test 'dataset.domains=[ACL]'

# 3. E5 passages
./scripts/02_index.sh experiment=dense_passages split=test 'dataset.domains=[ACL]'
./scripts/03_retrieve.sh experiment=dense_passages split=test 'dataset.domains=[ACL]'
./scripts/04_evaluate.sh experiment=dense_passages split=test 'dataset.domains=[ACL]'

# 4. ColBERT fulltext
./scripts/02_index.sh experiment=colbert_fulltext split=test 'dataset.domains=[ACL]'
./scripts/03_retrieve.sh experiment=colbert_fulltext split=test 'dataset.domains=[ACL]'
./scripts/04_evaluate.sh experiment=colbert_fulltext split=test 'dataset.domains=[ACL]'

# 5. ColQwen pages (scripts automatically select .venv-colqwen)
./scripts/02_index.sh experiment=colqwen_pages split=test 'dataset.domains=[ACL]'
./scripts/03_retrieve.sh experiment=colqwen_pages split=test 'dataset.domains=[ACL]'
./scripts/04_evaluate.sh experiment=colqwen_pages split=test 'dataset.domains=[ACL]'

# 6. OpenCLIP pages
./scripts/02_index.sh experiment=openclip_pages split=test 'dataset.domains=[ACL]'
./scripts/03_retrieve.sh experiment=openclip_pages split=test 'dataset.domains=[ACL]'
./scripts/04_evaluate.sh experiment=openclip_pages split=test 'dataset.domains=[ACL]'

# 7. OpenCLIP figures
./scripts/02_index.sh experiment=openclip_figures split=test 'dataset.domains=[ACL]'
./scripts/03_retrieve.sh experiment=openclip_figures split=test 'dataset.domains=[ACL]'
./scripts/04_evaluate.sh experiment=openclip_figures split=test 'dataset.domains=[ACL]'

# 8. Qwen text
./scripts/02_index.sh experiment=qwen3_vl_text split=test 'dataset.domains=[ACL]'
./scripts/03_retrieve.sh experiment=qwen3_vl_text split=test 'dataset.domains=[ACL]'
./scripts/04_evaluate.sh experiment=qwen3_vl_text split=test 'dataset.domains=[ACL]'

# 9. Qwen figures
./scripts/02_index.sh experiment=qwen3_vl_figures split=test 'dataset.domains=[ACL]'
./scripts/03_retrieve.sh experiment=qwen3_vl_figures split=test 'dataset.domains=[ACL]'
./scripts/04_evaluate.sh experiment=qwen3_vl_figures split=test 'dataset.domains=[ACL]'

# 10. Qwen pages
./scripts/02_index.sh experiment=qwen3_vl_pages split=test 'dataset.domains=[ACL]'
./scripts/03_retrieve.sh experiment=qwen3_vl_pages split=test 'dataset.domains=[ACL]'
./scripts/04_evaluate.sh experiment=qwen3_vl_pages split=test 'dataset.domains=[ACL]'

# 11. Qwen multimodal
./scripts/02_index.sh experiment=qwen3_vl_multimodal split=test 'dataset.domains=[ACL]'
./scripts/03_retrieve.sh experiment=qwen3_vl_multimodal split=test 'dataset.domains=[ACL]'
./scripts/04_evaluate.sh experiment=qwen3_vl_multimodal split=test 'dataset.domains=[ACL]'
```

Preparation is optional for every method. Use the same selector, split and domain
with `01_prepare.sh`, for example:

```sh
./scripts/01_prepare.sh experiment=qwen3_vl_multimodal split=test 'dataset.domains=[ACL]'
```

It exports canonical corpus, queries and qrels with provenance; page/figure
profiles add unit mappings and dense passages add passage records. It does not
build an index, precompute neural embeddings or restore missing datasets. Indexing
reads the original inputs directly rather than consuming prepared JSONL.

## Metrics, reuse and output locations

The default paper cutoff is `retrieval.top_k=100`. Default effectiveness metrics
include **RR@10, R@1, R@10, R@100**; the current
`config/evaluation/default.yaml` also enables **nDCG@10**. Aggregate RR@10 is
mean reciprocal rank at 10; R is recall. To report exactly the four core metrics,
append this override to **any** `04_evaluate.sh` command:

```sh
./scripts/04_evaluate.sh experiment=bm25_fulltext split=test 'dataset.domains=[ACL]' \
  'evaluation.metrics=[RR@10,R@1,R@10,R@100]'
```

Evaluation can be rerun with different metrics without retrieving again; metric
cutoffs cannot exceed the saved retrieval depth. Use explicit completed artifact
paths when automatic selection is ambiguous or when text source files are no
longer available:

```sh
# Set INDEX and RUN to the completed directories printed by previous stages.
INDEX=/absolute/path/to/completed/index
RUN=/absolute/path/to/completed/retrieval
./scripts/03_retrieve.sh experiment=bm25_fulltext split=test 'dataset.domains=[ACL]' "paths.index=$INDEX"
./scripts/04_evaluate.sh experiment=bm25_fulltext split=test 'dataset.domains=[ACL]' "paths.run=$RUN"
```

Substitute the matching method selector for other methods. Automatic text-index
selection rereads the selected corpus to match identity; visual index selection
uses saved manifests without opening original images/PDFs. Multiple compatible
visual indexes or retrieval runs require an explicit path.

Under `TOT_OUTPUT_ROOT`:

| Directory | Contents |
| --- | --- |
| `prepared/<identity>/` | Optional `corpus.jsonl`, `queries.jsonl`, `qrels.jsonl`, sidecars and applicable unit/passage mappings |
| `indexes/<identity>/` | `index-manifest.json`, backend files and paper/unit mappings: Terrier files, FAISS `index.faiss`, ColBERT `colbert/`, or ColQwen `qdrant/` |
| `models/<identity>/` | Retrieval `run.trec`, `topics.jsonl`, `qrels.tsv`, `query-ids.jsonl`, `corpus-ids.jsonl`; ColQwen also saves every page score in `page-scores.jsonl` |
| `evaluations/<identity>/` | Aggregate `metrics.csv` and individual-query `per-query.csv` |

New artifacts also contain `.hydra/{config,hydra,overrides}.yaml` and
`manifest.json` for resolved settings, identity and provenance. The printed
completed path is authoritative. Index names include model, representation,
domain and corpus split; visual indexes also include a content hash. Retrieval
and evaluation names include content identities. Weights/embeddings remain in
the cache hierarchy rather than `models/` retrieval outputs.

Repeated completed retrieval/evaluation/preparation jobs verify and reuse
identical artifacts. BM25/E5 index generation skips an existing destination,
including unfinished destinations; skipping does not establish index validity.
Use a fresh `paths.index` for a different build and pass it to retrieval.
ColBERT/visual indexes validate identity before completed reuse. ColQwen can
resume a validated partial page prefix by rerunning its index command; see
[ColQwen recovery](docs/colqwen-retrieval.md#resuming-an-interrupted-index).
Other interrupted builds are not generally resumable. Existing corrupt or
incompatible artifacts fail validation rather than being overwritten. See the
[configuration guide](docs/configuration.md) for precise stage contracts.

## GPU encoding and FAISS search

E5 and ColBERT require CUDA for indexing/retrieval; E5 checks CUDA even with
cached embeddings. ColQwen requires CUDA for encoding and exact MaxSim scoring.
BM25 uses CPU/Java. OpenCLIP and all Qwen profiles default to CUDA but explicitly
support `launcher=local_cpu model.precision=fp32`; repeat these settings across
stages. CPU fixture runs do not establish full-corpus performance.
Qwen defaults to BF16 with SDPA; E5 and OpenCLIP default to FP32 and also accept
FP16/BF16. Use precision supported by the GPU and reduce `model.batch_size` if
needed; all pages/images of one Qwen paper must still fit a single input.

E5, OpenCLIP and Qwen persist CPU-format exact FAISS `IndexFlatIP` indexes.
Encoding on CUDA does not imply FAISS runs on CUDA: **the lock installs
`faiss-cpu`**, so default search runs on CPU. `retrieval.search_device=auto`
selects CUDA only with an installed GPU-enabled FAISS build and a visible GPU;
`cpu` forces CPU and `cuda` fails when GPU FAISS is unavailable. A separately
provisioned compatible GPU FAISS environment can use `UV_NO_SYNC=1` to prevent
stage scripts from replacing its dependencies.

When supported, FAISS clones the index to GPU 0 in FP32. Search depths **over
2048** still use the CPU index, even with `search_device=cuda`. This includes
exhaustive E5 passage search and expanded OpenCLIP page/image searches for enough
unique papers and complete cutoff ties. Deep passage search avoids an unused
GPU clone; expanded visual searches can switch to CPU mid-query. Retrieval
manifests report CPU/GPU query-search counts. ColBERT's upstream index/search and
ColQwen's persisted Qdrant vectors with streamed CUDA MaxSim use their own paths;
the shared FAISS selector does not control them.

## Progress logs and resource reports

Stages print the selected sources/counts, artifact paths, cache hits and completion
status. Named phases cover source verification, execution, output verification
and sealing. Progress bars show encoding/indexing/rendering/search work where
available; opaque calls print a heartbeat every 30 seconds. Hydra job file
logging is disabled. Capture console output yourself when a durable log is useful:

```sh
mkdir -p logs
# In bash/zsh, propagate a failed stage through tee.
set -o pipefail
./scripts/02_index.sh experiment=qwen3_vl_pages split=test 'dataset.domains=[ACL]' \
  2>&1 | tee logs/qwen3_vl_pages-ACL-index.log
```

Each new stage's `manifest.json` has a `performance` section with elapsed/process
CPU time, phase timings, throughput, output/index/cache bytes, Python-process peak
RAM, hardware and CUDA details, PyTorch memory peaks, and optional device-wide
`nvidia-smi` samples. Timing covers the stage and verification/I/O after snapshots,
excluding CLI discovery and environment setup. RAM excludes JVM/child processes;
PyTorch peaks exclude FAISS/native allocations; device-wide GPU samples include
other processes. Cache encoding time can be historical; current execution time
is in `performance`. Reused artifacts retain their original reports. These are
scoped resource measurements, not complete hardware utilization benchmarks.

## Isolated official test-split runs

Add `--test` to **each** stage command above to select only official test papers
and test queries, while retaining all pages/passages/figures for those papers.
This uses `TOT_TEST_ROOT/results/` and `TOT_TEST_ROOT/cache/`, independent of
normal application outputs/caches. `TOT_DATA_ROOT` still supplies the original
inputs. It requires the full official test split, rejects skipped papers and
conflicting split overrides, and prevents consuming artifacts across modes.

The existing convenience script chains the same three stages for any core
selector, for example:

```sh
./scripts/00_test.sh --test experiment=qwen3_vl_figures 'dataset.domains=[ACL]'
```

Direct Hydra uses `mode=test` instead of the shell `--test` flag. GPU requirements
still apply. To validate a stage's composed settings without data/model access,
append `dry_run=true` to any stage command.

## Dataset adapter quickstart

```sh
# Fully offline two-document synthetic example after installation:
uv run python examples/convert_dataset.py \
  --local-dir tests/fixtures/adapter --domain ACL --output /tmp/scientific-tot-example
```

From Python, use a shared source to keep all reads on one revision:

```python
from scientific_tot import Dataset, load_corpus, load_queries, load_qrels

source = Dataset(local_dir="tests/fixtures/adapter")  # replace with your data root
corpus = load_corpus(dataset=source, domain="ACL", split="all", representation="fulltext")
queries = load_queries(dataset=source, domain="ACL", split="test", mode="visual-and-text")
qrels = load_qrels(dataset=source, domain="ACL", split="test", corpus=corpus)

corpus.write_jsonl("/tmp/corpus.jsonl")
queries.write_jsonl("/tmp/queries.jsonl")
qrels.write_jsonl("/tmp/qrels.jsonl")
print(corpus.report.to_dict())
```

`missing="error"` is the default; `missing="skip"` explicitly
excludes unusable records with a report. Unresolved relevance targets always
raise. Figure metadata and file/archive paths are exposed without loading images.

[API documentation](docs/api.md) covers component downloads, metadata-only reads,
Hub revisions, archive granularity, lean extractions, missing-data policy, and
record schemas. [Architecture](docs/architecture.md) records design decisions.

## Full document collection (default)

Both retrieval scripts default to **every available extraction in the selected
domain** (`--corpus-split all`), fulltext, and all test queries. There is no
20,000-document cap: the recorded ACL collection has 20,845 usable extractions.
Train/test corpus splits cover only assigned papers and would omit distractors.
Use your restored data root:

```sh
uv run python examples/pyterrier_bm25.py \
  --local-dir ../multimodal-tip-of-tongue-for-scientific-papers/data \
  --domain ACL --output results/acl-full-bm25
uv run python examples/dense_retrieval.py \
  --local-dir ../multimodal-tip-of-tongue-for-scientific-papers/data \
  --domain ACL --cache results/dense-cache --output results/acl-full-dense
```

Select one domain with `--domain ACL`, `--domain Biology`, `--domain Engineering`,
`--domain Medicine`, or `--domain Physics`. Small demos are explicit: add
`--documents 250 --queries 10`.
Omitting these limits, or setting either to `0`, uses all selected records.
`--query-split test` is the default; use `train+test` to evaluate both query splits.
Missing content still raises unless `--missing skip` is explicitly selected.

## PyTerrier BM25 fixture smoke test

```sh
# After the shared setup above; choose a fresh output path:
uv run python examples/pyterrier_bm25.py \
  --local-dir tests/fixtures/adapter --domain ACL \
  --documents 2 --queries 1 \
  --output /tmp/tot-pyterrier-demo
```

[PyTerrier guide](docs/pyterrier.md) provides the Python conversion API, real-data
subset/full-corpus commands, Java setup, and output conventions.
[Notebook](examples/pyterrier_bm25.ipynb) defaults to full-collection fulltext retrieval
and indexes full document text.
[Actual-data results](docs/pyterrier-results.md) record execution on dgx03.

## Pretrained dense retrieval

```sh
uv run python examples/dense_retrieval.py \
  --local-dir tests/fixtures/adapter --documents 2 --queries 1 \
  --cache results/dense-cache --output results/dense-example
```

[Dense retrieval tutorial](docs/dense-retrieval.md) covers configurable model IDs,
pinned revisions, model-specific prompts, CUDA setup and persistent library
caches. [Notebook](examples/dense_retrieval.ipynb) demonstrates encoding, FAISS
search, shared metrics and cache reuse. [Measured subset results](docs/dense-results.md)
verify the same canonical records as the BM25 baseline. The first dense run
downloads pretrained weights; subsequent runs reuse the cache.
No fine-tuning is performed.

Dense embedding generation requires CUDA. Both the example script and Hydra
`model=dense` (including `experiment=dense_passages`) select CUDA by default.
If no NVIDIA GPU is available to PyTorch, execution raises an error instead of
falling back to CPU, even when embeddings are cached. Hydra FAISS retrieval selects
CUDA when the installed FAISS build exposes a GPU, and reports CPU execution otherwise.
Saved-run evaluation remains on CPU and needs no GPU.

Check GPU availability after syncing the updated environment:

```sh
uv run python -c "import torch; print(torch.cuda.is_available())"
```

For a small comparison, these commands select 250 documents and ten test queries,
keeping all their relevant documents and using the same subset for both baselines.
They perform indexing, retrieval, and evaluation automatically; use fresh output
paths for reruns and reuse the dense cache:

```sh
export TOT_DATA_ROOT=/your/restored/data
uv run python examples/pyterrier_bm25.py \
  --local-dir "$TOT_DATA_ROOT" --domain ACL --documents 250 --queries 10 \
  --output results/small-acl-bm25
uv run python examples/dense_retrieval.py \
  --local-dir "$TOT_DATA_ROOT" --domain ACL --documents 250 --queries 10 \
  --baseline-dir results/small-acl-bm25/fulltext \
  --cache cache/dense-small --output results/small-acl-dense
```

The Hydra stage scripts currently use the full selected corpus; subset limits
are available in the example scripts above.

## Notebooks in the same environment

```sh
uv run python -m ipykernel install --sys-prefix --name scientific-tot --display-name "Scientific ToT"
uv run jupyter lab
```

Select **Scientific ToT** in both notebooks. They default to restored data at
`../multimodal-tip-of-tongue-for-scientific-papers/data`, all documents in `ACL`,
all test queries, and fulltext, with fresh output directories. Set `TOT_DATA_ROOT`
if your restored data lives elsewhere. Set `DOCUMENTS, QUERIES = 250, 10` for an
explicit demo; `None, None` (the default) leaves both unlimited. For a fixture
smoke test, launch with `TOT_DATA_ROOT=tests/fixtures/adapter uv run jupyter lab`.
Configure the domain and optional subset limits in the configuration cell.
BM25 selects the locked `jdk4py` runtime automatically, including on machines
whose system Java is too old. Its first run downloads Terrier's JAR.

The retrieval examples use original full document text only. Title/abstract
metadata is not required. `--missing skip` explicitly excludes records with
missing fulltext; all query positives must still resolve. CLI options use
`--missing skip`; `missing="skip"` is Python API syntax.

An existing experiment output is rejected to prevent stale indexes. Choose a new
`--output` for a rerun; reuse the dense `--cache` directory.

## Discovery and Hub use

```python
from scientific_tot import Dataset, DatasetAdapter

source = Dataset(revision="3b764a4c6855925ec802a2560712c4f5cc96f980")
print(source.discover())  # filenames/sizes only, no artifact downloads
adapter = DatasetAdapter(source)
queries = adapter.load_queries(domain="ACL", split="test")
# Optional explicit transfer of selected components:
adapter.download_components(domain="ACL", split="test", components=["queries"])
```

Omitting `revision` pins main on the first discovery. Explicit revisions must be
full commit SHAs. Local mode never accesses the network. Hub downloads use the
revision cache incrementally and select exact files. Domain TARs are the smallest
archive download unit; split filtering occurs after archive indexing. Corpus operations do not fetch separate PDF or clue components; a full-format
preprocessed TAR may itself contain bundled PDFs/images. Already-restored data from upstream
restoration utilities works directly; TARs can also be read without extraction.

Query IDs remain canonical and collection-scoped. The additional `query_key`
provides a reversible unique key across domains and appears in both queries and
qrels. Corpus IDs remain extraction paper IDs; packaged PDF IDs are separate.
Original provenance/license metadata is preserved. Generated query clues never
enter corpus text.

## Development

```sh
uv sync --locked
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
# Optional live check: metadata and one ~7 MB ACL test query file; no archives:
uv run pytest -m live
```

Default tests block network connections and use small synthetic fixtures.
[Verification](docs/verification.md) records executed checks and limitations.
[Integration plan](docs/integration-plan.md) tracks remaining work. The default
project environment includes all example dependencies and CUDA-enabled
Torch on Linux. Dense encoding requires an NVIDIA GPU with a compatible driver.
A model download is required only when running pretrained encoding.
Library consumers can still install the small core wheel and choose package extras.
ColQwen has a separate locked environment; textual workflows retain the default one. Code is MIT licensed;
dataset artifacts retain their own licenses.
