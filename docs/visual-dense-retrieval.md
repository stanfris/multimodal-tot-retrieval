# OpenCLIP and Qwen3-VL-Embedding retrieval

Six zero-shot profiles share Hydra stages, FAISS persistence, embedding caches,
the TREC writer and saved-run evaluation. All use the original textual ToT queries
and the selected paper collection independently for each scientific domain.

| Model profile / experiment | Document input | Indexed unit | Dimensions |
| --- | --- | --- | --- |
| `openclip` / `openclip_pages` | Rendered original PDF page | Page | 512 |
| `openclip_figures` | Individual supplied figure/table image | Image | 512 |
| `qwen3_vl_pages` | All rendered PDF pages together, in page order | Paper | 2048 |
| `qwen3_vl_text` | Full original Markdown text | Paper | 2048 |
| `qwen3_vl_figures` | All extracted figure/table images together | Paper | 2048 |
| `qwen3_vl_multimodal` | All extracted figure/table images, then full Markdown text, in one joint input | Paper | 2048 |

## Models and inputs

OpenCLIP uses official `open_clip` loading, `ViT-B-32`, `laion2b_s34b_b79k` weights,
and the checkpoint `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` at
`1a25a446712ba5ee05982a381eed697ef9b435cf`. Its standard evaluation transforms
resize/center-crop to 224 pixels and normalize RGB inputs. The standard 77-token
tokenizer truncates long queries; counts and proportions are printed and saved.
See [OpenCLIP](https://github.com/mlfoundations/open_clip).
OpenCLIP pages uses the existing PDFium renderer at scale 2.0; figures uses actual
supplied image assets without added captions. Each image/page receives a vector,
and the highest image/page score determines the paper score.

Qwen profiles share `Qwen/Qwen3-VL-Embedding-2B` at
`9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda`. The pinned checkpoint's official
`Qwen3VLEmbedder` provides image blocks, chat templates, preprocessing, forward
inference and last-token pooling. Multiple image blocks are put in the same user
message in source order (PDF page order for the pages profile). In multimodal
mode the complete Markdown text follows the last extracted figure image. One forward input and one normalized embedding represent each
paper; images are never independently embedded and averaged.
See the [official implementation](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/main/scripts/qwen3_vl_embedding.py).

The four Qwen modes have distinct document inputs:

- `text` reads the full original `markdown.md`, exactly as for BM25/E5 full text.
- `figures` reads every supplied extracted figure/table image, including lean
  `images/` staging and restored/TAR artifacts. No text is added.
- `pages` renders every original PDF page with the existing PDFium renderer
  (`representation.rendering.scale=2.0`) and supplies all pages together, in order.
  No extracted text or figures are added.
- `multimodal` uses the same extracted images as `figures`, followed by the same
  full original Markdown as `text`, in one joint input. It does not render PDFs.

Only pages needs original PDFs. Only text and multimodal need Markdown. No mode
combines rendered pages with text, and no page-text alignment, content-list JSON,
chunking or summarization is required. Source image/PDF hashes, page counts,
rendering settings and full text enter cache/index identity as applicable.
Figure-only Qwen requires at least one supplied figure per paper; otherwise it
fails explicitly. Multimodal papers without figures encode their full text.
Missing referenced Qwen figures fail rather than silently dropping image inputs.
Missing/corrupt PDFs follow the shared PDF loader's explicit missing policy;
test mode requires the complete test split without skipped papers.

The OpenCLIP figures missing policy is unchanged: papers without supplied images
stay in evaluation but receive no vector, and explicit `dataset.missing=skip`
can exclude unusable images.

All Qwen queries use `Retrieve images or text relevant to the user's query.`
Original query strings are supplied without rewriting; their embeddings are
shared across the four profiles.
The shared context budget is 32768 tokens, including image tokens, text and the
chat template. All available input is supplied to the official processor, which
runs once with right-side truncation at `model.max_length`. Images precede text,
so the remaining budget holds the beginning of the text and any excess suffix
is discarded. The same prefix truncation applies to long text-only documents
and queries. The configured limit and truncation side are recorded in encoding
metadata and cache/index identity. The processed tensors are reused for inference.
Official image resizing uses 4096–1843200 pixels per image. Image tokens must fit
within the context; the Transformers processor rejects truncation through image
tokens rather than creating mismatched image features.

Embeddings are L2-normalized in float32 and inserted incrementally into exact
`IndexFlatIP`. GPU encoding uses inference mode and SDPA for Qwen. A single CPU
prefetch worker decodes the next batch while the GPU handles the current one;
all images are closed after use, including when decoding/encoding fails.
Only a bounded number of paper batches are decoded at once.

## Commands

Install with `uv sync --locked`. New direct dependencies are `open-clip-torch`
(locked to 3.3.0) and `qwen-vl-utils` (0.0.14). Their dependencies include `timm`,
`ftfy`, and `av`. They are included in the existing default environment; a `visual`
extra is available for library consumers. The ColQwen Transformers 5 environment
remains separate.

The following executes indexing, retrieval, and evaluation for all six
configurations independently per domain. Each experiment selects the existing
MRR@10 (`RR@10`), Recall@1/10/100 and NDCG@10 metrics. The test query split
retains all available corpus documents as distractors.

```sh
export TOT_DATA_ROOT=/your/restored/data

for domain in ACL Biology Engineering Medicine Physics; do
  export TOT_OUTPUT_ROOT="./results/$domain"

  for experiment in openclip_pages openclip_figures qwen3_vl_text qwen3_vl_figures qwen3_vl_pages qwen3_vl_multimodal; do
    ./scripts/02_index.sh "experiment=$experiment" split=test "dataset.domains=[$domain]"
    ./scripts/03_retrieve.sh "experiment=$experiment" split=test "dataset.domains=[$domain]"
    ./scripts/04_evaluate.sh "experiment=$experiment" split=test "dataset.domains=[$domain]"
  done
done
unset TOT_OUTPUT_ROOT
```

For one domain/configuration, for example:

```sh
./scripts/02_index.sh experiment=qwen3_vl_multimodal split=test 'dataset.domains=[ACL]'
./scripts/03_retrieve.sh experiment=qwen3_vl_multimodal split=test 'dataset.domains=[ACL]'
./scripts/04_evaluate.sh experiment=qwen3_vl_multimodal split=test 'dataset.domains=[ACL]'
```

Replace the experiment value with `openclip_pages`, `openclip_figures`, `qwen3_vl_text`, or
`qwen3_vl_pages` or `qwen3_vl_figures` to run the others. Preparation is optional, using
`./scripts/01_prepare.sh` with the same experiment and domain overrides. Direct
`model=...` overrides also choose the appropriate representation, while the
experiment profiles additionally select the benchmark metrics.

CUDA is the default. OpenCLIP uses `model.precision=fp32`; Qwen uses `bf16` with
Transformers SDPA. Supported precision settings are `fp32`, `fp16`, and `bf16`;
choose one supported by the GPU. CPU smoke runs explicitly use
`launcher=local_cpu model.precision=fp32`. No device fallback occurs.
Adjust `model.batch_size` consistently across index/retrieve/evaluate commands to
fit memory. Qwen also exposes `model.max_length` and `model.query_instruction`;
`model.truncation_side=right` keeps the input prefix.
All Qwen profiles inherit their shared settings from `config/model/qwen3_vl_text.yaml`.

Indexes live under the existing `indexes/`, retrieval artifacts under `models/`,
and metrics under `evaluations/`. Distinct model names, strategies and content
hashes prevent index collisions. `unit-mapping.jsonl` records each vector's paper
and its source PDF/page count or extracted image paths/hashes and full text for
Qwen, or its page/image source for OpenCLIP.
Rebuild Qwen indexes with the matching mode; document inputs and artifact identity
separate text, figures, pages and multimodal outputs.
Weights and embeddings stay under shared `cache/models/` and
`cache/embeddings/`. Completed caches/indexes reload without document encoding.
Interrupted encoding is not resumable, matching the existing dense workflow.
Scripts select compatible completed indexes without reopening PDFs or image
assets; multiple compatible copies require an explicit `paths.index`.

## Search and efficiency

Qwen searches exactly one vector per paper and ranks papers directly; aggregation
is recorded as `null`. OpenCLIP starts at the requested paper cutoff and expands
its page/image search depth until enough unique papers are found, including
complete score ties at the cutoff, then takes maximum scores per paper.
`retrieval.top_k=100` returns up to 100 retrievable papers in both cases.

The index manifest records indexed units, retrievable papers, papers without
units, FAISS file size and original embedding generation time. Retrieval manifests
record search seconds and seconds per query. Qwen document encoding also reports
paper-level embeddings, input mode, rendered-page, supplied-figure and full-text counts.
These resource measurements remain separate from effectiveness metrics; they are not a
full-corpus performance benchmark. Exact FAISS still computes inner products
against all vectors, and the flat index resides in RAM. Unlike ColQwen, each unit
has only one vector: 2048 bytes for OpenCLIP or 8192 bytes for Qwen, before mapping
and file overhead. The cache also retains the NumPy vectors separately.

## Validation

```sh
uv run pytest tests/test_visual_dense.py
# Opt-in checkpoint downloads and real index/retrieve/evaluate stages:
uv run pytest -m visual_live tests/test_visual_dense.py
```

Offline tests exercise PDFium rendering, multi-image papers, original full text,
FAISS persistence, complete Hydra stages, saved-run evaluation and cache reuse
with mocked neural inference. They verify one Qwen vector per paper, all-images-
before-text ordering, image retention, token-limit truncation, PDF identity and missing-PDF behavior,
separate document indexes, shared queries and isolated test-mode execution.

Some macOS PyTorch/FAISS wheels bundle different OpenMP runtimes. The local test
process can select one shared runtime without changing application code:

```sh
DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.11/site-packages/torch/lib" \
  uv run pytest tests/test_visual_dense.py
```

The intended execution setup is CUDA/Linux. CPU fixture results do not establish
GPU compatibility or full-corpus throughput.

Executed validation for the four-mode configuration (2026-10-22):

- Full offline suite: **524 passed, 8 skipped, 14 deselected**.
- Real pinned checkpoint on CPU float32: **5 passed**, covering all four
  index/retrieve/evaluate pipelines and prefix truncation at 256 tokens.
- Each Qwen live index held **2 vectors for 2 papers**. Pages supplied all
  4 rendered pages; figures and multimodal supplied all 4 extracted images.
  Text and multimodal supplied both full texts, and queries shared their cache.
- All four isolated 110-paper Qwen pipelines passed. Each had one vector per
  paper; the pages fixture included a positive paper with two pages.
- All **32** stage-script dry runs passed (four profiles, four stages, normal and
  isolated test modes). Ruff lint/format, lock and Git whitespace checks passed.

CUDA/bfloat16 throughput has not been measured locally.

### Implementation files

The OpenCLIP figures extension adds `openclip_figures` model/experiment profiles.
Its follow-up validation passed **201 targeted tests**, **two real OpenCLIP
page/figure smoke tests**, and all four stage-script dry runs. It shares page
query embeddings, preserves independent figure indexes, and needs no new dependency.

- `integrations/visual_dense.py`: shared official encoders, representation inputs,
  batching, normalization and embedding cache integration.
- `integrations/dense_batch.py`: existing FAISS persistence/retrieval extended with
  adaptive document-level search and separate resource measurements.
- `adapter.py`, `data/artifacts.py`, `config_arguments.py`: image corpus selection
  and safe binary access to existing restored/TAR artifacts.
- `configuration.py`, `artifact_identity.py`, `config_artifacts.py`, `hydra_run.py`,
  `stage_cli.py`, `stages.py`: profile validation, provenance and stage integration.
- Six `config/model/` profiles and six matching `config/experiment/` profiles;
  `config/representation/figures.yaml` now has execution support.
- `scripts/01_prepare.sh` through `04_evaluate.sh`: portable empty-array expansion
  for macOS Bash, preserving existing arguments and ColQwen environment selection.
- `pyproject.toml`, `uv.lock`, `tests/test_visual_dense.py`, `tests/conftest.py`,
  README and configuration/API documentation: dependencies, tests and usage.
