# OpenCLIP and Qwen3-VL-Embedding retrieval

Five zero-shot configurations share the existing Hydra stages, dense FAISS
persistence, NumPy embedding cache, maximum-score paper aggregation, TREC writer,
and saved-run evaluator. All use the original textual ToT queries and the full
selected document collection, independently for each scientific domain.

| Model profile / experiment | Document input | Indexed unit | Dimensions |
| --- | --- | --- | --- |
| `openclip` / `openclip_pages` | Rendered original PDF page | Page | 512 |
| `openclip_figures` | Individual supplied MinerU images, including table images | Visual element | 512 |
| `qwen3_vl_page` | Rendered original PDF page | Page | 2048 |
| `qwen3_vl_multimodal` | Rendered page + that page's extracted text | Page-text pair | 2048 |
| `qwen3_vl_figures` | Individual supplied MinerU images, including table images | Visual element | 2048 |

## Models and preprocessing

OpenCLIP uses official `open_clip` model loading, `ViT-B-32` architecture,
`laion2b_s34b_b79k` weights, evaluation image transforms, and tokenizer. The pinned
checkpoint is `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` at
`1a25a446712ba5ee05982a381eed697ef9b435cf`. Its evaluation transform uses a
224-pixel bicubic resize/center crop, RGB conversion and the pretrained mean/std.
Its tokenizer retains the standard 77-token context and native truncation. The
truncated-query count and proportion are printed and recorded in query encoding
metadata, without changing ranking or metric schemas. See
[OpenCLIP](https://github.com/mlfoundations/open_clip).

OpenCLIP figures uses the same encoder, tokenizer and pretrained preprocessing as
OpenCLIP pages, with one embedding per supplied visual element and no captions.
Its page and figure profiles share query embeddings while keeping separate indexes.

All three Qwen configurations load `Qwen/Qwen3-VL-Embedding-2B` at
`9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda` using the checkpoint's bundled
`scripts/qwen3_vl_embedding.py:Qwen3VLEmbedder.process()` implementation. Application
code neither constructs a generative model nor pools hidden states. The official
processor, chat template, image resizing (4096–1843200 pixels), default document
instruction, and normalized embedding API own inference. Each multimodal document
input is a single `{"image": page_image, "text": page_text}` object.
See the [official Qwen model card](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B).

The query instruction is the same for all Qwen configurations:
`Retrieve images or text relevant to the user's query.` The wrapper's context
is set to 32768 tokens. An official-processor preflight checks the shared budget,
including expanded image tokens and the chat template, before inference. Inputs
exceeding it raise explicitly rather than silently losing text or images. This
adds a CPU preprocessing pass; model inference and the CPU-to-GPU input transfer
occur once. Query strings are passed intact, without rewriting or summarization.
Identical Qwen query inputs/settings share one content-addressed embedding cache
across all three document strategies.

ColQwen's existing PDFium renderer creates pages at `representation.rendering.scale=2.0`.
Page-only configurations supply no extracted text alongside the image. The
figures configurations open the supplied image assets, including authoritative
MinerU figure/table entries or the image files retained in lean staging. They
supply no captions or surrounding text to the model and do not render PDFs.
Source figure IDs are preserved; when lean staging has no figure IDs, internal
vector IDs use the original image path without fabricating authoritative IDs.
Both restored files and existing uncompressed TAR shards work.

All embeddings are L2-normalized in float32 before insertion/search. FAISS stores
one float32 vector per indexed unit in an exact CPU `IndexFlatIP`. GPU inference
runs with gradients disabled. Pages/images are decoded in bounded batches and
closed after each batch. NumPy memmaps stage vectors and FAISS receives them
incrementally, so images are never collected for the full corpus in RAM.

## Page-level MinerU alignment

The multimodal profile reads original `content_list` JSON referenced by
`paper.json.structured_outputs`, or discovers that export within the same
extraction directory. It prefers flat v1 when both v1 and v2 exports exist:

- Flat blocks must carry a valid zero-based `page_idx`.
- Nested v2 page arrays use their outer page position, and any explicit
  `page_idx` must agree.
- Original text, equations, lists, captions and table/code text are retained in
  source order, with deterministic separators. Image paths and geometry are
  excluded. Text is never taken from query clues or rewritten.
- A page with no extracted text receives an empty text string alongside its
  rendered image. Missing-text counts are reported.
- Missing content lists, ambiguous exports, unsupported text-bearing schemas,
  missing flat block page indices and out-of-range/conflicting page indices fail
  explicitly. The full document Markdown is never repeated for each page.

**Lean `markdown.md` + `images/` staging does not preserve page-level text
associations.** Restore the corresponding original MinerU structured output to
run `qwen3_vl_multimodal`. There is no reliable deterministic alignment to infer
from Markdown alone. An unalignable document is not silently encoded as image-only.

Documents without extracted visual elements remain in the figures baseline's
saved corpus and relevance judgments but receive no artificial vector. They
cannot be retrieved and their queries remain in evaluation. Even a collection
with zero eligible images produces an empty index/run that the existing evaluator
scores normally. Missing/corrupt referenced images fail under `dataset.missing=error`;
with explicit `skip`, unusable images are reported and excluded while their source
papers remain in the evaluation collection.

## Commands

Install with `uv sync --locked`. New direct dependencies are `open-clip-torch`
(locked to 3.3.0) and `qwen-vl-utils` (0.0.14). Their dependencies include `timm`,
`ftfy`, and `av`. They are included in the existing default environment; a `visual`
extra is available for library consumers. The ColQwen Transformers 5 environment
remains separate.

The following executes indexing, retrieval, and evaluation for all five
configurations independently per domain. Each experiment selects the existing
MRR@10 (`RR@10`), Recall@1, Recall@10 and Recall@100 metrics. The test query split
retains all available corpus documents as distractors.

```sh
export TOT_DATA_ROOT=/your/restored/data

for domain in ACL Biology Engineering Medicine Physics; do
  export TOT_OUTPUT_ROOT="./results/$domain"

  for experiment in openclip_pages openclip_figures qwen3_vl_page qwen3_vl_multimodal qwen3_vl_figures; do
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

Replace the experiment value with `openclip_pages`, `openclip_figures`, `qwen3_vl_page`, or
`qwen3_vl_figures` to run the others. Preparation is optional, using
`./scripts/01_prepare.sh` with the same experiment and domain overrides. Direct
`model=...` overrides also choose the appropriate representation, while the
experiment profiles additionally select the benchmark metrics.

CUDA is the default. OpenCLIP uses `model.precision=fp32`; Qwen uses `bf16` with
Transformers SDPA. Supported precision settings are `fp32`, `fp16`, and `bf16`;
choose one supported by the GPU. CPU smoke runs explicitly use
`launcher=local_cpu model.precision=fp32`. No device fallback occurs.
Adjust `model.batch_size` consistently across index/retrieve/evaluate commands to
fit memory. Qwen also exposes `model.max_length` and `model.query_instruction`;
all Qwen profiles inherit their shared settings from `config/model/qwen3_vl_page.yaml`.

Indexes live under the existing `indexes/`, retrieval artifacts under `models/`,
and metrics under `evaluations/`. Distinct model names, strategies and content
hashes prevent index collisions. `unit-mapping.jsonl` records the page/image source
of each vector. Weights and embeddings stay under shared `cache/models/` and
`cache/embeddings/`. Completed caches/indexes reload without document encoding.
Interrupted encoding is not resumable, matching the existing dense workflow.
Scripts select compatible completed indexes without reopening PDFs or image
assets; multiple compatible copies require an explicit `paths.index`.

## Search and efficiency

FAISS exact inner-product search starts at the requested document cutoff and
expands its page/image depth until enough unique papers are found or the index
is exhausted. It also resolves complete score ties at the cutoff. There is no
fixed over-retrieval cap. Existing maximum-passage aggregation produces the final
paper scores; canonical document IDs break ties. `retrieval.top_k=100` therefore
returns up to 100 unique retrievable papers.

The index manifest records indexed units, retrievable papers, papers without
units, FAISS file size and original embedding generation time. Retrieval manifests
record search/aggregation seconds and seconds per query. These resource
measurements remain separate from effectiveness metrics; they are not a
full-corpus performance benchmark. Exact FAISS still computes inner products
against all vectors, and the flat index resides in RAM. Unlike ColQwen, each unit
has only one vector: 2048 bytes for OpenCLIP or 8192 bytes for Qwen, before mapping
and file overhead. The cache also retains the NumPy vectors separately.

## Validation

```sh
uv run pytest tests/test_visual_dense.py
# Opt-in checkpoint downloads and real four-unit index/retrieve/evaluate stages:
uv run pytest -m visual_live tests/test_visual_dense.py
```

Offline tests exercise real PDFium rendering, supplied images, MinerU v1/v2
alignment, FAISS persistence, Hydra stages, shared aggregation/evaluation and
embedding cache reuse, with mocked neural inference. They check all five
strategies, long inputs, finite unit vectors, unique document rankings, missing
page text, papers with no images, empty visual indexes, separate Qwen document
indexes and shared Qwen query embeddings.

Some macOS PyTorch/FAISS wheels bundle different OpenMP runtimes. The local test
process can select one shared runtime without changing application code:

```sh
DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.11/site-packages/torch/lib" \
  uv run pytest tests/test_visual_dense.py
```

The intended execution setup is CUDA/Linux. CPU fixture results do not establish
GPU compatibility or full-corpus throughput.

### Executed validation (2026-10-21)

- Full offline suite: **453 passed, 8 skipped, 11 deselected**.
- Final targeted configuration/artifact/visual checks: **220 passed**; final visual
  offline checks: **12 passed**.
- Real official checkpoints, all four configurations: **4 passed in 53.02 s** using
  cached weights, CPU float32 and a long synthetic ToT query. Each test indexed
  four units from two papers, reloaded the index, produced two unique document
  rankings, and evaluated RR@10/R@1/R@10/R@100 through the existing evaluator.
- OpenCLIP reported 1/1 truncated queries; Qwen preserved the full query. Qwen
  multimodal/figures retrieval reused the page baseline's query embedding cache.
- All 16 prepare/index/retrieve/evaluate script dry runs passed; wheel/sdist
  builds, lock validation, Ruff lint/format and Git whitespace checks passed.

Small CPU fixture measurements, excluding checkpoint downloads:

| Configuration | Units | FAISS bytes | Model loading + embedding generation (s) | Search + aggregation (ms/query) |
| --- | ---: | ---: | ---: | ---: |
| OpenCLIP pages | 4 | 8,237 | 3.92 | 4.02 |
| Qwen pages | 4 | 32,813 | 14.93 | 7.84 |
| Qwen multimodal | 4 | 32,813 | 15.22 | 3.65 |
| Qwen figures | 4 | 32,813 | 12.86 | 3.59 |

The fixtures include blank PDF pages and synthetic visual elements; they validate
execution and data contracts, not retrieval effectiveness. CUDA/bfloat16 inference
and full-corpus performance were not exercised locally.

### Implementation files

The OpenCLIP figures extension adds `openclip_figures` model/experiment profiles.
Its follow-up validation passed **201 targeted tests**, **two real OpenCLIP
page/figure smoke tests**, and all four stage-script dry runs. It shares page
query embeddings, preserves independent figure indexes, and needs no new dependency.

- `integrations/visual_dense.py`: shared official encoders, representation inputs,
  batching, normalization and embedding cache integration.
- `integrations/mineru.py`: deterministic v1/v2 page-text alignment.
- `integrations/dense_batch.py`: existing FAISS persistence/retrieval extended with
  adaptive document-level search and separate resource measurements.
- `adapter.py`, `data/artifacts.py`, `config_arguments.py`: image corpus selection
  and safe binary access to existing restored/TAR artifacts.
- `configuration.py`, `artifact_identity.py`, `config_artifacts.py`, `hydra_run.py`,
  `stage_cli.py`, `stages.py`: profile validation, provenance and stage integration.
- Five `config/model/` profiles and five matching `config/experiment/` profiles;
  `config/representation/figures.yaml` now has execution support.
- `scripts/01_prepare.sh` through `04_evaluate.sh`: portable empty-array expansion
  for macOS Bash, preserving existing arguments and ColQwen environment selection.
- `pyproject.toml`, `uv.lock`, `tests/test_visual_dense.py`, `tests/conftest.py`,
  README and configuration/API documentation: dependencies, tests and usage.
