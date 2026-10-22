# OpenCLIP and Qwen3-VL-Embedding retrieval

Five zero-shot profiles share Hydra stages, FAISS persistence, embedding caches,
the TREC writer and saved-run evaluation. All use the original textual ToT queries
and the selected paper collection independently for each scientific domain.

| Model profile / experiment | Document input | Indexed unit | Dimensions |
| --- | --- | --- | --- |
| `openclip` / `openclip_pages` | Rendered original PDF page | Page | 512 |
| `openclip_figures` | Individual supplied figure/table image | Image | 512 |
| `qwen3_vl_images` | All supplied images in one joint input | Paper | 2048 |
| `qwen3_vl_text` | Full original Markdown text | Paper | 2048 |
| `qwen3_vl_multimodal` | All supplied images, then full Markdown text, in one joint input | Paper | 2048 |

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
message in dataset order. In multimodal mode the complete Markdown text follows
the last image. One forward input and one normalized embedding represent each
paper; images are never independently embedded and averaged.
See the [official implementation](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/blob/main/scripts/qwen3_vl_embedding.py).

Text comes from the adapter's original `markdown.md` full-text representation,
exactly as for BM25/E5 full text. No page matching, content-list JSON, PDF rendering,
chunking, summarization or repeated text per image is required for Qwen.
Images are the supplied extracted figures/tables, including lean `images/`
staging; restored files and TAR archives work. Text-only Qwen does not read images,
and image-only Qwen does not read Markdown. Source paths and hashes for every
image and the complete text are included in document cache identity.

Qwen image-only inputs require at least one supplied image per paper, otherwise
indexing fails with the paper ID. Multimodal papers with no images encode their
full text. Missing or unusable selected Qwen images fail explicitly, so an input
cannot silently become a subset of the paper's images. The existing OpenCLIP
figures missing policy is unchanged: papers without images stay in evaluation
but receive no vector, and explicit `dataset.missing=skip` can exclude unusable images.

All Qwen queries use `Retrieve images or text relevant to the user's query.`
Queries remain intact and their embeddings are shared across the three profiles.
The shared context budget is 32768 tokens, including image tokens, full text and
the chat template. The official processor runs once, with overflow detection;
its tensors are reused for inference. Over-budget inputs raise with the paper ID
instead of silently truncating text or omitting images. Official image resizing
uses 4096–1843200 pixels per image.

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

The following executes indexing, retrieval, and evaluation for all five
configurations independently per domain. Each experiment selects the existing
MRR@10 (`RR@10`), Recall@1/10/100 and NDCG@10 metrics. The test query split
retains all available corpus documents as distractors.

```sh
export TOT_DATA_ROOT=/your/restored/data

for domain in ACL Biology Engineering Medicine Physics; do
  export TOT_OUTPUT_ROOT="./results/$domain"

  for experiment in openclip_pages openclip_figures qwen3_vl_text qwen3_vl_multimodal qwen3_vl_images; do
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
`qwen3_vl_images` to run the others. Preparation is optional, using
`./scripts/01_prepare.sh` with the same experiment and domain overrides. Direct
`model=...` overrides also choose the appropriate representation, while the
experiment profiles additionally select the benchmark metrics.

CUDA is the default. OpenCLIP uses `model.precision=fp32`; Qwen uses `bf16` with
Transformers SDPA. Supported precision settings are `fp32`, `fp16`, and `bf16`;
choose one supported by the GPU. CPU smoke runs explicitly use
`launcher=local_cpu model.precision=fp32`. No device fallback occurs.
Adjust `model.batch_size` consistently across index/retrieve/evaluate commands to
fit memory. Qwen also exposes `model.max_length` and `model.query_instruction`;
all Qwen profiles inherit their shared settings from `config/model/qwen3_vl_text.yaml`.

Indexes live under the existing `indexes/`, retrieval artifacts under `models/`,
and metrics under `evaluations/`. Distinct model names, strategies and content
hashes prevent index collisions. `unit-mapping.jsonl` records each vector's paper
and its complete text/image inputs for Qwen, or its page/image source for OpenCLIP.
The former `qwen3_vl_page` and `qwen3_vl_figures` profiles are removed; rebuild
indexes with the new profiles. Old page-level multimodal indexes are incompatible.
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
paper-level embeddings, input mode, total supplied images and full-text counts.
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
before-text ordering, image retention, context overflow, missing-image behavior,
separate document indexes, shared queries and isolated test-mode execution.

Some macOS PyTorch/FAISS wheels bundle different OpenMP runtimes. The local test
process can select one shared runtime without changing application code:

```sh
DYLD_LIBRARY_PATH="$PWD/.venv/lib/python3.11/site-packages/torch/lib" \
  uv run pytest tests/test_visual_dense.py
```

The intended execution setup is CUDA/Linux. CPU fixture results do not establish
GPU compatibility or full-corpus throughput.

Executed validation on 2026-10-22:

- Full offline suite: **510 passed, 8 skipped, 12 deselected**.
- Real pinned Qwen checkpoint, CPU float32: **3 passed** (images, text and
  multimodal), with indexing, retrieval and evaluation for each profile.
- Each live Qwen index contained **2 paper vectors for 2 papers**. Image inputs
  retained all 4 supplied images; text inputs retained both original full texts.
  Retrieval reported no aggregation for any Qwen profile.
- All **24** Qwen stage-script dry runs passed (prepare/index/retrieve/evaluate,
  normal and isolated test modes). Ruff lint/format, lock validation and Git
  whitespace checks passed.

The historical page-level Qwen smoke measurements do not apply to these
whole-document profiles. CUDA/bfloat16 inference and full-corpus throughput have
not been exercised locally.

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
- Five `config/model/` profiles and five matching `config/experiment/` profiles;
  `config/representation/figures.yaml` now has execution support.
- `scripts/01_prepare.sh` through `04_evaluate.sh`: portable empty-array expansion
  for macOS Bash, preserving existing arguments and ColQwen environment selection.
- `pyproject.toml`, `uv.lock`, `tests/test_visual_dense.py`, `tests/conftest.py`,
  README and configuration/API documentation: dependencies, tests and usage.
