# Scientific Tip of the Tongue Retrieval.

This repository is the retrieval model implementation for **Multimodal Tip-of-the-Tongue Retrieval for Scientific Papers**, which introduces a
[multimodal scientific tip-of-the-tongue dataset](https://huggingface.co/datasets/kasys/multimodal-tip-of-the-tongue-retrieval-for-scientific-documents).
In this repository we provide implementations of retrieval models to be used for evaluation. This package does not depend on the dataset generation pipeline at runtime
and delegates ranking, pretrained encoding, and evaluation to PyTerrier,
Sentence Transformers, FAISS, Stanford ColBERT, Qdrant, and ir_measures.


## Downloading the Dataset from Huggingface

The base download script and its Python helpers are included in this repository;
no generation checkout is needed. Install [`uv`](https://docs.astral.sh/uv/), then create and activate the environment:

```bash
uv sync --locked
```

This installs `huggingface-hub`, `pandas`, and `pyarrow` along with the base
package.

The scripts below handle downloading each component of the dataset separately.
They should be run on a machine with sufficient storage space (at least 500GB of free space).

From the repository root, run the script with no arguments to download and restore everything (PDFs, preprocessed content, clues, query collections, and splits) under the default `data/` directory:

```bash
scripts/packaging/download_huggingface_dataset.sh
```

To download all generated components without PDFs, use `--generated`. This also uses `data/` by default:

```bash
scripts/packaging/download_huggingface_dataset.sh --generated
```

The `data` argument is the destination directory; omit it to use `data/`, or replace it with another path. `--all` explicitly selects every component. `--pdfs` downloads `metadata.parquet` and PDF shards, then restores PDFs under `data/pdf_datasets/<SOURCE>/` with size and SHA-256 verification. `--generated` selects parsed content, clues, query collections, and splits. Use `--overwrite` to existing files that differ. `--help` lists other options. 

This will result in the following dataset structure:
```text
data/
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

You are free to set any data or output path, please adjust `defaults.sh` to do so.


### Retrieval Baselines

The benchmark includes lexical, dense, late-interaction, and multimodal retrieval methods. All methods retrieve scientific papers, either directly or by aggregating scores from individual pages or figures.

| Method | Model | Document input | Indexing and aggregation | Input limit | Configuration (`experiment=...`) |
|---|---|---|---|---|---|
| BM25 | `BM25` (PyTerrier Implementation) | Full Markdown text | Paper-level lexical index | No explicit limit | `bm25_fulltext` |
| E5 | `e5-small-v2` | Markdown text | One vector per paper | 512 tokens | `dense_fulltext` |
| Qwen3 Text | `Qwen3-Embedding-0.6B` | Markdown text | One vector per paper | 32,768 tokens | `qwen3_text` |
| ColBERTv2 | `colbertv2.0` | Markdown text | Token-level late interaction per paper | 220 doc / 32 query tokens | `colbert_fulltext` |
| ColQwen2.5 | `colqwen2.5-v0.2` | Rendered PDF pages | Page-level late interaction; max score per paper | All pages | `colqwen_pages` |
| OpenCLIP | `ViT-B-32` | Rendered PDF pages | Page embeddings; max score per paper | All pages | `openclip_pages` |
| OpenCLIP | `ViT-B-32` | Extracted figures/tables | Image embeddings; max score per paper | All images | `openclip_figures` |
| Qwen3-VL | `Qwen3-VL-Embedding-2B` | Markdown text | One vector per paper | 32,768 tokens | `qwen3_vl_text` |
| Qwen3-VL | `Qwen3-VL-Embedding-2B` | Extracted figures/tables | One vector per paper | 32,768 tokens | `qwen3_vl_figures` |
| Qwen3-VL | `Qwen3-VL-Embedding-2B` | Rendered PDF pages | One vector per paper | First 18 pages; 32,768 tokens | `qwen3_vl_pages` |
| Qwen3-VL | `Qwen3-VL-Embedding-2B` | Figures/tables + Markdown | One multimodal vector per paper | 32,768 tokens | `qwen3_vl_multimodal` |

**Notes:**
- Select a retrieval method using its Hydra configuration, e.g. `experiment=qwen3_vl_multimodal`.
- Text is truncated when it exceeds the model's context window for both query and document.
- Qwen3-VL retains images in order until the token budget is exhausted. In multimodal mode, text is truncated to the remaining context.
- Page- and figure-level retrieval scores are aggregated to paper level using the maximum score.
- All methods use the default `.venv` except ColQwen2.5, which requires `.venv-colqwen`.




## Run baselines
In this section, we describe how to run the baselines.

### Install environments
If environments are not yet installed, please make sure to install them, and add the colqwen environment if you wish to use this method. 

```sh
uv sync --locked
```

```sh
UV_PROJECT_ENVIRONMENT=.venv-colqwen uv sync --locked \
  --no-default-groups --extra colqwen --group dev
```

### Run experiments
To run a method, we include scripts both to run the full benchmark, as well as to run indexing, retrieval and evaluation separately. 

We provide scripts to run the index, retrieval, and evaluation stages individually. Please select the dataset you wish to use from the five domains Biology, Medicine, ACL, Physics, and Engineering by adding the `--dataset.domains=[DOMAIN]` flag to the scripts. Below, we show an example to run indexing, retrieval and evaluation with Qwen3-emb on the ACL domain. 


```bash
./scripts/run_benchmark.sh 'benchmark.experiments=[qwen3_text]' 'benchmark.domains=[ACL]'
```
To run separately, use:
```bash
./scripts/02_index.sh experiment=qwen3_text 'dataset.domains=[ACL]'
./scripts/03_retrieve.sh experiment=qwen3_text 'dataset.domains=[ACL]'
./scripts/04_evaluate.sh experiment=qwen3_text 'dataset.domains=[ACL]'
```

As runs are likely to run for a significant amount of time, and require GPU usage, we recommend using a job scheduler, such as PBS or slurm. We provide an implementation script to run on a PBS cluster, to use this, add the `launcher=h200` flag to the scripts. Note, this script will run all experiments and domains, if you wish to run a subset, please edit the script accordingly. As evaluation is not computationally expensive, this can also be run locally.

### Logs, offline execution and interruption

Artifacts are written to `results/{prepared,indexes,models,evaluations}/`, with the
existing model/representation/domain/split/input identities, manifests and checksums. Every job also writes a log to `results/logs/`.

As in dataset generation, each submitted job has a named application log beside
its PBS files: `prepare.log`, `index.log`, `retrieve.log`, `evaluate.log`,
`test.log`, or `benchmark.log`. It is created before submission and records UTC
submission, worker-start and completion/failure timestamps. Worker stdout and
stderr, including native-library/child-process output and full error tracebacks,
are appended and flushed while the job runs. You can inspect these application
logs immediately without waiting for PBS to deliver its stdout/stderr files.


### Metrics, reuse and output locations

The default paper cutoff is `retrieval.top_k=100`. Default effectiveness metrics
include **RR@10, R@1, R@10, R@100**, and **nDCG@10**.

### License
This repository uses the [MIT License](https://github.com/stanfris/multimodal-tot-retrieval/blob/main/LICENSE).
