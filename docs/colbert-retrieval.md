# ColBERTv2 textual baseline

`experiment=colbert_fulltext` composes the existing ColBERT model, fulltext
representation and index stage. It uses Stanford's
[colbert-ai 0.2.22](https://github.com/stanford-futuredata/ColBERT) `Indexer` and
`Searcher`, including its tokenization, inference, residual compression, candidate
search and late-interaction scoring. There is no fine-tuning or custom scoring.

```sh
uv sync --locked
export TOT_DATA_ROOT=/your/restored/data
./scripts/01_prepare.sh experiment=colbert_fulltext  # optional
./scripts/02_index.sh experiment=colbert_fulltext
./scripts/03_retrieve.sh experiment=colbert_fulltext
./scripts/04_evaluate.sh experiment=colbert_fulltext
```

The default is CPU. For CUDA, add `launcher=local_gpu` consistently to index,
retrieve and evaluate commands. CPU execution on a CUDA machine requires
`CUDA_VISIBLE_DEVICES=''` before Python starts: the upstream implementation
selects several code paths from the number of visible GPUs, independently of
`RunConfig.gpus`. MPS is unsupported. The upstream CUDA k-means path uses
FAISS CPU training with our existing `faiss-cpu` dependency; model encoding and
residual compression/search use CUDA. CUDA requires a compatible driver and
CUDA toolkit for the upstream native extensions. CPU search also compiles C++
extensions and requires a working compiler and Ninja. Single-rank indexing uses
`avoid_fork_if_possible=True`, letting upstream exceptions propagate directly.

The checkpoint is `colbert-ir/colbertv2.0`, pinned by `model.revision` to
`c1e84128e85ef755c096a95bdb06b47793b13acf`. Since ColBERT's checkpoint APIs do
not accept revisions, the integration downloads that exact Hub snapshot into
`launcher.cache_root/models` and passes its local path. The model SHA and a
SHA256 over snapshot files are saved in provenance; retrieval verifies the model
bytes before inference. Explicit local checkpoint directories are also accepted;
their bytes enter the index identity, and the revision label remains a caller
assertion. Model weights and native extension builds stay outside experiment
outputs. No model is loaded during preparation or saved-run evaluation.

## Fulltext truncation

Fulltext selects original extracted Markdown, **not complete-document encoding**.
The established `doc_maxlen=220` is forwarded to ColBERT's `DocTokenizer`: its
WordPiece tokenizer truncates to 219 tokens including CLS/SEP, then inserts the
D marker, yielding at most 220 tokens (about 217 content tokens). Punctuation
masking follows the checkpoint/library configuration. Each paper is one library
PID; no passage splitting or aggregation occurs. This effective prefix truncation
policy and token limit are recorded in the index manifest.

`index_bsize=64`, `nbits=2`, `kmeans_niters=4` and `doc_maxlen=220` go to
`ColBERTConfig` for indexing. `query_maxlen=32` goes to the search configuration;
`retrieval.top_k` goes to `Searcher.search(k=...)`. Thread counts go to Torch
and FAISS. All original values are retained. Other representations are rejected.

## Artifacts and independent stages

The index directory contains the standard `.hydra/`, completion manifest,
`index-manifest.json`, ordered canonical paper/domain mapping in `ids.json`, and
upstream index files under `colbert/`. The manifest fingerprints all index files
and records corpus content, dataset/model commits, dependency versions, document
identities, truncation and configuration. ColBERT validates an existing completed
index against current content/settings and reuses it without overwriting.
Incomplete, incompatible and corrupt destinations fail explicitly. Existing
BM25/dense index-generation skip behavior is preserved.

Query split, mode, query token limit, retrieval depth and evaluation metrics do
not enter the index identity. Retrieval opens the existing compressed index,
loads only selected queries, validates their positives against saved paper/domain
identities, and maps returned zero-based PIDs to canonical paper IDs. Explicit
`paths.index` avoids the stage script's automatic selection, which reads corpus
content to identify the matching index. Retrieval itself never loads corpus text
or rebuilds the index. It writes the shared five-file batch contract under
`results/models/`, and the unchanged saved-run evaluator writes aggregate and
per-query metrics under `results/evaluations/` without dataset, model or index
access.

## Dependencies and verification

`colbert-ai==0.2.22` and `transformers==4.57.6` are pinned in the ColBERT extra
and default repository environment. Transformers 4 requires Hub <1; the lockfile
selects `huggingface-hub==0.36.2`. The adapter imports the public repository
entry classes from `huggingface_hub.hf_api`, available in both Hub 0.x and 1.x.
The library's optional `torch` extra pins obsolete Torch 1.13.1; we deliberately
use its base package with the repository's existing Torch >=2.6 instead.

```sh
uv run pytest                      # offline; no model downloads
uv run pytest -m colbert_live tests/test_colbert.py
uv run ruff check .
uv run ruff format --check .
uv lock --check
```

The live test uses two long synthetic documents so upstream automatic k-means
has enough training token embeddings. Very small collections of short documents
can fail its partition heuristic; the integration does not alter that algorithm.
Offline tests stub expensive library indexing/search and retain real canonical
adapter loading, artifact validation, batch serialization and metric computation.
See the [implementation report](colbert-implementation-report.md) for actual
execution results and host limitations.
