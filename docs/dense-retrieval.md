# Dense retrieval with Sentence Transformers and FAISS

This tutorial substitutes pretrained embeddings for Terrier BM25 while retaining
the canonical documents, queries, qrels and identifiers. Sentence Transformers
supplies encoding; FAISS `IndexFlatIP` supplies indexing and exact top-k search.
The same Phase 3 `evaluate_run` function calls PyTerrier Experiment and
ir_measures for RR@10, Recall@1, Recall@10 and NDCG@10. No neural model, vector search,
ranking algorithm or metric is implemented here, and no fine-tuning is performed.

## Install

Run from the repository root in the same `.venv` as BM25 and both notebooks:

```sh
uv sync --locked
```

The default project environment includes Sentence Transformers, Torch, FAISS,
NumPy, Parquet, PyTerrier, JupyterLab, and the bundled JDK. Dense retrieval does
not start Java. Linux uses the configured PyTorch CPU index to avoid CUDA
package downloads. No additional install or extra selection is needed.

## First small run

The offline data fixture is synthetic; the first encoding requires downloading
the pretrained model, then subsequent identical runs can reuse the cache:

```sh
uv run python examples/dense_retrieval.py \
  --local-dir tests/fixtures/adapter --documents 2 --queries 1 \
  --cache results/dense-cache --output results/dense-fixture
```

For the actual ACL subset already used by the BM25 example:

```sh
uv run python examples/dense_retrieval.py \
  --local-dir ../multimodal-tip-of-tongue-for-scientific-papers/data \
  --domain ACL --corpus-split train+test --query-split test \
  --mode visual-and-text --documents 250 --queries 10 --missing skip \
  --baseline-dir results/acl-small/fulltext \
  --cache results/dense-cache --output results/acl-dense/fulltext
```

Create the matching BM25 runs first using the [Phase 3 guide](pyterrier.md), or
omit `--baseline-dir` to run independently. When supplied, this option verifies
the ordered corpus ID list, exact corpus text hash, topics, query-ID mappings and
qrels before any model is loaded. A mismatch raises rather than silently creating
an incomparable run. Both examples use the same canonical adapter and deterministic fulltext subset
helper. Every positive target must resolve; no qrel is dropped to accommodate
missing content.

## Python API

```python
from scientific_tot import Dataset, DatasetAdapter
from scientific_tot.integrations.dense import EncodingConfig, dense_experiment

adapter = DatasetAdapter(Dataset(local_dir="tests/fixtures/adapter"))
corpus = adapter.load_corpus(domain="ACL", split="train+test", representation="fulltext")
queries = adapter.load_queries(domain="ACL", split="test", mode="visual-and-text")
config = EncodingConfig(device="cpu", batch_size=16)
metrics = dense_experiment(
    corpus,
    queries,
    "/tmp/tot-dense-example",
    cache="/tmp/tot-dense-cache",
    config=config,
)
print(metrics)
```

Corpus positional rows map directly to original canonical `doc_id` values through
a persisted `ids.json`; FAISS row positions never become new paper identifiers.
The Phase 3 conversion utility validates qrels and retains its reversible mapping
for query IDs reused across collections. Runs are standard TREC files written by
PyTerrier; topics, qrels and mappings retain the same formats as the BM25 outputs.

## Substitute your model and encoding instructions

The tutorial default is `intfloat/e5-small-v2` pinned to
`ffb93f3bd4047442299a41ebb6fa998a38507c52`. It is a small English pretrained
retrieval model chosen for a CPU tutorial, not a claim of experimental superiority.
Its model card requires query and passage prefixes, which this example supplies
through Sentence Transformers' `prompt` argument, without manually prepending
text. [E5 model card](https://huggingface.co/intfloat/e5-small-v2).

For another model, select a Hub ID and immutable revision:

```sh
uv run python examples/dense_retrieval.py \
  --local-dir tests/fixtures/adapter --documents 2 --queries 1 \
  --model sentence-transformers/all-MiniLM-L6-v2 --model-revision main \
  --encoding-profile model --device cpu --batch-size 16 \
  --cache results/dense-cache --output results/minilm-fixture
```

A branch/tag such as `main` resolves to a commit SHA before model loading and is
recorded as `model.resolved_revision`. Copy that SHA into `--model-revision` for
later reproduction. Supplying a commit avoids the metadata resolution request.
Dataset `--revision` is separate: it pins a Hub dataset or records a caller's
local provenance assertion. For an unversioned restored root, the manifest stores
`dataset_revision: null`, `dataset_revision_status: local/unrecorded`, exact
corpus/query hashes and the baseline manifest hash; it never invents a revision.

`--encoding-profile auto` applies required `query: ` and `passage: ` prompts for
the explicitly supported standard English/multilingual E5 small/base/large
families. For other models it delegates to the model's saved prompts and Router
via **`encode_query` and `encode_document`**, not generic `encode`. Named prompts
can be selected with `--query-prompt-name`/`--document-prompt-name`.
[Sentence Transformers encoding API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).

For instructions missing from a model's saved configuration, use
`--query-prompt 'Your task instruction: '` and `--document-prompt 'Document prefix: '`
with the exact text required by its model card. Prompt strings take precedence
over profiles; strings and names cannot be supplied together for the same task.
An empty string explicitly disables that task's prompt. E5 instruct variants
require an explicit task-specific query instruction; the example refuses to guess
one. `profile="model"` explicitly delegates prompts to model configuration; use
it only when that configuration includes the required instructions or the model
needs none. Read the model card when substituting a model: prompts and similarity
settings cannot be inferred universally from arbitrary Hub names.

Cosine is the default: Sentence Transformers normalizes vectors and FAISS performs
inner-product search. Select `--similarity dot` for a model intended for
unnormalized dot products. The example supports float32 sentence embeddings and
FAISS inner-product indexes; it does not support cross-encoders, late interaction,
custom remote model code, or every possible embedding architecture. Remote code
execution is disabled. [Default alternate model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

**Long text:** each document/query produces one vector. The model tokenizer
truncates to its configured maximum (512 tokens for the default E5 model),
including prompts. `--max-seq-length` explicitly overrides that budget; it must
remain within the model's supported context. The manifest records the effective
length and the policy `model_tokenizer_truncation_no_chunking`. A fulltext corpus
thus supplies full extraction text but does not imply every paper token is encoded.
This tutorial performs no custom chunking or aggregation.

## Batching, persistent caches and resources

`--batch-size` controls Sentence Transformers' own batching. Encoding, FAISS search
and evaluation run directly in the same process. Model weights use
its Hugging Face cache through `cache_folder`. Corpus/query embeddings persist
with NumPy `save`/`load`, using read-only memory mapping on reuse; FAISS's
`write_index`/`read_index` persists the index. Source texts, order, IDs, model SHA,
versions, instructions, precision and configuration determine cache keys. Completed
entries are published after file writes and verified with hashes before loading.
Corrupt or incomplete entries raise; remove the damaged entry and rerun to rebuild.
[FAISS index I/O](https://github.com/facebookresearch/faiss/wiki/Index-IO%2C-cloning-and-hyper-parameter-tuning).

Changing query text reuses the document index and rebuilds query embeddings.
Fully cached runs neither instantiate a neural model nor start Java. Models,
representations or encoding configurations with different keys coexist in the
cache. Old entries are retained until you remove them. Changes to software versions,
device or batch configuration deliberately create distinct entries. Cache paths,
hits, model prompts, dimensions and indexing settings appear in `manifest.json`.

CPU-only execution is supported and was tested on dgx03. The default model has
about 134 MB of float32 weights; two 250-by-384 float32 embedding/index arrays
require under 1 MB together, excluding model activations and Python overhead.
Actual RAM use also includes the adapter's eager selected corpus and restored-root
inventory; these can dominate on large source trees. The FAISS flat index scales
with corpus size and embedding dimension and stays on CPU in this example.
The `--threads` option controls FAISS CPU threads; `--encoding-threads` controls
PyTorch CPU threads. Both default to one for reproducible fixture runs and to
avoid native OpenMP thread-pool conflicts on macOS ARM. The actual encoding
thread count is recorded in the manifest. Adjust batch size to fit memory.
A related upstream issue is tracked in
[PyTorch #149201](https://github.com/pytorch/pytorch/issues/149201).

The shared locked Linux environment uses CPU Torch, so use `--device cpu` there.
CUDA encoding requires a different Torch build and is outside this reproducible
setup. No separate GPU environment is needed for any documented example.

The default uses all available documents in each selected domain, fulltext, and
all test queries: `--corpus-split all --documents 0 --queries 0`. No numeric cap
is imposed. Small demos require explicit `--documents` and `--queries` limits;
add `--corpus-split train+test` only to restrict the corpus to assigned papers.
Subset limits apply after canonical loading.
The current documented Hub dataset contains queries only, so experiments require
restored corpus/split data until those components are published.

## Notebook, outputs and checks

[Notebook](../examples/dense_retrieval.ipynb) shows configuration, encoding/search,
metrics, revision inspection and an identical cached rerun. Register its kernel:

```sh
uv run python -m ipykernel install --sys-prefix --name scientific-tot --display-name "Scientific ToT"
export TOT_DATA_ROOT=/path/to/restored/data
export TOT_DENSE_OUTPUT=/path/to/new/notebook-output
# Optional: matching saved baseline and a reusable cache:
export TOT_BM25_BASELINE=/path/to/bm25/fulltext
export TOT_DENSE_CACHE=/path/to/cache
```

Launch `uv run jupyter lab` and select **Scientific ToT**, the same kernel used
by BM25. By default, the notebook uses the restored sibling data directory,
all available documents, all test queries, and fresh output directories.
Set `TOT_DATA_ROOT` for another data root (or `tests/fixtures/adapter` for a smoke
test); `DOCUMENTS, QUERIES = 250, 10` explicitly selects a demo. Retrieval uses
original full document text and requires no bibliography metadata.

Use a fresh result directory for each experiment. Output includes `run.trec`,
aggregate/per-query metrics, topics, canonical qrel tables, query-ID mappings,
selected document IDs and a reproducibility manifest. FAISS indexes/embeddings
live in the separately reusable cache directory. [Actual-data results](dense-results.md)
record the CPU execution and cache-reuse check.

```sh
uv run pytest                    # offline fixtures, including real FAISS tests
uv run pytest -m dense_live       # optional pretrained download + encoding test
```

Offline tests stub only neural encoding, then exercise real FAISS search,
standard metrics, baseline rejection, corruption detection and selective cache
reuse. The live test encodes with the pinned pretrained model and checks that a
cached rerun produces identical TREC output. The existing Terrier test remains
selected independently with `-m terrier`.
