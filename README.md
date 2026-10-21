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

## Install and run

Use Python 3.11 (selected by `.python-version`) and [uv](https://docs.astral.sh/uv/).
Run all commands from the repository root. One locked `.venv` contains the adapter,
Parquet reader, BM25, dense retrieval, bundled Java runtime, JupyterLab, and test tools.
`uv run` uses this environment for textual baselines. ColQwen uses the separate
locked `.venv-colqwen` environment because Transformers 5 conflicts with ColBERT
(see below):

```sh
uv sync --locked
# Fully offline two-document synthetic example:
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

## Hydra configuration CLI

The four independent stages write directly into deterministic artifact directories:

```text
results/
  indexes/bm25_fulltext_d-ACL_c-all/
    .hydra/{config,hydra,overrides}.yaml
    manifest.json
    index-manifest.json
    data.properties                 # plus Terrier files
  models/bm25_fulltext_c-all_q-test_<retrieval-hash>/
    .hydra/  manifest.json  run.trec  topics.jsonl  qrels.tsv
    query-ids.jsonl  corpus-ids.jsonl
  evaluations/bm25_fulltext_c-all_q-test_<evaluation-hash>/
    .hydra/  manifest.json  metrics.csv  per-query.csv
  prepared/bm25_fulltext_c-all_q-test_<prepare-hash>/
    .hydra/  manifest.json  corpus.jsonl  queries.jsonl  qrels.jsonl
```

Dense indexes use the same layout with `dense_fulltext` or `dense_passages`
and FAISS/vector files. ColBERT indexes store the upstream compressed index
under `colbert/` with a canonical PID mapping. `models/` contains retrieval outputs; pretrained weights
and embeddings stay in the existing `cache/` hierarchy. There are no timestamped
artifact wrappers.

### Run BM25 and dense baselines on the test set

After `uv sync --locked`, point `TOT_DATA_ROOT` at your restored dataset. These
commands index every available ACL document and evaluate all test queries from
the `visual-and-text` query collection, using original fulltext for both baselines:

```sh
export TOT_DATA_ROOT=/your/restored/data

# BM25 baseline: index, retrieve test queries, then compute scores.
./scripts/02_index.sh experiment=bm25_fulltext split=test
./scripts/03_retrieve.sh experiment=bm25_fulltext split=test
./scripts/04_evaluate.sh experiment=bm25_fulltext split=test

# Pretrained dense baseline (intfloat/e5-small-v2, CUDA; no fine-tuning).
./scripts/02_index.sh model=dense representation=fulltext split=test
./scripts/03_retrieve.sh model=dense representation=fulltext split=test
./scripts/04_evaluate.sh model=dense representation=fulltext split=test
```

`split=test` selects test queries while keeping `split.corpus=all`, so distractor
documents remain in the retrieval collection. There are no document or query
limits. The first BM25 run downloads Terrier's JAR; the first dense run downloads
the pretrained weights. Preparation/export with `./scripts/01_prepare.sh` is
optional. The dataset must include corpus, query, and split artifacts as described
in [the data contract](docs/dataset-schema.md).

Each stage prints its completed output directory. Aggregate test-set scores are
in `results/evaluations/bm25_fulltext_c-all_q-test_<evaluation-hash>/metrics.csv`
and `results/evaluations/dense_fulltext_c-all_q-test_<evaluation-hash>/metrics.csv`;
`per-query.csv` contains individual query scores. Rankings are saved as `run.trec`
under the corresponding `results/models/` directory. Both baselines retrieve
top-100 and report RR@10, Recall@1, Recall@10, and NDCG@10 by default. To also
report Recall@100, pass `'evaluation.metrics=[RR@10,R@1,R@10,R@100,nDCG@10]'`
to each evaluation command.

For passage-based dense retrieval, replace `model=dense representation=fulltext`
with `experiment=dense_passages` in all three dense commands. This splits original
fulltext into overlapping passages and aggregates their scores by maximum per
paper.

### Evaluate each domain separately

The published domains are `ACL`, `Biology`, `Engineering`, `Medicine`, and
`Physics`. Pass the same `dataset.domains` override to indexing, retrieval, and
evaluation. For example, run both baselines separately for every domain:

```sh
export TOT_DATA_ROOT=/your/restored/data

for domain in ACL Biology Engineering Medicine Physics; do
  export TOT_OUTPUT_ROOT="./results/$domain"

  ./scripts/02_index.sh experiment=bm25_fulltext split=test "dataset.domains=[$domain]"
  ./scripts/03_retrieve.sh experiment=bm25_fulltext split=test "dataset.domains=[$domain]"
  ./scripts/04_evaluate.sh experiment=bm25_fulltext split=test "dataset.domains=[$domain]"

  ./scripts/02_index.sh model=dense representation=fulltext split=test "dataset.domains=[$domain]"
  ./scripts/03_retrieve.sh model=dense representation=fulltext split=test "dataset.domains=[$domain]"
  ./scripts/04_evaluate.sh model=dense representation=fulltext split=test "dataset.domains=[$domain]"
done
unset TOT_OUTPUT_ROOT
```

To run only selected domains, shorten the loop list, for example
`for domain in Biology Physics; do`. Each domain needs its restored corpus,
test queries, and split artifacts. Scores are saved separately under
`results/<DOMAIN>/evaluations/`; each baseline's `metrics.csv` reports that
domain's test-set scores, and `per-query.csv` reports its individual query scores.
The cache remains shared at `./cache`.

If indexing and retrieval have already completed, rerun only the corresponding
`04_evaluate.sh` command with the same domain and output root. For example:

```sh
TOT_OUTPUT_ROOT=./results/Biology ./scripts/04_evaluate.sh \
  experiment=bm25_fulltext split=test 'dataset.domains=[Biology]'
TOT_OUTPUT_ROOT=./results/Biology ./scripts/04_evaluate.sh \
  model=dense representation=fulltext split=test 'dataset.domains=[Biology]'
```

Each run selects exactly one domain. Use the separate runs above for per-domain
baseline results.

The scripts run from the repository root and share `./data`, `./cache`, and
`./results`; edit `scripts/defaults.sh` or use `TOT_DATA_ROOT`, `TOT_CACHE_ROOT`,
and `TOT_OUTPUT_ROOT`. Their uv cache defaults to `${TOT_CACHE_ROOT}/uv`.
BM25/fulltext defaults to ACL, all corpus documents, test queries, CPU, top-100,
and RR@10/Recall@1/Recall@10/NDCG@10. The scripts do not restore datasets or launch prerequisites.
Automatic index selection rereads the selected corpus to match its exact identity;
multiple compatible saved retrieval runs require an explicit `paths.run` choice.
Changing content/settings creates a new identity;
identical completed artifacts are always verified and reused.

Runs print corpus selection and counts, index/model metadata, cache hits, and stage
status as work proceeds. Progress bars track BM25 indexing/retrieval, dense encoding,
passage splitting, and staged FAISS query retrieval. Index stages select documents
only; query counts and test/train selection are printed during retrieval.

The bare Hydra CLI supports explicit artifact inputs and multiruns. Copy the
completed paths printed by each command into `INDEX` and `RUN` below:

```sh
uv run scientific-tot stage=index dataset.root=/your/restored/data
# Set INDEX to the completed results/indexes/... directory.
uv run scientific-tot stage=retrieve dataset.root=/your/restored/data "paths.index=$INDEX"
# Set RUN to the completed results/models/... directory.
uv run scientific-tot stage=evaluate dataset.root=/your/restored/data "paths.run=$RUN"
uv run scientific-tot stage=evaluate dataset.root=/your/restored/data "paths.run=$RUN" 'evaluation.metrics=[RR@10,R@100]'
# An explicit persistent destination is also supported:
uv run scientific-tot stage=index dataset.root=/your/restored/data paths.index=results/indexes/my-bm25-index
uv run scientific-tot dry_run=true experiment=dense_passages launcher=local_gpu
uv run scientific-tot --multirun stage=index split.queries=train,test
```

Index names use readable model/representation/corpus split labels without a checksum
suffix. Existing BM25/dense index generation skips existing destinations. ColBERT
instead verifies corpus/settings identity and sealed checksums before reuse. Use a different `paths.index` to build another index.
Other stage names combine readable model/representation/split labels with the first 16 hex
characters of a SHA256 of the resolved content identity. Full identities and
hashes remain in manifests alongside full configuration hashes. Index identities
include corpus bytes, dataset commit, representation, construction settings and
resolved document encoder settings/commit. Query split, scoring controls, retrieval
depth and metrics do not affect indexes. Retrieval identities include the exact
index manifest, query content/selection and ranking parameters. Evaluation identities
include the exact saved retrieval manifest and metric settings; evaluation needs
neither the original dataset nor the index.

For stages other than index generation, identical completed outputs are reused
only after manifest, snapshot and output checksum verification, without rewriting files. Incomplete, corrupted, legacy or
conflicting destinations fail explicitly. Concurrent writers reserve exclusively;
index multirun jobs sharing a destination skip subsequent builds. Other stages
reuse identical completed outputs and give different identities separate destinations. Index and retrieval consumers require schema-2 artifacts
and explicit document-domain identities. Rebuild older artifacts from original inputs.

[Configuration guide](docs/configuration.md) covers profiles, naming, provenance,
reuse and limitations. BM25 and dense `prepare`, `index`, `retrieve`, and `evaluate`
execute for text corpora. Dense passages preserve the configured LangChain splitter
and PyTerrier max/mean aggregation. ColBERT executes all four stages with fulltext.
ColQwen and the single-vector visual profiles execute the same four stages.
`search`, `export`, and the legacy `model=visual` profile remain configuration-only.
Standalone tutorial examples below retain
their established output layouts and APIs.

## ColBERTv2 textual baseline

```sh
./scripts/02_index.sh experiment=colbert_fulltext
./scripts/03_retrieve.sh experiment=colbert_fulltext
./scripts/04_evaluate.sh experiment=colbert_fulltext
```

This uses the pinned `colbert-ir/colbertv2.0` checkpoint and original extracted
fulltext, truncated by ColBERT to **220 tokens including special tokens** per
paper. It does not split documents into passages. Preparation is optional; model
weights download on first indexing/search. CUDA is selected by default and required for indexing/retrieval. Saved-run
evaluation requires no GPU. Native extensions require a compiler, Ninja and a CUDA toolkit.
[ColBERT guide](docs/colbert-retrieval.md) covers parameters, provenance,
requirements and the opt-in real-checkpoint test.

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
falling back to CPU, even when embeddings are cached. FAISS search and saved-run
evaluation remain on CPU; evaluating an existing run needs no GPU.

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

## Visual PDF retrieval with ColQwen2.5

`experiment=colqwen_pages` uses the pinned `vidore/colqwen2.5-v0.2` checkpoint
through Sentence Transformers 6.1's `MultiVectorEncoder`. PDFium renders original
PDF pages; the official checkpoint processor handles model preprocessing. Qdrant
persists page multi-vectors; Sentence Transformers performs batched exact MaxSim on CUDA. Maximum page-score
aggregation produces canonical paper rankings, and `retrieval.top_k` counts
papers. This aggregation is specific to our paper-level benchmark.

```sh
export TOT_DATA_ROOT=/your/restored/data
./scripts/01_prepare.sh experiment=colqwen_pages
./scripts/02_index.sh experiment=colqwen_pages
./scripts/03_retrieve.sh experiment=colqwen_pages
./scripts/04_evaluate.sh experiment=colqwen_pages
```

These scripts automatically use the locked `.venv-colqwen` environment and
select CUDA. ColQwen encoding requires an available NVIDIA GPU; CPU/MPS
overrides and CPU fallback are rejected. Preparation and saved-run evaluation
require no GPU. PDF rendering and SQLite reads use the CPU; encoding and MaxSim
scoring use CUDA. Originals must be restored
under `pdf_datasets/` with canonical extraction identity metadata. Indexes have
content hashes and are verified before reuse across query splits/modes/depths.
Retrieval consumes saved pages; evaluation consumes only saved rankings.
Exact search streams saved page vectors from read-only SQLite once for all queries,
using bounded page and query groups for CUDA scoring. This remains an exhaustive correctness baseline.
Pretrained CPU/GPU inference and full corpus performance have not been exercised locally. See the
[visual retrieval guide](docs/colqwen-retrieval.md) for pinned versions, rendering,
identity, resource requirements, and the selectable real-checkpoint test, and the
[implementation report](docs/colqwen-implementation-report.md) for executed checks.

## OpenCLIP and Qwen3-VL-Embedding baselines

Five experiments use single-vector FAISS retrieval with maximum-score document
aggregation: `openclip_pages`, `openclip_figures`, `qwen3_vl_page`, `qwen3_vl_multimodal` (page plus
aligned MinerU text), and `qwen3_vl_figures` (supplied figures/table images).
They share the existing stage scripts and document-level evaluation, reporting
MRR@10 and Recall@1/10/100. Qwen query embeddings are shared across its three
strategies; document embeddings and indexes remain separate.

```sh
for experiment in openclip_pages openclip_figures qwen3_vl_page qwen3_vl_multimodal qwen3_vl_figures; do
  ./scripts/02_index.sh "experiment=$experiment"
  ./scripts/03_retrieve.sh "experiment=$experiment"
  ./scripts/04_evaluate.sh "experiment=$experiment"
done
```

All run in the default locked environment. Multimodal page-text alignment
requires original MinerU content-list JSON, which lean Markdown/image staging
omits. The [visual dense retrieval guide](docs/visual-dense-retrieval.md) documents
inputs, dimensions, per-domain commands, preprocessing, efficiency and smoke tests.

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
