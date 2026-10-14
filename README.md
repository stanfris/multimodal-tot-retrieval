# scientific-tot-retrieval

Canonical dataset adapters for the
[scientific tip-of-the-tongue dataset](https://huggingface.co/datasets/kasys/open-source-scientific-documents).
Load selected original-content corpus records, queries, and validated relevance
judgments, then export standard JSONL for existing retrieval frameworks.

The published dataset is the source of truth. The
[generation repository](https://github.com/stanfris/multimodal-tip-of-tongue-retrieval-for-scientific-papers)
creates the artifacts; this package does not depend on that pipeline at runtime
and delegates ranking, pretrained encoding, and evaluation to PyTerrier,
Sentence Transformers, FAISS, and ir_measures.

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
`uv run` uses this same environment; no extra flags or separate environments are needed:

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

### Stage scripts with automatic reuse

```sh
./scripts/01_prepare.sh     # Optional validation/export; not an index prerequisite.
./scripts/02_index.sh       # Reuses a compatible completed index if one exists.
./scripts/03_retrieve.sh    # Finds the newest compatible completed index.
./scripts/04_evaluate.sh    # Finds the newest compatible completed retrieval run.
```

All scripts run from the repository root regardless of your current directory.
Defaults are BM25/fulltext, ACL, all corpus documents, test queries, CPU, top-100,
and nDCG@10/RR@10/R@100. Shared paths are `./data`, `./cache`, and `./results/hydra`.
Put the restored canonical dataset in `./data`, or edit `scripts/defaults.sh` once
to set its location; environment variables `TOT_DATA_ROOT`, `TOT_CACHE_ROOT`, and
`TOT_OUTPUT_ROOT` take precedence. The scripts do not generate or restore data.
The uv package cache defaults to `${TOT_CACHE_ROOT}/uv`; `UV_CACHE_DIR` can override it.

Hydra overrides still work; use the same experiment choices at each stage:

```sh
./scripts/02_index.sh experiment=dense_passages
./scripts/03_retrieve.sh experiment=dense_passages
./scripts/04_evaluate.sh experiment=dense_passages
./scripts/02_index.sh --rebuild    # Fresh index; retains earlier outputs.
./scripts/02_index.sh dry_run=true # Inspect defaults without building an index.
```

Automatic selection scans completed attempts under the configured output root,
restricted to the current `run.name`, then checks their existing compatibility
and integrity contracts. Evaluation also matches the requested retrieval depth.
Failed, incompatible, and corrupt candidates are excluded. Explicit `paths.index`
or `paths.run` overrides bypass automatic selection and retain normal validation.
No prerequisite stage is launched automatically. Index reuse checks recorded
identity; if local dataset files change in place, use `--rebuild` because reuse
does not reread the corpus to detect changes. For an explicit index destination,
existing destinations still fail rather than being overwritten.

Each executed stage retains Hydra's fresh timestamped output and provenance.
The stage scripts support single runs; use the CLI below for Hydra sweeps.
The bare CLI retains its explicit artifact-path contracts.

```sh
uv run scientific-tot dry_run=true
uv run scientific-tot stage=prepare dataset.root=/your/restored/data
uv run scientific-tot stage=index dataset.root=/your/restored/data
uv run scientific-tot dry_run=true experiment=dense_passages launcher=local_gpu
TOT_DATA_ROOT=/your/existing/data uv run scientific-tot dry_run=true experiment=bm25_fulltext
uv run scientific-tot --multirun dry_run=true experiment=bm25_fulltext,dense_passages split=train,test
```

[Configuration guide](docs/configuration.md) explains all YAML groups, overrides,
local/DGX03 data paths, model parameter contracts, presets, validation, and
reproducibility. BM25 and dense `prepare`, `index`, `retrieve`, and `evaluate`
execute for text corpora. Dense passages use the configured LangChain splitter
and PyTerrier max/mean aggregation, producing canonical paper-level rankings.
Indexes and batch artifacts carry configuration/provenance manifests; evaluation
consumes saved runs without indexing or model inference. Standalone examples below
retain their defaults. See the [implementation report](docs/implementation-report.md)
for changed files, cache behavior, validation, and limitations. The
[profile audit](docs/configuration-audit.md) records every YAML consumer, parameter
binding, necessary configuration change, and unsupported operation.

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

Select another domain with `--domain NeurIPS`, or multiple domains with
`--domain ACL NeurIPS`. Small demos are explicit: add `--documents 250 --queries 10`.
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
pinned revisions, model-specific prompts, CPU setup and persistent library
caches. [Notebook](examples/dense_retrieval.ipynb) demonstrates encoding, FAISS
search, shared metrics and cache reuse. [Measured subset results](docs/dense-results.md)
verify the same canonical records as the BM25 baseline. The first dense run
downloads pretrained weights; subsequent runs reuse the cache.
No fine-tuning is performed.

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
Configure domains and optional subset limits in the configuration cell.
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
project environment includes all example dependencies and uses CPU
Torch on Linux. A model download is required only when running pretrained encoding.
Library consumers can still install the small core wheel and choose package extras;
this does not change the single repository environment. Code is MIT licensed;
dataset artifacts retain their own licenses.
