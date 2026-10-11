# scientific-tot-retrieval

A small Python interface for discovering and selectively reading the
[scientific tip-of-the-tongue dataset](https://huggingface.co/datasets/kasys/open-source-scientific-documents).
The published dataset is the source of truth. The
[generation repository](https://github.com/stanfris/multimodal-tip-of-tongue-retrieval-for-scientific-papers)
creates and packages the artifacts; this package does not import or run that pipeline.

**Publication status checked 2026-10-11:** Hub revision
`3b764a4c6855925ec802a2560712c4f5cc96f980` contains query collections only.
There is no README, `metadata.parquet`, `splits/`, PDF, clue, or preprocessed
archive at that revision. The source code documents those additional formats,
but their publication and corpus coverage cannot currently be verified.
See [the data contract](docs/dataset-schema.md) for evidence and limitations.

## Install

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).
From a checkout of this repository:

```sh
uv sync --locked
uv run python examples/inspect_dataset.py
# Optional Arrow reader for a locally available or future published PDF index:
uv sync --locked --extra parquet
```

The core depends only on `huggingface_hub`. No model, GPU library, or retrieval
framework is required. `datasets` and integration extras are deferred until an
actual conversion or adapter requires them.

## Discover and select

```python
from scientific_tot import Dataset

dataset = Dataset(revision="3b764a4c6855925ec802a2560712c4f5cc96f980")
inventory = dataset.discover()  # API listing only; no artifact downloads
print(inventory.revision, inventory.components)
for collection in inventory.collections:
    print(collection.key, collection.queries_path)

collection = next(
    c
    for c in inventory.collections
    if c.domain == "ACL" and c.collection == "query_generation_test"
)
metadata = dataset.read_json(collection.metadata_path)  # about 11 KB
# Explicit opt-in: downloads this entire JSONL, then iterates its rows locally.
for query in dataset.iter_queries(collection):
    print(query["query_id"], query["query"], query["relevant_ids"])
```

Omitting `revision` resolves the current Hub commit on first discovery and pins
all subsequent reads on that instance. Save `inventory.revision` with experiment
outputs. Explicit revisions must be full 40-character commit SHAs, not mutable
branches/tags. Discovery results are cached for the lifetime of the instance.
Create another instance to refresh a changed local directory.

Query files can be large (the inspected training files are about 69–197 MB).
Reading one is always explicit. No method downloads or extracts archives.
`read_json` refuses metadata over 2 MB unless its limit is explicitly increased.

## Offline and local data

```python
dataset = Dataset(local_dir="/path/to/downloaded-or-restored/data")
print(dataset.discover())  # never calls the Hub
split = dataset.load_split()  # only if splits/pdf_dataset_split.json exists
paper = dataset.load_paper("preprocessed/ACL/ACL_2022.example-1.1")
# Full extractions require paper.json and nonempty markdown.md.
```

Point `local_dir` at the actual snapshot or restored data root, including a Hub
cache snapshot directory if desired. Missing components raise
`MissingComponentError` with the requested path. Local files have unverified
provenance; a supplied revision is the caller's assertion, not a checksum check.
Metadata and extra fields are retained. Clues in query metadata remain provenance
and must not enter a standard corpus index. `Paper.markdown` reads only extracted
original content. Lean markdown/images-only directories require an authoritative
ID mapping and are not loaded as full papers by this initial API.

For a separately obtained full extraction, resolve a packaged PDF ID with:

```python
from scientific_tot.data.identifiers import document_id_for_paper

rows = dataset.load_pdf_metadata().to_pylist()  # requires the parquet extra
pdf_id = document_id_for_paper(paper.metadata, rows)
```

The join uses exact source paths and fails for missing or ambiguous mappings.
It never regenerates IDs from titles or normalized filenames.

## Development

```sh
uv sync --locked --extra parquet
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
# Optional live inventory test (network required, no artifact downloads):
uv run pytest -m live
```

Default tests use tiny synthetic fixtures and block network connections.
[Architecture](docs/architecture.md) defines module boundaries and error policy;
[the integration plan](docs/integration-plan.md) describes corpus conversion,
standard IR exports, and later thin framework adapters. None of those retrieval
backends or exports is implemented in this initial foundation.

Code is MIT licensed. This does not license the dataset: source documents and
other artifacts require their own provenance/license checks.
