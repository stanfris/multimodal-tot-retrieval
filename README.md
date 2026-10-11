# scientific-tot-retrieval

Canonical dataset adapters for the
[scientific tip-of-the-tongue dataset](https://huggingface.co/datasets/kasys/open-source-scientific-documents).
Load selected original-content corpus records, queries, and validated relevance
judgments, then export standard JSONL for existing retrieval frameworks.

The published dataset is the source of truth. The
[generation repository](https://github.com/stanfris/multimodal-tip-of-tongue-retrieval-for-scientific-papers)
creates the artifacts; this package does not depend on that pipeline at runtime
and implements no ranking, embedding generation, or evaluation metrics.

**Publication status rechecked 2026-10-11:** revision
`3b764a4c6855925ec802a2560712c4f5cc96f980` contains query collections only.
Corpus archives, `metadata.parquet`, and `splits/` are absent. Queries can be
loaded from the Hub now; full corpus/qrel conversion requires appropriate local
artifacts or a future publication containing them. Missing data is reported,
never fabricated. [Evidence and data contract](docs/dataset-schema.md).

## Install and run

Python 3.11+ and [uv](https://docs.astral.sh/uv/) are required for development:

```sh
uv sync --locked
# Optional reader for metadata.parquet:
uv sync --locked --extra parquet
# Fully offline two-document synthetic example:
uv run python examples/convert_dataset.py \
  --local-dir tests/fixtures/adapter --domain ACL --output /tmp/scientific-tot-example
```

From Python, use a shared source to keep all reads on one revision:

```python
from scientific_tot import Dataset, load_corpus, load_queries, load_qrels

source = Dataset(local_dir="tests/fixtures/adapter")  # replace with your data root
corpus = load_corpus(dataset=source, domain="ACL", split="train+test", representation="fulltext")
queries = load_queries(dataset=source, domain="ACL", split="test", mode="visual-and-text")
qrels = load_qrels(dataset=source, domain="ACL", split="test", corpus=corpus)

corpus.write_jsonl("/tmp/corpus.jsonl")
queries.write_jsonl("/tmp/queries.jsonl")
qrels.write_jsonl("/tmp/qrels.jsonl")
print(corpus.report.to_dict())
```

`title_abstract` is also supported, using only available original fields. Absent
abstracts stay null. `missing="error"` is the default; `missing="skip"` explicitly
excludes unusable records with a report. Unresolved relevance targets always
raise. Figure metadata and file/archive paths are exposed without loading images.

[API documentation](docs/api.md) covers component downloads, metadata-only reads,
Hub revisions, archive granularity, lean extractions, missing-data policy, and
record schemas. [Architecture](docs/architecture.md) records design decisions.

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
uv sync --locked --extra parquet
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
# Optional live check: metadata and one ~7 MB ACL test query file; no archives:
uv run pytest -m live
```

Default tests block network connections and use small synthetic fixtures.
[Verification](docs/verification.md) records executed checks and limitations.
[Integration plan](docs/integration-plan.md) tracks remaining work. The core
requires only `huggingface_hub`; Arrow is optional. No GPU libraries or models
are required. Code is MIT licensed; dataset artifacts retain their own licenses.
