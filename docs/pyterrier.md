# BM25 with PyTerrier

The optional integration converts canonical records to PyTerrier's `docno/text`,
`qid/query`, and `qid/docno/label` structures. Terrier indexes and ranks; PyTerrier
Experiment and ir_measures calculate all metrics. The core adapter does not import
PyTerrier, pandas, or Java.

## Installation

Run from the repository root using the same environment as dense retrieval and notebooks:

```sh
uv sync --locked
```

This installs PyTerrier's Java bindings and the pinned `jdk4py==25.0.2.1`
runtime in `.venv`. BM25 selects that runtime automatically before starting
Java, so system Java 8 and a stale `JAVA_HOME` do not affect the example.
There is no separate `.venv-pyterrier`, manual JDK installation, or `--inexact`
step. The first experiment downloads Terrier's JAR if it is not cached.
Experiments pin Terrier 5.11. Choose fresh experiment output directories.

The README's two-document fixture works directly with fulltext. Bibliographic
fields are not required or loaded for retrieval. `--missing skip` is the CLI
spelling; `missing="skip"` belongs to the Python API. Skipping only excludes
records with unavailable fulltext, and every positive target must still resolve.

## Python API

This example runs from the repository root against the two-document fixture:

```python
from scientific_tot import Dataset, DatasetAdapter, load_qrels
from scientific_tot.integrations.pyterrier import to_pyterrier, bm25_experiment

source = Dataset(local_dir="tests/fixtures/adapter")
adapter = DatasetAdapter(source)
corpus = adapter.load_corpus(domain="ACL", split="train+test", representation="fulltext")
queries = adapter.load_queries(domain="ACL", split="test", mode="visual-and-text")
qrels = load_qrels(dataset=source, domain="ACL", split="test", corpus=corpus)
inputs = to_pyterrier(corpus, queries, qrels)  # optional independent qrels
print(inputs.topics)
print(inputs.qrels)
metrics = bm25_experiment(corpus, queries, "/tmp/tot-bm25-fixture")
print(metrics)
```

Choose a fresh output directory for every experiment. `to_pyterrier` performs no
Java initialization. Its document iterator can also be supplied directly to
`pt.terrier.IterDictIndexer(...).index(iter(inputs.documents))`; topics and qrels
are ordinary pandas DataFrames accepted by existing PyTerrier components.
Qrels are validated against corpus records before indexing; independently supplied
qrels must agree with canonical query relevance targets. Missing targets raise.

Document `docno` values preserve canonical extraction paper IDs exactly. The
integration sizes Terrier's metadata field for the longest ID and checks both
the indexed document count and every stored ID. Unique query IDs are unchanged;
when collections reuse IDs, `qid` is a reversible percent-encoding of canonical
`query_key`. `query-ids.jsonl` retains original IDs and keys.

## Small real-data example

```sh
uv run python examples/pyterrier_bm25.py \
  --local-dir ../multimodal-tip-of-tongue-for-scientific-papers/data \
  --domain ACL --corpus-split train+test --query-split test \
  --mode visual-and-text --documents 250 --queries 10 \
  --missing skip \
  --output results/acl-small
```

The demo chooses the first ten canonical queries, includes all their positive
papers, then fills the corpus with sorted canonical IDs. It is a reproducibility
smoke test, not a randomly sampled benchmark. Limits apply after canonical
loading/validation, so selecting 250 documents still inventories the restored
root and loads the selected split before making the demonstration subset.

## Full-corpus execution

Fulltext, `--corpus-split all`, and unlimited documents/test queries are now the
default. `--documents 0 --queries 0` explicitly expresses these unlimited limits.
Small subsets require positive limits. No 20,000-document cap is imposed.
```sh
uv run python examples/pyterrier_bm25.py \
  --local-dir /path/to/restored/data --domain ACL --output results/acl-full
```

`train+test` restricts documents to published split assignments; `all` includes
available papers outside those assignments. Query split and corpus split are
independent. Select one domain with `--domain ACL`, `--domain Biology`,
`--domain Engineering`, `--domain Medicine`, or `--domain Physics`.
For Hub data, omit `--local-dir` and supply `--revision` with a full commit SHA.
The current documented Hub snapshot contains queries only, so a full experiment
requires restored local corpus/split artifacts until those components are published.
The adapter reads available original extraction text; its missing-content report
is not a claim that all source PDFs have successful extractions.

## Notebook

Open [pyterrier_bm25.ipynb](../examples/pyterrier_bm25.ipynb), register the shared environment:

```sh
uv run python -m ipykernel install --sys-prefix --name scientific-tot --display-name "Scientific ToT"
export TOT_DATA_ROOT=/path/to/restored/data
export TOT_OUTPUT_DIR=/path/to/new/notebook-results
```

Launch `uv run jupyter lab` and select **Scientific ToT** in both notebooks.
The notebook defaults to the restored sibling data directory, fulltext,
`DOCUMENTS = QUERIES = None`, and `CORPUS_SPLIT = "all"`, with a fresh output
directory. Set `TOT_DATA_ROOT` to another restored root or `tests/fixtures/adapter`
for a smoke test. Set `DOCUMENTS, QUERIES = 250, 10` for a limited demo.
Execute all cells. The configuration cell controls the domain and subset limits.
The notebook converts, validates, indexes, evaluates, and reloads the standard
TREC output.

## Outputs and tests

The fulltext experiment writes `run.trec`, `metrics.csv`, `per-query.csv`,
`topics.jsonl`, `qrels.tsv`, `query-ids.jsonl`, `selected-documents.json` (CLI),
`manifest.json`, and the Terrier index. Runs are written with
`pt.io.write_results(..., format="trec")`. `qrels.tsv` is the named-column
PyTerrier table, not the four-column TREC qrels format.

`RR@10` in PyTerrier's aggregate output is MRR@10; `R@1` and `R@10` are
Recall@1 and Recall@10, and `nDCG@10` is NDCG@10. Every selected query is evaluated, including queries
with no retrieved results. No manual ranking or metric calculations are used.
Manifests include library versions, text hashes, corpus selection, content
reports, and index ID verification. See [measured results](pyterrier-results.md).

```sh
uv run pytest                  # offline fixtures; Java test excluded
uv run pytest -m terrier       # real Terrier indexing and established metrics
```

The Java test covers long document IDs, TREC output, and a zero-result query.
The optional live dataset test remains separately selected with `-m live`.

Implementation follows the documented [Terrier API](https://pyterrier.readthedocs.io/en/stable/terrier/api.html),
[retrieval interface](https://pyterrier.readthedocs.io/en/latest/terrier/retrieval.html),
and [Experiment API](https://pyterrier.readthedocs.io/en/stable/experiments.html).
