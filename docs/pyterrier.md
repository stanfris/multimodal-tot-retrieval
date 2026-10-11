# BM25 with PyTerrier

The optional integration converts canonical records to PyTerrier's `docno/text`,
`qid/query`, and `qid/docno/label` structures. Terrier indexes and ranks; PyTerrier
Experiment and ir_measures calculate all metrics. The core adapter does not import
PyTerrier, pandas, or Java.

## Installation

```sh
uv sync --locked --extra pyterrier --extra notebook
# Or install the package and optional integration:
pip install '.[pyterrier,notebook]'
```

Use a JDK (Java 17 recommended) and set `JAVA_HOME` to its installation directory.
The first explicit experiment starts Java and downloads Terrier's JAR if it is
not cached. No embeddings, models, PDFs, or image decoding are needed. The locked
Python environment uses PyTerrier 1.1.2 with its `java` extra, and experiments pin
Terrier 5.11. JDK setup is independent of the dataset package.

On dgx03, the system installation was only Java 8; the verified isolated setup is:

```sh
UV_PROJECT_ENVIRONMENT=.venv-pyterrier uv sync --locked --extra pyterrier --extra notebook
uv pip install --python .venv-pyterrier/bin/python jdk4py==25.0.2.1
export JAVA_HOME=$(.venv-pyterrier/bin/python -c 'import jdk4py; print(jdk4py.JAVA_HOME)')
```

Use `.venv-pyterrier/bin/python` for the examples in that environment. Subsequent
`uv sync` commands need `--inexact` to retain this separately installed runtime.

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
  --representation both --bibliography-from-markdown --missing skip \
  --output results/acl-small
```

The demo chooses the first ten canonical queries, includes all their positive
papers, then fills the corpus with sorted canonical IDs. It is a reproducibility
smoke test, not a randomly sampled benchmark. Limits apply after canonical
loading/validation, so selecting 250 documents still inventories the restored
root and loads the selected split before making the demonstration subset.

The real dgx03 ACL metadata has no abstract column and null title values. Its
Markdown extractions contain original paper titles and Abstract sections.
`--bibliography-from-markdown` explicitly selects a leading H1 title and a single
exact `Abstract` heading's text up to the next heading. It creates no summary or
invented metadata. Original supplied fields take precedence; unrecognized
sections stay null. Character offsets and source locators record the projection.
Title-only or abstract-only records are retained with a `partial_title_abstract`
issue; records with neither follow `--missing error|skip`. The original raw paper
metadata is retained. The core `title_abstract` loader continues to use original
bibliographic fields unless this explicit projection is requested.

Without this flag, the CLI loads original title/abstract metadata through the
adapter. When comparing both representations, their corpus ID sets must match.
A skipped positive target always fails relevance validation. Reports in each
manifest record omissions and partial bibliography coverage.

## Full-corpus execution

Remove both subset limits and select `all` to index every available extraction.
For the actual ACL corpus, bibliography coverage differs, so run independently:

```sh
uv run python examples/pyterrier_bm25.py \
  --local-dir /path/to/restored/data --domain ACL --corpus-split all \
  --query-split test --mode visual-and-text --representation fulltext \
  --missing skip --output results/acl-full
uv run python examples/pyterrier_bm25.py \
  --local-dir /path/to/restored/data --domain ACL --corpus-split all \
  --query-split test --mode visual-and-text --representation title_abstract \
  --bibliography-from-markdown --missing skip --output results/acl-full
```

`train+test` restricts documents to published split assignments; `all` includes
available papers outside those assignments. Query split and corpus split are
independent. Choose additional domains with `--domain ACL NeurIPS` as available.
If a comparison has unequal coverage, select a covered subset explicitly or run
one representation at a time; no silent intersection or qrel removal occurs.
For Hub data, omit `--local-dir` and supply `--revision` with a full commit SHA.
The current documented Hub snapshot contains queries only, so a full experiment
requires restored local corpus/split artifacts until those components are published.
The adapter reads available original extraction text; its missing-content report
is not a claim that all source PDFs have successful extractions.

## Notebook

Open [pyterrier_bm25.ipynb](../examples/pyterrier_bm25.ipynb), register the environment:

```sh
uv run python -m ipykernel install --sys-prefix --name scientific-tot-pyterrier
export TOT_DATA_ROOT=/path/to/restored/data
export TOT_OUTPUT_DIR=/path/to/new/notebook-results
```

Select `scientific-tot-pyterrier`. Execute all cells. The configuration cell
sets corpus/query limits and split; use `DOCUMENTS = QUERIES = None` and
`CORPUS_SPLIT = "all"` for the full available corpus. The notebook converts,
validates, indexes, evaluates, and reloads the standard TREC output.

## Outputs and tests

Each representation writes `run.trec`, `metrics.csv`, `per-query.csv`,
`topics.jsonl`, `qrels.tsv`, `query-ids.jsonl`, `selected-documents.json` (CLI),
`manifest.json`, and the Terrier index. Runs are written with
`pt.io.write_results(..., format="trec")`. `qrels.tsv` is the named-column
PyTerrier table, not the four-column TREC qrels format.

`RR@10` in PyTerrier's aggregate output is MRR@10; `R@10` and `R@100` are
Recall@10 and Recall@100. Every selected query is evaluated, including queries
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
