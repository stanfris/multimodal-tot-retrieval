# Actual ACL BM25 results

Executed on **dgx03, 2026-10-11**, using the existing restored dataset at
`/home/sfris/Documents/multimodal_tot/multimodal-tip-of-tongue-for-scientific-papers/data`.
These are real original extractions and published queries, not synthetic fixtures.
[Machine-readable evidence](benchmarks/pyterrier-acl.json) records metrics copied
from PyTerrier Experiment, library versions, text/run hashes, and coverage counts.

| Corpus | Representation | Documents | Queries | MRR@10 | Recall@10 | Recall@100 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Demo subset | Fulltext | 250 | 10 | 0.181111 | 0.500000 | 0.900000 |
| Demo subset | Title/abstract | 250 | 10 | 0.730000 | 1.000000 | 1.000000 |
| All available ACL extractions | Fulltext | 20,845 | 95 | 0.010526 | 0.021053 | 0.115789 |
| Available bibliography coverage | Title/abstract | 20,843 | 95 | 0.441704 | 0.589474 | 0.810526 |

All runs use Terrier BM25 defaults, English tokenization, Stopwords and
PorterStemmer, with cutoff 100. The installed versions are PyTerrier 1.1.2,
Terrier 5.11, pandas 3.0.6 and ir_measures 0.4.3 on Python 3.11.16. Metrics use
`pt.Experiment` with `RR@10`, `R@10`, and `R@100`; no custom calculations,
query rewriting, parameter tuning, or embeddings were applied.

The small subset takes the first ten queries and all positive papers, filling
with sorted IDs from the train+test corpus. The original split declares 2,310 ACL
papers. The lean extraction directory contains additional papers whose split
membership cannot be established from those split assignments; the parent report
records 18,535 unresolved source mappings. `missing="skip"` explicitly omits
those from the split-restricted demo. Demo results should not be used as full
benchmark estimates.

Full-corpus execution uses `--corpus-split all`, loading all 20,845 available ACL
Markdown extractions without requiring split membership for distractor documents.
The original test PDF split contains 110 papers; **95 published test queries**
are present for `visual-and-text`. No queries were fabricated for the other papers.
All 95 query relevance targets resolve in both indexed corpora. Each indexed
document ID was checked against its exact canonical ID, including metadata
capacity checks; every selected query is included in evaluation.

The bibliography experiment explicitly selects literal original title/Abstract
Markdown sections. The original metadata has null titles and no abstract column.
Two documents have neither a recognized title nor a delimited Abstract section
and are excluded with a report; neither is a positive target. Another 390 records
have only one usable field and are retained with `partial_title_abstract` issues.
In the demo subset, four records are partial and none are excluded. Fulltext and
bibliography therefore use different full-corpus coverage; their scores are not
a perfectly matched-document comparison. The CLI refused a combined full-corpus
comparison before indexing and the two reported experiments were explicitly run
independently. No abstract was summarized or invented.

The poor fulltext score is the observed baseline for unmodified verbose queries
and raw extracted Markdown. These runs demonstrate reproducibility and adapter
compatibility; they do not establish that BM25 has been tuned for this task.

## Reproduce and inspect

[Guide and commands](pyterrier.md) reproduce the subset and full-corpus runs.
Choose fresh output directories. Results copied into the local checkout are:

- `results/phase3-acl-small/{fulltext,title_abstract}/run.trec`
- `results/phase3-acl-full/{fulltext,title_abstract}/run.trec`
- The corresponding `metrics.csv`, `per-query.csv`, `qrels.tsv`, query mappings,
  selected document IDs and manifests.
- `results/phase3-acl-notebook-final-executed.ipynb`, an executed copy of the
  [source notebook](../examples/pyterrier_bm25.ipynb).

Results and indexes are ignored by Git. The source and compact benchmark evidence
are synchronized locally and on dgx03. Full Terrier indexes remain on dgx03;
local copied results exclude indexes. Restored source data was not modified.

Offline fixture tests and the real Java test both passed on dgx03. The Java test
also checks an ID longer than Terrier's default metadata capacity and a query
with no results. The notebook executes the entire conversion/indexing/evaluation
workflow on the real 250-document subset. Installation of the built wheel with
the `pyterrier` extra and the documented canonical conversion API was also
verified in a clean local environment outside the repository.
