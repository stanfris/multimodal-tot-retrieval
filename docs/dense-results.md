# Dense retrieval verification on actual data

Executed on **dgx03, 2026-10-11**, using the same 250-document, ten-query ACL
subset and canonical qrels as the saved Phase 3 BM25 baseline. Before encoding,
`--baseline-dir` verified ordered document IDs, exact corpus text hash, queries,
qrels and query-ID mappings. Exported topic, qrel and mapping files also match
BM25 byte-for-byte. [Machine-readable evidence](benchmarks/dense-acl.json).

| Representation | Documents | Queries | MRR@10 | Recall@10 | Recall@100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fulltext | 250 | 10 | 0.633333 | 0.800000 | 1.000000 |
| Title/abstract | 250 | 10 | 0.753333 | 0.900000 | 1.000000 |

The pretrained model is `intfloat/e5-small-v2` at commit
`ffb93f3bd4047442299a41ebb6fa998a38507c52`, with library-applied `query: ` and
`passage: ` prompts, normalized float32 384-dimensional embeddings, batch size
16, and CPU encoding. The effective context is 512 tokens, including prompts;
fulltext inputs are truncated by the model tokenizer. No chunking, fine-tuning,
query rewriting or parameter selection based on results was performed.
FAISS `IndexFlatIP` supplied top-100 search with one CPU search thread. Shared
PyTerrier Experiment/ir_measures implementations supplied every metric.

Installed versions: Sentence Transformers 5.7.0, Transformers 5.19.0,
Torch 2.14.1+cpu, FAISS CPU 1.15.1, NumPy 2.4.6, PyTerrier 1.1.2,
and ir_measures 0.4.3. No Java initialization or GPU is needed for dense execution.
This restored local dataset has no independently verified dataset commit; manifests
record `local/unrecorded`, exact corpus hashes and the matched baseline manifest.
The source data was not modified. Bibliography uses the same literal original
Markdown projection as Phase 3; four partial records are retained, none excluded.

The subset was chosen for reproducibility: the first ten canonical queries,
all their positives, then sorted document IDs as distractors. It is not a random
sample or full benchmark. These scores verify the tutorial's execution and do not
establish that this default model is superior generally or optimal for scientific
paper retrieval. [BM25 subset and full-corpus results](pyterrier-results.md) are
reported separately; full-corpus dense execution and GPU execution were not measured.

## Cache and testing evidence

The fulltext run was repeated with `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1`. Both embedding caches were hits; the persisted FAISS
index was loaded, no model was instantiated, and the resulting TREC run was
**byte-for-byte identical**. The bibliography run reused the same query cache
while encoding its different corpus representation. The executed notebook checks
its cached rerun's metrics and TREC bytes as well.

79 offline tests passed locally and on dgx03, including real FAISS search and
standard evaluation with only neural encoding stubbed. Tests verify required
prompts, immutable revision resolution, exact baseline identity, query-only cache
invalidation, and rejection of corrupted index files before library loading.
The explicitly selected pretrained-model test passed on dgx03, as did the existing
Terrier JVM integration test after evaluation was factored into a shared utility.
A clean core-only wheel installation loaded the canonical fixture with no Torch,
Sentence Transformers, FAISS, NumPy or PyTerrier installed.

The documented MiniLM substitution command also completed on the two-document
fixture, resolving `main` to an immutable model revision.

The [source notebook](../examples/dense_retrieval.ipynb) executed all cells on the
actual subset. Saved artifacts are available in the local checkout and on dgx03:

- `results/phase4-acl-small/{fulltext,title_abstract}/`: TREC runs, metrics, topics,
  qrels, mappings, selected document IDs and complete manifests.
- `results/phase4-acl-cached/fulltext/`: offline cached-run evidence.
- `results/phase4-dense-notebook-executed.ipynb`: executed tutorial.
- dgx03 `results/phase4-dense-cache/`: NumPy embeddings, FAISS indexes, immutable
  model snapshot cache and cache manifests.

Results/caches are ignored by Git. Code, tutorials and compact benchmark evidence
are synchronized locally and on dgx03. Follow the [reproduction guide](dense-retrieval.md)
to build your own cache and substitute another model.
