# Integration status and remaining phases

## Implemented in Phase 2

- Local/Hub canonical corpus, query, and independently validated positive qrel
  records, with domain/split/component selection and pinned incremental downloads.
- Fulltext and available title/abstract representations; exact source joins,
  lean identity lookup, metadata/licensing preservation, and coverage reports.
- Restored-file and read-only TAR support, including papers spanning shards;
  figure metadata and locators without decoding image bytes.
- Standard JSONL records with export hashes, reports, and provenance manifests.
- Executable offline example and a separately selected live query integration test.

## Implemented in Phase 3

- Optional PyTerrier/Java integration and canonical pandas/iterdict conversion.
- Standard Terrier BM25 indexes and TREC runs, with exact indexed ID validation.
- Established Experiment/ir_measures MRR@10 and Recall@10/100 evaluation.
- CLI and executable notebook for original fulltext and available bibliography;
  explicit literal Markdown title/Abstract projection with source spans.
- Offline conversion fixtures, optional real Java test, and actual dgx03 corpus
  experiments. See [PyTerrier guide](pyterrier.md) and [results](pyterrier-results.md).

## Implemented in Phase 4

- Optional Sentence Transformers/FAISS dense tutorial, with a pinned pretrained
  default, configurable Hub models/revisions and explicit encoding instructions.
- Canonical corpus/query/qrel conversion and subset selection shared with BM25;
  exact saved-baseline identity checks before encoding.
- Batched library encoding, NumPy embedding persistence and FAISS index I/O,
  verified cache reuse, and shared PyTerrier/ir_measures evaluation.
- CLI/notebook, CPU/GPU setup documentation, offline fixtures, an opt-in real
  pretrained model test, and actual dgx03 subset runs. See
  [dense tutorial](dense-retrieval.md) and [results](dense-results.md).

## Publication and further work

1. **Live corpus validation.** Reinspect a pinned Hub commit when metadata,
   splits, and extractions are published. Verify real archive layouts, source
   joins, and coverage. Current live availability is still query-only. Synthetic
   archive/index tests are not proof of live corpus completeness.
2. **Additional exports and scale.** Add explicit Arrow/Parquet corpus schemas,
   TREC-compatible export conventions, and a streaming conversion API if needed.
   Preserve original IDs and reversible query scope mappings. Any transactional
   bundle format should validate all files before publication.
3. **Further framework examples.** PyTerrier is implemented and verified. Add
   further maintained adapters only when required, using their ranking/evaluation
   implementations and optional dependency groups.
4. **Multimodal/page integration.** Provide established-library examples once
   actual figure/page metadata and PDF availability can be verified. Null IDs,
   missing images, and page boundaries require explicit policies. PDF rendering,
   model inference, vector search, and custom metrics remain outside the adapter.

The adapter deliberately does not package the generation pipeline, regenerate
metadata, implement ranking, generate embeddings, or download models.
