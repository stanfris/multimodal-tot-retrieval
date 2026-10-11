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

## Publication and further work

1. **Live corpus validation.** Reinspect a pinned Hub commit when metadata,
   splits, and extractions are published. Verify real archive layouts, source
   joins, and coverage. Current live availability is still query-only. Synthetic
   archive/index tests are not proof of live corpus completeness.
2. **Additional exports and scale.** Add explicit Arrow/Parquet corpus schemas,
   TREC-compatible export conventions, and a streaming conversion API if needed.
   Preserve original IDs and reversible query scope mappings. Any transactional
   bundle format should validate all files before publication.
3. **Thin framework examples.** Implement one maintained library adapter at a
   time, checking its current data interface. Let that library supply ranking
   and evaluation; add optional dependency groups only for adapters that exist.
4. **Multimodal/page integration.** Provide established-library examples once
   actual figure/page metadata and PDF availability can be verified. Null IDs,
   missing images, and page boundaries require explicit policies. PDF rendering,
   model inference, vector search, and custom metrics remain outside the adapter.

The adapter deliberately does not package the generation pipeline, regenerate
metadata, implement ranking, generate embeddings, or download models.
