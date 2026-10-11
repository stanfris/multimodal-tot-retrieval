# Remaining integration phases

1. **Publication and coverage.** Reinspect a pinned Hub commit when corpus
   metadata, split definitions, and extraction artifacts become available.
   Verify real archive layouts and checksums with upstream restoration tools.
   Build explicit paper/PDF and lean-artifact mappings, retaining IDs and
   rejecting collisions. Produce missing extraction, figure, and qrel reports.
   Do not reconstruct missing splits from query labels.
2. **Canonical corpus conversion.** Add incremental original-text corpus
   construction with explicit inclusion policies, source-path provenance, and
   manifests recording revisions, selected files, checksums, converter version,
   and exclusions. Support figure/page records only after verifying available
   metadata. Introduce Arrow or Hugging Face Datasets formats as needed.
3. **Standard exports.** Add corpus/query JSONL and TREC qrels adapters with
   declared binary relevance mapping. Preserve original IDs alongside any
   necessary reversible query namespace; verify referential integrity before
   publishing outputs. Keep quality judgments distinct from relevance labels.
4. **Thin integrations.** Implement one maintained framework adapter/example
   at a time (candidate interfaces include PyTerrier or BEIR-style datasets),
   checking their current APIs and dependencies at implementation time. Let
   those libraries provide ranking and evaluation. Add optional extras only
   for integrations that exist and test their data boundary with small fixtures.
5. **Multimodal integrations.** Provide figure/page inputs and established
   framework examples once authoritative page/figure mapping and missing-data
   policy are settled. Model inference, ColBERT/ColPali internals, vector
   search, and custom evaluation metrics remain outside this package.

No framework installation, model download, full-corpus transfer, retrieval
backend, or export implementation is part of the initial release.
