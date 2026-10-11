# Canonical dataset adapter

## Quick start

Run the complete offline example from the repository root:

```sh
uv sync --locked
uv run python examples/convert_dataset.py \
  --local-dir tests/fixtures/adapter --domain ACL \
  --output /tmp/scientific-tot-example
```

It emits two synthetic corpus records, one query, one validated positive qrel,
and three manifest/report files. These fixtures do not contain research data.
The example is also exercised by an offline test.

Use the same API on restored data or a Hub snapshot:

```python
from scientific_tot import Dataset, load_corpus, load_queries, load_qrels

# Change this to a downloaded snapshot or the upstream restorer's data root.
source = Dataset(local_dir="tests/fixtures/adapter")
corpus = load_corpus(dataset=source, domain="ACL", split="train+test", representation="fulltext")
queries = load_queries(dataset=source, domain="ACL", split="test", mode="visual-and-text")
qrels = load_qrels(dataset=source, domain="ACL", split="test", corpus=corpus)

for name, records in (("corpus", corpus), ("queries", queries), ("qrels", qrels)):
    records.write_jsonl(f"/tmp/scientific-tot-example/{name}.jsonl")
```

Each result is a `Records` sequence: `len(records)`, iteration, indexing, and
slicing return ordinary dictionaries. Every row is validated before a result is
returned. `records.report.to_dict()` exposes coverage; `records.provenance`
records source, revision, input filenames, and selection. `write_jsonl` writes
UTF-8 JSONL and `<filename>.manifest.json` with an output checksum, schema
version, record count, selection, and report. Files are individually replaced
atomically; the checksum detects a stale manifest after an interrupted pair of
writes. Existing output files are replaced. Conversion is eager and keeps
selected records in memory; select domains/splits appropriate to available RAM.
JSONL is the initial standardized export; Parquet corpus export is not implemented.

## Hub access and publication limitation

```python
from scientific_tot import Dataset, DatasetAdapter

source = Dataset(revision="3b764a4c6855925ec802a2560712c4f5cc96f980")
adapter = DatasetAdapter(source)
queries = adapter.load_queries(domain="ACL", split="test")
assert len(queries) == 110
```

This downloads only collection metadata and the selected ACL test JSONL (about
7 MB). The revision was rechecked during Phase 2 and still contains query
collections only. Corpus conversion from this revision raises
`MissingComponentError` because splits/extractions are absent. End-to-end
conversion currently requires appropriate local corpus artifacts. The adapter
will consume the source-verified files if they appear in a future revision; that
future publication has not been tested or assumed. No qrels are certified against
an imaginary corpus merely because a query supplies a relevant ID.

Without `revision`, the shared `Dataset` resolves main once and pins all reads
to that commit. Save `source.discover().revision`. Free functions also accept
`local_dir=` and `revision=` directly; do not mix those with `dataset=`. A shared
instance is preferable for multiple reads. Local mode never contacts the Hub.
`Dataset(cache_dir=...)` controls Hub storage. Previously downloaded files are
reused by `huggingface_hub.hf_hub_download` at their immutable revision. Offline
use takes `local_dir` pointing to the snapshot or restored data, not a remote
source that might fall back to the network.

## Selection and transfer granularity

`domain` accepts one exact published name or a sequence, such as
`domain=["ACL", "Physics"]`. Every requested query collection must exist;
unavailable selections are not silently ignored.

| Parameter | Semantics |
| --- | --- |
| Corpus `split="train"` / `"test"` | Select actual PDF paths in the authoritative split index |
| Corpus `split="train+test"` | Union of those lists, not the entire retrieval corpus |
| Corpus `split="all"` | Available domain corpus, including distractors; no split index required |
| Query `split="train+test"` or `"all"` | Both published query collections |
| `mode="visual-and-text"` | Published mode; directory is `visual_and_text` |
| `mode="visual-only"` | Source-supported; requires an actual matching collection |
| `include_metadata=False` | Avoid the optional PDF-index download/enrichment; preserve paper metadata |
| `include_figures=False` | Omit figure projection and image-availability reporting |
| `archives=False` | Require already-restored original files; refuse TAR downloads/reads |

`split="all"` can report missing extractions against `metadata.parquet` when
that index is available. Without it, coverage is limited to discovered
extraction artifacts; it cannot certify the universe of PDFs. `train+test`
coverage always uses the complete selected PDF split lists. Report counts
include unidentified extraction directories separately when their source path
cannot be resolved, since their membership cannot be established safely.

For transfer-only preparation, select components explicitly:

```python
adapter.download_components(
    domain="ACL",
    split="test",
    components=["queries"],
)
# If published in the selected revision:
# adapter.download_components(domain="ACL", components=["metadata", "splits"])
# adapter.download_components(domain="ACL", components=["preprocessed"])
```

Valid components are `metadata`, `splits`, `queries`, `preprocessed`, `clues`,
and `pdfs`. There is no default bulk selection. Corpus conversion reads
preprocessed content, optional PDF metadata, and the requested split definition;
it never downloads PDF or clue components. Metadata uses the global Parquet
index, then filters rows locally. Query transfers are domain/split/mode selective.
Preprocessed TARs have no per-paper member manifest in the verified contract:
the smallest safe download unit is a whole domain shard. All selected-domain
shards must be indexed because one paper may span shards. Splits filter corpus
records after that indexing, not archive transfer bytes. There is no claim of
remote byte-range or per-paper archive downloads. Full-format preprocessing
shards may themselves bundle PDFs and images; downloading such a shard transfers
all of those bytes even though conversion reads only selected text/metadata.
`load_metadata(domain="ACL", dataset=source)` accesses only the PDF index and
never downloads these archives (the `parquet` extra is required).

The archive reader indexes uncompressed TAR headers and seeks only requested
JSON/text members; images are not decoded or read into memory. TAR paths,
symlinks, special members, and duplicate logical files are checked. No archive
is extracted to disk. Already-restored output from the generation repository's
existing restoration utilities is supported unchanged; this adapter does not
copy their extraction/checksum implementation or import their runtime. Use those
utilities separately if physical PDF/image restoration is needed. The upstream
bulk downloader's default is not a pinned or lightweight download workflow.

## Representations and missing data

`representation="fulltext"` reads original `markdown.md` (or its recorded
`markdown_relpath`) and checks `markdown_sha256` when present. It never indexes
prompts, generated clues, or interpretations.

`representation="title_abstract"` joins available title and abstract with a blank
line. Either original field can be present alone; the other stays `null` and a
`partial_title_abstract` issue is reported. Missing both triggers the content
policy. Values come from original PDF-index metadata, then original paper
metadata if present. The verified PDF-index writer includes **title but no
abstract column**, so title-only results are expected. Explicit original abstract
fields in locally enriched metadata are retained; the adapter does not extract,
summarize, infer, or regenerate an abstract. Access to a bibliography field does
not authorize changing paper IDs.

Full extractions use authoritative `paper.json` identity and source paths. Lean
extractions preserve the existing directory name as `doc_id`, following the
source reader. To filter a lean extraction by split or join its PDF metadata,
that existing ID must match exactly one source-verified normal or collision
path candidate from the split/PDF index. Ambiguity is an error, not first-match
selection. No canonical ID is minted. Lean image names are retained as paths,
with `figure_id=null` when the original figure identity was discarded.

The default `missing="error"` raises `IncompleteCorpusError` with `.report` for
missing extractions/text, empty text, failed markdown checksums, or unresolved
source paths. `missing="skip"` explicitly excludes those records and retains the
report on the returned collection. It never inserts empty-text placeholders.
Malformed JSON, ambiguous identities, duplicate records, unsafe paths, and
contradictory source/split metadata always raise, even under `skip`.

Missing figures and bibliographic metadata are reported without excluding a
usable text record. Available licensing, DOI, and year fields are projected and
retained in original metadata. Missing metadata is null, not guessed. The corpus
reader does not require `_SUCCESS`, because restored/lean data can legitimately
omit it; it assesses requested content instead.

## Independent relevance validation

Qrels require an actual `corpus` result. No implicit corpus download happens in
`load_qrels`. If queries have already been loaded:

```python
from scientific_tot import validate_qrels

qrels = validate_qrels(queries, corpus)
```

Every positive document ID must exist in the corpus and belong to the query's
domain. Explicitly different source repositories or revisions are rejected.
`UnresolvedRelevanceError.report` lists all unresolved positives. There is no
option that silently drops unresolved qrels, even if corpus loading used
`missing="skip"`. Choosing a train-only corpus for test queries usually fails.
Local revision annotations are caller assertions; no provenance is independently
certified for a local directory.

Each member of upstream `relevant_ids` produces relevance `1`. `negative_ids`
and query-quality judgments are retained on queries but not converted into
judged negatives or graded relevance. An unlisted document is not a judged
negative. The result does not certify that a binary source judgment is correct;
it certifies that its target exists in the selected corpus.

## Standardized JSONL records

| Kind | Fields intended for retrieval frameworks |
| --- | --- |
| Corpus | `doc_id`, `text`, `title`, `abstract`, `domain`, `split`, `representation` |
| Queries | `query_id`, `query_key`, `text`, `domain`, `split`, `mode`, `collection` |
| Qrels | `query_id`, `query_key`, `doc_id`, `relevance`, `domain`, `split`, `mode` |

Corpus also includes `pdf_document_id` (separate namespace),
`source_pdf_relpath`, `license`, `doi`, `year`, `metadata.paper`, `metadata.pdf`,
`figures`, and artifact provenance. Figure entries carry `figure_id`, zero-based
`page_idx`, original `metadata`, and an `image` locator: dataset-relative `path`,
`available`, and, when known, `local_path`, `archive`, and `member`. For archived
images, `local_path` points to the TAR and `member` identifies its image; it is
not a decoded image path. An absent locator or absolute historical path is
reported unavailable; historical paths are retained only as provenance.

Canonical query IDs can repeat across domains. `query_id` is preserved exactly;
`query_key` is an explicit JSON encoding of
`[domain, collection, mode_directory, query_id]` for globally unique lookup. It
is identical in queries and qrels. A framework accepting only one query-ID field
should use `query_key`; no generation-pipeline knowledge is needed:

```python
framework_corpus = {row["doc_id"]: row["text"] for row in corpus}
framework_queries = {row["query_key"]: row["text"] for row in queries}
framework_qrels = [(r["query_key"], r["doc_id"], r["relevance"]) for r in qrels]
```

These are data mappings only, not a ranking or evaluation implementation.
Raw query metadata and unknown top-level query fields remain available but are
excluded from the `text` field used for retrieval.
