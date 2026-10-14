# Hydra configuration

Hydra is the experiment-facing configuration authority. `scientific-tot` (or
`python -m scientific_tot.hydra_entry`) composes the single YAML tree shipped in
both the checkout and wheel. Use `uv sync --locked` for the repository environment.
Validation needs only core dependencies and neither discovers data nor loads models.
The [profile audit](configuration-audit.md) traces every YAML profile and records
all differences from the existing implementations.

BM25 and dense `prepare`, `index`, `retrieve`, and `evaluate` execute with
`fulltext` or `title_abstract`; dense also supports `passages`. `search`, `export`,
and other models remain unimplemented. Existing BM25/dense
tutorial functions and scripts remain executable with their established defaults.
ColBERT, Byaldi, figure retrieval, and page rendering are
configuration-only profiles; no new retrieval model is implemented.

## Ownership and consumers

| Group | Profiles | Runtime consumer / support |
| --- | --- | --- |
| `dataset` | `default`, `hub` | `configured_adapter` → existing `Dataset` / `DatasetAdapter`; configured Hub downloads supported; artifact relocation unsupported |
| `split` | `all`, `train`, `test`, `train+test` | `corpus_arguments` / `query_arguments` → adapter selection |
| `representation` | `fulltext`, `title_abstract` | `corpus_arguments` → `DatasetAdapter.load_corpus`; original content is preserved |
| `representation` | `passages`, `figures`, `pages` | Dense passages execute through LangChain and PyTerrier aggregation; figures/pages remain configuration-only |
| `model` | `bm25`, `dense` | `bm25_arguments` / `dense_arguments` → existing optional integration functions when passed `settings=cfg` |
| `model` | `colbert`, `visual` | External API contracts and compatibility validation only |
| `retrieval` | `default` | PyTerrier `Retriever.num_results` / FAISS `Index.search` depth in the existing functions; BM25/dense retrieve; search remains unimplemented |
| `evaluation` | `default` | `evaluate_run` → `ir_measures.parse_measure` and PyTerrier `Experiment.eval_metrics` |
| `launcher` | `local_cpu`, `local_gpu` | Path normalization, output guards, model device, PyTorch/FAISS/Terrier threads and Terrier version |
| `stage` | `prepare`, `index`, `retrieve`, `evaluate`, `search`, `export` | `run_stage` artifact preflight / provenance, then BM25/dense prepare/index/retrieve/evaluate dispatch |
| `experiment` | `bm25_fulltext`, `dense_passages`, `visual_pages` | Hydra defaults-list composition; no duplicated model settings or runtime dispatcher |

`model` owns algorithm/library parameters. `representation` owns source content
and prospective transformations. `retrieval.top_k` owns ranking depth;
`evaluation` owns metrics only. Infrastructure roots, device selection, thread
counts, and JVM version live in `launcher`. `launcher` is an application group,
independent of Hydra's `hydra/launcher` plugin group. No scheduler dispatch is added.
A model's interpolated `device` is its external-library argument and must match
`launcher.device`; CPU FAISS and Terrier do not become GPU backends.

## Composition and overrides

Existing groups remain independently composable. `_self_` follows ordinary
root defaults; an optional global experiment preset follows it. Presets override
absolute group defaults without copying their bodies. Explicit CLI group choices
win over preset defaults. CPU/GPU and local/Hub inheritance are preserved.

```sh
uv run scientific-tot dry_run=true
uv run scientific-tot dry_run=true experiment=dense_passages launcher=local_gpu
uv run scientific-tot dry_run=true experiment=dense_passages representation=fulltext model.batch_size=32
uv run scientific-tot dry_run=true model=colbert representation=fulltext
uv run scientific-tot dry_run=true model=visual representation=figures
uv run scientific-tot dry_run=true split=train split.corpus=train
uv run scientific-tot dry_run=true 'model.retrieval.controls={bm25.b:0.5,bm25.k_1:1.5}'
uv run scientific-tot dry_run=true stage=retrieve paths.index=results/indexes/prior-index retrieval.top_k=200
uv run scientific-tot dry_run=true stage=evaluate paths.run=results/models/prior-run 'evaluation.metrics=[RR@10,R@100]'
uv run scientific-tot dry_run=true stage=search paths.index=results/indexes/prior-index 'stage.query=encoder diagram'
```

BM25 control names are **literal dotted library keys**. Override their dictionary
as shown; `model.retrieval.controls.bm25.b=...` would address a nonexistent nested
mapping. When changing a dense model identifier, also select that model's revision and
intended query/document prompts; the E5 profile's pinned commit and prompts are
not guessed or reset automatically.
Hydra owns configuration composition and override parsing, including its explicit
`+` and `++` additions. Ordinary overrides of unknown keys are rejected by Hydra.
Application validation checks supported values, backend compatibility, and safe
filesystem destinations; it does not reload YAML profiles to rebuild a key schema.
Missing values and interpolation errors retain OmegaConf's native diagnostics.

The corpus defaults to `all`, retaining unassigned distractors, independently of
query selection (`test` by default). `all` queries mean the train/test union.
`split.name` identifies the preset; `split.corpus` and `split.queries` are operative.
Text models require text representations; visual models require image representations.
Passage chunk sizes count characters. The metadata title/abstract adapter requires
its existing `[title, abstract]` projection; arbitrary field projections are refused.

Metrics use a deliberately validated subset of ir_measures syntax: `nDCG`, `RR`,
`R`, `P`, or `AP` with positive `@cutoffs`. Expressions must be distinct.
Evaluation checks cutoffs against the **saved retrieval depth**, not an evaluation
job's current retrieval settings. Index and search jobs do not run metric computation.

## Paths and machine profiles

```sh
export TOT_DATA_ROOT=/your/existing/restored/data
export TOT_CACHE_ROOT=/your/cache/scientific-tot
export TOT_OUTPUT_ROOT=/your/experiments/scientific-tot
uv run scientific-tot dry_run=true launcher=local_gpu experiment=visual_pages
```

There is no committed DGX03 path. Local data is read in place, with downloads
disabled. All relative managed paths, including `paths.index`, `paths.run`, cache
roots, and artifact overrides, resolve against the original invocation directory.
`hydra.job.chdir=true` does not change that base. Explicit local checkpoint
references beginning with `/`, `./`, `../`, or `~` also use that base; owner/repo
identifiers remain library identifiers. Symlinks resolve before checks.
Data, reusable caches, and experiment writes cannot overlap, in either direction.
The guard also covers custom Hydra run/sweep roots and external index destinations.

`dataset.artifacts` follows `dataset.root`. Individual overrides can describe
future layouts during a dry run, but `configured_adapter` refuses relocated
artifacts: the preserved Dataset adapter currently requires its canonical layout.
No alternate filesystem loader is introduced. `configured_adapter` constructs
one existing shared Dataset/adapter; it performs no discovery or download.
`corpus_arguments` and `query_arguments` provide its existing method arguments.
Fulltext corpus loading avoids optional metadata/figure enrichment, matching the
tutorial; title/abstract uses available original metadata.

`dataset=hub` explicitly enables network reads. Dataset revisions must be immutable
full 40-character SHAs or null; online Dataset discovery pins a null revision once.
Local revisions remain unverified assertions. `dataset.download.components` is the
future prepare-stage selection for `DatasetAdapter.download_components`, not a
promise that the Hub contains those artifacts. Physical restoration remains external.

Reusable cache layout is `${launcher.cache_root}/datasets` for Hub files and
`${launcher.cache_root}/embeddings` for dense embeddings/index cache entries,
including their library model cache. Results are separate. Dense calls with Hydra
settings must use the configured embedding cache path. Tutorial functions invoked
without Hydra settings retain their historical cache API/defaults.

## Experiment organization and stage contracts

`run.name` defaults to the effective
`<model>_<representation>_c-<corpus split>_q-<query split>` after all overrides;
`run.name=my-study` gives a custom plain filesystem-safe label. Names label a
study; artifact identities are independent of that label for indexes. Retrieval
includes the run label because it is serialized in `run.trec`.

```text
results/                           # launcher.output_root defaults to ./results
  indexes/<model>_<representation>_c-<corpus>_<hash>/
    .hydra/{config,hydra,overrides}.yaml
    manifest.json
    index-manifest.json
    data.properties or index.faiss # backend files directly alongside snapshots
  models/<model>_<representation>_c-<corpus>_q-<queries>_<hash>/
    .hydra/  manifest.json  run.trec  topics.jsonl  qrels.tsv
    query-ids.jsonl  corpus-ids.jsonl
  evaluations/<model>_<representation>_c-<corpus>_q-<queries>_<hash>/
    .hydra/  manifest.json  metrics.csv  per-query.csv
  prepared/<model>_<representation>_c-<corpus>_q-<queries>_<hash>/
    .hydra/  manifest.json  corpus.jsonl  queries.jsonl  qrels.jsonl
cache_root/                        # outside output_root, unchanged
  datasets/
  embeddings/                      # includes existing library model caches
```

`artifact_identity.py` is the centralized authority for names and locations.
Hydra's `artifact_dir` resolver calls it before directory creation; YAML contains
no separate hash/path algorithm. The suffix is 16 hex characters of the full
SHA256 stored as `artifact_sha256`, with the complete `artifact_identity` stored
alongside the existing full-configuration checksum. Short-name collisions fail
verification rather than overwriting an artifact.

Index identities use selected original corpus text/IDs/order, dataset repository
and resolved commit, corpus selection, document representation/preprocessing,
backend construction settings, and resolved document encoder commit/settings.
Relevant backend library versions and thread settings are retained. Filesystem
input locations remain exact provenance but do not determine corpus content
identity. Index identities exclude query artifacts/split/mode, evaluation metrics,
ranking depth, BM25 scoring controls, dense query prompts and passage aggregation.
Retrieval identities reference the exact source index manifest (path and checksum),
query content/selection/revision, model/ranking settings, prompts and depth.
Evaluation identities reference the exact source retrieval manifest and evaluation
settings. `prepare` includes selected corpus, queries, judgments and representation.
Model weights are not loaded to resolve names; mutable dense model references are
resolved to immutable commits and pinned for the actual build.

| Stage | Inputs and intended output | Current boundary |
| --- | --- | --- |
| `prepare` | Selected corpus/queries → `corpus.jsonl`, `queries.jsonl`, `qrels.jsonl` with coverage/provenance sidecars | BM25/dense text representations supported; every positive validated |
| `index` | Corpus → `results/indexes/<index-id>` or explicit `paths.index` | Persistent Terrier or FAISS index; verified identical artifacts reused |
| `retrieve` | Verified `paths.index` + batch queries → `run.trec`, `topics.jsonl`, `qrels.tsv`, `query-ids.jsonl`, `corpus-ids.jsonl` | BM25/dense batch retrieval; canonical paper IDs and artifact hashes |
| `evaluate` | Completed retrieval `paths.run` → aggregate/per-query metrics | Saved artifact/configuration validation; ir_measures aggregate/per-query metrics |
| `search` | Verified `paths.index` + required `stage.query` → ad hoc ranking | Index verification wired; interactive execution unimplemented |
| `export` | Canonical records → record artifacts | Unimplemented |

`evaluate` does not accept `paths.index` or invoke retrieval. `prepare`/`export`
accept neither input artifact path. `retrieve` and `search` never build an index.
The existing combined tutorial functions are not dispatched by any stage.

```sh
uv run scientific-tot stage=prepare dataset.root=/your/restored/data
uv run scientific-tot stage=index dataset.root=/your/restored/data
uv run scientific-tot stage=index dataset.root=/your/restored/data paths.index=/fresh/index
```

Preparation loads corpus and queries and validates every positive target. Indexing
loads only the selected corpus; query files and qrels are unnecessary. `dataset.missing=skip` cannot
omit a positive target. Local restored data is read in place; Hub mode pins
all reads to one resolved commit and downloads configured components through the
existing adapter. Required absent components fail explicitly. No dataset restoration
or document conversion algorithm is added. `prepare` writes canonical JSONL with
existing `Records.write_jsonl` sidecars; it does not start Java. `index` builds only an index, without query reads, retrieval or evaluation.
Explicit index downloads filter out query components and use the corpus split. BM25 indexing uses
`model.indexing` and `launcher.threads.indexing`, checks metadata capacity against
UTF-8 paper IDs, and verifies stored IDs and document count before completion.
Failed builds retain their output but have no completed index manifest.
The manifest records corpus content SHA256, dataset revision, local revision
verification status, library/Python/Terrier versions, and absolute output location.

The default offline tests exercise preparation, retrieval, evaluation and a mocked BM25 backend using
`tests/fixtures/adapter`. A real offline test also builds and reopens a persistent
index when Terrier 5.11 and helper jars are already cached in PyTerrier's home;
otherwise that test skips explicitly. Maven downloads and network access are
blocked during that test.

## Snapshots, provenance, and reuse

`FreshOutput` protects inputs/caches and reserves a new single-run artifact directory
**before Hydra writes**. Existing completed identical artifacts are fully verified
and reused without rewriting snapshots, manifests or outputs. Failed, incomplete,
corrupted or incompatible destinations fail explicitly. An exclusive `.run.lock`
also guards direct callers racing on an empty directory. Outputs are inventoried
and checksummed before the read-only manifest is marked completed. Sources are
checked again after construction; changed inputs cannot claim completion under
the reserved identity. Resolve failures before a destination is known leave no
artifact; failures after reservation preserve their snapshots and failed manifest.

The pinned Hydra basic sweeper validates all composed job paths and identities
before writing `multirun.yaml`. Distinct identities resolve to separate directories;
identical jobs can share a destination and reuse the first completed artifact.
Concurrent identical jobs encountering an unfinished reservation fail safely.
Conflicting identities or ancestor/descendant output paths fail in preflight.
Only `${artifact_dir:}` is supported for `hydra.sweep.subdir`. Mutable sweep
orchestration metadata lives in an exclusively reserved temporary directory outside
`results/`, never as an artifact wrapper. An explicit `hydra.sweep.dir` remains
supported and must be fresh; temporary orchestration directories can be removed
after a sweep. Other sweepers need explicit preflight integration. Automatic Hydra
snapshots and job file logging remain disabled.

`.hydra/` contains resolved absolute application settings, Hydra settings and
overrides; all three snapshots are read-only. Requested revisions remain in those
snapshots, while manifests record resolved dataset/model revisions and builder
provenance. Git commit and dirty status are recorded for a checkout; installed
wheels outside a checkout report null. `dry_run=true` resolves/validates configuration
without reading datasets/models, writes no snapshots/manifests, and uses a clearly
separate `dry-runs/` destination. Dry-run hashes describe requested configuration,
not a resolved artifact identity.

`write_index_manifest(index, cfg, provenance=...)` defines the builder's
completion contract: immutable full configuration, checksum, index-relevant settings,
resolved dataset/model revisions, and corpus SHA256. It uses exclusive creation.
`verify_index` is wired into retrieve/search preflight before backend dispatch.
It rejects missing/unsupported manifests, incomplete artifacts, changed configuration,
model/corpus/representation mismatch, and mismatched pinned revision provenance.
Query split, metric selection, ranking depth, BM25 retrieval controls, and dense
query prompt may change without rebuilding the corpus index. Other model build
settings and original dataset locations are conservatively required to match.
Hub index consumers require the build dataset commit. Dense index consumers
require the immutable model commit used at build time;
mutable refs/null revisions must be pinned before consuming an index, preventing
query encoding from resolving newer weights. Local dense checkpoint execution
remains unsupported by the existing Hub model-revision resolver.

Schema-2 index manifests inventory both Terrier and FAISS output bytes with exact
references/checksums. Completed stage manifests additionally seal snapshots and all
outputs; identical artifact reuse verifies that entire inventory. This does not
certify local dataset origins. Schema-1 Hydra indexes and saved batches may still
be consumed via explicit paths under their original verification contracts, but
cannot be reused as schema-2 outputs. Unmanifested tutorial indexes remain
unsupported. There is no implicit migration: rebuild from original inputs for new
artifacts and deterministic names. `verify_retrieval` requires a completed retrieve
manifest, intact config checksum and all five batch files with exact paths/checksums.

## Existing integration functions

For low-level integration/review, compose and resolve a config, construct the
existing adapter with `configured_adapter`, then call its methods with
`corpus_arguments(cfg)` and `query_arguments(cfg)`. Existing `bm25_experiment` and
`dense_experiment` accept optional `settings=cfg`. Those settings control actual
library calls, with data provenance/selection checks; they still combine operations
and are not stage implementations. Do not combine dense `settings` and an explicit
`EncodingConfig`. Calls without settings preserve tutorial defaults.

## Verification

```sh
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Tests cover every group/profile, compatible and incompatible compositions, overrides,
environment paths, chdir, Cartesian sweeps, sweep preflight, no-overwrite safeguards,
read-only snapshots, index identity/integrity, retrieval artifact preflight, adapter
selection, and actual external-library argument propagation. Neural encoding and
JVM construction are stubbed for offline parameter checks; real FAISS and metric
computation are exercised. Network/model/JVM integration tests remain opt-in.


## Independent retrieval and evaluation

```sh
uv run scientific-tot stage=retrieve dataset.root=/your/restored/data paths.index=/your/index retrieval.top_k=100
uv run scientific-tot stage=evaluate dataset.root=/your/restored/data paths.run=/your/retrieve 'evaluation.metrics=[RR@10,R@100]'
uv run scientific-tot model=dense representation=fulltext stage=index dataset.root=/your/restored/data
uv run scientific-tot model=dense representation=fulltext stage=retrieve dataset.root=/your/restored/data paths.index=/your/dense/index
uv run scientific-tot model=dense representation=passages stage=index dataset.root=/your/restored/data
```

Repeat the dataset/model/representation settings of the relevant build or batch
when consuming its artifacts. Retrieval may change query selection, query prompts,
ranking depth, BM25 controls, or passage aggregation without rebuilding the index.
Evaluation checks corpus/model/representation/query selection and encoding settings
against the saved retrieval configuration. Its metric expressions may change,
provided their cutoffs fit the saved retrieval depth. The saved depth is authoritative;
evaluation does not run retrieval to fill additional ranks.

Batch rankings use one-based TREC ranks. Paper IDs remain canonical; unique query
IDs stay unchanged, and colliding collection IDs use the existing reversible
percent-encoded query-key mapping. `query-ids.jsonl` retains the originals.
`corpus-ids.jsonl` permits positive/document validation without opening the index
or dataset during evaluation. All five saved inputs have absolute references and
SHA256 checksums. Invalid mappings, duplicate/unknown IDs, non-finite scores,
ambiguous runs, inconsistent ranks, missing positives or changed files are rejected.
`metrics.csv` and `per-query.csv` are separate evaluation outputs; per-query rows
include canonical query IDs/keys and domains. Entirely empty rankings are evaluated
by ir_measures directly, preserving zero results for every selected judged query.

Dense `index` uses the existing `EncodingConfig`, `_encode_tasks`, `_read_cache`,
and `_write_cache`. It encodes documents only and persists `index.faiss`, `ids.json`,
and `vectors.jsonl` with a canonical vector-to-paper/domain mapping. Dense `retrieve`
encodes only queries, independently of original document availability. Task caches
are content-addressed and checked before reuse; document keys exclude query prompts
and passage aggregation. Fresh index destinations may reuse unchanged cached
embeddings and FAISS artifacts. Model revisions are resolved to commits for builds;
consumers must pin the actual commit. Explicit prompts, including empty strings,
are passed unchanged to Sentence Transformers.

`representation=passages` loads canonical original fulltext, then uses
`RecursiveCharacterTextSplitter` with YAML chunk size, overlap, separators, and
whitespace policy. Passage IDs preserve paper IDs separately and escape the
PyTerrier passage delimiter reversibly. `representation.aggregation.method=max`
uses `pyterrier.text.max_passage`; `mean` uses `mean_passage`. FAISS scores every
indexed passage per query before aggregation, then the paper cutoff is applied.
The method, library, search depth, and paper ranking level are saved in manifests.
This exact aggregation can be expensive for large passage collections.

`launcher=local_cpu` encodes on CPU; `launcher=local_gpu` forwards CUDA to Sentence
Transformers where supported. FAISS remains the existing CPU `IndexFlatIP`; no GPU
FAISS implementation is introduced. Local model-path loading remains unsupported
by the existing revision resolver. Standalone combined tutorial functions retain
their established behavior; passage execution uses the separate Hydra stages.
