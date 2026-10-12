# Hydra configuration

The `scientific-tot` command (also `python -m scientific_tot.hydra_entry`) composes
YAML settings with Hydra 1.3 and OmegaConf. Install with `uv sync --locked`.
Configuration inspection needs only the core package: it imports no retrieval
libraries, loads no model weights, and reads/downloads no dataset artifacts.

This architecture follows the generation repository's
[defaults list](https://github.com/stanfris/multimodal-tip-of-tongue-retrieval-for-scientific-papers/blob/main/config/config.yaml),
entry point, environment interpolation, and composition tests. Generation settings,
MinerU, model serving, and custom scheduler machinery are not carried over.
The library API and existing example scripts remain available; they are not yet
connected to this CLI. Their existing tutorial defaults are unchanged.

## Configuration ownership

| Group | Responsibility | Profiles |
| --- | --- | --- |
| `dataset` | Source, Hub identity/revision, domains, local artifact paths, selective downloads, missing-data policy | `default` (local), `hub` |
| `split` | Query selection and independently selected corpus split | `train`, `test`, `train+test`, `all` |
| `representation` | Original content source, preprocessing, passage splitting, page rendering | `title_abstract`, `fulltext`, `passages`, `figures`, `pages` |
| `model` | External retrieval backend and its actual parameters | `bm25`, `dense`, `colbert`, `visual` |
| `stage` | One pipeline operation | `prepare`, `index`, `evaluate`, `search`, `export` |
| `evaluation` | Metric expressions and ranking depth | `default` |
| `launcher` | Execution device and machine-dependent data, cache, output roots | `local_cpu`, `local_gpu` |
| `experiment` | Composition of existing group profiles | `bm25_fulltext`, `dense_passages`, `visual_pages` |

All five stage execution boundaries currently raise `NotImplementedError`.
This is deliberately a configuration phase. In subsequent phases, connect each
operation in `scientific_tot.hydra_run.execute_stage` to concrete adapters and
library calls. `index` must build only an index; `evaluate` must consume a
previous index through `paths.index`. Do not route either stage through the
existing examples that combine indexing and evaluation.

`prepare` will select/download/load configured components; `export` will write
canonical records; `search` will consume `paths.index` and `stage.query`.
Custom artifact locations are configuration contracts for these later stages;
the current Dataset API still expects its canonical on-disk layout. No new
loader, passage construction, or image processing is implemented here.

## Composition and overrides

`config/config.yaml` selects independent groups. `_self_` applies that file's
values after its ordinary defaults. The optional experiment comes afterwards.
Experiment YAML uses `# @package _global_` and `override /model`,
`override /representation`, and `override /stage` to reuse group definitions.
CPU/GPU and local/Hub profiles use defaults-list inheritance with `_self_` last.
Explicit CLI group selections override experiment defaults.

```sh
uv run scientific-tot dry_run=true
uv run scientific-tot dry_run=true model=dense representation=passages launcher=local_gpu
uv run scientific-tot dry_run=true model=colbert representation=fulltext
uv run scientific-tot dry_run=true model=visual representation=figures launcher=local_gpu
uv run scientific-tot dry_run=true experiment=dense_passages model.batch_size=32
uv run scientific-tot dry_run=true experiment=visual_pages representation=figures stage=export
uv run scientific-tot dry_run=true split=train split.corpus=train
uv run scientific-tot dry_run=true stage=search paths.index=results/prior/index 'stage.query=diagram of an encoder'
uv run scientific-tot dry_run=true stage=evaluate paths.index=results/prior/index
```

The corpus defaults to `all` even when queries select `test` or `train`, retaining
unassigned distractor papers. Set `split.corpus` explicitly for a restricted
corpus. `split=all` selects train and test queries; no synthetic third query split
is created. `split.name` identifies the chosen preset; `split.queries` is the
operative query selection and can also be overridden.

Text models accept title/abstract, fulltext, and passages. Visual models accept
figures or pages. Selecting an incompatible pair fails validation. Corpus profiles
use original document metadata/text/figures/PDF pages; generated clues are never
original content. Query collections are separate. Passages use the established
`RecursiveCharacterTextSplitter` and character counts; pages declare pypdfium2
rendering. Those optional libraries are not installed or invoked by this task.

## Existing local data, including DGX03

There is no committed DGX03 path. Point the configuration to its existing restored
root; `dataset=default` uses local data in place with downloads disabled:

```sh
export TOT_DATA_ROOT=/your/existing/restored/data
export TOT_CACHE_ROOT=/your/cache/scientific-tot
export TOT_OUTPUT_ROOT=/your/experiments/scientific-tot
uv run scientific-tot dry_run=true launcher=local_gpu experiment=visual_pages
```

For a local fixture inspection:

```sh
TOT_DATA_ROOT=tests/fixtures/adapter uv run scientific-tot dry_run=true
uv run scientific-tot dry_run=true dataset.root=tests/fixtures/adapter 'dataset.domains=[ACL]'
```

`dataset.artifacts.{preprocessed,metadata,queries,splits,pdfs}` follow `dataset.root`
automatically; override individual paths for artifacts stored elsewhere. All
relative filesystem paths, including individual overrides, resolve once against
the original invocation directory, not the YAML directory or Hydra output directory.
`hydra.job.chdir=false` is the default, and runtime normalization also supports
an explicit `hydra.job.chdir=true`. Symlinks resolve before overlap checks.
Writable output/cache directories must be separate from dataset inputs.
Dry runs validate path structure without requiring data or GPU hardware to exist
on the inspecting machine. A later executing stage must verify required files.

Hub access is an explicit selection:

```sh
uv run scientific-tot dry_run=true dataset=hub \
  dataset.repo_id=kasys/open-source-scientific-documents \
  'dataset.download.components=[queries,splits]' \
  dataset.revision=3b764a4c6855925ec802a2560712c4f5cc96f980
```

This still performs no download. Dataset revisions, when specified, must be full
immutable commit SHAs. A revision supplied for local data is an unverified
provenance assertion. Omitting a Hub revision will let the existing Dataset API
resolve and pin it at first actual discovery in a later execution integration.
Domain archives are the adapter's download unit; split filtering does not make
archive transfers smaller. The published revision may lack corpus or split
artifacts: composition is not an assertion of their availability.

## Model parameter contracts

Profiles describe supported external-library arguments, rather than a universal
model interface. Their identifiers, revisions where supported, and indexing
options live in YAML. No backend is installed or loaded by configuration checks.

| Profile | Parameter mapping |
| --- | --- |
| BM25 | `indexing` → PyTerrier `IterDictIndexer` (`type`, `blocks`, `meta`); `retrieval` → `Retriever` (`wmodel`, `controls`) |
| Dense | `model_id`, `revision`, `device` → SentenceTransformer; `batch_size`, encoding normalization/prompts → encode calls; `factory=Flat`, metric → FAISS flat inner-product or L2 index |
| ColBERT | `checkpoint` → ColBERT Indexer/Searcher; index/query lengths, `index_bsize`, `nbits`, `kmeans_niters` → ColBERTConfig |
| Visual | `model_id`, `device` → Byaldi `RAGMultiModalModel.from_pretrained`; collection storage, overwrite, image dimensions → `index` |

References: [PyTerrier indexing](https://pyterrier.readthedocs.io/en/latest/terrier-indexing.html),
[Sentence Transformers API](https://sbert.net/docs/package_reference/sentence_transformer/SentenceTransformer.html),
[FAISS indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes),
[ColBERT settings](https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/infra/config/settings.py),
and [Byaldi API](https://github.com/AnswerDotAI/byaldi/blob/main/byaldi/RAGModel.py).

Byaldi's verified public wrapper has no revision or batch-size argument. ColBERT's
verified Indexer uses a checkpoint and its own `index_bsize`; a generic revision
argument is not exposed. For pinned runs with these backends, use an immutable
local checkpoint directory and preserve its identity in the eventual execution
manifest. Do not add parameters based on assumptions about transitive libraries.
The dense profile pins E5's model commit and explicit prompts. BM25 has no weights
or model revision. CUDA in a launcher selects the model device where its API
supports it; it does not claim GPU support for FAISS CPU or PyTerrier. No Slurm or
PBS dispatch is implemented; `launcher` remains separate from `hydra/launcher`,
so future Submitit integration can use Hydra's existing plugin mechanism.

## Inspecting, presets, and sweeps

`dry_run=true` resolves interpolation, normalizes paths, validates compatibility,
and prints the effective application YAML. It performs no stage execution,
index writes, provenance writes, dataset discovery, or model downloads. Hydra
creates empty run directories (and sweep metadata for multiruns).

```sh
uv run scientific-tot --cfg job --resolve experiment=bm25_fulltext
uv run scientific-tot --info defaults
uv run scientific-tot --multirun dry_run=true \
  experiment=bm25_fulltext,dense_passages,visual_pages split=train,test
uv run scientific-tot --multirun dry_run=true model=dense \
  representation=fulltext,passages model.batch_size=8,16
```

`--cfg job --resolve` is Hydra's built-in composition inspection; it does not run
application validation or filesystem normalization. Use `dry_run=true` for both.
Programmatic composition uses `hydra.initialize_config_dir` with
`scientific_tot.configuration.CONFIG_DIR`, then `hydra.compose`; validation is
`resolve_config(cfg, base_dir)`. It needs no live Hydra runtime singleton.

To add a reproducible preset, create `config/experiment/<name>.yaml` using the
existing presets' global package and absolute group overrides. Keep model and
representation settings in their groups. Pin dataset/model commits where
available, and pass environment paths when launching. No group bodies need copying.

## Output protection and provenance

Hydra selects timestamped run directories, or numbered jobs inside a sweep.
A small callback reserves a fresh run/sweep root before Hydra writes anything;
existing roots are rejected even for dry runs. Guard validation failures are fatal,
including malformed dataset paths, so Hydra cannot continue and overwrite sweep metadata. Use a new `hydra.run.dir` or
`hydra.sweep.dir` when overriding the defaults.

For attempted execution, the entry point exclusively creates `.hydra/` and saves
resolved, absolute application settings, Hydra settings, and override information.
Hydra's automatic output subdirectory and job file logging are disabled so they
cannot overwrite a previous experiment before the task checks its output.
A `manifest.json` records status, requested dataset/model revision, model identifier,
retrieval repository Git commit when running from a checkout, and output/index
locations. Unknown revisions/commits are null. Installed wheels outside a checkout
have no live Git commit. A later integration must record revisions discovered at
execution rather than retaining unknown values.

Unimplemented stages return a nonzero exit status and record a failed attempt,
never a successful placeholder. Failed run directories remain available for
inspection and cannot be reused. An existing `paths.index` is read by search or
evaluation and must be separate from the new run directory. Index construction
must refuse existing indexes in the future execution integration.

## Verification

```sh
uv run pytest tests/test_configuration.py
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv build
```

Tests use Hydra's composition API plus subprocess CLI checks for dry runs,
validation, real Cartesian multiruns, failed-stage provenance, and preservation of
existing runs/sweeps. Every model/representation combination is checked without
network access or model downloads. The YAML tree is included in both source and
wheel distributions; installing the wheel needs no repository-root config folder.
