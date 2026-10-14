# YAML profile audit and runtime mapping

The audit covers every file under `config/`. The authoritative experiment interface
is the [Hydra guide](configuration.md). The adapter's content and identity policy
is unchanged. BM25/dense prepare/index/retrieve/evaluate execute for text; dense passages also execute. Other
stage/model combinations remain unimplemented. Parameter bindings below also serve
existing low-level integration functions via `settings`.

## Full profile inventory

| YAML file(s) | Settings traced to implementation / status |
| --- | --- |
| `config.yaml` | Defaults → Hydra composition; `dry_run` → entry point; `run.name` → callback/output organization and run manifest; `paths` → normalization/stage preflight; `hydra` → output reservation, exclusive snapshots and sweeps |
| `dataset/default.yaml` | `name` is a dataset label; source/repo/revision/root/cache → `configured_adapter` → `Dataset`; domains/missing/query_mode → adapter method kwargs; artifacts describe canonical Dataset layout (relocation rejected by configured adapter); download.enabled gates Hub reads; components → existing `download_components` in BM25/dense prepare/index |
| `dataset/hub.yaml` | Inherits local profile's common values; source=hub and download.enabled=true select Hub API/cache; Dataset pins discovery revision |
| `split/all.yaml` | corpus=all includes distractors; queries=all → adapter's train/test union |
| `split/train.yaml` | corpus=all, queries=train → independent adapter selections |
| `split/test.yaml` | corpus=all, queries=test → independent adapter selections |
| `split/train+test.yaml` | corpus=all, queries=train+test → independent adapter selections |
| `representation/fulltext.yaml` | name → `load_corpus(representation=fulltext)`; modality/source → compatibility and index identity checks; original markdown retained as loaded |
| `representation/title_abstract.yaml` | name → existing metadata projection in `load_corpus`; fields fixed to title/abstract (overrides rejected); modality/source → compatibility and identity checks |
| `representation/passages.yaml` | RecursiveCharacterTextSplitter consumes chunk size/overlap/separators/whitespace; reversible passage IDs retain canonical parents; PyTerrier max/mean aggregation produces paper rankings |
| `representation/figures.yaml` | Existing adapter can attach original figure metadata/locators to text records. A figure-only corpus and include_captions transformation are not implemented; profile is a validated prospective representation |
| `representation/pages.yaml` | Future pypdfium2 page rendering scale; original PDF source/modality validated. No renderer, page IDs, or page-to-paper aggregation implemented |
| `model/bm25.yaml` | type → `IndexingType.CLASSIC` enum; blocks/meta → `IterDictIndexer`; wmodel/controls → `Retriever`. `bm25_arguments` binds these; configured metadata capacity is checked before truncation |
| `model/dense.yaml` | model_id/revision/device → `SentenceTransformer`; batch_size → encode_query/encode_document; normalize_embeddings → encoding (through existing cosine/dot config); prompts → encode prompt arguments; Flat/inner_product → existing `IndexFlatIP`. `dense_arguments` binds these; unsupported L2 is rejected |
| `model/colbert.yaml` | Future checkpoint → Indexer/Searcher; index_bsize, nbits, kmeans_niters, doc_maxlen, query_maxlen → ColBERTConfig. No ColBERT integration, device binding, or runtime consumer implemented |
| `model/visual.yaml` | Future model_id/device → Byaldi from_pretrained; storage/overwrite/image dimensions → index. No executing visual integration or generic revision/batch-size argument |
| `retrieval/default.yaml` | top_k → Retriever.num_results / FAISS.search k in existing integration functions; future Byaldi.search k / ColBERT.search k contracts; distinct from metrics |
| `evaluation/default.yaml` | backend identifies ir_measures; metrics → parse_measure → Experiment eval_metrics. Default remains nDCG@10, RR@10, R@100 |
| `launcher/local_cpu.yaml` | kind/device → local validation and supported model device argument; dataset/cache/output roots → original-cwd normalization and guards; indexing threads → Terrier indexer/retriever; encoding threads → torch.set_num_threads; faiss threads → faiss.omp_set_num_threads; terrier_version → pt.terrier.set_version/JVM-version check |
| `launcher/local_gpu.yaml` | Inherits infrastructure settings and overrides device=cuda; FAISS stays CPU and Terrier stays JVM; no GPU promise for unimplemented ColBERT |
| `stage/prepare.yaml` | BM25/dense → configured adapter, positive validation, canonical JSONL exports and coverage sidecars; dense passages also export splitter output |
| `stage/export.yaml` | Name → stage contract; body unimplemented |
| `stage/index.yaml` | BM25 → configured IterDictIndexer; dense → existing document encoder/cache and persistent FAISS/mapping; fresh destinations and completed manifests, no ranking/evaluation |
| `stage/retrieve.yaml` | Name → index snapshot/configuration verification; BM25/dense query-only batch retrieval with persistent indexes |
| `stage/evaluate.yaml` | Name → completed retrieval artifact/config/depth verification; shared saved-run aggregate/per-query evaluation, with no inference |
| `stage/search.yaml` | Name/query → required ad hoc query and verified index contract; body unimplemented |
| `experiment/bm25_fulltext.yaml` | Hydra absolute overrides select BM25/fulltext/index; reusable group bodies remain independent |
| `experiment/dense_passages.yaml` | Hydra absolute overrides select dense/passages/index; executes the passage builder and cached FAISS indexing |
| `experiment/visual_pages.yaml` | Hydra absolute overrides select visual/pages/index; validated preset, execution unsupported |

`config_arguments.py` contains plain bindings to existing APIs, not a new dataset
or model abstraction. `configuration.py` validates the YAML-owned key shapes and
values without optional backends; `config_artifacts.py` defines artifact identity;
`hydra_run.py` reserves outputs and performs stage preflight.

## Findings, changes, and defaults

| Finding | Resolution / justification |
| --- | --- |
| Every stage was a stub, but parameter tables described eventual execution as if present | Explicitly distinguish binding support, prospective contracts, and CLI stage execution throughout profiles/docs |
| Top-100 retrieval was hardcoded in dense/BM25; evaluation.top_k had no consumer | Move the unchanged value 100 to independently composable retrieval/default; settings propagate to actual search calls |
| evaluate was specified as index-consuming and therefore conflated retrieval/metrics | Add retrieve; evaluate requires a completed paths.run, search/retrieve require paths.index |
| YAML metrics nDCG@10/RR@10/R@100 differed from tutorial RR@10/R@10/R@100 | Preserve both existing defaults. Hydra settings explicitly propagate YAML metrics; tutorial calls without settings retain their published metrics |
| Dense YAML accepted L2, but execution always constructed IndexFlatIP | Reject L2 and non-Flat factories; no new index implementation introduced |
| Dense had an independent dataclass/defaults/prompts interface | Bind settings explicitly to the existing EncodingConfig and external calls; reject settings+config conflicts. Keep the dataclass as a backward-compatible tutorial/library API |
| BM25 YAML index options and controls were ignored by existing tutorial code | Bind them when settings is passed; translate classic to the library enum. Reject insufficient docno capacity instead of silently increasing a configured capacity or truncating IDs |
| Dense precision/remote-code policy, model truncation, and Terrier tokenisation were implicit | Keep float32 NumPy embeddings for the existing FAISS representation, trust_remote_code=false, saved model tokenizer truncation, and Terrier English/Porter/Stopwords defaults. These are implementation invariants, not ignored YAML controls. No tunable YAML precision, tokenizer, max_seq_length, prompt_name, encoding profile, or baseline/demo-subset settings are claimed; existing tutorial APIs retain those separate options where present |
| Thread counts/JVM version were hidden runtime defaults | Expose existing values (all threads 1, Terrier 5.11) in infrastructure; preserve calls without settings |
| Fulltext/title_abstract strip_whitespace=true had no adapter consumer; adapter retains original text | Remove the unused preprocessing fields rather than changing original document bytes. Passage strip_whitespace is passed to the library splitter |
| Top-level seed=42 was never used | Remove the misleading unused seed; no stochastic behavior or scientific defaults changed. Future stochastic execution must wire its actual seed argument explicitly |
| title_abstract.fields was mutable despite fixed adapter projection | Keep established fields and refuse other projections |
| Artifact overrides were accepted even though Dataset requires canonical paths | Keep dry-run contracts; reject relocated artifacts in the existing configured adapter, without adding a filesystem abstraction |
| Output timestamps lacked study identity and could mix caches/results | Use content identities under indexes/models/evaluations/prepared; guard output/cache overlap including custom Hydra roots and external indexes |
| Single-run guards did not validate all swept jobs before writing metadata | Expand the pinned basic sweeper and preflight every composition before reservation; reject unsafe job layouts and duplicate external index destinations |
| Configuration snapshots were preserved but writable; Git SHA omitted dirty state | Make snapshots/final manifests read-only, hash resolved config, record dirty status, retain failed attempts and exclusive creation |
| Index consumers could not verify build configuration | Add completed index-manifest schema with immutable config, digest, resolved revisions and corpus content digest; reject incompatible/legacy/missing snapshots before dispatch |

Scientific values remain unchanged: BM25 b=0.75/k_1=1.2 and indexing defaults,
E5 checkpoint/commit, batch size 16, normalization and query/passage prompts,
corpus=all/query=test, YAML metrics, passage size=1000/overlap=200/separators,
page scale=2, ColBERT settings, visual checkpoint and overwrite=false.
The ColBERT profile keeps nbits=2 even though the inspected upstream settings
currently default to 1; this repository's established setting is deliberate and
is not silently changed to track an external default.

Unsupported functionality remains explicit: search/export stages,
other model backends, image/page pipelines, relocated artifacts,
scheduler/custom sweeper integration, binary-index integrity checking/migration,
and independent verification of local dataset origins. A dry run can inspect
planned configurations without pretending those operations exist.

## External API verification

Executable arguments were traced in the installed PyTerrier/Sentence Transformers
libraries and exercised with constructor/encoding stubs plus real FAISS/evaluation.
The prospective contracts were checked against primary API sources:
[PyTerrier indexing](https://pyterrier.readthedocs.io/en/latest/terrier-indexing.html),
[SentenceTransformer](https://sbert.net/docs/package_reference/sentence_transformer/model.html),
[FAISS indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes),
[ColBERT settings](https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/infra/config/settings.py),
and [Byaldi wrapper](https://github.com/AnswerDotAI/byaldi/blob/main/byaldi/RAGModel.py).
Byaldi exposes model path/device and image indexing options but no revision
argument; ColBERT exposes a checkpoint/config. Future reproducible execution of
those profiles needs an immutable local checkpoint with recorded content identity.
