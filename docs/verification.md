# Verification — Phase 2, 2026-10-11

Executed locally on macOS, CPython 3.11.14:

| Check | Result |
| --- | --- |
| `HF_HUB_OFFLINE=1 .venv/bin/pytest` | 64 passed; one live test deselected |
| `.venv/bin/pytest -m live` | One passed; pinned inventory, collection metadata, and 110 ACL test queries |
| `.venv/bin/ruff check .` | Passed |
| `.venv/bin/ruff format --check .` | Passed |
| `uv lock --check --offline` | Passed; no new runtime dependencies |
| `uv build --offline` | Source distribution and wheel built successfully |
| Wheel installed in a fresh uv environment outside checkout | Public canonical API, full offline corpus/query/qrel conversion, and JSONL export passed from `/private/tmp`, without Arrow or development dependencies |
| Executable local example | Two synthetic documents, one query, one resolving qrel; all three exports and manifests produced |

Offline tests cover domain/split filtering; strict/skip missing-text policies;
missing extractions and figures; original metadata/licensing; title-only and
explicit original abstract fields; preservation of canonical IDs; collision-safe
lean lookup; unsafe archives; cross-shard papers; image locators without decoding;
mocked incremental/pinned selective downloads; duplicate IDs; malformed JSON;
query scope across domains; export checksums; and unresolved relevance targets.

CI is configured for Python 3.11 and 3.13. It runs locked dependency sync,
offline tests, lint/format checks, wheel build, and the executable converter
against the installed wheel in a separate environment. GitHub-hosted CI and
Python 3.13 have not been executed locally.

The live Hub head was rechecked: it remains
`3b764a4c6855925ec802a2560712c4f5cc96f980`, with 31 files and queries only.
The optional integration test confirms query conversion and the expected
missing-corpus error. No live corpus/qrel conversion is claimed. The downloaded
ACL test file from Phase 1 was reused through the Hub cache. No dataset PDF,
extraction archive, full dataset, model, or model weights were downloaded.
Archive/PDF-index support is tested with small synthetic artifacts reflecting
the pinned generation implementation, not a fabricated live schema.

## Repository structure

```text
scientific-tot-retrieval/  # project in the provided multimodal-tot-retrieval workspace
├── .github/workflows/ci.yml
├── .gitignore
├── .python-version
├── LICENSE
├── README.md
├── pyproject.toml
├── uv.lock
├── docs/
│   ├── api.md
│   ├── architecture.md
│   ├── dataset-schema.md
│   ├── hub-inventory.json
│   ├── integration-plan.md
│   └── verification.md
├── examples/
│   ├── convert_dataset.py
│   └── inspect_dataset.py
├── src/scientific_tot/
│   ├── __init__.py
│   ├── adapter.py
│   ├── api.py
│   ├── records.py
│   └── data/
│       ├── __init__.py
│       ├── artifacts.py
│       ├── hub.py
│       ├── identifiers.py
│       ├── schemas.py
│       └── validation.py
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   ├── README.md
    │   ├── adapter/  # complete synthetic two-paper dataset
    │   ├── query.json
    │   └── split.json
    ├── test_adapter.py
    ├── test_data.py
    └── test_live.py
```

## Phase 3 verification (2026-10-11)

PyTerrier integration checks: 73 offline tests passed locally and on dgx03;
1 explicitly selected Java integration test passed on dgx03. Ruff check and
format checks passed, wheel/source distribution builds succeeded, and a clean
wheel installation with the PyTerrier extra converted the canonical fixture.
The real-data CLI indexed/evaluated both a 250-document subset and all available
ACL extractions. The notebook was executed on dgx03. See
[actual data results and coverage](pyterrier-results.md) and
[reproducible API/examples](pyterrier.md). Core imports do not initialize Java.

Real-data loading revealed repeated full-inventory membership construction.
Local discovery now caches the path set for the selected inventory and prunes
hidden cache directories before traversal; it preserves offline inventory and
exact-file selection semantics. Default tests still block network and exclude
live/Java checks. CI adds a separate opt-in Java job with a configured JDK.


## Phase 4 verification (2026-10-11)

79 offline fixture tests passed locally and on dgx03. The opt-in pretrained-model
integration and the existing Terrier integration each passed on dgx03. Dense
fulltext and title/abstract runs used the exact saved BM25 subset, with verified
corpus hashes, IDs, topics and qrels. An offline cached run reproduced its TREC
file exactly; the dense notebook executed with actual data. The built core wheel
was installed into a clean environment without any neural/retrieval packages and
loaded canonical records. See [dense results](dense-results.md) and the
[executable tutorial](dense-retrieval.md). No full-corpus dense or GPU benchmark
was performed.
