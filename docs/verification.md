# Verification — 2026-10-11

Executed locally on macOS, CPython 3.11.14:

| Check | Result |
| --- | --- |
| `HF_HUB_OFFLINE=1 .venv/bin/pytest` | 31 passed; live test deselected |
| `.venv/bin/pytest -m live` | 1 passed; 31 offline tests deselected; pinned Hub inventory only |
| `.venv/bin/ruff check .` | Passed |
| `.venv/bin/ruff format --check .` | Passed for the complete project |
| `uv lock --check --offline` | Passed |
| `uv build --offline` | Source distribution and wheel built successfully |
| Wheel installed in a fresh uv environment outside the checkout | Import and offline JSON/split reads passed from `/private/tmp`; no Arrow or dev dependencies installed |
| Downloaded ACL test queries loaded through `Dataset(local_dir=...)` | All 110 records validated offline |

A clean-wheel check caught an overly broad `data/` ignore rule that excluded
package modules after Git initialization. The rule now anchors to `/data/`;
the rebuilt wheel passed import and file-reading checks outside the checkout.

CI is configured for Python 3.11 and 3.13, including locked dependency sync,
offline tests, lint/format checks, wheel build, and isolated wheel import.
GitHub-hosted CI and Python 3.13 have not been run in this local setup.

No full dataset, PDF/archive, retrieval model, or model weights were downloaded.
Inspections fetched the Hub file inventory, ten small mode metadata files, ACL
test root metadata, and one ACL test JSONL. The upstream generation test suite
was inspected but not executed. Published full-corpus IDs, split membership,
archive layouts, and figure completeness remain unverified because those
components are absent from the inspected revision.

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
│   ├── architecture.md
│   ├── dataset-schema.md
│   ├── hub-inventory.json
│   ├── integration-plan.md
│   └── verification.md
├── examples/inspect_dataset.py
├── src/scientific_tot/
│   ├── __init__.py
│   └── data/
│       ├── __init__.py
│       ├── hub.py
│       ├── identifiers.py
│       ├── schemas.py
│       └── validation.py
└── tests/
    ├── conftest.py
    ├── fixtures/
    │   ├── query.json
    │   └── split.json
    ├── test_data.py
    └── test_live.py
```

Git initialized on `main`; no remote or commit created. Export/config/script
folders are deferred until they contain implemented functionality.
