"""Finalize an interrupted, fully populated local ColQwen index without re-encoding.

Run only against your own trusted index: Qdrant stores pickled PointStruct objects.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from omegaconf import OmegaConf

from scientific_tot.artifact_identity import artifact_name, index_identity
from scientific_tot.config_arguments import configured_adapter
from scientific_tot.config_artifacts import artifact_reference, digest, index_settings, verify_index
from scientific_tot.integrations import colqwen


def recover_index(index: Path, *, requested=None) -> None:
    import portalocker

    index = index.resolve()
    with portalocker.Lock(index / ".run.lock", mode="a+", timeout=0):
        _recover_locked(index, requested=requested)


def _recover_locked(index: Path, *, requested=None) -> None:
    """Finalize, or resume for a stage invocation holding the artifact's run lock."""
    manifest_path = index / "manifest.json"
    original = json.loads(manifest_path.read_text())
    cfg = OmegaConf.load(index / ".hydra/config.yaml")
    if (
        original.get("schema_version") != 2
        or original.get("stage") != "index"
        or original.get("status") not in {"started", "failed"}
        or cfg.stage.name != "index"
        or cfg.model.name != "colqwen"
        or Path(cfg.paths.index).resolve() != index
        or original.get("index_location") != str(index)
        or digest(OmegaConf.to_container(cfg, resolve=True)) != original["configuration_sha256"]
    ):
        raise ValueError("Expected an interrupted ColQwen index at its original saved location")
    if requested is not None and index_settings(requested) != index_settings(cfg):
        raise ValueError("Requested index settings differ from the saved index configuration")

    print("Reading original PDFs and reconstructing index identity...", flush=True)
    corpus = colqwen.load_pdf_corpus(configured_adapter(cfg), cfg)
    identity = {
        "stage": "index",
        "index": index_identity(
            cfg,
            {
                "dataset_revision": corpus.provenance["revision"],
                "model_revision": cfg.model.revision,
                "corpus_sha256": colqwen.pdf_corpus_digest(corpus),
            },
        ),
    }
    if artifact_name(cfg, identity) != index.name:
        raise ValueError(
            "Current PDFs/settings/dependency versions differ from the original index hash"
        )

    if not (index / "index-manifest.json").exists():
        colqwen.build_index(corpus, cfg, existing="resume" if requested is not None else "finalize")
    # Interruption during sealing: verify completed backend metadata instead of rewriting it.
    built = verify_index(index, cfg, allow_building=True)
    if built["artifact_identity"] != identity:
        raise ValueError("Recovered metadata differs from the original index identity")
    provenance = built["provenance"]
    manifest = {
        **original,
        "status": "completed",
        "artifact_identity": identity,
        "artifact_sha256": digest(identity),
        "resolved_dataset_revision": corpus.provenance["revision"],
        "resolved_model_revision": cfg.model.revision,
        "artifacts": {"index_provenance": provenance},
        "recovery": {
            "original_manifest": original,
            "validation": "All stored page IDs, payloads, vector shapes and finite values checked",
            "effective_model_reconstructed": True,
        },
        "output_files": {
            str(path.relative_to(index)): artifact_reference(path)
            for path in sorted(index.rglob("*"))
            if path.is_file() and path.name not in {"manifest.json", ".run.lock"}
        },
    }
    manifest.pop("error", None)
    # Publish completion atomically, retaining the original manifest until everything passes.
    with tempfile.NamedTemporaryFile(mode="w", dir=index, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(manifest, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            temporary.chmod(0o444)
            temporary.replace(manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Completed recovered index: {index}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path, help="Original interrupted index directory")
    args = parser.parse_args()
    recover_index(args.index)


if __name__ == "__main__":
    main()
