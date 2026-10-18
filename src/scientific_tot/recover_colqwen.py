"""Finalize an interrupted, fully populated local ColQwen index without re-encoding.

Run only against your own trusted index: Qdrant stores pickled PointStruct objects.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from scientific_tot.artifact_identity import artifact_name, index_identity
from scientific_tot.config_arguments import configured_adapter
from scientific_tot.config_artifacts import artifact_reference, digest, verify_index
from scientific_tot.integrations import colqwen


def validate_points(database: Path, mapping: list[dict], dimension: int) -> None:
    """Read one persisted point at a time; never instantiate an in-memory Qdrant index."""
    from qdrant_client.local.persistence import CollectionPersistence
    from tqdm.auto import tqdm

    with closing(sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        count = db.execute("SELECT COUNT(*) FROM points").fetchone()[0]
        print(f"Stored pages: {count:,}; expected pages: {len(mapping):,}", flush=True)
        if count != len(mapping):
            raise ValueError("Stored page count differs from corpus; cannot finalize this index")
        seen = bytearray(count)
        for key, blob in tqdm(
            db.execute("SELECT id, point FROM points"), total=count, desc="Validating stored pages"
        ):
            point = pickle.loads(blob)
            position = point.id
            if type(position) is not int or not 0 <= position < count or seen[position]:
                raise ValueError(f"Invalid or duplicate page position: {position}")
            if (
                key != CollectionPersistence.encode_key(position)
                or point.payload != mapping[position]
            ):
                raise ValueError(f"Stored page identity/payload differs at position {position}")
            vectors = np.asarray(point.vector, dtype=np.float32)
            if (
                vectors.ndim != 2
                or vectors.shape[0] == 0
                or vectors.shape[1] != dimension
                or not np.isfinite(vectors).all()
            ):
                raise ValueError(f"Invalid page vectors at position {position}")
            seen[position] = 1


def recover_index(index: Path) -> None:
    import portalocker

    index = index.resolve()
    # Share Qdrant's own lock so recovery cannot overlap a writer or another recovery.
    with (index / "qdrant/.lock").open("r+") as lock:
        portalocker.lock(lock, portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING)
        _recover_locked(index)


def _recover_locked(index: Path) -> None:
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
    if (index / "index-manifest.json").exists():
        raise FileExistsError(
            "Index metadata already exists; inspect it before attempting recovery"
        )

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

    metadata = json.loads((index / "qdrant/meta.json").read_text())
    vectors = metadata["collections"]["pages"]["vectors"]
    if (
        vectors["distance"] != "Dot"
        or vectors["multivector_config"]["comparator"] != "max_sim"
        or vectors["hnsw_config"]["m"] != 0
    ):
        raise ValueError("Stored collection is not the expected exact DOT/MAX_SIM index")
    validate_points(
        index / "qdrant/collection/pages/storage.sqlite",
        colqwen.page_mapping(corpus),
        vectors["size"],
    )

    print(
        "Loading pinned model to reconstruct processor/checkpoint metadata (no encoding)...",
        flush=True,
    )
    model, effective = colqwen.load_model(cfg)
    if model.get_embedding_dimension() != vectors["size"]:
        raise ValueError("Stored vector dimension differs from the pinned model")
    del model

    print("Writing metadata and checksumming index files; this can take a while...", flush=True)
    provenance = colqwen.publish_index_metadata(corpus, cfg, effective)
    built = verify_index(index, cfg, allow_building=True)
    if built["artifact_identity"] != identity:
        raise ValueError("Recovered metadata differs from the original index identity")
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
