"""Check installed H200 environments without downloading weights or changing packages."""

from __future__ import annotations

import importlib
import platform
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

from scientific_tot.stage_cli import plan


def check(profile: str, cfg) -> None:
    if profile not in {"default", "colqwen"}:
        raise ValueError("Select the default or colqwen environment")
    if platform.system() != "Linux" or sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use Linux and Python 3.11 for the H200 deployment")
    from packaging.requirements import Requirement

    # Dependency constraints remain owned by pyproject.toml and uv.lock.
    project = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    extras = project["project"]["optional-dependencies"]
    groups = ["colqwen"] if profile == "colqwen" else ["dense", "colbert", "visual", "pyterrier"]
    failures = []
    for text in sorted({text for group in groups for text in extras[group]}):
        requirement = Requirement(text)
        try:
            # GPU FAISS may replace the faiss-cpu distribution. Validate the
            # actual module version here and its selected backend below.
            installed = (
                importlib.import_module("faiss").__version__
                if requirement.name == "faiss-cpu"
                else version(requirement.name)
            )
            if installed not in requirement.specifier:
                failures.append(f"{requirement}: installed {installed}")
        except ImportError:
            failures.append(f"Missing {requirement}")
    if failures:
        raise RuntimeError("Re-run the appropriate locked uv sync:\n" + "\n".join(failures))
    import torch

    print(
        f"Environment={profile}; Python={platform.python_version()}; "
        f"Torch={torch.__version__}; CUDA runtime={torch.version.cuda}; "
        f"Transformers={version('transformers')}",
        flush=True,
    )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch cannot access CUDA. Check driver/runtime compatibility, "
            "CUDA_VISIBLE_DEVICES and your GPU allocation; keep the lock unchanged."
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one allocated H200 with CUDA_VISIBLE_DEVICES")
    free, total = torch.cuda.mem_get_info()
    print(
        f"GPU={torch.cuda.get_device_name(0)}; capability={torch.cuda.get_device_capability(0)}; "
        f"free={free / 2**30:.1f} GiB / total={total / 2**30:.1f} GiB",
        flush=True,
    )
    if "H200" not in torch.cuda.get_device_name(0):
        raise RuntimeError("The selected GPU is not an H200; check the allocation")
    if free < 8 * 2**30:
        raise RuntimeError(
            "Less than 8 GiB free. Release other GPU jobs before the model smoke test"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "BF16 is unavailable; check the selected device and locked Torch runtime"
        )
    with torch.inference_mode():
        sample = torch.ones((16, 16), device="cuda", dtype=torch.bfloat16)
        if not torch.isfinite(sample @ sample).all():
            raise RuntimeError("CUDA BF16 matrix multiplication returned nonfinite values")
        torch.cuda.synchronize()
        del sample
    modules = ["sentence_transformers", "pypdfium2", "pyarrow", "ir_measures", "pyterrier"]
    modules += (
        ["peft", "qdrant_client"]
        if profile == "colqwen"
        else [
            "open_clip",
            "qwen_vl_utils",
            "colbert",
            "jdk4py",
        ]
    )
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            raise RuntimeError(
                f"Cannot import {module} in {profile}: {exc}. "
                "Restore the locked environment; do not upgrade individual packages."
            ) from exc
    if profile == "colqwen":
        from sentence_transformers import SentenceTransformer

        if not hasattr(SentenceTransformer, "encode_document"):
            raise RuntimeError("Installed Sentence Transformers lacks encode_document")
        print("ColQwen dependencies and CUDA verified; weights/inference not tested.", flush=True)
        return

    import faiss
    import numpy as np
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel  # noqa: F401

    from scientific_tot.integrations.faiss_search import FaissSearch

    index = faiss.IndexFlatIP(2)
    vector = np.array([[1, 0]], dtype=np.float32)
    index.add(vector)
    search = FaissSearch(
        index, device=cfg.retrieval.search_device, threads=cfg.launcher.threads.faiss, depth=1
    )
    scores, positions = search.search(vector, 1)
    if positions[0, 0] != 0 or scores[0, 0] != 1:
        raise RuntimeError("FAISS exact inner-product smoke check failed")
    print(
        f"FAISS={getattr(faiss, '__version__', 'unknown')}; verified {search.device} search. "
        "The lock installs faiss-cpu; CUDA encoding does not require GPU FAISS.",
        flush=True,
    )
    from scientific_tot.integrations.colbert import prepare_cuda

    prepare_cuda(cfg)
    print(
        "ColBERT CUDA extensions compiled/loaded. Model inference remains to be tested.", flush=True
    )


def main() -> None:
    if len(sys.argv) < 2:
        raise ValueError(
            "Usage: python -m scientific_tot.h200_check default|colqwen [Hydra key=value ...]"
        )
    cfg, _ = plan("prepare", ["launcher=h200", *sys.argv[2:], "dry_run=true"])
    check(sys.argv[1], cfg)


if __name__ == "__main__":
    main()
