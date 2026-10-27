"""Exact batched results and explicit CUDA/CPU routing without pretrained models."""

from types import SimpleNamespace

import faiss
import numpy as np
import pytest

from scientific_tot.integrations.faiss_search import FaissSearch


def test_batched_search_matches_exact_cpu():
    rng = np.random.default_rng(42)
    index = faiss.IndexFlatIP(8)
    index.add(rng.standard_normal((19, 8)).astype(np.float32))
    queries = rng.standard_normal((5, 8)).astype(np.float32)
    expected_scores, expected_ids = index.search(queries, 7)
    search = FaissSearch(index, device="cpu")
    batches = list(search.initial_batches(queries, 7, 2))
    assert [start for start, _, _ in batches] == [0, 2, 4]
    np.testing.assert_allclose(
        np.concatenate([scores for _, scores, _ in batches]), expected_scores
    )
    np.testing.assert_array_equal(np.concatenate([ids for _, _, ids in batches]), expected_ids)
    assert search.report()["cpu_query_searches"] == 5


def test_unavailable_cuda_is_explicit(monkeypatch, capsys):
    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 0)
    index = faiss.IndexFlatIP(2)
    index.add(np.ones((1, 2), dtype=np.float32))
    assert FaissSearch(index).device == "cpu"
    assert "GPU FAISS unavailable" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="GPU-enabled FAISS"):
        FaissSearch(index, device="cuda")


def test_cuda_routing_preserves_deep_exact_results(monkeypatch, capsys):
    clones = []
    monkeypatch.setattr(faiss, "get_num_gpus", lambda: 1)
    monkeypatch.setattr(faiss, "StandardGpuResources", object, raising=False)
    monkeypatch.setattr(faiss, "GpuClonerOptions", SimpleNamespace, raising=False)

    def clone(resources, device, index, options):
        assert resources is not None and device == 0 and options.useFloat16 is False
        clones.append(index)
        return index  # Exercise routing with real exact FAISS search on this CPU host.

    monkeypatch.setattr(faiss, "index_cpu_to_gpu", clone, raising=False)
    index = faiss.IndexFlatIP(2)
    index.add(np.arange(4100, dtype=np.float32).reshape(2050, 2))
    query = np.ones((1, 2), dtype=np.float32)
    search = FaissSearch(index, device="cuda", depth=2)
    search.search(query, 2)
    actual_scores, actual_ids = search.search(query, 2050)
    expected_scores, expected_ids = index.search(query, 2050)
    np.testing.assert_array_equal(actual_ids, expected_ids)
    np.testing.assert_allclose(actual_scores, expected_scores)
    assert search.report()["gpu_query_searches"] == 1
    assert search.report()["cpu_query_searches"] == 1
    assert "depth > 2048" in capsys.readouterr().out
    assert FaissSearch(index, device="cuda", depth=2050).device == "cpu"
    assert len(clones) == 1  # Avoid copying an unused exhaustive passage index to VRAM.


def test_torch_cuda_exact_search_without_gpu_faiss(monkeypatch):
    import torch

    # Execute the real Torch search operations on CPU; only device placement is stubbed.
    as_tensor = torch.as_tensor
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "as_tensor", lambda data, **kwargs: as_tensor(data, device="cpu"))
    rng = np.random.default_rng(12)
    index = faiss.IndexFlatIP(8)
    index.add(rng.standard_normal((29, 8)).astype(np.float32))
    queries = rng.standard_normal((5, 8)).astype(np.float32)
    search = FaissSearch(index, device="torch_cuda", depth=7)
    batches = list(search.initial_batches(queries, 7, 2))
    scores, positions = index.search(queries, 7)
    np.testing.assert_allclose(np.concatenate([s for _, s, _ in batches]), scores, atol=2e-6)
    np.testing.assert_array_equal(np.concatenate([p for _, _, p in batches]), positions)
    assert search.report()["backend"] == "torch"
    assert search.report()["gpu_query_searches"] == 5


def test_torch_cuda_requires_gpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="available CUDA GPU"):
        FaissSearch(faiss.IndexFlatIP(2), device="torch_cuda")
