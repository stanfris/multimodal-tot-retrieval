"""Exact search over FAISS indexes using FAISS or PyTorch CUDA, in bounded query batches."""

import numpy as np

from scientific_tot.reporting import phase, tqdm


class FaissSearch:
    def __init__(self, index, *, device="auto", threads=1, depth=0):
        import faiss

        faiss.omp_set_num_threads(threads)
        self.cpu = index
        self.gpu = None
        self.torch_vectors = None
        self.resources = None  # FAISS GPU indexes borrow their resources.
        self.cpu_queries = 0
        self.gpu_queries = 0
        self.deep_notice = False
        available = hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0
        if device == "torch_cuda":
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    "retrieval.search_device=torch_cuda requires an available CUDA GPU"
                )
            if not isinstance(index, faiss.IndexFlatIP):
                raise ValueError("torch_cuda search requires a FAISS IndexFlatIP")
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            with phase("Load exact FP32 index on CUDA (PyTorch)"):
                self.torch_vectors = torch.as_tensor(index.reconstruct_n(), device="cuda")
            self.device = "cuda"
            tqdm.write("Vector search: CUDA via PyTorch; FP32 exact inner product")
            return
        if device == "cuda" and not available:
            raise RuntimeError(
                "retrieval.search_device=cuda requires a GPU-enabled FAISS installation "
                "and a visible CUDA GPU. Install FAISS GPU in the execution environment."
            )
        if device != "cpu" and available and index.ntotal and depth <= 2048:
            import torch

            torch.cuda.empty_cache()  # Release encoder allocations before FAISS takes VRAM.
            with phase("Load exact FP32 FAISS index on CUDA"):
                self.resources = faiss.StandardGpuResources()
                options = faiss.GpuClonerOptions()
                options.useFloat16 = False
                self.gpu = faiss.index_cpu_to_gpu(self.resources, 0, index, options)
        self.device = "cuda" if self.gpu is not None else "cpu"
        reason = (
            "requested"
            if device == "cpu"
            else (
                "empty index"
                if not index.ntotal
                else "exact depth exceeds GPU top-k limit (2048)"
                if depth > 2048
                else "GPU FAISS unavailable"
            )
        )
        tqdm.write(
            f"FAISS search: {self.device}; FP32 exact inner product"
            + (f" ({reason})" if self.device == "cpu" else "")
        )

    def search(self, vectors, depth):
        if self.torch_vectors is not None:
            import torch

            with torch.inference_mode():
                queries = torch.as_tensor(
                    np.ascontiguousarray(vectors, dtype=np.float32), device="cuda"
                )
                scores, positions = torch.topk(queries @ self.torch_vectors.T, depth, dim=1)
            self.gpu_queries += len(vectors)
            return scores.cpu().numpy(), positions.cpu().numpy()
        # FAISS GPU top-k is limited to 2048. Deep exhaustive passage searches and
        # expanded visual ties retain their exact semantics on the persisted CPU index.
        use_gpu = self.gpu is not None and depth <= 2048
        if self.gpu is not None and not use_gpu and not self.deep_notice:
            tqdm.write("FAISS: depth > 2048; using exact CPU search for deep rankings/ties")
            self.deep_notice = True
        if use_gpu:
            self.gpu_queries += len(vectors)
        else:
            self.cpu_queries += len(vectors)
        index = self.gpu if use_gpu else self.cpu
        return index.search(np.ascontiguousarray(vectors, dtype=np.float32), depth)

    def initial_batches(self, vectors, depth, batch_size):
        # Bound returned score/position buffers for full-corpus passage search.
        size = min(batch_size, max(1, 1_000_000 // max(depth, 1)))
        for start in range(0, len(vectors), size):
            chunk = vectors[start : start + size]
            if depth:
                scores, positions = self.search(chunk, depth)
            else:
                scores = np.empty((len(chunk), 0), dtype=np.float32)
                positions = np.empty((len(chunk), 0), dtype=np.int64)
            yield start, scores, positions

    def report(self):
        return {
            "device": self.device,
            "backend": "torch" if self.torch_vectors is not None else "faiss",
            "precision": "fp32",
            "gpu_query_searches": self.gpu_queries,
            "cpu_query_searches": self.cpu_queries,
            "gpu_max_depth": None if self.torch_vectors is not None else 2048,
        }
