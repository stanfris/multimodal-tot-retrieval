"""Shared stage activity, timings and explicitly scoped resource measurements."""

from __future__ import annotations

import platform
import resource
import shutil
import subprocess
import sys
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from importlib.metadata import version
from time import perf_counter, process_time

from tqdm.auto import tqdm

_active = ContextVar("stage_monitor", default=None)


@contextmanager
def phase(description):
    """Name work even when an upstream library cannot expose a progress counter."""
    monitor = _active.get()
    previous = monitor.activity if monitor else None
    if monitor:
        monitor.activity = description
    started = perf_counter()
    tqdm.write(f"▶ {description}")
    try:
        yield
    except BaseException:
        elapsed = perf_counter() - started
        tqdm.write(f"✗ {description} failed after {elapsed:.1f}s")
        if monitor:
            monitor.phases.append({"name": description, "seconds": elapsed, "status": "failed"})
        raise
    else:
        elapsed = perf_counter() - started
        tqdm.write(f"✓ {description} — {elapsed:.1f}s")
        if monitor:
            monitor.phases.append({"name": description, "seconds": elapsed, "status": "completed"})
    finally:
        if monitor:
            monitor.activity = previous


def package_version(name):
    """FAISS CPU and CUDA distributions expose the same importable module."""
    if name == "faiss":
        import faiss

        return faiss.__version__
    return version(name)


class StageMonitor:
    """Heartbeat for opaque library calls; process peaks exclude JVM/worker processes."""

    def __init__(self, cfg):
        self.started = perf_counter()
        self.cpu_started = process_time()
        self.started_at = datetime.now(UTC).isoformat()
        self.activity = f"{cfg.stage.name}: {cfg.model.name}"
        self.phases = []
        self.stop = threading.Event()
        self.samples = []
        self.smi = shutil.which("nvidia-smi")
        self.torch = None
        self.devices = []
        if (
            cfg.stage.name in {"index", "retrieve"}
            and cfg.model.name != "bm25"
            and cfg.launcher.device == "cuda"
        ):
            import torch

            if torch.cuda.is_available():
                self.torch = torch
                self.devices = [torch.cuda.current_device()] if torch.cuda.device_count() else []
                for number in self.devices:
                    torch.cuda.synchronize(number)
                    torch.cuda.reset_peak_memory_stats(number)
        self.token = _active.set(self)
        self.worker = threading.Thread(target=self._heartbeat, daemon=True)
        self.worker.start()

    def _sample_gpu(self):
        if not self.smi or self.torch is None:
            return
        try:
            result = subprocess.run(
                [
                    self.smi,
                    "--query-gpu=uuid,utilization.gpu,memory.used,driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return  # Optional driver telemetry; PyTorch memory peaks remain available.
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                uuid, utilization, memory, driver = (value.strip() for value in line.split(","))
                if utilization.isdigit() and memory.isdigit():
                    self.samples.append(
                        {
                            "uuid": uuid,
                            "utilization_percent": int(utilization),
                            "memory_used_bytes": int(memory) * 1024**2,
                            "driver_version": driver,
                        }
                    )

    def _heartbeat(self):
        ticks = 0
        while not self.stop.wait(10):
            self._sample_gpu()
            ticks += 1
            if ticks % 3 == 0:
                tqdm.write(f"… {self.activity}; stage elapsed {perf_counter() - self.started:.0f}s")

    def finish(self):
        self.stop.set()
        self.worker.join()
        self._sample_gpu()
        _active.reset(self.token)
        gpus = []
        if self.torch is not None:
            for number in self.devices:
                self.torch.cuda.synchronize(number)
                properties = self.torch.cuda.get_device_properties(number)
                gpus.append(
                    {
                        "device": f"cuda:{number}",
                        "name": properties.name,
                        "total_memory_bytes": properties.total_memory,
                        "peak_allocated_bytes": self.torch.cuda.max_memory_allocated(number),
                        "peak_reserved_bytes": self.torch.cuda.max_memory_reserved(number),
                    }
                )
        torch = sys.modules.get("torch")
        return {
            "started_at": self.started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "wall_seconds": perf_counter() - self.started,
            "cpu_seconds": process_time() - self.cpu_started,
            "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            "ram_scope": "Python process lifetime high-water mark; excludes JVM/child processes",
            "gpu_memory_scope": "PyTorch allocator peaks since stage start on the active device; "
            "FAISS/native allocations are visible only in device-wide nvidia-smi samples",
            "phases": self.phases,
            "hardware": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "torch": torch.__version__ if torch else None,
                "cuda_runtime": torch.version.cuda if torch else None,
                "gpus": gpus,
                "sdpa_backends_enabled": {
                    "flash": torch.backends.cuda.flash_sdp_enabled(),
                    "memory_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
                    "math": torch.backends.cuda.math_sdp_enabled(),
                }
                if self.torch is not None
                else None,
            },
            "gpu_samples": self.samples,
            "gpu_sample_scope": "device-wide nvidia-smi samples every 10s and at completion; "
            "includes other processes; unavailable telemetry is an empty list",
        }
