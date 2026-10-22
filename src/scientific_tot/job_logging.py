"""Append live job output while retaining the scheduler's stdout/stderr stream."""

from __future__ import annotations

import codecs
import shlex
import subprocess
import traceback
from datetime import UTC, datetime
from pathlib import Path


def log_event(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"[{datetime.now(UTC).isoformat()}] {message}\n")


def run_logged(command: list[str], path: Path, *, label: str, cwd: Path) -> int:
    """Stream both child file descriptors, including native-library output, to an append log."""
    log_event(path, f"{label} worker started")
    exact = shlex.join(command)
    print(f"Stage log: {path}\n$ {exact}", flush=True)
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(f"$ {exact}\n")
        try:
            with subprocess.Popen(
                command,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            ) as process:
                assert process.stdout is not None
                decoder = codecs.getincrementaldecoder(process.stdout.encoding)()
                last = ""
                while chunk := process.stdout.buffer.read1(8192):
                    text = decoder.decode(chunk)
                    print(text, end="", flush=True)
                    stream.write(text)
                    stream.flush()
                    if text:
                        last = text
                stream.write(decoder.decode(b"", final=True))
                if last and not last.endswith("\n"):
                    stream.write("\n")
                code = process.wait()
        except BaseException:
            traceback.print_exc(file=stream)
            stream.flush()
            log_event(path, f"{label} worker failed")
            raise
    log_event(path, f"{label} worker {'completed' if code == 0 else 'failed'} (exit={code})")
    return code
