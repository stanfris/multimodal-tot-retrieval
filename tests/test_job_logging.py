"""Job logs retain failures, native stderr, and output visible before completion."""

import subprocess
import sys
import time

from scientific_tot.job_logging import run_logged


def test_output_stderr_failure_and_append(tmp_path, capsys):
    log = tmp_path / "index.log"
    log.write_text("previous attempt\n")
    command = [
        sys.executable,
        "-u",
        "-c",
        "import os; print('progress'); os.write(2, b'native warning\\n'); "
        "raise ValueError('encoding failed')",
    ]
    code = run_logged(command, log, label="index", cwd=tmp_path)
    assert code != 0
    text = log.read_text()
    assert text.startswith("previous attempt\n")
    for message in (
        "worker started",
        "progress",
        "native warning",
        "Traceback",
        "ValueError: encoding failed",
        "worker failed",
    ):
        assert message in text
    assert "native warning" in capsys.readouterr().out


def test_log_is_visible_while_child_is_running(tmp_path):
    log = tmp_path / "index.log"
    release = tmp_path / "release"
    code = (
        "import pathlib, time; print('first batch', end='', flush=True); "
        f"release=pathlib.Path({str(release)!r}); "
        "exec('while not release.exists():\\n time.sleep(0.02)')"
    )
    script = (
        "from pathlib import Path; from scientific_tot.job_logging import run_logged; "
        f"raise SystemExit(run_logged({[sys.executable, '-u', '-c', code]!r}, "
        f"Path({str(log)!r}), label='index', cwd=Path({str(tmp_path)!r})))"
    )
    with subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as process:
        try:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                if log.exists() and "first batch" in log.read_text():
                    break
                time.sleep(0.02)
            assert log.exists() and "first batch" in log.read_text()
            assert process.poll() is None
            release.touch()
            output, _ = process.communicate(timeout=10)
            assert process.returncode == 0, output
            assert "worker completed (exit=0)" in log.read_text()
        finally:
            release.touch()
            if process.poll() is None:
                process.kill()
