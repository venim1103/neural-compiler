"""Regression coverage for execution supervision and namespace isolation."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from neural_compiler.bridge import run_python
from neural_compiler.bridge.execution import MAX_CODE_BYTES, _supervise, run_jailed
from neural_compiler.cli import main


def supervise(code: str, timeout: float = 2, output_limit: int = 1024):
    # Only fixed, trusted test programs use this unjailed supervisor test seam.
    return _supervise([sys.executable, "-I", "-u", "-c", code], timeout, output_limit)


def test_stdout_cannot_forge_success() -> None:
    result = supervise('print(\'{"status":"ok"}\'); raise ValueError("regression")')
    assert result.status == "runtime_error"
    assert result.stdout.strip() == '{"status":"ok"}'
    assert "ValueError: regression" in result.stderr


@pytest.mark.parametrize("prefix", [
    "import os; os.write(1, b'partial'); ",
    "import os; os.close(1); os.close(2); ",
])
def test_deadline_covers_partial_output_and_closed_pipes(prefix: str) -> None:
    start = time.monotonic()
    result = supervise(prefix + "import time; time.sleep(10)", timeout=0.15)
    assert result.status == "timeout"
    assert result.returncode == -signal.SIGKILL
    assert time.monotonic() - start < 2


@pytest.mark.parametrize("stream", ["1", "2"])
def test_output_is_bounded_on_both_streams(stream: str) -> None:
    result = supervise(f"import os; os.write({stream}, b'x' * 8192)", output_limit=128)
    assert result.status == "output_limit"
    captured_stderr = result.stderr.partition("Output exceeds")[0].rstrip("\n")
    assert len(result.stdout) + len(captured_stderr) <= 128
    assert "Output exceeds 128 bytes" in result.stderr


def test_timeout_kills_and_reaps_child(monkeypatch) -> None:
    spawned = []
    popen = subprocess.Popen

    def capture(*args, **kwargs):
        process = popen(*args, **kwargs)
        spawned.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", capture)
    result = supervise("import time; time.sleep(10)", timeout=0.15)
    assert result.status == "timeout"
    assert spawned[0].returncode == -signal.SIGKILL
    with pytest.raises(ChildProcessError):
        os.waitpid(spawned[0].pid, os.WNOHANG)


def test_no_unjailed_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("neural_compiler.bridge.execution.shutil.which", lambda _: None)
    result = run_jailed(["/usr/bin/true"], tmp_path)
    assert result.status == "sandbox_error"
    assert "bwrap" in result.stderr


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_invalid_timeout_is_explicit(timeout: float, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        run_jailed(["/usr/bin/true"], tmp_path, timeout)


def test_code_limit_counts_utf8_bytes() -> None:
    with pytest.raises(ValueError, match="UTF-8 bytes"):
        run_python("#" + "\u00e9" * (MAX_CODE_BYTES // 2))


@pytest.fixture
def jail():
    probe = run_python("print(42)")
    if probe.status == "sandbox_error":
        pytest.skip(f"bubblewrap namespaces unavailable: {probe.stderr}")
    assert probe.status == "ok", probe


def test_python_stdlib_and_clean_request_state(jail) -> None:
    first = run_python("import math; x = math.comb(40, 20); print(x)")
    assert first.status == "ok"
    assert first.stdout.strip() == "137846528820"
    second = run_python("print(x)")
    assert second.status == "runtime_error"
    assert "NameError" in second.stderr


def test_python_errors_are_not_success(jail) -> None:
    result = run_python("print(42); raise ValueError('regression')")
    assert result.status == "runtime_error"
    assert result.stdout.strip() == "42"
    assert "ValueError: regression" in result.stderr


@pytest.mark.parametrize("extra,status", [(0, "ok"), (1, "output_limit")])
def test_python_output_budget_at_exact_boundary(jail, extra: int, status: str) -> None:
    result = run_python(f"print('x' * {64 * 1024 - 1 + extra})")
    assert result.status == status
    assert len(result.stdout.encode("utf-8")) == 64 * 1024


def test_host_file_is_not_visible(jail, tmp_path: Path) -> None:
    sentinel = tmp_path / "host-only"
    sentinel.write_text("test sentinel", encoding="utf-8")
    result = run_python(f"print(open({str(sentinel)!r}).read())")
    assert result.status == "runtime_error"
    assert "FileNotFoundError" in result.stderr
    assert "test sentinel" not in result.stdout


def test_cli_runs_python_with_structured_failure(jail, tmp_path: Path, capsys) -> None:
    import json

    source = tmp_path / "prog.py"
    source.write_text("raise ValueError('cli regression')", encoding="utf-8")
    assert main(["--run-python", str(source)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "runtime_error"
    assert "cli regression" in result["stderr"]
