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


def test_run_jailed_command_assembly_and_fd_plumbing(monkeypatch, tmp_path: Path) -> None:
    from contextlib import contextmanager
    from neural_compiler.bridge import execution
    from neural_compiler.bridge.cgroups import CgroupLimits
    from neural_compiler.bridge.seccomp import SeccompPolicy

    recorded: list[tuple[list[str], tuple[int, ...]]] = []

    @contextmanager
    def fake_filter(_policy):
        yield 17

    @contextmanager
    def fake_cgroup(_limits):
        m_fd = os.memfd_create("test-cgroup-member", os.MFD_CLOEXEC)
        s_fd = os.memfd_create("test-cgroup-status", os.MFD_CLOEXEC)
        try:
            yield m_fd, s_fd
        finally:
            os.close(m_fd)
            os.close(s_fd)

    def fake_supervise(command, timeout, output_limit, pass_fds=()):
        recorded.append((list(command), tuple(pass_fds)))
        return execution.ExecutionResult("ok", "done\n", returncode=0)

    monkeypatch.setattr(execution.shutil, "which", lambda _: "/usr/bin/bwrap")
    monkeypatch.setattr(execution, "export_filter", fake_filter)
    monkeypatch.setattr(execution, "request_cgroup", fake_cgroup)
    monkeypatch.setattr(execution, "_supervise", fake_supervise)

    res_ro = run_jailed(
        ["/usr/bin/python3", "/work/a.py"],
        tmp_path,
        writable=False,
        runtime_prefix=Path("/usr"),
    )
    assert res_ro.status == "ok"
    cmd_ro, fds_ro = recorded[-1]
    assert fds_ro == ()
    assert "--ro-bind" in cmd_ro and "--bind" not in cmd_ro
    assert cmd_ro.count("/usr") == 2

    res_rw = run_jailed(
        ["/work/prog"],
        tmp_path,
        writable=True,
        runtime_prefix=Path("/opt/venv"),
        seccomp=SeccompPolicy(("read", "write", "exit_group")),
        cgroup=CgroupLimits(tmp_path, 4096, 16),
    )
    assert res_rw.status == "ok"
    cmd_rw, fds_rw = recorded[-1]
    assert len(fds_rw) == 3 and fds_rw[0] == 17
    assert cmd_rw[:4] == ["/usr/bin/python3", "-I", "-c", execution._CGROUP_LAUNCHER]
    assert "/usr/bin/bwrap" in cmd_rw
    bind_idx = cmd_rw.index("--bind")
    assert cmd_rw[bind_idx + 1 : bind_idx + 3] == [str(tmp_path.resolve()), "/work"]
    assert "/opt/venv" in cmd_rw
    seccomp_idx = cmd_rw.index("--seccomp")
    assert cmd_rw[seccomp_idx + 1] == "17"


def test_supervise_classifies_bwrap_and_sigxcpu_failures() -> None:
    bwrap_res = _supervise(
        [
            sys.executable,
            "-I",
            "-c",
            "import sys; sys.stderr.write('bwrap: uid map failed\\n'); raise SystemExit(1)",
        ],
        timeout=2,
        output_limit=1024,
    )
    assert bwrap_res.status == "sandbox_error"
    assert "bwrap:" in bwrap_res.stderr

    xcpu_res = _supervise(
        [
            sys.executable,
            "-I",
            "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGXCPU)",
        ],
        timeout=2,
        output_limit=1024,
    )
    assert xcpu_res.status == "timeout"
    assert "CPU time limit exceeded" in xcpu_res.stderr


def test_run_python_static_import_policy_prefilter(monkeypatch) -> None:
    from neural_compiler.bridge import execution
    from neural_compiler.bridge.imports import CANDIDATE_MATH_MODULES

    monkeypatch.setattr(
        execution, "run_jailed", lambda *a, **kw: pytest.fail("Must not launch jail")
    )
    disallowed = run_python(
        "import subprocess\nprint(1)\n", allowed_imports=CANDIDATE_MATH_MODULES
    )
    assert disallowed.status == "sandbox_error"
    assert "subprocess" in disallowed.stderr

    unresolved = run_python(
        "__import__('os')\n", allowed_imports=CANDIDATE_MATH_MODULES
    )
    assert unresolved.status == "sandbox_error"
    assert "__import__" in unresolved.stderr

    syntax_err = run_python(
        "def broken(:\n", allowed_imports=CANDIDATE_MATH_MODULES
    )
    assert syntax_err.status == "sandbox_error"
    assert "syntax" in syntax_err.stderr.lower()

    launched: list[list[str]] = []
    monkeypatch.setattr(
        execution,
        "run_jailed",
        lambda cmd, *a, **kw: launched.append(list(cmd)) or execution.ExecutionResult("ok", "6\n"),
    )
    allowed = run_python(
        "import math\nprint(math.factorial(3))\n",
        allowed_imports=CANDIDATE_MATH_MODULES,
    )
    assert allowed.status == "ok" and allowed.stdout.strip() == "6"
    assert len(launched) == 1
