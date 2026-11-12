"""Unit and jail tests for in-jail C99 and assembly compilation."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from neural_compiler.bridge import (
    ExecutionResult,
    SeccompPolicy,
    run_asm,
    run_c,
    run_python,
)
from neural_compiler.bridge.cgroups import CgroupLimits
from neural_compiler.bridge.execution import MAX_CODE_BYTES


def test_run_c_and_run_asm_command_contract(monkeypatch, tmp_path: Path) -> None:
    from neural_compiler.bridge import native

    calls: list[dict[str, object]] = []
    files_seen: dict[str, str] = {}

    def fake_run_jailed(
        command,
        workdir,
        timeout=20,
        *,
        writable=False,
        output_limit=65536,
        runtime_prefix=None,
        seccomp=None,
        cgroup=None,
    ):
        for entry in Path(workdir).iterdir():
            files_seen[entry.name] = entry.read_text(encoding="utf-8")
        calls.append(
            {
                "command": list(command),
                "writable": writable,
                "timeout": timeout,
                "seccomp": seccomp,
                "cgroup": cgroup,
            }
        )
        if writable:
            return ExecutionResult("ok", returncode=0)
        return ExecutionResult("ok", stdout="42\n", returncode=0)

    monkeypatch.setattr(native, "run_jailed", fake_run_jailed)
    policy = SeccompPolicy(("read", "write", "exit_group"))
    limits = CgroupLimits(tmp_path, 4096, 16)

    c_res = run_c("int main(void){return 0;}", timeout=10, seccomp=policy, cgroup=limits)
    assert c_res.status == "ok" and c_res.stdout.strip() == "42"
    assert len(calls) == 2
    assert calls[0]["command"] == [
        "/usr/bin/gcc",
        "-std=c99",
        "-pedantic-errors",
        "-O2",
        "-o",
        "/work/prog",
        "/work/prog.c",
        "-lm",
    ]
    assert calls[0]["writable"] is True
    assert calls[0]["seccomp"] is None
    assert calls[0]["cgroup"] is limits
    assert calls[1]["command"] == ["/work/prog"]
    assert calls[1]["writable"] is False
    assert calls[1]["seccomp"] is policy
    assert calls[1]["cgroup"] is limits

    calls.clear()
    files_seen.clear()
    asm_res = run_asm("mov eax, 42\nret", timeout=10)
    assert asm_res.status == "ok"
    assert "solve.s" in files_seen and "main.c" in files_seen
    assert calls[0]["command"] == [
        "/usr/bin/gcc",
        "-no-pie",
        "-o",
        "/work/prog",
        "/work/main.c",
        "/work/solve.s",
    ]


def test_run_c_maps_compile_failure_and_deadline_exhaustion(monkeypatch) -> None:
    from neural_compiler.bridge import native

    def fail_compile(command, workdir, timeout=20, **kwargs):
        if kwargs.get("writable"):
            return ExecutionResult("runtime_error", stderr="syntax error", returncode=1)
        pytest.fail("Compiled binary must not run when compilation fails")

    monkeypatch.setattr(native, "run_jailed", fail_compile)
    err = run_c("not valid c")
    assert err.status == "compile_error"
    assert "syntax error" in err.stderr

    clock = iter([10.0, 25.0])
    monkeypatch.setattr(native.time, "monotonic", lambda: next(clock))

    def slow_compile(command, workdir, timeout=20, **kwargs):
        if kwargs.get("writable"):
            return ExecutionResult("ok", returncode=0)
        pytest.fail("Binary must not launch after compilation exhausts deadline")

    monkeypatch.setattr(native, "run_jailed", slow_compile)
    timed_out = run_c("int main(void){return 0;}", timeout=5)
    assert timed_out.status == "timeout"
    assert "deadline" in timed_out.stderr.lower()


def test_run_c_validates_inputs_before_jail(monkeypatch) -> None:
    from neural_compiler.bridge import native

    monkeypatch.setattr(
        native, "run_jailed", lambda *a, **kw: pytest.fail("Must validate first")
    )
    with pytest.raises(ValueError, match="UTF-8"):
        run_c("/*" + "a" * MAX_CODE_BYTES + "*/")
    with pytest.raises(ValueError, match="finite positive"):
        run_c("int main(void){return 0;}", timeout=0)
    with pytest.raises(ValueError, match="output_limit"):
        run_c("int main(void){return 0;}", output_limit=0)


def test_native_bridge_real_jail_when_toolchain_available() -> None:
    if shutil.which("bwrap") is None or shutil.which("gcc") is None:
        pytest.skip("bwrap or gcc unavailable on runner")
    probe = run_python("print(0)")
    if probe.status == "sandbox_error":
        pytest.skip(f"bubblewrap namespaces unavailable: {probe.stderr}")
    c_res = run_c('#include <stdio.h>\nint main(void){printf("42\\n");return 0;}')
    assert c_res.status == "ok" and c_res.stdout.strip() == "42"
    asm_res = run_asm(".intel_syntax noprefix\n.global solve\nsolve:\nmov eax, 42\nret")
    assert asm_res.status == "ok" and asm_res.stdout.strip() == "42"
