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
    run_hex,
    run_python,
    run_wat,
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
    with pytest.raises(ValueError, match="Unsupported C compiler"):
        run_c("int main(void){return 0;}", compiler="clang")  # type: ignore[arg-type]


def test_run_c_supports_tcc_fast_path_inside_jail(monkeypatch) -> None:
    from neural_compiler.bridge import native

    commands: list[list[str]] = []

    def fake_run_jailed(command, workdir, timeout=20, **kwargs):
        commands.append(list(command))
        return ExecutionResult("ok", stdout="42\n", returncode=0)

    monkeypatch.setattr(native, "run_jailed", fake_run_jailed)
    res = run_c("int main(void){return 0;}", compiler="tcc")
    assert res.status == "ok" and res.stdout.strip() == "42"
    assert commands[0] == [
        "/usr/bin/tcc",
        "-Wall",
        "-Werror",
        "-o",
        "/work/prog",
        "/work/prog.c",
        "-lm",
    ]
    assert commands[1] == ["/work/prog"]


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


def test_run_wat_and_run_hex_jail_contract_and_exit_mapping(
    monkeypatch, tmp_path: Path
) -> None:
    import sys
    from neural_compiler.bridge import native

    seen: list[dict[str, object]] = []

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
        work = Path(workdir)
        seen.append(
            {
                "command": list(command),
                "writable": writable,
                "runtime_prefix": runtime_prefix,
                "program": (work / "program").read_text(encoding="utf-8"),
                "has_runtime": (work / "runtime.py").is_file(),
                "seccomp": seccomp,
                "cgroup": cgroup,
            }
        )
        return ExecutionResult("ok", stdout="42\n", returncode=0)

    monkeypatch.setattr(native, "run_jailed", fake_run_jailed)
    policy = SeccompPolicy(("read", "write", "exit_group"))
    limits = CgroupLimits(tmp_path, 4096, 16)

    wat_res = run_wat('(module (func (export "solve") (result i64) (i64.const 42)))', fuel=5000, seccomp=policy, cgroup=limits)
    assert wat_res.status == "ok" and wat_res.stdout.strip() == "42"
    assert seen[-1]["command"] == [
        sys.executable,
        "-I",
        "/work/runtime.py",
        "wat",
        "/work/program",
        "5000",
    ]
    assert seen[-1]["writable"] is False
    assert seen[-1]["runtime_prefix"] == Path(sys.prefix)
    assert seen[-1]["has_runtime"] is True
    assert seen[-1]["seccomp"] is policy and seen[-1]["cgroup"] is limits

    hex_res = run_hex("b8 2a 00 00 00 c3", max_insns=1000)
    assert hex_res.status == "ok"
    assert seen[-1]["command"][3:] == ["hex", "/work/program", "1000"]

    # Exit codes 2 and 3 map to compile_error and timeout; invalid hex fails before jail.
    monkeypatch.setattr(
        native,
        "run_jailed",
        lambda *a, **kw: ExecutionResult("runtime_error", stderr="parse failed", returncode=2),
    )
    assert run_wat("bad wat").status == "compile_error"

    monkeypatch.setattr(
        native,
        "run_jailed",
        lambda *a, **kw: ExecutionResult("runtime_error", stderr="out of fuel", returncode=3),
    )
    assert run_wat("(module)").status == "timeout"

    monkeypatch.setattr(
        native, "run_jailed", lambda *a, **kw: pytest.fail("Invalid hex must not launch jail")
    )
    assert run_hex("not-hex").status == "compile_error"
    assert run_hex("   ").status == "compile_error"
    with pytest.raises(ValueError, match="positive integer"):
        run_wat("(module)", fuel=0)
