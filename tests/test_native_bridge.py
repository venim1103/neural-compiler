"""Unit and jail tests for in-jail C99 and assembly compilation."""

from __future__ import annotations

import ctypes
import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_compiler.bridge import (
    BridgePolicy,
    ExecutionResult,
    SeccompPolicy,
    execute_program,
    run_asm,
    run_c,
    run_hex,
    run_python,
    run_wat,
)
from neural_compiler.bridge.cgroups import CgroupLimits
from neural_compiler.bridge.execution import MAX_CODE_BYTES


@pytest.mark.parametrize(
    "runner,program",
    [
        (run_python, "print(42)"),
        (run_c, "int main(void){return 0;}"),
        (run_asm, "mov eax, 42\nret"),
        (run_wat, '(module (func (export "solve") (result i64) i64.const 42))'),
        (run_hex, "b8 2a 00 00 00 c3"),
    ],
)
def test_boolean_timeout_fails_before_subprocess_launch(monkeypatch, runner, program):
    def forbid_launch(*args, **kwargs):
        raise AssertionError("Boolean timeouts must fail before subprocess launch")

    monkeypatch.setattr(
        "neural_compiler.bridge.execution.subprocess.Popen", forbid_launch
    )
    with pytest.raises(ValueError, match="finite positive"):
        runner(program, timeout=True)


def test_bridge_policy_rejects_boolean_timeout():
    with pytest.raises(ValueError, match="finite positive"):
        BridgePolicy(timeout=True)


@pytest.mark.parametrize(
    "field,counter", [("wat_fuel", ctypes.c_uint64), ("hex_max_insns", ctypes.c_size_t)]
)
@pytest.mark.parametrize("overflow", [0, 1], ids=["maximum", "overflow"])
def test_bridge_policy_matches_runtime_counter_range(field, counter, overflow):
    budget = (1 << (8 * ctypes.sizeof(counter))) - 1 + overflow
    if overflow:
        with pytest.raises(ValueError, match=rf"{field}.*counter"):
            BridgePolicy(**{field: budget})
    else:
        BridgePolicy(**{field: budget})


@pytest.mark.parametrize(
    "runner,program,budget_name,counter,module",
    [
        pytest.param(
            run_wat,
            '(module (func (export "solve") (result i64) i64.const 42))',
            "fuel",
            ctypes.c_uint64,
            "wasmtime",
            id="wasm",
        ),
        pytest.param(
            run_hex,
            "b8 2a 00 00 00 c3",
            "max_insns",
            ctypes.c_size_t,
            "unicorn",
            id="hex",
        ),
    ],
)
@pytest.mark.parametrize(
    "overflow", [0, 1, 2], ids=["maximum", "wrap-zero", "wrap-one"]
)
def test_runtime_budget_matches_unsigned_counter_range(
    monkeypatch, runner, program, budget_name, counter, module, overflow
):
    budget = (1 << (8 * ctypes.sizeof(counter))) - 1 + overflow
    if overflow:
        monkeypatch.setattr(
            "neural_compiler.bridge.native.tempfile.TemporaryDirectory",
            lambda **_: pytest.fail("Overflowing counter reached workspace setup"),
        )
        with pytest.raises(ValueError, match="counter"):
            runner(program, timeout=3, output_limit=1024, **{budget_name: budget})
        return
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} runtime dependency unavailable; no unjailed fallback")
    probe = run_python("print(0)", timeout=3, output_limit=1024)
    if probe.status == "sandbox_error":
        pytest.skip(f"bubblewrap namespaces unavailable: {probe.stderr}")
    assert probe.status == "ok", probe
    result = runner(program, timeout=3, output_limit=1024, **{budget_name: budget})
    assert result.status == "ok" and result.stdout == "42\n", result


@pytest.mark.parametrize("extra,status", [(0, "ok"), (1, "output_limit")])
def test_python_dispatch_honors_custom_output_budget(extra: int, status: str) -> None:
    probe = run_python("print(0)")
    if probe.status == "sandbox_error":
        pytest.skip(f"bubblewrap namespaces unavailable: {probe.stderr}")
    assert probe.status == "ok", probe
    result = execute_program(
        "python", f"print('x' * {127 + extra})", BridgePolicy(output_limit=128)
    )
    assert result.status == status
    assert len(result.stdout.encode("utf-8")) == 128


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

    c_res = run_c(
        "int main(void){return 0;}", timeout=10, seccomp=policy, cgroup=limits
    )
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

    clock = iter([10.0, 10.0, 25.0])
    monkeypatch.setattr(native, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    compiled = []

    def slow_compile(command, workdir, timeout=20, **kwargs):
        if kwargs.get("writable"):
            compiled.append(command)
            return ExecutionResult("ok", returncode=0)
        pytest.fail("Binary must not launch after compilation exhausts deadline")

    monkeypatch.setattr(native, "run_jailed", slow_compile)
    timed_out = run_c("int main(void){return 0;}", timeout=5)
    assert timed_out.status == "timeout"
    assert "Compilation exhausted" in timed_out.stderr
    assert len(compiled) == 1


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


@pytest.mark.parametrize(
    "execute,code,write_delay,expected_timeouts",
    [
        (run_c, "int main(void){return 0;}", 11, []),
        (run_asm, "mov eax, 42\nret", 1, [8, 6]),
        (
            run_wat,
            '(module (func (export "solve") (result i64) (i64.const 42)))',
            1,
            [8],
        ),
        (run_hex, "b8 2a 00 00 00 c3", 6, []),
    ],
    ids=["c", "asm", "wat", "hex"],
)
def test_native_preparation_consumes_shared_deadline(
    monkeypatch, execute, code, write_delay, expected_timeouts
):
    from neural_compiler.bridge import native

    now = 0
    received = []
    write_text = Path.write_text

    def delayed_write(path, *args, **kwargs):
        nonlocal now
        result = write_text(path, *args, **kwargs)
        now += write_delay
        return result

    def fake_run_jailed(command, workdir, timeout=20, **kwargs):
        nonlocal now
        received.append(timeout)
        if kwargs.get("writable"):
            now += 2
        return ExecutionResult("ok", stdout="42\n", returncode=0)

    monkeypatch.setattr(native, "time", SimpleNamespace(monotonic=lambda: now))
    monkeypatch.setattr(Path, "write_text", delayed_write)
    monkeypatch.setattr(native, "run_jailed", fake_run_jailed)
    result = execute(code, timeout=10)
    assert received == expected_timeouts
    assert result.status == ("ok" if expected_timeouts else "timeout")
    if not expected_timeouts:
        assert "setup" in result.stderr.lower()


@pytest.mark.parametrize(
    "execute,stage",
    [(run_python, "write"), (run_c, "directory"), (run_wat, "runtime")],
)
def test_preparation_io_failures_are_explicit_without_launch(
    monkeypatch, execute, stage
):
    from neural_compiler.bridge import execution, native

    def fail_io(*args, **kwargs):
        raise OSError("fixture preparation failure")

    def reject_launch(*args, **kwargs):
        pytest.fail("Preparation failure must not launch a child")

    if stage == "write":
        monkeypatch.setattr(Path, "write_text", fail_io)
    elif stage == "directory":
        monkeypatch.setattr(native.tempfile, "TemporaryDirectory", fail_io)
    else:
        monkeypatch.setattr(Path, "read_text", fail_io)
    monkeypatch.setattr(execution, "run_jailed", reject_launch)
    monkeypatch.setattr(native, "run_jailed", reject_launch)
    result = execute("print(42)")
    assert result.status == "sandbox_error"
    assert "fixture preparation failure" in result.stderr
    assert result.returncode is None


def test_workspace_cleanup_failure_preserves_completed_result(monkeypatch):
    from contextlib import contextmanager
    from neural_compiler.bridge import execution

    temporary_directory = execution.tempfile.TemporaryDirectory

    @contextmanager
    def failing_cleanup(*args, **kwargs):
        with temporary_directory(*args, **kwargs) as directory:
            yield directory
            raise OSError("fixture cleanup failure")

    monkeypatch.setattr(execution.tempfile, "TemporaryDirectory", failing_cleanup)
    monkeypatch.setattr(
        execution,
        "run_jailed",
        lambda *a, **kw: ExecutionResult("ok", "42\n", "guest diagnostic", 0),
    )
    result = run_python("print(42)")
    assert result.status == "sandbox_error"
    assert result.stdout == "42\n"
    assert result.returncode == 0
    assert "guest diagnostic" in result.stderr
    assert "fixture cleanup failure" in result.stderr


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


@pytest.mark.parametrize(
    "runner,program,dependency,budget",
    [
        (
            run_wat,
            '(module (func (export "solve") (result i64) i64.const 42))',
            "wasmtime",
            {"fuel": 1000},
        ),
        (run_hex, "b8 2a 00 00 00 c3", "unicorn", {"max_insns": 100}),
    ],
)
@pytest.mark.parametrize("failure", ["missing", "loader"])
def test_jailed_runtime_import_failures_are_infrastructure_errors(
    monkeypatch, runner, program, dependency, budget, failure
):
    from neural_compiler.bridge import native

    probe = run_python("print(0)", timeout=3, output_limit=1024)
    if probe.status == "sandbox_error":
        pytest.skip(f"bubblewrap namespaces unavailable: {probe.stderr}")
    assert probe.status == "ok", probe
    run_jailed = native.run_jailed
    guard = f"""
import builtins, runpy
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name == {dependency!r}:
        if {failure!r} == "missing":
            raise ModuleNotFoundError("fixture dependency unavailable", name=name)
        raise OSError("fixture binding unavailable")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
runpy.run_path("/work/runtime.py", run_name="__main__")
"""

    def guarded_worker(command, workdir, timeout=20, **options):
        assert command[2] == "/work/runtime.py"
        (Path(workdir) / "importguard.py").write_text(guard, encoding="utf-8")
        return run_jailed(
            [*command[:2], "/work/importguard.py", *command[3:]],
            workdir,
            timeout,
            **options,
        )

    monkeypatch.setattr(native, "run_jailed", guarded_worker)
    result = runner(program, timeout=3, output_limit=2048, **budget)
    assert result.status == "sandbox_error" and result.returncode == 4, result
    assert result.stdout == ""
    assert dependency in result.stderr.lower()
    assert "fixture" in result.stderr and "unavailable" in result.stderr
    assert "Traceback" not in result.stderr


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

    wat_res = run_wat(
        '(module (func (export "solve") (result i64) (i64.const 42)))',
        fuel=5000,
        seccomp=policy,
        cgroup=limits,
    )
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
        lambda *a, **kw: ExecutionResult(
            "runtime_error", stderr="parse failed", returncode=2
        ),
    )
    assert run_wat("bad wat").status == "compile_error"

    monkeypatch.setattr(
        native,
        "run_jailed",
        lambda *a, **kw: ExecutionResult(
            "runtime_error", stderr="out of fuel", returncode=3
        ),
    )
    assert run_wat("(module)").status == "timeout"

    monkeypatch.setattr(
        native,
        "run_jailed",
        lambda *a, **kw: pytest.fail("Invalid hex must not launch jail"),
    )
    assert run_hex("not-hex").status == "compile_error"
    assert run_hex("   ").status == "compile_error"
    with pytest.raises(ValueError, match="positive integer"):
        run_wat("(module)", fuel=0)
