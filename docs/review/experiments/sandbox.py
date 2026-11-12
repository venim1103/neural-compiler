"""Sandboxed executors for model-generated programs (used by exp_loop.py).

Python/C/asm run natively inside bubblewrap (no network, read-only /usr, private
/tmp, new PID/IPC/UTS namespaces); WAT and raw machine-code hex use separate
jailed wasmtime/unicorn processes. Each returns (status, output).

PROTOTYPE, NOT SAFE FOR UNTRUSTED CODE. Python/C/asm now use the supervised
per-request bridge, with resource limits, bounded output and in-jail compilation.
There is no warm guest worker or persistent REPL. Remaining P1 requirements:
seccomp, per-request cgroups/PID caps, and an independent containment review.
"""

import os
import sys
import tempfile
import time
from pathlib import Path

from neural_compiler.bridge.execution import run_jailed, run_python as execute_python, validate_code
from neural_compiler.bridge import runtime

HARNESS = '#include <stdio.h>\nlong long solve(void);\nint main(void){printf("%lld\\n", solve());return 0;}\n'


def _jailed(cmd, workdir, timeout):
    result = run_jailed(cmd, Path(workdir), timeout)
    return result.status, result.stdout.strip() if result.status == "ok" else (result.stderr or result.stdout)[-300:]


def run_python(code, timeout=20):
    result = execute_python(code, timeout)
    return result.status, result.stdout.strip() if result.status == "ok" else (result.stderr or result.stdout)[-300:]


def _compile_and_run(command, workdir, timeout):
    deadline = time.monotonic() + timeout
    result = run_jailed(command, Path(workdir), timeout, writable=True)
    if result.status != "ok":
        status = "compile_error" if result.status == "runtime_error" else result.status
        return status, (result.stderr or result.stdout)[-300:]
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return "timeout", "Compilation exhausted the wall-clock deadline"
    return _jailed(["/work/prog"], workdir, remaining)


def run_c(code, timeout=20):
    validate_code(code)
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "prog.c")
        with open(src, "w") as f:
            f.write(code)
        return _compile_and_run(
            ["/usr/bin/gcc", "-std=c99", "-pedantic-errors", "-O2", "-o", "/work/prog", "/work/prog.c", "-lm"],
            td, timeout,
        )


def run_asm(code, timeout=20):
    validate_code(code)
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "solve.s"), "w") as f:
            f.write(code + "\n")
        with open(os.path.join(td, "main.c"), "w") as f:
            f.write(HARNESS)
        return _compile_and_run(
            ["/usr/bin/gcc", "-no-pie", "-o", "/work/prog", "/work/main.c", "/work/solve.s"],
            td, timeout,
        )


def _run_runtime(language, code, budget, timeout):
    validate_code(code)
    if budget <= 0:
        raise ValueError("execution budget must be positive")
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        (work / "runtime.py").write_text(Path(runtime.__file__).read_text(encoding="utf-8"), encoding="utf-8")
        (work / "program").write_text(code, encoding="utf-8")
        result = run_jailed(
            [sys.executable, "-I", "/work/runtime.py", language, "/work/program", str(budget)],
            work, timeout, runtime_prefix=Path(sys.prefix),
        )
    status = result.status
    if status == "runtime_error":
        status = {2: "compile_error", 3: "timeout"}.get(result.returncode, status)
    return status, result.stdout.strip() if status == "ok" else (result.stderr or result.stdout)[-300:]


def run_wat(code, fuel=20_000_000_000, timeout=20):
    return _run_runtime("wat", code, fuel, timeout)


def run_hex(code, max_insns=200_000_000, timeout=20):
    validate_code(code)
    try:
        blob = bytes.fromhex(code)
    except ValueError:
        return "compile_error", "not an even-length hex string (whitespace only between bytes)"
    if not blob:
        return "compile_error", "empty hex string"
    return _run_runtime("hex", code, max_insns, timeout)


EXECUTORS = {"python": run_python, "c": run_c, "wat": run_wat, "asm": run_asm, "hex": run_hex}
