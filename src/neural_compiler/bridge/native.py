"""In-jail C99 and assembly compilation + execution through the supervised bridge.

This is a P1 bridge prerequisite, not a production security boundary: compiled
binaries still require accepted seccomp/cgroup policies and independent
containment qualification before running untrusted workloads.
"""

from __future__ import annotations

import math
import sys
import tempfile
import time
from collections.abc import Set
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import runtime
from .cgroups import CgroupLimits
from .execution import (
    MAX_OUTPUT_BYTES,
    ExecutionResult,
    run_jailed,
    run_python,
    validate_code,
)
from .seccomp import SeccompPolicy

ASM_HARNESS = (
    "#include <stdio.h>\n"
    "long long solve(void);\n"
    'int main(void){printf("%lld\\n", solve());return 0;}\n'
)


def _compile_and_run(
    compile_command: list[str],
    workdir: Path,
    timeout: float,
    *,
    output_limit: int = MAX_OUTPUT_BYTES,
    seccomp: SeccompPolicy | None = None,
    cgroup: CgroupLimits | None = None,
) -> ExecutionResult:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    if output_limit <= 0:
        raise ValueError("output_limit must be positive")
    deadline = time.monotonic() + timeout
    compiled = run_jailed(
        compile_command,
        workdir,
        timeout,
        writable=True,
        output_limit=output_limit,
        cgroup=cgroup,
    )
    if compiled.status != "ok":
        status = (
            "compile_error" if compiled.status == "runtime_error" else compiled.status
        )
        return ExecutionResult(
            status,
            compiled.stdout,
            compiled.stderr,
            compiled.returncode,
        )
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return ExecutionResult(
            "timeout",
            stderr="Compilation exhausted the wall-clock deadline",
        )
    return run_jailed(
        ["/work/prog"],
        workdir,
        remaining,
        writable=False,
        output_limit=output_limit,
        seccomp=seccomp,
        cgroup=cgroup,
    )


CCompiler = Literal["gcc", "tcc"]

_C_COMPILER_COMMANDS: dict[CCompiler, list[str]] = {
    "gcc": [
        "/usr/bin/gcc",
        "-std=c99",
        "-pedantic-errors",
        "-O2",
        "-o",
        "/work/prog",
        "/work/prog.c",
        "-lm",
    ],
    "tcc": [
        "/usr/bin/tcc",
        "-Wall",
        "-Werror",
        "-o",
        "/work/prog",
        "/work/prog.c",
        "-lm",
    ],
}


def run_c(
    code: str,
    timeout: float = 20,
    *,
    compiler: CCompiler = "gcc",
    output_limit: int = MAX_OUTPUT_BYTES,
    seccomp: SeccompPolicy | None = None,
    cgroup: CgroupLimits | None = None,
) -> ExecutionResult:
    """Compile C inside a writable jail (`gcc` C99 or `tcc`), then run read-only."""
    if compiler not in _C_COMPILER_COMMANDS:
        raise ValueError(f"Unsupported C compiler: {compiler!r}")
    validate_code(code)
    with tempfile.TemporaryDirectory(prefix="neural-compiler-c-") as directory:
        work = Path(directory)
        (work / "prog.c").write_text(code, encoding="utf-8")
        return _compile_and_run(
            list(_C_COMPILER_COMMANDS[compiler]),
            work,
            timeout,
            output_limit=output_limit,
            seccomp=seccomp,
            cgroup=cgroup,
        )


def run_asm(
    code: str,
    timeout: float = 20,
    *,
    output_limit: int = MAX_OUTPUT_BYTES,
    seccomp: SeccompPolicy | None = None,
    cgroup: CgroupLimits | None = None,
) -> ExecutionResult:
    """Assemble and link a `long long solve(void)` routine inside the jail."""
    validate_code(code)
    with tempfile.TemporaryDirectory(prefix="neural-compiler-asm-") as directory:
        work = Path(directory)
        (work / "solve.s").write_text(code + "\n", encoding="utf-8")
        (work / "main.c").write_text(ASM_HARNESS, encoding="utf-8")
        return _compile_and_run(
            [
                "/usr/bin/gcc",
                "-no-pie",
                "-o",
                "/work/prog",
                "/work/main.c",
                "/work/solve.s",
            ],
            work,
            timeout,
            output_limit=output_limit,
            seccomp=seccomp,
            cgroup=cgroup,
        )


def _run_jailed_runtime(
    language: str,
    code: str,
    budget: int,
    timeout: float,
    *,
    output_limit: int = MAX_OUTPUT_BYTES,
    seccomp: SeccompPolicy | None = None,
    cgroup: CgroupLimits | None = None,
) -> ExecutionResult:
    validate_code(code)
    if type(budget) is not int or budget <= 0:
        raise ValueError("execution budget must be a positive integer")
    runtime_source = Path(runtime.__file__).read_text(encoding="utf-8")
    with tempfile.TemporaryDirectory(
        prefix=f"neural-compiler-{language}-"
    ) as directory:
        work = Path(directory)
        (work / "runtime.py").write_text(runtime_source, encoding="utf-8")
        (work / "program").write_text(code, encoding="utf-8")
        result = run_jailed(
            [
                sys.executable,
                "-I",
                "/work/runtime.py",
                language,
                "/work/program",
                str(budget),
            ],
            work,
            timeout,
            writable=False,
            output_limit=output_limit,
            runtime_prefix=Path(sys.prefix),
            seccomp=seccomp,
            cgroup=cgroup,
        )
    if result.status == "runtime_error":
        mapped = {2: "compile_error", 3: "timeout"}.get(result.returncode)
        if mapped is not None:
            return ExecutionResult(
                mapped,
                result.stdout,
                result.stderr,
                result.returncode,
            )
    return result


def run_wat(
    code: str,
    fuel: int = 20_000_000_000,
    timeout: float = 20,
    *,
    output_limit: int = MAX_OUTPUT_BYTES,
    seccomp: SeccompPolicy | None = None,
    cgroup: CgroupLimits | None = None,
) -> ExecutionResult:
    """Validate, compile, and execute WAT (`export "solve" -> i64`) in a jailed Wasmtime worker."""
    return _run_jailed_runtime(
        "wat",
        code,
        fuel,
        timeout,
        output_limit=output_limit,
        seccomp=seccomp,
        cgroup=cgroup,
    )


def run_hex(
    code: str,
    max_insns: int = 200_000_000,
    timeout: float = 20,
    *,
    output_limit: int = MAX_OUTPUT_BYTES,
    seccomp: SeccompPolicy | None = None,
    cgroup: CgroupLimits | None = None,
) -> ExecutionResult:
    """Emulate raw x86-64 machine-code hex in a jailed Unicorn process."""
    validate_code(code)
    try:
        blob = bytes.fromhex(code)
    except ValueError:
        return ExecutionResult(
            "compile_error",
            stderr="not an even-length hex string (whitespace only between bytes)",
        )
    if not blob:
        return ExecutionResult("compile_error", stderr="empty hex string")
    return _run_jailed_runtime(
        "hex",
        code,
        max_insns,
        timeout,
        output_limit=output_limit,
        seccomp=seccomp,
        cgroup=cgroup,
    )


Language = Literal["python", "c", "asm", "wat", "hex"]
ALL_LANGUAGES: tuple[Language, ...] = ("python", "c", "asm", "wat", "hex")


@dataclass(frozen=True)
class BridgePolicy:
    """Shared execution and containment configuration across bridge tiers."""

    timeout: float = 20.0
    output_limit: int = MAX_OUTPUT_BYTES
    seccomp: SeccompPolicy | None = None
    cgroup: CgroupLimits | None = None
    allowed_imports: Set[str] | None = None
    allow_unresolved_imports: bool = False
    c_compiler: CCompiler = "gcc"
    wat_fuel: int = 20_000_000_000
    hex_max_insns: int = 200_000_000

    def __post_init__(self) -> None:
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("timeout must be a finite positive number")
        if type(self.output_limit) is not int or self.output_limit <= 0:
            raise ValueError("output_limit must be a positive integer")
        if self.c_compiler not in _C_COMPILER_COMMANDS:
            raise ValueError(f"Unsupported C compiler: {self.c_compiler!r}")
        if type(self.wat_fuel) is not int or self.wat_fuel <= 0:
            raise ValueError("wat_fuel must be a positive integer")
        if type(self.hex_max_insns) is not int or self.hex_max_insns <= 0:
            raise ValueError("hex_max_insns must be a positive integer")


def execute_program(
    language: Language,
    code: str,
    policy: BridgePolicy = BridgePolicy(),
    *,
    timeout: float | None = None,
) -> ExecutionResult:
    """Dispatch `code` to the jailed executor for `language` under `policy`."""
    effective_timeout = policy.timeout if timeout is None else timeout
    if language == "python":
        return run_python(
            code,
            timeout=effective_timeout,
            output_limit=policy.output_limit,
            seccomp=policy.seccomp,
            cgroup=policy.cgroup,
            allowed_imports=policy.allowed_imports,
            allow_unresolved_imports=policy.allow_unresolved_imports,
        )
    if language == "c":
        return run_c(
            code,
            timeout=effective_timeout,
            compiler=policy.c_compiler,
            output_limit=policy.output_limit,
            seccomp=policy.seccomp,
            cgroup=policy.cgroup,
        )
    if language == "asm":
        return run_asm(
            code,
            timeout=effective_timeout,
            output_limit=policy.output_limit,
            seccomp=policy.seccomp,
            cgroup=policy.cgroup,
        )
    if language == "wat":
        return run_wat(
            code,
            fuel=policy.wat_fuel,
            timeout=effective_timeout,
            output_limit=policy.output_limit,
            seccomp=policy.seccomp,
            cgroup=policy.cgroup,
        )
    if language == "hex":
        return run_hex(
            code,
            max_insns=policy.hex_max_insns,
            timeout=effective_timeout,
            output_limit=policy.output_limit,
            seccomp=policy.seccomp,
            cgroup=policy.cgroup,
        )
    raise ValueError(f"Unsupported execution language: {language!r}")
