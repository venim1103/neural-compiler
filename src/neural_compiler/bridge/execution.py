"""Per-request Linux namespace execution.

This is a P0 foundation, not a production security boundary: accepted policies and
independent containment qualification are still required for untrusted workloads.
"""

from __future__ import annotations

import math
import os
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
from collections.abc import Set
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .cgroups import CgroupError, CgroupLimits, request_cgroup
from .seccomp import SeccompError, SeccompPolicy, export_filter

Status = Literal[
    "ok",
    "compile_error",
    "runtime_error",
    "timeout",
    "output_limit",
    "sandbox_error",
]


@dataclass(frozen=True)
class ExecutionResult:
    status: Status
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None


MAX_CODE_BYTES = 64 * 1024
MAX_OUTPUT_BYTES = 64 * 1024
MEMORY_BYTES = 512 * 1024 * 1024

_LAUNCHER = """
import os, resource, sys
resource.setrlimit(resource.RLIMIT_AS, (int(sys.argv[1]), int(sys.argv[1])))
resource.setrlimit(resource.RLIMIT_CPU, (int(sys.argv[2]), int(sys.argv[2]) + 1))
resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
os.execv(sys.argv[3], sys.argv[3:])
"""

_CGROUP_LAUNCHER = """
import os, sys
membership, status = map(int, sys.argv[1:3])
try:
    os.set_inheritable(status, False)
    if os.write(membership, b"0") != 1:
        raise OSError("Incomplete cgroup membership write")
    os.close(membership)
    os.execv(sys.argv[3], sys.argv[3:])
except OSError as exc:
    os.write(status, ("Cgroup setup failed: " + str(exc)).encode("utf-8")[:4096])
    os._exit(125)
"""


def validate_code(code: str) -> None:
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        raise ValueError(f"Program exceeds {MAX_CODE_BYTES} UTF-8 bytes")


def _supervise(
    command: list[str],
    timeout: float,
    output_limit: int,
    pass_fds: tuple[int, ...] = (),
) -> ExecutionResult:
    deadline = time.monotonic() + timeout
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            pass_fds=pass_fds,
        )
    except OSError as exc:
        return ExecutionResult("sandbox_error", stderr=str(exc))

    assert process.stdout is not None and process.stderr is not None
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    status: Status = "ok"
    diagnostic = ""
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status, diagnostic = "timeout", "Wall-clock deadline exceeded"
                    break
                for key, _ in selector.select(remaining):
                    chunk = os.read(key.fileobj.fileno(), 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    buffers[key.data].extend(chunk[: max(0, output_limit - total)])
                    total += len(chunk)
                    if total > output_limit:
                        status, diagnostic = (
                            "output_limit",
                            f"Output exceeds {output_limit} bytes",
                        )
                        break
                if status != "ok":
                    break
        if status == "ok":
            try:
                process.wait(timeout=max(0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                status, diagnostic = "timeout", "Wall-clock deadline exceeded"
    finally:
        # On timeout or supervisor failure, kill before reaping so this PID cannot
        # be reused while it is still being used as the process-group identifier.
        if process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        process.stdout.close()
        process.stderr.close()

    stdout = buffers["stdout"].decode("utf-8", errors="replace")
    stderr = buffers["stderr"].decode("utf-8", errors="replace")
    if diagnostic:
        stderr = f"{stderr}\n{diagnostic}".strip()
    elif process.returncode:
        if stderr.startswith("bwrap:"):
            status = "sandbox_error"
        elif process.returncode == -signal.SIGXCPU:
            status = "timeout"
            stderr = f"{stderr}\nCPU time limit exceeded".strip()
        else:
            status = "runtime_error"
            if not stderr:
                stderr = f"Process exited with code {process.returncode}"
    return ExecutionResult(status, stdout, stderr, process.returncode)


def run_jailed(
    command: list[str],
    workdir: Path,
    timeout: float = 20,
    *,
    writable: bool = False,
    output_limit: int = MAX_OUTPUT_BYTES,
    runtime_prefix: Path | None = None,
    seccomp: SeccompPolicy | None = None,
    cgroup: CgroupLimits | None = None,
) -> ExecutionResult:
    """Run a trusted command specification against guest files in a private jail.

    Guest stdout is data, never a status/control protocol. There is deliberately
    no unjailed fallback if bubblewrap or user namespaces are unavailable.
    Optional cgroups require an existing caller-delegated subtree; this function
    never enables controllers or changes host delegation.
    """
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be a finite positive number")
    if output_limit <= 0:
        raise ValueError("output_limit must be positive")
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return ExecutionResult("sandbox_error", stderr="bubblewrap (bwrap) is required")
    args = [
        bwrap,
        "--ro-bind",
        "/usr",
        "/usr",
        "--symlink",
        "usr/lib",
        "/lib",
        "--symlink",
        "usr/lib64",
        "/lib64",
        "--symlink",
        "usr/bin",
        "/bin",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--unshare-all",
        "--die-with-parent",
        "--clearenv",
        "--setenv",
        "PATH",
        "/usr/bin:/bin",
    ]
    if runtime_prefix is not None and runtime_prefix != Path("/usr"):
        args += ["--ro-bind", str(runtime_prefix), str(runtime_prefix)]
    args += [
        "--bind" if writable else "--ro-bind",
        str(workdir.resolve()),
        "/work",
        "--chdir",
        "/work",
    ]
    result: ExecutionResult | None = None
    try:
        with (
            request_cgroup(cgroup)
            if cgroup is not None
            else nullcontext(None) as group_fds,
            export_filter(seccomp)
            if seccomp is not None
            else nullcontext(None) as filter_fd,
        ):
            if filter_fd is not None:
                args += ["--seccomp", str(filter_fd)]
            args += [
                "/usr/bin/python3",
                "-I",
                "-c",
                _LAUNCHER,
                str(MEMORY_BYTES),
                str(math.ceil(timeout)),
                *command,
            ]
            inherited = () if filter_fd is None else (filter_fd,)
            if group_fds is not None:
                # Join before bubblewrap can spawn guest processes. The status fd
                # closes on exec, so guest output cannot forge setup diagnostics.
                args = [
                    "/usr/bin/python3",
                    "-I",
                    "-c",
                    _CGROUP_LAUNCHER,
                    *(str(fd) for fd in group_fds),
                    *args,
                ]
                inherited += group_fds
            result = _supervise(args, timeout, output_limit, pass_fds=inherited)
            if group_fds is not None:
                os.lseek(group_fds[1], 0, os.SEEK_SET)
                diagnostic = os.read(group_fds[1], 4096)
                if diagnostic:
                    result = ExecutionResult(
                        "sandbox_error",
                        result.stdout,
                        diagnostic.decode("utf-8", errors="replace"),
                        result.returncode,
                    )
        return result
    except (SeccompError, CgroupError) as exc:
        return ExecutionResult(
            "sandbox_error",
            "" if result is None else result.stdout,
            str(exc) if result is None else f"{result.stderr}\n{exc}".strip(),
            None if result is None else result.returncode,
        )


def run_python(
    code: str,
    timeout: float = 20,
    *,
    seccomp: SeccompPolicy | None = None,
    cgroup: CgroupLimits | None = None,
    allowed_imports: Set[str] | None = None,
    allow_unresolved_imports: bool = False,
) -> ExecutionResult:
    """Execute Python in fresh globals with bounded output and a hard deadline."""
    validate_code(code)
    if allowed_imports is not None:
        from .imports import evaluate_import_policy

        try:
            report = evaluate_import_policy(
                code,
                allowed_imports,
                allow_unresolved=allow_unresolved_imports,
            )
        except SyntaxError as exc:
            return ExecutionResult(
                "sandbox_error",
                stderr=f"Import policy syntax check failed: {exc}",
            )
        if not report.compliant:
            blocked = [site.module for site in (*report.disallowed, *report.unresolved)]
            return ExecutionResult(
                "sandbox_error",
                stderr=f"Blocked by static import policy: {', '.join(blocked)}",
            )
    with tempfile.TemporaryDirectory(prefix="neural-compiler-") as directory:
        work = Path(directory)
        (work / "prog.py").write_text(code, encoding="utf-8")
        return run_jailed(
            ["/usr/bin/python3", "-I", "-u", "/work/prog.py"],
            work,
            timeout,
            seccomp=seccomp,
            cgroup=cgroup,
        )
