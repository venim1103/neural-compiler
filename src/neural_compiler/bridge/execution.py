"""Per-request Linux namespace execution.

This is a P0 foundation, not a production security boundary: an accepted syscall
policy and per-request cgroups are still required before accepting untrusted workloads.
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
from dataclasses import dataclass
from contextlib import nullcontext
from pathlib import Path
from typing import Literal

from .seccomp import SeccompError, SeccompPolicy, export_filter

Status = Literal["ok", "runtime_error", "timeout", "output_limit", "sandbox_error"]


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
) -> ExecutionResult:
    """Run a trusted command specification against guest files in a private jail.

    Guest stdout is data, never a status/control protocol. There is deliberately
    no unjailed fallback if bubblewrap or user namespaces are unavailable.
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
    try:
        with (
            export_filter(seccomp)
            if seccomp is not None
            else nullcontext(None) as filter_fd
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
            return _supervise(
                args,
                timeout,
                output_limit,
                pass_fds=() if filter_fd is None else (filter_fd,),
            )
    except SeccompError as exc:
        return ExecutionResult("sandbox_error", stderr=str(exc))


def run_python(
    code: str,
    timeout: float = 20,
    *,
    seccomp: SeccompPolicy | None = None,
) -> ExecutionResult:
    """Execute Python in fresh globals with bounded output and a hard deadline."""
    validate_code(code)
    with tempfile.TemporaryDirectory(prefix="neural-compiler-") as directory:
        work = Path(directory)
        (work / "prog.py").write_text(code, encoding="utf-8")
        return run_jailed(
            ["/usr/bin/python3", "-I", "-u", "/work/prog.py"],
            work,
            timeout,
            seccomp=seccomp,
        )
