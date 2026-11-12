"""Bounded kernel-filter fixtures; this is not a production syscall policy."""

from __future__ import annotations

import ctypes.util
import errno
import os

import pytest

from neural_compiler.bridge import run_python
from neural_compiler.bridge.seccomp import SeccompError, SeccompPolicy, export_filter

# Trusted fixture-only interpreter/launcher syscalls. Process creation and
# networking are deliberately absent; no product allowlist is adopted here.
FIXTURE_SYSCALLS = (
    "read",
    "write",
    "close",
    "lseek",
    "pread64",
    "newfstatat",
    "fstat",
    "statx",
    "openat",
    "readlink",
    "readlinkat",
    "getcwd",
    "getdents64",
    "mmap",
    "mprotect",
    "munmap",
    "madvise",
    "brk",
    "futex",
    "set_tid_address",
    "set_robust_list",
    "rseq",
    "rt_sigaction",
    "rt_sigprocmask",
    "rt_sigreturn",
    "sigaltstack",
    "arch_prctl",
    "getrandom",
    "getpid",
    "gettid",
    "prlimit64",
    "getrlimit",
    "clock_gettime",
    "clock_nanosleep",
    "fcntl",
    "ioctl",
    "exit",
    "exit_group",
    "getuid",
    "geteuid",
    "getgid",
    "getegid",
    "execve",
    "wait4",
    "sched_getaffinity",
)


@pytest.fixture
def kernel_filter():
    if ctypes.util.find_library("seccomp") is None:
        pytest.skip("libseccomp is unavailable; no unfiltered fallback")
    result = run_python("print(0)")
    if result.status == "sandbox_error":
        pytest.skip(f"Namespace support unavailable: {result.stderr}")
    assert result.status == "ok", result
    return SeccompPolicy(FIXTURE_SYSCALLS)


def test_opt_in_filter_preserves_trusted_python_arithmetic(kernel_filter):
    result = run_python("import math; print(math.comb(40, 20))", seccomp=kernel_filter)
    assert result.status == "ok", result
    assert result.stdout.strip() == "137846528820"


def test_kernel_denies_one_process_creation_and_socket_attempt(kernel_filter):
    code = """
import os, socket
try:
    pid = os.fork()
except PermissionError as exc:
    print("fork", exc.errno)
else:
    if pid == 0:
        os._exit(0)
    os.waitpid(pid, 0)
    print("unexpected fork success")
try:
    sock = socket.socket()
except PermissionError as exc:
    print("socket", exc.errno)
else:
    sock.close()
    print("unexpected socket success")
"""
    result = run_python(code, seccomp=kernel_filter)
    assert result.status == "ok", result
    assert result.stdout.splitlines() == [
        f"fork {errno.EPERM}",
        f"socket {errno.EPERM}",
    ]


def test_missing_filter_dependency_does_not_execute_without_policy(monkeypatch):
    monkeypatch.setattr(
        "neural_compiler.bridge.execution.shutil.which", lambda _: "/usr/bin/true"
    )
    monkeypatch.setattr(
        "neural_compiler.bridge.seccomp.ctypes.util.find_library", lambda _: None
    )
    result = run_python("print('must not execute')", seccomp=SeccompPolicy(("write",)))
    assert result.status == "sandbox_error"
    assert "libseccomp" in result.stderr and not result.stdout


def test_unknown_syscall_fails_explicitly(kernel_filter):
    result = run_python(
        "print('must not execute')", seccomp=SeccompPolicy(("not_a_real_syscall",))
    )
    assert result.status == "sandbox_error"
    assert "Unknown syscall" in result.stderr and not result.stdout


def test_exported_filter_descriptor_is_closed(kernel_filter):
    with export_filter(kernel_filter) as fd:
        assert os.read(fd, 8)
    with pytest.raises(OSError) as error:
        os.fstat(fd)
    assert error.value.errno == errno.EBADF


def test_library_load_failure_is_explicit(monkeypatch):
    monkeypatch.setattr(
        "neural_compiler.bridge.seccomp.ctypes.util.find_library", lambda _: "missing"
    )
    with pytest.raises(SeccompError, match="Cannot load"):
        with export_filter(SeccompPolicy(("write",))):
            pytest.fail("A failed filter must not be usable")
