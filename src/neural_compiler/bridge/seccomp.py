"""Optional syscall-filter export for bubblewrap; no production policy is supplied."""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import re
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

_ALLOW = 0x7FFF0000
_DENY = 0x00050000 | errno.EPERM


class SeccompError(RuntimeError):
    """The requested syscall filter could not be constructed."""


@dataclass(frozen=True)
class SeccompPolicy:
    allowed_syscalls: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.allowed_syscalls or len(set(self.allowed_syscalls)) != len(
            self.allowed_syscalls
        ):
            raise ValueError(
                "An explicit, nonempty, unique syscall allowlist is required"
            )
        if any(
            re.fullmatch(r"[a-z][a-z0-9_]*", name) is None
            for name in self.allowed_syscalls
        ):
            raise ValueError("Syscall names must be lowercase identifiers")


@contextmanager
def export_filter(policy: SeccompPolicy) -> Generator[int, None, None]:
    """Yield a private BPF fd; unknown syscalls/missing libraries fail closed."""
    library = ctypes.util.find_library("seccomp")
    if library is None:
        raise SeccompError("libseccomp is required for the requested syscall policy")
    try:
        seccomp = ctypes.CDLL(library)
    except OSError as exc:
        raise SeccompError(f"Cannot load libseccomp: {exc}") from exc
    seccomp.seccomp_init.argtypes = [ctypes.c_uint32]
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.restype = None
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    seccomp.seccomp_export_bpf.restype = ctypes.c_int
    context = seccomp.seccomp_init(_DENY)
    if context is None:
        raise SeccompError("libseccomp could not allocate the filter context")
    fd = None
    try:
        for name in policy.allowed_syscalls:
            number = seccomp.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number < 0:
                raise SeccompError(
                    f"Unknown syscall for this host architecture: {name}"
                )
            result = seccomp.seccomp_rule_add(context, _ALLOW, number, 0)
            if result < 0:
                raise SeccompError(
                    f"Cannot allow syscall {name}: {os.strerror(-result)}"
                )
        try:
            fd = os.memfd_create("neural-compiler-seccomp", os.MFD_CLOEXEC)
        except OSError as exc:
            raise SeccompError(
                f"Cannot allocate the syscall-filter descriptor: {exc}"
            ) from exc
        result = seccomp.seccomp_export_bpf(context, fd)
        if result < 0:
            raise SeccompError(
                f"Cannot export the syscall filter: {os.strerror(-result)}"
            )
        os.lseek(fd, 0, os.SEEK_SET)
        yield fd
    finally:
        if fd is not None:
            os.close(fd)
        seccomp.seccomp_release(context)
