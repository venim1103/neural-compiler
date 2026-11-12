"""Opt-in cgroup-v2 requests; callers must supply an already delegated subtree."""

from __future__ import annotations

import math
import os
import select
import time
from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import mkdtemp


class CgroupError(RuntimeError):
    """Requested cgroup setup or cleanup could not be completed."""


@dataclass(frozen=True)
class CgroupLimits:
    delegated_root: Path
    memory_bytes: int
    pids: int
    swap_bytes: int = 0
    cpu_max: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        for name, value, minimum in (
            ("memory_bytes", self.memory_bytes, 1),
            ("pids", self.pids, 1),
            ("swap_bytes", self.swap_bytes, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if self.cpu_max is not None and (
            not isinstance(self.cpu_max, tuple)
            or len(self.cpu_max) != 2
            or any(type(value) is not int or value <= 0 for value in self.cpu_max)
        ):
            raise ValueError("cpu_max must be a positive integer (quota, period) pair")


def _require_cgroup2(root: Path) -> None:
    if os.statvfs(root).f_flag & os.ST_RDONLY:
        raise CgroupError(f"Cgroup delegation is read-only: {root}")
    filesystem = ""
    depth = -1
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, separator, right = line.partition(" - ")
        fields, details = left.split(), right.split()
        if not separator or len(fields) < 6 or len(details) < 3:
            raise CgroupError("Cannot parse cgroup mount information")
        mount_name = fields[4]
        for escaped, literal in (
            ("\\040", " "),
            ("\\011", "\t"),
            ("\\012", "\n"),
            ("\\134", "\\"),
        ):
            mount_name = mount_name.replace(escaped, literal)
        mount = Path(mount_name)
        if (root == mount or mount in root.parents) and len(mount.parts) > depth:
            filesystem, depth = details[0], len(mount.parts)
    if filesystem != "cgroup2":
        raise CgroupError(f"An existing cgroup-v2 delegation is required: {root}")


def _write_control(path: Path, value: str) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        data = value.encode("ascii")
        if os.write(fd, data) != len(data):
            raise CgroupError(f"Incomplete cgroup control write: {path}")
    finally:
        os.close(fd)


def _kill_group(group: Path) -> None:
    _write_control(group / "cgroup.kill", "1")
    deadline = time.monotonic() + 1
    with (group / "cgroup.events").open("rb") as events:
        poller = select.poll()
        poller.register(events.fileno(), select.POLLPRI | select.POLLERR)
        while True:
            events.seek(0)
            lines = events.read().splitlines()
            if b"populated 0" in lines:
                return
            if b"populated 1" not in lines:
                raise CgroupError(f"Missing cgroup population state: {group}")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CgroupError(f"Cgroup cleanup deadline exceeded: {group}")
            poller.poll(math.ceil(remaining * 1000))


@contextmanager
def request_cgroup(limits: CgroupLimits) -> Generator[tuple[int, int], None, None]:
    """Yield membership/status fds for one owned group; never enable controllers."""
    group: Path | None = None
    try:
        root = limits.delegated_root.resolve(strict=True)
        _require_cgroup2(root)
        with ExitStack() as cleanup:
            group = Path(mkdtemp(prefix="neural-compiler-", dir=root))
            cleanup.callback(group.rmdir)
            for name, value in (
                ("memory.max", limits.memory_bytes),
                ("pids.max", limits.pids),
                ("memory.swap.max", limits.swap_bytes),
                ("memory.oom.group", 1),
            ):
                _write_control(group / name, str(value))
            if limits.cpu_max is not None:
                _write_control(group / "cpu.max", " ".join(map(str, limits.cpu_max)))
            if not (group / "cgroup.kill").is_file():
                raise CgroupError(f"cgroup.kill is required: {group}")
            cleanup.callback(_kill_group, group)
            membership_fd = os.open(
                group / "cgroup.procs", os.O_WRONLY | os.O_CLOEXEC | os.O_NOFOLLOW
            )
            cleanup.callback(os.close, membership_fd)
            status_fd = os.memfd_create("neural-compiler-cgroup", os.MFD_CLOEXEC)
            cleanup.callback(os.close, status_fd)
            yield membership_fd, status_fd
    except OSError as exc:
        raise CgroupError(
            f"Request cgroup failed ({group or limits.delegated_root}): {exc}"
        ) from exc
