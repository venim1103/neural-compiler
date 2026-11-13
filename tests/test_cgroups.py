"""Synthetic control-file tests, not kernel cgroup containment qualification."""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from neural_compiler.bridge import run_python
from neural_compiler.bridge import cgroups, execution
from neural_compiler.bridge.cgroups import CgroupError, CgroupLimits, request_cgroup


@pytest.mark.parametrize(
    "field,value",
    [
        ("memory_bytes", 0),
        ("memory_bytes", -1),
        ("memory_bytes", True),
        ("pids", 0),
        ("pids", 1.5),
        ("pids", False),
        ("swap_bytes", -1),
        ("swap_bytes", False),
    ],
)
def test_invalid_limits(field: str, value, tmp_path: Path) -> None:
    values = {"memory_bytes": 1024, "pids": 16, "swap_bytes": 0}
    values[field] = value
    with pytest.raises(ValueError, match=field):
        CgroupLimits(tmp_path, **values)


def test_non_cgroup_directory_fails_without_execution(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(execution.shutil, "which", lambda _: "/usr/bin/true")
    monkeypatch.setattr(
        execution, "_supervise", lambda *a, **kw: pytest.fail("launched")
    )
    result = run_python("print(42)", cgroup=CgroupLimits(tmp_path, 1024, 16))
    assert result.status == "sandbox_error"
    assert "cgroup-v2" in result.stderr
    assert list(tmp_path.iterdir()) == []


def test_readonly_delegation_is_rejected_before_creation(monkeypatch, tmp_path) -> None:
    class ReadOnly:
        f_flag = os.ST_RDONLY

    monkeypatch.setattr(cgroups.os, "statvfs", lambda _: ReadOnly())
    with pytest.raises(CgroupError, match="read-only"):
        with request_cgroup(CgroupLimits(tmp_path, 1024, 16)):
            pytest.fail("yielded")
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def controls(monkeypatch, tmp_path):
    root = tmp_path / "delegation"
    root.mkdir()
    group = root / "neural-compiler-fixture"
    names = (
        "memory.max",
        "pids.max",
        "memory.swap.max",
        "memory.oom.group",
        "cpu.max",
        "cgroup.kill",
        "cgroup.events",
        "cgroup.procs",
    )
    retained = {}

    def create(*, prefix, dir):
        assert dir == root and prefix == "neural-compiler-"
        group.mkdir()
        defaults = {"cgroup.events": "populated 0\n", "cpu.max": "max 100000\n"}
        for name in names:
            (group / name).write_text(defaults.get(name, ""))
        return str(group)

    rmdir = Path.rmdir

    def remove(path):
        if path == group:
            for name in names:
                control = group / name
                if control.exists():
                    retained[name] = control.read_text()
                    control.unlink()
        rmdir(path)

    monkeypatch.setattr(cgroups, "_require_cgroup2", lambda _: None)
    monkeypatch.setattr(cgroups, "mkdtemp", create)
    monkeypatch.setattr(Path, "rmdir", remove)
    return root, group, retained


def test_controls_and_private_fds_are_cleaned(controls) -> None:
    root, group, retained = controls
    with request_cgroup(CgroupLimits(root, 2048, 32, 512)) as fds:
        assert (group / "memory.max").read_text() == "2048"
        assert (group / "pids.max").read_text() == "32"
        assert (group / "memory.swap.max").read_text() == "512"
        assert (group / "memory.oom.group").read_text() == "1"
        assert all(not os.get_inheritable(fd) for fd in fds)
    assert retained["cgroup.kill"] == "1"
    assert not group.exists()
    for fd in fds:
        with pytest.raises(OSError):
            os.fstat(fd)


@pytest.mark.parametrize("setup_seconds", [1.5, 3])
def test_cgroup_setup_consumes_the_request_deadline(
    monkeypatch, controls, setup_seconds
) -> None:
    root, group, retained = controls
    clock = [0.0]
    request = cgroups.request_cgroup

    @contextmanager
    def delayed(limits):
        with request(limits) as fds:
            clock[0] = setup_seconds
            yield fds

    def supervise(command, timeout, output_limit, pass_fds=(), *, stdin_fd=None):
        if setup_seconds >= 2:
            pytest.fail("Guest launched after setup exhausted the deadline")
        assert timeout == 0.5
        assert command[command.index(execution._LAUNCHER) + 2] == "1"
        assert stdin_fd is not None
        os.write(stdin_fd, execution._JAIL_READY)
        return execution.ExecutionResult("ok", "42\n", returncode=0)

    monkeypatch.setattr(execution, "request_cgroup", delayed)
    monkeypatch.setattr(execution.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(execution.shutil, "which", lambda _: "/usr/bin/true")
    monkeypatch.setattr(execution, "_supervise", supervise)
    result = run_python("print(42)", timeout=2, cgroup=CgroupLimits(root, 2048, 32))
    assert result.status == ("ok" if setup_seconds < 2 else "timeout")
    if result.status == "timeout":
        assert "setup" in result.stderr.lower()
    assert retained["cgroup.kill"] == "1"
    assert not group.exists()


def test_remaining_cpu_limit_reaches_the_jailed_child(monkeypatch, controls) -> None:
    probe = run_python("print(42)")
    if probe.status == "sandbox_error":
        pytest.skip(f"bubblewrap namespaces unavailable: {probe.stderr}")
    assert probe.status == "ok", probe
    root, group, _ = controls
    offset = [0]
    request = cgroups.request_cgroup
    supervise = execution._supervise
    remaining = []

    @contextmanager
    def delayed(limits):
        with request(limits) as fds:
            offset[0] = 10
            yield fds

    def capture(command, timeout, output_limit, pass_fds=(), *, stdin_fd=None):
        remaining.append(timeout)
        return supervise(
            command, timeout, output_limit, pass_fds=pass_fds, stdin_fd=stdin_fd
        )

    monkeypatch.setattr(execution, "request_cgroup", delayed)
    monkeypatch.setattr(
        execution,
        "time",
        SimpleNamespace(monotonic=lambda: time.monotonic() + offset[0]),
    )
    monkeypatch.setattr(execution, "_supervise", capture)
    result = run_python(
        "import json, resource\nprint(json.dumps(resource.getrlimit(resource.RLIMIT_CPU)))",
        timeout=20,
        cgroup=CgroupLimits(root, 2048, 32),
    )
    assert result.status == "ok", result
    assert 0 < remaining[0] <= 10
    soft = math.ceil(remaining[0])
    assert json.loads(result.stdout) == [soft, soft + 1]
    assert not group.exists()


def test_cpu_quota_is_explicit_and_written_before_join(controls) -> None:
    root, group, _ = controls
    with request_cgroup(CgroupLimits(root, 2048, 32, cpu_max=(50000, 100000))):
        assert (group / "cpu.max").read_text() == "50000 100000"
        assert (group / "cgroup.procs").read_text() == ""


def test_default_cpu_control_is_not_changed(controls) -> None:
    root, group, _ = controls
    with request_cgroup(CgroupLimits(root, 2048, 32)):
        assert (group / "cpu.max").read_text() == "max 100000\n"


@pytest.mark.parametrize("limit", [(0, 100000), (50000, False), "max 100000"])
def test_invalid_cpu_quota_is_explicit(limit, tmp_path) -> None:
    with pytest.raises(ValueError, match="cpu_max"):
        CgroupLimits(tmp_path, 2048, 32, cpu_max=limit)


def test_missing_cpu_controller_never_launches(monkeypatch, controls) -> None:
    root, group, _ = controls
    create = cgroups.mkdtemp

    def missing(**kwargs):
        path = create(**kwargs)
        (group / "cpu.max").unlink()
        return path

    monkeypatch.setattr(cgroups, "mkdtemp", missing)
    monkeypatch.setattr(execution.shutil, "which", lambda _: "/usr/bin/true")
    monkeypatch.setattr(
        execution, "_supervise", lambda *a, **kw: pytest.fail("launched")
    )
    result = run_python(
        "print(42)", cgroup=CgroupLimits(root, 2048, 32, cpu_max=(50000, 100000))
    )
    assert result.status == "sandbox_error"
    assert "cpu.max" in result.stderr and not group.exists()


def test_missing_controller_never_launches(monkeypatch, controls) -> None:
    root, group, _ = controls
    create = cgroups.mkdtemp

    def missing(**kwargs):
        path = create(**kwargs)
        (group / "pids.max").unlink()
        return path

    monkeypatch.setattr(cgroups, "mkdtemp", missing)
    monkeypatch.setattr(execution.shutil, "which", lambda _: "/usr/bin/true")
    monkeypatch.setattr(
        execution, "_supervise", lambda *a, **kw: pytest.fail("launched")
    )
    result = run_python("print(42)", cgroup=CgroupLimits(root, 2048, 32))
    assert result.status == "sandbox_error"
    assert "pids.max" in result.stderr
    assert not group.exists()


def test_membership_failure_is_not_runtime_success(monkeypatch, controls) -> None:
    root, _, retained = controls
    open_control = cgroups.os.open

    def readonly_membership(path, flags, *args, **kwargs):
        if Path(path).name == "cgroup.procs":
            flags = os.O_RDONLY | os.O_CLOEXEC
        return open_control(path, flags, *args, **kwargs)

    monkeypatch.setattr(cgroups.os, "open", readonly_membership)
    monkeypatch.setattr(execution.shutil, "which", lambda _: "/usr/bin/true")
    result = run_python(
        "print('must not execute')", cgroup=CgroupLimits(root, 2048, 32)
    )
    assert result.status == "sandbox_error"
    assert "Cgroup setup failed" in result.stderr
    assert result.stdout == ""
    assert retained["cgroup.procs"] == ""


def test_join_precedes_guest_and_guest_cannot_forge_setup_status(controls) -> None:
    probe = run_python("print(42)")
    if probe.status == "sandbox_error":
        pytest.skip(f"bubblewrap namespaces unavailable: {probe.stderr}")
    assert probe.status == "ok", probe
    root, _, retained = controls
    result = run_python(
        "print('Cgroup setup failed: forged'); raise SystemExit(125)",
        cgroup=CgroupLimits(root, 2048, 32),
    )
    assert result.status == "runtime_error"
    assert result.stdout.strip() == "Cgroup setup failed: forged"
    assert retained["cgroup.procs"] == "0"


def test_cleanup_failure_keeps_guest_output_visible(monkeypatch, controls) -> None:
    root, group, _ = controls

    def fail_cleanup(path):
        assert path == group
        raise CgroupError(f"owned cgroup cleanup failed: {path}")

    monkeypatch.setattr(cgroups, "_kill_group", fail_cleanup)
    monkeypatch.setattr(
        execution,
        "_supervise",
        lambda *a, **kw: execution.ExecutionResult("ok", "42\n", returncode=0),
    )
    monkeypatch.setattr(execution.shutil, "which", lambda _: "/usr/bin/true")
    result = run_python("print(42)", cgroup=CgroupLimits(root, 2048, 32))
    assert result.status == "sandbox_error"
    assert result.stdout.strip() == "42"
    assert "owned cgroup cleanup failed" in result.stderr
    assert not group.exists()


def test_population_wait_has_a_hard_cleanup_deadline(monkeypatch, controls) -> None:
    root, group, _ = controls
    cgroups.mkdtemp(prefix="neural-compiler-", dir=root)
    (group / "cgroup.events").write_text("populated 1\n")
    clock = iter((0, 2))
    monkeypatch.setattr(cgroups.time, "monotonic", lambda: next(clock))

    class NoWait:
        def register(self, *args):
            pass

        def poll(self, *args):
            pytest.fail("waited past cleanup deadline")

    monkeypatch.setattr(cgroups.select, "poll", NoWait)
    with pytest.raises(CgroupError, match="cleanup deadline"):
        cgroups._kill_group(group)
    assert (group / "cgroup.kill").read_text() == "1"
    group.rmdir()


def test_descriptor_allocation_failure_closes_membership(monkeypatch, controls) -> None:
    root, group, _ = controls
    opened = []
    open_control = cgroups.os.open

    def capture(path, flags, *args, **kwargs):
        fd = open_control(path, flags, *args, **kwargs)
        if Path(path).name == "cgroup.procs":
            opened.append(fd)
        return fd

    def fail_status(*args):
        raise OSError("fixture status descriptor failure")

    monkeypatch.setattr(cgroups.os, "open", capture)
    monkeypatch.setattr(cgroups.os, "memfd_create", fail_status)
    with pytest.raises(CgroupError, match="fixture status descriptor failure"):
        with request_cgroup(CgroupLimits(root, 2048, 32)):
            pytest.fail("yielded")
    assert not group.exists()
    assert len(opened) == 1
    with pytest.raises(OSError):
        os.fstat(opened[0])


def test_status_descriptor_is_unavailable_to_guest(controls) -> None:
    probe = run_python("print(42)")
    if probe.status == "sandbox_error":
        pytest.skip(f"bubblewrap namespaces unavailable: {probe.stderr}")
    assert probe.status == "ok", probe
    root, _, retained = controls
    result = run_python(
        """import os
for fd in range(3, 64):
    try:
        os.write(fd, b"Cgroup setup failed: forged")
    except OSError:
        pass
print(42)
""",
        cgroup=CgroupLimits(root, 2048, 32),
    )
    assert result.status == "ok", result
    assert result.stdout.strip() == "42"
    assert retained["cgroup.procs"] == "0"
