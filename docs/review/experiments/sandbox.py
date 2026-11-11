"""Sandboxed executors for model-generated programs (used by exp_loop.py).

Python/C/asm run natively inside bubblewrap (no network, read-only /usr, private
/tmp, new PID/IPC/UTS namespaces); WAT runs in wasmtime with fuel metering; raw
machine-code hex runs in the unicorn CPU emulator. Each returns (status, output).

PROTOTYPE, NOT SAFE FOR UNTRUSTED CODE. Known gaps, all to be fixed in the real
bridge (PROJECT_DIRECTION.md, phase P1):
- JailedPyWorker's timeout bounds only the wait for the first output byte; guest
  code shares the worker process and stdout, so it can stall or forge replies.
  The fix is fork-per-request with a control channel the guest cannot reach and an
  outer wall-clock deadline that kills and reaps the worker.
- The worker is stateless (fresh globals per request), and guest exceptions come
  back as text with status "ok".
- C/asm are compiled OUTSIDE the jail: `#include "/host/file"` or `.incbin` can
  read host files at compile time.
- wasm runs in-process; no seccomp, rlimits or cgroups (fork/memory bombs).
"""

import json
import os
import select
import subprocess
import tempfile

import wasmtime

BWRAP = ["bwrap", "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib",
         "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/bin", "/bin",
         "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
         "--unshare-all", "--die-with-parent", "--new-session", "--clearenv"]

HARNESS = '#include <stdio.h>\nlong long solve(void);\nint main(void){printf("%lld\\n", solve());return 0;}\n'


def _jailed(cmd, workdir, timeout):
    full = BWRAP + ["--ro-bind", workdir, "/work", "--chdir", "/work"] + cmd
    try:
        p = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", ""
    if p.returncode != 0:
        return "runtime_error", (p.stderr or p.stdout)[-300:]
    return "ok", p.stdout.strip()


def run_python(code, timeout=20):
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "prog.py"), "w") as f:
            f.write(code)
        return _jailed(["/usr/bin/python3", "prog.py"], td, timeout)


def run_c(code, timeout=20):
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "prog.c")
        with open(src, "w") as f:
            f.write(code)
        p = subprocess.run(["gcc", "-O2", "-o", os.path.join(td, "prog"), src, "-lm"],
                           capture_output=True, text=True)
        if p.returncode != 0:
            return "compile_error", p.stderr[-300:]
        return _jailed(["./prog"], td, timeout)


def run_asm(code, timeout=20):
    with tempfile.TemporaryDirectory() as td:
        with open(os.path.join(td, "solve.s"), "w") as f:
            f.write(code + "\n")
        with open(os.path.join(td, "main.c"), "w") as f:
            f.write(HARNESS)
        p = subprocess.run(["gcc", "-no-pie", "-o", os.path.join(td, "prog"),
                            os.path.join(td, "main.c"), os.path.join(td, "solve.s")],
                           capture_output=True, text=True)
        if p.returncode != 0:
            return "compile_error", p.stderr[-300:]
        return _jailed(["./prog"], td, timeout)


_cfg = wasmtime.Config()
_cfg.consume_fuel = True
_ENGINE = wasmtime.Engine(_cfg)


def run_wat(code, fuel=20_000_000_000):
    try:
        module = wasmtime.Module(_ENGINE, code)
    except Exception as e:  # parse or validation failure
        return "compile_error", str(e)[-300:]
    store = wasmtime.Store(_ENGINE)
    store.set_fuel(fuel)
    try:
        inst = wasmtime.Instance(store, module, [])
        solve = inst.exports(store)["solve"]
        return "ok", str(solve(store))
    except Exception as e:
        msg = str(e)
        return ("timeout" if "fuel" in msg else "runtime_error"), msg[-300:]


def run_hex(code, max_insns=200_000_000):
    from unicorn import UC_ARCH_X86, UC_MODE_64, Uc, UcError
    from unicorn.x86_const import UC_X86_REG_RAX, UC_X86_REG_RSP
    h = "".join(ch for ch in code if ch in "0123456789abcdefABCDEF")
    if not h or len(h) % 2:
        return "compile_error", "not an even-length hex string"
    blob = bytes.fromhex(h)
    base, stack, ret = 0x100000, 0x800000, 0x300000
    uc = Uc(UC_ARCH_X86, UC_MODE_64)
    uc.mem_map(base, 0x100000)
    uc.mem_map(stack - 0x100000, 0x200000)
    uc.mem_map(ret, 0x1000)
    uc.mem_write(base, blob)
    rsp = stack - 0x1000
    uc.mem_write(rsp, ret.to_bytes(8, "little"))
    uc.reg_write(UC_X86_REG_RSP, rsp)
    try:
        uc.emu_start(base, ret, count=max_insns)
    except UcError as e:
        return "runtime_error", str(e)
    if uc.reg_read(UC_X86_REG_RSP) != rsp + 8:
        return "timeout", "did not return"
    val = uc.reg_read(UC_X86_REG_RAX)
    return "ok", str(val - (1 << 64) if val >> 63 else val)


EXECUTORS = {"python": run_python, "c": run_c, "wat": run_wat, "asm": run_asm, "hex": run_hex}

WORKER_SRC = r"""
import sys, io, contextlib, json
for line in sys.stdin:
    code = json.loads(line)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<llm>", "exec"), {"__name__": "__main__"})
        out = buf.getvalue()
    except Exception as e:
        out = buf.getvalue() + f"{type(e).__name__}: {e}"
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
"""


class JailedPyWorker:
    """Warm Python interpreter inside bubblewrap; one exec per request."""

    def __init__(self):
        self.td = tempfile.mkdtemp()
        self.p = subprocess.Popen(BWRAP + ["--ro-bind", self.td, "/work", "--chdir", "/work",
                                           "/usr/bin/python3", "-c", WORKER_SRC],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)

    def run(self, code, timeout=20):
        self.p.stdin.write(json.dumps(code) + "\n")
        self.p.stdin.flush()
        ready, _, _ = select.select([self.p.stdout], [], [], timeout)
        if not ready:
            self.p.kill()
            self.__init__()
            return "timeout", ""
        return "ok", json.loads(self.p.stdout.readline()).strip()

    def close(self):
        self.p.stdin.close()
        self.p.wait()
