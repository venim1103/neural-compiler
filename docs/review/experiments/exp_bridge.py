"""Experiment E2: what does the execution bridge cost?

Measures, per task: wasmtime compile (from WAT and from binary), instantiate and
call with fuel metering; Python in a fresh subprocess vs a warm worker; C via
`tcc -run` and gcc -O2 + exec; native execution time of each program.

Run: PYTHONPATH=/tmp/exppkgs python exp_bridge.py
"""

import json
import os
import statistics
import subprocess
import sys
import tempfile
import time

import wasmtime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import C_MAIN, TASKS  # noqa: E402

REPS = 7


def ms(f, reps=REPS):
    ts = []
    out = None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts), out


WORKER_SRC = r"""
import sys, io, contextlib, json
for line in sys.stdin:
    code = json.loads(line)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<llm>", "exec"), {"__name__": "__main__"})
    sys.stdout.write(json.dumps(buf.getvalue()) + "\n")
    sys.stdout.flush()
"""


class PyWorker:
    """Warm Python process; a real bridge would add a seccomp/namespace jail."""

    def __init__(self):
        self.p = subprocess.Popen([sys.executable, "-c", WORKER_SRC], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True, bufsize=1)

    def run(self, code):
        self.p.stdin.write(json.dumps(code) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def main():
    cfg = wasmtime.Config()
    cfg.consume_fuel = True
    engine = wasmtime.Engine(cfg)
    worker = PyWorker()
    rows = {}
    with tempfile.TemporaryDirectory() as td:
        for name, t in TASKS.items():
            r = {}
            body = os.path.join(td, f"{name}.c")
            with open(body, "w") as f:
                f.write(t["c"])
            wasm = os.path.join(td, f"{name}.wasm")
            subprocess.run(["clang", "--target=wasm32", "-Oz", "-nostdlib", "-Wl,--no-entry",
                            "-Wl,--export=solve", "-o", wasm, body], check=True)
            subprocess.run(["wasm-strip", wasm], check=True)
            wat = subprocess.run(["wasm2wat", "--no-debug-names", wasm], check=True,
                                 capture_output=True, text=True).stdout
            with open(wasm, "rb") as f:
                wbin = f.read()

            r["wasm_compile_from_wat_ms"], _ = ms(lambda: wasmtime.Module(engine, wat))
            r["wasm_compile_from_bin_ms"], module = ms(lambda: wasmtime.Module(engine, wbin))

            def inst_and_call():
                store = wasmtime.Store(engine)
                store.set_fuel(10_000_000_000)
                inst = wasmtime.Instance(store, module, [])
                return inst.exports(store)["solve"](store), store.get_fuel()

            def inst_only():
                store = wasmtime.Store(engine)
                store.set_fuel(1000)
                return wasmtime.Instance(store, module, [])

            r["wasm_instantiate_ms"], _ = ms(inst_only)
            r["wasm_inst_call_ms"], (val, fuel_left) = ms(inst_and_call)
            r["wasm_fuel_used"] = 10_000_000_000 - fuel_left
            assert val == int(subprocess.run([sys.executable, "-c", t["py_idiomatic"]],
                                             capture_output=True, text=True).stdout)

            r["py_subprocess_ms"], _ = ms(lambda: subprocess.run(
                [sys.executable, "-c", t["py_idiomatic"]], capture_output=True, text=True), reps=3)
            r["py_worker_idiomatic_ms"], _ = ms(lambda: worker.run(t["py_idiomatic"]), reps=3)
            r["py_worker_plain_ms"], _ = ms(lambda: worker.run(t["py_plain"]), reps=3)

            full = os.path.join(td, f"{name}_full.c")
            with open(full, "w") as f:
                f.write(C_MAIN.format(body=t["c"]))
            r["tcc_run_ms"], _ = ms(lambda: subprocess.run(["tcc", "-run", full], check=True,
                                                          capture_output=True), reps=5)
            r["tcc_compile_only_ms"], _ = ms(lambda: subprocess.run(
                ["tcc", "-c", "-o", os.path.join(td, f"{name}.tcc.o"), full], check=True), reps=5)
            c2w = os.path.join(td, f"{name}_c2w.wasm")
            r["clang_c_to_wasm_O2_ms"], _ = ms(lambda: subprocess.run(
                ["clang", "--target=wasm32", "-O2", "-nostdlib", "-Wl,--no-entry", "-Wl,--export=solve",
                 "-o", c2w, body], check=True), reps=3)
            exe = os.path.join(td, name)
            r["gcc_O2_compile_ms"], _ = ms(lambda: subprocess.run(["gcc", "-O2", "-o", exe, full],
                                                                 check=True), reps=3)
            r["native_exec_ms"], _ = ms(lambda: subprocess.run([exe], check=True, capture_output=True))
            rows[name] = r

    worker.close()
    cols = list(next(iter(rows.values())).keys())
    print("task".ljust(20) + "".join(c.replace("_ms", "")[:14].rjust(15) for c in cols))
    for name, r in rows.items():
        print(name.ljust(20) + "".join(
            (f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c])).rjust(15) for c in cols))
    med = {c: statistics.median(r[c] for r in rows.values()) for c in cols}
    print("MEDIAN".ljust(20) + "".join(
        (f"{med[c]:.3f}" if isinstance(med[c], float) else str(med[c])).rjust(15) for c in cols))
    with open(os.path.join(os.path.dirname(__file__), "exp_bridge_results.json"), "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
