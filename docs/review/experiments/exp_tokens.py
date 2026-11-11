"""Experiment E1: how many tokens does each program representation cost?

For each task in tasks.py: verify all implementations agree, compile the C
version to wasm32 (clang) and x86-64 (gcc), run the wasm with wasmtime, then
count Qwen3.5 tokens for every representation an LLM could emit.

Run (devcontainer): python exp_tokens.py
"""

import json
import os
import re
import subprocess
import sys
import tempfile

import wasmtime
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tasks import C_MAIN, TASKS  # noqa: E402

tok = Tokenizer.from_file(hf_hub_download("Qwen/Qwen3.5-4B", "tokenizer.json"))


def ntok(text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False).ids)


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


ASM_DROP = re.compile(r"^\s*\.(file|ident|size|type|p2align|align|cfi_\w+|section\s+\.note|globl|text|addrsig)\b")


def clean_asm(asm: str) -> str:
    out = []
    for line in asm.splitlines():
        if ASM_DROP.match(line) or not line.strip():
            continue
        out.append(line)
    return "\n".join(out) + "\n"


def wat_instruction_estimate(wat: str) -> int:
    """Tokens if every wasm opcode and every immediate were one vocabulary item."""
    n = 0
    for line in wat.splitlines():
        s = line.strip()
        if not s or s.startswith(";;"):
            continue
        n += len(s.replace("(", " ").replace(")", " ").split())
    return n


def main():
    engine = wasmtime.Engine()
    results = {}
    with tempfile.TemporaryDirectory() as td:
        for name, t in TASKS.items():
            r = {}
            py_i = run([sys.executable, "-c", t["py_idiomatic"]]).stdout.strip()
            py_p = run([sys.executable, "-c", t["py_plain"]]).stdout.strip()
            assert py_i == py_p, (name, py_i, py_p)
            expected = int(py_i)

            cfile = os.path.join(td, f"{name}.c")
            with open(cfile, "w") as f:
                f.write(C_MAIN.format(body=t["c"]))
            exe = os.path.join(td, name)
            run(["gcc", "-O2", "-o", exe, cfile])
            assert int(run([exe]).stdout) == expected, name

            body = os.path.join(td, f"{name}_body.c")
            with open(body, "w") as f:
                f.write(t["c"])

            wat = {}
            wasm_bytes = {}
            for opt in ("-O2", "-Oz"):
                wasm = os.path.join(td, f"{name}{opt}.wasm")
                run(["clang", "--target=wasm32", opt, "-nostdlib", "-Wl,--no-entry",
                     "-Wl,--export=solve", "-o", wasm, body])
                run(["wasm-strip", wasm])
                wat[opt] = run(["wasm2wat", "--no-debug-names", wasm]).stdout
                with open(wasm, "rb") as f:
                    wasm_bytes[opt] = f.read()
                module = wasmtime.Module(engine, wasm_bytes[opt])
                store = wasmtime.Store(engine)
                inst = wasmtime.Instance(store, module, [])
                assert inst.exports(store)["solve"](store) == expected, (name, opt)

            asm = clean_asm(run(["gcc", "-O2", "-S", "-o", "-", "-fno-asynchronous-unwind-tables",
                                 "-fno-stack-protector", "-fcf-protection=none", body]).stdout)
            obj = os.path.join(td, f"{name}.o")
            run(["gcc", "-O2", "-c", "-fno-asynchronous-unwind-tables", "-fno-stack-protector",
                 "-fcf-protection=none", "-o", obj, body])
            code = b""
            for sec in (".text", ".rodata", ".rodata.str1.1", ".data"):
                binf = os.path.join(td, f"{name}{sec}.bin")
                run(["objcopy", "-O", "binary", f"--only-section={sec}", obj, binf])
                with open(binf, "rb") as f:
                    code += f.read()

            r["expected"] = expected
            r["py_idiomatic"] = ntok(t["py_idiomatic"])
            r["py_plain"] = ntok(t["py_plain"])
            r["c_body"] = ntok(t["c"])
            r["c_full"] = ntok(C_MAIN.format(body=t["c"]))
            r["wat_O2"] = ntok(wat["-O2"])
            r["wat_Oz"] = ntok(wat["-Oz"])
            r["wat_opcode_vocab_Oz"] = wat_instruction_estimate(wat["-Oz"])
            r["wasm_bytes_Oz"] = len(wasm_bytes["-Oz"])
            r["wasm_hex_Oz"] = ntok(wasm_bytes["-Oz"].hex())
            r["x86_asm_O2"] = ntok(asm)
            r["x86_bytes_O2"] = len(code)
            r["x86_hex_O2"] = ntok(code.hex())
            results[name] = r
            if name == "fib_mod":
                with open(os.path.join(os.path.dirname(__file__), "sample_fib_mod.wat"), "w") as f:
                    f.write(wat["-Oz"])
                with open(os.path.join(os.path.dirname(__file__), "sample_fib_mod.s"), "w") as f:
                    f.write(asm)

    cols = ["py_idiomatic", "py_plain", "c_body", "c_full", "wat_O2", "wat_Oz",
            "wat_opcode_vocab_Oz", "wasm_bytes_Oz", "wasm_hex_Oz", "x86_asm_O2", "x86_bytes_O2", "x86_hex_O2"]
    print("task".ljust(20) + "".join(c[:12].rjust(13) for c in cols))
    tot = {c: 0 for c in cols}
    for name, r in results.items():
        print(name.ljust(20) + "".join(str(r[c]).rjust(13) for c in cols))
        for c in cols:
            tot[c] += r[c]
    print("TOTAL".ljust(20) + "".join(str(tot[c]).rjust(13) for c in cols))
    base = tot["py_idiomatic"]
    print("x py_idiomatic".ljust(20) + "".join(f"{tot[c] / base:.2f}".rjust(13) for c in cols))
    with open(os.path.join(os.path.dirname(__file__), "exp_tokens_results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
