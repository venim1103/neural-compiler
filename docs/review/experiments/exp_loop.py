"""Experiment E3: Qwen3.5-4B writing programs that a host bridge executes.

Part B: zero-shot, batched over 12 tasks, one batch per target representation
        (Python, C, WAT, x86-64 asm, raw x86-64 machine-code hex). Every program
        is executed in a sandbox (sandbox.py) and checked against the reference.
Part C: an in-decode-loop bridge. Greedy decoding stops at the end of a
        ```python block, the program runs in a warm jailed worker, its output is
        prefilled into the *same* KV/recurrent cache, and decoding resumes.

Run (devcontainer): python exp_loop.py
"""

import json
import os
import re
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sandbox as sb  # noqa: E402
from tasks import TASKS  # noqa: E402

M = "Qwen/Qwen3.5-4B"
EXPECTED = {k: v["expected"] for k, v in json.load(open(os.path.join(HERE, "exp_tokens_results.json"))).items()}

LANG_SPEC = {
    "python": ("python", "Write a Python 3 program that computes the following and prints only the final integer answer."),
    "c": ("c", "Write a complete C program (C99, standard library only) that computes the following and prints only "
               "the final integer answer using printf."),
    "wat": ("wat", "Write a WebAssembly module in WAT (WebAssembly text format) that computes the following. The module "
                   "must have no imports and must export a function named \"solve\" that takes no parameters and returns "
                   "the answer as an i64. If you need linear memory, declare it inside the module."),
    "asm": ("asm", "Write x86-64 assembly for the GNU assembler using `.intel_syntax noprefix` that defines a global "
                   "function `solve` following the System V AMD64 ABI. It takes no arguments and returns the answer as a "
                   "64-bit integer in rax. Do not call any functions and do not make system calls. You may use .data or "
                   ".bss sections."),
    "hex": ("hex", "Write the raw x86-64 machine code of a position-independent function that follows the System V AMD64 "
                   "ABI, takes no arguments, and returns the answer as a 64-bit integer in rax. It must not call anything "
                   "and must only use registers and its own stack. Output the machine code bytes as one hex string."),
}
MAX_NEW = {"python": 400, "c": 500, "wat": 900, "asm": 900, "hex": 600}

TIR_SYSTEM = ("You can run Python code. When a question needs computation, write a short Python program in a "
              "```python code block that prints the result, then stop. The system will run it and reply with its "
              "output in an ```output block. After you see the output, reply with: Final answer: <number>")

tok = AutoTokenizer.from_pretrained(M)
tok.padding_side = "left"
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()
gen_eos = model.generation_config.eos_token_id
EOS = set(gen_eos if isinstance(gen_eos, list) else [gen_eos]) | {tok.pad_token_id}


def chat_text(messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def extract(text):
    blocks = re.findall(r"```[a-zA-Z0-9_+-]*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    m = re.search(r"```[a-zA-Z0-9_+-]*\n(.*)", text, re.S)
    return m.group(1) if m else text


def judge(output, expected):
    lines = [ln.strip() for ln in output.strip().splitlines() if ln.strip()]
    return bool(lines) and lines[-1] == str(expected)


@torch.no_grad()
def batch_generate(prompts, max_new):
    texts = [chat_text([{"role": "user", "content": p}]) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    res = []
    for row in out[:, enc["input_ids"].shape[1]:].tolist():
        n = next((i for i, t in enumerate(row) if t in EOS), len(row))
        res.append((tok.decode(row[:n]), n))
    return res, dt, out.shape[1] - enc["input_ids"].shape[1]


def part_b():
    names = list(TASKS)
    summary, programs = {}, {}
    for lang, (fence, _) in LANG_SPEC.items():
        prompts = [f"{LANG_SPEC[lang][1]}\n\nTask: {TASKS[n]['prompt']}\n\nReply with only the code in a single "
                   f"```{fence} block." for n in names]
        outs, dt, steps = batch_generate(prompts, MAX_NEW[lang])
        stats = {"ok": 0, "compile_error": 0, "runtime_error": 0, "timeout": 0, "wrong": 0}
        toks = []
        for n, (text, ntok) in zip(names, outs):
            code = extract(text)
            status, output = sb.EXECUTORS[lang](code)
            good = status == "ok" and judge(output, EXPECTED[n])
            stats["ok" if good else ("wrong" if status == "ok" else status)] += 1
            toks.append(ntok)
            programs[f"{lang}/{n}"] = dict(tokens=ntok, status=status, correct=good, output=output[-200:], code=code)
        summary[lang] = dict(correct=stats["ok"], of=len(names), **{k: v for k, v in stats.items() if k != "ok"},
                             mean_tokens=round(statistics.mean(toks), 1), batch_steps=steps,
                             batch_ms_per_step=round(dt * 1e3 / max(steps, 1), 1))
        print(lang, summary[lang], flush=True)
    return summary, programs


@torch.no_grad()
def forward(ids, cache):
    out = model(input_ids=ids, past_key_values=cache, use_cache=True)
    return out.logits[:, -1, :], out.past_key_values


@torch.no_grad()
def decode_until(logits, cache, stop, max_new):
    """Greedy decode; stop(text) -> True ends the phase.

    Returns ids, logits, cache, per-token ms, and `pending`: True when the last sampled token was
    NOT yet fed to the model (phase ended on stop/EOS), False when the token budget ran out.
    """
    ids, lat = [], []
    for _ in range(max_new):
        nxt = int(logits.argmax(-1))
        ids.append(nxt)
        if nxt in EOS or stop(tok.decode(ids)):
            return ids, logits, cache, lat, True
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits, cache = forward(torch.tensor([[nxt]], device="cuda"), cache)
        torch.cuda.synchronize()
        lat.append((time.perf_counter() - t0) * 1e3)
    return ids, logits, cache, lat, False


def program_done(text):
    return bool(re.search(r"```python\n.*?```", text, re.S))


@torch.no_grad()
def part_c(names):
    worker = sb.JailedPyWorker()
    worker.run("print(0)")
    rows = {}
    for n in names:
        prompt = chat_text([{"role": "system", "content": TIR_SYSTEM}, {"role": "user", "content": TASKS[n]["prompt"]}])
        p_ids = tok(prompt, return_tensors="pt")["input_ids"].cuda()
        torch.cuda.synchronize()
        t_start = time.perf_counter()
        logits, cache = forward(p_ids, None)
        ids1, logits, cache, lat1, pending = decode_until(logits, cache, program_done, 400)
        text1 = tok.decode(ids1)
        t_prog = time.perf_counter()

        code = extract(text1) if program_done(text1) else ""
        status, out = worker.run(code) if code else ("no_program", "")
        t_exec = time.perf_counter()
        res_ids = tok(f"\n```output\n{out}\n```\n", return_tensors="pt")["input_ids"].cuda()
        # If the phase ended on the closing fence, that token was sampled but not yet fed; feed it
        # together with the result. If the budget ran out, every sampled token is already cached.
        # (Prototype protocol: in-band ```output block, not Qwen3.5's native <tool_response> turn.)
        head = [ids1[-1]] if pending else []
        feed = torch.cat([torch.tensor([head], device="cuda", dtype=res_ids.dtype), res_ids], dim=1)
        logits, cache = forward(feed, cache)
        torch.cuda.synchronize()
        t_prefill = time.perf_counter()

        ids2, logits, cache, lat2, _ = decode_until(logits, cache, lambda s: False, 120)
        text2 = tok.decode(ids2)
        t_end = time.perf_counter()

        # Stateless alternative: re-prefill the entire context from scratch.
        full = torch.cat([p_ids, torch.tensor([ids1], device="cuda"), res_ids], dim=1)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        forward(full, None)
        torch.cuda.synchronize()
        t_reprefill = time.perf_counter() - t0

        m = re.search(r"Final answer:\s*\**\s*(-?[\d,]+)", text2)
        final = m.group(1).replace(",", "") if m else None
        rows[n] = dict(
            program_tokens=len(ids1), exec_status=status, exec_output=out[-80:],
            program_correct=judge(out, EXPECTED[n]), final_answer=final,
            final_correct=final == str(EXPECTED[n]),
            decode_ms_per_token=round(statistics.median(lat1 + lat2), 1),
            gen_program_ms=round((t_prog - t_start) * 1e3), exec_ms=round((t_exec - t_prog) * 1e3, 2),
            result_tokens=feed.shape[1], prefill_result_ms=round((t_prefill - t_exec) * 1e3, 1),
            answer_tokens=len(ids2), gen_answer_ms=round((t_end - t_prefill) * 1e3),
            reprefill_full_context_tokens=full.shape[1], reprefill_full_context_ms=round(t_reprefill * 1e3, 1),
            transcript=text1 + f"\n```output\n{out}\n```\n" + text2,
        )
        print(n, {k: v for k, v in rows[n].items() if k != "transcript"}, flush=True)
    worker.close()
    return rows


if __name__ == "__main__":
    # Warm-up (Triton autotuning / CUDA init) so timings reflect steady state.
    batch_generate(["Say hi."] * 12, 8)
    batch_generate(["Say hi."], 8)
    c_rows = part_c(["double_palindromes", "lcg_median", "collatz", "digits_2_100", "lattice_paths", "modpow"])
    b_summary, b_programs = part_b()
    with open(os.path.join(HERE, "exp_loop_results.json"), "w") as f:
        json.dump(dict(part_b=b_summary, part_c=c_rows, programs=b_programs), f, indent=1)
    print("PART B", json.dumps(b_summary, indent=1))
