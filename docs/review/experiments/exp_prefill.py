"""Experiment E3b: what does returning a result cost, with warm kernels?

For context lengths N, compares (a) one decode step, (b) prefilling k result
tokens on top of the live cache (in-engine bridge), and (c) re-prefilling the
whole N+k context from scratch (stateless tool-call API without prefix cache).

Run (devcontainer): python exp_prefill.py
"""

import json
import os
import statistics
import time

import torch
from transformers import AutoModelForCausalLM

M = "Qwen/Qwen3.5-4B"
HERE = os.path.dirname(os.path.abspath(__file__))
model = AutoModelForCausalLM.from_pretrained(M, dtype=torch.bfloat16, device_map="cuda").eval()
VOCAB_LO, VOCAB_HI = 1000, 100000


def ids(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(VOCAB_LO, VOCAB_HI, (1, n), generator=g).cuda()


@torch.no_grad()
def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1e3, out


@torch.no_grad()
def run():
    rows = []
    for n in (256, 1024, 4096):
        ctx = ids(n, n)
        for k in (1, 16, 64):
            res = ids(k, 7 * k)
            inc, full = [], []
            for rep in range(4):  # rep 0 warms Triton autotuning for these shapes
                # logits_to_keep=1: like a serving engine, only the last position's logits are needed.
                cache = model(input_ids=ctx, use_cache=True, logits_to_keep=1).past_key_values
                t_inc, _ = timed(lambda: model(input_ids=res, past_key_values=cache, use_cache=True,
                                               logits_to_keep=1))
                t_full, _ = timed(lambda: model(input_ids=torch.cat([ctx, res], 1), use_cache=True,
                                                logits_to_keep=1))
                if rep:
                    inc.append(t_inc)
                    full.append(t_full)
            row = dict(context=n, new_tokens=k, incremental_ms=round(statistics.median(inc), 1),
                       reprefill_ms=round(statistics.median(full), 1))
            rows.append(row)
            print(row, flush=True)
    return rows


if __name__ == "__main__":
    rows = run()
    with open(os.path.join(HERE, "exp_prefill_results.json"), "w") as f:
        json.dump(rows, f, indent=1)
