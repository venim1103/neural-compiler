# Neural Compiler (`neural-compiler`)

**LLM-as-Compiler and Native CPU Execution Bridge for Tool-Integrated Reasoning and Weight-Compiled Neural Systems.**

## Overview

`neural-compiler` investigates how a language model can compile algorithmic intent within its own weights and execute the resulting programs with minimal latency and high reliability.

Based on empirical benchmarking on `Qwen/Qwen3.5-4B` and a two-round mathematical and systems audit of weight-compiled virtual machines (see [**`WHITEPAPER.md`**](./WHITEPAPER.md) and [`docs/review/`](./docs/review/)), this project follows a two-part architecture:

1. **The LLM as the Intent-to-Program Compiler:** Rather than running a software compiler inside neural weights (`Qwenx86`'s in-weights `qcc` requires 169,293 machine trace tokens and 71.2 s to compile and run `6 * 7`), the model directly emits compact executable programs:
   - **Default Tier (Python):** Idiomatic Python with a curated standard library (`math`, `itertools`, `fractions`, arbitrary-precision integers). Achieves **1.00× token cost** (530 tokens across our 12-task benchmark) and **10/12 zero-shot accuracy** on `Qwen3.5-4B`.
   - **Compute Tier (Freestanding / C99):** Compiled inside an isolated sandbox via `tcc` (**1.6 ms** compile) or `clang --target=wasm32 -O2` (**45.8 ms** compile), delivering **7–32× faster execution** than CPython on loop-intensive tasks with **8–10/12 zero-shot accuracy**.
   - **Research Track (Middle-Level IRs):** (R1) A compact, statically typed surface DSL with host intrinsics lowered to native/Wasm via Cranelift, compared against (R2) grammar/validator-constrained `LLM-WAT` emission.
2. **The Host Execution Bridge as the Program-to-Silicon Compiler:** A hardened, out-of-process host bridge intercepts Qwen3.5's native `<tool_call>` (`248058`) / `</tool_call>` (`248059`) delimiters from the token stream, compiles and runs the code inside a namespace + `seccomp` + resource-limited jail (`bubblewrap` / `nsjail` and fuel-metered `wasmtime`), and prefills the `<tool_response>` turn directly into the **same live KV + Gated DeltaNet recurrent cache**.

## Key Empirical Findings (`Qwen3.5-4B`)

| Metric / Experiment | Measured Result | Architectural Implication |
| :--- | :--- | :--- |
| **Token Efficiency (E1, 12 tasks)** | Python: **530 tokens** (`1.00x`) vs. C: **1,673** (`3.16x`), raw x86-64 bytes: **1,288** (`2.43x`), Wasm binary: **2,246** (`4.24x`), WAT text: **6,281** (`11.85x`) | Token cost is governed by abstraction level and standard-library intrinsics (`math.comb`, `pow`, bignum), not binary vs. text encoding. |
| **Bridge Latency (E2, medians)** | Warm jailed Python: **0.07 ms**; `wasmtime` compile: **0.64–0.75 ms**; `wasmtime` run: **54 µs**; `tcc -run`: **1.7 ms** | Host compilation + execution takes less time than generating a **single decode token** ($\ge 2.5$ ms on H100; 63–137 ms on RTX A3000). |
| **Zero-Shot Emission Accuracy (E3)** | Python: **10/12**; C: **10/12** (`gcc`) / **8/12** (strict C99 + ASan/UBSan); WAT, x86-64 asm, raw x86-64 hex: **0/12** | Base 4B models reliably emit high-level code but fail on register allocation, stack typing, and relative offsets in raw IR/assembly without SFT/RL and constrained decoding. |
| **Return-Path Prefill Cost (E3b, 4k ctx)** | Live KV + GDN cache prefill (16–64 tokens): **99–123 ms** ($\approx 1.5\text{–}2$ decode steps) vs. full re-prefill: **1,489–1,514 ms** | Preserving the live hybrid KV + Gated DeltaNet state (or using hybrid prefix caching) is the primary latency lever, saving $>1.3$ s per tool call at 4k context. |

## Repository Structure

- [**`WHITEPAPER.md`**](./WHITEPAPER.md) — Revised technical whitepaper covering (1) the LLM-as-compiler + native CPU bridge architecture, empirical benchmarks (E1–E3b), security model, and SFT/RL training recipe, and (2) the mathematically corrected analytical reference for weight-compiled neural virtual machines (`Loom`, `Qwenputer`, `Qwenx86`, `Percepta`).
- [**`docs/review/PROJECT_DIRECTION.md`**](./docs/review/PROJECT_DIRECTION.md) — Architecture decision record, literature synthesis, and phased roadmap (`P0`–`P5`) with quantitative go/no-go gates.
- [**`docs/review/WHITEPAPER_REVIEW.md`**](./docs/review/WHITEPAPER_REVIEW.md) & [**`WHITEPAPER_REVIEW_ROUND2.md`**](./docs/review/WHITEPAPER_REVIEW_ROUND2.md) — Two-round technical and numerical audit of the initial NeuroVM draft, backed by [`docs/review/verify_whitepaper.py`](./docs/review/verify_whitepaper.py).
- [**`docs/review/experiments/`**](./docs/review/experiments/) — Reproducible benchmark suite (`exp_tokens.py`, `exp_bridge.py`, `exp_loop.py`, `exp_prefill.py`, `sandbox.py`, `tasks.py`) and recorded JSON results.
- [**`.devcontainer/`**](./.devcontainer/README.md) — Multi-profile Ubuntu 24.04 development container (CPU, CUDA/Docker, CUDA/Podman-WSL) with hash-locked dependencies and the compile-and-execute toolchain (`clang`, `lld`, `tcc`, `wabt`, `bubblewrap`, `wasmtime`, `unicorn`, and `flash-linear-attention` on CUDA).

## Quick Start

Verify the container environment, unit tests, and compile-and-execute toolchain:

```bash
python .devcontainer/smoke.py --tests --toolchain
python docs/review/verify_whitepaper.py
```

Reproduce the representation, bridge latency, and live-cache prefill benchmarks (see [`docs/review/PROJECT_DIRECTION.md`](./docs/review/PROJECT_DIRECTION.md#appendix--reproducing-the-experiments)):

```bash
cd docs/review/experiments
python exp_tokens.py    # E1: Token cost across 9 representations
python exp_bridge.py    # E2: Host compilation and sandbox execution latency
python exp_loop.py      # E3: Qwen3.5-4B zero-shot & in-decode-loop bridge (CUDA)
python exp_prefill.py   # E3b: Live KV+GDN cache resume vs. full re-prefill (CUDA)
```
