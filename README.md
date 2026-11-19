# Neural Compiler (`neural-compiler`)

**LLM-as-Compiler and Native CPU Execution Bridge for Tool-Integrated Reasoning and Weight-Compiled Neural Systems.**

## Overview

`neural-compiler` investigates how a language model can compile algorithmic intent within its own weights and execute the resulting programs with minimal latency and high reliability.

Based on empirical benchmarking on `Qwen/Qwen3.5-4B` and a two-round mathematical and systems audit of weight-compiled virtual machines (see [**`WHITEPAPER.md`**](./WHITEPAPER.md) and [`docs/review/`](./docs/review/)), this project follows a two-part architecture:

1. **The LLM as the Intent-to-Program Compiler:** Rather than running a software compiler inside neural weights (`Qwenx86`'s in-weights `qcc` requires 169,293 machine trace tokens and 71.2 s to compile and run `6 * 7`), the model directly emits compact executable programs:
   - **Default Tier (Python):** Idiomatic Python with a curated standard library (`math`, `itertools`, `fractions`, arbitrary-precision integers). Achieves **1.00× token cost** (530 tokens across our 12-task benchmark) and **10/12 zero-shot accuracy** on `Qwen3.5-4B`.
   - **Compute Tier (Freestanding / C99, planned fast path):** Reference `tcc` (**1.6 ms** compile) and `clang --target=wasm32 -O2` (**45.8 ms** compile) measurements motivate this tier. Reference C/Wasm execution was **7–32× faster** than CPython on loop-intensive tasks, with **8–10/12 zero-shot C accuracy**. These are not end-to-end hardened-jail timings.
   - **Research Track (Middle-Level IRs):** (R1) A compact, statically typed surface DSL with host intrinsics lowered to native/Wasm via Cranelift, compared against (R2) grammar/validator-constrained `LLM-WAT` emission.
2. **The Host Execution Bridge as the Program-to-Silicon Compiler (target architecture):** The P0 prototype now monitors Qwen3.5's native tool delimiters and prefills chat-template-derived response turns into the **same live KV + Gated DeltaNet recurrent cache**. The planned production bridge adds `seccomp`/cgroup containment and serving integration. Historical E3 results used markdown fences, not the new native path.

## Implementation Status

The project is at **P0**, not production-ready. [The Python bridge foundation](./src/neural_compiler/bridge/execution.py) now provides fresh per-request namespaces, wall-clock supervision, process-group kill/reap, resource limits, separate stdout/stderr, and a 64 KiB program/output budget. It fails explicitly if bubblewrap is unavailable; it never falls back to host execution.

The experimental C/assembly executors compile **inside** the jail; Wasm and x86 emulation run in separate jailed processes. [Optional seccomp filters](./src/neural_compiler/bridge/seccomp.py) can apply an explicit caller-supplied syscall allowlist through bubblewrap, failing closed if setup fails. [Optional delegated cgroup requests][cgroup-requests] configure memory/swap, PID and optional aggregate CPU-bandwidth limits before starting bubblewrap; callers must supply an already delegated cgroup-v2 subtree with the required controllers enabled. Setup/cleanup failures are explicit; the bridge never enables controllers or changes host delegation. **Production syscall/import policies, real controller qualification, and serving-engine integration remain unfinished. Do not expose these executors to untrusted workloads.**

[cgroup-requests]: ./src/neural_compiler/bridge/cgroups.py

[Native protocol framing](./src/neural_compiler/bridge/protocol.py) supports one `run(code)` call at a time, rejects incomplete/malformed calls, and escapes control delimiters in structured results. Calls are capped at 4,096 body tokens; response turns are capped at 2,048 tokens with a visible `truncated` flag when output is shortened. [Live sessions](./src/neural_compiler/bridge/session.py) preserve pending sampled tokens exactly once. Multi-call, EOS, and token-budget continuation are checked against full-prefill logits on a tiny CPU hybrid Qwen model; this is not a new 4B accuracy/latency result.

[The offline evaluation harness](./src/neural_compiler/eval/harness.py) supplies explicit
task/checker contracts, matched no-tool/Python-tool/current-best paths, shared budgets and
failure-inclusive timing. It does not load models or accept datasets; statistical held-out
evaluation remains a separate gate.

[Live sessions](./src/neural_compiler/bridge/session.py) also support bounded, process-local
CPU checkpoints and independent exact-boundary resubmission. Unsupported caches and stale
model state are rejected. This is an offline correctness prerequisite, not a production
prefix-cache integration or measured serving speedup.

The 12 fixed tasks and 6 forced-tool examples are preliminary evidence, not proof of autonomous tool use, generalization, or production latency. See [the implementation audit and next gates](./docs/review/PROJECT_DIRECTION.md#implementation-audit-2026-11-12).

## Key Empirical Findings (`Qwen3.5-4B`)

| Metric / Experiment | Measured Result | Architectural Implication |
| :--- | :--- | :--- |
| **Token Efficiency (E1, 12 tasks)** | Python: **530 tokens** (`1.00x`) vs. C: **1,673** (`3.16x`), raw x86-64 bytes: **1,288** (`2.43x`), Wasm binary: **2,246** (`4.24x`), WAT text: **6,281** (`11.85x`) | Token cost is governed by abstraction level and standard-library intrinsics (`math.comb`, `pow`, bignum), not binary vs. text encoding. |
| **Bridge Latency (historical E2 prototype medians)** | Warm jailed Python: **0.07 ms**; `wasmtime` compile: **0.64–0.75 ms**; `wasmtime` run: **54 µs**; `tcc -run`: **1.7 ms** | Warm/in-process microbenchmarks motivate a bridge; they exclude the new per-request isolation costs and do not establish hardened p95 latency. |
| **Zero-Shot Emission Accuracy (E3)** | Python: **10/12**; C: **10/12** (`gcc`) / **8/12** (strict C99 + ASan/UBSan); WAT, x86-64 asm, raw x86-64 hex: **0/12** | Base 4B models reliably emit high-level code but fail on register allocation, stack typing, and relative offsets in raw IR/assembly without SFT/RL and constrained decoding. |
| **Return-Path Prefill Cost (E3b, 4k ctx)** | Live KV + GDN cache prefill (16–64 tokens): **99–123 ms** ($\approx 1.5\text{–}2$ decode steps) vs. full re-prefill: **1,489–1,514 ms** | Preserving the live hybrid KV + Gated DeltaNet state (or using hybrid prefix caching) is the primary latency lever, saving $>1.3$ s per tool call at 4k context. |

## Repository Structure

- [**`src/neural_compiler/bridge/`**](./src/neural_compiler/bridge/) — Supervised execution, native protocol framing, live-cache sessions, and standalone experimental runtime harness.
- [**`WHITEPAPER.md`**](./WHITEPAPER.md) — Revised technical whitepaper covering (1) the LLM-as-compiler + native CPU bridge architecture, empirical benchmarks (E1–E3b), security model, and SFT/RL training recipe, and (2) the mathematically corrected analytical reference for weight-compiled neural virtual machines (`Loom`, `Qwenputer`, `Qwenx86`, `Percepta`).
- [**`docs/review/PROJECT_DIRECTION.md`**](./docs/review/PROJECT_DIRECTION.md) — Architecture decision record, literature synthesis, and phased roadmap (`P0`–`P5`) with quantitative go/no-go gates.
- [**`docs/review/WHITEPAPER_REVIEW.md`**](./docs/review/WHITEPAPER_REVIEW.md) & [**`WHITEPAPER_REVIEW_ROUND2.md`**](./docs/review/WHITEPAPER_REVIEW_ROUND2.md) — Two-round technical and numerical audit of the initial NeuroVM draft, backed by [`docs/review/verify_whitepaper.py`](./docs/review/verify_whitepaper.py).
- [**`docs/review/experiments/`**](./docs/review/experiments/) — Reproducible benchmark suite (`exp_tokens.py`, `exp_bridge.py`, `exp_loop.py`, `exp_prefill.py`, `sandbox.py`, `tasks.py`) and recorded JSON results.
- [**`.devcontainer/`**](./.devcontainer/README.md) — Multi-profile Ubuntu 24.04 development container (CPU, CUDA/Docker, CUDA/Podman-WSL) with hash-locked dependencies and the compile-and-execute toolchain (`clang`, `lld`, `tcc`, `wabt`, `bubblewrap`, `wasmtime`, `unicorn`, and `flash-linear-attention` on CUDA).

## Quick Start

Inside the prepared devcontainer, install the local package without changing its locked dependencies,
then verify the environment, unit tests, and compile-and-execute toolchain:

```bash
python -m pip install --no-deps --no-build-isolation -e .
python .devcontainer/smoke.py --tests --toolchain
python docs/review/verify_whitepaper.py
```

Execute a local Python program through the P0 bridge (Linux with working bubblewrap namespaces):

```bash
neural-compiler --run-python path/to/program.py
```

This prints JSON with `status`, `stdout`, `stderr`, and `returncode`; the CLI exits nonzero for failed execution.
CLI program files, including `--check-imports` inputs, must be UTF-8 and at most
64 KiB before newline normalization; oversized files are rejected after a bounded read.
On the audit host, 30 calls to the already-loaded Python API measured **23.92 ms p50 / 25.38 ms p95**,
including fresh jail/interpreter startup. This excludes CLI/model startup and is not a production SLO.

### Standalone research runtimes

The existing [jailed Wasm and x86-emulation APIs][jailed-runtime-apis] are also
available through mutually exclusive CLI actions in the prepared devcontainer:

```bash
neural-compiler --timeout 3 --run-wat path/to/program.wat
neural-compiler --timeout 3 --run-hex path/to/program.hex
```

For tighter work limits, add `--wat-fuel COUNT` with `--run-wat`, or
`--hex-max-insns COUNT` with `--run-hex`. Counts must be positive integers
representable by the runtime counter; invalid counts and mismatched actions
fail before reading the program file. Exhaustion is reported as `timeout`,
independently of the wall-clock deadline. Omitting these options preserves the
API defaults.

WAT must export `solve` with no arguments and one `i64` result. Hex input is an
x86-64 function that returns with `ret`, with the signed 64-bit result in `RAX`.
Both actions use the same bounded UTF-8 file reader, JSON result format and
nonzero failure exit as `--run-python`. They preserve the API's default work caps
(20 billion Wasmtime fuel units or 200 million Unicorn instructions) in addition
to the supplied wall-clock timeout. Wasmtime/Unicorn must already be installed;
neither action installs dependencies or falls back to execution outside the jail.
Missing or unloadable runtime imports report `sandbox_error`, not a guest-program
failure that can be repaired by changing the submitted code.
These remain experimental tiers, not production containment or the default
model-emission path.

[jailed-runtime-apis]: ./src/neural_compiler/bridge/native.py

### Offline evaluation evidence

Save existing result rows from the [matched evaluation harness][offline-harness]
using the [versioned evidence API][evidence-records]:

```python
from pathlib import Path

from neural_compiler.eval import dump_case_results

Path(".runs").mkdir(exist_ok=True)
with open(".runs/case_results.json", "w", encoding="utf-8") as destination:
    dump_case_results(results, destination)
```

Replay the failure-inclusive matched report without loading a model or executing
saved answers:

```bash
neural-compiler --eval-report .runs/case_results.json
```

The command prints strict JSON, requires all three paths for every case, and
rejects invalid schemas or mismatched paired observations rather than dropping
records. A successful exit means report generation succeeded, not that an
accuracy threshold or dataset-acceptance gate passed. Missing legacy provenance
remains unknown. Evidence streams are caller-owned, not authenticated or atomic
file transactions; no dataset or model study is accepted by saving a file.
Summary statistics, including fixed-seed bootstrap intervals, do not depend on
the order of case records.

For reusable exact task specifications, the same stream API provides
`dump_tasks(tasks, destination)` and `load_tasks(source)`. The separate schema-v1
task document supports only the built-in integer, rational and integer-sequence
checkers, never serialized Python callables. References use canonical decimal
strings and preserve arbitrary precision without changing interpreter digit
limits. Duplicate task IDs and unknown checker kinds fail explicitly. Task and
result files can contain reference answers or private text; keep them in ignored
`.runs/` storage rather than git. Loading specifications is not dataset acceptance.

For offline cohort preparation, `inspect_task_overlap(development, evaluation)`
reports exact declared ID/family overlaps and evaluation IDs reusing development
prompt text, even if the answer format changed. It never invokes checkers or
selects/accepts a split. Duplicate IDs within a cohort fail; empty cohorts retain
explicit zero counts rather than a pass decision. These exact-text/label checks
do not detect semantic aliases or establish absence of pretraining exposure.

Optionally recheck recorded final-answer outcomes against those specifications:

```bash
neural-compiler --eval-report .runs/case_results.json --eval-tasks .runs/tasks.json
```

This rejects mismatched task IDs, families, observed task text or final-answer
classifications. It never rewrites timeout, protocol, budget or isolation failures
as correct answers. The added `reference_validation` counts distinguish checked
answers/text from nonfinal failures and unknown text observations. This verifies
agreement with the supplied references, not their authenticity, dataset quality,
full model prompts, matched hardware or production qualification.

Add `--eval-development-tasks .runs/development_tasks.json` alongside
`--eval-tasks` with either report action to include the informational
`task_overlap` inventory. Overlaps are reported, not automatically rejected as a
benchmark policy; successful replay still does not accept a split. Invalid
development specifications fail before any report is printed.

Serving-comparison rows have corresponding `dump_serving_results(results, stream)`
and `load_serving_results(stream)` APIs. These preserve nested case evidence,
token/logit-prefix boundaries, cache counters and detached dense CPU logits.
Float16, bfloat16, float32 and float64 logit values use raw hexadecimal bytes with
explicit dtype, shape and native byte order, retaining masks, NaNs and signed
zero as observations without nonstandard JSON numbers or pickle. Readers reject
other byte orders or inconsistent metadata before tensor reconstruction.
These are offline observations, not live-cache checkpoints, model weights,
training graphs, a production serving integration or measured speedups.

Replay a saved serving report with:

```bash
neural-compiler --serving-report .runs/serving_results.json
```

Add `--eval-tasks .runs/tasks.json` for the same optional reference checks.
Incomplete mode coverage, duplicate pairs and mismatched observations fail
explicitly. A single recorded mode is supported but has no paired mode deltas.
By default, report replay validates the encoded logit fields but aggregates
observations without reconstructing the CPU logits. For numerical checks, use
`neural-compiler --serving-report .runs/serving.json --check-serving-logits`
(optionally with `--eval-tasks`). Checks run before any report is printed;
successful output adds `logit_validation` counts and tolerances. Invalid
observations fail without a partial report. This flag requires `--serving-report`.

For an explicit [recorded-logit check][serving-observations],
`validate_serving_logits(results)` requires full-reprefill reference observations
plus another mode, identical token histories and measured logit boundaries.
When the prompt length was recorded, the history must account for the prompt,
generated and injected tokens, and the logit prefix must include the full prompt.
It checks exact live/prefix agreement when both are present and uses float64
arithmetic for full-reprefill agreement at the existing CPU fixture tolerances
(`atol=1e-6`, `rtol=1e-5`).
Matching negative-infinity masks are allowed; NaNs, positive-infinity maxima and
all-masked logits fail. Returned counts identify which comparisons ran. This is
agreement of supplied CPU values, not task accuracy, authenticated execution,
GPU/kernel qualification or a production cache guarantee.

[offline-harness]: ./src/neural_compiler/eval/harness.py
[evidence-records]: ./src/neural_compiler/eval/records.py
[serving-observations]: ./src/neural_compiler/eval/serving.py

Reproduce the representation, bridge latency, and live-cache prefill benchmarks (see [`docs/review/PROJECT_DIRECTION.md`](./docs/review/PROJECT_DIRECTION.md#appendix--reproducing-the-experiments)):

```bash
cd docs/review/experiments
python exp_tokens.py    # E1: Token cost across 9 representations
python exp_bridge.py    # E2: Trusted-reference host compilation and execution microbenchmarks
python exp_loop.py      # E3: Qwen3.5-4B zero-shot & in-decode-loop bridge (CUDA)
python exp_loop.py --native  # Native Part C only; writes separate exp_native_results.json
python exp_prefill.py   # E3b: Live KV+GDN cache resume vs. full re-prefill (CUDA)
```

The native experiment allows up to five serial calls, including error-repair turns, and records
execution status, result-token/prefill costs, prompt prefill, and end-to-end inference time.
Malformed calls and exhausted call budgets are explicit failures. Model loading happens only
when the script is run; importing it or requesting `--help` does not load/download weights.
