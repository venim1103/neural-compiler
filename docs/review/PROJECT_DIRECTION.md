# Project direction: LLM-as-compiler + native execution bridge (2026-11-11)

Goal as stated by the user:
> The LLM, within its own weights and as efficiently as possible, compiles things it can then run
> (maybe directly binary, or a middle level that is then compiled to full binary). A "bridge"
> connects that output directly to the CPU for execution, and the result feeds back into the model's
> inference. Fine-tuning is allowed.

This document answers **what the best approach is**. It draws on:
- four experiments run for this review on the target model (Qwen3.5-4B, local RTX A3000 12 GB);
- three verified literature sweeps;
- an independent critique of a first draft, folded in below.

The second-round whitepaper review is in [WHITEPAPER_REVIEW_ROUND2.md](WHITEPAPER_REVIEW_ROUND2.md).

**Reading map:** [architecture][architecture-map] and [roadmap][roadmap-map]
are accepted targets; [implementation audit][audit-map] and the dated updates
below distinguish implemented, tested and still-gated work. E1-E3 remain
historical prototype evidence, not renewed model-study authority.

[architecture-map]: #4-recommended-architecture
[roadmap-map]: #6-roadmap-with-gono-go-gates
[audit-map]: #implementation-audit-2026-11-12

---

## 0. Decision in one paragraph

Make the **LLM the compiler from intent to program**, and make the **bridge the compiler from program
to machine code**.

- **The model** writes the *most compact correct program* it can.
- **The bridge** is a small trusted host component. It lowers, validates and sandboxes the program,
  runs it on the CPU, and prefills the result into the **same live KV/recurrent cache**. Decoding then
  resumes.
- **Fine-tuning** (SFT, then RL with an execution reward) teaches Qwen3.5-4B when to compute, how to
  write programs for this bridge, and how to use the results.

**Start with Python**, the most compact form and the most accurate zero-shot. **Add C** for compute-heavy
work.

**Research track — the user's "middle level":** a compact, typed surface language with shared
intrinsics (bignum, math), which the bridge compiles to Wasm or native code in milliseconds. Compare it
against direct emission of a Wasm dialect.

**Raw binary is not the primary path.** Its structural costs (tokens, time, safety, portability) hold
whatever the training. Its accuracy after fine-tuning is **untested**; §3 says what evidence would
change this.

---

## 1. What "compile within its own weights" can mean

| Meaning | Example | Evidence | Verdict |
|---|---|---|---|
| (a) A compiler *program* runs inside the weights | Qwenx86 `qcc`; the whitepaper's "Tier 1" | Qwenx86's own demo: compiling and running a C program that prints 6*7 took **169,293 machine tokens, 71.2 s**. Host: `tcc` compiles in 1.6 ms; wasmtime compiles a module in ≈0.7 ms (E2). | ❌ 10⁴–10⁵× slower; no upside when the CPU is available |
| (b) **The LLM *is* the compiler** from intent to an executable program | Tool-integrated reasoning (PAL, ToRA, ReTool, rStar2) | Real interpreter vs LM "executing" its own code: GSM8K 72.0 vs 23.2 (PAL, 2211.10435). E3: 6/6 answers zero-shot through our bridge. | ✅ This is where "in-weights" adds value |
| (c) A program is baked *into* weights | Percepta "first Futamura projection" | Only useful when execution happens inside the network | ➖ Not needed for CPU execution |
| (d) Hidden states → machine code with fewer or no autoregressive steps (parallel/blockwise emission) | — | Untested; no evidence either way. Multi-token prediction (Qwen3.5 ships one MTP layer) is the proven way to cut sequential steps. | ➖ Not a starting point |

So the question becomes: **what form should the model emit, and how much of the lowering should the
bridge do?**

---

## 2. Evidence

### E1 — Token cost of each representation (12 tasks, Qwen3.5 tokenizer)

Script: [experiments/exp_tokens.py](experiments/exp_tokens.py). The 12 tasks are written as idiomatic
Python, plain-loop Python and freestanding C.

- **Executed and cross-checked (0 mismatches):** both Python versions, the gcc-native C, and both
  clang wasm builds.
- **Assembly and machine-code sizes:** taken from **unlinked** gcc objects of the same C
  (`.text + .rodata + .data`). They are approximate sizes, not runnable artifacts.

| Representation the model would emit | Tokens | × idiomatic Python | × C program |
|---|---:|---:|---:|
| Python, idiomatic (stdlib, bignum) | 530 | **1.00** | 0.32 |
| Python, plain loops | 1,036 | 1.95 | 0.62 |
| C, full program (`main` + `printf`) | 1,673 | 3.16 | 1.00 |
| x86-64 machine code, raw bytes (1 token/byte, approx.) | 1,288 | 2.43 | 0.77 |
| WAT with a 1-token-per-opcode vocabulary (estimate) | 1,844 | 3.48 | 1.10 |
| wasm binary, raw bytes | 2,246 | 4.24 | 1.34 |
| x86-64 machine code as hex (approx.) | 2,482 | 4.68 | 1.48 |
| x86-64 assembly text (gcc -O2) | 3,676 | 6.94 | 2.20 |
| WAT text (clang -Oz) | 6,281 | 11.85 | 3.75 |

How to read the table:
- **Compiler outputs are reference points, not minimum-token programs.** In E3 the model's own WAT and
  asm attempts were longer still.
- **Idiomatic Python's advantage comes largely from its library primitives** (`math.comb`, `pow`,
  bignum), which the freestanding forms lack.
- So token cost is driven mostly by **abstraction level and available intrinsics**, not by "binary vs
  text". It also implies that a low-level IR given the *same* intrinsics would shrink. That is the
  research track below.

### E2 — Cost of the bridge itself (medians over the 12 tasks)

Script: [experiments/exp_bridge.py](experiments/exp_bridge.py). The jailed worker is in
[experiments/sandbox.py](experiments/sandbox.py); it is a prototype, see its docstring.

| Bridge step | Cost |
|---|---|
| wasmtime compile (Cranelift) from WAT / from binary | 0.75 ms / 0.64 ms |
| wasmtime instantiate / instantiate+run (with fuel) | 12 µs / 54 µs (max 4.3 ms) |
| Warm Python worker, unjailed / inside bubblewrap (no net, read-only FS) | 0.21 ms / **0.07 ms** per exec (trivial program) |
| Cold `python -c` / cold bubblewrap spawn | 10 ms / 16 ms |
| C: `tcc` compile-only / `tcc -run` | 1.6 ms / 1.7 ms |
| C: clang C→wasm32 -O2 / `gcc -O2` | **45.8 ms** / 31.4 ms |
| Loop-heavy task (double palindromes): Python worker / wasm / native exec | 30–75 ms / 4.3 ms / 1.5 ms |

### E3 — The model itself (Qwen3.5-4B, bf16, greedy, thinking off, zero-shot)

Script: [experiments/exp_loop.py](experiments/exp_loop.py). Every program was executed: bubblewrap,
wasmtime with fuel, or the unicorn emulator for raw bytes. C was re-checked with the production
toolchains and with ASan/UBSan.

| Target | Correct | Failure modes | Mean tokens |
|---|---:|---|---:|
| Python | **10/12** | 2 wrong (one printed the count, not the sum) | **70** |
| C | **10/12** gcc · 9/12 clang C99 / tcc · **8/12 strict** | 1 int overflow; 1 truncated; 1 relies on a GCC nested function; 1 out-of-bounds write (ASan) | 257 |
| WAT | 0/12 | Invented ops (`i32.sqrt`); i32/i64 stack-type errors; 10/12 looped on `(local $tempN …)` until the 900-token cap | 852 |
| x86-64 asm (GAS) | 0/12 | NASM directives in GAS; operand-size mismatches; undefined symbols | 653 |
| Raw x86-64 hex | 0/12 | 7 not even valid hex; 5 crashed on unmapped memory | 600 (all at cap) |

This is one greedy sample, n=12, zero-shot. It shows **today's starting point**, not what each
representation could reach after training.

**In-decode-loop bridge, Part C (6 tasks, code use requested by the system prompt).** The model
writes a program, generation pauses, a warm jailed worker runs it, the output is prefilled into the
same cache, and decoding resumes.
- **6/6 final answers correct; 5/6 programs correct.** The LCG program sampled x0..x50 instead of
  x1..x51 and got the right median by luck.
- Execution totalled **0.13 %** of program-generation time (180 ms vs 138.7 s across the 6 tasks).
- This is a smoke test of the mechanism. It says nothing about autonomous tool selection.
- It used an in-band ```output fence, not Qwen3.5's native `<tool_response>` turn.

**E3b — return-path cost.** Script: [experiments/exp_prefill.py](experiments/exp_prefill.py). Kernels
are warm, and only last-position logits are computed, as in serving:

| Context | Decode step | Prefill 16–64 result tokens into live cache | Re-prefill whole context, no prefix cache |
|---:|---:|---:|---:|
| 256 | 63 ms | 78–81 ms | 157–179 ms |
| 1k | 66 ms | 82–84 ms | 440–449 ms |
| 4k | 63 ms | 99–123 ms | **1,489–1,514 ms** |

What the timings mean:
- Absolute speeds are slow here: HF eager, laptop GPU under WSL, and decode varied from 63 to
  137 ms/token across runs. The first prefill of a new shape also paid ≈13 s of one-off Triton
  autotuning; production engines must pre-warm or capture graphs.
- **Host-side bridge work** for small programs with warm executors is 0.05–2 ms on the wasm, tcc and
  warm-Python paths (clang→wasm adds ≈46 ms). That is less than one decode step on current GPUs: even
  an H100 must read ≈8.4 GB of bf16 weights per step, ≥2.5 ms at batch 1.
- **Returning the result** costs ≈1.2–2 decode steps of prefill.
- **Prefix caching changes the comparison.** A stateless request that hits a prefix cache does not
  re-prefill everything. Hybrid caches can resume from saved recurrent-state checkpoints; see the
  literature below.

### Literature (verified on primary sources)

**Direct low-level emission is weak unaided:**
- C→x86, 33 % IO-correct (2108.07639).
- Claude-3.5 direct-to-assembly success@1 28–35 % (2511.04132). Most of that paper's *error
  diagnostics* are assembler-stage; that counts error instances, not failed programs.
- Base 1.5–3B coders score ≈0 % on assembly translation (2506.14606).

**It becomes strong only with heavy scaffolding.** These are different tasks from ours:
- About 1M compiler-made pairs: 93–99 % on easy -O0 asm *translation*, but 15–50 % at -O2.
- RL with an execution reward: SuperCoder, 61→95 %, *optimizing* existing asm (2505.11480).
- Decomposition plus ≤5 repair rounds: LEGO-Compiler (2505.20356).

No paper evaluates LLMs *emitting* WAT/Wasm or bytecode.

**Execute-and-feed-back works at small scale only after training:**
- An untrained 4B made 0.07 tool calls and *lost* accuracy (Trice, 2605.06326).
- PAL-prompted CodeLlama-7B scored 16.6 on MATH, vs 44.6 after tool SFT (ToRA).
- The converged recipe is a small SFT on real trajectories, then RL with an outcome reward and the
  tool output masked from the loss (ReTool 2504.11536, ToRL 2503.23383, SimpleTIR 2509.02479,
  rStar2-Agent 2508.20722).

**Serving:**
- Treating a tool call as a new request with recomputation costs 37–40 % of forward time (InferCept
  2402.01869).
- Qwen3.5's 24 Gated-DeltaNet layers hold recurrent state that cannot be *truncated* like KV. It can
  be checkpointed or restored at exact boundaries: SGLang's `extra_buffer` mode, and vLLM's hybrid
  prefix caching (marked experimental).
- So there are two candidate designs: **live pause/resume**, or **prefix-cached resubmission**.
  Test both early.

**Constrained decoding** cuts syntax errors by 96 % (SynCode) and compile errors by 52–75 %
(type-constrained, 2504.09246), at µs-per-token mask cost (XGrammar).

---

## 3. Why raw binary is not the primary path (and what would change that)

1. **Accuracy (zero-shot evidence only).** 0/12 here; ≈0 % for base models in the literature. Errors
   concentrate exactly where binary is hardest: ModRM/REX encodings, rel32 offsets, ABI, stack and
   addressing. Its potential after fine-tuning is **untested**.
2. **No token saving.** Even compiler-written machine code costs 2.4× idiomatic Python (E1). Most of
   that gap is missing intrinsics, which a host library could supply to *any* representation.
3. **No time saving.** The step it skips costs 0.6–1.6 ms (wasmtime compile, `tcc`), less than one
   decode token.
4. **Unsafe by construction.** Arbitrary machine code can issue any syscall. It needs an OS jail around
   everything, or a NaCl-style instruction validator. Wasm provides software fault isolation with a
   simple, fast validator.
5. **Not portable.** It is ISA-specific, and so is its training data.

**What would change the verdict:** a *matched* comparison, with equal fine-tuning budget, constrained
decoding and repair rounds, in which binary or IR emission reaches parity in accuracy at a **lower wall
time per correct answer** than the Python/C tiers. Reasons 2–5 make that unlikely, but it is a testable
claim, and P4 below is where it gets tested.

---

## 4. Recommended architecture

```
 GPU: inference engine (Qwen3.5-4B, fine-tuned)        Host: bridge (separate jailed process)
┌──────────────────────────────────────────────┐ ids  ┌────────────────────────────────────────────┐
│ … <tool_call>  (single token 248058)         │─────►│ Stream monitor (per sequence)              │
│ <function=run><parameter=code> program …     │8 B/  │  · optional constrained-decoding mask      │
│ </tool_call>   (248059) → pause THIS sequence│token │  · streams/parses the program as it arrives│
│                                              │      │ Pre-warmed executor pool                   │
│ prefill turn boundary + <tool_response>…     │◄─────│  · python: CPython in bwrap/nsjail+seccomp │
│ into the SAME live KV + GDN state → resume   │ ids  │  · c: tcc / clang→wasm, compiled IN the jail│
│ (other sequences in the batch keep decoding) │      │  · wasm/WAT: wasmtime+Cranelift, fuel/epoch│
└──────────────────────────────────────────────┘      │ Result formatter: truncate, localize errors│
                                                      └────────────────────────────────────────────┘
```

| Concern | Choice | Why |
|---|---|---|
| Protocol | Qwen3.5's native `<tool_call>…</tool_call>`, with the response as a proper chat-template turn (`<\|im_end\|>`, user `<tool_response>…</tool_response>`, new assistant turn) | Delimiters are single tokens (248058/248059/248066/248067). The model already knows the format. Detection is a token-ID compare. |
| Default language | Python with a curated stdlib (math, fractions, itertools, bignum) | 1.0× tokens; 10/12 zero-shot |
| Fast path | C → `tcc` (1.6 ms) or clang→wasm (≈46 ms), compiled inside the jail | 7–32× faster than CPython on the loop-heavy tasks in E2 (≈1–3× on tiny ones). C is 8–10/12 zero-shot, depending on toolchain strictness. |
| Research path (the "middle level") | **(R1)** a compact *typed* surface language (e.g. a statically typed Python subset or small AST/DSL) with host intrinsics, compiled by the bridge to Wasm/native. **(R2)** direct "LLM-WAT" emission with incremental-validator-constrained decoding. Both lower via Cranelift. | R1 aims at Python-like brevity *and* native speed and safety. R2 is valid by construction but token-heavy. Candidate R1 compilers (LPython, Codon, Numba, mypyc) were **not evaluated here**. |
| Isolation | Out-of-process jail (namespaces + seccomp + rlimits/cgroups, no network, read-only FS) *and* Wasm SFI where used. Caps on size, time, memory and output. **Compile inside the jail too:** C `#include "/any/host/file"` and asm `.incbin` read host files at compile time and leak them through error messages. | Every sandbox class has had escapes. LLM-app code-exec CVEs are common (LangChain, llama_index and smolagents scored CVSS 9.8–10). |
| Deadlines | Fork-per-request (or per-session child) with an outer wall-clock deadline that kills and reaps. Control channel unreachable from guest code. | The prototype's warm worker bounds only the first byte; guest code can stall or forge replies (critique, reproduced) |
| State | Per-session namespaces (REPL semantics) with structured statuses (`ok` / `error` / `timeout`) | Enables "toolmaking": define once, call cheaply later. The prototype is stateless and reports exceptions as `ok`. |
| Return path | Text tokens prefilled into the live cache, or a prefix-cache hit | ≈1–2 decode steps; exact. Embedding injection needs a trained adapter and buys nothing for small outputs. |
| GPU↔CPU link | None special: sampled IDs already reach the host every step | The whitepaper's pinned-MMIO bridge saves µs against ms-scale costs |
| Speed levers, in priority order | (1) fewer tokens: high-level language + intrinsics; (2) fewer failed attempts: fine-tuning, constrained decoding, localized errors; (3) faster decode: Qwen3.5's MTP head for speculative decoding, quantization; (4) live resume or prefix-cache hits; (5) warm executors | Follows from E1–E3 |

---

## 5. Fine-tuning plan

| Stage | What | Notes |
|---|---|---|
| Data | Problems with checkable answers (math, number theory, combinatorics, algorithms, strings), held out **by task family**, plus text-only data (10–30 %). Randomized instances per family. | Avoid memorizable classics. Randomization catches "right for the wrong reason" (E3 Part C's LCG). |
| SFT (cold start) | 3k–16k **real, executed** trajectories in the native tool format. Teacher: a larger Qwen3.5 or rejection sampling from the 4B, run through **the same bridge**. | Real trajectories beat stitched ones: AIME24 33.2 vs 4.4 (2510.11701) |
| RL | GRPO with a binary outcome reward; tool-output tokens masked from the loss; drop void turns; 5–10 call cap; same bridge, formatting and timeouts as serving | ReTool, ToRL, SimpleTIR, rStar2 |
| Research track (R1/R2) | R1: SFT on (task → typed-DSL) programs, from translated Python solutions verified by execution. R2: SFT on (task/C → LLM-WAT) pairs from clang -O1 → wasm2wat → normalizer. Then RL. | The literature needed ~10⁶ pairs for asm; budget accordingly |
| Hardware | SFT: LoRA/QLoRA of the 4B is likely to fit on 12 GB at ≤2k tokens (to verify). Fast iteration on Qwen3.5-0.8B/2B. RL: rent ≥24–80 GB, since local HF decode (63–137 ms/token across our runs) is too slow for rollouts. | Use vLLM/SGLang for rollouts; Qwen3.5 recipes exist |

---

## 6. Roadmap with go/no-go gates

**Primary metric everywhere:** wall time (or GPU-seconds) **per correct answer**, with confidence
intervals, on held-out task families.

**Baselines, always reported:**
- the no-tool model;
- the zero-shot Python-tool model;
- the current best tier.

| Phase | Deliverable | Gate |
|---|---|---|
| P0 (≈1–2 wk) | Bridge MVP in the repo, ported from `experiments/`: native tool protocol with correct turn boundaries; fork-per-request executors with real deadlines and structured statuses; eval harness (~200 tasks, randomized instances). **Serving spike:** live pause/resume vs prefix-cached resubmission on Qwen3.5's hybrid state. | Host-side bridge work ≤1 decode step at p95. Result return ≤2 decode steps. Both serving designs measured. |
| P1 (≈1 wk) | Hardening: nsjail/bwrap + seccomp + cgroups, compile-in-jail, caps, watchdog. Red-team suite: fork/memory bombs, infinite loops, `#include`/`.incbin` exfiltration, `/proc`, huge output, network, forged replies. | 100 % contained |
| P2 (≈2 wk) | SFT cold start | Time per correct answer and accuracy beat both baselines (CI-separated); no text regression beyond noise; recovery from tool errors demonstrated |
| P3 (2–4 wk, bigger GPU) | RL with execution reward | Time per correct answer improves on P2 (token count may rise if accuracy pays for it) |
| P4 (research, after P1) | R1 typed DSL and/or R2 LLM-WAT, with constrained decoding and SFT/RL | Beats the Python/C tiers on time per correct answer for at least one task class at equal accuracy; **otherwise stop** |
| P5 | vLLM/SGLang production integration (chosen serving design, MTP speculation, batching) | p95 tool-call overhead ≤2 decode steps under load |

---

## 7. What happens to the current repo

- **The NeuroVM whitepaper.** In-weights *execution* is not needed for this goal, so archive it as
  background, marked as superseded. The parts worth keeping are §6.1 (the IR question, re-framed) and
  §7.2 (the bridge, redone safely).
- **The scaffold.** `snap_bipolar` and `gdn_exact_*` are not used by this direction. Proposed layout:
  `src/neural_compiler/bridge/` (protocol, monitor, sandbox, `executors/{python,c,wasm}`) and
  `src/neural_compiler/eval/`. The prototypes in `experiments/` are the starting point; they are
  **not hardened**.
- **New dependencies:** now part of the devcontainer (see
  [.devcontainer/README.md](../../.devcontainer/README.md#compile-and-execute-toolchain)):
  - `wasmtime` and `unicorn` in all locks;
  - `bubblewrap`, `clang`, `lld`, `tcc` and `wabt` via apt;
  - `flash-linear-attention` + `fla-core` in the CUDA lock, for Qwen3.5 speed. HF falls back to slow
    reference kernels without them.
  - `nsjail` is not packaged for Ubuntu 24.04; add it in P1 if chosen over bubblewrap.

---

## Appendix — reproducing the experiments

All dependencies are part of the devcontainer; see
[.devcontainer/README.md](../../.devcontainer/README.md#compile-and-execute-toolchain). The GPU
experiments need a CUDA profile, which provides the fla kernels.

```bash
cd docs/review/experiments
python exp_tokens.py    # E1 (+ writes exp_tokens_results.json used by E3)
python exp_bridge.py    # E2
python exp_loop.py      # E3 (CUDA profile; downloads Qwen/Qwen3.5-4B, ~25 min on an RTX A3000)
python exp_prefill.py   # E3b (CUDA profile)
```

Notes:
- Raw results are in `experiments/exp_*_results.json`. E3 stores every generated program with its
  execution status.
- `exp_loop.py` has since been patched for a failure-path bug (a duplicate token feed when the budget
  runs out), which never triggered in the recorded run.
- E3b was re-run with `logits_to_keep=1`. The first version computed full-vocabulary logits for every
  prefix position, which inflated all timings.

## Implementation audit (2026-11-12)

The architecture above is a **target**, not a description of a completed product. The revised
whitepaper and README previously overstated the implemented protocol, isolation, and latency.

### Verified issues and repairs

| Finding | Evidence | Repair |
|---|---|---|
| Python exceptions were reported as success | `raise ValueError(...)` returned `("ok", "ValueError: ...")` | Host-supervised process exit determines status; stdout is never a control reply |
| Warm-worker timeout covered only the first byte | A partial response took 301 ms despite a 50 ms timeout, then raised `JSONDecodeError` | Replace the guest-sharing worker with per-request execution; deadline covers all output and process exit; kill and reap on timeout |
| Compilation could read host files and was unbounded | C/asm invoked host gcc before entering bubblewrap | Compile in a writable private work directory inside the jail; compile and execution share one wall-clock budget; C uses strict C99 |
| Wasm/unicorn ran inside the supervisor | Fuel/instruction limits did not bound native runtime memory or compilation time | Separate jailed processes with an outer deadline and resource limits; Wasm linear memory and tables are bounded |
| Invalid hex was silently rewritten | `b82a000000c3xyz` was accepted as a program returning 42 | Reject non-hex text; permit whitespace only between bytes; require the emulator actually reaches the return sentinel |
| Experimental result handling assumed the old statuses | New resource/isolation failures could not be counted | E3 now records `output_limit`/`sandbox_error` and requires successful execution for program correctness |

The reusable Python API and CLI are in [execution.py](../../src/neural_compiler/bridge/execution.py)
and [cli.py](../../src/neural_compiler/cli.py). Regression coverage includes deadlines with partial
output/closed pipes, process reaping, output caps, explicit failures, host-file isolation, strict C99,
and Wasm fuel/memory/signature errors.

**Deliberate tradeoff:** cold per-request isolation replaces the unsafe warm Python worker.
The historical 0.07 ms measurement does **not** describe this executor. Recorded JSON results are
kept unchanged as historical evidence; E3 reruns use the new executors and last-position logits.
The E2 trusted-reference microbenchmarks are not a hardened-bridge latency gate.
On the audit host, 30 `print(6 * 7)` calls through the already-loaded Python API measured
**23.92 ms p50 / 25.38 ms p95**, including jail/interpreter startup. A partial-output timeout
returned `timeout` in 101.46 ms against a 100 ms deadline, instead of blocking for the guest's
whole response. These shared-host measurements exclude CLI/model startup and do not establish
an H100 or concurrent-serving SLO.

### Approach risks and the next development gates

1. **Keep Python first, but keep the conclusion provisional.** E1 compares handwritten programs
   with compiler output and unmatched libraries. Twelve fixed tasks and six prompted tool calls
   do not establish autonomous tool selection, held-out-family accuracy, or an advantage over
   the no-tool model. Use randomized instances, matched intrinsics/budgets, and both baselines.
2. **Finish containment before training or exposing a service.** Namespaces and rlimits alone
   are insufficient: seccomp, per-request cgroups/PID caps, an import policy, and an independent
   containment review remain prerequisites. A finite test suite cannot prove universal containment.
3. **Native protocol/cache-correct continuation now has a P0 implementation.** See the continuation
   below. A prefix-cache serving spike is still outstanding; do not assume live pause/resume wins.
4. **Measure the real objective.** Record full prompt-to-answer time, including isolation startup,
   compilation, result prefill, failed attempts, and repair turns; report p50/p95 and time per
   correct answer with uncertainty. Add warm pooling only after correctness and containment.
5. **Defer DSL/RL investment until these gates pass.** A typed DSL with arbitrary-precision host
   intrinsics is research, not an implemented low-latency compiler. Grammar/type validity is not
   semantic correctness.

### P0 continuation: native tool turns and live hybrid state

Implemented [protocol.py](../../src/neural_compiler/bridge/protocol.py) and
[session.py](../../src/neural_compiler/bridge/session.py):

- Token IDs are derived from the tokenizer and checked for single-token, distinct delimiters.
  Each sequence owns its own bounded monitor. Only one `run(code: str)` call is accepted at a
  time; malformed, nested, partial, and oversized calls fail before execution.
- Return turns are derived from the installed chat template, including assistant thinking
  boundaries. The real template includes an empty `<think>` block when thinking is disabled;
  the old whitepaper's handwritten transition omitted it.
- Results are JSON with status/stdout/stderr/returncode; angle brackets are escaped so guest
  output cannot become chat-control tokens. Response turns have a 2,048-token cap. Shortened
  output carries `truncated: true`; an envelope that cannot fit fails explicitly.
- Stop/EOS tokens remain pending and are prefilled once with the result. If a generation budget
  is exhausted, all sampled tokens are already cached and none are replayed.
- Offline tests exercise two serial calls on a real, tiny Qwen3.5 model with one linear-attention
  and one full-attention layer. Incremental logits match full-prefill reference logits within
  `atol=1e-6, rtol=1e-5`, with the same cache object retained. CPU checks use Torch reference
  kernels, not FLA/Triton; CUDA-kernel equivalence is not claimed.

The opt-in native Part C path is:

```bash
cd docs/review/experiments
python exp_loop.py --native
```

It uses the same supervised Python executor, permits five serial calls (including repairs),
and records structured protocol failures and end-to-end inference timings. It writes
`exp_native_results.json` separately from historical results. Tests include actual jailed
runtime-error recovery and a real locally cached Qwen tokenizer round trip; the latter skips
when absent and never downloads it. Importing the experiment no longer loads model weights.

**Not completed:** new 4B native accuracy/latency measurements, randomized held-out-family
evaluation with no-tool and Python-tool baselines, prefix-cached resubmission, continuous
batching, and P1 containment. Those remain the next gates before SFT/RL.

### 2026-11-12: bounded offline evaluation prerequisites

Implemented [the evaluation harness](../../src/neural_compiler/eval/harness.py) without
loading models, importing task data, or accepting a benchmark policy. Task identity/family,
prompt, and checker are separate; references are never passed to inference. The integer
checker requires a complete final answer on the last line, rather than crediting an earlier
matching integer. Other checker contracts can be supplied by callers.

The no-tool, Python-tool and caller-labelled current-best paths share prompts, checker
and resource budgets. Only tool availability differs for no-tool. "Current best" is a
comparison label, not a claim that a backend is best; the caller supplies the actual
already-loaded session factory and executor. No new C/DSL tier is selected here.

- Total generated tokens, turn/call/response tokens, serial calls, prompt/context size
  and elapsed time have explicit finite budgets. Limits do not reset after repair turns.
  Prompt overflow prevents prefill; result overflow prevents cache injection.
- Failed executions are retained with timing and status; runtime errors can be repaired.
  Missing isolation fails explicitly, rather than pretending it is a recoverable model error.
- End-to-end timing includes prompt prefill, decoding, failed attempts, execution and result
  injection. Callers must supply device synchronization. The model watchdog is cooperative
  at token/turn boundaries; the subprocess executor has its separate hard deadline.
- The cost ratio uses total elapsed time, including failures, divided by correct answers.
  Zero successes produce `seconds_per_correct: null` and `no_correct_answers`, not zero
  or a success-shaped fallback. Raw family/case/path results remain available.

**Finite routine ledger:** one initial evaluation/protocol check (33 cases passed), one
extended budget-edge check (21 evaluation cases passed), and one root gate (75 passed).
Ruff lint, scoped format check, whitespace check and Pylance harness diagnostics passed.
No checks failed; existing Torch JIT deprecation warnings remain visible. The shared tiny
CPU model/tokenizer fixtures were extracted without changing their reference kernels or
logit tolerances. Fixtures include three seeded arithmetic examples and actual jailed error
recovery; they are unit controls, not accepted task data or measured model accuracy.

No downloads, pretrained inference, GPU studies, dependencies or host privileges changed.
Next: exact-boundary hybrid checkpoint/resubmission correctness. Dataset/family/split
acceptance, matched statistical evaluation and serving-engine latency measurements remain
separately scoped; this harness alone does not satisfy those gates.

### 2026-11-12: exact-boundary CPU hybrid checkpoints

[LiveSession](../../src/neural_compiler/bridge/session.py) can now capture an exact
sampled-token boundary and fork independent sessions from it, without full re-prefill.
The snapshot includes full-attention KV, convolution history, recurrent state, logits,
token history and the pending sampled token. Restore clones the saved state; advancing
one branch does not mutate the source, sibling branches or the retained snapshot.

This is deliberately a **process-local CPU prerequisite**, not a vLLM/SGLang prefix-cache
integration or a 4B performance claim. It supports only non-offloaded `DynamicCache`
with standard dynamic/full-attention and linear-attention layer types. Unsupported cache
types, non-CPU/autograd tensors, inconsistent token boundaries and stale model weights/
buffers/configuration fail explicitly. Restore requires the same unchanged eval-mode model.
The default snapshot cap is 64 MiB of tensor storage, including logits, and is checked
before copying. The source, retained snapshot and each fork occupy separate memory;
this per-snapshot cap is not an aggregate server memory-management policy.

**Finite routine ledger and repair:** the first focused cache/protocol/evaluation check
ran 45 cases (44 passed, one failed). The ownership check found dictionary-valued
recurrent/conv state where the implementation and fixture initially expected lists.
The repair counts dictionary tensor storage, verifies all four state types are independently
cloned, and tests the precise storage cap. Revised focused check: 46 passed; final focused
checkpoint check: nine passed; root gate: 84 passed. Ruff lint, scoped format, whitespace
and Pylance session diagnostics passed. Existing Torch JIT warnings remain visible.

Tests compare serial native-call forks with live continuation and independent full-prefill
logits (`atol=1e-6, rtol=1e-5`), including stop/EOS/length boundaries, divergent sibling
responses and subsequent decoding without injection. Live and identically-fed restored
sessions also match exactly in the fixture. No tolerance was relaxed. No downloads,
pretrained inference, GPU runs, dependency changes or host security changes occurred.

Next: enforce matched case/family coverage in the offline reporting layer, then continue
P1 containment prerequisites. Real-engine checkpoint formats, GPU correctness, cache
eviction, matched latency studies and containment qualification remain outstanding.

### 2026-11-12: paired case/family reporting

The [offline harness](../../src/neural_compiler/eval/harness.py) now preflights unique
task IDs, all three comparison paths and identical resource budgets before any inference.
It runs the same task/checker object through each supplied path. Reporting rejects duplicate
results, missing baseline/current-best pairs and inconsistent family identities. Costs are
reported by path and by family, retaining failed-attempt time; invalid/nonfinite elapsed
times and mixed-path single summaries fail explicitly.

These are consistency checks, not cohort/split acceptance, statistical confidence bounds,
or proof of matched hardware/intrinsics. A complete table does not qualify task quality or
generalization; current-best remains a caller-provided label.

**Routine ledger:** 27 focused evaluation cases passed, then the root gate passed 90 tests.
Ruff lint, scoped format check, whitespace and Pylance harness diagnostics passed.
No failed attempts, downloads, datasets, pretrained inference or GPU studies occurred.

Read-only containment inspection found installed `libseccomp.so.2` and a cgroup-v2 mount
with CPU/memory/PID controllers, but the mount is **read-only**. Do not remount it or change
host privileges. Next: implement opt-in seccomp-filter plumbing with bounded trusted
fixtures; policy acceptance and writable delegated per-request cgroups remain separate.

### 2026-11-12: opt-in syscall-filter plumbing

Added [explicit seccomp-policy export](../../src/neural_compiler/bridge/seccomp.py) and
bubblewrap filter-descriptor plumbing in [the executor](../../src/neural_compiler/bridge/execution.py).
Callers may supply `SeccompPolicy(allowed_syscalls=(...))` to `run_python` or `run_jailed`.
Unlisted calls receive `EPERM`; unknown syscall names, missing libraries and export/setup
failures produce explicit isolation errors without an unfiltered fallback.

**No production allowlist is selected, and default execution behavior is unchanged.**
The policy in tests is a trusted interpreter/launcher fixture only. It permits launcher
execution and required I/O, and deliberately excludes process creation and sockets.
The kernel fixture performs one bounded fork attempt (reaping any unexpected child) and
one socket-creation attempt. Both were denied with `EPERM`; ordinary arithmetic still
ran under that same filter. Descriptors are private, passed only to bubblewrap, and closed
after execution/export scope. These checks do not prove complete containment.

**Routine ledger:** the focused filter/executor/runtime check passed 40 cases, then the
root gate passed 96 tests, with real bubblewrap/filter checks rather than skips on this host.
Ruff lint, scoped format and whitespace passed. Pylance found a deprecated contextmanager
return annotation; it was corrected to `Generator` without changing runtime behavior.
No failed execution checks, downloads, dependency installs, privileged host changes,
fork/memory bombs or untrusted services were used.

Remaining externally gated work: the cgroup-v2 mount is read-only, and neither vLLM nor
SGLang is installed. Do not remount/change privileges or install serving engines under
routine test authority. Product syscall/import-policy acceptance and independent containment
evidence, accepted held-out datasets/splits and exact pinned model-study budgets remain
outstanding. Offline infrastructure is not renewed authority for those actions.

### 2026-11-12: source-only import-policy inventory

Added [an import inventory](../../src/neural_compiler/bridge/imports.py) as a bounded
offline prerequisite for policy review, **not** a Python sandbox or admission mechanism.
It records direct imports and source lines across all syntactic branches, conservatively
flags relative/dynamic import sites (including simple aliases), and compares known direct
imports against caller-supplied module/package names. It parses source only and never
imports or executes the inspected program.

Unknown runtime/library imports, reflection and indirect calls remain unresolved. An empty
inventory or no unlisted direct imports is not evidence of safety. No product import policy
was selected and the executor's behavior is unchanged. Runtime enforcement/containment
requires separate evidence; the inventory must not be used as a substitute.

**Routine ledger:** 11 focused cases and the root suite (107 cases) passed. The subsequent
lint gate caught E731 on a sorting-key lambda assignment; it was replaced with an equivalent
typed function. The scoped format check then required spacing around the new function;
the repository formatter applied it. Lint/scoped-format/whitespace checks were revalidated without repeating the
unchanged successful runtime suite. Pylance reported no import-inventory diagnostics.
Fixtures cover relative imports, aliasing, dead branches, package-prefix boundaries and
explicit source-byte/syntax failures. No source was executed, no dataset accepted, no
downloads/dependency changes or host modifications made.

The next gates still require accepted task/family/split and pinned study scopes, real
serving-engine dependencies/resources, or a writable delegated cgroup environment and
approved production containment policies. Current offline fixtures and infrastructure do
not supply those missing inputs or justify relaxing isolation to obtain measurements.

### 2026-11-12: opt-in delegated cgroup requests

Implemented [per-request cgroup-v2 plumbing][cgroup-requests] through `run_python` and
`run_jailed`, with no change to default execution. Callers supply
`CgroupLimits(delegated_root=Path(...), memory_bytes=..., pids=..., swap_bytes=0)`.
The root must already be delegated, writable cgroup v2 with memory/PID controls enabled.
The bridge creates only its unique child group; it never enables controllers, remounts a
filesystem or changes host privileges.

The request configures `memory.max`, `memory.swap.max`, `pids.max` and
`memory.oom.group`. PID limits count launcher/helper processes and threads too; callers must
budget those, not just guest Python. A trusted wrapper joins itself before executing
bubblewrap, so guests inherit the group without a post-spawn attachment race. Its private
setup-status descriptor closes on exec, keeping guest output separate from trusted
isolation diagnostics. Cleanup closes private descriptors, requests `cgroup.kill`, waits
at most one second for depopulation and removes the owned subgroup. Setup or cleanup
failure produces `sandbox_error`, preserving already captured guest output. A failed
cleanup can leave an owned subgroup requiring removal; its path is reported.

**Scope and limits:** the actual host mount was rechecked read-only; no real cgroup controls
were written. Temporary synthetic control files test configuration, missing controls,
membership failure, descriptor lifetime, bounded cleanup and error preservation.
Two existing-style jailed fixtures check guest output/exit codes cannot forge setup status
and try at most 61 descriptor writes to check the setup descriptor is not inherited.
These are plumbing checks, not evidence that kernel memory/PID caps work, production policy
acceptance or independent containment qualification. Default jailed/native regression
controls ran without skips. Real-controller evidence still needs a writable delegated
environment; no privilege/delegation workaround is authorized here.

**Finite routine ledger:** initial cgroup/executor check: 33 passed; three added cleanup/
descriptor edge cases: three passed. Root gate: 125 passed; Ruff lint, scoped format and
whitespace checks passed. Existing 14 Torch JIT warnings remain visible. No execution/static
checks failed. One documentation patch partially applied because its final context differed;
the applied hunks were retained and only the missing hunk was retried after a bounded read.
Pylance source diagnostics found no errors and the inspected existing call signature
remained compatible. No downloads, dependencies, fork/memory bombs, model studies or
untrusted services were used.

[cgroup-requests]: ../../src/neural_compiler/bridge/cgroups.py

### 2026-11-12: evaluation correctness beyond qualification blockers

The prior stopping decision conflated missing real-engine/controller/study qualification
with all local implementation work. Those gates remain protected, but do not prevent
compatible offline prerequisites. A subsequent local check reproduced and repaired two
issues in [the evaluator][evaluation-harness]:

- `IntegerChecker` called `str(expected)`, crashing on a valid 4,500-digit reference at
  Python's current 4,300-digit decimal conversion limit. It now compares integers after
  parsing at most nine decimal digits per chunk, without changing the interpreter limit.
  The accepted final-line grammar, sign/leading-zero behavior and checker outcomes remain
  unchanged. Signed correct, valid wrong and mixed-block large answers are covered.
- Final checking and terminal device synchronization could finish outside the existing
  wall budget while the case was still scored correct. The completion boundary now checks
  the same measured elapsed time reported in `CaseResult`; ordinary answer outcomes past
  the budget become timeouts. Existing explicit protocol/budget/isolation failures remain
  visible. This remains a cooperative deadline, not interruption of arbitrary checker or
  device code.

**Bounded ledger:** three integer repro cases failed with the decimal conversion error;
ten focused grammar/integer cases passed after repair. Two deterministic terminal-clock
repros failed by returning `correct` past the budget; four terminal/execution checks passed
after repair, including an explicit-protocol-failure control. One shared root gate passed
132 cases. Ruff lint/scoped format/whitespace and Pylance source diagnostics passed;
existing 14 Torch JIT warnings remain visible. No accepted data, pretrained runs, GPU jobs,
dependency changes or host modifications occurred.

Next: verify current primary-source serving-engine compatibility before proposing an exact
dependency integration scope. Public-source research and offline implementation do not
require resolving all qualification gates first.

[evaluation-harness]: ../../src/neural_compiler/eval/harness.py

### 2026-11-12: serving dependency compatibility prerequisite

Read-only metadata inspection found the current base environment has Torch
`2.10.0+cu128` and Transformers `5.22.1`; neither vLLM nor SGLang is installed.
Workspace free space was 364.16 GiB at inspection, not a guarantee of future download
headroom or GPU fit.

Primary package metadata rules out silently installing either inspected vLLM candidate
into this environment:

| Candidate | Declared Python | Declared Torch | Declared Transformers |
| --- | --- | --- | --- |
| [vLLM 0.33.0 metadata][vllm-033-metadata] | `>=3.11,<3.15` | `==2.14.0` | `>=5.16.1,<5.21.0` |
| [vLLM 0.17.1 metadata][vllm-017-metadata] | `>=3.10,<3.14` | `==2.10.0` | `>=4.56.0,<5` |

These are inspected candidates, **not adopted pins or resolver-verified environments**.
Both declare Apache-2.0 metadata; that is not transitive licence review.
The [developer-preview hybrid-cache design][vllm-hybrid-design] describes full-attention/
Mamba prefix intersection and recurrent-state caching at block boundaries. It explicitly
warns about its implementation basis; it does not establish Qwen3.5 exact tool-boundary
state, live pause/resume support, numerical equivalence or measured serving speedup.

**Research ledger:** one broad search failed with HTTP 504. Six bounded primary requests
were spent: release metadata, hybrid-cache documentation, two SGLang documentation URLs
(both 404), and two projected PyPI metadata responses. The original four-request plan was
narrowly extended to six after the first SGLang 404 and incompatible latest-vLLM metadata;
failed attempts were retained, not refunded. SGLang compatibility remains unverified;
no further guessed documentation routes were retried. No wheels, weights or datasets were
downloaded, no environment changed and no model/GPU/server jobs ran.

The concrete next protected scope is **isolated serving-dependency integration**, keeping
the default Python 3.10/3.12 project requirements and environment intact. It needs explicit
authority for new dependencies/optional environment setup; it does not imply dataset
acceptance, pretrained-model runs or writable cgroup delegation. Public-source research
and compatible local correctness work remain separate from those permissions.

[vllm-033-metadata]: https://pypi.org/pypi/vllm/0.33.0/json
[vllm-017-metadata]: https://pypi.org/pypi/vllm/0.17.1/json
[vllm-hybrid-design]: https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/

### 2026-11-12: repeated stopping diagnosis and continuation safeguards

The current session log confirms the second ending was an assistant completion
decision: after the approval tool reported it could not collect a response, the
assistant recorded dependency integration as blocked and called `task_complete`.
The preceding source gate passed 132 tests and the source/documentation pushes
were verified; no user stop/pause instruction caused that ending.

[The project guide][project-guide] already required active continuation. It now
explicitly applies the final-answer gate to terminal completion tools, distinguishes
slice exit conditions from session stopping, and separates user-requested pauses
from assistant-generated endings. An unavailable approval tool defers only the
protected action, not other authorized work. An all-path blocker claim requires
a concrete sweep of each queue and its offline prerequisites; a self-selected
blocked todo list is not sufficient evidence. Workflow/documentation repairs are
checkpoints during an already active continuation, not a new permission to finish.

Protected study/data/dependency/host boundaries remain unchanged. This strengthens
agent behavior within available tools; it does not promise unlimited runtime,
invent background monitoring or authorize bypassing safety/platform limits.

**Documentation ledger:** ASCII guide content, all 11 local guide reference targets,
the direction-to-guide link and whitespace passed. Protected Boundaries and Rules
sections were checked unchanged against the preceding revision. The first literal
wording check falsely failed on an existing line wrap; the validator was corrected
to normalize whitespace and propagate failures, then passed. No code suite was
rerun; its latest unchanged result remains 132 passing tests.

Next actual authorized action after this documentation checkpoint: inspect the
CPU hybrid checkpoint's eval-mode consistency guard and use bounded existing-style
fixtures if a concrete gap is reproduced, rather than ending at a dependency approval.

[project-guide]: ../../AGENTS.md

### 2026-11-12: actual continuation and complete model-mode guards

The instruction checkpoint was followed by an actual offline cache-contract check,
not a terminal completion call. Three tiny CPU repro cases showed session creation,
checkpoint capture and restore accepting child modules in training mode while the
parent model's training flag remained false.

The session now shares a complete module-tree eval-mode predicate across creation,
prefill/decode, capture and restore. Changed mode is rejected before decoding/injection
can mutate the token boundary. This closes an unsupported-state path, not a numerical
accuracy claim; exact-boundary ownership/pending-token/reference-logit controls remain
unchanged (`atol=1e-6, rtol=1e-5`), as do source/token/cache budgets.

**Bounded ledger:** three initial mode cases failed; the focused checkpoint gate passed
12 cases after repair, including unchanged-state checks after refused continuation.
Root gate: 135 passed, with existing 14 Torch JIT warnings and no skips. Ruff lint,
scoped format, whitespace and Pylance session diagnostics passed. Fixtures were tiny
untrained CPU models using the existing Torch-reference path; no new dependencies,
pretrained/GPU jobs, accepted data or host changes occurred.

Next: inspect whether checkpoint model-version metadata is tied to the model that
produced the initial prefix, rather than merely the model present at capture time.
This is another bounded offline correctness prerequisite, not serving qualification.

### 2026-11-12: checkpoint identity bound to the initial prefix

Two CPU repro cases confirmed that pre-capture weight/configuration changes were
accepted: the checkpoint stamped the current model onto a cache/logits prefix
computed by the earlier model. Version-tracked CPU sessions now retain their initial
prefill model identity and check it before continuation/capture. Restored sessions
inherit that same identity; snapshots no longer relabel stale cache state with current
weights or configuration.

GPU and unversioned/inference-tensor models retain their existing live-only path;
unsupported CPU checkpoint requests fail explicitly rather than disabling the guard
and returning a snapshot. This is still process-local correctness infrastructure,
not serving integration, concurrent model mutation safety or GPU equivalence evidence.

**Bounded ledger:** two mutation cases failed initially and the unversioned live-only
control passed. After repair, 15 focused checkpoint cases and the root gate's 138 cases
passed, with existing 14 Torch JIT warnings and no skips. Changed-model decode/injection
refusals preserve the token boundary. Ruff lint, scoped format, whitespace and Pylance
source diagnostics passed. Reference-logit tolerances and cache/token budgets were not
changed; fixtures remain tiny untrained CPU models with no downloads, dependency changes,
accepted data, GPU jobs or host modifications.

Next: advance the separate P1 queue with explicit caller-supplied aggregate CPU-quota
plumbing for delegated cgroups, using kernel contract review and synthetic control files.
Real controller enforcement remains blocked by the read-only mount.

### 2026-11-12: user-requested WIP checkpoint and pause

Pending delegated-cgroup plumbing adds an optional positive integer `cpu_max`
`(quota, period)` pair, writes it before membership setup, and leaves the CPU control
unchanged when omitted. Six focused synthetic cases passed and the changed test file
was formatted. This WIP has **not** received the root regression/lint gate; the latest
full result remains 138 passing tests before these edits. No real cgroup controls were
written, and kernel enforcement remains unqualified on the read-only mount.

CI investigation established that [the latest pushed baseline run][ci-failure-wip]
fails at pytest on both Python 3.10 and 3.12, after successful installation and Ruff.
The Python 3.12 job reports three failures, 108 passes and 27 skips: two cgroup setup
tests and the missing-libseccomp test expect deeper dependency errors, but the runner
has no bubblewrap executable and correctly fails earlier. An earlier checkpoint also
failed at pytest on both runtimes. The CI repair is **not implemented**: next isolate
these no-launch unit fixtures from ambient bubblewrap availability, including the new
CPU-controller case, while preserving the executor's fail-closed ordering and real
namespace-test skips.

The Actions tool targeted an unrelated enterprise host (404); public repository API
metadata succeeded, anonymous job-log reads returned 403 twice, and existing configured
GitHub authentication permitted a bounded job-log read without credential/settings
changes or forwarding authentication on the download redirect. No dependencies,
downloads, model/GPU jobs or host-security changes occurred. The user requested this
local WIP commit and a pause; no further implementation or push is part of this checkpoint.

[ci-failure-wip]: https://github.com/venim1103/neural-compiler/actions/runs/43236827251

### 2026-11-12: CI repair and multi-family offline evaluation prerequisites

Four no-launch cgroup and seccomp unit tests (`tests/test_cgroups.py`, `tests/test_seccomp.py`)
were isolated from ambient `bubblewrap` availability by stubbing `shutil.which("bwrap")`
(`c2c9b27`), restoring green Python 3.10 and 3.12 CI runs (117 passed, 27 skipped).

The offline evaluation package (`src/neural_compiler/eval/`) now includes:
- `RationalChecker` and `IntegerSequenceChecker` alongside `IntegerChecker`, sharing bounded
  chunked decimal parsing and propagating `Task.answer_format` into the system prompt.
- Deterministic parameterized synthetic task-family generators (`src/neural_compiler/eval/families.py`:
  `modular_exponentiation`, `binomial_coefficient`, `linear_congruence`, `harmonic_partial_sum`,
  `recurrence_prefix`) for offline harness and reporting verification.
- Enriched `summarize()` and `matched_report()` metrics (`generated_tokens`, `injected_tokens`,
  `total_tokens`, `tool_calls`, `execution_seconds`, `paired_cases`, and canonical per-`Outcome`
  breakdowns), verified end-to-end on the tiny CPU hybrid Qwen fixture.

These generators remain synthetic unit fixtures, not accepted benchmark splits or pretrained
model accuracy evidence. Next: advance Queue 2 (bounded CPU checkpoint manager/eviction and
resubmission comparison harness) or Queue 3 (promoting C jail compilation and candidate
import/seccomp profiles).

### 2026-11-12: bounded checkpoint store and offline P0 serving comparison harness

Queue 2 offline prerequisites are implemented and verified on the tiny CPU hybrid Qwen fixture:
- Added `CheckpointStore` (`src/neural_compiler/bridge/session.py`) with strict byte-budget
  and entry-count LRU eviction, longest-prefix checkpoint lookup (`match_prefix` /
  `resume_prefix`), exact pending-token boundary preservation, and hit/miss/eviction counters.
- Added `LivePauseResumeSession`, `PrefixCachedSession`, `FullReprefillSession`,
  `run_serving_comparison()`, and `summarize_serving()` (`src/neural_compiler/eval/serving.py`)
  implementing the `DecodeSession` protocol so `EvaluationHarness` can compare live session
  continuation against prefix-cached resubmission and full re-prefill under identical tasks
  and budgets.
- Verified on the tiny CPU hybrid Qwen model (`tests/test_checkpoints.py`,
  `tests/test_serving_comparison.py`) that `prefix_cached_resubmission` matches
  `live_pause_resume` logits bit-for-bit (`atol=0, rtol=0`) and matches `full_reprefill_resubmission`
  within `atol=1e-6, rtol=1e-5`, while reducing prefill tokens across multi-call turns.

This remains offline CPU correctness infrastructure, not GPU latency or serving-engine
qualification. Next: advance Queue 3 (jailed C/assembly compilation module and candidate
import/seccomp allowlist profiles).

### 2026-11-12: in-jail C99/assembly bridge module and candidate P1 policy baselines

Queue 3 offline/bridge prerequisites are implemented and verified:
- Added `run_c()` and `run_asm()` (`src/neural_compiler/bridge/native.py`, exported from
  `neural_compiler.bridge`) compiling strict C99 (`-std=c99 -pedantic-errors -O2`) and
  x86-64 assembly (`long long solve(void)`) inside a writable bubblewrap jail before
  executing `/work/prog` in a read-only jail with remaining wall-clock budget, explicit
  `compile_error` status mapping, and optional post-compile `seccomp`/`cgroup` enforcement.
- Added candidate P1 review baselines (`CANDIDATE_MATH_MODULES` and `evaluate_import_policy()`
  in `src/neural_compiler/bridge/imports.py`; `CANDIDATE_PYTHON_COMPUTE_SYSCALLS`,
  `candidate_python_compute_policy()`, `SeccompPolicy.allows()`, and `SeccompPolicy.merge()`
  in `src/neural_compiler/bridge/seccomp.py`), explicitly scoped as review candidates rather
  than adopted production security boundaries.
- Added runner-independent unit tests (`tests/test_execution.py`, `tests/test_native_bridge.py`,
  `tests/test_import_inventory.py`, `tests/test_seccomp.py`) covering `run_jailed` flag/FD
  assembly, supervisor `bwrap:` / `SIGXCPU` classification, compile-vs-run isolation boundaries,
  and candidate policy evaluation.

Real kernel containment qualification (`bwrap` user namespaces, kernel `seccomp` enforcement,
writable `cgroup2` delegation) and GPU/serving-engine runs on `Qwen/Qwen3.5-4B` remain
separate hardware/host gates.

### 2026-11-12: unified jailed Wasm/x86 bridge wrappers and CLI policy/native entry points

- Added `run_wat()` and `run_hex()` (`src/neural_compiler/bridge/native.py`, exported from
  `neural_compiler.bridge`) wrapping the isolated `runtime.py` Wasmtime/Unicorn worker in
  `run_jailed` with `ExecutionResult` status mapping (`2 -> compile_error`, `3 -> timeout`),
  pre-jail hex/budget validation, and optional `seccomp`/`cgroup` forwarding.
- Extended `neural-compiler` CLI (`src/neural_compiler/cli.py`) with `--run-c`, `--run-asm`,
  `--check-imports` (static candidate import-policy triage), and `--timeout`.
- Verified with runner-independent unit tests in `tests/test_native_bridge.py` and
  `tests/test_smoke.py`.

### 2026-11-12: uncertainty intervals, paired deltas, jailed tcc fast path, and import pre-filter

- Extended `summarize()` and `matched_report()` (`src/neural_compiler/eval/harness.py`) with
  point accuracy, 95% Wilson score intervals (`accuracy_ci_95`), deterministic seeded bootstrap
  95% CIs on failure-inclusive `seconds_per_correct` (`seconds_per_correct_ci_95`), p50/p95
  wall-clock and execution quantiles, and `paired_deltas` 2×2 contingency counts (`both_correct`,
  `target_only_correct`, `baseline_only_correct`, `neither_correct`) plus paired time deltas.
- Added `compiler: Literal["gcc", "tcc"] = "gcc"` to `run_c()` (`src/neural_compiler/bridge/native.py`)
  and `--compiler` to `neural-compiler` CLI (`src/neural_compiler/cli.py`), compiling via
  `/usr/bin/tcc -Wall -Werror` inside the writable jail before read-only execution.
- Added optional static AST import pre-filtering (`allowed_imports`, `allow_unresolved_imports`)
  to `run_python()` (`src/neural_compiler/bridge/execution.py`) failing closed as `sandbox_error`
  before launching `bwrap`.
- Updated `AGENTS.md` `Current Work` to reflect completed offline prerequisites across Queues 1–3
  and remaining privileged-Linux / CUDA-GPU gates.

### 2026-11-12: unified BridgePolicy dispatcher and red-team contract suite

- Added `BridgePolicy` and `execute_program()` (`src/neural_compiler/bridge/native.py`,
  exported from `neural_compiler.bridge`) unifying `"python"`, `"c"` (`gcc`/`tcc`), `"asm"`,
  `"wat"`, and `"hex"` execution under a single validated policy configuration suitable for
  `EvaluationHarness` `current_best` path wiring.
- Added `tests/test_red_team_contracts.py` verifying adversarial protocol delimiter smuggling
  and re-feed safety, runaway stdout flood containment (`output_limit` + `SIGKILL` process-group
  reaping), static import-policy evasion vectors, and `BridgePolicy` parameter bounds.
- All offline prerequisites across Queues 1–3 are implemented and verified in CI; further
  empirical/qualification work requires a privileged Linux host (namespaces, kernel `seccomp`,
  writable `cgroup2`) or a CUDA GPU (`Qwen/Qwen3.5-4B`).

### 2026-11-13: review of intervening implementation and contract repairs

The clean local worktree was still at WIP `636d8fc`; the configured upstream contained
seven intervening commits through `57e441f`. A fast-forward preserved both histories.
Direct review covered the new native dispatcher/CLI and candidate policies, synthetic
families/checkers/uncertainty metrics, checkpoint store, serving adapters and their tests.
[The latest hosted baseline][review-ci-baseline] and its preceding run were green;
local baseline validation passed 201 cases with 14 existing Torch JIT warnings and no skips.

Five reproduced contract/reporting issues were repaired:

- `execute_program("python", ...)` ignored `BridgePolicy.output_limit`. Python now forwards
  the same explicit cap to the supervisor as the other tiers; its unchanged default is
  64 KiB, and exact-boundary/one-byte-over tests exercise the real jail.
- Bootstrap cost intervals discarded zero-success resamples, conditioning uncertainty on
  success. All 1,000 seeded resamples now count. Undefined zero-success costs sort above
  finite costs; nonfinite interval bounds are emitted as JSON `null`, with explicit
  `seconds_per_correct_ci_status` (`defined`, `unbounded_upper`, or `no_correct_answers`).
  The existing point ratio and zero-correct `ratio_status` semantics are unchanged.
- Call execution time now rejects nonfinite/negative measurements before they can poison
  execution totals/quantiles.
- Serving summaries reject duplicate/incomplete mode pairs, different family identities
  and different evaluation paths rather than comparing unmatched cohorts.
- Prefix-cached adapters no longer derive the per-snapshot default from the 256 MiB
  aggregate store budget. The shared 64 MiB snapshot default is retained, additionally
  bounded by store capacity; explicit adapter overrides remain explicit.

**Bounded ledger:** the first eight repro cases had six failures and two passing controls;
the snapshot-default repro also failed. After repair, 11 focused contract/logit controls
and four snapshot/eviction/dispatcher controls passed. Pylance found all 35 Python executor
call sites compatible and no harness/serving source diagnostics.
The final root gate passed 210 cases with the same 14 warnings and no skips; Ruff lint,
format checks for all nine changed Python files, whitespace and local documentation
reference checks passed. Execution/native/session Pylance diagnostics were also clear.
Existing jailed Python/C/assembly and Wasm/Unicorn regression controls actually ran.

Five trusted jailed Wasm/hex controls were attempted once: result/parse-error checks passed;
the fuel-loop check hit `output_limit` because its trap diagnostic exceeded the deliberately
small 128-byte cap, not the expected `timeout`. This failed fixture expectation was retained,
the cap was not raised, and unchanged successful checks were not repeated. Both remaining
hex result/instruction-cap controls passed. Output-limit precedence remains explicit.

These are offline/local contract checks, not pretrained-model accuracy, GPU/serving latency,
accepted statistical evaluation or independent production containment evidence. Namespace
and seccomp fixtures can run on this host without privilege changes; writable cgroup
qualification and accepted production policies remain outstanding. The previous update's
blanket all-prerequisites-complete/privileged-host conclusion was too strong. No new
dependencies, model/data downloads, pretrained/GPU jobs or host-security changes occurred.
This user-requested review does not resume separate project milestones.

[review-ci-baseline]: https://github.com/venim1103/neural-compiler/actions/runs/43314457168

### 2026-11-13: failed-turn boundaries in full re-prefill comparison

The review repair's [hosted baseline][review-repair-ci] completed successfully.
Autonomous continuation then reproduced a separate Queue 2 correctness gap:
`FullReprefillSession` reported stale token history when a decode callback raised,
and subsequent injection omitted the pending sampled token. Live and prefix-cached
adapters retained those tokens.

The full re-prefill adapter now reports active session history and snapshots that
history before resubmission. Malformed native calls therefore retain all sampled
tokens in the failure record; callback-abort recovery feeds the pending token exactly
once. No exceptions are swallowed and failed cases remain failed.

**Bounded ledger:** both tiny CPU repros failed before repair. Five focused failure,
recovery, multi-call reference-logit and eviction controls passed afterward. Root
gate: 212 passed, 14 existing warnings, no skips; Ruff lint/scoped format/whitespace
and Pylance serving diagnostics passed. Exact cached/live equality and full-prefill
`atol=1e-6, rtol=1e-5` remain unchanged. No model/data downloads, pretrained/GPU jobs,
dependency changes or host modifications occurred.

Next: advance the reporting contract's validation of externally assembled case/call
records so impossible counters or unknown outcomes cannot silently enter aggregates.
This is another bounded offline prerequisite, not a dataset or benchmark-policy gate.

[review-repair-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43410204789

### 2026-11-13: impossible evaluation records fail before aggregation

Public case/call records previously accepted negative, fractional or boolean token
counts and unknown outcome/path labels. This allowed impossible token totals and
failures absent from the canonical outcome matrix. Record construction now rejects
these values explicitly, using the same canonical paths/outcomes as the runner and
matched reporter. Valid zero counts, accepted labels and all metric definitions are
unchanged.

**Bounded ledger:** six scalar synthetic repros failed before repair; eight focused
record/report controls passed afterward. Root gate: 218 passed, 14 existing warnings,
no skips; Ruff lint/scoped format/whitespace and Pylance harness diagnostics passed.
There were no accepted data, inference studies, dependency or host changes.

Next: check whether stateless resubmission retains initial model identity when a
fresh session is constructed, rather than silently accepting mid-sequence model changes.

### 2026-11-13: stateless model identity and deterministic CI classification

Three tiny CPU repros showed full re-prefill accepting mid-sequence weight/configuration
changes and prefix-cached resubmission doing the same after its store was cleared.
Adapters now retain the initial session's supported CPU metadata and reuse the owning
live-session guard before resubmission. No live cache is retained solely for this guard.
Full re-prefill commits new history only after successful reconstruction; refused
decode/injection preserves the exact boundary. GPU/unversioned live-only behavior remains
unchanged; this is not concurrent model-mutation safety.

Seven focused drift/refusal, unversioned, pending-token, eviction and multi-call logit
controls passed after three initial failures. The first root gate passed 221 cases.
At that checkpoint, [earlier hosted CI][classification-ci-failure] showed one Python 3.12
failure (181 passed, 30 skipped): the SIGXCPU classification fixture received a wall-clock
timeout/SIGKILL instead of its expected CPU-limit diagnostic. Python 3.10 passed, and
[the newer record-validation CI][record-validation-ci] passed both runtimes. The log does
not establish host load or core-dump handling as the cause.

Classification now uses deterministic completed-process pipes/exit codes, independent of
interpreter startup and signal handling. Seven focused classification plus unchanged real
deadline/output/process-reaping controls passed; no deadlines/assertions were loosened or
tests skipped. Final root gate: 222 passed, 14 existing warnings, no skips; Ruff lint,
all four changed-file format checks, whitespace and session/serving diagnostics passed.
No dependencies, model/data downloads, pretrained/GPU jobs or host changes occurred.

Next: recover exact primary SGLang package/revision/cache documentation routes for the
still-unverified serving compatibility prerequisite; do not install dependencies or run
models under this read-only research scope.

[classification-ci-failure]: https://github.com/venim1103/neural-compiler/actions/runs/43411265117
[record-validation-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43411652981

### 2026-11-13: primary SGLang compatibility prerequisite recovered

Six primary requests succeeded: [PyPI metadata][sglang-metadata], release metadata,
tag resolution, the canonical documentation root, its linked index and the linked
[hybrid-memory page][sglang-hybrid-memory]. The four-request routine plan was extended
to six solely to follow the root's actual index/page links, not retry guessed 404 routes.
Inspected release `v0.5.23` resolves to source commit
`297e54fd105e974cdf401e5b4d20253a22c16f90`; neither is an adopted execution pin.

SGLang `0.5.23` declares Python `>=3.11`, Torch `==2.14.1`, Transformers `==5.19.0`
and tokenizers `==0.23.2`, versus verified installed `2.10.0+cu128`, `5.22.1` and
`0.23.3`. It also declares CUDA 13 packages. Installing it into the working base
would change incompatible dependencies; isolated integration remains separately scoped.
Apache-2.0 package metadata is not transitive licence review.

Current docs explicitly describe Qwen3.5 GDN support and conv/recurrent state slots
alongside full-attention KV. Their SWA-plus-state `extra_buffer` path cuts prefill chunks
on a checkpoint grid; this is not proof of our exact native tool boundary. Current docs
are not assumed version-bound to the inspected tag, nor numerical/GPU-fit/latency evidence.
No wheels, source checkout, weights or datasets were downloaded or executed.

The model-identity/classification repair's [hosted baseline][identity-repair-ci] passed.
No code gate was repeated for this read-only research update; its latest result is 222
local passing cases. Next: enforce positive integer standalone native framing budgets.

[sglang-metadata]: https://pypi.org/pypi/sglang/0.5.23/json
[sglang-hybrid-memory]: https://docs.sglang.io/docs/advanced_features/unified_memory.md
[identity-repair-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43412525283

### 2026-11-13: standalone native framing budgets reject invalid caps

The standalone protocol constructor accepted NaN, infinity, boolean and fractional
caps, unlike the evaluation budget contract. NaN/infinity could defeat the token
comparison. Both caps now require positive integers before tokenizer initialization;
the 4,096/2,048 defaults, native grammar and visible-truncation behavior are unchanged.
The touched protocol file was formatted with the existing repository formatter.

**Bounded ledger:** four synthetic repros failed before repair; all 21 protocol
controls passed afterward. Root gate: 226 passed, 14 existing warnings, no skips;
Ruff lint, both scoped format checks, whitespace and protocol diagnostics passed.
No downloads, pretrained/GPU runs, dependency or host changes occurred. These are
offline contract checks, not production containment or serving qualification.

Next: inspect case-level budget accounting at the serving/harness boundary for an
offline prerequisite; real-engine integration and accepted held-out data remain gated.

### 2026-11-13: CPU continuation identity includes generation EOS settings

The existing identity stamp omitted `generation_config.eos_token_id`. Stateless
re-prefill/cache-miss paths could therefore adopt new termination settings while
live/restored sessions retained the initial EOS set. Supported CPU prefix stamps
now include that setting, and the existing guards refuse drift before decode,
injection, capture or restore. Private checkpoint fields and valid token/turn
semantics are unchanged; GPU/unversioned live-only behavior remains unchanged.
This is not concurrent mutation safety or real-engine integration evidence.

**Bounded ledger:** five tiny CPU repros failed before repair; all 36 focused
serving/checkpoint controls passed afterward. The first scoped format check found
one pre-existing long line in the touched checkpoint test, which the formatter
repaired. The unchanged successful focused gate was not repeated. Root gate:
231 passed, 14 existing warnings, no skips; Ruff lint/scoped format/whitespace
passed, with no source diagnostics or test errors. Existing exact cached/live
and `atol=1e-6, rtol=1e-5` full-prefill controls are unchanged. No downloads,
pretrained/GPU runs, dependency or host changes occurred.

Next: continue the P1 resource-budget boundary prerequisites with bounded trusted
fixtures; real cgroup enforcement still requires separately qualified host support.

### 2026-11-13: jail setup consumes the execution allowance

Optional cgroup/seccomp setup previously finished before the child received its
full timeout, extending the remaining allowance passed by native compilation.
The executor now starts accounting before setup, passes only the remaining
allowance to child wall/CPU limits and refuses to launch when setup exhausts it.
Owned contexts still clean up, and timeout/setup failures remain explicit.
Synchronous host setup and cleanup are not made interruptible; this is not a
new hard whole-host deadline or cgroup kernel-enforcement qualification.

**Bounded ledger:** two synthetic control-file/clock repros failed before repair:
a 2-second request still granted 2 seconds after 1.5 seconds of setup, and launched
after 3 seconds of setup. Both now pass, including cleanup and the exact remaining
0.5-second allowance. All 56 focused cgroup/execution/native controls passed.
The first root gate passed 233 cases. One coupled real-child CPU-limit integration
case plus one namespace probe extended the routine verification plan: after a
synthetic 10-second setup debit under a 20-second allowance, the actual jailed
child reported the kernel CPU limit derived from the remaining supervisor
allowance. This read-only resource query used no busy loop, sleep or large
allocation and is not cgroup controller qualification. The new selector passed;
the revised root gate passed 234 cases, with 14 existing warnings and no skips.
Ruff lint, both scoped format checks, whitespace and executor diagnostics passed. Actual jailed
Python/C/assembly/Wasm/Unicorn controls ran; no deadlines, output caps or
assertions were relaxed. No downloads, pretrained/GPU jobs, dependencies or
host changes occurred.

Next: inspect remaining bounded resource-limit evidence prerequisites without
changing the shared host or accepting production policies.

The EOS checkpoint's [hosted baseline][eos-repair-ci] also completed successfully.

[eos-repair-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43415037944

### 2026-11-13: checkpoint capacity failures remain matched failed cases

Snapshot/store byte overflow previously raised generic `ValueError` and aborted
the serving comparison, losing complete failure-inclusive reports. Genuine
capacity failures now raise `CheckpointBudgetError`, still a `ValueError`, and
the harness records the existing `budget_exhausted` outcome. Unsupported caches,
invalid arguments and model-identity errors remain explicit errors, not budget
failures. There is no live-session fallback.

At startup the fresh per-case store has already cold-prefilled the prompt before
capture can overflow. Its known prompt/prefill count is retained, while unavailable
logits remain `None`. Turn-boundary failures retain sampled tokens and prevent
execution; other matched modes still run. Existing schemas, defaults, metrics and
token/logit tolerances are unchanged.

**Bounded ledger:** initial/turn synthetic threshold repros both aborted before
repair; the unsupported-cache control passed without being reclassified.
All 104 focused serving/checkpoint/evaluation controls passed afterward, including
all six intended matched records across the two threshold fixtures. No large
snapshot was allocated. Root gate: 237 passed, 14 existing warnings, no skips;
Ruff lint, four scoped format checks, whitespace and all three source diagnostics
passed. No data acceptance, pretrained/GPU runs, downloads, dependencies or
host changes occurred; these are not fit or statistical qualification measurements.

Next: inspect final-answer validity precedence for caller-supplied checker results
before advancing other matched-family reporting prerequisites.

The setup-deadline checkpoint's [hosted baseline][setup-deadline-ci] passed.

[setup-deadline-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43416605746

### 2026-11-13: invalid final-answer formats cannot count as correct

Caller-supplied checkers can distinguish a semantically correct value from valid
final-answer formatting. The harness previously prioritized `correct=True` over
`valid=False`, counting invalid output as success. It now checks validity first
and retains the supplied diagnostic. Built-in checker behavior, outcome labels,
metrics and generation accounting are unchanged.

**Bounded ledger:** one tiny scripted CPU repro failed before repair; all 66
evaluation controls passed afterward. Root gate: 238 passed, 14 existing warnings,
no skips; Ruff lint, both scoped format checks, whitespace and harness diagnostics
passed. No accepted labels/data, pretrained/GPU runs, downloads, dependencies or
host changes occurred.

Next: recover the cached target configuration and owning hybrid-state shapes for
a bounded offline checkpoint-budget estimate, not a model run or fit claim.

### 2026-11-13: target checkpoint storage lower bounds, metadata only

The locally cached [target configuration][qwen-cache-config] at snapshot
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` has 24 linear-attention and eight
full-attention layers. Installed Transformers 5.22.1's Torch-reference source
uses float32 recurrent states shaped `(batch, value_heads, key_dim, value_dim)`.
With batch one and the configured `32 × 128 × 128` states, recurrent storage
alone is **48 MiB**.

**Source-derived estimate, not a measurement:** assuming unquantized BF16 KV,
the eight full-attention layers require 32 KiB per forwarded token. Recurrent
plus KV lower bounds are 64 MiB at 512 tokens, 304 MiB at 8,192, and 560 MiB at
16,384. These omit convolution history, logits, extra buffers, weights, vision
activations and allocator overhead. Thus even the 512-token bound cannot fit
the default 64 MiB snapshot once positive extra storage is included. Forwarded
tokens exclude an unforwarded pending stop token.

**Bounded ledger:** eight initial metadata probes were extended to ten to follow
the actually listed snapshot and its installed owning state source; the direct
configuration read succeeded despite earlier glob misses. Three scalar estimates
completed. No model forward, weight deserialization, downloads, dependency or
host changes occurred. The cached revision is an inspected metadata identity,
not renewed execution authority. This is neither actual cache storage, GPU fit
nor an SGLang/vLLM allocation measurement; no cap increase is authorized by it.
The unchanged code suite was not rerun for these read-only findings.

Next: expose the already-existing explicit adapter snapshot override through the
comparison helper, preserving all defaults and requiring a separately scoped
actual run before nondefault limits become qualification evidence.

[qwen-cache-config]: https://huggingface.co/Qwen/Qwen3.5-4B/blob/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a/config.json

### 2026-11-13: explicit adapter snapshot overrides reach the comparison helper

`run_serving_comparison` now forwards optional `max_checkpoint_bytes` to the
existing cached adapter. `None` preserves the 64 MiB default additionally bounded
by the store; the aggregate store cap remains independent. Invalid overrides fail
before sampler/session creation. No default, metric, policy or actual-run budget
was increased, and exposing the argument does not authorize a model study.

**Bounded ledger:** three new selectors failed before forwarding existed. One
two-case tiny CPU fixture now reports the default overflow and succeeds only with
an explicit 65 MiB synthetic threshold; no large snapshot was allocated. Boolean
and zero overrides fail before prefill. All 42 focused serving/checkpoint controls
passed. The initial signature check identified three existing dependent executor
fixtures without the protocol's optional timeout default; their signatures were
aligned without changing current execution. All 11 resolved call sites are now
compatible. Root gate: 241 passed, 14 existing warnings, no skips; Ruff lint,
both scoped format checks, whitespace and serving diagnostics passed.

Next: verify target model-loader/output compatibility from local configuration
and installed classes before any separately authorized pretrained-model run.

### 2026-11-13: target loader metadata supports the existing causal interface

The cached architecture label prompted an output-type compatibility hypothesis.
Five bounded source/metadata probes disproved it: with the verified Pylance
interpreter `/opt/venv/bin/python` and installed Transformers 5.22.1,
`AutoModelForCausalLM` resolves the cached top-level `Qwen3_5Config` to
`Qwen3_5ForCausalLM`, whose declared configuration is `Qwen3_5TextConfig`.
Required `input_ids`, `past_key_values`, `use_cache` and `logits_to_keep`
arguments exist. `Qwen3_5CausalLMOutputWithPast` inherits
`CausalLMOutputWithPast`, so the existing output guard need not be weakened.

No model was instantiated, no weights loaded, and no unchanged code gate repeated.
Registry/signature compatibility does not prove weight loading, real 4B cache
behavior, GPU fit, accuracy or serving latency. Actual pretrained runs, pins,
dependencies and resource scopes remain separately gated.

Next: inspect the pinned SGLang release's actual cache/API boundary hooks, rather
than treating current documentation as version-bound checkpoint evidence.

### 2026-11-13: pinned SGLang API and checkpoint-source evidence

At inspected source `297e54fd105e974cdf401e5b4d20253a22c16f90`,
[the engine declaration][sglang-engine-source] accepts token `input_ids`,
sampling parameters, session identifiers/parameters and `cache_salt`.
[Its base interface][sglang-engine-base] declares cache flushing and memory
occupation release/resume; memory resume is not proof of token-boundary
pause/inject/resume. The engine page was truncated, so absence of additional
concrete methods is not established.

[The optional Mamba checkpoint pool][sglang-checkpoint-source] compresses cached
recurrent state to int8, then dequantizes into active slots and copies convolution
history. This is not our exact tensor-copy checkpoint path; its source's quality
claims are not accepted accuracy/exactness evidence. Keep compression out of an
exactness baseline unless separately qualified. The pool also states its extra
allocation is not charged to `max_total_num_tokens`; actual resource qualification
must account for it independently. The earlier Transformers storage lower bounds
must not be presented as this compressed engine's actual allocation.

**Bounded ledger:** six primary requests, one failed and five successful. The
first recursive root tree exceeded the 4 MiB read cap; the cap was not increased.
Two narrower metadata queries followed the actual Python subtree. The original
three-query plan was extended to five for that recovery, then six to follow the
actual `EngineBase` import. Source pages retained their 20,000-character caps.
No source checkout, engine execution/installation, weights, datasets or host
changes occurred. No unchanged code suite was rerun. Exact native stop-token
accounting, boundary cache restoration, multi-call logits and hardware fit remain
unverified for this engine; inspected sources are not adopted execution pins.

[sglang-engine-source]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/entrypoints/engine.py
[sglang-engine-base]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/entrypoints/EngineBase.py
[sglang-checkpoint-source]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/mem_cache/mamba_checkpoint_pool.py

### 2026-11-13: jail startup status is independent of guest stderr

The supervisor previously classified any nonzero exit whose stderr began with
`bwrap:` as an isolation failure. A trusted jailed print/exit fixture reproduced
this misclassification. Stderr now remains guest data; a separate startup
acknowledgement distinguishes genuine setup failures from guest runtime errors.

The trusted resource-limit launcher reports readiness through an anonymous
host-created Unix datagram endpoint temporarily used as stdin. It restores
read-only `/dev/null` at fd0 and closes its private writer before guest exec.
The parent reads at most the acknowledgement length plus one byte, rejects
missing/invalid reports explicitly and closes both endpoints. The endpoint is
unnamed IPC, not a listening service; guest syscall allowlists are unchanged.
Existing wall/output limits, cgroup status handling and no-fallback behavior remain.

**Bounded ledger:** the initial five selectors yielded four failures and one
passing genuine setup-error control. An `--info-fd` implementation passed four
but failed that control: bubblewrap reports its child PID before mounts finish.
That report was not accepted as readiness evidence. The private-stdin repair
passed five selectors, then its `dup2` operation broke two existing seccomp
controls (99 passed). Reopening fd0 with already-supported operations preserved
all 101 controls without broadening policies. One bounded owned-jail monitor
probe then found the regular-file acknowledgement reachable through
`/proc/1/fd/0`; self-descriptor closure alone was insufficient. The final
datagram design passed the strengthened descriptor selector, including monitor
reopen refusal and parent cleanup. One overlong-report parameter extended the
new fixture count to six. All 102 coupled execution controls and the root gate
passed: **247 tests**, 14 existing warnings, no skips. Root Ruff lint, all three
scoped format checks, whitespace and execution diagnostics passed; all four
resolved supervisor callers are signature-compatible.

Actual jailed Python/C/assembly/Wasm/Unicorn and trusted seccomp fork/socket-denial
controls ran. Namespace-local descriptor checks are not complete ptrace/process
memory or production containment qualification. The missing local manpage lookup
and failed approaches remain spent evidence. No downloads, dependencies, model/
GPU jobs, service exposure or shared-host privilege/security changes occurred.
Owned descriptor channels are closed; no persistent runtime artifacts were added.

Next: advance another bounded offline evaluation prerequisite while retaining
the separate dataset, real-engine and writable-cgroup qualification gates.

The startup-channel checkpoint's [hosted CI][startup-channel-ci] succeeded at
`bbb1aa8`. Public metadata also confirmed successful runs for `e73de2b` and
`d44d647`, which had not previously been inspected. The configured GitHub
connector returned 404 from a different host; the public GitHub API provided
this evidence without any connector, credential or security changes.

[startup-channel-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43429641390

### 2026-11-13: matched synthetic suites validate seeds before arithmetic

The single-family generator rejected non-integer seeds, but the matched-suite
entry point first added its family offset. Booleans became integers and silently
bypassed validation; string seeds raised `TypeError` instead of the explicit
contract error. The matched entry point now applies the same exact-integer guard
before arithmetic. Valid integer seeds, offsets, generated cases, checkers and
report metrics are unchanged.

**Bounded ledger:** three invalid-seed repros failed and five existing input
controls passed. After repair, all 69 evaluation controls and the root gate
passed: **250 tests**, 14 existing warnings, no skips; root Ruff lint and family
diagnostics were clear. The scoped formatter initially flagged the existing
single-family signature layout and directly dependent matched-suite call. Both
were normalized by Ruff; scoped format, lint and whitespace checks then passed.
The unchanged successful behavior gate was not repeated for formatting alone.
No datasets, labels, pretrained-model/trajectory runs, dependencies or host changes
were accepted or performed. Synthetic fixtures remain unit evidence, not a study.

Next: check matched-family aggregation inputs against the existing pairing
contract, without changing accepted metric definitions.

### 2026-11-13: checker-result flags must satisfy their boolean contract

Matched-report pairing already rejects missing/duplicate paths and mismatched
families. A related scoring boundary was still permissive: custom `CheckResult`
objects could contain truthy strings in `correct` or `valid`, and both malformed
cases entered correctness scoring. The result constructor now rejects
nonboolean flags with an explicit `ValueError`. This implements the existing
typed contract, not a new correctness metric. Genuine `True`/`False` flags retain
their behavior, including semantically correct but invalidly formatted answers.

**Bounded ledger:** two tiny forced-answer CPU repros failed before repair.
All 71 focused evaluation controls and the root gate now pass: **252 tests**,
14 existing warnings, no skips; root lint, both scoped format checks, whitespace
and harness diagnostics are clear. Existing checker diagnostics are preserved.
No dataset/cohort acceptance, pretrained inference, trajectory collection,
dependency, GPU or host changes occurred.

Next: expose the existing paired comparisons within each matched family,
keeping aggregate fields and comparison definitions unchanged.

### 2026-11-13: paired comparison effects remain visible within each family

`matched_report` now adds `paired_deltas_by_family`, keyed by the existing
validated family identities. Each family retains the same
`python_tool_vs_no_tool` and `current_best_vs_python_tool` comparisons as the
aggregate report: correctness contingency counts and failure-inclusive elapsed
deltas. A shared wrapper reuses the existing comparison implementation. Existing
aggregate fields, per-path/per-family cost maps and metric definitions are unchanged.
This exposes heterogeneous effects, not new confidence intervals or accepted
held-out-family evidence.

**Bounded ledger:** one six-row, two-family synthetic fixture demonstrates
opposing accuracy effects that cancel in both aggregate comparisons, with
distinct elapsed deltas including failed answers. The first attempt failed on
omitted required fixture counters; after correcting those, the missing-field
repro failed at the required output key. The repaired feature selector passed.
All 94 focused evaluation/serving controls and the root gate passed: **253 tests**,
14 existing warnings, no skips; root lint, both scoped format checks, whitespace
and harness diagnostics were clear. No inference was added to this fixture,
and no dataset, model-study, dependency, GPU or host changes occurred.

The matched-seed checkpoint's [hosted CI][matched-seed-ci] succeeded at `5d66671`;
the checker and family-report checkpoints have not yet had hosted results
inspected.

[matched-seed-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43430515411

Next: finish bounded inspection of the pinned engine's remaining declaration;
do not infer exact token-boundary checkpoint hooks from a truncated page or
install an incompatible engine without a separately scoped runtime directive.

### 2026-11-13: pinned engine sessions are an integration candidate, not a cache proof

The complete concrete `Engine` declaration exposes `open_session`/
`close_session` in addition to generation and memory-release/resume operations.
Its `open_session` documentation describes `streaming=True` as a low-overhead,
append-only conversation path; `capacity_of_str_len` is documented as string
capacity, not a measured token/tensor memory limit. This is a concrete candidate
for future multi-turn integration, not proof that host iterator backpressure
pauses decoding or that native call-boundary Gated DeltaNet state is preserved.

[Pinned request definitions][sglang-session-requests] include `session_id`,
`session_params` and token `input_ids`; `SessionParams` contains `id`, `rid`,
`offset`, `replace` and `drop_previous_output`. Do not apply those rewind/replace
controls to hybrid state without exact-boundary qualification. The concrete
declaration defines no dedicated token pause/inject/checkpoint method, but it
inherits mixins and exposes `collective_rpc`; this is not an absence claim about
inherited, scheduler or private interfaces. Memory resume remains a different
operation from demonstrated decode continuation.

**Bounded ledger:** three capped 20,000-character source pages still ended within
the 86,357-byte/2,097-line engine file. One necessary parse-only recovery read
capped the body at 128 KiB and metadata output at 20,000 characters, then inspected
the complete concrete method inventory without importing/executing source.
One directly referenced request-definition read (111,191 bytes, same caps)
established the fields above. Five requests succeeded; no partial oversized body
was accepted. Source was not checked out or installed, and no plugins, subprocess
launchers, weights, data or model/host jobs were run. No unchanged code tests were
repeated for these documentation facts.

Exact stop-token inclusion, pending-token placement, multi-call hybrid logits,
session cache retention/precision, resource accounting and fit remain runtime
qualification gates under a separate dependency/model directive. Keep the
existing CPU fixture evidence and compressed checkpoint estimates distinct.

[sglang-session-requests]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/io_struct.py

Next: inspect the actual session implementation's append-only state ownership,
without executing the engine or turning parameter names into correctness evidence.

### 2026-11-13: append-only session record ownership traced at the pinned source

[Streaming session slots][sglang-streaming-session] own a shared `ReqKvInfo`
record and prefix-tree lock; successive turns borrow that same record.
Non-aborted completion retains it, while session release frees both KV rows and
Mamba pool/ping-pong allocations. Runtime abort releases the borrowed record and
leaves the previous successful request as the next prefill history. This is
source-level ownership evidence, not a numerical Gated DeltaNet continuation test.

[The controller][sglang-session-controller] rejects concurrent streaming turns,
replace/drop controls and nonzero offsets. Successful completion commits one
request node and token-array lengths; abort retains the preceding node.
Its plain-append helper trims unconfirmed array tails, then appends the previous
request's retained output IDs and the new input suffix. A future live-session
adapter therefore must not submit our full-history resubmission input as an
append suffix. The helper also strips a leading appended BOS; native template
and stop-token accounting still require explicit end-to-end qualification.

Close while a request is active defers cache release. Finished-length/speculative
overshoot trimming, one-token logit reserve and backend page alignment appear
in the cache path; none of these establishes the exact pending-token/recurrent
state boundary without the actual backend checks.

**Bounded ledger:** the complete pinned Python tree was 1,794,535 bytes under the
4 MiB cap. The actual streaming file was 12,478 bytes. The 20,999-byte controller
exceeded the 20,000-character output cap, so its complete method inventory was
emitted first, then two narrowly scoped reads emitted complete lifecycle and
append helper definitions. All source bodies remained under 128 KiB; five
requests succeeded, with no imports/exec, source checkout, engine/dependency,
weights/data/GPU or shared-host changes. No unchanged code tests were rerun.

[sglang-streaming-session]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/session/streaming_session.py
[sglang-session-controller]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/session/session_controller.py

Next: preserve these candidate/qualification limits and continue bounded local
cache-boundary checks, without installing or executing the pinned engine.

### 2026-11-13: reject decode budgets before touching a continuation boundary

The [live decoder][decode-budget-session] checked only positivity before
forwarding a pending stop token. Fractional budgets then failed in `range`
after that work, while boolean budgets were treated as integers. The
[stateless adapters][decode-budget-serving] also restored a checkpoint or
re-prefilled the full history before rejecting even a zero budget.

One shared positive-exact-integer guard now runs before those operations.
Valid decode behavior, pending-token accounting, model-identity checks and
existing token caps remain unchanged; invalid requests raise explicit
`ValueError` without forwarding tokens or reconstructing a session.

**Bounded ledger:** four tiny CPU failing-first cases reproduced pending-token
forwarding for `1.5`/`True` and stateless reconstruction for zero. After the
guard repair, all four passed; the coupled checkpoint/serving/protocol gate
passed **67 tests** in 5.50 s. The required root gate passed **257 tests** in
14.37 s, with 14 existing Torch JIT warnings and no skips; root Ruff,
four-file format checks, whitespace and both owning-file Pylance diagnostics
passed. No dependencies, pretrained/data/GPU jobs or host changes were made.
Hosted CI for the preceding `52dbc5e` push succeeded (run `43432446483`);
that workflow status is not additional containment qualification.

[decode-budget-session]: ../../src/neural_compiler/bridge/session.py
[decode-budget-serving]: ../../src/neural_compiler/eval/serving.py

Next: checkpoint and synchronize the verified local repair, then continue the
remaining bounded offline evaluation/cache/containment prerequisites.

### 2026-11-13: snapshot overrides are admitted before prefix-cache work

The snapshot and store primitives already rejected non-integer/nonpositive
limits. However, the [prefix adapter][decode-budget-serving] checked its
selected snapshot override only on a later capture: cold construction did
prefill work first, while warm exact-prefix hits skipped constructor capture
and could accept an invalid override indefinitely.

The adapter now reuses the snapshot primitive's extracted private validator
before prefix lookup. Existing `max_cache_bytes` diagnostics, default bounds,
valid cache hits and genuine `CheckpointBudgetError` overflow classification
remain unchanged; no new snapshot or store cap was introduced.

**Bounded ledger:** cold-zero and warm-NaN CPU cases initially both reached
prefix resumption; after the guard repair both passed in 0.06 s. One formatting
failure identified only the new constructor call's line layout; scoped Ruff
normalization repaired it. Coupled checkpoint/serving controls passed **48
tests** in 3.74 s; the root gate passed **259 tests** in 15.22 s with 14 existing
warnings and no skips. Root lint/whitespace, three-file format checks and both
source Pylance diagnostics passed. No dependencies, model studies, dataset,
GPU jobs or shared-host changes occurred; no temporary artifacts remain.

Next: retain the valid low-cap overflow controls, checkpoint the repair and
continue bounded offline evaluation/containment prerequisites.

### 2026-11-13: native preparation shares the execution wall allowance

The [native bridge][native-preparation-source] started C/assembly deadlines
after writing source files, and gave Wasm/hex workers a fresh allowance after
copying their trusted runtime and source. Deterministic fixtures showed these
outer setup stages were not charged even though inner jail setup already was.

A shared validated deadline now starts before temporary-directory/runtime
preparation. Compilation and worker launch receive only the remaining time;
expired setup returns `timeout` without launching a child. Compile/run
classification, writable compile/read-only execution mounts, seccomp/cgroup
forwarding, source/output bounds and backend work limits remain unchanged.
Setup I/O and cleanup are still synchronous, not interruptible hard host
deadlines; this repair is not production containment qualification.

**Bounded ledger:** four failing-first controls covered C setup expiry,
assembly's remaining compile/run allowances, Wasm worker allowance and hex
setup expiry. After repair, those plus the existing compilation-expiry case
passed. The latter now explicitly confirms compilation occurred, using three
isolated clock observations. One formatting failure concerned only the new
clock fixture's line layout; scoped normalization repaired it.
Native/execution/cgroup controls passed **67 tests** in 1.75 s; root validation
passed **263 tests** in 13.12 s, with 14 existing warnings and no skips.
Root lint/whitespace, two-file formatting, native diagnostics and both resolved
private-helper callers' signature compatibility passed. Existing jailed
Python/C/assembly/Wasm/Unicorn and trusted seccomp controls ran; writable-cgroup
kernel enforcement remains unqualified. No downloads, dependencies, model/
data/GPU jobs, services or host changes occurred; temporary workdirs were cleaned.

The preceding serving checkpoints' hosted CI succeeded: [decode admission
`5b2a085`][decode-admission-ci] and [snapshot admission
`422d184`][snapshot-admission-ci]. This is workflow-status evidence, not a new
kernel/engine qualification result.

[native-preparation-source]: ../../src/neural_compiler/bridge/native.py
[decode-admission-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43436029043
[snapshot-admission-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43436974498

Next: checkpoint the verified deadline repair, then continue bounded offline
prerequisites while retaining the separate real-engine and containment gates.

### 2026-11-16: Python import/source preparation consumes the same allowance

The [Python wrapper][python-preparation-source] likewise passed a fresh timeout
after static import-policy evaluation and source-file preparation. It now starts
accounting before these stages, passes only the remaining allowance to the jail,
and returns `timeout` without launch when setup consumes it. Python, native
wrappers and jail admission reuse one extracted timeout/output validator;
existing finite-positive time and positive-integer output contracts are unchanged.
Rejected/syntactically invalid import policies remain explicit `sandbox_error`.
The wrapper docstring now distinguishes the hard subprocess deadline from
synchronous, non-interruptible setup and cleanup.

**Bounded ledger:** two failing-first fixtures wrapped the actual compliant
math-import policy and source write with isolated deterministic clocks. Two
seconds of preparation originally still granted 10 seconds; 12 seconds still
launched. After repair, the fixtures passed with 8 seconds or no child, in
0.03 s. Coupled execution/native/import/seccomp/cgroup controls passed **88
tests** in 2.31 s; root validation passed **265 tests** in 21.28 s, with 14
existing warnings and no skips. Root lint/whitespace, three-file formatting
and both source Pylance diagnostics passed. Existing real jailed tier/seccomp
controls ran, without new kernel containment qualification.
One unrelated protocol-source range lookup exceeded that file's length; the
correct bounded read recovered it without editing. No new dependencies,
model/data/GPU jobs, services or shared-host changes occurred.

The native deadline checkpoint's [hosted CI][native-preparation-ci] succeeded
at `05ff5d9` (run `43438563896`).

[python-preparation-source]: ../../src/neural_compiler/bridge/execution.py
[native-preparation-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43438563896

Next: checkpoint and synchronize this dependent repair, then continue bounded
offline cache/evaluation prerequisites without reopening protected studies.

### 2026-11-16: CLI source ingestion stops at the byte-budget sentinel

All four [CLI file actions][cli-ingestion-source] previously read complete files
before execution/import inventory enforced the 64 KiB source cap. A shared
local reader now reads at most 65,537 bytes, rejects overflow before decoding
or processing, and retains universal-newline normalization for in-limit UTF-8
source. The raw-byte cap also prevents newline normalization from admitting
oversized files. Existing explicit file/UTF-8/size errors and execution JSON
remain; this is bounded ingestion, not untrusted-workload qualification.
The README usage section documents this input contract without progress history.

**Bounded ledger:** instrumented C/import-inventory reads both initially used
unbounded text ingestion; the exact-65,536-byte multibyte/CRLF compatibility
control already passed. All three passed after repair. Source fixtures were
at most 65,538 bytes; no source was actually executed by the new fixtures.
An initial patch's next-test anchor was wrong and changed nothing; a bounded
read recovered the correct anchor. One ledger SQL syntax error was corrected
before execution. A formatter failure concerned only two shortened CLI calls;
scoped normalization repaired them.
Coupled execution/native/import controls passed **57 tests** in 1.43 s; root
validation passed **268 tests** in 15.40 s with 14 existing warnings and no
skips. Root lint/whitespace, two-file formatting and CLI diagnostics passed.
No dependencies, model/data/GPU jobs, services or host changes occurred.

[cli-ingestion-source]: ../../src/neural_compiler/cli.py

Next: checkpoint/synchronize the bounded reader, then retain native framing,
exact cache boundaries and separately gated serving/containment qualification.

### 2026-11-16: pinned internal stop-token retention is source-confirmed

The pinned [request finish implementation][sglang-request-finish] uses
`update_finish_state`, not the older `check_finished` name. Token/EOS stops set
`finished_len` to the matched token's position plus one. Streaming completion
then retains `output_ids[:finished_len]`, and the append helper carries that
retained output before the new suffix. This establishes internal stop-token
inclusion at this source revision, not public streaming output visibility or
whether the terminal token has already entered recurrent/KV state.

String/regex stops precede token/EOS checks; speculative stop positions beyond
`max_new_tokens` are demoted to a length finish at that cap. `ignore_eos` skips
the entire token-stop helper, including explicit `stop_token_ids`: a future
native-delimiter adapter must account for that coupling rather than assuming
independent controls. Public stop trimming, `no_stop_trim`, speculative/page
alignment and pending-token numerical equivalence still need actual backend
qualification before implementation/adoption.

**Bounded ledger:** streaming/controller bodies were 12,478/20,999 bytes.
The imported request source declared 178,701 bytes, exceeding the initial
128 KiB cap; that attempt stopped and was not parsed as complete evidence.
A narrowly revised 192 KiB recovery parsed it completely. Its legacy-name
selector did not emit the actual finish methods; one final bounded query
recovered the complete method inventory and five finish branches, under an
18,000-character combined definition-output cap. Five requests total, including
the failed size-bound attempt, used the same pinned revision with 20-second
request limits. No imports/exec, checkout, installation, model/data/GPU jobs
or host changes occurred; no source bodies were retained. Documentation-only
validation checks whitespace/references; the unchanged code suite was not rerun.

Hosted CI succeeded for [Python preparation `653b5ba`][python-preparation-ci]
and [CLI ingestion `be14539`][cli-ingestion-ci]. UTC/commit metadata confirmed
these checkpoints occurred on November 16; their two update headings were
corrected rather than inheriting the older November 13 date.

[sglang-request-finish]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/schedule_batch.py
[python-preparation-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43827061233
[cli-ingestion-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43827565988

Next: preserve the internal-versus-public/pending boundary distinction and
continue compatible offline prerequisites without instantiating the engine.

### 2026-11-16: pinned detokenization separates text trimming from token output

The request's `output_ids_through_stop` property explicitly includes the stop
position and feeds incremental detokenization. The pinned
[detokenizer][sglang-detokenizer] may remove a matched token before decoding text
when `no_stop_trim` is false, with exceptions for configured tool-close IDs.
String-stop trimming similarly changes rendered text. This is not evidence that
all target-specific tool markers are configured as exceptions.

`handle_batch_token_id_out` forwards `recv_obj.output_ids` unchanged into
`BatchStrOutput`, separately from trimmed `output_strs`. A future adapter must
therefore distinguish token-array framing from rendered-text visibility;
`no_stop_trim` alone is not proof of exact terminal-token accounting. Final
public response assembly and recurrent/KV pending-token behavior remain
qualification checks, not implemented engine integration.

**Bounded ledger:** two same-revision, parse-only requests recovered the
178,701-byte request source under 192 KiB and 23,531-byte detokenizer under
128 KiB, each with a 20-second timeout. Complete selected methods fit their
16,000/15,000-character output caps; the optional third request was unnecessary.
No imports/exec, checkout, dependency changes, engine/model/data/GPU or host
actions occurred; no source files were retained. Documentation references and
whitespace were checked; no unchanged code gate was rerun.

[sglang-detokenizer]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/detokenizer_manager.py

Next: trace final response assembly only as needed for the exact token-ID
qualification plan; do not substitute trimmed text for native protocol tokens.

### 2026-11-16: no-tool records cannot include execution or injected tokens

The [evaluation harness][no-tool-record-source] disables execution/injection
for `no_tool`, but manually constructed `CaseResult` rows could still carry
executed calls or positive injected-token counts under that label. Both entered
summaries, contaminating the baseline's execution/token costs without rejection.

The result producer now rejects those impossible combinations with explicit
`ValueError`. Model-generated tool syntax can still produce `protocol_error`
with zero executed calls; tool-enabled paths retain failure-inclusive calls
and response accounting. No metric formulas, baseline prompts, accepted cohorts
or outcome definitions changed. One existing report fixture incorrectly copied
a tool-enabled row into all three paths; its no-tool row now removes calls and
injection and verifies zero baseline execution cost.

**Bounded ledger:** two producer-only failing-first cases covered a failed
execution with zero response tokens and positive injection without calls.
After repair, both plus the related report control passed. Coupled
evaluation/serving controls passed **100 tests** in 7.69 s; root validation
passed **270 tests** in 14.45 s, with 14 existing warnings and no skips.
Root lint/whitespace, two-file formatting and harness diagnostics passed.
Existing tiny CPU matched/no-tool controls actually ran; this is not accepted
task data or pretrained-model evidence. No dependencies, model/data/GPU jobs,
services or host changes occurred.

[no-tool-record-source]: ../../src/neural_compiler/eval/harness.py

Next: checkpoint/synchronize the verified baseline repair and continue
compatible offline prerequisites without weakening the remaining study gates.

### 2026-11-16: injected totals reconcile with per-call response tokens

The harness assigns each successful tool injection's length both to the case
total and its corresponding call record; failures before injection retain zero
response tokens. Nevertheless, manual tool-path rows could report a different
total, dropping or inventing injected tokens in aggregate costs.

`CaseResult` now requires `injected_tokens` to equal the sum of its calls'
`response_tokens`, after the existing individual-counter/no-tool checks.
Mismatches raise explicit `ValueError`; neither the metric definition nor
actual runtime accounting changes. Existing valid multi-call and failed-call
controls remain unchanged; no synthetic fixture needed correction.

**Bounded ledger:** two producer-only failing-first cases reported totals of
three/five against a failed runtime call's four response tokens. Both plus
related response/report controls passed after repair. Coupled evaluation/
serving controls passed **102 tests** in 9.14 s; root validation passed **272
tests** in 16.65 s, with 14 existing warnings and no skips. Root lint/whitespace,
two-file formatting and harness diagnostics passed. Tiny CPU matched controls
ran, with no accepted data, pretrained/model/GPU study, dependency or host work.

Next: checkpoint/synchronize the reconciled records and continue the remaining
bounded offline protocol/cache/execution prerequisites.

### 2026-11-16: execution result and call labels use the closed status contract

`ExecutionResult` declared a six-label `Status` contract but admitted arbitrary
runtime labels, and `CallRecord` likewise accepted unrecognized statuses.
Unknown labels could reach native result JSON or execution records rather than
surfacing a trusted producer error.

Both producers now reuse one private validator derived from the existing
`Status` literal. Unknown labels raise explicit `ValueError` before serialization
or reporting. All six existing statuses and payload fields remain unchanged;
no execution failure is renamed, defaulted to success or silently discarded.

**Bounded ledger:** two failing-first producer cases used `success` and
`unrecorded_failure`; both passed after repair. Coupled execution/evaluation/
native controls passed **123 tests** in 6.68 s; root validation passed **274
tests** in 13.45 s with 14 existing warnings and no skips. Root lint/whitespace,
three-file formatting and both source diagnostics passed. Test diagnostics
contained only 17 existing unused-parameter hints, not new errors; those were
left alone. Existing jailed/tiny CPU controls ran, without model/data/GPU,
dependency or shared-host changes. Potential downstream scoring impact is a
design risk, not a newly measured pretrained-model outcome.

Hosted CI succeeded for the preceding [no-tool baseline checkpoint
`9111792`][no-tool-record-ci] and [token reconciliation
`4bc3129`][response-token-ci]; neither workflow status replaces study or
containment qualification.

[no-tool-record-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43828769679
[response-token-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43829173823

Next: checkpoint/synchronize closed-status admission and continue bounded
offline prerequisites under the existing protected study/containment limits.

### 2026-11-16: pinned response assembly distinguishes delta and cumulative IDs

The pinned [tokenizer manager][sglang-final-response] has an asynchronous
`_handle_batch_output`. Its non-beam string/token output paths extend request
state with received IDs without text-based stop trimming. Incremental streaming
branches publish delta IDs; other streaming branches publish cumulative IDs,
using the live list for unfinished output and a copy at completion. Nonstreaming
completion also returns a copy. `generate_request` yields its response objects.

A future adapter therefore must select/verify the actual output mode and
deduplicate cumulative emissions rather than appending every yielded list.
Rendered text and raw IDs remain separate. This consumer-side source evidence
does not qualify scheduler-side emission/overshoot ordering, target-specific
tool-close configuration, exact pending hybrid state or actual engine behavior.
No concrete/inherited/RPC pause-interface absence is claimed: this file imports
pause/continue request types and a control mixin, which were not investigated
or executed in this slice.

**Bounded ledger:** HEAD reported 177,003 bytes; a complete parse fit the
256 KiB/20-second source cap. The whole handler exceeded the 15,000-character
definition-output cap and was not emitted. One targeted recovery failed because
its selector accepted only synchronous functions; correcting it to accept the
actual asynchronous definition recovered complete ID assignments, dictionaries
and then/else ancestry under a 14,000-character statement cap. Four requests
total, including the failed selector query, at the same pin. No imports/exec,
checkout, engine/dependency/model/data/GPU or host changes occurred; no source
bodies were retained. Documentation whitespace/references were checked; the
unchanged 274-test code gate was not rerun.

[sglang-final-response]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/tokenizer_manager.py

Next: retain mode-specific token accounting in the engine qualification plan
and continue compatible local prerequisites without installing the engine.

### 2026-11-16: successful execution results cannot contain failed exits

The supervisor and native workers already classify nonzero process exits as
failures, but manually constructed `ExecutionResult` objects could combine
`ok` with a nonzero or signal return code. The result producer now rejects
that contradiction explicitly. Successful nonprocess results may still use
`None`; successful process results use zero. Failure statuses retain their
existing codes, including setup failures with a zero child exit.

**Bounded ledger:** two producer-only cases for exit code one and SIGKILL
initially failed, then passed with the existing deterministic classification
controls. Coupled execution/native/evaluation controls passed **125 tests** in
8.45 s; root validation passed **276 tests** in 17.34 s, with 14 existing Torch
JIT warnings and no skips. Root lint/whitespace, two-file formatting and source
diagnostics passed. One earlier focused run additionally emitted a CUDA
initialization warning and two FLA CPU-fallback warnings; no hardware cause
or CUDA/Triton qualification is inferred. No dependencies, model/data/GPU jobs
or shared-host changes were made.

Next: checkpoint the verified result-contract repair and continue useful
offline evaluation and serving prerequisites within the existing gates.

### 2026-11-16: pinned pause controls are global server controls

The pinned [pause/continue requests][sglang-session-requests] have no request or
session selector. Their documented modes are global abort, in-place scheduler
pause and retract-to-waiting-queue pause. In-place mode documents old-KV reuse;
retract mode permits cache flushing/recomputation. These are documented server
contracts, not measured hybrid-cache equivalence.

The [tokenizer manager][sglang-final-response] sets a shared pause condition.
Non-abort modes dispatch to the scheduler; abort mode issues `abort_all=True`
until the model-update lock is free. Continue clears that shared condition and
dispatches the continuation request. The latter defaults to requesting
`torch_empty_cache`; no CUDA action was invoked during this inspection.

These controls are not a proven per-sequence tool-call injection boundary.
Append-only streaming sessions remain the source-backed candidate for that
integration; concrete scheduler behavior, multi-request interference and exact
pending recurrent/KV state still require separate qualification.

**Bounded ledger:** three parse-only requests inspected 43,973-byte control
mixin and 111,191-byte request definitions under 128 KiB, then the 177,003-byte
actual producer under the previously verified 256 KiB cap. Each had a
20-second timeout and 16,000-character selected-output cap. The mixin did not
own pause methods, so the third lookup followed the known producer rather than
claiming interface absence. No scheduler consumer or engine was executed,
source bodies were not retained, and no dependencies/model/data/GPU/host
changes occurred. Documentation whitespace/references were checked; the
unchanged 276-test code gate was not rerun.

Next: keep global pause distinct from sequence-local append and qualify the
actual scheduler/cache semantics before any serving-engine adoption.

### 2026-11-16: token IDs are validated before model or prefix-cache work

The live decoder constructed long tensors from caller-provided IDs without
validating their representation. Fractional IDs were truncated for inference
but retained in token history. Cache keys also allowed booleans to alias
integer IDs under Python equality, silently restoring a different typed input.

One shared exact-integer validator now runs before model forwarding and both
store key-lookup paths. Invalid IDs raise explicit `ValueError` before forwarding
or changing hit/miss/LRU state. No new vocabulary/context limit or negative-ID
policy was introduced. A rejected response preserves the pending stop token,
history, logits and forwarded cache length; subsequent valid injection still
matches full-prefill logits at the unchanged `atol=1e-6`, `rtol=1e-5`.

**Bounded ledger:** three tiny CPU failing-first cases covered fractional
prompt/response IDs and a boolean cached-prefix alias. All passed after repair.
Coupled checkpoint/serving/protocol controls passed **72 tests** in 5.72 s;
root validation passed **279 tests** in 14.43 s with 14 existing warnings and
no skips. Root lint/whitespace, two-file formatting and source diagnostics
passed. One earlier line-based symbol lookup returned no result and was not
treated as caller evidence; no signature changed. No dependencies, pretrained
model/data/GPU jobs, services or shared-host changes occurred.

Hosted CI also succeeded for [closed-status admission `65c62bd`][closed-status-ci]
and [exit consistency `ddc8c6b`][exit-consistency-ci], without adding engine or
containment qualification.

[closed-status-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43829630999
[exit-consistency-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43847404756

Next: checkpoint/synchronize token-ID admission and continue useful offline
protocol/cache/evaluation prerequisites without claiming engine qualification.

### 2026-11-16: outer workspace I/O failures remain explicit execution results

Jail setup already translated infrastructure `OSError` failures to
`sandbox_error`, preserving any completed result. Outer Python/native temporary
directory, source-write and trusted-runtime-read failures instead escaped that
contract; cleanup errors could also discard completed output from the caller.

The wrappers now classify only these infrastructure I/O exceptions through the
same extracted result-preserving constructor. Cleanup failures retain stdout,
existing diagnostics and child return code while surfacing `sandbox_error`.
Invalid-input `ValueError` and unrelated non-I/O exceptions are not caught.
Compilation/runtime status mapping, deadlines, jail mounts and all work/output
caps remain unchanged; no unjailed fallback or best-effort cleanup success is
introduced.

**Bounded ledger:** four failing-first synthetic cases covered Python source
write, C temporary-directory creation, Wasm trusted-runtime read and cleanup
after a completed Python result. All passed after repair, including preserved
output/diagnostics and zero child launches on preparation failure. Coupled
execution/native/seccomp/cgroup controls passed **85 tests** in 2.07 s; root
validation passed **283 tests** in 16.36 s with 14 existing warnings and no
skips. Root lint/whitespace, three-file formatting and both source diagnostics
passed. Existing real jailed Python/C/assembly/Wasm/Unicorn and trusted seccomp
controls ran, without production containment qualification. No dependencies,
model/data/GPU jobs, services or host changes occurred; fixture workdirs cleaned.

Next: checkpoint/synchronize infrastructure failure handling, then continue
compatible offline evaluation/cache prerequisites under the existing gates.

### 2026-11-16: pinned scheduler emits through-stop token deltas

The actual scheduler imports a composed output streamer, rather than inheriting
the output implementation. Its [generation accumulator][sglang-output-streamer]
selects `req.output_ids_through_stop`, emits the suffix after `send_token_offset`,
then advances that offset to the retained length. Its payload constructor passes
those deltas directly into `BatchTokenIDOutput.output_ids`. Finished requests
emit once, marking `finished_output`; unfinished emission may be deferred by
stream intervals or a possible stop-string prefix.

Combined with the earlier request/detokenizer/tokenizer-manager reads, this
closes the inspected Python non-beam source path: stop-inclusive internal IDs
become scheduler deltas, remain separate from text trimming, and are exposed
as delta or cumulative IDs according to public response mode. It is not a
runtime acceptance result. Scheduler ordering, speculative/page effects,
actual target markers and terminal recurrent/KV state still require integration
qualification; Rust/disaggregated/beam modes are not certified by this trace.

**Bounded ledger:** the 86,357-byte Engine established scheduler ownership.
The first scheduler read exceeded 256 KiB and was not parsed; HEAD then reported
282,156 bytes, recovered completely under a revised 320 KiB cap. Actual imports
led to the 35,754-byte output streamer under 128 KiB. Its complete acceptance
method fit the 15,500-character output cap; the payload method exceeded the
remaining combined space, so one final query emitted that complete definition
under 16,000 characters. Six requests total, including the failed size-bound
attempt and metadata query, each with a 20-second timeout at the same pin.
No imports/exec, checkout, engine/dependency/model/data/GPU or host actions
occurred; no source files were retained. Documentation whitespace/references
were checked, not the unchanged 283-test code gate.

[sglang-output-streamer]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/scheduler_components/output_streamer.py

The token-ID checkpoint's [hosted CI][token-id-ci] succeeded at `c70f54f`.
The workspace checkpoint was still running when inspected; no hosted success
is inferred for it.

[token-id-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43848048691

Next: retain the mode-specific source contract as a qualification prerequisite,
without substituting it for numerical/cache/resource tests on a real engine.

### 2026-11-16: remaining queue gates checked against current resources

The last source gate passed **283 tests**, root lint/whitespace, scoped formatting
and diagnostics. Its [hosted workspace checkpoint][workspace-io-ci] subsequently
succeeded at `a67a0ee`; the earlier in-progress observation remains historical.
No unchanged successful tests were rerun for this reassessment.

| Queue | Verified compatible prerequisites | Remaining action and constraint |
|---|---|---|
| Held-out evaluation | Exact checkers, five deterministic synthetic families, three matched paths, uncertainty/family reporting and consistent result records; tiny CPU end-to-end controls passed. | Accept task/family/split inputs and study policy/resource scope before collecting trajectories or measuring a pretrained model. Synthetic fixtures are not accepted evaluation data. No additional concrete offline implementation defect was identified in this sweep. |
| Serving comparison | Live/exact-prefix/full-prefill CPU adapters, bounded LRU, multi-call reference logits and pending-token checks; pinned Python non-beam token-output source path traced. | Neither SGLang nor vLLM is installed. Dependency integration needs a separately authorized, compatible isolated environment; actual engine, hybrid-state, CUDA/FLA and resource measurements remain unqualified. Source evidence cannot substitute for those checks. |
| Containment/native tiers | Actual trusted namespace/native/seccomp fixtures passed; import inventory, resource deadlines and delegated-cgroup plumbing exist. | The current cgroup-v2 mount is read-only and its subtree controllers are not enabled. Real controller/PID-cap qualification needs a writable predelegated subtree, plus accepted production policies and independent evidence. Do not enable controllers, remount, or expose an untrusted service. |

**Current resource evidence:** `/opt/venv` is Python 3.12.3 with Torch
2.10.0+cu128, Transformers 5.22.1, tokenizers 0.23.3, Wasmtime 50.0.1 and
Unicorn 2.1.4. Bubblewrap, gcc and tcc are present. Free disk is 363.91 GiB,
so disk headroom is not the current blocker. No GPU/model initialization or
host modification was performed. The source/test/resource sweep identified
no further concrete useful authorized implementation step; the remaining
milestones require the protected inputs or environment above, not another
routine testing/planning approval. This is a bounded handoff, not project
completion, a user pause, or evidence that all possible bugs are absent.

[workspace-io-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43848533674

### 2026-11-16: synchronized result-return timing prerequisite

On resumed autonomous work, the [evaluation harness][return-timing-harness]
provided executor/case wall times but no separate result-return measurements.
This is a useful offline P0 prerequisite missed by the previous bounded sweep;
it does not require accepted tasks, model studies or a new serving dependency.

`CallRecord.response_seconds` now measures response framing, context admission,
cache restoration/prefill through `inject`, and the supplied synchronization
hook. Failed framing, admission or injection attempts retain their elapsed
time and explicit existing outcome. No attempted return (including isolation
failure) and legacy records use `None`, not a zero-time substitute. Successful
injection still records every response token once.

The harness synchronizes after each attempted return so asynchronous prefill
completion is charged to that stage rather than the next decode turn. This
intentionally changes timing boundaries; comparisons use the same harness and
hook. Summaries add measured response-time totals/p50/p95 and
`response_timing_samples`, making incomplete measurement coverage visible.
The primary failure-inclusive seconds-per-correct metric, checker/outcome
contracts, confidence calculations and resource limits remain unchanged.
These diagnostics neither isolate a pure GPU decode step nor establish the
P0 latency gate on a real engine.

**Bounded ledger:** four failing-first cases exposed the missing field
(0.94 s); eight new synthetic clock/CPU cases then passed (0.67 s).
They cover delayed completion, failed restore, rejected framing/context,
unmeasured-record coverage and invalid timings. Coupled evaluation/serving/
checkpoint controls: **137 passed** (9.75 s). Required root gate:
**291 passed** (14.84 s), root Ruff, two-file format and whitespace passed;
owning-source Pylance diagnostics were empty. Fourteen existing Torch warnings,
no skips. Existing jailed native and trusted containment controls ran in the
root gate; no new dependency, data/model/GPU, service or host actions occurred.
Fixture workdirs cleaned normally. Finite plan: eight new cases, one
failing-first run, one focused/coupled/root pass; no assertions or limits relaxed.

Next: reuse existing paired-delta arithmetic for the serving report, which
currently validates matching cases but exposes only independently aggregated
mode summaries.

[return-timing-harness]: ../../src/neural_compiler/eval/harness.py

### 2026-11-16: matched serving deltas, including family effects

The [serving summary][paired-serving-summary] now retains its existing
`by_mode` aggregates and adds `paired_cases`, `paired_deltas` and
`paired_deltas_by_family`. It reuses the evaluation report's paired arithmetic
for shared correctness counts, accuracy delta and wall-time mean/median delta;
no metric formula or confidence contract changed.

Comparisons use the first present mode in the existing canonical order as
baseline: live pause/resume when present, otherwise prefix-cached resubmission.
Every other present mode is compared with that baseline. Single-mode summaries
produce empty paired deltas, not comparisons with absent modes. Existing
complete-ID/mode/family/path preflight and failure inclusion remain intact.
The synthetic regression has equal aggregate accuracy/time deltas but opposing
family effects, demonstrating why independent mode aggregates are insufficient.
This does not establish matched hardware/prompts, accepted cohorts, serving
latency or statistical qualification.

**Bounded ledger:** three failing-first report cases exposed missing paired
fields (0.11 s); all three passed after reuse (0.06 s). One scoped format check
reported a wrapped test lookup, repaired with Ruff without changing behavior.
Coupled evaluation/serving controls: **115 passed** (11.22 s).
Root: **294 passed** (18.23 s), root Ruff, three-file format and whitespace
passed, both owning-source diagnostics empty. Private helper signature checks
confirmed two original callers before label generalization and all three
callers afterward. Fourteen existing Torch warnings, no skips; native jailed
and trusted containment controls ran. No model/data/GPU, engine/dependency,
service or host action occurred; fixtures cleaned normally. Three new report
cases stayed within the four-case plan; no assertions or budgets relaxed.

Next: check the return-stage deadline boundary. Formatting is charged to case
wall time, but an expired wall budget currently has no admission check between
response formatting/context validation and model injection.

[paired-serving-summary]: ../../src/neural_compiler/eval/serving.py

### 2026-11-16: reject expired result prefill before model work

The return-stage probe reproduced model injection after response formatting
had consumed the evaluation's wall allowance. The harness now reuses its
existing remaining-time guard after context validation and before `inject`.
An expired return reports the existing timeout outcome, retains its measured
response work and executor record, and injects no tokens. In-limit responses
still prefill normally. Existing explicit framing/context failures retain
their classification. No new deadline, limit or model interruptibility claim
was introduced; already-started model work remains cooperative.

**Bounded ledger:** the expired-case prefill spy failed before the repair,
while the unexpired control passed (0.69 s). The two new cases plus two
framing/context controls passed after the one-line guard (0.84 s).
Coupled evaluation/serving/checkpoint controls: **142 passed** (9.83 s).
Required root: **296 passed** (17.31 s), root Ruff, two-file format and
whitespace passed; owning-source diagnostics empty. Fourteen existing Torch
warnings, no skips; actual jailed native and trusted containment controls ran.
The two-case/one-repair plan was not expanded. Fixtures cleaned normally;
no model/data/GPU, engine/dependency, service or host action occurred.

Next: retain per-family primary cost and resource summaries alongside the
new serving deltas, reusing the existing mode aggregation rather than
introducing different family-level metric definitions.

### 2026-11-16: per-family serving costs share the aggregate contract

`summarize_serving` now includes `by_family[family][mode]`, using the same
aggregation as `by_mode` for failure-inclusive seconds per correct answer,
uncertainty, elapsed/execution/response diagnostics and resource counters.
Families with no correct answers retain explicit undefined cost; their time
and failures are not dropped. Prefill/cache counters are summed within each
group, while checkpoint storage uses the existing peak semantics.
The aggregate and paired outputs remain available unchanged.

**Bounded ledger:** one existing opposing-family case was extended; it failed
on the missing family field before the edit (0.08 s). That case plus two
mode-subset controls passed (0.13 s). Coupled evaluation/serving:
**117 passed** (11.50 s). Root: **296 passed** (19.07 s), root Ruff, two-file
format and whitespace passed; owning-source diagnostics empty. Fourteen
existing Torch warnings, no skips; jailed native/trusted containment controls
ran. No new cohort cases, accepted labels, model/data/GPU, engine/dependency,
service or host actions occurred; fixtures cleaned normally.
This reporting prerequisite is not held-out-family or serving qualification.

Next: align the harmonic unit fixture's wording with the accepted exact-rational
checker. The fixture asks for a reduced representation, while the existing
checker/tests intentionally accept mathematically equivalent fractions.

### 2026-11-16: harmonic fixture wording matches exact equivalence

The seed-17 unit fixture asked for a reduced representation of the sum from
9 through 18, but its doubled fraction `1269742/1633632` was accepted as
correct/valid by the existing exact-rational checker. The accepted checker
and regression explicitly permit equivalent fractions (`22/12` for `11/6`).
Only the unsupported word `reduced` was removed from the synthetic prompt;
the reference value, parameter ranges, seeds, answer format and scoring are
unchanged. This neither adopts task data nor changes accepted study prompts.

**Bounded ledger:** one generated unit-fixture/value probe; **18 existing**
rational/generator/end-to-end controls passed (3.80 s). Root: **296 passed**
(24.11 s), root Ruff, one-file format and whitespace passed; owning-source
diagnostics empty. Fourteen existing Torch warnings, no skips; jailed native/
trusted containment controls ran. No constant-string tests or new cohort
cases were added. No model/data/GPU, engine/dependency, service or host action
occurred; fixtures cleaned normally. Source inspection also confirmed the
existing 16 MiB file-size and disabled-core limits; no resource policy changed.

The resumed slices added return timing, paired/family serving metrics and
prefill deadline admission while preserving the accepted primary metric.
They are offline prerequisites, not a target-model latency or containment
qualification. Further measurement still needs accepted evaluation/study
inputs; real serving needs its separately authorized isolated dependencies
and runtime scope; real controller/PID-cap qualification needs a writable
predelegated subtree and accepted production policies. The earlier resource
and milestone table remains a gate inventory, not a claim that no possible
offline improvement or bug can exist.

### 2026-11-16: declare configured executors without changing native framing

The harness accepted custom executors but always declared `run(code)` as
Python. `EvaluationHarness` and `run_serving_comparison` now accept an optional
keyword-only `tool_description`. It changes only the description of a private
copy of the existing tool declaration; the function name, code parameter,
schema, result envelope and tokenizer-derived boundaries are unchanged.
Python's default prompt and the disabled no-tool baseline retain their exact
token sequences. Invalid descriptions fail before sampler/model setup.

For a separately scoped C comparison, callers can pass `executor=run_c` and
`tool_description="Compile and run C99 code, returning structured stdout or errors."`
together. The description does not select or dispatch an executor: callers
remain responsible for matching declared capabilities to the actual backend,
intrinsics and accepted study settings. No C cohort or benchmark policy was
adopted here.

**Bounded ledger:** four failing-first description cases exposed the missing
keyword (0.21 s); the existing Python prompt control passed. Seven focused
cases then passed (1.57 s). The plan grew by one callback-preflight case to
verify invalid descriptions cannot invoke sampler setup. The trusted C fixture
compiled/executed once and explicitly replayed that real result across three
CPU cache strategies, keeping the result's timing fields identical. Native
source, outcome, complete token stream and reference logits matched at the
unchanged `atol=1e-6, rtol=1e-5`; this replay is not a latency/accuracy study.
Across focused/coupled/root gates, the fixture used nine supervised invocations
(one namespace probe plus C compile/run per gate), within the twelve-invocation
plan. Existing root controls were additional required regressions.

Coupled evaluation/serving/checkpoint controls: **149 passed** (14.30 s).
Root: **303 passed** (23.57 s), root Ruff, four-file format and whitespace
passed, both owning-source diagnostics empty. Fourteen existing Torch warnings,
no skips. A static check caught the new test executor's mismatched parameter/
default contract; its signature was corrected, and all thirteen serving
callers are compatible. Formatting was repaired narrowly. The source and
fixture program stayed unchanged after the root pass, so static-only repairs
did not trigger another unchanged runtime gate. Earlier constructor-helper
flags were checked against actual defaults, runtime binding and empty source
diagnostics rather than worked around by changing existing callers.
No model/data/GPU, dependencies/downloads, service or host actions occurred;
fixtures cleaned normally.

Next: retain the already-counted prompt tokens in case/report diagnostics.
Different executor declarations change input length, but the current report
only exposes generated/injected tokens. Preserve that existing total definition
and report observed prompt size separately, with explicit missing coverage.

### 2026-11-16: retain observed prompt size separately from generated tokens

`CaseResult.prompt_tokens` now retains the valid tokenized prompt size already
used by admission checks, including prompts rejected before model prefill.
Legacy records and invalid template outputs remain explicitly unmeasured
(`None`). This is requested input size, not a claim that rejected tokens were
forwarded. Counts must be exact nonnegative integers.

Summaries add the sum of observed prompt tokens and `prompt_token_samples`;
serving mode/family summaries inherit those diagnostics. The existing
`total_tokens` definition remains generated plus injected tokens, and primary
cost/accuracy/uncertainty formulas and budgets are unchanged. The C declaration
fixture checks the exact initial session-token count rather than a text-length
proxy.

**Bounded ledger:** the existing rejected-prompt case exposed the missing field
(0.28 s) without model prefill. Five new metadata/coverage cases and two existing
controls passed (1.50 s). Coupled evaluation/serving/checkpoints:
**154 passed** (74.68 s, within the 120-second plan); no cause was inferred
for the increased elapsed time. Root: **308 passed** (38.12 s), root Ruff,
three-file format and whitespace passed; owning-source diagnostics empty.
Fourteen existing Torch warnings, no skips. Existing jailed native/trusted
controls ran; the C replay fixture used nine supervised invocations across
the three gates, not refunded from earlier scopes. Fixtures cleaned normally.
No model/data/GPU, dependencies/downloads, service or host action occurred.

The declaration checkpoint's [hosted CI][executor-description-ci] succeeded
at `d09d103`; the prior [fixture alignment CI][harmonic-alignment-ci] also
succeeded at `a9bf176`. These statuses do not certify kernel/containment coverage.

Next: validate sampled IDs before they enter live history/pending state.
Currently a callback stop can return a fractional/boolean sampler ID before
the forward-time ID guard runs, leaving a malformed pending token.

[executor-description-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43856600039
[harmonic-alignment-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43852744628

### 2026-11-16: reject malformed samples before terminal-boundary mutation

Live decoding now applies the existing exact-integer token-ID guard immediately
after sampling, before history/pending writes or the callback. Previously a
callback stop could return a fractional/boolean ID without reaching the
forward-time guard, poisoning the pending boundary. Valid prior pending work
still completes once; the repair neither rewinds recurrent state nor changes
greedy sampling, vocabulary/EOS policy or the mutable session sampler hook.

**Bounded ledger:** four tiny CPU cases (fraction/boolean, with/without valid
pending work) failed before the repair (0.13 s), then passed (0.12 s). They
verify no malformed callback/history entry, bitwise unchanged fresh-prefix
logits and valid recovery/injection against full prefill at the existing
`atol=1e-6`, `rtol=1e-5`. Coupled checkpoint/serving controls:
**60 passed** (6.65 s); root: **312 passed** (21.93 s). Root Ruff, two-file format
and whitespace passed; owning-source diagnostics empty. Fourteen existing Torch
warnings, no skips; existing jailed native/trusted controls ran and fixtures
cleaned normally. One failing-first, one focused, one coupled and one root
attempt were spent; no old ledger was refunded. No model/data/GPU, dependency,
download, service or host-security action occurred.

Next: falsify the decoder-consumer contract with bounded adapter fixtures.
The harness currently trusts callback-stop adherence, returned token IDs,
finish reasons and requested turn limits. A provider that continues after a
tool boundary can overwrite the recognized call and incorrectly score later
text. Reject provider-contract faults explicitly, rather than manufacturing
model outcomes, dropping tokens or rolling back cache state.

### 2026-11-16: enforce decoder callback/result conformance before scoring

The evaluation harness now checks the actual callback stream against the
requested turn limit and returned IDs, validates finish reasons, and rejects
continued decoding or inconsistent results after a callback stop. Previously
continued callbacks could overwrite a recognized native call; unreported or
substituted tokens and unsupported finish reasons could also evade accounting.
Provider faults raise explicitly before execution/scoring, rather than becoming
model outcomes or corrected token streams. Final synchronization runs even when
such faults escape. Valid decoder behavior and existing outcome/cost definitions
are unchanged; the interface documents the already-implemented live semantics.

**Bounded ledger:** six tiny CPU provider faults (substituted IDs, unknown finish
reason, spurious stop, ignored real stop, one-token turn overrun, omitted
callbacks) were not rejected before the repair (0.75 s). All six passed afterward
(0.89 s), with no executor invocation and synchronization after decoder exit.
Coupled evaluation/checkpoint/serving: **164 passed** (23.71 s); root:
**318 passed** (21.14 s). Root Ruff, two-file format and whitespace passed;
owning-source diagnostics empty. Fourteen existing Torch warnings, no skips;
existing jailed native/trusted controls ran. One failing-first, one focused,
one coupled and one root attempt were spent within the recorded limits.
Fixtures cleaned normally; the bounded failure-output inspection retained no
new model/data artifacts. No model/data/GPU, dependency, download, service or
host-security action occurred. These guards are not engine/kernel/cache
qualification.

The next-candidate-harness hypothesis was disproved by prior art:
`run_matched` already accepts per-path harnesses and checks common budgets.
No API change was made. Next: test whether an interrupted/invalid model forward
can mutate recurrent cache while leaving a live session reusable or
checkpointable. Preserve valid pre-forward checkpoints and fail closed on a
potentially damaged session, without automatic recurrent-state rollback.

### 2026-11-16: invalidate sessions after potentially mutating model failure

A live model invocation now marks the session unusable until valid cache/logits
are committed. Exceptions or invalid returned output preserve their original
error and block later decoding, injection and checkpoint capture. A forward can
change recurrent tensors without advancing attention length, so the previous
snapshot-length guard alone could accept damaged state. Valid pre-forward
checkpoints remain independently recoverable; restored sessions initialize
healthy status. No cache rollback, silent reset or broad exception catch was
added. Invalid-input preflight still preserves a healthy session.

**Bounded ledger:** the first four-case fixture attempt failed before exercising
production behavior because recurrent states are batch-indexed dictionaries
(0.24 s). After using the existing batch-zero convention, all four cases exposed
unsafe checkpoint acceptance (0.17 s). Exception/invalid-output cases on fresh
and restored sessions then passed (0.12 s), including unchanged attention length,
observed recurrent mutation, rejected reuse and valid prior-checkpoint recovery
at existing full-prefill tolerances. Coupled checkpoint/serving:
**64 passed** (6.07 s); root: **322 passed** (23.65 s). Root Ruff, two-file format
and whitespace passed; owning-source diagnostics empty. Fourteen existing Torch
warnings, no skips; existing jailed native/trusted controls ran. The setup
failure and revised failing-first attempt remain spent, alongside one focused,
one coupled and one root attempt. Fixtures cleaned normally. No model/data/GPU,
dependency, download, service or host-security action occurred.

One bounded CI inspection confirmed the [sample-boundary checkpoint's hosted
success][sample-boundary-ci] at `d852f63`, which also includes prompt accounting.
This is workflow status, not production containment or kernel qualification.

Next: check integer samples outside the model's logit-column domain at a
callback stop. The exact-integer guard alone still lets such a sample enter
history/pending state without reaching model input validation.

[sample-boundary-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43860590657

### 2026-11-16: admit only valid model-logit column samples

Sample admission now checks the current model-logit column domain as well as
exact integer type, before history/pending mutation or callback delivery.
Previously `-1` or the logit width could return at a callback stop without any
model input validation. Width is captured before the mutable sampler executes;
no tokenizer/config vocabulary constant, EOS setting or read-only sampler API
was introduced. Default greedy output is unchanged.

**Bounded ledger:** four new negative/upper-bound cases, each with/without valid
pending work, failed before the guard while four earlier type controls passed
(0.23 s). All eight passed afterward (0.27 s), including exact valid-prefix
accounting and recovery/full-prefill checks at unchanged tolerances. Coupled
checkpoint/serving: **68 passed** (5.79 s); root: **326 passed** (21.40 s).
Root Ruff, two-file format and whitespace passed; owning-source diagnostics
empty. Fourteen existing Torch warnings, no skips; existing jailed native/trusted
controls ran. One failing-first, one focused, one coupled and one root attempt
were spent; fixtures cleaned normally. No model/data/GPU, dependency, download,
service or host-security action occurred.

Next: test ownership of captured serving logits. The CPU
`detach().cpu()` result can alias a live sampler session's tensor; a retained
mutable session could change earlier comparison evidence after capture.
Copy diagnostic output rather than restricting the sampler interface.

### 2026-11-16: capture independent serving logit evidence

Serving comparison now captures a detached CPU copy of final logits. The
previous `detach().cpu()` capture shared storage on CPU, so a retained mutable
sampler session could alter earlier evidence, or editing evidence could alter
the session. The copy preserves shape, dtype and boundary semantics without
restricting sampler access or changing execution/cost definitions.

**Bounded ledger:** one three-mode CPU fixture reproduced the alias (0.29 s):
editing a retained session changed all 262 captured elements by one. After the
repair, bidirectional source/evidence independence passed with exact comparisons
(0.19 s), with no executor invocation. Coupled checkpoint/serving:
**69 passed** (5.35 s); root: **327 passed** (18.95 s). Root Ruff, two-file format
and whitespace passed; owning-source diagnostics empty. Fourteen existing Torch
warnings, no skips; existing jailed native/trusted controls ran. One failing-first,
one focused, one coupled and one root attempt were spent. Fixtures cleaned
normally; no model/data/GPU, dependency, download, service or host action occurred.
The GPU transfer/copy path was not qualified by these CPU fixtures.

Next: make the captured logit-prefix boundary explicit in comparison results.
`final_tokens` includes a pending stop/EOS sample, while captured logits can
still belong to the prefix before it. Preserve that convention, report the
observed prefix length separately, and leave legacy/unobserved records unknown.

### 2026-11-16: identify the captured serving logit prefix

`ServingCaseResult.logit_prefix_tokens` now identifies the observed prefix
associated with captured logits. Final history still includes pending sampled
stop/EOS IDs; no token is forwarded merely to obtain this observation. Active
native pending state, or the adapters' released callback-stop boundary,
determines the count. Legacy/unobserved results remain `None`; supplied counts
must be exact integers within final history. Cost/summary definitions and the
existing token/logit conventions are unchanged.

**Bounded ledger:** three EOS/length/sandbox-return boundary cases exposed the
missing observation (1.18 s). Those cases and three invalid-count cases passed
(0.99 s); nine captured boundaries matched full-prefill reference logits at the
existing tolerances. The cold initial-checkpoint budget control remains
unobserved rather than falsely zero. Coupled checkpoint/serving:
**75 passed** (6.16 s); root: **333 passed** (22.19 s). Root Ruff, two-file format
and whitespace passed; owning-source diagnostics empty. Fourteen existing Torch
warnings, no skips; existing jailed native/trusted controls ran. New boundary
fixtures invoked no guest executor; their sandbox status was a synthetic return,
not containment evidence. One failing-first, one focused, one coupled and one
root attempt were spent. Fixtures cleaned normally; no model/data/GPU, dependency,
download, service or host action occurred.

Next: falsify model-output shape admission before healthy cache commitment.
The owning forward currently checks missing logits but does not explicitly
enforce the existing single-sequence, nonempty causal-logit shape; malformed
batch/rank/vocabulary output can otherwise be retained as healthy state.

### 2026-11-16: validate causal logit shape before healthy cache commitment

Live forwards now require a nonempty rank-three Torch logit tensor for one
sequence before committing cache/logits as healthy. The existing missing-logit
error remains distinct; nonempty full-sequence logits are still supported by
selecting their last position. No dtype/masked-value policy or per-token finite
scan was added. Invalid output leaves the potentially mutated session unusable,
with recovery only through a valid prior checkpoint or fresh prefill.

**Bounded ledger:** four malformed-output cases failed before admission
(0.18 s): extra batch/rank and empty vocabulary were accepted, while empty token
axis raised an opaque indexing error after a real cache update. All four passed
afterward (0.11 s), including rejected reuse and unchanged-model prior-checkpoint
recovery at existing reference tolerances. Coupled checkpoint/serving:
**79 passed** (6.32 s); root: **337 passed** (18.30 s). Root Ruff, two-file format
and whitespace passed; owning-source diagnostics empty. Fourteen existing Torch
warnings, no skips; existing jailed native/trusted controls ran. One failing-first,
one focused, one coupled and one root attempt were spent. Fixtures cleaned
normally; no model/data/GPU, dependency, download, service or host action occurred.

Next: move remaining serving configuration validation ahead of sampler-factory
side effects and cold prefill. Unknown paths/store limits can currently be
silently ignored for empty tasks or rejected only after calling the factory;
a provided factory returning `None` can silently select greedy sampling.

### 2026-11-16: preflight serving settings before factory side effects

Serving comparison now validates known paths and existing store limits before
calling sampler factories, including empty task inputs. Store-limit validation
is shared with `CheckpointStore`, preserving exact-positive-integer constraints,
defaults and diagnostics. A supplied factory must return a callable; only an
absent factory selects default greedy sampling. Invalid configuration does not
become a model outcome or trigger cold prefill.

**Bounded ledger:** nine cases failed before the repair (0.39 s): three empty
inputs accepted invalid settings, three populated inputs reached a factory spy,
and three invalid factory results reached a model-forward spy. No model forward
actually ran in these probes. All nine passed afterward (0.13 s). Coupled
checkpoint/serving: **88 passed** (6.67 s); root: **346 passed** (17.21 s).
Root Ruff, three-file format and whitespace passed; both owning-source diagnostics
empty, and both shared-helper callers were signature-compatible. Fourteen
existing Torch warnings, no skips; existing jailed native/trusted controls ran.
One failing-first, one focused, one coupled and one root attempt were spent.
Fixtures cleaned normally; no model/data/GPU, dependencies, downloads, service
or host action occurred.

A single bounded CI inspection confirmed success at [causal-shape
admission][causal-shape-ci], [independent evidence capture][independent-logits-ci]
and [failed-forward handling][failed-forward-ci]. These are hosted workflow
results, not engine/kernel/production-containment qualification.

Next: distinguish a response cut by the requested generation limit from a
voluntary malformed native-call termination. The harness currently finalizes
protocol parsing before checking a length stop, which can misclassify a
host-truncated call as a protocol failure.

[causal-shape-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43869402237
[independent-logits-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43865992206
[failed-forward-ci]: https://github.com/venim1103/neural-compiler/actions/runs/43863627658

### 2026-11-16: distinguish token-limit cuts from malformed call termination

The harness checks a validated decoder length stop before finalizing native-call
parsing. A host-truncated response now reports the existing `budget_exhausted`
outcome, retaining partial text and exact generated-token accounting. Voluntary
EOS inside an incomplete call remains `protocol_error`; incomplete calls never
execute or reach the answer checker. Existing outcome/cost definitions and
token/call limits are unchanged.

**Bounded ledger:** four scripted tiny CPU cases (no-tool/Python-tool crossed
with length/EOS, at most 200 generated IDs each) reproduced two length
misclassifications; two EOS controls passed (0.74 s). All four passed after the
repair (0.68 s), with exact generated counts and zero calls/injected IDs.
Coupled evaluation/serving: **155 passed** (11.94 s); root: **350 passed**
(16.35 s). Root Ruff, two-file format and whitespace passed; owning-source
diagnostics empty. Fourteen existing Torch warnings, no skips; existing jailed
native/trusted controls ran. One failing-first, one focused, one coupled and
one root attempt spent; fixtures cleaned normally. No guest ran in the new
cases, and no pretrained-model/data/GPU study, dependency, download, service
or host change occurred. The earlier preflight-output temporary file was removed.

Next: enforce the complementary decoder-provider contract. A claimed length
stop must actually consume the requested turn allowance; short or empty
provider output is not evidence that the host generation budget was exhausted.

### 2026-11-16: reject premature decoder length claims

A decoder reporting `length` must return the requested turn's full token count.
Short/empty length claims now escape as provider errors before decoding/scoring
or budget-outcome attribution; final synchronization still runs. Existing
callback/returned-ID agreement checks and genuine length/EOS/stop behavior remain
unchanged. No metric definitions or token allowances changed.

**Bounded ledger:** two added malformed-provider controls failed before the
repair (0.16 s), then passed (0.07 s). Each uses at most one sampled ID with
ordinary tiny CPU prefill and no guest execution. Scoped Ruff passed; the first
format check found the added import needed wrapping. One formatter application
and one style-only recheck repaired that failure without repeating the successful
focused test. Coupled evaluation/serving: **157 passed** (10.75 s); root:
**352 passed** (16.75 s). Root Ruff, two-file format and whitespace passed;
owning-source diagnostics empty. Fourteen existing Torch warnings, no skips;
existing jailed native/trusted controls ran. One failing-first, one focused,
one coupled and one root attempt spent, plus the failed/repaired style checks.
Fixtures cleaned normally; no pretrained-model/data/GPU study, dependency,
download, service or host change occurred.

Next: align chat-template admission with exact integer token-ID validation
before invoking the session factory. The harness's current `isinstance` guard
accepts booleans even though native sessions reject them as invalid token IDs.

### 2026-11-16: reject invalid template IDs before factory admission

Chat-template admission now requires exact Python integer IDs, matching native
session validation. Boolean IDs are rejected as a template protocol error before
the session factory or prompt observation is committed. Existing malformed
non-list/fractional controls and valid tokenizer prompts retain their behavior.

**Bounded ledger:** three synthetic inputs exposed one boolean-factory admission
failure; two malformed controls passed (0.13 s). All three passed afterward
(0.04 s), with invalid prompts unobserved. Factory spies prevented model/guest
execution. Scoped Ruff passed; one initial format failure needed a wrapped test
call, repaired by one formatter application/style-only recheck without repeating
the successful focus. Coupled evaluation/serving: **159 passed** (10.82 s);
root: **354 passed** (17.98 s). Root Ruff, two-file format and whitespace passed;
owning-source diagnostics empty. Fourteen existing Torch warnings, no skips;
existing jailed native/trusted controls ran. One failing-first, focused, coupled
and root attempt spent, plus failed/repaired style checks; fixtures cleaned.
No pretrained-model/data/GPU study, dependencies, downloads, service or host
changes occurred.

Next: complete sampler-factory input preflight. Noncallable factory results now
fail explicitly, but a noncallable factory itself remains silently accepted
for empty task lists.

### 2026-11-16: complete sampler-factory input preflight

Serving comparison now rejects a supplied noncallable sampler factory before
task iteration, retaining the callable-result check and `None`-only greedy
default. Empty task lists no longer hide invalid factory configuration.

**Bounded ledger:** four empty/nonempty-by-boolean/integer cases failed before
the repair (0.15 s): empty tasks accepted the inputs, populated tasks raised
an indirect call-site `TypeError`. After explicit admission, all four passed
(0.09 s), with forward spies preventing model/guest execution. One formatter
application wrapped the new fixture before scoped style checks. Coupled
checkpoint/serving: **92 passed** (5.16 s); root: **358 passed** (16.60 s).
Root Ruff, two-file format and whitespace passed; owning-source diagnostics
empty. Fourteen existing Torch warnings, no skips; existing jailed native/trusted
controls ran. One failing-first, focused, coupled and root attempt spent;
fixtures cleaned normally. No pretrained-model/data/GPU study, dependency,
download, service or host change occurred.

Next: reassess the held-out-family, serving and containment queues against
owning code/tests and current resources, selecting another compatible offline
prerequisite rather than treating a green checkpoint as completion.

### 2026-11-16: reject empty templates and reassess remaining gates

Template admission also rejects empty ID lists before committing a prompt
observation or calling the session factory. This matches native prefill's
nonempty-sequence contract, retaining the existing protocol-error diagnostic.

**Bounded ledger:** the added empty-list case reached the factory spy; three
existing malformed controls passed (0.13 s). All four passed after repair
(0.04 s), with no model/guest execution. One scoped formatter application
wrapped the admission guard. Coupled evaluation/serving: **164 passed**
(11.92 s); root: **359 passed** (16.74 s). Root Ruff, two-file format and
whitespace passed; owning-source diagnostics empty. Fourteen existing Torch
warnings, no skips; existing jailed native/trusted controls ran. One failing-first,
focused, coupled and root attempt spent; fixtures cleaned normally. No
pretrained-model/data/GPU study, dependency, download, service or host change.

**Queue reassessment:** bounded reads of family generators, exact checkers and
reporting, three serving adapters and their multi-call controls, and jailed
runtime/cgroup/policy code and controls confirmed the compatible offline
prerequisites remain implemented. No additional concrete source repair was
identified after empty-template admission. Next milestone actions need:

- **Held-out evaluation:** accepted family/cohort/split inputs and a fresh exact
  pretrained-run/trajectory scope. Deterministic synthetic generators and matched
  uncertainty/paired reporting exist; those fixtures are not accepted task data.
- **Serving:** separately scoped engine dependencies/runtime and kernel/model
  qualification. Current `/opt/venv` is Python 3.12.3; neither SGLang nor vLLM
  is available. FLA is present, but CPU-reference fixtures do not qualify its
  accelerator kernels. CPU snapshot/LRU/three-strategy adapters already exist.
- **Containment/multi-tier:** writable predelegated cgroup-v2 controllers and
  accepted production syscall/import policies for qualification. Current
  `/sys/fs/cgroup` is read-only, advertises `cpu memory pids`, and has empty
  subtree control. Five jailed tier dispatchers and trusted/synthetic controls
  exist, without production qualification or an unjailed fallback.

Disk inspection found 362.85 GiB free, not download authority. No privileges,
controllers, credentials, dependencies or services were changed. The short
reading map above exposes decisions versus implementation evidence without
rewriting historical reviews. Latest actual user intent remains autonomous
continuation, not the historical WIP pause; protected prerequisites require a
new directive rather than fabricated data, qualifications or routine reruns.

### 2026-11-17: reject nonfinite greedy selections before token commitment

The default greedy selector now checks the maximum returned by its reduction
before committing a sampled token. NaN, positive infinity and entirely masked
negative-infinity rows raise an explicit runtime error rather than becoming
ordinary generated IDs. Negative-infinity masks with a finite winner and
first-index tie handling remain supported. Explicit sampler hooks are unchanged;
this is not a new whole-output finite scan or a model-accuracy claim.

**Bounded ledger:** six nonfinite-row/pending-boundary cases failed before
repair; one masked finite-tie control passed (0.24 s). All seven passed afterward
(0.16 s), including no callback/history mutation, exactly-once completion of
previously pending work and subsequent valid injection/reference logits at
existing tolerances. Coupled checkpoint/serving: **99 passed** (6.12 s);
root: **366 passed** (20.63 s). Root Ruff, two-file format, whitespace and
owning-source diagnostics passed. Fourteen existing Torch warnings, no skips;
existing jailed native/trusted controls ran. One failing-first, focused, coupled
and root attempt spent; fixtures cleaned normally. No guest ran in the new
fixtures, and no pretrained-model/data/GPU study, dependency, download, service
or host change occurred. GPU synchronization/performance was not measured.

### 2026-11-17: keep trusted exec failures out of guest error outcomes

The jail resource launcher previously acknowledged readiness and closed its
private reporter before `exec`; a missing executable or invalid executable
format was therefore mislabeled `runtime_error` (and could become a C
`compile_error`). It now retains its non-inheritable duplicate until `exec`,
which closes it atomically on success. A caught exec `OSError` sends one bounded
private followup diagnostic and exits nonzero; the supervisor returns
`sandbox_error`. Guest stdout/stderr and exit codes cannot supply this signal.
This changes failure attribution, not namespace/limit/policy requirements.

**Bounded ledger:** two real trusted jailed missing/invalid-format fixtures
reproduced the misclassification (0.18 s). Five focused checks passed afterward
(0.35 s), including old/new stderr non-forgery strings and descriptor closure.
Native/execution/protocol/seccomp/cgroup/red-team regression: **141 passed**
(5.05 s); initial root: **369 passed** (18.75 s). A concrete closure question
then strengthened the existing fixture: inspect every guest `/proc/self/fd`
entry, not just original stdin's descriptor, to exclude the launcher's duplicated
socket. That focus passed (0.10 s); required root with the changed assertion:
**369 passed** (18.67 s). Root Ruff, two-file format, whitespace and owning-source
diagnostics passed. One formatter application; fourteen existing Torch warnings,
no skips. Actual jailed Python/C/assembly/Wasm/Unicorn and trusted seccomp controls
ran. One failing-first, repaired focus, coupled and initial root attempt spent,
plus the narrowly revised descriptor focus/root; no unchanged successful coupled
rerun. Temporary fixtures cleaned normally; no studies/downloads/dependencies,
untrusted service, privileges or host-policy changes. This is not production
containment qualification.

The suspected failed-handoff prefill undercount was disproved by owning-code
inspection: `resume_prefix` does not capture checkpoints, later capture failures
occur after counters are committed, and the cold constructor failure already
retains its prompt for accounting. No speculative patch or redundant test was
added. One bounded hosted inspection confirmed success at [greedy
admission][greedy-admission-ci], `bd292bf` and `93623b9`; this does not assert a
hosted result for the subsequent exec-reporting change.

[greedy-admission-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44034930751

### 2026-11-17: pin offline tokenizer evidence and recheck next milestones

The cached-tokenizer two-call integration now requests the reviewed
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` revision explicitly instead of mutable
cached `main`. It remains local-only and skips if that exact artifact is absent.
Six local cache-entry probes found both references currently resolve to that
same snapshot (tokenizer JSON 12,807,982 bytes, config 16,710, template 7,756);
this is a reproducibility repair, not evidence that prior framing was wrong.
Historical experiment scripts/results were not rerun or rewritten.

**Bounded ledger:** the existing exact-template two-call integration passed
(0.77 s); root: **369 passed** (23.51 s). Root Ruff, one-file format, whitespace
and test-file diagnostics passed. Fourteen existing Torch warnings, no skips.
One six-entry read-only cache probe, one focused and one root attempt spent;
no separate constant-checking test or unchanged coupled rerun. No model weights,
downloads, new dependencies, GPU/pretrained/data studies or host changes.
One subsequent hosted inspection found [exec-reporting CI][exec-report-ci]
still in progress; the completed greedy-admission CI remained green.

The remaining queue was checked against owning code, nearby regressions and
current resources, not just todo status. Synthetic family generation, exact
checkers and matched uncertainty reporting are implemented; advancing accepted
evaluation needs accepted family/cohort/split inputs and fresh exact run scope.
CPU snapshots/LRU/three-strategy comparisons are implemented; real serving needs
separately scoped engine dependencies/runtime and qualification (neither SGLang
nor vLLM is installed). Jailed tier/policy/cgroup prerequisites and trusted
controls are implemented; real containment needs accepted production policies
and writable predelegation. The current cgroup-v2 mount is still read-only,
advertises `cpu memory pids`, and has empty subtree control. Free disk is
362.24 GiB, not download authority. No further concrete compatible source repair
was identified; additional synthetic work would not satisfy those milestones.

[exec-report-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44035539763

### 2026-11-17: reject reused task labels with mismatched observed text

The [harness][task-text-harness] now records optional `task_text_sha256`, derived
from an ASCII canonical JSON pair of exact user-task prompt and answer-format
strings. Model/tool wrappers deliberately differ by path and are not hashed.
Both matched evaluation and [serving reports][task-text-serving] reject differing
observed text under reused case/family labels, including partially observed
groups. Legacy/missing observations remain `None`, with `task_text_samples` and
`task_text_complete_pairs` exposing coverage rather than imputing a match.
The field is appended with a default, preserving prior positional records.
Hashes do not certify checker/reference identity, complete model prompts,
budgets, hardware, cohort acceptance or statistical qualification. Existing
accuracy, cost, uncertainty and outcome definitions are unchanged.

**Bounded ledger:** four report-type-by-text-field fixtures reproduced false
matched acceptance (0.69 s), using twelve tiny CPU prefills and fewer than 200
scripted generated IDs per run. All four passed after repair (0.56 s).
Malformed-digest, legacy/partial coverage and genuine matched-run controls
passed together: **12 passed** (0.84 s). Coupled evaluation/serving:
**175 passed** (13.24 s); root: **380 passed** (20.00 s). Root Ruff, three-file
format and whitespace passed; both owning-source diagnostics empty and both
shared-helper callers signature-compatible. Test diagnostics contained only
unused-parameter hints, not type errors. Fourteen existing Torch warnings, no
skips; existing jailed native/trusted controls ran. One failing-first, repaired
focus, expanded edge focus, coupled and root attempt spent; one formatter
application, temporary fixtures cleaned. No guest ran in the new fixtures, and
no study/data acceptance, model weights, downloads, dependencies or host changes.

A bounded hosted inspection confirmed both `ea2a0a4` (exec reporting) and
`46f7d7c` (tokenizer pin) CI completed successfully. Next compatible prerequisite:
retain configured evaluation budgets alongside case evidence so independently
collected records cannot silently pair different resource allowances.

[task-text-harness]: ../../src/neural_compiler/eval/harness.py
[task-text-serving]: ../../src/neural_compiler/eval/serving.py

### 2026-11-17: preserve configured budgets in independently paired evidence

`CaseResult.budgets` now retains the immutable configured `Budgets` observation,
including prompt-admission failures. Both report consumers use shared paired
provenance validation to reject differing observed allowances within a case,
even when another observation is absent. `budget_samples` and
`budget_complete_pairs` expose coverage; legacy `None` is not replaced with
current defaults. Records retain positional compatibility. Per-case matching
does not impose a new cross-case budget policy, change metric definitions, or
certify actual enforcement, complete execution policies, references or hardware.
The existing pre-inference `run_matched` budget check remains unchanged.

**Bounded ledger:** two independently collected report fixtures accepted
different serial-call allowances before repair (0.56 s; six tiny CPU prefills).
Both rejected the mismatch afterward (0.36 s). Expanded provenance,
legacy/partial, invalid-observation and prefill-admission controls:
**18 passed** (0.96 s). Coupled evaluation/serving: **183 passed** (16.07 s);
root: **388 passed** (22.38 s). Root Ruff, three-file format and whitespace
passed; both owning-source diagnostics empty and both shared-helper callers
signature-compatible. Fourteen existing Torch warnings, no skips; existing
jailed native/trusted controls ran. One failing-first, repaired focus, expanded
edge focus, coupled and root attempt spent; one formatter application,
temporary fixtures cleaned. New fixtures used no guest executor. No model/data
studies or acceptance, dependencies, downloads, services or host changes occurred.

These are additional offline reporting prerequisites, not acceptance of a
held-out cohort or real-engine measurements. Next milestone actions still need
accepted evaluation inputs/exact run scope, separately scoped serving-engine
dependencies/runtime qualification, or writable cgroup predelegation and accepted
production policies. Further provenance fields alone would not satisfy those
gates.
