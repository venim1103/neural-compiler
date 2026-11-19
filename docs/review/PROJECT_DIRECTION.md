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

## Archived implementation updates (2026-11-12 through 2026-11-16)

Whole dated updates, including failures, approvals, spent ledgers and evidence
limits, are preserved verbatim in the [historical archive][direction-history-map].
They are history, not renewed experiment authority. Existing deep links land
on the corresponding archive entry below.

- <a id="2026-11-12-bounded-offline-evaluation-prerequisites"></a>[2026-11-12: bounded offline evaluation prerequisites][direction-history-001]
- <a id="2026-11-12-exact-boundary-cpu-hybrid-checkpoints"></a>[2026-11-12: exact-boundary CPU hybrid checkpoints][direction-history-002]
- <a id="2026-11-12-paired-casefamily-reporting"></a>[2026-11-12: paired case/family reporting][direction-history-003]
- <a id="2026-11-12-opt-in-syscall-filter-plumbing"></a>[2026-11-12: opt-in syscall-filter plumbing][direction-history-004]
- <a id="2026-11-12-source-only-import-policy-inventory"></a>[2026-11-12: source-only import-policy inventory][direction-history-005]
- <a id="2026-11-12-opt-in-delegated-cgroup-requests"></a>[2026-11-12: opt-in delegated cgroup requests][direction-history-006]
- <a id="2026-11-12-evaluation-correctness-beyond-qualification-blockers"></a>[2026-11-12: evaluation correctness beyond qualification blockers][direction-history-007]
- <a id="2026-11-12-serving-dependency-compatibility-prerequisite"></a>[2026-11-12: serving dependency compatibility prerequisite][direction-history-008]
- <a id="2026-11-12-repeated-stopping-diagnosis-and-continuation-safeguards"></a>[2026-11-12: repeated stopping diagnosis and continuation safeguards][direction-history-009]
- <a id="2026-11-12-actual-continuation-and-complete-model-mode-guards"></a>[2026-11-12: actual continuation and complete model-mode guards][direction-history-010]
- <a id="2026-11-12-checkpoint-identity-bound-to-the-initial-prefix"></a>[2026-11-12: checkpoint identity bound to the initial prefix][direction-history-011]
- <a id="2026-11-12-user-requested-wip-checkpoint-and-pause"></a>[2026-11-12: user-requested WIP checkpoint and pause][direction-history-012]
- <a id="2026-11-12-ci-repair-and-multi-family-offline-evaluation-prerequisites"></a>[2026-11-12: CI repair and multi-family offline evaluation prerequisites][direction-history-013]
- <a id="2026-11-12-bounded-checkpoint-store-and-offline-p0-serving-comparison-harness"></a>[2026-11-12: bounded checkpoint store and offline P0 serving comparison harness][direction-history-014]
- <a id="2026-11-12-in-jail-c99assembly-bridge-module-and-candidate-p1-policy-baselines"></a>[2026-11-12: in-jail C99/assembly bridge module and candidate P1 policy baselines][direction-history-015]
- <a id="2026-11-12-unified-jailed-wasmx86-bridge-wrappers-and-cli-policynative-entry-points"></a>[2026-11-12: unified jailed Wasm/x86 bridge wrappers and CLI policy/native entry points][direction-history-016]
- <a id="2026-11-12-uncertainty-intervals-paired-deltas-jailed-tcc-fast-path-and-import-pre-filter"></a>[2026-11-12: uncertainty intervals, paired deltas, jailed tcc fast path, and import pre-filter][direction-history-017]
- <a id="2026-11-12-unified-bridgepolicy-dispatcher-and-red-team-contract-suite"></a>[2026-11-12: unified BridgePolicy dispatcher and red-team contract suite][direction-history-018]
- <a id="2026-11-13-review-of-intervening-implementation-and-contract-repairs"></a>[2026-11-13: review of intervening implementation and contract repairs][direction-history-019]
- <a id="2026-11-13-failed-turn-boundaries-in-full-re-prefill-comparison"></a>[2026-11-13: failed-turn boundaries in full re-prefill comparison][direction-history-020]
- <a id="2026-11-13-impossible-evaluation-records-fail-before-aggregation"></a>[2026-11-13: impossible evaluation records fail before aggregation][direction-history-021]
- <a id="2026-11-13-stateless-model-identity-and-deterministic-ci-classification"></a>[2026-11-13: stateless model identity and deterministic CI classification][direction-history-022]
- <a id="2026-11-13-primary-sglang-compatibility-prerequisite-recovered"></a>[2026-11-13: primary SGLang compatibility prerequisite recovered][direction-history-023]
- <a id="2026-11-13-standalone-native-framing-budgets-reject-invalid-caps"></a>[2026-11-13: standalone native framing budgets reject invalid caps][direction-history-024]
- <a id="2026-11-13-cpu-continuation-identity-includes-generation-eos-settings"></a>[2026-11-13: CPU continuation identity includes generation EOS settings][direction-history-025]
- <a id="2026-11-13-jail-setup-consumes-the-execution-allowance"></a>[2026-11-13: jail setup consumes the execution allowance][direction-history-026]
- <a id="2026-11-13-checkpoint-capacity-failures-remain-matched-failed-cases"></a>[2026-11-13: checkpoint capacity failures remain matched failed cases][direction-history-027]
- <a id="2026-11-13-invalid-final-answer-formats-cannot-count-as-correct"></a>[2026-11-13: invalid final-answer formats cannot count as correct][direction-history-028]
- <a id="2026-11-13-target-checkpoint-storage-lower-bounds-metadata-only"></a>[2026-11-13: target checkpoint storage lower bounds, metadata only][direction-history-029]
- <a id="2026-11-13-explicit-adapter-snapshot-overrides-reach-the-comparison-helper"></a>[2026-11-13: explicit adapter snapshot overrides reach the comparison helper][direction-history-030]
- <a id="2026-11-13-target-loader-metadata-supports-the-existing-causal-interface"></a>[2026-11-13: target loader metadata supports the existing causal interface][direction-history-031]
- <a id="2026-11-13-pinned-sglang-api-and-checkpoint-source-evidence"></a>[2026-11-13: pinned SGLang API and checkpoint-source evidence][direction-history-032]
- <a id="2026-11-13-jail-startup-status-is-independent-of-guest-stderr"></a>[2026-11-13: jail startup status is independent of guest stderr][direction-history-033]
- <a id="2026-11-13-matched-synthetic-suites-validate-seeds-before-arithmetic"></a>[2026-11-13: matched synthetic suites validate seeds before arithmetic][direction-history-034]
- <a id="2026-11-13-checker-result-flags-must-satisfy-their-boolean-contract"></a>[2026-11-13: checker-result flags must satisfy their boolean contract][direction-history-035]
- <a id="2026-11-13-paired-comparison-effects-remain-visible-within-each-family"></a>[2026-11-13: paired comparison effects remain visible within each family][direction-history-036]
- <a id="2026-11-13-pinned-engine-sessions-are-an-integration-candidate-not-a-cache-proof"></a>[2026-11-13: pinned engine sessions are an integration candidate, not a cache proof][direction-history-037]
- <a id="2026-11-13-append-only-session-record-ownership-traced-at-the-pinned-source"></a>[2026-11-13: append-only session record ownership traced at the pinned source][direction-history-038]
- <a id="2026-11-13-reject-decode-budgets-before-touching-a-continuation-boundary"></a>[2026-11-13: reject decode budgets before touching a continuation boundary][direction-history-039]
- <a id="2026-11-13-snapshot-overrides-are-admitted-before-prefix-cache-work"></a>[2026-11-13: snapshot overrides are admitted before prefix-cache work][direction-history-040]
- <a id="2026-11-13-native-preparation-shares-the-execution-wall-allowance"></a>[2026-11-13: native preparation shares the execution wall allowance][direction-history-041]
- <a id="2026-11-16-python-importsource-preparation-consumes-the-same-allowance"></a>[2026-11-16: Python import/source preparation consumes the same allowance][direction-history-042]
- <a id="2026-11-16-cli-source-ingestion-stops-at-the-byte-budget-sentinel"></a>[2026-11-16: CLI source ingestion stops at the byte-budget sentinel][direction-history-043]
- <a id="2026-11-16-pinned-internal-stop-token-retention-is-source-confirmed"></a>[2026-11-16: pinned internal stop-token retention is source-confirmed][direction-history-044]
- <a id="2026-11-16-pinned-detokenization-separates-text-trimming-from-token-output"></a>[2026-11-16: pinned detokenization separates text trimming from token output][direction-history-045]
- <a id="2026-11-16-no-tool-records-cannot-include-execution-or-injected-tokens"></a>[2026-11-16: no-tool records cannot include execution or injected tokens][direction-history-046]
- <a id="2026-11-16-injected-totals-reconcile-with-per-call-response-tokens"></a>[2026-11-16: injected totals reconcile with per-call response tokens][direction-history-047]
- <a id="2026-11-16-execution-result-and-call-labels-use-the-closed-status-contract"></a>[2026-11-16: execution result and call labels use the closed status contract][direction-history-048]
- <a id="2026-11-16-pinned-response-assembly-distinguishes-delta-and-cumulative-ids"></a>[2026-11-16: pinned response assembly distinguishes delta and cumulative IDs][direction-history-049]
- <a id="2026-11-16-successful-execution-results-cannot-contain-failed-exits"></a>[2026-11-16: successful execution results cannot contain failed exits][direction-history-050]
- <a id="2026-11-16-pinned-pause-controls-are-global-server-controls"></a>[2026-11-16: pinned pause controls are global server controls][direction-history-051]
- <a id="2026-11-16-token-ids-are-validated-before-model-or-prefix-cache-work"></a>[2026-11-16: token IDs are validated before model or prefix-cache work][direction-history-052]
- <a id="2026-11-16-outer-workspace-io-failures-remain-explicit-execution-results"></a>[2026-11-16: outer workspace I/O failures remain explicit execution results][direction-history-053]
- <a id="2026-11-16-pinned-scheduler-emits-through-stop-token-deltas"></a>[2026-11-16: pinned scheduler emits through-stop token deltas][direction-history-054]
- <a id="2026-11-16-remaining-queue-gates-checked-against-current-resources"></a>[2026-11-16: remaining queue gates checked against current resources][direction-history-055]
- <a id="2026-11-16-synchronized-result-return-timing-prerequisite"></a>[2026-11-16: synchronized result-return timing prerequisite][direction-history-056]
- <a id="2026-11-16-matched-serving-deltas-including-family-effects"></a>[2026-11-16: matched serving deltas, including family effects][direction-history-057]
- <a id="2026-11-16-reject-expired-result-prefill-before-model-work"></a>[2026-11-16: reject expired result prefill before model work][direction-history-058]
- <a id="2026-11-16-per-family-serving-costs-share-the-aggregate-contract"></a>[2026-11-16: per-family serving costs share the aggregate contract][direction-history-059]
- <a id="2026-11-16-harmonic-fixture-wording-matches-exact-equivalence"></a>[2026-11-16: harmonic fixture wording matches exact equivalence][direction-history-060]
- <a id="2026-11-16-declare-configured-executors-without-changing-native-framing"></a>[2026-11-16: declare configured executors without changing native framing][direction-history-061]
- <a id="2026-11-16-retain-observed-prompt-size-separately-from-generated-tokens"></a>[2026-11-16: retain observed prompt size separately from generated tokens][direction-history-062]
- <a id="2026-11-16-reject-malformed-samples-before-terminal-boundary-mutation"></a>[2026-11-16: reject malformed samples before terminal-boundary mutation][direction-history-063]
- <a id="2026-11-16-enforce-decoder-callbackresult-conformance-before-scoring"></a>[2026-11-16: enforce decoder callback/result conformance before scoring][direction-history-064]
- <a id="2026-11-16-invalidate-sessions-after-potentially-mutating-model-failure"></a>[2026-11-16: invalidate sessions after potentially mutating model failure][direction-history-065]
- <a id="2026-11-16-admit-only-valid-model-logit-column-samples"></a>[2026-11-16: admit only valid model-logit column samples][direction-history-066]
- <a id="2026-11-16-capture-independent-serving-logit-evidence"></a>[2026-11-16: capture independent serving logit evidence][direction-history-067]
- <a id="2026-11-16-identify-the-captured-serving-logit-prefix"></a>[2026-11-16: identify the captured serving logit prefix][direction-history-068]
- <a id="2026-11-16-validate-causal-logit-shape-before-healthy-cache-commitment"></a>[2026-11-16: validate causal logit shape before healthy cache commitment][direction-history-069]
- <a id="2026-11-16-preflight-serving-settings-before-factory-side-effects"></a>[2026-11-16: preflight serving settings before factory side effects][direction-history-070]
- <a id="2026-11-16-distinguish-token-limit-cuts-from-malformed-call-termination"></a>[2026-11-16: distinguish token-limit cuts from malformed call termination][direction-history-071]
- <a id="2026-11-16-reject-premature-decoder-length-claims"></a>[2026-11-16: reject premature decoder length claims][direction-history-072]
- <a id="2026-11-16-reject-invalid-template-ids-before-factory-admission"></a>[2026-11-16: reject invalid template IDs before factory admission][direction-history-073]
- <a id="2026-11-16-complete-sampler-factory-input-preflight"></a>[2026-11-16: complete sampler-factory input preflight][direction-history-074]
- <a id="2026-11-16-reject-empty-templates-and-reassess-remaining-gates"></a>[2026-11-16: reject empty templates and reassess remaining gates][direction-history-075]

## Recent implementation updates (2026-11-17 onward)

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

### 2026-11-17: persist validated offline case evidence

The user explicitly delegated policy/approach selection and fully autonomous
continuation. Routine policy consent is no longer a missing-input gate; this
does not turn consent into factual containment qualification. The selected next
milestone is a usable offline evidence-save/report-replay workflow, without
model execution, new dependencies or shared-host changes.

[Case evidence I/O][case-records] exports `dump_case_results(results, stream)`
and `load_case_results(stream)` from the evaluation package. Schema version 1
contains a `cases` array of complete nested result records. It preserves call
failures, response timing, optional prompt/task-text/budget observations and
legacy unknowns. The reader rejects duplicate/unknown fields, missing required
fields, unsupported versions, boolean numeric values, nonfinite numbers and
partial budget objects. Existing record contracts validate counts, times and
statuses; paired-report coverage/provenance checks remain with report consumers.
The writer validates before writing strict ASCII-escaped JSON and reports short
writes explicitly. Streams remain caller-owned; this is not an atomic-file or
untrusted-service interface, accepted dataset, authenticated evidence or study.

**Finite scope and results:** at most 18 new synthetic fixtures, at most six
records each; no model/guest execution in those fixtures. All **18 passed**
(0.07 s), including failure-inclusive deterministic report equality, Unicode
round-trip, malformed input, unknown optional observations, pre-write rejection
and short-write reporting. Coupled evidence/evaluation/serving: **201 passed**
(12.00 s); root: **406 passed** (16.90 s). Root Ruff, three-file format and
whitespace checks passed; both owning-source diagnostics were empty and all
eight dump/load callers were signature-compatible. Fourteen existing Torch
warnings, no skips; existing jailed/native controls ran in the root gate.
One focus, coupled and root invocation spent, with one formatter application,
no failures or aborts. Temporary files were managed by test fixtures; no study
artifacts, downloads or dependencies were added. Next concrete action: expose
offline report replay in the existing CLI with explicit input/coverage failures.

[case-records]: ../../src/neural_compiler/eval/records.py

### 2026-11-17: replay matched reports through the public CLI

`neural-compiler --eval-report FILE` now reads saved case evidence and prints the
existing failure-inclusive matched report as strict JSON. It is mutually
exclusive with execution/import/version actions and performs no model or guest
execution. File/schema errors and incomplete or mismatched paired coverage
remain explicit exceptions, consistent with existing CLI file handling; no
partial report is printed. Exit zero means successful report generation, not
accuracy acceptance. The public README documents saving existing harness rows
and replaying them; the 64 KiB program-source limit is not applied to evidence.

**Finite scope and results:** three new synthetic CLI fixtures, at most three
records each, with executor and live-session-constructor traps. First focus:
one failed/two passed (0.08 s), because the test imported the session protocol
from the concrete-session module. Inspection identified `LiveSession.__init__`
as the actual model-prefill entry. Second focus: one failed/two passed (0.06 s),
because the test compared decoded JSON arrays against Python CI tuples.
Normalizing the expected report to JSON wire types repaired that assertion;
no CLI source repair was needed. A narrowly revised routine plan allowed one
additional focused attempt after that concrete fixture correction: **3 passed**
(0.07 s). No unchanged failed rerun or protected experiment allowance was used.

Required root gate: **409 passed** (27.50 s), fourteen existing Torch warnings,
no skips; existing jailed/native controls ran. Root Ruff, two-file formatting,
whitespace and new README link targets passed. Owning-source diagnostics were
empty, all twelve `main` callers were signature-compatible; test diagnostics
were only unused-parameter hints in the execution trap. Three focus invocations,
one root invocation and two formatter applications spent; no aborts. Temporary
evidence files were fixture-managed. No models, accepted task data, dependencies,
downloads, services or host changes were introduced.

### 2026-11-17: reject invalid observed serving work

The serving record admitted negative, boolean and fractional prefill/cache
counters into reports, unlike the base case-record integer contract.
`ServingCaseResult` now rejects those values at construction for `prefill_tokens`,
`checkpoint_bytes_peak`, `cache_hits`, `cache_misses` and `checkpoints_evicted`.
Zero and positive integer observations retain their existing meaning. Aggregate
formulas, cache limits, logit-prefix semantics and serving behavior are unchanged;
this is input validation, not new containment or engine qualification.

**Finite scope and results:** fifteen synthetic record-only rejection fixtures,
one record each (five fields by three invalid values), all reproduced acceptance
before repair: **15 failed** (0.24 s). All **15 passed** afterward (0.04 s).
Coupled serving/evaluation/evidence: **216 passed** (16.37 s); root:
**424 passed** (25.62 s), fourteen existing Torch warnings, no skips. Root Ruff,
two-file format and whitespace checks passed; owning-source diagnostics empty.
One failing-first, repaired focus, coupled and root invocation spent; no aborts,
formatter changes or retained artifacts. Existing trusted jailed/native controls
ran in the root gate; the new fixtures used no model or guest execution.

### 2026-11-17: persist exact task specifications without executable checkers

The evidence module now also exports `dump_tasks(tasks, stream)` and
`load_tasks(stream)`. Its separate schema-v1 `tasks` document retains task
identity, family, exact prompt, answer-format instruction and a closed checker
specification (`integer`, `rational` or `integer_sequence`). Only exact built-in
checker types can be saved; opaque callables/subclasses are rejected, not
pickled, imported or evaluated. Loading constructs known checker objects without
calling them. Duplicate IDs fail before writing and on reading.

Reference integers are canonical signed decimal strings; rational references
have numerator/positive-denominator strings, sequence references are arrays of
integer strings. Chunked conversion reuses the checker's existing exact parser
and never changes interpreter-global digit limits. Shared strict envelope/stream
helpers preserve the existing schema-v1 `cases` document and short-write errors.
The README documents these APIs and ignored `.runs/` storage for private task,
reference and result files. This is specification I/O, not accepted labels,
cohort/split adoption, trajectory collection or authenticated reference evidence.

**Finite scope and results:** twelve new synthetic fixtures, at most two tasks
each and 5,001 reference digits. Integer, rational and sequence round-trips,
opaque-checker non-execution, duplicate IDs and invalid specifications passed.
Full record focus, including unchanged case-schema controls: **30 passed**
(0.09 s); coupled record/evaluation: **162 passed** (10.88 s); root:
**436 passed** (21.98 s), fourteen existing Torch warnings, no skips. Root Ruff,
three-file format, whitespace and README local-link targets passed. Both owning
sources had empty diagnostics; five new API callers were signature-compatible;
test diagnostics contained only an unused checker-trap parameter hint. One focus,
coupled and root invocation and one formatter application spent, no failures or
aborts. Temporary streams/files remained fixture-owned; no accepted data, model
runs, downloads, dependencies, services or host changes.

### 2026-11-17: recheck saved outcomes against exact references

`validate_references(tasks, results)` now checks unique/exact task-ID coverage,
family identity, observed task-text digests and agreement of final-answer
classifications with built-in exact checkers. It rejects opaque checker types
without calling them. The harness and validator share the unchanged canonical
task-text hash and validity-first outcome rules. Reference checks never rewrite
records, apply an accuracy threshold or require identical references across
different tasks. Pairing remains the report consumer's responsibility.

`--eval-tasks FILE` optionally enables these checks with `--eval-report FILE`;
using it without a report is an explicit argument error before file access.
Default replay output stays unchanged. Opted-in reports add
`reference_validation` counts for tasks, records, checked answers/text and
unchecked nonfinal records. Protocol, budget, timeout and sandbox failures remain
failures, even if their text resembles a final answer. Absent text observations
stay unknown. Checks establish agreement with supplied references, not source
authenticity, label quality, complete model prompts, matched hardware or accepted
study data.

**Finite scope and results:** sixteen new synthetic fixtures, at most four
records/tasks each; no model/guest calls. All final categories across three
checker types, differing per-case references, inconsistent labels/identities/
text, opaque-checker non-execution, nonfinal failures and CLI gating passed.
Full record/CLI focus, including old controls: **53 passed** (0.13 s); coupled
record/evaluation/CLI: **185 passed** (8.27 s); root: **452 passed** (21.85 s).
Root Ruff, five-file format, whitespace and README local-link targets passed.
Three owning-source diagnostics were empty; six validator and fourteen CLI
callers were signature-compatible. Test diagnostics were only unused trap
parameters. Fourteen existing Torch warnings, no skips; existing jailed/native
controls ran. One focus, coupled/root invocation and formatter application
spent, no code failures or aborts. Fixture streams/files were temporary.

A configured GitHub MCP lookup queried a different host and returned 404.
One supported public-REST alternative confirmed [task-specification CI][task-spec-ci]
(`3b3da30`), `a062c59` and `67c3e17` all completed successfully. No credentials,
security settings, remotes or host privileges were changed.

[task-spec-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44043640416

### 2026-11-17: persist complete CPU serving observations without pickle

`dump_serving_results(results, stream)` / `load_serving_results(stream)` use a
separate strict schema-v1 `serving_cases` envelope. All serving fields and nested
case evidence are retained: mode, integer token history, optional logit-prefix
boundary, work/cache counters and optional logits. Metadata/counter validation
precedes tensor reconstruction. Unknown fields/modes and inconsistent payloads
fail explicitly; no missing logits or prefix observations are imputed.

Detached dense CPU float16/bfloat16/float32/float64 logits carry exact dtype,
shape, native byte order and raw lowercase hexadecimal bytes. This preserves
mask infinities, NaNs and signed zero without nonstandard JSON floats, pickle,
NumPy or model loading. Readers reject another byte order and inconsistent
lengths before reconstructing independent CPU tensors. Non-CPU, gradient-bearing
or unsupported-layout/dtype inputs fail rather than being moved or silently
dropped. These are value observations, not live-cache checkpoints, weights,
training graphs, cross-endian conversion or numerical/serving qualification.

**Finite scope and results:** fourteen new synthetic fixtures, at most three
serving rows/four logit elements each (one six-case base helper per fixture).
Four-dtype bitwise and report round-trips, seven malformed-input pre-reconstruction
traps, distinct missing/empty logits and non-CPU pre-write rejection passed.
Full record focus: **57 passed** (0.14 s); coupled record/serving:
**123 passed** (4.42 s); root: **466 passed** (17.43 s). Root Ruff, three-file
format, whitespace and README local-link targets passed. Both owning sources
had empty diagnostics; seven new API callers were signature-compatible. Test
diagnostics were only unused trap parameters. Fourteen existing Torch warnings,
no skips; existing jailed/native controls ran. One focus, coupled/root invocation
and formatter application spent; no failures/aborts, model forwards, guest
execution, GPU access, dependencies, downloads or retained new artifacts.
Next concrete action: expose serving-report replay with the same optional exact
reference checks and explicit pairing errors.

### 2026-11-17: replay saved serving reports through the public CLI

`--serving-report FILE` now loads complete serving observations and emits the
existing strict-JSON serving summary without inference or program execution.
It is mutually exclusive with the other CLI actions. Optional `--eval-tasks`
reference validation now works with either report action; the default matched
report shape and reference-before-aggregation ordering remain unchanged.
An empty serving input is not mistaken for matched-case evidence. Duplicate or
incomplete mode pairs and mismatched observations fail without partial output.
Single-mode reports remain supported without paired mode deltas.

This aggregates captured evidence; it does not execute a serving engine,
certify numerical equivalence of logits or accept a benchmark cohort.
README usage and the concise Current Work map distinguish those limits.

**Finite scope and results:** three new synthetic CLI fixtures, at most three
serving rows/three CPU logit values each, with live-model-constructor and executor
traps. Serving replay with/without exact tasks and incomplete-pair rejection
passed. Full CLI focus, including old controls: **13 passed** (0.06 s); coupled
CLI/record/serving: **136 passed** (4.45 s); root: **469 passed** (18.96 s).
Root Ruff, two-file format, whitespace and README local-link targets passed.
Owning-source diagnostics were empty; fifteen `main` callers were signature-
compatible; test diagnostics were only unused trap parameters. Fourteen existing
Torch warnings, no skips; existing jailed/native controls ran. One focus,
coupled/root invocation and formatter check/application spent (no formatting
changes), no failures/aborts, model forwards, guest execution, GPU access,
downloads, dependencies or retained fixture artifacts.

### 2026-11-17: reject serving pairs with known prompt-length mismatches

Serving comparisons now reject differing observed `prompt_tokens` within the
same case, including groups with another unknown observation. Unlike the
no-tool/tool comparison, serving modes already require the same evaluation path
and prompt. Missing counts remain unknown; different tasks can retain different
lengths. Existing sample-coverage fields and metric formulas are unchanged.
Equal counts do not certify identical prompts, tokenizers, models or hardware.
The new check runs after shared task-text/budget validation, preserving existing
more-specific diagnostics.

**Finite scope and results:** three new record-only fixtures, at most four rows.
First focus: **2 failed/1 passed** (0.10 s), reproducing false matched acceptance.
After repair: **3 passed** (0.08 s); coupled serving/record: **126 passed**
(4.45 s). Root attempt one: **2 failed/470 passed** (18.31 s): the initial check
masked task-text diagnostics for records with both mismatches. Moving it after
shared provenance validation repaired the precedence rather than weakening
assertions. A narrowly revised necessary focus (three new and six existing
provenance fixtures): **9 passed** (0.78 s), with at most eighteen tiny CPU
prefills/600 scripted IDs. Required root revalidation: **472 passed** (17.83 s).
The unchanged successful coupled selector was not rerun for reassurance.

Root Ruff, two-file format and whitespace passed; owning-source diagnostics
empty and fifteen summary callers signature-compatible. Fourteen existing
Torch warnings, no skips; existing jailed/native controls ran. All spent failed
attempts are retained above; no aborts, tolerance changes, studies, downloads,
dependencies or host changes. A bounded public-REST check confirmed
[`861bb78` serving replay CI][serving-replay-ci] and `a0fc479` both succeeded.

[serving-replay-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44046390586

### 2026-11-17: explicitly check recorded same-boundary CPU logits

`validate_serving_logits(results)` now checks supplied observations independently
of statistical aggregation. Shared pairing/provenance admission preserves the
default summary shape and diagnostic order without rerunning bootstraps.
A full-reprefill reference and another mode are required for every case, with
identical complete token histories, measured equal nonempty logit prefixes, and
matching detached nonempty single-sequence CPU floating dtype/shape.

The checks reproduce the existing two-call hybrid fixture contract, not a new
or relaxed numerical policy: live/prefix values agree exactly when both modes
exist; comparisons against full reprefill use `atol=1e-6`, `rtol=1e-5`.
Tolerance parameters are not exposed. Matching negative-infinity masks remain
valid; NaN/positive-infinity/all-masked maxima fail. Results explicitly count
cases, full-reference comparisons and exact live/prefix pairs and retain the
applied tolerances. No task outcomes are rewritten. Agreement of supplied CPU
values does not authenticate execution, certify task quality, or qualify GPU
kernels, production caches or serving integration.

**Finite scope and results:** sixteen new record-only fixtures, at most three
rows/four logits each, including measured inside/outside absolute-tolerance
boundaries and live/prefix drift that full-reference tolerance alone permits.
The existing two-call/three-mode tiny CPU control gained postcapture checks and
an explicit 2,048-scripted-ID cap; task, model and existing limits are unchanged.
Focused check: **17 passed** (1.09 s); coupled serving/record:
**142 passed** (4.74 s); root: **488 passed** (19.92 s). Root Ruff, three-file
format, whitespace and README local-link target passed. Both owning sources
had empty diagnostics; three checker and fifteen summary callers were signature-
compatible. Fourteen existing Torch warnings, no skips; existing jailed/native
controls ran. One focus, coupled/root invocation and formatter application spent;
no failures/aborts, new pretrained studies, guest workloads, GPU access,
downloads, dependencies or retained artifacts. Next action: gate CLI numerical
checks behind an explicit serving-report-only flag.

### 2026-11-17: opt-in numerical checks during serving-report replay

`--check-serving-logits` now requires `--serving-report` before file access,
including when combined with version, execution or matched-report actions.
Checks run before statistical aggregation and JSON output; success adds only
the explicit `logit_validation` counts/tolerances. Numerical failures emit no
partial report. Optional exact task-reference checks compose with this flag;
neither check rewrites recorded task outcomes. Default replay remains unchanged
and still accepts nonfinite tensor values as observations without claiming
numerical agreement. README usage and Current Work now describe the opt-in path.

**Finite scope and results:** three additional synthetic CLI cases, at most
three rows/three CPU logits each: successful three-mode/reference-checked replay
with model/guest execution trapped, a `2e-6` mismatch against a zero reference,
and four argument-guard combinations using absent files. CLI focus:
**14 passed**, two unrelated primitive checks deselected (0.08 s); root:
**491 passed** (17.51 s). Root Ruff, two-file format and whitespace passed;
owning-source diagnostics empty, seventeen CLI and four numerical-check callers
signature-compatible. One focused/root invocation and formatter application
spent, no failures/aborts; fourteen existing Torch warnings, no skips. Existing
jailed/native controls ran; no additional workloads, model studies, dependencies,
downloads, GPU access or retained artifacts. This checks supplied CPU
observations only, not engine authenticity, GPU kernels or production serving.

### 2026-11-17: reject boolean evaluation times before scoring

The in-memory time guards used `math.isfinite`, which accepts booleans. Unlike
the strict evidence codec, they admitted `False` as zero elapsed/execution/return
time and `True` as a positive time allowance. This allowed invalid observations
into cost metrics and made constructor admission inconsistent with persistence.
The owning `Budgets`, `CallRecord` and `CaseResult` guards now explicitly reject
booleans across all five time fields. Existing numeric durations, valid measured
zeros and unmeasured `None` return timing retain their behavior. No metric,
deadline, tolerance, record schema or observed outcome was redefined.

**Finite scope and results:** ten constructor-replacement cases, no model/guest
execution. Initial focus: **8 failed, 2 passed** (0.22 s); the two false positive-
budget cases were already rejected by the existing positivity guard. After the
boolean repair: **10 passed** (0.05 s). Coupled evaluation/record:
**199 passed** (7.45 s); root: **501 passed** (19.41 s). Root Ruff, two-file
format, whitespace and owning-source diagnostics passed. One failing focus,
one necessary repaired focus, one coupled/root invocation spent, no aborts;
fourteen existing Torch warnings, no skips. Existing jailed/native controls ran,
with no additional workloads, studies, dependencies, downloads or GPU access.
No new artifacts. Public hosted CI for the preceding pushed CLI checkpoint
`0a12d50` also [completed successfully][checked-serving-replay-ci].
Next check: the same finite-positive timeout contract at executor entry points.

[checked-serving-replay-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44049031348

### 2026-11-17: share strict timeout admission across execution tiers

The shared execution-budget guard also treated `True` as a one-second timeout.
Six public entry-point fixtures reached the trapped subprocess-launch seam:
direct jail, Python, C, assembly, Wasmtime and Unicorn. The shared guard now
rejects boolean timeouts before launch. A follow-up owning-source check found
the same duplicated guard in `BridgePolicy`; its constructor now reuses the
shared duration/output validator instead of retaining a divergent copy.
Valid timeouts, deadline accounting, output caps, compilation flags and
containment behavior are unchanged. This input-contract repair is not
production containment qualification.

**Finite scope and spent checks:** six added admission cases, with launches
trapped; initial focus **6 failed, 4 passed** (0.17 s), repaired **10 passed**
(0.04 s). First coupled execution/native **60 passed** (1.45 s), root
**507 passed** (18.18 s). The discovered policy duplication justified one
additional constructor fixture and a bounded necessary revalidation: policy
focus **1 failed** (0.04 s), repaired combined **11 passed** (0.04 s), coupled
**61 passed** (1.40 s), root **508 passed** (17.93 s). Earlier attempts remain
spent, not replaced/refunded. Each focused check stayed within 30 s, coupled
within 60 s and root within 180 s; no aborts or safety-limit changes.

Root Ruff, four-file format and whitespace passed; both owning sources had empty
diagnostics and all four shared-validator callers were signature-compatible.
Fourteen existing Torch warnings, no skips. Required existing jailed
Python/C/assembly and Wasm/Unicorn controls actually ran, with no new guest
workloads beyond those controls, dependencies, downloads, GPU access, host
security changes or retained artifacts. Next inspect cooperative model-decode
time admission for the same contract mismatch.

### 2026-11-17: reject non-callable stop callbacks before hybrid state work

Decode sessions expose token budgets/callbacks, not an independent time input;
cooperative wall time remains enforced by the harness's validated `Budgets`
and callback checks. That trace exposed a separate admission defect:
non-callable stop callbacks reached pending-token forwarding in live sessions,
checkpoint restoration in prefix mode and model re-prefill in full mode.
The shared preflight is now `_validate_decode_request`, checking token budget
then optional callback callability before any of those stateful operations.
All three consumers use it; public decode signatures and valid callbacks/`None`
retain their behavior. Live-session documentation states the admission guarantee.

**Finite scope and results:** six added tiny CPU cases reuse existing live
pending-boundary and released prefix/full fixtures. Each focus had ten cases,
at most four history IDs each, ten scripted IDs and thirty initial prefill IDs;
post-admission forward/restore seams were trapped, not extra model execution.
Initial focus **6 failed, 4 passed** (0.39 s); repaired **10 passed** (0.22 s).
Coupled checkpoint/serving **139 passed** (5.71 s), including the existing
multi-call/full-prefill numerical controls; root **514 passed** (19.75 s).
One failing focus, one repaired focus, one coupled/root invocation spent, no
aborts; original context/token/tensor/tolerance limits unchanged. Root Ruff,
four-file format and whitespace passed; both owning-source diagnostics empty,
all three shared-preflight callers signature-compatible. Fourteen existing
Torch warnings, no skips. No studies, dependencies, downloads, GPU access or
retained artifacts. Hosted CI for pushed timeout repair `e62abfb`
[completed successfully][strict-timeout-ci].

Next: inspect task-specification cohort overlaps as an offline held-out-family
prerequisite, without accepting a split or collecting task/model data.

[strict-timeout-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44050852632

### 2026-11-17: inspect declared development/evaluation task overlaps offline

`inspect_task_overlap(development, evaluation)` now inventories supplied task
specifications without executing checkers or creating labels. It independently
reports sorted exact case-ID/family intersections and evaluation IDs whose
prompt text repeats development text, even when answer format, ID or family
changed. This prepares the existing held-out-by-family requirement without
choosing/accepting a split or adopting a new benchmark policy. Comparisons are
exact declared strings, not semantic aliases or pretraining-exposure evidence.

Within-cohort IDs must be unique, matching the existing specification contract;
cross-cohort repetition is reported, not silently rejected or removed. Empty
cohorts retain explicit task counts and produce no pass/qualification decision.
Default report shapes and persisted schemas are unchanged. Public exports,
README and Current Work describe the observation-only utility.

**Finite scope and results:** seven synthetic metadata fixtures/eight API
queries, at most four tasks/query and 256-byte fixture prompts, no tensor/model/
guest execution. Focus **7 passed** (0.12 s); coupled evaluation/record
**206 passed** (7.60 s); root **521 passed** (19.53 s). One focus, coupled/root
invocation and formatter application spent; no failures/aborts, fourteen
existing Torch warnings, no skips. Root Ruff, three-file format, whitespace,
both owning-source diagnostics and four signature-compatible call sites passed.
An initial workspace-symbol query preceded indexing; direct function-location/
signature checks and refreshed diagnostics confirmed the new API. No new task
acceptance, datasets, studies, dependencies, downloads, GPU access or artifacts.
Next: expose this same informational inventory during optional reference-checked
CLI replay, leaving ordinary output and acceptance boundaries unchanged.

### 2026-11-17: optionally include task-cohort inventory during report replay

`--eval-development-tasks FILE` now composes with `--eval-tasks` and either
matched or serving replay. It requires the reference-task option before file
access; the existing reference option still requires a report action. Both
specifications use strict task I/O, and validation/inventory finishes before
aggregation or output. Opted-in JSON adds only `task_overlap`; ordinary report
shapes remain unchanged. Actual overlaps are informational and can accompany
successful report generation, not a split-acceptance or absence-of-exposure
decision. Invalid development specifications emit no partial report.

**Finite scope and results:** initially three new synthetic CLI cases, at most
three case rows/three CPU logits and two task specifications: combined serving/
logit/reference/inventory replay with execution trapped, four missing-reference
argument combinations and malformed-development-schema failure. Initial CLI
focus **17 passed** (0.09 s), root **524 passed** (19.37 s). Exact-contract
review required a positive matched-action inventory case as well: one bounded
extra fixture, **1 passed** (0.05 s), then the necessary final root
**525 passed** (21.46 s). Earlier successes remain spent; no unchanged full
focus rerun, failures, aborts or relaxed limits.

Root Ruff, two-file format and whitespace passed; owning-source diagnostics
empty, nineteen CLI/five inventory call sites signature-compatible. Fourteen
existing Torch warnings, no skips. No new model/guest workloads beyond existing
root controls, accepted data, studies, dependencies, downloads, GPU access or
artifacts. README documents both actions and the informational boundary.
Next inspect direct serving-adapter sampler admission before model prefill.

### 2026-11-17: preflight direct adapter sampler overrides

The comparison runner already rejected invalid sampler-factory products, but
direct live/prefix/full adapter constructors bound non-callable overrides only
after model prefill or prefix-store access. Six failing-first fixtures reached
the trapped model-forward seam. All constructors now share an optional-sampler
callability guard before those operations; prefix checkpoint-budget validation
retains its existing precedence. Existing `None`/callable behavior, factory
diagnostics, public signatures, decode budgets and numerical tolerances remain
unchanged. Module documentation states the direct-admission guarantee.

**Finite scope and results:** six `False`/integer sampler cases across the three
constructors, existing tiny CPU model construction, three prompt IDs each, no
new forwards or sampled IDs. The empty prefix store stayed within its 1 MiB cap
with entry/byte/hit/miss/eviction observations unchanged after rejection.
Initial focus **6 failed** (0.56 s); repaired **6 passed** (0.19 s); coupled
checkpoint/serving **145 passed** (10.87 s), including unchanged multi-call
reference-logit controls; root **531 passed** (30.29 s). One failing focus,
one repaired focus, one coupled/root invocation spent, no aborts. Root Ruff,
two-file format, whitespace, owning-source diagnostics and three shared-guard
callers passed. Fourteen existing Torch warnings, no skips. No new studies,
dependencies, downloads, GPU/host-security changes or artifacts.

Hosted CI for pushed cohort replay `c3a2b82`
[completed successfully][task-overlap-replay-ci]. Next reassess the three
milestone gates against current source prerequisites and available resources;
do not infer fresh model-study or production-containment authority from fixtures.

[task-overlap-replay-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44054028534

### 2026-11-17: prepare opt-in kernel cgroup PID and memory checks

The [direct-sampler checkpoint passed hosted CI][cgroup-probe-prior-ci].
A fresh source/resource sweep found a useful Queue 3 prerequisite: the cgroup
tests exercised synthetic control files but had no real controller-limit probes.
The [cgroup suite][cgroup-probe-tests] now includes bounded trusted jailed controls and opt-in
real PID/memory checks through the existing `run_python`/`request_cgroup` path.
No execution defaults, production policies or public interfaces changed.

The PID source attempts at most eight forks, holds children on a pipe, gives
each child a three-second alarm and reaps them in `finally`. Its unrestricted
jailed control must create eight children; a separately reported host
process-creation restriction is not a successful control. The memory source
touches at most 96 MiB. Both have five-second request deadlines and 1,024-byte
output caps. The opt-in PID request uses a total eight-PID/128 MiB cgroup;
the memory request uses eight PIDs/64 MiB with the existing zero-swap/OOM-group
settings. Fresh owned groups must be removed after each request.

**Attribution matters:** [Linux v6.8 PID accounting][cgroup-probe-kernel-68]
can increment the forking group's generic event counter when an ancestor's
limit failed; peak watermarks can include reverted charges. The new checks
therefore require `pids.events.local`/`memory.events.local`, and reject the
legacy `pids_localevents` mode before PID qualification.
[Modern default PID accounting][cgroup-probe-kernel-618] attributes the local
limit event to the group whose limit failed. PID success requires both a
bounded `EAGAIN` fork denial and a positive local `max` delta. Memory success
requires explicit execution failure after its start marker and positive local
`max`/`oom`/`oom_kill` deltas, not an ordinary `MemoryError` or timeout.
Counters are read in the supervisor before removal; `-s` emits their deltas
and the observed status/return code as JSON.

Run the real checks only with an owner-provided writable delegation supporting
these local event files and the existing required cgroup controls:

```sh
NEURAL_COMPILER_TEST_CGROUP_ROOT=/owner/delegated/subtree \
  python -m pytest -q -s tests/test_cgroups.py \
  -k opt_in_kernel_cgroup_limits
```

An unset root explicitly skips both checks. A configured invalid/read-only
root, missing counter/controller or ambiguous legacy PID accounting fails;
there is no namespace/filter fallback, controller enabling or remount.

**Finite ledger:** initial four-case focus **2 passed, 2 skipped** (0.67 s):
both jailed controls actually ran, including all eight forks and memory
touching. One owned ordinary-directory negative run produced exactly
**2 expected failures** (0.10 s), with explicit cgroup-v2 admission errors,
no guest execution and its temporary root cleaned. Primary-source inspection
then identified the ancestor-attribution issue above; one new guard fixture
made two admission queries, **1 passed** (0.06 s). Earlier successes remain
spent, with no unchanged focus rerun. Coupled cgroup/execution/native/seccomp
gate **98 passed, 2 skipped** (2.26 s); root **534 passed, 2 skipped**
(23.00 s), fourteen existing Torch warnings. Root Ruff, scoped format and
whitespace passed. Initial formatting needed three call-wrapping changes;
one formatter application repaired them. Pylance reported no errors, only
existing unused-parameter hints. An initial kernel-symbol lookup missed
v6.8's inline accounting; bounded corrected primary-source excerpts succeeded.

**Evidence limits:** the opt-in controller checks have **not run on a writable
kernel cgroup**. This host is Linux 6.18.58.1 WSL2, with a read-only cgroup-v2
mount, visible `cpu memory pids` controllers and empty subtree control.
The test-root opt-in was unset; the two skips are missing evidence, not passes.
Pre-control `MemAvailable` was 10,066,180 KiB, a point-in-time observation,
not a hardware guarantee. No host privileges, mounts, controllers, dependencies,
model/data artifacts or GPU state changed; no new model study or tracing run
occurred. Small public kernel-source lookups were retained only as references;
no generated artifacts or probe children remain.

The fixture preparation does not establish production containment, CPU/swap
qualification, other execution-tier controller behavior, accepted study data,
or real-engine performance. Next reassess the remaining offline evaluation
and serving prerequisites while retaining the precise delegated-host gate.

[cgroup-probe-prior-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44055130311
[cgroup-probe-tests]: ../../tests/test_cgroups.py
[cgroup-probe-kernel-68]: https://github.com/torvalds/linux/blob/v6.8/kernel/cgroup/pids.c
[cgroup-probe-kernel-618]: https://github.com/torvalds/linux/blob/v6.18/kernel/cgroup/pids.c

### 2026-11-17: guard live configuration without tensor version counters

The [cgroup-probe preparation passed hosted CI][versionless-prefix-prior-ci].
The next source sweep found a separate P0 cache correctness gap: live models
whose tensors lack version counters intentionally remain usable without CPU
checkpoints, but their prefix configuration was not recorded. The shared
prefix guard also gated configuration comparison on tensor-version tracking.
Changed RoPE/EOS settings could therefore reach a model forward against a
stale live hybrid prefix or silently rebuild a released full-reprefill turn.

The [live session][versionless-prefix-session] now records serialized model/EOS
configuration after every initial prefill, independently of optional weight
version tracking. The shared guard compares each available observation
independently. This also protects released full-reprefill adapters, which
already retain and validate the captured configuration. There are no public
signature, execution-budget, numerical-tolerance or checkpoint-eligibility
changes. Unversioned live continuation remains supported; version-tracked CPU
weights are still required for checkpoints. This does **not** make mutations
of unversioned tensor contents detectable.

**Finite scope and results:** four new tiny CPU inference-weight cases
(RoPE/EOS changes crossed with decode/inject) retain a forced pending token;
two released full-reprefill cases cross EOS changes with decode/inject. Per
focused invocation, the new cases use six three-ID prefills and six forced
samples, with subsequent model forwards trapped. The existing usable
unversioned/no-checkpoint control remains included. Initial focus
**6 failed, 1 passed** (1.15 s), all failures reaching the trapped forward/
rebuild seam; repaired focus **7 passed** (0.21 s). Pending/history IDs,
cached length, usability, exact logits and released-adapter prefill counters
remain unchanged after rejection. Coupled checkpoint/serving regressions
**151 passed** (10.83 s), including the multi-call/full-reference controls;
root **540 passed, 2 skipped** (24.63 s), fourteen existing Torch warnings.
One failing focus, one repaired focus, one coupled/root invocation spent,
no aborts or relaxed limits. Root Ruff, three-file format, whitespace and
owning-source diagnostics passed; Pylance identified the five shared-guard
call sites before editing.

The two root skips remain the explicitly unconfigured writable-cgroup
qualification checks, not newly passing kernel evidence. These CPU fixtures
do not establish CUDA/FLA behavior, pretrained accuracy, serving latency/fit,
or production checkpoint support. No new dependencies, model/data downloads,
accepted task cohorts, GPU jobs, host changes or retained artifacts.
Next continue the remaining evaluation/serving prerequisite sweep rather than
treating this source checkpoint as project completion.

[versionless-prefix-prior-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44059553073
[versionless-prefix-session]: ../../src/neural_compiler/bridge/session.py

### 2026-11-17: remaining milestone gates after the new source prerequisites

Controller-probe preparation `60a13a2` and versionless-prefix repair `f0978b3`
were pushed normally; actual remote SHAs matched the local checkpoints.
The [latest source checkpoint passed hosted CI][post-guard-source-ci].
Its required local gate remains **540 passed, 2 explicitly unconfigured
cgroup skips**, root Ruff, scoped formatting and whitespace.

A fresh owning-code/test/resource sweep checked each remaining queue, rather
than treating those green checkpoints as project completion:

| Queue | Current local evidence | Actual remaining gate |
|---|---|---|
| Held-out evaluation | The [matched runner][post-guard-matched] preflights all three labelled paths and equal budgets; exact checkers, synthetic generators, uncertainty, task/result I/O, reference replay and declared overlap inspection are exported and root-tested. Harness execution and saved-reference validation consistently give format validity precedence. | Actual held-out cohort/family/split acceptance and a fresh finite pretrained/trajectory study scope. Synthetic unit fixtures are not accepted evaluation data, and existing experiment ledgers do not renew runs. No additional concrete offline source repair was identified in this sweep. |
| Serving comparison | The [comparison driver][post-guard-serving] uses the shared caller-owned model with live/exact-prefix/full-reprefill adapters. The latest coupled CPU gate passed 151 cases, including pending-token, versionless-configuration and multi-call reference-logit controls. | SGLang and vLLM are still absent from the selected environment. Real-engine adoption needs a reviewed compatible dependency/runtime scope, then actual hybrid-state/kernel/resource measurements. Existing Transformers-forward fixtures do not qualify a serving engine; no new decoder abstraction was needed to repeat them. |
| Containment/native tiers | Trusted jailed Python/C/assembly and Wasm/Unicorn controls ran in the root gate. The [new PID/memory probes][cgroup-probe-tests] have passing bounded jailed controls and closed invalid-root admission, but their real controller cases remain unexecuted. | Every visible cgroup-v2 mount was checked: the only mount is `/sys/fs/cgroup`, read-only and nonwritable, with no supplied test delegation. Real controller qualification needs an already writable owner-delegated subtree; do not remount, enable controllers or expose untrusted workloads. |

Fresh capability evidence: `/opt/venv/bin/python` is Python 3.12.3;
workspace free disk is 362.1 GiB. The cgroup mount reports `rw,nsdelegate`
superblock options but has a read-only VFS mount, as independently confirmed
by `statvfs` and write-access checks. Visible controllers are `cpu memory pids`;
subtree control is empty and `NEURAL_COMPILER_TEST_CGROUP_ROOT` is unset.
No GPU/model initialization, dependency installation or host change was used
to obtain these observations.

Policy/approach selection remains delegated, not a missing routine-consent
gate. The remaining actions above need actual inputs, scoped runtime adoption
or unavailable kernel delegation/qualification. The bounded sweep found no
additional concrete useful authorized source prerequisite; it does not prove
the absence of all bugs, finish the research project or record a user pause.
No unchanged successful tests were rerun for this documentation-only handoff.

[post-guard-source-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44061421911
[post-guard-matched]: ../../src/neural_compiler/eval/harness.py
[post-guard-serving]: ../../src/neural_compiler/eval/serving.py

### 2026-11-17: account for independently materialized checkpoint logits

A fresh bounded CPU repro confirmed that [snapshot preflight][snapshot-clone-bytes-session]
deduplicated source storage across cache and logits, although capture deep-copies
the cache and independently clones logits. A cache-alias view therefore omitted
the separate logit allocation; broadcast logits materialized more bytes than
their source storage; strided logits discarded backing slack but were overcharged.
The store retains the supplied checkpoint without cloning it, so its admission
and LRU totals inherited the incorrect producer metadata.

Preflight now counts unique cache backing storage plus the independently
materialized logit elements. Existing CPU/no-grad/storage validation, independent
snapshot/restore ownership, public signatures, default limits and numerical
tolerances are unchanged. Rejection still precedes cache copying; copies are
not allocated to discover whether a request exceeds its budget.

**Finite scope and results:** [three new synthetic view cases][snapshot-clone-bytes-tests]
each use one three-ID tiny CPU hybrid prefill and at most 1 MiB of snapshot
tensor storage. A successful case makes two captures, one rejected capture
whose copy seam is trapped, and one rejected store insertion. Initial focus
**3 failed** (0.35 s): declared versus actual retained tensor bytes were
**1664/1696** (cache alias), **1668/2712** (broadcast) and **3760/2712**
(strided). Repaired focus **3 passed** (0.09 s), checking actual cloned
storage, exact-budget acceptance, one-byte-under rejection before copying,
and unchanged store entries/bytes after rejection. Coupled checkpoint/serving
regressions **154 passed** (11.94 s), including the unchanged independent-clone
and multi-call/full-reference controls; root **543 passed, 2 skipped**
(34.43 s), fourteen existing Torch warnings. One failing focus, one repaired
focus, one coupled/root invocation spent, with no aborts or relaxed limits.
Root Ruff, two-file formatting, whitespace and owning-source diagnostics passed;
test diagnostics contain only five existing unused mock-argument hints.

The two root skips remain the explicitly unconfigured writable-cgroup
qualification probes. Trusted jailed Python/C/assembly and Wasm/Unicorn
controls ran. These are retained CPU tensor-byte measurements, not process-RSS,
GPU fit, latency, pretrained accuracy or production-serving qualification.
No new dependencies, downloads, accepted cohorts, model studies, host changes
or retained generated artifacts. Continue the remaining source prerequisites;
this repaired budget does not complete the outstanding evaluation,
real-engine or containment gates.

[snapshot-clone-bytes-session]: ../../src/neural_compiler/bridge/session.py
[snapshot-clone-bytes-tests]: ../../tests/test_checkpoints.py

### 2026-11-17: enforce the frozen syscall policy's declared tuple input

The [snapshot-byte checkpoint passed hosted CI][immutable-syscalls-prior-ci].
The next policy-source check found that the frozen
[SeccompPolicy][immutable-syscalls-source] accepted mutable lists, sets and
mapping keys as its allowlist. It also accepted a bare string such as `write`,
iterating its characters as syscall names. Frozen field assignment did not
make those mutable input containers immutable.

Constructor admission now requires the declared tuple container before the
existing nonempty, uniqueness and name checks, matching the nearby cgroup
tuple-validation pattern. Valid tuple policies, merge behavior, syscall
contents, filter construction and execution limits are unchanged. This rejects
undeclared input shapes; it does not adopt a production policy.

**Finite scope and results:** four new constructor cases (string/list/set/
mapping, at most two intended names each) plus the unchanged valid-tuple/merge
control. The selected fixtures make no bridge requests, filter exports or
model forwards. Initial focus **4 failed, 1 passed** (0.04 s), all failures
showing the wrong container was accepted; repaired focus **5 passed** (0.04 s).
One coupled syscall/execution/native/import/cgroup invocation **114 passed,
2 skipped** (2.53 s); root **547 passed, 2 skipped** (29.84 s), fourteen
existing Torch warnings. One failing focus, one repaired focus, one coupled/root
invocation spent, with no aborts or relaxed limits. Root Ruff, two-file format,
whitespace and source diagnostics passed; test diagnostics contain one existing
unused fixture-argument hint. Trusted jailed Python/C/assembly and Wasm/Unicorn
controls ran, including the unchanged filtered Python positive/denial controls.

The two skips explicitly require `NEURAL_COMPILER_TEST_CGROUP_ROOT` to name a
writable delegated subtree. There are no new controller, syscall-inventory,
production-containment, model-quality or serving measurements. No dependency,
download, host-security, accepted-cohort or retained-artifact changes. Continue
bounded evidence prerequisites without changing the candidate syscall list.

[immutable-syscalls-prior-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44076972163
[immutable-syscalls-source]: ../../src/neural_compiler/bridge/seccomp.py

### 2026-11-17: bounded trusted Python startup syscall observations

The [immutable-policy checkpoint passed hosted CI][runtime-probes-ci].
Two separate bounded trusted requests advanced the candidate-policy evidence
without changing its syscall list. A temporary command wrapper prepended the
installed strace to the existing [jailed Python command][runtime-probes-execution];
the supervisor, read-only work mount, namespaces and normal
`/usr/bin/python3 -I -u /work/prog.py` flags were unchanged. These were default
no-filter fixtures, not a bypass of an enabled filter. Both retained a five-second
request deadline and 16,384-byte combined output cap.

One `strace -f -c -qq -- <guest command>` request evaluated
`math.comb(40, 20)`. Strace **6.8** on kernel
`6.18.58.1-microsoft-standard-WSL2` reported **30 syscall names, 324 counted
calls and 46 errors**. Status/return code were **ok/0**, stdout was exactly
`137846528820\n`, and parsed per-row calls/errors matched the complete summary
footer. The names outside the current candidate were:

| Observed syscall | Calls | Errors |
|---|---:|---:|
| `access` | 1 | 1 |
| `socket` | 2 | 0 |
| `connect` | 2 | 2 |

That new evidence prompted one narrower attribution request, not another
summary run. It selected only `access,socket,connect,write` with 96-byte strings
and inserted a fixed trusted stderr body marker before importing math.
All five access/socket/connect calls preceded that marker; none followed it.
The access checked `/etc/ld.so.preload` and failed with `ENOENT`; both sockets
were `AF_UNIX`, and both connects targeted `/var/run/nscd/socket` and failed
with `ENOENT`. The request again returned **ok/0** and the exact arithmetic
answer. This attributes the observed probes before the arithmetic body,
not to a required guest network capability.

One installed-tool version query (two-second cap), one summary request and
one evidence-driven attribution request were spent; no failed attempts,
aborts, relaxed caps, raw trace files or retained guest source. Only owned
in-jail children were traced. The [unchanged filtered arithmetic control][runtime-probes-tests]
already passed in the **547-pass/two-skip** root gate with those three names
absent from its allowlist. Observed use is not proof a call must be allowed:
no socket/connect/access permission was added.

These are local post-launcher interpreter observations, not an exhaustive
syscall contract, trusted-launcher inventory, latency measurement or production
containment qualification. No host-security, dependency, model/data or policy
adoption changes. This evidence-only update does not rerun an unchanged code
gate; scoped references and whitespace are checked before its checkpoint.

[runtime-probes-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44078636047
[runtime-probes-execution]: ../../src/neural_compiler/bridge/execution.py
[runtime-probes-tests]: ../../tests/test_seccomp.py

### 2026-11-17: reject unrepresentable native runtime work counters

The [native runtime wrapper][runtime-counter-native] admitted any positive
Python integer, but the installed Wasmtime **50.0.1** binding passes fuel as
`ctypes.c_uint64` and Unicorn **2.1.4** passes instruction count as
`ctypes.c_size_t`. Both are 64 bits here: `18446744073709551615` survives
conversion, while one above becomes zero. A bounded actual jailed WAT program
returning constant 42, requested with fuel `2**64`, reproduced the failure:
**timeout/3**, empty stdout and an all-fuel-consumed trap instead of 42.

Admission now verifies that the positive budget survives its actual unsigned
C-counter conversion unchanged, before deadlines, workspace creation or worker
launch. The public runner docstrings document those representation limits.
Defaults, representable budgets, source/output/wall limits, compiler settings
and status mappings are unchanged. This enforces the underlying binding's
range, not a new workload threshold. VM engines remain inside jailed workers.

**Finite scope and results:** one diagnostic constant-WAT request, capped at
three seconds and 1,024 output bytes; [six new boundary cases][runtime-counter-tests]
cross the two runtimes with their exact unsigned maximum and the next two
integers. Each maximum case uses one bounded namespace probe and one actual
constant worker; each overflow case traps workspace creation. Programs contain
no WAT loops/imports or exactly `mov eax,42; ret` in hex. Initial focus
**4 failed, 2 passed** (0.34 s): every overflow reached the setup trap, while
both maximum-value workers returned 42. Repaired focus **6 passed** (0.22 s);
coupled native/execution regressions **67 passed** (2.49 s); root
**553 passed, 2 skipped** (30.68 s), fourteen existing Torch warnings.
One diagnostic, one failing focus, one repaired focus, one coupled/root
invocation spent; no aborts or relaxed limits.

Scoped Ruff passed. The first format check found only a new long decorator;
the existing formatter repaired it and the two-file format check passed.
Root Ruff/whitespace and source diagnostics passed; test diagnostics contain
only pre-existing unused mock-argument hints. Two binding-definition lookups
hit a Pylance internal error; installed distribution metadata/source reads and
standard-library ABI probes supplied the facts without importing VM engines.

Trusted jailed Python/C/assembly and Wasm/Unicorn controls actually ran,
including the new exact-maximum worker cases. The only root skips remain the
unconfigured writable-cgroup probes. No dependency, download, model/data,
accepted-policy, host-security or retained-artifact changes. These fixtures
do not establish production containment or throughput at enormous budgets.
Continue useful offline/native prerequisites rather than equating a green
counter boundary with completion of the remaining project gates.

[runtime-counter-native]: ../../src/neural_compiler/bridge/native.py
[runtime-counter-tests]: ../../tests/test_native_bridge.py

### 2026-11-17: apply counter admission to shared bridge policies

The [direct-runtime counter checkpoint passed hosted CI][policy-counter-ci].
Its dependent-callsite check found that [BridgePolicy][policy-counter-source]
still accepted out-of-range `wat_fuel` / `hex_max_insns` until dispatch.
The existing representation check is now one shared private validator used by
both direct runtime calls and policy construction. Unsupported frozen
configurations reject immediately; exact maxima, defaults, old field-specific
positive-integer errors and execution behavior remain unchanged.

This narrowly revised source scope retained the preceding spent ledger.
Four new constructor-only cases cross both fields with their exact unsigned
maximum and one above. Initial focus **2 failed, 2 passed** (0.08 s);
repaired focus **4 passed** (0.03 s). No bridge requests, filter exports or
model forwards occur in those cases. The new source regression gate includes
the preceding six runtime-boundary cases as unchanged controls: coupled
native/execution **71 passed** (2.10 s), root **557 passed, 2 skipped**
(32.76 s), fourteen existing Torch warnings. One failing focus, one repaired
focus and one coupled/root invocation spent, with no new VM diagnostic,
aborts, relaxed limits or reset of earlier attempts.

Root Ruff, two-file formatting, source diagnostics and whitespace passed.
Relevant jailed Python/C/assembly and Wasm/Unicorn controls ran, including
exact-maximum worker checks. The two skips still require writable cgroup
delegation. No dependency, model/data, accepted-policy, host-security or
retained-artifact changes. This completes the identified caller-consistency
repair, not production containment or the outstanding project milestones.

[policy-counter-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44081345922
[policy-counter-source]: ../../src/neural_compiler/bridge/native.py

### 2026-11-17: trusted native-tier runtime syscall summaries

The [shared-policy checkpoint passed hosted CI][tier-runtime-ci].
Four default no-filter runtime-tail summaries now complement the Python
observations. Requests were serialized in one reused controller under a
one-worker resource budget. The temporary wrapper prepended
`strace -f -c -qq --` only to read-only [runtime execution][tier-runtime-source];
C/assembly compilation remained untraced inside its normal writable jail.
WAT/Unicorn engines remained in jailed workers with their usual runtime-prefix
mount and interpreter flags. Every request retained a five-second deadline
and 16,384-byte combined output cap; no configured filter/controller was disabled.

| Trusted constant-42 runtime | Printed syscall names | Counted calls | Errors |
|---|---:|---:|---:|
| Strict C99 / gcc | 18 | 34 | 6 |
| Linked x86-64 assembly | 18 | 34 | 6 |
| WAT worker, fuel 1,000 | 33 | 1,303 | 141 |
| Hex worker, instruction budget 100 | 31 | 1,580 | 177 |

All four successful traces returned **ok/0**, exact `42\n`, one read-only
runtime trace per tier and complete row/footer count agreement. The C/assembly
names outside the **Python-only** candidate were `access` (one failed call);
hex additionally observed two successful sockets and two failed connects.
WAT additionally observed `statfs` (one failed call) and `uname` (one successful
call), as well as that access/socket/connect set. These are contextual
observations, not required permissions or candidate additions. WAT totals
include its worker's validation/JIT; C/assembly compilation is excluded, so
this is not a matched compile-time or latency comparison.

The initial assembly fixture omitted its dialect/export framing and correctly
returned **compile_error/1** before runtime (`operand size mismatch for mov`).
After inspecting the verbatim-source contract, one narrowly revised attempt
used the [existing actual-jail golden][tier-runtime-tests] with
`.intel_syntax noprefix`, `.global solve` and `solve:`. It passed without
changing compiler flags or controls. The successful C summary was not rerun;
the then-unattempted WAT/hex requests ran once each. Total: **five** native
requests (C one, assembly two, WAT one, hex one), three compiler requests and
four runtime traces; one recorded fixture failure, no aborted jobs, refunded
attempts, raw trace files or retained temporary source/binaries.

These are tiny trusted local observations on the previously recorded strace/
kernel environment, not exhaustive syscall contracts, production containment,
large-program fit or task-quality evidence. No policies, dependencies,
model/data, accepted cohorts or host security changed. The unchanged
**557-pass/two-cgroup-skip** code gate was not rerun for this evidence-only
update; scoped references and whitespace are validated before checkpointing.

[tier-runtime-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44082394849
[tier-runtime-source]: ../../src/neural_compiler/bridge/native.py
[tier-runtime-tests]: ../../tests/test_native_bridge.py

### 2026-11-17: prepare a bounded trusted CPU-bandwidth probe

The [existing controller fixture matrix][cpu-probe-tests] now also prepares a
CPU case. Its single-process trusted program loops for a nominal **0.25
process-CPU seconds**, with start/completion markers, the existing five-second
wall deadline and 1,024-byte output cap. The default control actually runs in
the normal Python jail without requesting a cgroup; no unjailed fallback exists.

The opt-in case requests `cpu.max = 25000 100000`, 128 MiB memory, eight PIDs
and the existing zero-swap setting in one newly owned request group. It requires
successful completion plus positive `cpu.stat` deltas for `nr_periods`,
`nr_throttled` and `throttled_usec`, read before normal group removal. This is
a prepared check for triggered fair-class bandwidth throttling, not a
wall-time-ratio estimate or production CPU-rate qualification.
[Linux's interface contract][cpu-probe-interface] distinguishes these counters
from ordinary usage; the [scheduler exporter][cpu-probe-exporter] reads the task
group's CFS bandwidth state and its [period timer][cpu-probe-accounting] records
throttled periods. PID/memory still use their local event files. Test diagnostic
JSON calls the values `counter_deltas`; guest protocol/status framing is unchanged.

To select only the new real-controller case on an already owner-delegated,
CPU-enabled subtree:

```sh
NEURAL_COMPILER_TEST_CGROUP_ROOT=/owner/delegated/subtree \
  python -m pytest -q -s tests/test_cgroups.py \
  -k 'opt_in_kernel_cgroup_limits and cpu'
```

An unset root skips it. A configured invalid/read-only root or missing CPU
controller/counters fails; the fixture never enables controllers or remounts.
The current root is unset and the only visible cgroup-v2 mount is read-only,
so **no real CPU/PID/memory controller probe passed or wrote host controls**.

**Finite ledger:** new two-case CPU focus **1 passed, 1 skipped** (0.33 s);
coupled cgroup/native/seccomp gate **74 passed, 3 skipped** (1.95 s);
root **558 passed, 3 skipped** (30.89 s), fourteen existing Torch warnings.
Three nominal 0.25-second jailed CPU loops ran across these required checks;
the controller cases skipped each time. No failed/aborted attempts or reruns.
Root Ruff, changed-file formatting/whitespace and Pylance diagnostics passed
(existing unused-parameter hints only). The formatter left the file unchanged.
Trusted jailed C/assembly, Wasm/Unicorn and filtered Python controls actually ran.
The three skips are missing writable-delegation evidence, not containment passes.
No new dependencies, model/data downloads, GPU jobs, host changes, raw logs or
retained guest artifacts; the three bounded primary-source reads were in memory.
Continue the remaining offline evaluation/serving prerequisite sweep.

[cpu-probe-tests]: ../../tests/test_cgroups.py
[cpu-probe-interface]: https://github.com/torvalds/linux/blob/v6.18/Documentation/admin-guide/cgroup-v2.rst
[cpu-probe-exporter]: https://github.com/torvalds/linux/blob/v6.18/kernel/sched/core.c#L10082
[cpu-probe-accounting]: https://github.com/torvalds/linux/blob/v6.18/kernel/sched/fair.c#L6258

### 2026-11-17: remaining gates after the native and CPU prerequisites

The CPU-probe source checkpoint `734f826` was pushed with its actual remote
SHA verified; [hosted CI passed][post-cpu-source-ci]. Its local source gate is
**558 passed, 3 explicit delegated-controller skips**, root Ruff, scoped
formatting/whitespace and diagnostics. No unchanged code gate was rerun for
this evidence-only update.

A fresh owning-code/test/resource sweep retained the following concrete gates:

| Queue | Available local prerequisite | Remaining milestone input/evidence |
|---|---|---|
| Held-out evaluation | The [matched runner][post-cpu-matched] preflights caller-owned tasks and identical budgets; exact checkers, synthetic generators, strict task/result I/O, reference replay, uncertainty and overlap inspection are exported and root-tested. The private Wilson helper has one fixed-z summary caller; a configurable-confidence edge hypothesis did not apply. | Accepted held-out cohort/family/split inputs and an exact finite pretrained/trajectory study scope. Synthetic fixtures and historical spent experiments are not that acceptance or renewed run authority. No further concrete offline source repair was identified in this bounded sweep. |
| Serving comparison | The [three adapters and driver][post-cpu-serving] validate modes before dispatch and preserve the existing CPU checkpoint/pending-token comparisons. Their source and pairing/observation checks were revisited; no speculative replacement abstraction was added. | Actual terminal-environment distribution metadata confirms SGLang/vLLM are absent. Real-engine adoption needs a reviewed compatible dependency/runtime scope and actual hybrid-state/kernel/resource measurements; repeated Transformers-forward fixtures cannot substitute for them. |
| Containment/native tiers | The [native runtime bridge][post-cpu-native] and trusted jailed controls ran in the root gate. PID/memory/CPU controller probes are prepared; the [Python-only syscall candidate][post-cpu-seccomp] remains explicitly nonproduction with actual filtered arithmetic and process/socket-denial controls. | An already writable owner-delegated subtree and actual controller evidence, plus production workload/policy qualification. The three real-controller cases remain unexecuted. Native constant-42 inventories do not justify syscall widening or production policy adoption. |

Fresh capability reads: `/opt/venv/bin/python`, Python 3.12.3; 361 GiB free disk
(point-in-time, not model-fit evidence). The sole visible cgroup-v2 mount is
`/sys/fs/cgroup`: VFS read-only and nonwritable, visible `cpu memory pids`,
empty subtree control and no `NEURAL_COMPILER_TEST_CGROUP_ROOT`. No permitted
remount/controller/privilege workaround exists. No additional concrete
reversible prerequisite was identified after preparing the CPU probe; advancing
these milestones now needs the inputs above, not routine planning/test consent.
This is not a claim of exhaustive bug absence or project completion.
No new dependencies, model/data downloads, studies, GPU jobs, host changes or
retained probe artifacts; no owned jobs remain running.

[post-cpu-source-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44085847500
[post-cpu-matched]: ../../src/neural_compiler/eval/harness.py
[post-cpu-serving]: ../../src/neural_compiler/eval/serving.py
[post-cpu-native]: ../../src/neural_compiler/bridge/native.py
[post-cpu-seccomp]: ../../src/neural_compiler/bridge/seccomp.py

### 2026-11-17: reject incomplete live-forward cache boundaries

An additional owning-code pass found a concrete P0 prerequisite after the
previous bounded sweep: [live forwards][forward-boundary-source] checked the
returned cache type and logit shape, but not its sequence length. A model could
return valid-shaped logits with an empty/stale real `DynamicCache`; initial
prefill or injection then reported a healthy boundary with missing token state.
The optional checkpoint-length guard detected this too late for live-only use.

The producer now records the pre-forward cache length plus input-ID count and
requires the returned cache to match before committing cache/logits or marking
the session usable. A mismatch explicitly raises `RuntimeError`; an in-place
partially modified cache remains poisoned, and recovery uses an independent
previous checkpoint. No recurrent state is sliced, rolled back or replayed.
The installed `Cache.get_seq_length()` API selects the full-attention layer for
mixed hybrid caches; the target Qwen text forward already uses that same API
for position IDs. No public signatures, token budgets, snapshot eligibility/
byte limits, numerical tolerances or valid model behavior changed.

**Finite scope and results:** [three new regression cases][forward-boundary-tests]
cover initial empty-cache output and stale-cache output after original/restored
prefills, reusing the existing partial-recurrent-update/closed-reuse/recovery
fixture. Each new case uses at most three real tiny CPU model forwards, at most
five IDs per forward, plus its synthetic malformed-output seam. Initial focus
**3 failed** (0.16 s), all because the invalid boundary was accepted; repaired
focus **3 passed** (0.20 s). Coupled checkpoint/serving regressions **157 passed**
(13.01 s), including exact pending-token and multi-call/full-prefill reference
controls; root **561 passed, 3 skipped** (33.37 s), fourteen existing Torch
warnings. One failing focus, one repaired focus and one coupled/root invocation
spent; no aborts, refunded attempts or relaxed assertions. Root Ruff, two-file
formatting/whitespace and diagnostics passed (existing unused-parameter hints
only); the formatter left both files unchanged.

The three skips remain the unconfigured real PID/memory/CPU controller probes,
not containment passes. Trusted jailed Python/C/assembly and Wasm/Unicorn
controls actually ran in the root gate. No new dependencies, model/data
downloads, pretrained/GPU jobs, host changes or retained artifacts. These tiny
CPU controls do not establish real-engine, CUDA/FLA or production evidence.
Next inspect final-turn answer decoding against the existing exact-format
contract; do not treat this repaired slice as project completion.

[forward-boundary-source]: ../../src/neural_compiler/bridge/session.py
[forward-boundary-tests]: ../../tests/test_checkpoints.py

### 2026-11-17: preserve generated answer content and exact EOS boundaries

The [cache-boundary source checkpoint passed hosted CI][answer-content-cache-ci].
The next [evaluation producer][answer-content-source] check reproduced a real
false positive: `Final answer: 4<tool_response>2` followed by EOS was decoded
with all special tokens removed, becoming `Final answer: 42` and scoring
**correct**. A token-limited marked answer also lost its marker in saved
diagnostics. This was answer-content erasure, not a checker-definition change.

Answer decoding now preserves nonterminal special tokens. For a completed
answer it excludes exactly the sampled terminal EOS, using the existing
`DecodeResult.reason` contract rather than assuming the tokenizer's default
EOS or that every configured EOS is special. Generated/history token accounting
still includes that token exactly once. Token-limited answers preserve their
entire sampled content. An impossible zero-token EOS result now explicitly
raises before any scored record, consistently with other invalid providers.
The sampled-ID contract is clarified in the existing result docstring; no
schema, tool framing, checker/reference, metric, threshold or budget changed.

**Finite scope and results:** [five new cases][answer-content-tests] cover the
interrupted integer, marked length-limit diagnostics, configured alternate
special/ordinary EOS, and empty-EOS provider rejection. Four scripted cases
retain <=64 sampled IDs, <=65 tiny CPU model forwards, <=512 prompt/1,024 context
IDs and a five-second cooperative deadline each; the existing empty-provider
fixture uses one prefill and no sampled IDs. No executor calls occur in these
new cases. Initial focus **4 failed, 1 passed** (0.69 s); repaired focus
**5 passed** (0.56 s). The former false positive is now **invalid_answer** and
its exact marker-bearing text is retained. The alternate ordinary EOS now
terminates correctly without entering the answer.

Coupled evaluation/record-replay/serving regressions **308 passed** (25.67 s);
root **566 passed, 3 skipped** (29.77 s), fourteen existing Torch warnings.
One failing focus, one repaired focus and one coupled/root invocation spent;
no aborts, refunded attempts or relaxed assertions. Root Ruff, three-file
formatting/whitespace and diagnostics passed (unused-parameter hints only);
the formatter left all files unchanged. Existing default-EOS, exact reference,
multi-call/pending-token and full-prefill controls remain green.

The three skips still require writable delegated PID/memory/CPU controllers.
Trusted jailed Python/C/assembly and Wasm/Unicorn controls actually ran. No
dependencies, model/data downloads, pretrained/GPU jobs, host changes, accepted
datasets, regenerated historical measurements or retained artifacts. Next check
EOS configuration admission at live-session initialization: sampled token IDs
are strict integers, while the configured EOS set still uses Python equality.

[answer-content-cache-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44089672751
[answer-content-source]: ../../src/neural_compiler/eval/harness.py
[answer-content-tests]: ../../tests/test_evaluation.py

### 2026-11-17: validate configured EOS IDs before prefill

The [answer-content checkpoint passed hosted CI][eos-admission-prior-ci].
The dependent [live-session initializer][eos-admission-source] previously
converted EOS configuration directly into a set. Float/bool IDs could alias
sampled integer IDs; `[1, True]` even discarded the invalid member during set
construction. An unsupported tuple became one unmatched set member rather
than an explicit configuration error. All four malformed cases reached the
trapped model-prefill seam in the new failing-first check.

The initializer now normalizes the existing int/list/None representation and
reuses the existing strict token-ID validator **before** deduplication and
prefill. Valid scalar IDs, disabled EOS (`None`/empty list), duplicate/multiple
integer EOS IDs and their pending-token behavior are preserved. No new
nonnegativity/vocabulary restriction, signature, schema, execution policy or
budget was adopted. The installed Transformers `GenerationConfig` EOS-field
documentation independently specifies the optional int/list[int] form.

**Finite scope and results:** [seven new cases][eos-admission-tests] comprise
four malformed scalar/container inputs with all model forwards trapped and
three valid disabled/multi-EOS controls. Each valid control uses at most two
tiny CPU forwards, at most three IDs per forward and one forced sample.
Initial focus **4 failed, 3 passed** (0.33 s); repaired focus **7 passed**
(0.13 s). Coupled checkpoint/serving/evaluation regressions **318 passed**
(26.94 s); root **573 passed, 3 skipped** (26.38 s), fourteen existing Torch
warnings. One failing focus, one repaired focus and one coupled/root invocation
spent; no aborts, refunded attempts or relaxed assertions. Root Ruff, two-file
formatting/whitespace and diagnostics passed (unused-parameter hints only);
the formatter left both files unchanged. Existing ordinary/special EOS,
exact-answer replay and multi-call/full-prefill reference controls remain green.

The three skips still represent absent writable controller qualification.
Trusted jailed Python/C/assembly and Wasm/Unicorn controls actually ran. No
dependencies, model/data downloads, pretrained/GPU jobs, host changes or
retained artifacts. Next take a bounded read-only primary-source review of a
real engine's raw-token and stop-boundary APIs before any dependency adoption;
missing installed packages alone do not exhaust that compatible prerequisite.

[eos-admission-prior-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44091648417
[eos-admission-source]: ../../src/neural_compiler/bridge/session.py
[eos-admission-tests]: ../../tests/test_checkpoints.py

### 2026-11-17: pinned pause controls are not an exact tool boundary

The [EOS-admission checkpoint passed hosted CI][eos-admission-ci] at `fe07b1b`.
The next source-review prerequisite reused inspected SGLang commit
`297e54fd105e974cdf401e5b4d20253a22c16f90`, rather than selecting another
release or repeating the already documented raw-token output trace. The
explicitly uninspected gap was pause/continue control ownership and semantics.

[The request definitions][sglang-session-requests] default pause to **abort**.
`in_place` and `retract` are distinct: [the scheduler][sglang-pause-scheduler]
sets a global paused flag in both cases, but `in_place` leaves `running_batch`,
`last_batch`, `chunked_req` and `result_queue` untouched. Retraction can drain
pending results and requeue/rebuild requests; it is not the unchanged live-cache
path. Continue optionally empties the CUDA allocator cache, enabled by default.
None of these methods was executed.

The actual pause/continue methods are on [TokenizerManager][sglang-final-response].
Its pause condition gates new generation requests. Non-abort pause/continue
await `_async_dispatch_to_scheduler`; that method awaits `async_sock_send`,
whose complete helper only sends a ZMQ message. This inspected control path
does not return a sampled-token/cache-boundary acknowledgement. In-place pause
also deliberately retains pending overlap results. An application must not
equate completion of that send with the exact emitted tool-close boundary or
label it as our qualified `live_pause_resume` path. Per-request tool-result
insertion, exact hybrid state and multi-call logits still need an actual
integration hook and acceptance evidence. Other inherited/RPC interfaces are
not claimed absent.

The complete concrete [Engine signatures and request constructors][sglang-engine-source]
expose log-probability, hidden-state and custom-logit-processor options, but no
raw-logits return argument. Log probabilities and hidden states are not the
full raw-logit observations required by our existing comparison contract.
Custom-processor/RPC paths exist but were not enabled or qualified; this is
not a claim that no integration hook can be implemented.

**Finite source ledger:** eight successful HTTPS responses, **1,000,220 bytes**
total, within eight-response/1 MiB total/512 KiB per-response caps and
15-second I/O timeouts. Import ownership led through the control mixin back to
the actual manager; two subsequent reads recovered the dispatch/socket helpers,
not the settled output-handler trace. The final complete Engine parse recovered
signatures/constructors omitted from the earlier truncated-page evidence.
No failed/aborted requests or increased caps.
[The pinned root LICENSE][sglang-source-license] directly identifies
Apache-2.0; this is not transitive/model licence qualification or adoption of
an execution pin. No source bodies retained, engine imports/execution, installs,
checkouts, model/data/GPU jobs, services or host changes. Scoped references and
whitespace checked; the unchanged **573 passed, 3 skipped** code gate was not
repeated for this documentation-only result.

**Queue recheck:** [matched evaluation][pause-gate-eval] already preflights
common budgets/task identities, alongside deterministic synthetic generators
and saved-reference replay. Accepted held-out cohorts/splits and a fresh exact
study scope remain absent; unit fixtures are not accepted evaluation data.
[Serving comparisons][pause-gate-serving] retain their CPU-only evidence.
Terminal metadata still finds no SGLang/vLLM; the inspected release's
incompatible Torch/Transformers requirements need separately scoped isolated
adoption, followed by actual boundary/logit/kernel/resource qualification.
[PID/memory/CPU controller fixtures][pause-gate-cgroups] are ready, but the
sole visible cgroup2 mount remains read-only/nonwritable and
`NEURAL_COMPILER_TEST_CGROUP_ROOT` is unset. Writable owner delegation and
accepted production workload/policy qualification remain necessary; neither
host changes nor a wider syscall policy is a workaround. No additional concrete
authorized implementation prerequisite was identified in this owning-code and
resource sweep; these are remaining project gates, not project completion.

[eos-admission-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44093324066
[sglang-pause-scheduler]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/scheduler.py
[sglang-source-license]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/LICENSE
[pause-gate-eval]: ../../src/neural_compiler/eval/harness.py
[pause-gate-serving]: ../../src/neural_compiler/eval/serving.py
[pause-gate-cgroups]: ../../tests/test_cgroups.py

### 2026-11-18: distinguish input construction from a mutating forward

A fresh continuation found a compatible local recovery defect in
[LiveSession][input-materialization-source]. `_forward` marked its cache
unusable before evaluating the input-tensor constructor. Injecting `2**64`
raised Torch's `ValueError: Overflow when unpacking long long` without entering
the model, yet permanently blocked checkpointing and subsequent valid use of
the unchanged session. The defect reproduced with and without a pending token.

Input tensors are now materialized after the existing input/model/prefix
guards and **before** marking the cache unusable. Actual model entry still
poisons the session until its returned logits/cache boundary is validated;
partially mutated recurrent state is never reused. No exception is swallowed,
invalid input discarded, new token-range policy adopted or cache rollback added.
[Two regressions][input-materialization-tests] trap model entry on the failing
input, preserve token history/pending state/cache identity/logits, capture a
healthy checkpoint and then verify valid injection against full-prefill logits
at the unchanged `atol=1e-6, rtol=1e-5`.

**Finite routine ledger:** two new cases, each at most three tiny CPU forwards,
six input IDs per forward and one forced sample. The first focus **2 failed**
(0.29 s) because the test initially expected `RuntimeError`, while this Torch
raises `ValueError`; that was a fixture error, not proof of cache poisoning.
After correcting the assertion to the observed exception, the unchanged-source
focus **2 failed** (0.21 s) at healthy checkpoint recovery. Repaired focus
**2 passed** (0.06 s), coupled checkpoint/protocol/serving **187 passed**
(9.15 s), root **575 passed, 3 skipped** (22.45 s), fourteen existing Torch
warnings. Both failed attempts remain spent; no aborts or relaxed work/logit
limits. Root Ruff, two-file formatting, whitespace and source diagnostics
passed; test diagnostics contain unused-parameter hints only. Genuine
mutating-forward failures and multi-call/full-prefill controls remain green.
The three skips still lack writable real-controller qualification; actual
jailed native/runtime controls ran. No dependencies, downloads, pretrained/GPU
jobs, services, shared-host changes or retained artifacts.

### 2026-11-18: pinned observer hook is a more relevant integration candidate

The previous pause review explicitly left custom processors uninspected.
At the same inspected source pin, [custom processors][sglang-custom-processors]
serialize a class with `dill`, wrapped in JSON; their call returns a same-shape
logit tensor. This is a trusted executable extension mechanism, not a safe
recorded-evidence format or justification to enable arbitrary request code.
No processor was deserialized, imported, installed or executed.

The pinned [sampling observer contract][sglang-sampling-observer] is a more
relevant candidate for observational instrumentation: it defines `is_active`,
`before_grammar` and `after_sample`, the latter returning graph-safe device
auxiliary output for scheduler-side copying. Its documented integration spans
ModelRunner, Scheduler and TokenizerManager plugin hooks.
[ModelRunner][sglang-observer-runner] explicitly excludes speculative and DLLM
paths from its default observer-support predicate. Its complete `sample`
method also assigns local `observer_state` only in the active-observer branch,
then tests that local after either branch: inactive-path handling is a
source-visible adoption risk, not an observed engine-run failure.

Observer placement relative to all preprocessing, request/row identity, complete
auxiliary-output transport, exact tool-boundary state and target kernels remain
unqualified. Do not equate this interface with working raw-logit exports or
implement a pretend engine adapter around it. A future isolated integration
should investigate this explicit observer path before using a mutating custom
processor; this does not adopt a dependency/revision or change the comparison
contract.

**Finite source ledger:** six successful primary HTTPS responses, **179,311
bytes** total, within six-response/768 KiB total/256 KiB per-response caps,
15-second I/O timeouts and 16,000 emitted source characters. Two narrow directory
metadata reads led to the actual sampling files and measured ModelRunner size.
No failures/aborts, widened caps, checkouts, retained bodies or engine execution.
This closes the bounded source review, not runtime/cache/resource qualification.

[input-materialization-source]: ../../src/neural_compiler/bridge/session.py
[input-materialization-tests]: ../../tests/test_checkpoints.py
[sglang-custom-processors]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/sampling/custom_logit_processor.py
[sglang-sampling-observer]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/sampling/sampling_observer.py
[sglang-observer-runner]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/model_executor/model_runner.py

### 2026-11-18: observer capture is after penalties, not inherently raw

A bounded follow-up resolved the newly discovered observer's tensor stage.
[ModelRunner preprocessing][sglang-observer-runner] delegates observation to
[SamplingBatchInfo][sglang-observer-batch]. Its complete observed path applies
pre-grammar transformations, invokes `before_grammar`, applies any grammar mask,
then applies post-grammar transformations. The pre-grammar methods apply
accumulated additive/scaling penalties or the non-overlap penalizer orchestrator;
the post-grammar method applies logit bias. Therefore the hook's name does not
establish that it exposes unchanged model logits. Matching our current raw-logit
reference requires demonstrated identity preprocessing or an earlier capture,
not relabelling transformed scores as raw observations.

The [auxiliary-output contracts][sglang-observer-auxiliary] require device tensors
not to alias graph-static buffers that can be overwritten before the later host
copy. Host `consume` is specified to run after token commits and before streaming;
its commit sequence aligns with `batch.reqs`. The contract permits per-request
buffering for a streamer subclass. A snapshot of the tensor and explicit request/
commit association are prerequisites, not evidence that default responses export
those observations. An output-streamer selector found no auxiliary-named methods;
inline/inherited/subclass transport was not ruled out or fully traced. No full
transport, runtime boundary, graph or target-kernel qualification is claimed.

**Finite source ledger:** five successful primary responses, **205,262 bytes**
total, within five-response/768 KiB aggregate/256 KiB per-response caps,
15-second I/O timeouts and 14,000 selected source characters. A second
SamplingBatchInfo query followed the actual named pre/post-transform helpers,
not a repeated settled token-output trace. No failed/aborted requests, widened
caps, retained bodies, imports/execution, checkouts, dependencies, downloads of
model/data artifacts, pretrained/GPU jobs, services or host changes. This
documentation-only follow-up does not repeat the unchanged **575 passed,
3 skipped** code gate.

The [input-recovery checkpoint passed hosted CI][input-materialization-ci]
at `2a8876b`. A fresh resource/queue recheck retains the existing boundaries:
the [offline evaluation implementation][pause-gate-eval] is available, but
accepted held-out cohorts/splits and a fresh exact study scope remain absent;
the [serving implementation][pause-gate-serving] remains a CPU-fixture
comparison, while current terminal metadata still finds no SGLang/vLLM.
Installed Torch `2.10.0+cu128` and Transformers `5.22.1` still differ from the
inspected release's dependency requirements. Isolated compatible runtime
adoption, an actual exact-boundary/logit hook and measured target behavior
remain separate next steps, not replaceable with a synthetic HTTP/plugin adapter.
The [three real controller probes][pause-gate-cgroups] still need writable
owner delegation: `/sys/fs/cgroup` remains the sole visible cgroup2 mount,
read-only/nonwritable with empty subtree control, and the test-root variable is
unset. No additional concrete permitted local implementation prerequisite was
identified after the cache repair and these source follow-ups; project
qualification remains incomplete, without changing host privileges or adopting
production workload/syscall policies.

[sglang-observer-batch]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/sampling/sampling_batch_info.py
[sglang-observer-auxiliary]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/auxiliary_output.py
[input-materialization-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44277647761

### 2026-11-18: expose existing jailed Wasm/hex tiers through the CLI

Fresh autonomous continuation selected a compatible P1 usability prerequisite,
not another model study or policy-adoption gate. The [native API][cli-runtime-api]
already exported jailed Wasmtime/Unicorn execution, while [the CLI][cli-runtime-source]
exposed only Python/C/assembly. New mutually exclusive `--run-wat FILE` and
`--run-hex FILE` actions reuse the existing bounded UTF-8/newline-normalizing
reader, `--timeout`, `ExecutionResult` JSON and nonzero failure exit. Existing
fuel/instruction/output defaults, isolation and status mapping are unchanged.
Neither language becomes the default model-emission path.

[Public usage][cli-runtime-usage] now describes the no-argument/i64 WAT export,
x86-64 return/result convention, prepared dependencies, work/wall-clock caps
and experimental containment limits. No progress log or test counts were
added to the public introduction.

**Finite routine ledger:** [two dispatch/status cases][cli-runtime-smoke],
four added exact/oversized byte-budget cases, and [two actual jailed CLI
successes][cli-runtime-execution]. Each trusted runtime source is under
128 bytes and returns `42` in fewer than ten guest operations, with a
three-second subprocess deadline and no loops or model execution. Namespace/
dependency absence would skip explicitly, never use a host fallback; both
runtime cases actually ran. Initial focus **8 failed, 3 passed** (0.68 s):
all eight new cases hit unrecognized CLI arguments. Repaired focus **11 passed**
(0.30 s), coupled CLI/execution/native **99 passed** (2.08 s), root **583 passed,
3 skipped** (31.92 s), fourteen existing Torch warnings. One failing focus,
one repaired focus, one coupled/root invocation spent, no aborts, changed
thresholds or repeated unchanged green gates. One of three scoped Python
files was formatted; root Ruff, scoped formatting/whitespace and CLI diagnostics
passed; test diagnostics contain unused-parameter hints only.

The three skips still lack writable real-controller qualification. Existing
jailed Python/C/assembly/Wasm/Unicorn controls ran without production containment,
model/GPU or serving-performance claims. No dependencies, downloads,
pretrained/GPU jobs, services or host changes. Worker workspaces are cleaned;
tiny fixture files follow pytest's normal temporary-directory retention.
Next expose opt-in access to the same API fuel/instruction caps,
preserving defaults and rejecting mismatched CLI options before file access,
then verify actual bounded exhaustion rather than wall-time alone.

[cli-runtime-api]: ../../src/neural_compiler/bridge/native.py
[cli-runtime-source]: ../../src/neural_compiler/cli.py
[cli-runtime-usage]: ../../README.md#standalone-research-runtimes
[cli-runtime-smoke]: ../../tests/test_smoke.py
[cli-runtime-execution]: ../../tests/test_execution.py

### 2026-11-18: expose bounded runtime work controls without new defaults

[The CLI][cli-runtime-source] now accepts `--wat-fuel COUNT` with `--run-wat`
and `--hex-max-insns COUNT` with `--run-hex`. These directly expose existing
[native API][cli-runtime-api] arguments, not new execution policies or
thresholds. Omitted options still omit those keywords and preserve API defaults;
explicit values reuse the existing positive/unsigned-ABI-counter validator
before source-file access. Mismatched action/option pairs fail through argparse.
The [public usage][cli-runtime-usage] distinguishes these work controls from
the independent wall deadline and retains the experimental safety caveats.

**Finite routine ledger:** ten new cases comprise two supplied-cap dispatches,
four zero/ABI-overflow rejections before `Path.open`, two mismatched-action
errors and two actual jailed exhaustion cases in the existing runtime matrix.
The unchanged default-dispatch and actual `42` controls remain. The trusted WAT
loop is bounded by **1,000 fuel**, the hex loop by **100 instructions**; each
source is under 128 bytes with a separate three-second deadline and the existing
64 KiB output cap. Both actually returned worker code `3`/`timeout` with fuel/
instruction diagnostics, not a wall-clock-only failure or unjailed simulation.

Initial selector **10 failed, 4 passed** (0.95 s), all from unsupported new
options; repaired selector **14 passed** (0.49 s). Coupled CLI/execution/native
**109 passed** (2.29 s), root **593 passed, 3 skipped** (31.15 s), fourteen
existing Torch warnings. One failing focus, one repaired focus, one coupled/root
invocation spent; no aborts, changed safety/work thresholds or repeated unchanged
green gates. Scoped formatter left all three files unchanged. Root Ruff,
scoped formatting/whitespace and CLI diagnostics passed; test diagnostics are
unused-parameter hints only. Real controller skips are not qualification.
Existing jailed Python/C/assembly/Wasm/Unicorn controls ran. No dependencies,
downloads, pretrained/GPU jobs, services or host changes; worker workspaces
clean, tiny fixture files retained under pytest's normal policy.

Next examine the directly coupled setup-failure path: Wasmtime/Unicorn are
already optional prepared-environment prerequisites, and their absence should
not be presented to inference as a guest-program error eligible for repair.
Keep dependency setup failures explicit without importing native runtimes into
the supervisor, changing dependencies or trusting guest stderr as control.

### 2026-11-18: separate jailed runtime import failures from guest errors

The [trusted jailed worker][cli-runtime-worker] imported optional native runtimes
before constructing the engine/guest. Missing modules and native-loader errors
escaped with child code `1`, which [the parent][cli-runtime-api] classified as
`runtime_error`. Four actual jailed import-fault fixtures reproduced this,
so an unavailable environment could incorrectly appear eligible for code repair.
Only the Wasmtime/Unicorn import statements now catch `ImportError`/`OSError`;
they emit an explicit runtime-unavailable diagnostic and trusted worker code `4`,
mapped to the existing public `sandbox_error`. Guest, compile and work-exhaustion
statuses, raw return codes, budgets and result schema are unchanged. No native
engine import moved into the supervisor, no stderr text became a control signal,
and no dependency installation or execution fallback was introduced.

**Finite routine ledger:** four [end-to-end fixtures][cli-runtime-regressions]
cross Wasmtime/Unicorn with missing-module/broken-loader faults. Each uses one
Python namespace control and one guarded worker, with a three-second deadline,
at most 2,048 output bytes, guest source under 128 bytes and a trusted import
guard under 1 KiB. The guard exists only in the owned temporary worker directory;
normal imports are forwarded, with no installed-package or host import changes.
WAT fuel remains 1,000 and hex instructions 100, although failure precedes
engine/guest construction.

Failing focus **4 failed** (0.36 s), all as `runtime_error`/code `1` with uncaught
tracebacks; repaired focus **4 passed** (0.31 s), coupled native/execution/CLI
**113 passed** (2.34 s), root **597 passed, 3 skipped** (21.55 s), fourteen existing
Torch warnings. One failing focus, one repaired focus, one coupled/root invocation
spent; no aborts, changed limits or repeated unchanged green gates. One of three
scoped Python files was formatted. Root Ruff, scoped formatting/whitespace and
product diagnostics passed; test diagnostics contain unused-parameter hints only.
Existing jailed Python/C/assembly/Wasm/Unicorn controls ran. The three real
controller skips still lack writable delegation and are not qualification.
The [preceding work-budget checkpoint passed hosted CI][cli-runtime-budget-ci].

The [public usage][cli-runtime-usage] and native docstrings now distinguish
unavailable runtime imports from guest-code errors. Worker workspaces are
cleaned; pytest retains tiny fixture files under its normal policy. No new
dependencies, downloads, model/GPU jobs, services or host changes. This is
prototype error classification, not production containment evidence. Next
inspect whether the frozen shared dispatch policy retains a caller-owned mutable
import allowlist, and reproduce any change in admission before deciding on a
compatible repair.

[cli-runtime-worker]: ../../src/neural_compiler/bridge/runtime.py
[cli-runtime-regressions]: ../../tests/test_native_bridge.py
[cli-runtime-budget-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44280678618

### 2026-11-18: snapshot the frozen dispatch policy's import allowlist

The [shared dispatch policy][cli-runtime-api] retained the caller's mutable
`Set[str]` even though the dataclass was frozen. Two instrumented dispatch
fixtures reproduced admission changing after construction: adding `math`
changed a rejection into a launch, while removing it changed a launch into
a rejection. The constructor now snapshots a supplied allowlist as `frozenset`;
the accepted `Set[str] | None` API, `None` default, candidate module contents,
standalone `run_python` behavior and other execution controls are unchanged.
The class docstring states the construction-time snapshot semantics. This
repairs configuration ownership, not the known limits of static import checks
or an adoption of any production import/syscall policy.

**Finite routine ledger:** two [regression cases][dispatch-import-regressions],
each with two dispatches of the same sub-128-byte `import math` program,
one-second/1,024-byte configured limits and an instrumented jail boundary.
These cases exercise the real static admission path but launch no child;
temporary accepted-source workspaces are cleaned by the existing executor.
Failing focus **2 failed** (0.05 s), confirming both opposite status changes;
repaired focus **2 passed** (0.03 s), coupled policy/native/execution **101 passed**
(2.85 s), root **599 passed, 3 skipped** (24.63 s), fourteen existing Torch
warnings. One failing focus, one repaired focus, one coupled/root invocation
spent, with no aborts, changed limits or repeated unchanged green gates.
Scoped formatting left both files unchanged; root Ruff, scoped whitespace and
product diagnostics passed; test diagnostics contain unused-parameter hints.
Actual jailed Python/C/assembly/Wasm/Unicorn controls ran in regression. The
three skipped real-controller probes remain unqualified without delegation.

The [runtime-import checkpoint passed hosted CI][runtime-import-ci]. No new
dependencies, downloads, model/GPU jobs, services or host changes; tiny fixture
retention remains pytest's normal policy. Next resume the offline-evidence
prerequisite queue with bounded source/fixture inspection, retaining the
separate gates for accepted task data, trajectory collection and model studies.

[dispatch-import-regressions]: ../../tests/test_red_team_contracts.py
[runtime-import-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44281549074

### 2026-11-18: enforce immutable typed call histories in case evidence

The [frozen case record][no-tool-record-source] declared
`calls: tuple[CallRecord, ...]`, but accepted caller-owned lists and consumable
iterators. Their contents could therefore outlive the constructor's no-tool
and injected-token consistency checks. A tuple containing another object failed
later with `AttributeError` while summing response counts. The producer now
requires the declared tuple/record shape before either invariant is evaluated,
with an explicit `ValueError`. It does not silently convert invalid inputs.
Valid harness rows and strict JSON reconstruction already supply tuples;
outcomes, metric formulas, versioned schemas and accepted data policies are
unchanged.

**Finite routine ledger:** four [constructor-only cases][case-call-regressions]
use an empty list, one-item list, empty iterator and wrong-record tuple, with
one result construction per case and one fixed call fixture. No new case loads
a model, executes a guest or performs network work. Failing focus **4 failed**
(0.23 s): three missing rejections and the later `AttributeError`; repaired focus,
including existing no-tool/injected-total controls, **8 passed** (0.06 s).
Coupled evaluation/evidence **215 passed** (8.87 s), root **603 passed, 3 skipped**
(21.41 s), fourteen existing Torch warnings. One failing focus, one repaired
focus, one coupled/root invocation spent, with no aborts, changed thresholds or
repeated unchanged green gates. One of two scoped Python files was formatted;
root Ruff, scoped formatting/whitespace and product diagnostics passed. Test
diagnostics contain unused-parameter hints only.

Existing tiny CPU hybrid, strict evidence replay and actual jailed
Python/C/assembly/Wasm/Unicorn controls ran. The three real-controller skips
remain unqualified without writable delegation. The [dispatch-import checkpoint
passed hosted CI][dispatch-import-ci]. No dependencies, downloads, model/GPU
studies, services or host changes; constructor fixtures leave no artifacts,
and existing pytest temporary retention is unchanged. Next inspect the owning
serving-record/token-history checks and their JSON reconstruction, preserving
optional numerical replay and exact-boundary qualification limits.

[case-call-regressions]: ../../tests/test_evaluation.py
[dispatch-import-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44282147850

### 2026-11-18: validate serving token histories before evidence allocation

The [serving record][pause-gate-serving] accepted mutable histories and
boolean/floating/negative token IDs. Shared integer validation previously ran
only during opt-in numerical comparison. In ordinary [JSON replay][serving-token-io],
negative IDs passed integer parsing and reached logit-tensor reconstruction.
Four constructor fixtures reproduced acceptance, and the negative-ID wire case
hit the existing tensor-allocation trap.

The constructor now requires the declared immutable tuple, reuses shared strict
integer-ID validation and rejects negative history IDs before checking the
observed logit boundary. Consequently JSON reconstruction fails before logits
are allocated. Empty histories remain valid for missing/failed observations;
vocabulary bounds are not invented without model metadata. Live-session/EOS
behavior, raw logit-bit preservation, optional numerical checking and its fixed
tolerances are unchanged. This is structural evidence validation, not engine
integration or authenticated model measurements.

**Finite routine ledger:** five new cases: four constructor-only histories,
with one case/serving row each and at most one token, plus the existing wire
fixture extended with a negative ID. The latter uses six fixed base rows,
one serving row, three IDs and a 1x4 float32/16-byte tensor. No new fixture
loads a model or executes a guest. Failing focus **5 failed** (0.18 s),
repaired focus including existing empty-observation controls **7 passed**
(0.06 s), coupled serving/evidence **159 passed** (5.08 s), root **608 passed,
3 skipped** (22.21 s), fourteen existing Torch warnings. One failing focus,
one repaired focus, one coupled/root invocation spent, no aborts, changed
thresholds or repeated unchanged green gates. All three scoped Python files
were already formatted; root Ruff, scoped whitespace and product diagnostics
passed. Test diagnostics contain unused-parameter hints.

Existing exact-boundary/multi-call tiny CPU fixtures, raw-bit evidence round trips
and actual jailed Python/C/assembly/Wasm/Unicorn controls ran. The three real
controller skips still require writable delegation. The [immutable-call checkpoint
passed hosted CI][case-call-ci]. No dependencies, downloads, pretrained/GPU
studies, services or host changes; new records/streams remain in memory and
existing pytest retention is unchanged. Next recheck current queue/resources
against the owning code and existing offline prerequisites rather than
mistaking these unit passes for held-out, real-engine or containment qualification.

[serving-token-io]: ../../src/neural_compiler/eval/records.py
[case-call-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44282933535

[sglang-engine-source]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/entrypoints/engine.py
[sglang-session-requests]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/io_struct.py
[no-tool-record-source]: ../../src/neural_compiler/eval/harness.py
[sglang-final-response]: https://github.com/sgl-project/sglang/blob/297e54fd105e974cdf401e5b4d20253a22c16f90/python/sglang/srt/managers/tokenizer_manager.py

[direction-history-map]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md
[direction-history-001]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-bounded-offline-evaluation-prerequisites
[direction-history-002]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-exact-boundary-cpu-hybrid-checkpoints
[direction-history-003]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-paired-casefamily-reporting
[direction-history-004]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-opt-in-syscall-filter-plumbing
[direction-history-005]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-source-only-import-policy-inventory
[direction-history-006]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-opt-in-delegated-cgroup-requests
[direction-history-007]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-evaluation-correctness-beyond-qualification-blockers
[direction-history-008]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-serving-dependency-compatibility-prerequisite
[direction-history-009]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-repeated-stopping-diagnosis-and-continuation-safeguards
[direction-history-010]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-actual-continuation-and-complete-model-mode-guards
[direction-history-011]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-checkpoint-identity-bound-to-the-initial-prefix
[direction-history-012]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-user-requested-wip-checkpoint-and-pause
[direction-history-013]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-ci-repair-and-multi-family-offline-evaluation-prerequisites
[direction-history-014]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-bounded-checkpoint-store-and-offline-p0-serving-comparison-harness
[direction-history-015]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-in-jail-c99assembly-bridge-module-and-candidate-p1-policy-baselines
[direction-history-016]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-unified-jailed-wasmx86-bridge-wrappers-and-cli-policynative-entry-points
[direction-history-017]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-uncertainty-intervals-paired-deltas-jailed-tcc-fast-path-and-import-pre-filter
[direction-history-018]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-12-unified-bridgepolicy-dispatcher-and-red-team-contract-suite
[direction-history-019]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-review-of-intervening-implementation-and-contract-repairs
[direction-history-020]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-failed-turn-boundaries-in-full-re-prefill-comparison
[direction-history-021]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-impossible-evaluation-records-fail-before-aggregation
[direction-history-022]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-stateless-model-identity-and-deterministic-ci-classification
[direction-history-023]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-primary-sglang-compatibility-prerequisite-recovered
[direction-history-024]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-standalone-native-framing-budgets-reject-invalid-caps
[direction-history-025]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-cpu-continuation-identity-includes-generation-eos-settings
[direction-history-026]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-jail-setup-consumes-the-execution-allowance
[direction-history-027]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-checkpoint-capacity-failures-remain-matched-failed-cases
[direction-history-028]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-invalid-final-answer-formats-cannot-count-as-correct
[direction-history-029]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-target-checkpoint-storage-lower-bounds-metadata-only
[direction-history-030]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-explicit-adapter-snapshot-overrides-reach-the-comparison-helper
[direction-history-031]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-target-loader-metadata-supports-the-existing-causal-interface
[direction-history-032]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-pinned-sglang-api-and-checkpoint-source-evidence
[direction-history-033]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-jail-startup-status-is-independent-of-guest-stderr
[direction-history-034]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-matched-synthetic-suites-validate-seeds-before-arithmetic
[direction-history-035]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-checker-result-flags-must-satisfy-their-boolean-contract
[direction-history-036]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-paired-comparison-effects-remain-visible-within-each-family
[direction-history-037]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-pinned-engine-sessions-are-an-integration-candidate-not-a-cache-proof
[direction-history-038]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-append-only-session-record-ownership-traced-at-the-pinned-source
[direction-history-039]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-reject-decode-budgets-before-touching-a-continuation-boundary
[direction-history-040]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-snapshot-overrides-are-admitted-before-prefix-cache-work
[direction-history-041]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-13-native-preparation-shares-the-execution-wall-allowance
[direction-history-042]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-python-importsource-preparation-consumes-the-same-allowance
[direction-history-043]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-cli-source-ingestion-stops-at-the-byte-budget-sentinel
[direction-history-044]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-pinned-internal-stop-token-retention-is-source-confirmed
[direction-history-045]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-pinned-detokenization-separates-text-trimming-from-token-output
[direction-history-046]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-no-tool-records-cannot-include-execution-or-injected-tokens
[direction-history-047]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-injected-totals-reconcile-with-per-call-response-tokens
[direction-history-048]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-execution-result-and-call-labels-use-the-closed-status-contract
[direction-history-049]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-pinned-response-assembly-distinguishes-delta-and-cumulative-ids
[direction-history-050]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-successful-execution-results-cannot-contain-failed-exits
[direction-history-051]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-pinned-pause-controls-are-global-server-controls
[direction-history-052]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-token-ids-are-validated-before-model-or-prefix-cache-work
[direction-history-053]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-outer-workspace-io-failures-remain-explicit-execution-results
[direction-history-054]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-pinned-scheduler-emits-through-stop-token-deltas
[direction-history-055]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-remaining-queue-gates-checked-against-current-resources
[direction-history-056]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-synchronized-result-return-timing-prerequisite
[direction-history-057]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-matched-serving-deltas-including-family-effects
[direction-history-058]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-reject-expired-result-prefill-before-model-work
[direction-history-059]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-per-family-serving-costs-share-the-aggregate-contract
[direction-history-060]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-harmonic-fixture-wording-matches-exact-equivalence
[direction-history-061]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-declare-configured-executors-without-changing-native-framing
[direction-history-062]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-retain-observed-prompt-size-separately-from-generated-tokens
[direction-history-063]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-reject-malformed-samples-before-terminal-boundary-mutation
[direction-history-064]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-enforce-decoder-callbackresult-conformance-before-scoring
[direction-history-065]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-invalidate-sessions-after-potentially-mutating-model-failure
[direction-history-066]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-admit-only-valid-model-logit-column-samples
[direction-history-067]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-capture-independent-serving-logit-evidence
[direction-history-068]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-identify-the-captured-serving-logit-prefix
[direction-history-069]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-validate-causal-logit-shape-before-healthy-cache-commitment
[direction-history-070]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-preflight-serving-settings-before-factory-side-effects
[direction-history-071]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-distinguish-token-limit-cuts-from-malformed-call-termination
[direction-history-072]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-reject-premature-decoder-length-claims
[direction-history-073]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-reject-invalid-template-ids-before-factory-admission
[direction-history-074]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-complete-sampler-factory-input-preflight
[direction-history-075]: PROJECT_DIRECTION_HISTORY_2026-11-12_16.md#2026-11-16-reject-empty-templates-and-reassess-remaining-gates

### 2026-11-18: archive whole historical updates without losing evidence

This topic had reached **4,353 lines**. The [historical archive][direction-history-map]
now holds **75 complete updates / 2,202 original lines** from November 12-16,
including failed checks, approvals, spent ledgers, pauses and evidence limits.
The historical body is byte-for-byte unchanged; its SHA-256 is
`6584ec06067bb5a5617c4955e45db5fdb546532e6a687daf361deb0fd8cd389a`.
Architecture, implementation audit and newer updates remain here. Each original
dated-section fragment is retained above and links to the corresponding archive
heading. Both documents stay in the same directory, preserving relative targets;
four shared reference definitions are also retained for newer source citations.
Neither archive placement nor historical approvals authorize new experiments.

**Finite documentation ledger:** one bounded inventory and one local bulk move
of a 141,089-byte block. A subsequent zero-reference-copy result was anomalous:
the first scanner incorrectly treated unmatched backticks as spanning paragraphs,
reducing 169 apparent links to five and hiding four needed definitions. Inspection
caught this before commit. The revised checker uses the already-installed
CommonMark parser; a tiny metadata probe and one reference inventory identified
the four definitions, restored by a scoped patch. No package was installed.
The first full check then failed because its URL variable shadowed the parser
helper before fragment validation. Renaming that local and caching each
document's anchors repairs the checker without weakening its assertions.
Scoped checks cover exact block preservation, retained content, all original
anchors, reference resolution, local targets, cross-document fragments, both
2,500-line topic bounds and whitespace. The unchanged **608 passed, 3 skipped**
source gate is not rerun for this documentation-only move.

Fresh resource inspection still finds Python 3.12.3 in `/opt/venv`, Torch
2.10.0+cu128, Transformers 5.22.1, Wasmtime 50.0.1 and Unicorn 2.1.4, with about
360.91 GiB free disk. SGLang/vLLM and a local `.runs/` directory are absent.
The sole visible cgroup2 mount remains read-only/nonwritable, with cpu/memory/pids
controllers, empty subtree control and no delegated test-root variable. These
are current prerequisites, not hardware-fit or containment measurements; no
downloads, model/GPU jobs, services, dependencies or host settings changed.

### 2026-11-19: close false-positive scoring in the native experiment path

The [native experiment entry point][native-scoring-source] still used a first-match
answer regex rather than the offline harness's exact checker. Scripted fixtures
reproduced three false successes: a correct-looking answer at the **400-token
length cutoff**, an earlier `42` followed by final `41`, and an isolation failure
injected as a repairable result before an eventual `42`. This is a separate
entry point from the already-verified evaluation harness.

Native final scoring now requires completed EOS termination, removes only that
sampled terminal token and uses the existing `IntegerChecker` last-line contract.
Length cuts cannot receive credit; nonterminal special tokens are not stripped.
`sandbox_error` stops the case without response injection or another decode.
Executed calls are recorded before result framing, with zero injected tokens
and unmeasured prefill time when no return is attempted. A shared terminal-row
builder retains sampled-token counts, timing and transcript on protocol failures;
counting in the callback also retains tokens sampled before parsing raises.
Completed-row field names, five-call default and 400-token turn cap remain;
historical Part B/C code, the loader and checked-in result JSON are unchanged.
No new native accuracy or latency study was run.

**Finite routine ledger:** four new scripted, no-model cases, one synthetic
task each, at most 400 sampled IDs (only the exact cutoff reaches 400), 512 prompt
tokens and 1,024 total history IDs. Fake execution makes at most one call and
launches no child. The existing actual jailed two-call/tiny CPU hybrid regression
now also checks retained malformed-call token counts/timing. Initial focus
**3 failed, 1 passed** (0.08 s); repaired focus **4 passed** (0.04 s); coupled
protocol/evaluation **183 passed** (8.99 s); root **612 passed, 3 skipped**
(18.04 s), fourteen existing Torch warnings. One failing focus, one repaired
focus, one coupled/root invocation spent, no aborts or relaxed limits.

Root Ruff and whitespace passed; Pylance reports only unused-parameter hints.
The initial full-file format check exposed pre-existing formatting throughout
the historical script. Only the edited native routine was range-formatted:
its AST-extracted formatting check and the complete changed test-file check
pass, and an exact prefix/suffix check confirms unrelated script text is intact.
This is not a claim that the historical file now passes full-file formatting.
Actual jailed Python/C/assembly/Wasm/Unicorn controls ran in regression; the three
real-controller skips still lack writable delegation. Fixtures remain in memory
or normal pytest retention; no dependencies, downloads, pretrained/GPU jobs,
services or host changes. Next inspect native token-limit and failure-accounting
boundaries against the same harness contract before any future study.

[native-scoring-source]: experiments/exp_loop.py

### 2026-11-19: preserve finite native call caps and token-limit classification

The [same native routine][native-scoring-source] admitted NaN, infinity, fractional
and boolean `max_calls` values. In particular, `len(calls) >= NaN` never holds,
so that configuration defeated the intended serial-call limit. It now requires
a positive integer before protocol/model setup; the default remains five.
The routine also called `protocol.finish()` before inspecting a length stop,
misclassifying an incomplete call at the 400-token cutoff as a malformed call.
Length stops now retain their actual reason and sampled counts without execution,
injection or final-answer credit. EOS-ended malformed calls still fail as
protocol errors, and the turn limit is unchanged.

**Finite routine ledger:** four constructor-free invalid-cap cases plus one
400-ID incomplete-call case in the existing scripted matrix; at most 512 prompt
IDs and 1,024 history IDs, no new model/guest calls. Initial focus **5 failed**
(0.09 s); repaired **5 passed** (0.04 s); coupled protocol/evaluation **188 passed**
(8.70 s); root **617 passed, 3 skipped** (18.82 s), fourteen existing Torch
warnings. One failing focus, one repaired focus, one coupled/root invocation
spent; no aborts or relaxed limits. Root Ruff/whitespace, full test-file format
and AST-extracted native-routine format passed. Exact prefix/suffix comparison
preserves historical script text; the previous full-file formatting debt remains.
Pylance reports only unused parameters. Jailed runtime controls ran, while the
three actual cgroup-controller probes still skip without delegation. No data,
model/GPU studies, downloads, dependencies, services or host changes; fixtures
remain in memory/normal pytest retention. Next recheck real milestone resources
and remaining compatible prerequisites; these tests do not qualify native studies.

### 2026-11-19: include prompt framing in native total latency

The [native timer][native-scoring-source] started after protocol construction and
prompt tokenization, unlike the already-required full prompt-to-answer metric.
It now starts before that work, after initial device synchronization, with a
separate prefill timer preserving `prompt_prefill_ms`. A controlled no-model
clock reproduced `total_ms=30` instead of `60` for 30 ms framing, 20 ms prefill
and 10 ms decoding; after repair it records `60` total and `20` prefill. These are
fixture values, not measured model/hardware latency.

**Finite routine ledger:** one in-memory case, 17 sampled IDs, at most 512 prompt/
1,024 history IDs; no sleeps, guest execution or model load. Failing focus
**1 failed** (0.07 s); repaired **1 passed** (0.03 s); coupled **189 passed**
(8.49 s); root **618 passed, 3 skipped** (18.23 s), fourteen existing Torch
warnings. One failing focus, one repaired focus, one coupled/root invocation,
no aborts or changed limits. Root Ruff/whitespace, full test-file format and
scoped native-routine formatting passed; unrelated script text preserved exactly,
including its existing formatting debt. Pylance reports only unused parameters.
Jailed controls ran; controller skips remain unqualified. [Scoring CI][native-scoring-ci]
and [boundary CI][native-boundary-ci] passed at their exact pushed revisions.
No historical result regeneration, dependencies, downloads, model/GPU studies,
services or host changes. Next actual studies still require accepted task/run
scope; real serving needs an adopted compatible runtime, and controller
qualification needs an owner-provided writable delegation.

[native-scoring-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44505688956
[native-boundary-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44506102479

### 2026-11-19: reconcile token counts in serving logit replay

Matching token IDs and CPU logits could pass replay even when an observed prompt
count made the history too short or too long, or the logit prefix ended before
the complete prompt. The opt-in validator now reconciles each measured history
with its prompt, generated and injected counts, and requires the prefix to
cover the prompt. Unknown prompt counts remain unknown; schema, ordinary
summaries and numerical tolerances are unchanged. This rejects contradictory
evidence; it does not authenticate execution or qualify a serving runtime.

**Finite routine ledger:** four in-memory cases (three rows, four IDs and
three logits per row), no model or guest setup in the focus. Initial focus
**3 failed, 1 passed** (0.33 s); repaired **4 passed** (0.04 s); root
**622 passed, 3 skipped** (30.88 s), fourteen existing Torch warnings.
Ruff, changed Python formatting, whitespace and source diagnostics passed.
Controller skips still lack writable delegation. No data downloads,
dependencies, model studies, services or host changes. The previous
[native-timing CI][native-timing-ci] passed at its exact revision.

[native-timing-ci]: https://github.com/venim1103/neural-compiler/actions/runs/44506620598

### 2026-11-19: make evaluation intervals independent of record order

Reversing the same eight saved cases changed a fixed-seed cost interval from
about `[1.82, 18.44]` to `[1.94, 16.47]` seconds per correct answer in a
synthetic fixture. The bootstrap draws array positions, so single-path and
serving summaries now assign positions by case ID; paired time deltas also
aggregate in case-ID order. The estimator, seed, draw count and treatment of
zero-success resamples are unchanged. This improves replay reproducibility,
not the validity of an unaccepted study.

**Finite routine ledger:** one matched 24-row fixture, two report calls, and
the existing zero-success bootstrap control; no new model or guest setup.
Initial focus **1 failed** (0.18 s); repaired **2 passed** (0.16 s); root
**623 passed, 3 skipped** (33.66 s), fourteen existing Torch warnings.
Ruff, changed Python formatting (after fixing the new assertion), whitespace
and source diagnostics passed. Real controller probes still lack writable
delegation. No data downloads, model studies, dependencies or host changes.

### 2026-11-19: cap evaluation turns before sampling

The harness now includes remaining context in the decode request, after
counting prompt, generated and injected IDs. Length stops at that bound retain
their partial text without answer credit; EOS at the exact limit still scores.
A call that fills the context does not execute because its reply cannot fit.
Decode providers that exceed the requested limit still raise explicitly.

Four scripted cases with three prompt IDs each, no model or guest: initial
**4 failed** (0.22 s); repaired focus with adjacent contracts **19 passed**
(1.51 s); root **627 passed, 3 skipped** (24.22 s). Ruff, changed formatting
and source diagnostics passed. Real controller probes still lack writable
delegation. No dependencies, downloads, data or model studies.

The follow-up skips prefill if the prompt or a framed reply would leave no token
for final generation. Observed prompt length, completed calls and attempted
return timing are retained. A reply leaving one slot still allows decoding;
oversized replies keep their existing outcome. Three scripted reply cases:
**1 failed, 2 passed** (0.19 s), repaired focus **18 passed** (2.57 s), root
**630 passed, 3 skipped** (21.98 s). The strengthened scripted prompt case
then reproduced **1 failure** (0.19 s); combined focus **9 passed** (0.70 s)
and final root **630 passed, 3 skipped** (50.10 s). Ruff, scoped formatting,
whitespace and source diagnostics passed. No new model or guest setup in the
scripted cases; real controller probes still lack writable delegation.

### 2026-11-19: apply the store cap before copying a serving snapshot

The prefix-cached adapter now takes the smaller of the per-snapshot limit and
store capacity before capture, at prompt and tool-call boundaries. A snapshot
that cannot fit no longer allocates a cache copy before rejection. Admitted
explicit overrides and LRU accounting are unchanged.

Two bounded tiny-CPU cases used a synthetic one-byte excess (no oversized
allocation). Initial **2 failed** (0.22 s); repaired cache/serving focus
**12 passed** (3.16 s); root **632 passed, 3 skipped** (28.25 s). Ruff, changed
formatting, whitespace and source diagnostics passed. Real controller probes
still lack delegation; this is not a measured serving-memory or latency result.

Serving evidence writes now validate row metadata without decoding the logit
payload they just encoded. Loading still validates the payload before
reconstructing tensors; schema v1 and bitwise round trips are unchanged.
One three-value strided CPU fixture reproduced **1 failure** (0.08 s); the
repaired evidence suite had **59 passes** (0.13 s), and root had
**633 passed, 3 skipped** (22.28 s). Ruff, scoped formatting, whitespace and
source diagnostics passed. No dependencies, downloads, model or host changes.
