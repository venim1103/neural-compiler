# Architecture and Mathematical Foundations of Neural Compilation: LLM-as-Compiler with a Native CPU Execution Bridge and Audited Weight-Compiled Virtual Machines

## Abstract

A central question in neurosymbolic systems is how a language model can compile algorithmic intent within its own weights and execute the resulting computation with high speed, determinism, and safety. Prior work has pursued two distinct interpretations of "in-weights compilation and execution":
1. **Weight-Compiled Virtual Machines (`Qwenputer`, `Qwenx86`, `qwenGPU`, `Loom`, `WASM Interpreter`, `Percepta`):** Digital processors, arithmetic logic units (ALUs), and software compilers (`qcc`) are analytically compiled into transformer weights so that forward passes emulate hardware clock cycles.
2. **LLM-as-Compiler with a Native Hardware Execution Bridge:** The language model acts as the compiler from natural-language/algorithmic intent to a compact executable program (or intermediate representation), while a trusted host bridge lowers, validates, sandboxes, and executes that program natively on the physical CPU, injecting the result back into the model's live KV and recurrent state cache.

This whitepaper presents a two-part synthesis grounded in empirical measurements on `Qwen/Qwen3.5-4B` (32 layers, 24 Gated DeltaNet + 8 full-attention layers) and a rigorous numerical audit (`docs/review/verify_whitepaper.py`):
- **Part I (Primary System Architecture — LLM-as-Compiler + Native CPU Bridge):** We demonstrate experimentally (`E1`–`E3b`) why running a software compiler or emulator inside neural weights is $4\times 10^4\text{–}10^5\times$ slower than native host execution (`Qwenx86`'s in-weights `qcc` takes 169,293 machine trace tokens and 71.2 s to compile and run `6 * 7`, whereas host `tcc -run` takes 1.7 ms and `wasmtime` Cranelift compilation takes 0.64–0.75 ms). We detail our production architecture: multi-tier target emission (Python default at $1.00\times$ token cost and 10/12 zero-shot accuracy; C fast path at $7\text{–}32\times$ faster loop execution; and a middle-level IR research track comparing a typed surface DSL against validator-constrained `LLM-WAT`), an out-of-process jailed execution bridge (`bubblewrap`/`nsjail` + `seccomp` + fuel-metered `wasmtime` with in-jail compilation), live KV + Gated DeltaNet cache resumption (saving $>1.3$ s per call at 4k context over stateless re-prefill), and a two-stage SFT + GRPO RL training pipeline.
- **Part II (Audited Reference Specification for Weight-Compiled Neural Virtual Machines):** For applications requiring self-contained weight-space execution or formal circuit verification, we provide the corrected mathematical foundations of depth-looped (`Loom`) and token-driven (`Qwenputer`, `Percepta`) neural computers—resolving critical errors in prior hybrid drafts regarding MSB-vs-LSB borrow-chain indexing, floating-point precision ceilings (`bf16`/`TF32`/`fp32`/`fp64`), `ReGLU` vs. `SwiGLU` multiplication, ADC neuron scaling, continuous restoring clamps, parabolic addressing limits, and Gated DeltaNet memory banking.

---

**Implementation status (2026-11-12):** Part I describes the target architecture, not a
production implementation. The repository now contains a supervised, per-request execution
foundation with in-jail C/assembly compilation and isolated Wasm/unicorn harnesses. Native tool
turns, seccomp/cgroup hardening, warm pooling, and serving integration remain unfinished.
Historical warm/in-process microbenchmarks exclude these isolation costs; the six-task E3 bridge
used markdown fences. See [the implementation audit](./docs/review/PROJECT_DIRECTION.md#implementation-audit-2026-11-12).

## 1. Taxonomy of "In-Weights Compilation" and Precedent Systems

### 1.1 Four Meanings of Compiling Within Model Weights

| Paradigm | Representative Systems | Empirical Profile | Architectural Verdict |
| :--- | :--- | :--- | :--- |
| **(A) Running a compiler binary inside neural VM weights** | `Qwenx86` (`qcc` firmware in GDN ROM) | Compiling and running a C program printing `6 * 7` takes **169,293 machine trace tokens (71.2 s)**. Host `tcc -run` takes **1.7 ms**; `wasmtime` compiles in **0.64–0.75 ms**. | **$4\times 10^4\text{–}10^5\times$ slower** than host silicon; no latency or accuracy upside when a CPU is present. |
| **(B) The LLM *is* the compiler from intent to executable program** | Tool-Integrated Reasoning (`PAL`, `ToRA`, `ReTool`, `rStar2`), **Primary `neural-compiler` Path** | Real interpreter vs. LM self-execution raises GSM8K from 23.2% to 72.0% (`PAL`). In our `E3` Part C benchmark, `Qwen3.5-4B` achieved **6/6 correct final answers** with host execution taking **0.13%** of generation time. | **Primary production path.** Maximizes accuracy and minimizes wall time per correct answer. |
| **(C) Specializing a single program into FFN weights** | `Percepta` ("First Futamura Projection" mode) | Specializes WebAssembly bytecode directly into feed-forward parameters for fast in-network trace streaming (>30k tok/s on CPU). | Useful only when execution must remain inside neural forward passes. |
| **(D) Parallel/blockwise emission of IR/bytecode from hidden states** | Multi-Token Prediction (MTP) heads (`Qwen3.5` ships 1 MTP layer) | Reduces sequential decode steps via speculative drafting verified by the base model. | Orthogonal decode accelerator; compatible with Paradigm (B). |

### 1.2 Verified Comparison of Weight-Compiled Neural Computers

A primary-source audit of existing weight-compiled execution engines establishes their exact layer topologies, numerical constraints, and measured throughputs:

| System | Base Model / Architecture | Target ISA & Scope | Topology & Head Layout | Memory Substrate | Primary-Source Verified Throughput & Fidelity |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwenputer** | `Qwen3.5-4B` (32 native layers, $d=2560$, vocab 248,320) | 32-bit RISC-V (`RV32IM` + `Zba` + Bignum unit) | 11 prepended chip layers (~1,400 extra residual dims); chip tokens **skip Qwen's 32 layers** | Gated DeltaNet state ($\alpha=1, \beta=1$, key dim 128) | **1.0–1.55 trace tok/instr**; **3,699 instr/s** at B=1 (13.9k at B=32; ~24k tok/s in latest executor) on L4; text **KL = $6.1\times 10^{-4}$** (noise floor $5\times 10^{-4}$) |
| **Qwenx86** | `Qwen3.5-4B` | 64-bit x86 (MSVC integer subset + flag $\mu$-ops) + `qcc` | 13 prepended chip layers; v2/v3 offload memory & calls to a **host Python bridge** | 64 GDN RAM heads (64 KB RAM, 256 KB ROM, 128 KB rodata) | 169,293 machine tokens in **71.2 s** (~2,377 tok/s) for `qcc` compile+run of `6*7`; GSM8K-28: 26/28 vs. 23/28 base |
| **qwenGPU** | `Qwen3.5-2B` (24 native layers) | 4-lane SIMT GPU (NVIDIA PTX subset, predicates, barriers) | 45 prepended chip layers (69 layers total) | Gated DeltaNet state (15 regs/lane, shared RAM, ROM) | 10–15 trace tokens/s (Apple Silicon MPS); text **KL $\approx 7\times 10^{-4}$** |
| **Loom** | Standalone Looped Transformer (`arXiv:2604.08816`) | 22-opcode single-cycle ISA (`SUBLEQ` + 21 extended ops) | **8 layers** iterated cyclically; active heads per layer: **$[1, 3, 2, 0, 2, 0, 0, 0]$** | Bipolar matrix $X \in \{-1, +1\}^{d \times n}$, **$N=8$-bit words**, $m \in [64, 224]$ data words | 1 instruction per 8-layer pass ($d=146, 155, 164$ at $n=512, 1024, 2048$); paper validates up to 1,000 steps (README: millions with **argmax**); Alveo U200 FPGA |
| **WASM Interpreter** | Standalone Autoregressive Transformer | WebAssembly subset (25 **`i32`-only** opcodes, no `call`/`return`, loops $\le 256$) | 8 heterogeneous layers ($d=100$, 30 heads: $[13, 1, 0, 8, 2, 1, 1, 4]$), **`ReGLU` FFN** | Causal `SumAttn` + parabolic keys + **host-precomputed** program positional encodings | 112/112 compliance tests on its supported subset |
| **Percepta WASM** | Standalone Autoregressive (`Percepta-Core/transformer-vm`) | WebAssembly stack VM (35 opcodes, call-depth tracking, Futamura projection) | 2D attention heads with **`HullKVCache` ($O(\log t)$ lookup)** + **`ReGLU` FFN** (`ReLU(b)*a`) | Token-embedded execution history + convex-hull KV cache | **>30,000 trace tokens/s** (C++ CPU streaming engine) |

---

## Part I: LLM-as-Compiler + Native CPU Execution Bridge

### 2. Empirical Analysis of Representations, Bridge Latency, and Cache Resumption

To ground architectural decisions in measured hardware performance rather than speculation, four experiments (`E1`, `E2`, `E3`, `E3b` in `docs/review/experiments/`) were executed on `Qwen/Qwen3.5-4B` across 12 representative algorithmic and number-theoretic tasks (`primes_sum`, `collatz`, `lcm_1_25`, `fib_mod`, `divisors`, `binomial`, `digits_2_100`, `modpow`, `lcg_median`, `double_palindromes`, `lattice_paths`, `vowels`).

#### 2.1 Experiment E1: Token Cost Across Target Representations
Every token emitted by a 4B model requires a full forward pass reading ~8 GB of `bf16` weights from GPU HBM ($\ge 2.5$ ms/token on an H100; 63–137 ms/token on an RTX A3000). Token count is therefore the dominant multiplier on total wall-clock latency.

| Emitted Representation | Total Tokens (12 Tasks, Qwen3.5 Tokenizer) | Relative to Idiomatic Python | Relative to Full C Program |
| :--- | ---: | ---: | ---: |
| **Python, idiomatic** (`math`, stdlib, native bignum) | **530** | **1.00×** | 0.32× |
| **Python, plain loops** (no stdlib / no bignum) | 1,036 | 1.95× | 0.62× |
| **x86-64 machine code, raw bytes** (1 tok/byte, unlinked `.text+.rodata+.data`) | 1,288 | 2.43× | 0.77× |
| **C, full program** (`main` + `printf`, freestanding body) | 1,673 | 3.16× | **1.00×** |
| **WAT with 1-token-per-opcode vocabulary** (estimate) | 1,844 | 3.48× | 1.10× |
| **WebAssembly binary**, raw bytes (1 tok/byte) | 2,246 | 4.24× | 1.34× |
| **x86-64 machine code as hex text** (unlinked object) | 2,482 | 4.68× | 1.48× |
| **x86-64 assembly text** (`gcc -O2`) | 3,676 | 6.94× | 2.20× |
| **WAT text** (`clang --target=wasm32 -Oz` + `wasm2wat`) | 6,281 | 11.85× | 3.75× |

**Key Insight:** Even compiler-optimized raw x86-64 machine code requires **2.43× more tokens** than idiomatic Python (and Wasm binary requires **4.24× more**). Idiomatic Python's brevity comes from high-level primitives (`math.comb`, `math.lcm`, modular `pow`, arbitrary-precision integers). Consequently, **token efficiency is determined by abstraction level and available host intrinsics**, not by binary versus text encoding.

#### 2.2 Experiment E2: Host Bridge Compilation and Execution Latency

| Host Bridge Operation (Medians Across 12 Tasks) | Measured Latency | Comparison to 1 LLM Decode Step |
| :--- | ---: | :--- |
| **Warm Python worker inside `bubblewrap` jail** (trivial program) | **0.07 ms** (0.21 ms unjailed) | $< 0.03\times$ of an H100 decode step |
| **`wasmtime` instantiate / instantiate + run** (with fuel metering) | **12 µs / 54 µs** (max 4.3 ms) | $< 0.02\times$ of an H100 decode step |
| **`wasmtime` Cranelift compile** (from binary / from WAT text) | **0.64 ms / 0.75 ms** | $\approx 0.3\times$ of an H100 decode step |
| **`tcc` C compile-only / `tcc -run`** | **1.6 ms / 1.7 ms** | $\approx 0.7\times$ of an H100 decode step |
| **Loop-heavy task (`double_palindromes`):** Native C / Wasm / CPython | **1.5 ms / 4.3 ms / 30–75 ms** | C/Wasm is **7–32× faster** than CPython on tight loops |
| **Cold process spawn:** `python3 -c` / cold `bubblewrap` | 10 ms / 16 ms | Eliminated by pre-warmed worker pool |
| **Heavy optimizing compile:** `gcc -O2` / `clang --target=wasm32 -O2` | 31.4 ms / 45.8 ms | Use `tcc` (1.6 ms) for fast C or Cranelift (0.7 ms) for Wasm |

**Key Insight:** With pre-warmed workers, `tcc`, or `wasmtime`, host compilation and execution take **0.07 ms to 1.7 ms**—less than the time required to decode a single token. Furthermore, sampled token IDs (`8 bytes/token`) already reach host CPU memory at every decode step; custom GPU-to-CPU pinned-memory (`cudaHostAllocMapped`) polling saves at most a few microseconds against millisecond-scale decode steps while introducing severe security and GPU occupancy problems.

#### 2.3 Experiment E3: Zero-Shot Model Emission and In-Decode-Loop Bridge (`Qwen3.5-4B`)

| Target Representation | Zero-Shot Accuracy ($n=12$, Greedy) | Mean Emitted Tokens | Observed Failure Modes |
| :--- | :--- | ---: | :--- |
| **Python 3** | **10 / 12** (83.3%) | **70** | 2 algorithmic logic errors (e.g., printed prime count instead of sum) |
| **C (C99)** | **10 / 12** (`gcc`) · **9 / 12** (`clang`/`tcc`) · **8 / 12** (strict C99 + ASan/UBSan) | **257** | 1 `int` overflow, 1 token cap truncation, 1 GCC nested function extension, 1 ASan out-of-bounds write |
| **WebAssembly Text (WAT)** | **0 / 12** (0.0%) | 852 | Hallucinated opcodes (`i32.sqrt`), `i32`/`i64` stack-type mismatches, infinite `(local $tempN ...)` loops |
| **x86-64 Assembly (GAS)** | **0 / 12** (0.0%) | 653 | Mixed NASM/GAS directives, operand-size mismatches, undefined labels |
| **Raw x86-64 Machine Code Hex** | **0 / 12** (0.0%) | 600 (cap) | 7 invalid hex strings; 5 segfaulted on unmapped addresses / bad `rel32` offsets |

In **Part C (In-Decode-Loop Bridge)** across 6 tasks (`double_palindromes`, `lcg_median`, `collatz`, `digits_2_100`, `lattice_paths`, `modpow`), `Qwen3.5-4B` emitted Python code, paused decoding at the closing fence, executed the program in the warm jailed worker, prefilled the output into the **same live KV + Gated DeltaNet cache**, and resumed decoding:
- **6 / 6 final answers were correct** (5 / 6 programs exact; 1 LCG off-by-one index happened to share the same median).
- Total host execution time across all 6 tasks was **180 ms** versus **138.7 s** of token generation (**0.13% overhead**).

#### 2.4 Experiment E3b: Return-Path Prefill Cost (Live Hybrid Cache vs. Full Re-Prefill)
`Qwen3.5-4B` is a hybrid architecture containing **24 Gated DeltaNet layers and 8 full-attention layers**. Measuring prefill latency with warm kernels and `logits_to_keep=1` (computing logits only for the final token, as in production serving) reveals why stateful cache continuation is critical:

| Context Length ($N$) | 1 Decode Step | Incremental Prefill of 16–64 Result Tokens into Live Cache | Stateless Full-Context Re-Prefill ($N + k$ Tokens, No Prefix Cache) |
| ---: | ---: | ---: | ---: |
| **256 tokens** | 63 ms | **78–81 ms** ($\approx 1.25\times$ decode) | 157–179 ms ($2.2\times$ slower) |
| **1,024 tokens** | 66 ms | **82–84 ms** ($\approx 1.25\times$ decode) | 440–449 ms ($5.3\times$ slower) |
| **4,096 tokens** | 63 ms | **99–123 ms** ($\approx 1.7\times$ decode) | **1,489–1,514 ms** (**$12\text{–}15\times$ slower**) |

Because Gated DeltaNet recurrent states $S_t$ cannot be arbitrarily sliced backward like standard softmax KV caches, stateless tool-calling without hybrid state checkpointing forces full sequence recomputation (which consumes 37–40% of forward time in serving benchmarks such as InferCept, `arXiv:2402.01869`). Resuming directly from the live in-memory KV + GDN state (or using hybrid prefix-cache checkpoints at tool-call boundaries, as in SGLang's `extra_buffer` mode and vLLM's hybrid prefix caching) reduces result-injection overhead to **1.2–2 decode steps**.

---

### 3. Production System Architecture: The `neural-compiler` Bridge

```
 GPU: Inference Engine (Qwen3.5-4B, Fine-Tuned)         Host CPU: Hardened Execution Bridge (Out-of-Process)
┌──────────────────────────────────────────────┐  IDs  ┌────────────────────────────────────────────────────┐
│ ... <tool_call>  (single token ID 248058)    │──────►│ Per-Sequence Token Stream Monitor                  │
│ <function=run><parameter=code> program ...   │ 8 B / │  • Detects token 248058..248059 in O(1)            │
│ </tool_call>     (single token ID 248059)    │ token │  • Optional XGrammar / validator mask on logits    │
│   └─► Pause THIS sequence (batch continues)  │       ├────────────────────────────────────────────────────┤
│                                              │       │ Pre-Warmed Jailed Executor Pool (Fork-per-Request) │
│ Prefill turn boundary + <tool_response>...   │◄──────│  • Python: CPython in bwrap/nsjail + seccomp       │
│ (tokens 248066..248067) into SAME live       │  IDs  │  • C: tcc or clang->wasm32 COMPILED INSIDE JAIL    │
│ KV + Gated DeltaNet state & resume decoding  │       │  • Wasm / DSL: wasmtime + Cranelift (fuel + epoch) │
└──────────────────────────────────────────────┘       │  • Dedicated control pipe + hard kill/reap timer   │
                                                       └────────────────────────────────────────────────────┘
```

#### 3.1 Native Protocol and Turn Boundaries
Instead of ad-hoc markdown fences, the bridge uses `Qwen3.5`'s native tool tokens, which are single vocabulary IDs: `<tool_call>` (`248058`), `</tool_call>` (`248059`), `<tool_response>` (`248066`), and `</tool_response>` (`248067`). When the stream monitor observes `248059`, it pauses the sequence, executes the payload, and prefills the complete chat-template turn transition (`<|im_end|>\n<|im_start|>user\n<tool_response>\n...\n</tool_response><|im_end|>\n<|im_start|>assistant\n`) directly into the live cache.

#### 3.2 Multi-Tier Language & Intermediate Representation Strategy
1. **Tier 1 — Curated Python (Default):** Provides arbitrary-precision arithmetic, `math`, `fractions`, `itertools`, and `collections` at $1.0\times$ token cost. Ideal for symbolic math, combinatorics, and rapid prototyping.
2. **Tier 2 — Jailed C (`tcc` / `clang --target=wasm32`):** Used when algorithmic time complexity requires native loop speed ($7\text{–}32\times$ faster than CPython on nested loops).
3. **Research Track — The "Middle-Level" IR (`R1` vs. `R2`):**
   - **R1 (Compact Typed Surface DSL + Host Intrinsics):** A statically typed Pythonic/AST subset with built-in bignum and math intrinsics that lowers to native code or Wasm via Cranelift in $<2$ ms—combining Python's low token count with native loop speed and static safety.
   - **R2 (Validator-Constrained `LLM-WAT`):** Normalized WebAssembly text emitted under an incremental stack-and-type validator mask (XGrammar / SynCode style, $\mu\text{s/token}$ overhead), guaranteeing 100% parse-and-validate success by construction.

#### 3.3 Security Model and Sandbox Hardening
Giving LLM-generated code direct host access (`POSIX_READ/WRITE`, `LLVM_JIT_COMPILE`, or raw pointers) is a critical remote code execution (RCE) vulnerability, underscored by recent CVSS 9.8–10.0 sandbox escapes in LLM frameworks (LangChain, LlamaIndex, `smolagents`) and JIT compiler advisories (e.g., Wasmtime Winch `CVE-2026-34987`). In `neural-compiler`, **the host bridge is the security boundary**:
- **Defense-in-Depth Isolation:** Every execution runs out-of-process inside Linux user/PID/mount/network/IPC namespaces (`bubblewrap` / `nsjail`) with a strict `seccomp-bpf` syscall allowlist, `rlimits`/`cgroups` memory and PID caps, a read-only root filesystem, and an empty `tmpfs` scratch directory.
- **In-Jail Compilation:** C and assembly are **compiled inside the jail**, preventing compile-time file exfiltration via `#include "/path/to/secret"` or `.incbin`.
- **Unreachable Control Channel & Hard Deadlines:** To prevent guest code from forging JSON status replies on `stdout` or hanging after printing a single byte, the supervisor forks per request (or per session from a pre-warmed template process), communicates over a dedicated control descriptor closed before `exec` (or after guest completion), and enforces an outer wall-clock deadline that sends `SIGKILL` to the entire process group and reaps all children.
- **Structured Outcomes & Optional REPL Persistence:** Returns structured `(status, payload)` tuples (`ok`, `compile_error`, `runtime_error`, `timeout`) with concise, line-localized diagnostics to enable self-correction, and supports per-session state retention so the model can define helper functions once and invoke them across turns.

---

### 4. Fine-Tuning Recipe (SFT + RL) and Phased Engineering Roadmap

Untrained 4B models under-utilize tools or lose accuracy when prompted naively (`Trice`, `arXiv:2605.06326`). Following the converged literature recipe (`ToRA` `arXiv:2309.17452`, `ReTool` `arXiv:2504.11536`, `ToRL` `arXiv:2503.23383`, `SimpleTIR` `arXiv:2509.02479`, `rStar2-Agent` `arXiv:2508.20722`), training proceeds in two stages:
1. **Supervised Cold-Start (SFT):** 3k–16k **real, bridge-executed** trajectories in `Qwen3.5`'s native `<tool_call>` / `<tool_response>` format across held-out task families with randomized parameters (preventing memorization and catching lucky-answering bugs), mixed with 10–30% general text data. Real executed trajectories substantially outperform stitched synthetic traces (`AIME24` 33.2% vs. 4.4%, `arXiv:2510.11701`).
2. **Reinforcement Learning (GRPO with Execution Outcome Reward):** Group Relative Policy Optimization using binary verification rewards, **masking `<tool_response>` tokens from the policy loss**, filtering void turns, and enforcing a 5–10 call budget under the exact production bridge timeouts.

#### Phased Roadmap and Go/No-Go Gates
The primary evaluation metric across all phases is **wall-clock time (and GPU-seconds) per correct answer** with confidence intervals on held-out task families:

| Phase | Deliverable | Quantitative Go/No-Go Gate |
| :--- | :--- | :--- |
| **P0** | **Production Bridge MVP & Serving Spike:** Port `experiments/` into `src/neural_compiler/bridge/` (`protocol`, `monitor`, `sandbox`, `executors/{python,c,wasm}`) and `src/neural_compiler/eval/` (~200 randomized tasks). Benchmark live pause/resume vs. hybrid prefix-cached resubmission. | Host bridge overhead $\le 1$ decode step at p95; result return $\le 2$ decode steps; zero duplicate-token or turn-boundary bugs. |
| **P1** | **Sandbox Hardening & Red-Team Suite:** `bwrap`/`nsjail` + `seccomp` + `cgroups`, in-jail C/Wasm compilation, unreachable control pipe, outer kill+reap timer. | **100% containment** against fork bombs, memory bombs, infinite loops, `#include`/`.incbin` exfiltration, `/proc` reads, and forged stdout replies. |
| **P2** | **SFT Cold Start (`Qwen3.5-0.8B/2B/4B`):** LoRA/QLoRA training on verified multi-turn execution trajectories, including compile/runtime error recovery. | Wall time per correct answer and accuracy beat both no-tool and zero-shot Python baselines (CI-separated); no text KL regression beyond noise. |
| **P3** | **Execution-Rewarded RL (GRPO):** Rollouts via vLLM/SGLang with live bridge execution and masked tool-output loss. | Statistically significant improvement over P2 on wall time per correct answer and multi-step error recovery. |
| **P4** | **Middle-Level IR Research Track (`R1` Typed DSL / `R2` Constrained `LLM-WAT`):** Matched SFT/RL budget + constrained decoding + repair rounds vs. Python/C tiers. | Must beat Python/C tiers on **wall time per correct answer** at equal or higher accuracy on at least one task class; otherwise archive. |
| **P5** | **Serving Engine Integration:** Production vLLM/SGLang plugin with `Qwen3.5` MTP speculative decoding and continuous batching. | p95 tool-call round-trip overhead $\le 2$ decode steps under concurrent batch load. |

---

## Part II: Audited Mathematical Reference for Weight-Compiled Neural Virtual Machines

When the engineering goal is instead a **self-contained neural artifact** (shipping a deterministic state machine or ISA interpreter entirely inside `.safetensors` weights for interpretability, formal verification, or FPGA synthesis), every circuit equation must respect floating-point mantissa bounds and piecewise-linear representability. Below is the mathematically verified specification correcting all issues identified in Round 1 (`M1`–`M14`, `F1`–`F15`, `A1`–`A8`) and Round 2 (`N1`–`N13`, `C1`–`C4`).

### 5. Corrected Combinational ALU and Precision Ceilings

#### 5.1 Direct Borrow-Chain Subtraction: MSB-First Indexing with LSB Borrow Summation
In `Loom` (`arXiv:2604.08816`), two $N$-bit signed two's-complement integers $a, b$ are represented as MSB-first bipolar vectors $a_i, b_i \in \{-1, +1\}$ for $i \in \{0, \dots, N-1\}$ ($+1 \equiv 1$, $-1 \equiv 0$). Because arithmetic borrows propagate from **less significant bits ($j > i$) into bit $i$**, the weighted partial difference $D_i$ must sum from $j = i$ down to the LSB $N - 1$:

$$D_i = \sum_{j=i}^{N-1} 2^{N-2-j} \left(a_j - b_j\right)$$

*(Audit note `M1`: Summing over $j < i$ instead of $j \ge i$ inverted the borrow direction and failed on 64,255 / 65,536 input pairs at $N=8$; Loom's $j \ge i$ summation above achieves **0 / 65,536 errors**.)*

With $P_i = 2^{N-1-i}$, a single two-layer ReLU feed-forward network ($\text{FFN}(x) = W_2\,\text{ReLU}(W_1 x + b_1) + b_2$) evaluates six thresholded pre-activations per bit:

$$z_{i,1} = D_i + P_i + 1, \quad z_{i,2} = D_i + P_i, \quad z_{i,3} = -D_i, \quad z_{i,4} = -D_i - 1, \quad z_{i,5} = D_i - P_i + 1, \quad z_{i,6} = D_i - P_i$$

$$R_i = 2 \Big[ \big(\text{ReLU}(z_{i,1}) - \text{ReLU}(z_{i,2})\big) + \big(\text{ReLU}(z_{i,3}) - \text{ReLU}(z_{i,4})\big) + \big(\text{ReLU}(z_{i,5}) - \text{ReLU}(z_{i,6})\big) \Big] - 3 \in \{-1, +1\}$$

#### 5.2 Exact Floating-Point Word-Width Limits ($N_{\max}$ per Data Type)
Because the pre-activations $z_{i,1}$ reach $\approx 3 \cdot 2^{N-1}$ while the step transition width is $1.0$, the unit offset $+1$ is lost to floating-point rounding as soon as $3 \cdot 2^{N-1}$ exceeds the format's exact integer mantissa ($2^{M+1}$):

| Floating-Point Format | Mantissa Bits ($M$) | Max Exact Integer | Single-Layer Subtractor Word Limit ($N_{\max}$) | Measured Failure Rate (`verify_whitepaper.py`) |
| :--- | ---: | ---: | :--- | :--- |
| **`bfloat16`** | 7 bits | $256$ | **$N \le 7$** (Fails at $N=8$) | $N=8$: **13.4% wrong**; $N=16$: **99.6% wrong** |
| **`TF32` (TensorCore)** | 10 bits | $2,048$ | **$N \le 9$** | Must disable TF32 (`torch.backends.cuda.matmul.allow_tf32 = False`) for $N > 9$ |
| **`float32`** | 23 bits | $16,777,216$ | **$N \le 23$** (Safe for $N=8, 16$) | $N=16$: **0% wrong**; $N=24$: **13.0% wrong**; $N=32$: **100% wrong** |
| **`float64`** | 52 bits | $9.0 \times 10^{15}$ | **$N \le 52$** (Safe for $N=32$) | $N=32$: **0% wrong**; $N=64$: **100% wrong** |

The rates above are historical measurements, not backend-independent constants. The
2026-11-12 CPU rerun produced 385/3004 failures at `bf16`, $N=8$, and 3004/3004 at
$N=16$. Boundary checks (1,003 seeded/adversarial pairs per width) found no failures at
`bf16` $N=7$ or `fp64` $N=52$, and 128 failures at each next width. The precision ceiling
is $N_{\max}=M$, since the largest pre-activation is $3\cdot2^{N-1}$.

**Architectural Consequence (`M2`, `A4`):** `Loom` uses **$N = 8$** in `fp32` ($d = 9\ell + 8N + 1 = 155$ at $\ell = 10, n = 1024$). Supporting 32-bit integers in `fp32` requires either multi-limb execution ($2 \times 16$-bit limbs with carry ripple across layers/cycles) or `fp64` weights.

#### 5.3 Bitwise Operations, Shifts, and Multiplication (`M3`–`M5`, `C4`)
- **Minuend Rewriting for `OR` and `XOR` (`M3`):** Because subtracting a positive correction can only clear bits ($+1 \to -1$), `OR` cannot be staged as a subtrahend adjustment (which fails on 56,788 / 65,536 pairs at $N=8$). Instead, Layer 2's FFN computes bitwise `OR` and `XOR` directly into the **minuend register (`scr_min`)** and sets the subtrahend **`scr_sub = 0`** so Layer 4's subtractor passes the result through unchanged.
- **Inversion and Gated Shifts (`M4`):** In bipolar encoding, bitwise `NOT` is a weight sign flip $\times(-1)$ requiring zero hidden units; opcode-gated 1-bit shifts require $2N$ (`SHL`) and $2N-1$ (`SHR`) gating neurons in Layer 2.
- **Why Single-Layer 32-Bit Multiplication Fails (`M5`, `C4`):**
  1. *Gate Activation:* `SwiGLU` uses $\text{SiLU}(x) = x\sigma(x)$, which is non-linear for positive inputs ($\text{SiLU}(1)\cdot 7 = 5.12 \neq 7$). Exact bilinear multiplication requires a **ReLU-gated GLU (`ReGLU`)**: $\text{ReGLU}(a, b) = \text{ReLU}(a) \odot b$, as used in both `wasm-interpreter-transformer` and `Percepta`.
  2. *Mantissa Overflow:* Even with 8-bit limb chunking ($255 \times 255 = 65,025$), `bf16` rounds $65,025 \to 65,024$; `fp32` is mandatory.
  3. *ADC Exponential Neuron Cost:* Accumulating four 8-bit partial products yields an 18-bit column sum. Decomposing an $K=18$-bit scalar back into bipolar bits inside a single 1-layer FFN requires a periodic step function with $\sum_{j=0}^{17} 4 \cdot 2^{17-j} = \mathbf{1,048,572}$ ReLU units (or 18 sequential layers). Therefore, compact looped VMs implement multiplication iteratively via the `MULACC` instruction across $N$ clock cycles.

---

### 6. Addressing, Error Correction, and Memory Topologies in Analytical VMs

#### 6.1 Continuous Restoring Clamp and Dual-Stage Snapping (`M6`, `M8`, `M9`)
When using softmax attention with inverse temperature $\lambda = 10$ over Hamming-margin-2 bipolar coordinates ($\text{Score}(j) = \ell - 2\,d_{\text{Hamming}}(p, \text{pos}_j)$), off-target columns leak up to $(n-1)e^{-2\lambda} \approx 2.1 \times 10^{-6}$ at $n=1024$. A discontinuous step function cannot be represented by a finite ReLU layer (`M6`). Instead, drift is erased using a **continuous piecewise-linear restoring clamp** with slope $k = \frac{1}{1 - \epsilon}$ ($\epsilon = 0.1 \implies k = \frac{10}{9}$):

$$\text{Clamp}_\epsilon(x) = \text{ReLU}(k x + 1) - \text{ReLU}(k x - 1) - 1 = \begin{cases} -1.0 & \text{if } x \le -(1 - \epsilon) \\ k x & \text{if } x \in (-(1 - \epsilon), 1 - \epsilon) \\ +1.0 & \text{if } x \ge 1 - \epsilon \end{cases}$$

In `Loom`'s verified 8-layer pipeline (head layout $[1, 3, 2, 0, 2, 0, 0, 0]$), error correction occurs at **two** stages (`M9`, `F2`):
- **Layer 3 (2 Heads):** Resolves indirect `LOAD`/`FIND` pointers and folds scratchpad clamping into its FFN.
- **Layer 5 (2 Heads):** Executes direct/indirect memory writes and dual-column `SWAP`.
- **Layer 8 (0 Heads, Dedicated Error-Correction FFN):** Applies the restoring clamp across all persistent memory and `PC` rows so that softmax writeback leakage from Layer 5 is eliminated before Layer 1 of the next cycle. (When run with exact **argmax** attention, drift is zero by construction.)

#### 6.2 Parabolic Addressing Limits vs. 2D Convex-Hull Lookup (`M7`, `F14`)
Parabolic key-query matching ($Q_i = [i, 1]^T, K_j = [2j, -j^2]^T \implies Q_i^T K_j = 2ij - j^2$) relies on a unit margin between $j = i$ ($i^2$) and $j = i \pm 1$ ($i^2 - 1$). Numerical testing confirms that floating-point rounding erases this unit margin at address **$i = 100$ in `bf16`** and **$i \approx 5,000$ in `fp32`**. For long execution traces, `Percepta` (`Percepta-Core/transformer-vm`) replaces 1D parabolic scaling with **2D attention heads backed by a convex-hull KV cache (`HullKVCache`)**, achieving exact $O(\log t)$ lookup across millions of steps.

#### 6.3 Banked Matrix RAM vs. Gated DeltaNet Fast Weights (`M10`, `M11`, `A1`, `A2`)
- **Token-Driven Co-Processors (`Qwenputer`):** Because `Qwenputer` runs inside a stock `Qwen3_5ForCausalLM` graph, it repurposes `Qwen3.5`'s native Gated DeltaNet heads ($\alpha_t = 1$, per-head write gate $\beta_t^{(h)} \in \{0, 1\}$, orthonormal basis keys $e_i \in \{0, 1\}^{128}$) as addressable RAM. Note that standard Gated DeltaNet (`arXiv:2412.06464`) already supports per-head gating via scalar $\beta_t^{(h)}$; `Gated DeltaNet-2` (`arXiv:2605.22791`) channel gates are unnecessary when gates are uniform across channels (`M10`).
- **Standalone Depth-Looped VMs (`Loom`):** Because a Loom layer uses non-causal masks, symmetric $Q=K$, ReLU FFNs, and per-column biases $b_1 \in \mathbb{R}^{r \times n}$ (which are 99.9% sparse—~8,000 nonzeros out of 4.7M entries at $n=1024$), it cannot execute inside stock causal RMSNorm+SiLU LLM layers without custom kernels (`A1`, `F4`). Inside a custom depth loop, external memory is most efficiently implemented as a **direct one-hot-addressed tensor bank** `RAM[bank, row] = val` rather than storing 1 bit per float in a $64 \times 256 \times 64$ GDN state (which consumes 4 MiB in `fp32` to store 128 KiB of bits, `M11`).

---

## 7. References

1. **Qwenputer:** *A deterministic 32-bit RISC-V CPU compiled into Qwen3.5-4B weights*, `divagr.com/blog/qwenputer` & `github.com/divagr18/qwenputer` (`experimental/riscv/docs/rv32.md`).
2. **Qwenx86:** *Qwenx86-4B-compact: 64-bit x86 CPU and `qcc` compiler in 13 prepended layers*, `huggingface.co/Quazim0t0/Qwenx86-4B-compact`.
3. **qwenGPU:** *4-lane SIMT PTX GPU compiled into 45 layers ahead of Qwen3.5-2B*, `emergent.sh/news/qwengpu-qwen3-5-2b-runs-cuda-45-compiled-gpu`.
4. **Loom:** Mehmet Kerem Turkcan et al., *Loom: A Scalable Analytical Neural Computer Architecture*, `arXiv:2604.08816` & `github.com/mkturkcan/Loom`.
5. **Percepta Transformer VM:** *WebAssembly execution via 2D attention heads, `HullKVCache`, and Futamura projection*, `github.com/Percepta-Core/transformer-vm`.
6. **WASM Interpreter Transformer:** *Analytical WebAssembly subset interpreter*, `huggingface.co/eastlondoner/wasm-interpreter-transformer`.
7. **Looped Transformers as Programmable Computers:** Angeliki Giannou et al., *ICML 2023*, `arXiv:2301.13196`.
8. **Gated Delta Networks:** Songlin Yang et al., *Improving Mamba2 with Delta Rule*, `arXiv:2412.06464`.
9. **Gated DeltaNet-2:** *Decoupling Erase and Write in Linear Attention*, `arXiv:2605.22791`.
10. **LoopKV & LoopSpec:** *Non-uniform KV cache compression and pipelined self-speculative decoding for looped transformers*, `arXiv:2611.04137` & `arXiv:2609.17184`.
11. **LoopCD:** *Training-free contrastive latent decoding for looped language models*, `arXiv:2610.02185`.
12. **Tool-Integrated Reasoning & SFT/RL Recipes:** `PAL` (`arXiv:2211.10435`), `ToRA` (`arXiv:2309.17452`), `InferCept` (`arXiv:2402.01869`), `ToRL` (`arXiv:2503.23383`), `ReTool` (`arXiv:2504.11536`), `rStar2-Agent` (`arXiv:2508.20722`), `SimpleTIR` (`arXiv:2509.02479`), `Trice` (`arXiv:2605.06326`).
13. **Neural Compilation & Constrained Decoding:** Armengol-Estapé et al. (`arXiv:2108.07639`), `SynCode` (`arXiv:2403.01632`), `XGrammar` (`arXiv:2411.15100`), Mündler et al. type-constrained decoding (`arXiv:2504.09246`), `SuperCoder` (`arXiv:2505.11480`), `LEGO-Compiler` (`arXiv:2505.20356`), `arXiv:2506.14606`, `arXiv:2511.04132`.
