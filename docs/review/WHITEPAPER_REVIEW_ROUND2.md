# WHITEPAPER.md — Second-round review (2026-11-11)

Subject: [WHITEPAPER.md](../../WHITEPAPER.md). This round:
1. **Re-verifies** the round-1 findings ([WHITEPAPER_REVIEW.md](WHITEPAPER_REVIEW.md)): re-ran the
   numerical audit and re-fetched the primary sources.
2. **Checks claims not covered in round 1.**
3. **Re-reads the whitepaper against the clarified goal:** the model compiles programs in its weights,
   and a bridge runs them on the real CPU.
4. **Corrects my own round-1 review** where needed.

The recommended direction is in [PROJECT_DIRECTION.md](PROJECT_DIRECTION.md).

---

## 1. Re-verification of round-1 findings

| Round-1 item | How re-checked | Result |
|---|---|---|
| M1–M11 (subtractor, precision limits, OR staging, snap, parabolic keys, SiLU/bf16 multiplier, ADC size, softmax leakage, RAM cost) | Re-ran `verify_whitepaper.py` on CPU | ✅ Identical numbers (e.g. 64,255/65,536 wrong; fp32 exact to N≤23; bf16 255·255=65,024; parabolic keys fail at 100 in bf16, 5,000 in fp32) |
| F6: Qwenputer chip tokens already skip Qwen's layers | `qwenputer/experimental/riscv/docs/rv32.md` line 34 | ✅ "Chip tokens run only the chip layers and are invisible to Qwen's layers." |
| F7: tokens per instruction and instr/s | rv32.md lines 308–316, 468, 489 | ✅ 1.0–1.55 tok/instr; executor v2 0.177 ms/token = 3,699 instr/s at batch 1 (L4); 3.6k instr/s compact executor. README: ~24,000 trace tokens/s. |
| F5: KL is not 0.0 | Qwenputer README lines 41, 186 | ✅ KL 6e-4 (noise floor 5e-4); MMLU 71.9 vs 71.7 base, KL 7e-4 |
| F1/F9: Loom configurations | Loom arXiv HTML | ✅ 146×512 (Sudoku), 155×1024 (default), 164×2048. With N=8, d = 9ℓ+8N+1 at ℓ = 9/10/11. The blueprint's d_vm=164 is Loom's N=8, n=2048 configuration. |
| A7: native is ≈10⁴× faster | Qwenx86 model card, plus local `tcc` measurement | ✅ and now quantified (N4 below): 71.2 s vs 1.7 ms (`tcc -run`) |

No round-1 number had to be changed. One qualitative statement was wrong (§3, C1).

---

## 2. New findings

| # | Section | Finding | Sev |
|---|---|---|---|
| N1 | §1 table, §1.2 | **The WASM Interpreter Transformer is over-described.**<br>• It has no `call`/`return`, so "stack operands across arbitrary call depths" is false.<br>• It supports 25 **i32-only** opcodes, loops of **≤256 iterations**, and 4 file descriptors × 32 bytes.<br>• Its positional encodings are **program-specific, computed by a compile-time analysis pass**: instruction indices, local-variable source locations and memory-address mappings. Part of the work is precomputed on the host.<br>• Its FFN is "SwiGLU (**ReLU gate**)", which independently confirms round-1 M5: exact products need a ReLU gate, not Qwen's SiLU. | 🟠 |
| N2 | §1 table, §1.2 | **Percepta.**<br>• The README confirms ~30K tok/s (C++ engine, O(log n) hull cache), 35 opcodes and call-depth tracking.<br>• Its FFN primitive is `reglu(a,b)=ReLU(b)·a`, again a ReLU gate.<br>• The whitepaper's "7 layers, d=36, 18 heads" could **not** be found on the primary README (UNVERIFIED).<br>• Not mentioned in the whitepaper: Percepta's "first Futamura projection" mode, which bakes one program into the FFN weights. It is the only existing mechanism that literally "compiles a program into weights". | 🟡 |
| N3 | §5.1 | The Sudoku result (1,085 → 284 instructions, −73.8 %) is quoted correctly but presented as a property of "our 8-layer looped core". It is **Loom's** measured result for its STORE opcode. | 🟡 |
| N4 | §1, §6.2 Tier 1 | **The cost of in-weights compilation is left out.**<br>• The whitepaper quotes Qwenx86's "169k trace tokens in 71.2 s" as throughput. That run is the **entire compile+run of a C program printing 6*7**: 169,293 machine tokens, 71.2 s.<br>• Host: `tcc -run` 1.7 ms; wasmtime compile ≈0.7 ms → **≈4×10⁴–10⁵× slower**.<br>• Qwenx86's own accuracy check: GSM8K first 28 questions, 26/28 with calc vs 23/28 plain. That is a tool-use gain of ordinary size, at that cost. | 🟠 |
| N5 | §2.2 | **LoopKV, precisely** (arXiv 2611.04137):<br>• It picks 4-bit, 2-bit or reuse of the final loop's KV *per (loop, layer) cell*. It is not "early iterations to 2-bit".<br>• It stores **35 % less KV than 4-bit KIVI** (storage, not "memory traffic").<br>• It gives **1.7–2.2× goodput vs KIVI-4** under a latency SLO (Ouro-1.4B), not vs BF16. | 🟡 |
| N6 | §1.1 | **Qwenx86 itself moved toward a host bridge.** Its v2/v3 memory is driven by host Python ("Python carries out each call and hands the result back"), and the in-weights tools are invoked through ordinary `<tool_call>`. | 🟡 (context) |
| N7 | §2.1 | "A single C program execution spans 50,000 to 358,500 trace tokens". 169,293 matches Qwenx86's qcc demo; **358,500 has no source**. | 🟡 |
| N8 | §3.5 (3) | **SumAttn is a non-standard attention mode.**<br>• "Removing the softmax denominator" is the WASM interpreter's custom "sum-mode". Percepta uses averaging attention plus rescaling.<br>• Stock Qwen kernels have neither.<br>• bf16 prefix sums are exact only up to 256. | 🟡 |
| N9 | §8 refs 9–10 | The ResearchGate links return HTTP 403 to automated fetches and cannot be verified. Use arXiv 2605.22791 and 2609.17184. | 🟡 |
| N10 | §6.1 | **Under the clarified goal, the ISA table uses the wrong criteria.** "Neural decode cost" and "loop depth footprint" matter only for in-weights execution.<br>• For CPU execution the criteria are: model emission accuracy, tokens per program, cheap validation (safety), lowering speed and ecosystem.<br>• On those, Wasm ≫ the quad-tuple ISA.<br>• Register ISAs (RV32, eBPF, x86) are the hardest for an LLM to emit: registers, offsets, ABI.<br>• Measured (PROJECT_DIRECTION E1/E3, zero-shot, n=12): WAT, asm and hex 0/12; Python 10/12; C 10/12 with gcc (8/12 strict: clang C99 + sanitizers). | 🟠 |
| N11 | §7.2 | **Under the clarified goal, the bridge idea is right but the mechanism solves a non-problem.**<br>• Sampled token IDs already reach the host every decode step (8 bytes/token); a program is 0.3–3 KB.<br>• Measured costs: a warm jailed executor 0.07 ms; wasm compile ≈0.7 ms; returning the result ≈1.2–2 decode steps (live-cache prefill).<br>• Zero-copy pinned memory saves µs (polled-flag round trip ≈1 µs per NVIDIA GDRCopy) against ms-scale actions: tcc ≈1.6 ms, clang/gcc 28–76 ms for a snippet, decode 2.5–137 ms/token.<br>• The design lever that actually matters is **not recomputing the context**: live-cache resume costs 0.10–0.12 s vs 1.5 s for a full re-prefill at 4k context here (InferCept: recompute = 37–40 % of forward time). Hybrid prefix caching is the other option. | 🟠 |
| N12 | §7.2 | **Security, with evidence.**<br>• LLM-app code-exec flaws are a known CVE class: LangChain LLMMathChain/PALChain 9.8, llama_index 9.8, smolagents sandbox escape 10.0, Terrarium Pyodide→host root.<br>• Wasm runtimes had sandbox-escape Criticals in 2026: Wasmtime CVE-2026-34987 (Winch) and others.<br>• Giving model code `POSIX_*`, `LLVM_JIT_COMPILE` and physical buffer descriptors (round-1 A6) is therefore not a theoretical risk. Any bridge needs an out-of-process jail plus Wasm SFI, size/time/memory caps, and fast patching. | 🔴 (confirms A6) |
| N13 | README.md | **The README repeats claims the reviews disproved:** a "Mathematically Consistent Hybrid ALU" (M1–M5), snapping that "guarantee[s] zero bit drift" (M6, M9), and "architectural audit corrections".<br>It also describes the NeuroVM direction the project no longer pursues. Rewrite it when the repo is restructured. | 🟡 |

---

## 3. Corrections and updates to my round-1 review

| # | Round-1 text | Correction |
|---|---|---|
| C1 | A6: "Triton has no first-class `__nanosleep` or system-scope polling" | **Partly wrong.**<br>• Triton 3.6 has system-scope atomics (`tl.atomic_cas/atomic_add(..., sem=..., scope="sys")`) and `tl.load(..., volatile=True)`; `nanosleep` is reachable through `tl.inline_asm_elementwise`.<br>• The real obstacles are persistent-kernel occupancy, watchdog/TDR on display GPUs, and memory-ordering rules for mapped flags.<br>• The conclusion (A6 is unsafe and unnecessary) stands. |
| C2 | M2: "TF32 … (a PyTorch default on some paths)" | Clarification: TF32 is **off by default for matmuls** in PyTorch ≥1.12, but on for cuDNN convolutions, and it is commonly enabled for speed. The finding (TF32 limits exact N to ≈9) is unchanged. |
| C3 | §5 "Pick one" value propositions | The user chose a goal **not on that list**: LLM-as-compiler with native CPU execution. The round-1 roadmap (Loom reproduction P0–P6) is **superseded**; see PROJECT_DIRECTION §6. |
| C4 | M5 | **Strengthened.** Both analytic WASM transformers (N1, N2) use ReLU-gated GLUs for exact products. |

---

## 4. The whitepaper under the clarified goal

| Whitepaper part | Status for "compile in weights, execute on CPU" |
|---|---|
| §3–§5, §7.1: looped neural VM, Loom ALU, snap, GDN-2 RAM | **Not needed.** Execution moves to the CPU. Keep only as background reading. |
| §6.2 Tier 1: in-weights `qcc` | **Drop.** ≈4×10⁴–10⁵× slower than host compilation (N4). |
| §6.2 Tier 2: host Clang/LLVM | **Keep, re-framed.** The host compiles what the model writes: C→wasm, or tcc in a jail. |
| §6.1: IR choice | **Keep the question, change the criteria** (N10). Answer: Python/C for the product; a Wasm dialect for the research track. |
| §7.2: hardware bridge | **Keep the idea, replace the mechanism.** Token-stream monitor, pre-warmed jailed executors, live-cache resume (N11–N12). |
| Abstract: "eliminates … security boundaries" | **Invert.** The bridge *is* the security boundary. |

## 5. Updated verdict

- **The whitepaper as a plan:** do not implement it. It is mathematically broken in places (round 1),
  and its central machinery is unnecessary for the stated goal.
- **What carries over:** two ideas.
  - The model should "compile" inside its own weights.
  - A bridge should connect that output to real hardware.
- **How to implement them:**
  - the model emits compact programs;
  - a sandboxed, pre-warmed bridge on the host lowers and runs them;
  - the result is returned into the live cache;
  - fine-tuning and RL teach the model to use it.

## Appendix A — sources checked in this round

- **Analytic transformers:**
  - WASM Interpreter Transformer model card (huggingface.co/eastlondoner/wasm-interpreter-transformer).
  - Percepta `transformer-vm` README.
- **Qwen-based neural computers:**
  - Qwenx86-4B-compact model card and README (lines 176, 236).
  - Qwenputer README and `rv32.md`.
  - Qwen3.5-2B config: 24 layers, which matches qwenGPU's 45+24.
- **Loom and LoopKV:** Loom arXiv HTML; LoopKV arXiv 2611.04137.
- **Link status:** Chipformer and Giannou et al. links resolve (HTTP 200); ResearchGate returns 403.
- **Literature sweeps:** neural compilation, tool-integrated reasoning and serving, bridge and sandbox
  technology. The key items and numbers are cited in PROJECT_DIRECTION §2. Primary IDs include:
  - 2407.02524, 2108.07639, 2505.20356, 2511.04132, 2505.11480, 2506.14606, 2504.09246, 2403.01632,
    2411.15100;
  - 2211.10435, 2309.17452, 2504.11536, 2503.23383, 2509.02479, 2508.20722, 2510.11701, 2605.06326,
    2402.01869, 2406.00059;
  - Wasmtime docs and advisories, NVD CVE entries, NVIDIA GDRCopy and CUDA docs.
