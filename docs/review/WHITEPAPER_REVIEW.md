# WHITEPAPER.md — Technical Review (2026-11-11)

> **Update (round 2):**
> - The second-round review is in [WHITEPAPER_REVIEW_ROUND2.md](WHITEPAPER_REVIEW_ROUND2.md). It
>   re-verifies every number below and corrects one statement in A6 (C1).
> - The roadmap in §6 is superseded by the clarified goal in [PROJECT_DIRECTION.md](PROJECT_DIRECTION.md).

Scope: `WHITEPAPER.md`, `README.md`, and the current scaffold of `venim1103/neural-compiler`.
Method: (1) line-by-line read; (2) primary-source check of every cited system (Loom paper + repo,
Qwenputer repo + `rv32.md` design log, Qwenx86 model card, qwenGPU coverage, Qwen3.5-4B `config.json`,
Gated DeltaNet-2, LoopKV/LoopSpec/LoopCD, Percepta, WASM-interpreter-transformer); (3) numerical
verification of every equation (`verify_whitepaper.py`, same folder).

---

## 0. Verdict

The whitepaper is **not a buildable specification**. It is a mix of (a) Loom's published 8-bit
design, (b) Qwenputer/Qwenx86 results, and (c) new extensions — and the new parts are where the
errors are. The new parts are: the 32/64-bit words, the bf16 claims, the GDN-2 banks, the
SwiGLU multiplier, the fused SRAM loop and the MMIO bridge. Several core equations are wrong as
written. Several numeric claims about prior work are wrong. The central performance argument
(depth-looped beats token-autoregressive) is based on figures that the cited projects' own
measurements contradict.

The **underlying research direction is legitimate** (Loom, Qwenputer and Percepta all work), but
the project should restart from a **narrow, verified core**, not from this blueprint.

Severity legend: 🔴 blocks implementation / mathematically false · 🟠 design contradiction or
wrong fact about prior work · 🟡 imprecision, cosmetic or unverified.

---

## 1. Mathematical errors (verified numerically)

| # | Section | Claim | Finding | Sev |
|---|---|---|---|---|
| M1 | §3.2 | $D_i=\sum_{j=0}^{i-1}2^{N-2-j}(A_j-B_j)+\tfrac12(A_i-B_i)$ | **Wrong.** It sums the *more* significant bits, but borrows come from the *less* significant bits. Exhaustive N=8 test: wrong on **64,255 / 65,536** pairs. Loom's original $D_i=\sum_{j=i}^{N-1}2^{N-2-j}(a_j-b_j)$ is correct: **0 / 65,536** wrong. The threshold/ReLU part ($z_1..z_6$, $R_i$) is correct. | 🔴 |
| M2 | §3.2, §3.4, §7 | Single-layer subtractor for N=32/64, "exact in fp32/bf16" | **False.** The pre-activations reach about $3\cdot2^{N-1}$, and the "+1" step offsets vanish once values pass the float's integer-exact range. Measured: fp32 exact for **N ≤ 23**; N=24 fails 13%; N=32 fails **100%**. bf16 fails **13% even at N=8**. fp64 handles N=32 but not N=64. GPU matmuls with TF32 enabled (a PyTorch default on some paths) have a 10-bit mantissa, which leaves roughly N ≲ 9. Loom itself uses **N=8**. | 🔴 |
| M3 | §3.3 | OR/XOR "staged identically" to AND via the subtractor | AND as $A-(A\wedge\neg B)$ is correct (0 wrong). OR as $A-(\neg A\wedge B)$ is **wrong on 56,788 / 65,536**, because a subtraction can only clear bits. Loom does not do this: it rewrites the *minuend* to the bitwise result and sets `scr_sub = 0`. | 🔴 |
| M4 | §3.3 | Bit inversion for −B costs 2N units; shifts have "zero non-linear overhead" | In bipolar encoding NOT is ×(−1) and costs nothing. But **opcode-gated** selection of a shifted copy does need gating neurons (Loom: SHL 2N, SHR 2N−1 rows). Loom also only supports shift-by-1; variable shifts need a barrel shifter. | 🟡 |
| M5 | §3.4 | "Limb-chunked DAC→SwiGLU→ADC" exact multiplier in one layer | Three separate failures. (a) SwiGLU uses **SiLU**, and SiLU(u)·v ≠ u·v: for example SiLU(1)·7 = 5.12, not 7. Exactness needs a ReLU gate (ReGLU), which Qwen does not have. (b) **bf16** 255·255 = 65,024, not 65,025, yet the paper claims bf16 elsewhere. (c) The **ADC** is not specified, and it is the hard part. Each column sum needs 18 bits, and extracting bits from an 18-bit scalar in one ReLU layer takes about **10⁶ units** (periodic staircase); otherwise it needs ~18 sequential layers. Carries between limb columns add more depth. A 32×32 product does not fit in "Layer 4". | 🔴 |
| M6 | §5.2 | $\text{Snap}_\epsilon$: snap inside $[\pm1\pm\epsilon]$, "x otherwise" | The function as written is **discontinuous** (it jumps from 0.9 to 1.0 at 1−ε), so no finite ReLU network can implement it. It also leaves drift ≥ ε uncorrected. The correct form is a continuous restoring clamp, e.g. $\text{ReLU}(kx+1)-\text{ReLU}(kx-1)-1$ with $k=1/(1-\epsilon)$, plus residual compensation. | 🟠 |
| M7 | §3.5.2 | Parabolic keys give unique peak with margin ≥1 | Correct in exact arithmetic only. The scores grow like $i^2$, so the margin of 1 is lost to rounding: **bf16 fails at address 100**, **fp32 at about 5,000**. The paper's 14-bit (16,384-word) space is out of range. Percepta avoids this with 2-D heads plus a convex-hull cache. | 🟠 |
| M8 | §3.5.1 | Softmax "zero cross-talk" | With Loom's λ=10 the worst-case leaked mass is about 2e-6 at n=1024 and 3.4e-5 at n=16,384. That is small but non-zero, and it is why a snap is needed at all. Loom's README credits exact **argmax** attention, not the snap, for running "millions of steps without drift". | 🟡 |
| M9 | §5.2 / §7 | Snap folded into L3 removes the need for a separate correction layer | **Misreport.** Loom has *both* an L3 scratchpad snap *and* a dedicated **L8** persistent-state snap. The whitepaper's Layer 8 drops that duty, so drift from L5's write heads in memory/PC rows goes uncorrected until the next L3 and leaks into L4's large ALU weights. | 🟠 |
| M10 | §4.2 | GDN-2 channel gates are needed for banked RAM | **Not needed.** Standard Gated DeltaNet already has a **per-head β**; setting β=0 on non-target heads freezes them, so per-head bank selection works with plain GDN. Because the paper sets $b_t=w_t=\mathbb I(\cdot)\mathbf 1$ (scalar per head), GDN-2 reduces exactly to GDN here. | 🟡 |
| M11 | §4.2 | 64 banks × 256 rows × 64 bits = 128 KB | The logical size is right, but each bit is stored as one float: the state is **4 MiB fp32 / 2 MiB bf16** for 128 KiB of data. The GDN-2 equation also puts the gates on the wrong sides compared with the paper ($S_t=(I-k(b\odot k)^T)D_tS_{t-1}+k(w\odot v)^T$). | 🟡 |
| M12 | §3.1 | Looping gives Turing completeness | True only up to a finite bound, as stated. Fine. | ✅ |
| M13 | §3.5.1 | Hamming-score argmax, $\le\ell-2$ off-target | Correct. | ✅ |
| M14 | §3.4 (1) | DAC $u_m=\sum 2^{k-1}(L_{m,k}+1)$ | Correct, but it is indexed LSB-first, while the rest of the paper is MSB-first. | 🟡 |

---

## 2. Factual errors about prior work (checked against primary sources)

| # | Whitepaper says | Primary source says | Sev |
|---|---|---|---|
| F1 | Loom handles N=32/64 words; 320/640 hidden units | Loom: **N=8** (d=9ℓ+8N+1=155 at ℓ=10). Memory **m = 64–224 words**; 928 instruction slots by default. | 🟠 |
| F2 | Loom heads per layer: L2=2, L5=1, L7=1, L8 = "terminal snap & host bus" | Loom: L2=**3**, L5=**2** (needed for SWAP), L7=**0**, L8=**error correction**. | 🟡 |
| F3 | Loom: "zero bit divergence across millions of instructions" | The paper's validation runs go **up to 1,000 ISA steps**. The README claims millions of steps *with argmax attention*. | 🟡 |
| F4 | §2.2: VM weights 8.3–28 MB "fit in SM SRAM, no HBM" | 4.7M params ≈ 16 MB fp32 (Loom n=1024). Most of that is **per-column biases**, and **99.9% is zeros** (about 8,000 nonzeros). An SM has ≤ 228 KB of SRAM, so dense weights cannot sit in one SM's SRAM; sparse weights fit trivially. | 🟠 |
| F5 | Qwenputer/Qwenx86: "0.0 KL" on text | Qwenputer: **KL 6.1e-4** (noise floor 5e-4). qwenGPU: **KL ≈ 7e-4**. | 🟠 |
| F6 | §2.1 Bottleneck 1: every trace token runs all LLM layers + LM head, so 99% of bandwidth is wasted | Qwenputer: "**Chip tokens run only the chip layers and are invisible to Qwen's layers**"; its Triton executor runs only the circuit. The bottleneck the paper is built to solve **is already solved**. | 🔴 (premise) |
| F7 | §2.1 Bottleneck 2: 3–6 tokens per instruction, about 1,000 instr/s | Qwenputer: **1.0–1.55 tokens per instruction**. Executor v2 on an L4: **3,699 instr/s at batch 1** (0.177 ms per step) and 13.9k instr/s at batch 32. A newer executor reaches 24k tokens/s. Exact over more than 2.6M instructions of checked runs. | 🔴 (premise) |
| F8 | Qwen3.5-4B: 36 layers; GDN key head dim 256 | config.json: **32 layers**, hidden 2560, GDN **key head dim 128**, 16 key / 32 value heads, SiLU. Vocab 248,320 is correct. | 🟡 |
| F9 | Unified blueprint d_vm = 164 | 164 is Loom's **N=8, ℓ=11** value. With N=32 the formula gives d = 9·11+8·32+1 = **356**. Qwenputer's RV32 core needs **~1,400 extra dims and 11 layers**. | 🟠 |
| F10 | §7.2 MMIO registers at "Rows 250–255" of scratchpad column 0 | X has only 164 rows. | 🟡 |
| F11 | Ref 9 (GDN-2) cited via ResearchGate | Primary source: **arXiv:2605.22791** (NVlabs). | 🟡 |
| F12 | Ref 10 LoopSpec via ResearchGate; LoopKV "compresses to 2-bit" | LoopSpec: **arXiv:2609.17184** (KAIST). LoopKV mixes 4-bit, 2-bit and KV reuse per (loop, layer); −35% storage compared with uniform 4-bit. | 🟡 |
| F13 | LoopCD is a "contrastive latent regularizer" that mitigates drift | LoopCD is a **training-free contrastive decoding** method for accuracy, not a stability regularizer. It is irrelevant to an analytic VM. | 🟡 |
| F14 | Percepta: ">30k tok/s CPU-specialized streaming" | The key mechanism, **2-D heads with HullKVCache for O(log t) lookups**, is not mentioned. It is the main idea worth borrowing for any trace-based design. Its code (`Percepta-Core/transformer-vm`) is not cited. | 🟡 |
| F15 | ISA table: eBPF "8–10 layers"; RISC-V "11–13 layers" | The eBPF figure is not sourced anywhere. The RISC-V figure is a property of Qwenputer's circuit, not of the ISA. | 🟡 |

---

## 3. Architecture-level contradictions

**A1 — "Orthogonal residual cohabitation" does not fit a Loom-style VM.** 🔴
A Loom layer is not a standard LM layer. It has:
- per-column biases $b_1\in\mathbb R^{r\times n}$;
- no normalization;
- a **non-causal** mask;
- symmetric Q=K at λ=10;
- ReLU FFNs.

Qwen3.5 layers instead use RMSNorm, causal gated attention or GDN, and SiLU. So either:
- (a) X lives in extra residual channels of n token positions, which needs ~512–2048 reserved context positions (the very "context pollution" the abstract says it removes) plus custom non-causal layers; or
- (b) X is a separate tensor run by a separate kernel, and then "cohabitation" means nothing. That is a co-resident tool executed by the host.

Qwenputer's real achievement is that it loads as a stock `Qwen3_5ForCausalLM`, and the blueprint would give that up.

**A2 — Using GDN as RAM only pays off in the token paradigm the paper rejects.** 🟠
GDN state evolves over *tokens*. Inside a depth loop with no tokens, "GDN-2 banked RAM" with one-hot keys is just `S[:, i] = v`, a plain array. Its only benefit, running on stock Qwen kernels, exists only in Paradigm A.

**A3 — No bridge from tokens to bytes or from results back to the LLM.** 🔴
The paper never says how text tokens of a `<compile_and_run>` block reach the VM's byte buffer. Qwen's tokens are multi-byte BPE, and Qwenputer handles this with a 16-bytes-per-token capture window. Nor does it say how the result returns. "Project VM output register into LLM residual stream" does not work: a frozen LLM cannot read an arbitrary injected vector without training. Every working system returns results as **tokens**.

**A4 — Word size vs layer budget.** 🔴
In fp32 the "strictly 8 layers, one instruction per pass" claim holds only for N ≤ 16–23. Real 32-bit semantics need multi-limb ops: about 4–10 passes per 32-bit ADD, and N passes per multiply with MULACC. Qwenputer runs a full RV32 instruction in about 1.0–1.55 tokens, so per *semantic* instruction the looped core is **not clearly better**.

**A5 — The in-weights `qcc` (Tier 1) does not fit Loom-scale memory.** 🟠
Loom has 64–224 data words and ≤ 1,792 instruction slots. A recursive-descent C compiler needs tens of KB of code and heap. Qwenx86 manages it with 64 KB RAM + 256 KB ROM in GDN state, which is the token paradigm again.

**A6 — The MMIO bridge contradicts the abstract and opens a security hole.** 🔴
The abstract sells "eliminating security boundaries". In practice §7.2 gives model-generated code `POSIX_READ/WRITE`, `LLVM_JIT_COMPILE` and raw "virtual/physical buffer descriptors". That is an un-sandboxed RCE primitive. It also brings back host round-trips, which undoes the stated latency benefit, and it is more dangerous than an ordinary sandboxed tool call.
There are implementation problems too:
- "Unified Memory" is not `cudaHostAllocMapped`;
- Triton exposes no first-class `__nanosleep`. *(Corrected in round 2, C1: Triton 3.6 does have
  system-scope atomics (`scope="sys"`) and volatile loads; the real obstacles are occupancy,
  watchdogs and memory ordering.)*
- persistent spin kernels risk watchdog timeouts and keep the GPU occupied.

**A7 — The fused SRAM loop is the right lever but oversold.** 🟡
With about 8,000 nonzeros, the VM step is purely latency-bound, so a persistent single-kernel step with sparse argmax is a good idea. Qwenputer lists it as its own next step. A realistic ceiling is 10⁴–10⁵ instr/s. A native interpreter of the same ISA does about **10⁸ instr/s**, and `tcc` compiles C in milliseconds. Speed is **not** a value proposition.

**A8 — "Mathematically verified" is unsupported.** 🟠
The repo has three toy tests. `snap_bipolar` is a `torch.where` (not a neural circuit), and `gdn_exact_write` builds a dense d_k×d_k projector for what is `S[:, i] = v`.

---

## 4. What is sound and worth keeping

- **Loom's actual design (N=8):** the 6-threshold borrow chain with the correct $D_i$, opcode-as-operand routing, STORE via address rewrite, merged branch+PC, argmax attention, and a fixed-size state. It is verified and reproducible.
- **Qwenputer's one-hot GDN-row RAM:** exact, verified over millions of instructions, and compatible with a stock HF model.
- **A two-level memory idea:** a small fast register/scratch file plus a large associative RAM. This is the best original idea in the paper, if it is implemented as a plain one-hot-key matrix memory first.
- **Deterministic halting** on PC==0 or a trap, instead of ACT.
- **Reference-oracle discipline:** an ISA interpreter or QEMU in lockstep with the neural path (both Loom and Qwenputer do this).

## 5. Where real value could come from

Native execution or a sandboxed tool will always be about 10⁴× faster, so the defensible goals are:
1. **Self-contained artifact:** exact computation shipped inside one model file with no tool runtime (Qwenputer's pitch).
2. **Differentiability and internal visibility:** the LLM can attend to execution state, and gradients can flow through it (Percepta's pitch). This only matters if you **train** around it.
3. **Interpretability and education:** weights that are fully explained.
4. **Verified rule enforcement inside perception models** (Loom §1): small state machines, not full C.

Pick one; the plan differs a lot between them.

## 6. Recommended re-scoped roadmap (with go/no-go gates)

| Phase | Deliverable | Gate |
|---|---|---|
| P0 | Python ISA reference interpreter (Loom-22 at N=8, parameterised N) + tiny assembler | Unit tests per opcode |
| P1 | Analytic weight builder reproducing Loom (N=8, d=155, n=1024), fp32, CPU, argmax + softmax paths | Lockstep with P0 on Loom's 111-test style suite; exhaustive ALU at N=8 |
| P2 | Precision study: N=16 fp32 single-layer; N=32 via 2× 16-bit limbs + carry (extra layer) **or** fp64 | Exhaustive/random lockstep; documented max N per dtype |
| P3 | External one-hot matrix RAM (plain tensor; optional GDN-kernel form later) | Same suite with m ≫ 224 |
| P4 | Compiler: either a C-subset→ISA compiler (Loom-style) **or** RV32IM as the ISA (reuse gcc/clang, riscv-tests, QEMU oracle) | Compiled programs match QEMU/ref |
| P5 | Performance: sparse argmax step, then persistent CUDA/Triton kernel | instr/s measured against Qwenputer's 3.7k/s baseline |
| P6 (optional) | LLM integration via **token** I/O bridge (Qwenputer-style), not residual projection | KL vs base ≤ noise floor; end-to-end tool-call accuracy |

Drop or postpone indefinitely: the in-weights `qcc` ROM, the MMIO/syscall/cuBLAS/NPU bridge, the
GDN-2 dependency, the single-layer SwiGLU multiplier, and the bf16 execution claims.

## 7. Required corrections if the whitepaper is kept

1. Replace the §3.2 $D_i$ with Loom's formula and state each dtype's exact N limit.
2. Rewrite §3.3 OR/XOR to use minuend rewriting.
3. Remove or redesign §3.4 (ReGLU, an explicit ADC with its layer and neuron cost, fp32 only).
4. Replace §5.2 Snap with the continuous form; restore the L8 snap.
5. Fix the facts in §1/§2 (KL values, 32 layers, tokens per instruction, instr/s, early exit already present in Qwenputer, Loom N=8).
6. Make d_vm consistent with N and ℓ.
7. Remove or sandbox §7.2.
8. Add the token↔byte I/O design.
9. Cite primary arXiv IDs.
