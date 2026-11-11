# Architecture and Mathematical Foundations of Weight-Compiled Looped Neural Virtual Machines, In-Weights Compilers, and Zero-Copy Hardware Bridges

## Abstract

Embedding deterministic computing machinery directly into the parameter spaces of neural networks eliminates the serialization latency, context-window pollution, and security boundaries of external neurosymbolic tool-calling pipelines. Existing weight-compiled neural computers have evolved along two distinct axes: (1) **token-autoregressive co-processors**—such as `Qwenputer`, `Qwenx86`, `qwenGPU`, and the `WASM Interpreter Transformer`—which prepend compiled circuit layers to pretrained large language models (LLMs) and use linear attention fast weights (`Gated DeltaNet`) or prefix-sum attention as addressable memory, and (2) **depth-looped state engines**—exemplified by `Loom`—which execute complete instruction cycles in-place via iterative depth recurrence ($X^{(t+1)} = \mathcal{F}_\theta(X^{(t)})$) over a fixed bipolar tensor without generating intermediate tokens.

This whitepaper synthesizes both paradigms into a unified, mathematically verified master architecture: the **Orthogonal Looped Neural Virtual Machine (NeuroVM)**. We resolve the core structural trade-offs of prior systems by combining:
1. **Orthogonal Residual Subspace Cohabitation** with an **Early-Exit Fused SRAM Triton Loop**, allowing an 8-layer compiled processor to cohabit a frozen foundation model with zero logit divergence ($0.0$ KL) while bypassing downstream language layers and vocabulary projections during internal clock cycles.
2. **A Mathematically Consistent Hybrid ALU** that unifies bit-sliced 6-threshold ReLU direct borrow-chain subtraction ($10N$ hidden units) and opcode-as-operand routing with an explicit **Bipolar-to-Scalar DAC / Limb-Chunked SwiGLU Multiplier / Scalar-to-Bipolar ADC** pipeline.
3. **Folded Layer-3 Deadzone Snapping** ($\epsilon = 0.1$), preserving digital bipolar invariants ($\{-1, +1\}$) across millions of recurrent passes within an 8-layer budget.
4. **Multi-Head Banked Gated DeltaNet-2 Fast-Weight RAM**, scaling exact rank-one orthonormal memory addressing ($S_t^{(h)} = S_{t-1}^{(h)}(I - e_i e_i^T) + v_t e_i^T$) across $H = 64$ parallel memory banks without key-dimension explosion or cross-talk.
5. **A Dual-Tier Compilation Pipeline and Zero-Copy Pinned MMIO Hardware Bridge**, pairing an in-weights recursive-descent compiler firmware (`qcc`-class) in ROM with a host Clang/LLVM serializer and a `cudaHostAllocMapped` trap interface for native CPU syscalls, CUDA Tensor Core kernels, and NPU execution.

---

## 1. Architectural Precedents in Weight-Compiled Neural Computers

Weight-compiled neural virtual machines derive network parameters algebraically from formal instruction set architecture (ISA) specifications and digital logic equations rather than stochastic gradient descent. Across current literature, six primary architectures establish the empirical baseline for neural execution engines:

| System | Base Model / Framework | Target ISA / Abstract Machine | Weight Construction Methodology | Architectural Topology | Memory & State Substrate | Verified Execution Throughput |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwenputer** | `Qwen3.5-4B` | 32-bit RISC-V (`RV32IM` + `Zba` + Bignum unit) | Analytical compilation via `chipformer` DSL | 11 compiled layers prepended to frozen 4B LLM | Gated DeltaNet recurrent state ($\alpha=1, \beta=1$) | 3,200–5,500 trace tokens/s (NVIDIA L4 via fused Triton) |
| **Qwenx86** | `Qwen3.5-4B` | 64-bit x86 (MSVC integer subset + flag $\mu$-ops) | Analytical compilation (`chipformer` fork) | 13 compiled layers prepended to frozen 4B LLM | Multi-head Gated DeltaNet (64 KB RAM, 256 KB ROM, 128 KB data) | 169k trace tokens in 71.2s (~2,377 tokens/s on RTX GPU) |
| **qwenGPU** | `Qwen3.5-2B` | 4-lane SIMT GPU (NVIDIA PTX subset) | Analytical compilation via extended `chipformer` | 45 compiled layers prepended ahead of 24 LLM layers | Gated DeltaNet state (15 regs/lane, shared RAM, ROM) | 10–15 trace tokens/s (Apple Silicon MPS) |
| **Loom** | Standalone Looped Transformer | 22-opcode single-cycle ISA (`SUBLEQ` + 21 ops) | Analytical derivation from ISA circuit spec | 8 fixed-depth layers iterated cyclically in-place | Persistent bipolar tensor $X \in \{-1, +1\}^{d \times n}$ | 1 instruction per 8-layer pass; FPGA-verified (Alveo U200) |
| **WASM Interpreter** | Standalone Autoregressive Transformer | WebAssembly Bytecode (25 opcodes, stack VM) | Analytical linear algebra compilation | 8 heterogeneous layers ($d=100$, 30 total heads) | Autoregressive trace via `SumAttn` & parabolic keys | Exact bytecode trace autoregression (112/112 tests) |
| **Percepta WASM** | Standalone Autoregressive (2D Attention) | WebAssembly (stack machine runtime) | Analytical compilation | 7 layers ($d=36$, 18 2D attention heads) | Token-embedded execution state history | >30,000 trace tokens/s (CPU-specialized streaming) |

### 1.1 Prepend Token-Driven Co-Processors (`Qwenputer`, `Qwenx86`, `qwenGPU`)
In `Qwenputer`, an 11-layer analytically compiled 32-bit RISC-V processor (`RV32IM` with `Zba` address-generation extensions and a hardware bignum unit) is prepended directly upstream of `Qwen3.5-4B`'s native language layers. The residual stream is widened so that the compiled "chip layers" operate in dedicated orthogonal subspaces without perturbing pretrained language weights ($0.0$ KL divergence on natural text). Execution advances via autoregressive token generation: emitting structured trace tokens (memory addresses, data words, and clock markers) updates the recurrent Gated DeltaNet state matrices, while a host driver (`RvChat`) hides internal trace tokens from the conversational context window.

`Qwenx86-4B-compact` extends this topology to 13 prepended layers to support a 64-bit x86 integer subset (targeting MSVC output) and an expanded memory map: 64 KB of writeable RAM distributed across 64 RAM heads, 256 KB of code ROM, and 128 KB of read-only constants. To handle x86 condition flags (`ZF`, `SF`, `CF`, `OF`) efficiently, `Qwenx86` introduces dedicated flag micro-operations (`FADD`, `FSUB`, `FAND`, `FOR`, `FXOR`, `FINC`, `FDEC`) and decomposes 64-bit operations into paired 32-bit lanes. Crucially, `Qwenx86` embeds a complete C compiler (`qcc`) and runtime firmware directly into its weight ROM, enabling the model to emit C code inside a native `<tool_call>`, compile the source into x86 machine code inside its own weights, and execute the resulting binary within a single generation session.

`qwenGPU` scales analytical weight compilation from scalar CPUs to data-parallel SIMT execution by prepending 45 compiled layers to `Qwen3.5-2B` (forming a 69-layer hybrid model). The compiled layers implement a 4-lane SIMT processor capable of executing compiled NVIDIA PTX kernels, including per-lane predicate registers, warp shuffles, divergent branch reconvergence, shared memory, and `__syncthreads()` barriers.

### 1.2 Stack-Based Autoregressive Bytecode Engines (`WASM Interpreter`, `Percepta`)
The `WASM Interpreter Transformer` (`eastlondoner/wasm-interpreter-transformer`) compiles a 25-opcode WebAssembly stack machine into an 8-layer heterogeneous transformer ($d = 100$) with 30 total attention heads distributed across Layers 0–7 as $[13, 1, 0, 8, 2, 1, 1, 4]$. Rather than relying on linear RNN states, it reconstructs machine state over the autoregressive trace using causal prefix sums (`SumAttn`) to track dynamic operand stack depth and parabolic key-query matching to fetch local variables and stack operands across arbitrary call depths.

### 1.3 Depth-Recurrent State Engines (`Loom`)
Departing from sequence autoregression, `Loom` (`arXiv:2604.08816`) implements a general-purpose virtual machine via depth recurrence over an 8-layer transformer block. All machine state—scratchpad registers, data RAM, instruction ROM, and the program counter (`PC`)—is encoded in a fixed-size bipolar matrix $X \in \{-1, +1\}^{d \times n}$. Each forward pass through the 8-layer block executes one complete machine instruction in-place ($X^{(t+1)} = \mathcal{F}_\theta(X^{(t)})$) with zero token emission, zero vocabulary projection overhead, and zero floating-point drift across millions of sequential instructions.

---

## 2. Paradigm Analysis: Sequence Autoregression vs. Depth Recurrence

Choosing how time and state evolve inside a neural computer dictates its hardware utilization profile, memory complexity, and maximum sustainable clock rate.

```
Paradigm A: Token-Autoregressive Co-Processor (Qwenputer / Qwenx86)
[Token t] -> [13 Chip Layers] -> [36 Frozen LLM Layers] -> [LM Head (248k Vocab)] -> [Sample Token t+1]
  * Cost per micro-step: Full 4B model pass + vocabulary projection + sequence growth.

Paradigm B: Orthogonal Looped Co-Processor (Unified NeuroVM Architecture)
[LLM emits <tool_call>]
       │
       ▼ (Enter Fused In-SRAM Triton Loop)
  ┌──► [8 Compiled Looped VM Layers (Pinned in L2 / SRAM)] ──┐
  │                        │                                  │
  └────── Recirculate X^{(t+1)} (PC != 0 & Trap == 0) ────────┘
                           │
       ▼ (Exit on PC == 0 or REG_MMIO_CMD != 0)
[Resume LLM Decoding / Dispatch Zero-Copy Host Hardware Bridge]
```

### 2.1 Bottlenecks of Pure Token Autoregression
Driving virtual machine execution via token generation imposes three structural overheads:
1. **Full-Backbone Parameter Traffic:** When 11–13 chip layers are prepended to a 4B LLM without an early-exit loop, every machine trace token forces the runtime to execute all 32–36 downstream language layers and project hidden states through the unembedding matrix $W_u \in \mathbb{R}^{d \times |V|}$ ($|V| = 248,320$ in `Qwen3.5`). Because a single C program execution spans $50,000$ to $358,500$ trace tokens, 99% of memory bandwidth is spent reading frozen language weights that perform no virtual machine computation.
2. **Multi-Token Instruction Dilution:** Because each trace token only carries a narrow slice of state update (an address pointer, a 32-bit word, or a step marker), executing a single ISA instruction requires 3 to 6 autoregressive token steps, reducing 5,000 trace tokens/s to ~1,000 actual instructions/s.
3. **Kernel Launch and Sampling Latency:** Autoregressive stepping synchronizes across the GPU command queue at every token unless captured inside a static CUDA graph or custom multi-step kernel.

### 2.2 Advantages and Scaling Laws of Depth-Looped Execution
In a depth-looped architecture, sequence generation is suspended during program execution. The state tensor $X \in \mathbb{R}^{d \times n}$ recirculates through a compact $K$-layer block ($K = 8$) whose total weight footprint is only 8.3 MB to 28 MB (`fp32`/`bf16`). Because both the state tensor $X$ and the 8 layer parameter matrices fit inside GPU L2 cache and Streaming Multiprocessor (SM) SRAM, the entire execution loop runs without touching device High Bandwidth Memory (HBM).

When looped architectures are alternatively used for latent chain-of-thought reasoning across token sequences, naive serving allocates a $K \times T$ key-value cache grid. Recent serving optimizations—**LoopKV** (which compresses early recurrent draft iterations to 2-bit precision or shares terminal iteration KV tensors, cutting KV memory traffic by 35% and boosting throughput by $1.7\times\text{–}2.2\times$) and **LoopSpec** (pipelined self-speculative decoding across recurrent depths)—mitigate multi-token loop costs. For deterministic virtual machine execution, however, **pure in-place state recurrence + linear fast weights** eliminates sequence KV caching during the execution loop entirely.

---

## 3. Mathematical Foundations of Neural Execution Circuits

### 3.1 Circuit Complexity ($\text{TC}^0$) and Computational Universality
A single feed-forward pass through a fixed-depth transformer with bounded precision is restricted to the circuit complexity class $\text{TC}^0$ (constant-depth threshold circuits), which cannot solve problems requiring unbounded sequential state composition (such as arbitrary-length graph reachability or general program evaluation). However, iterating a fixed transformer block $\mathcal{F}_\theta: \mathbb{R}^{d \times n} \to \mathbb{R}^{d \times n}$ via recurrent feedback $X^{(t+1)} = \mathcal{F}_\theta(X^{(t)})$ forms a discrete-time dynamical system that emulates a Universal Register Machine, achieving full Turing completeness bounded only by the finite word size $N$ and memory capacity.

### 3.2 Direct Borrow-Chain Arithmetic via 6-Threshold ReLU Networks
Classical digital logic synthesis inside neural networks requires a 3-layer FFN cascade to subtract two $N$-bit numbers: Layer A performs one's complement bit inversion ($\neg B$), Layer B propagates carries to form two's complement ($-B = \neg B + 1$), and Layer C ripples carries across $A + (-B)$. Following `Loom`, this 3-layer cascade is compressed into a **single FFN layer** using direct borrow-chain subtraction over bipolar bit vectors.

Let two $N$-bit signed two's-complement integers $A, B$ be encoded in bipolar format $A_i, B_i \in \{-1, +1\}$ for bit indices $i \in \{0, \dots, N-1\}$ (ordered MSB-first, where $+1$ represents binary `1` and $-1$ represents binary `0`). For each bit position $i$, a linear projection computes the weighted partial difference $D_i$:

$$D_i = \sum_{j=0}^{i-1} 2^{N-2-j} (A_j - B_j) + \frac{1}{2}(A_i - B_i)$$

Here, the weight $2^{N-2-j}$ equals half the binary positional value of bit $j$, scaling the bipolar difference $(A_j - B_j) \in \{-2, 0, +2\}$ so that $D_i$ takes exact integer values. Let $P = 2^{N-1-i}$. We construct six pre-activation thresholds per bit using three antagonistic ReLU pairs:

$$z_{i,1} = D_i + P + 1, \quad z_{i,2} = D_i + P$$
$$z_{i,3} = -D_i, \quad z_{i,4} = -D_i - 1$$
$$z_{i,5} = D_i - P + 1, \quad z_{i,6} = D_i - P$$

The output difference bit $R_i \in \{-1, +1\}$ is obtained in a single two-layer perceptron pass ($\text{FFN}(x) = W_2\,\text{ReLU}(W_1 x + b_1) + b_2$):

$$R_i = 2 \Big[ \text{ReLU}(z_{i,1}) - \text{ReLU}(z_{i,2}) + \text{ReLU}(z_{i,3}) - \text{ReLU}(z_{i,4}) + \text{ReLU}(z_{i,5}) - \text{ReLU}(z_{i,6}) \Big] - 3$$

Because each antagonistic pair $\text{ReLU}(u + 1) - \text{ReLU}(u)$ forms an exact unit step transition over $[-1, 0]$, this 6-unit combination evaluates the local difference bit and resolves MSB-to-LSB borrow propagation simultaneously across all $N$ bits. Including $4N$ auxiliary neurons to clear temporary staging registers, the entire $N$-bit subtractor requires exactly **$10N$ hidden neurons** ($320$ hidden units for $N = 32$, or $640$ for $N = 64$).

### 3.3 Opcode-as-Operand Routing
Rather than instantiating parallel ALU layers for each opcode, all arithmetic, bitwise, shift, and comparison instructions are algebraically reduced in Layer 2's FFN into **operand staging** for Layer 4's single shared subtraction core (`scr_min - scr_sub`):
* **Addition (`ADD`):** Staged as $A - (-B)$. Layer 2 computes two's complement negation $-B$ using $2N$ hidden units for bit inversion and $2N$ units for increment carry propagation, placing $-B$ into `scr_sub`.
* **Bitwise `AND`:** In bipolar encoding, $A_i \land B_i = +1 \iff A_i = +1 \text{ and } B_i = +1$. Layer 2 stages `scr_min` $= A$ and activates $N$ correction units that place $+1$ into `scr_sub` at bit $i$ if and only if $A_i = +1$ and $B_i = -1$, causing the subtractor to pull bit $i$ down to $-1$.
* **Bitwise `OR` and `XOR`:** Staged identically using $N$ hidden units (`OR`, correcting when $A_i = -1, B_i = +1$) and $2N$ hidden units (`XOR`, detecting directional mismatches).
* **Bit Shifts (`SHL`, `SHR`) & Rotations:** Performed via fixed index-permutation matrices in Layer 2's linear projection with zero non-linear neuron overhead.

### 3.4 Resolving the Bipolar-vs-Scalar Multiplication Mismatch (Limb-Chunked SwiGLU)
A critical mathematical pitfall in naive hybrid specifications is mixing bit-sliced bipolar vectors ($b \in \{-1, +1\}^N$) with scalar bilinear `SwiGLU` multiplication ($\text{Mul}(A, B) = \text{ReLU}(A) \odot B$). Applying element-wise $\text{ReLU}(A) \odot B$ directly to bipolar vectors only performs per-bit gating rather than integer multiplication, while projecting a full 32-bit or 64-bit integer into a single scalar overflows the exact integer mantissa of `bf16` ($2^8 = 256$) or `fp32` ($2^{24} = 16,777,216$).

To execute hardware-accurate multiplication without multi-cycle software loops (while retaining `MULACC` as a fallback), our ALU implements a **Limb-Chunked DAC-SwiGLU-ADC Pipeline**:
1. **8-Bit Bipolar-to-Scalar DAC (Input Projection $W_{\text{DAC}}$):** Each 32-bit bipolar operand is partitioned into four 8-bit limbs $L_m \in \{-1, +1\}^8$ ($m \in \{0, 1, 2, 3\}$). A linear projection converts each bipolar byte into an exact unsigned scalar integer in $[0, 255]$:
   $$u_m = \sum_{k=0}^{7} 2^{k-1} \left(L_{m,k} + 1\right) \in \{0, 1, \dots, 255\}$$
2. **Bilinear SwiGLU Partial Products:** For all 16 byte pairs $(m, n) \in \{0..3\}^2$, a gated linear unit computes the exact partial product:
   $$P_{m,n} = \text{ReLU}(u_m^{(A)}) \odot u_n^{(B)} \in [0, 65025]$$
   Because $65,025 < 2^{16}$, every partial product and limb accumulation is 100% exact in `fp32` (and within 16-bit integer precision).
3. **Scalar-to-Bipolar ADC (Binary Decomposition):** The accumulated limb sums are decomposed back into bipolar bits $\{-1, +1\}$ using a periodic sawtooth/step ReLU threshold bank before entering the residual stream.

### 3.5 Deterministic Addressing Mechanics: Coordinate Argmax, Parabolic Keys, and Sum-Attention
Neural virtual machines employ three complementary attention mechanisms depending on the memory topology:

1. **Symmetric Coordinate Argmax (Looped State Tensor $X$):**
   Each column $j \in \{0, \dots, n-1\}$ of $X$ stores a static $\ell$-bit bipolar address encoding $\text{pos}_j \in \{-1, +1\}^\ell$ ($\ell = \lceil \log_2 n \rceil$). Using symmetric projections $Q = K$, matching a target pointer $p \in \{-1, +1\}^\ell$ against all columns yields:
   $$\text{Score}(j) = p^T \text{pos}_j = \ell - 2 \, d_{\text{Hamming}}(p, \text{pos}_j)$$
   This score attains a strict global maximum of $+\ell$ if and only if $\text{pos}_j = p$ (and $\le \ell - 2$ for all $j \neq p$). Passing scores through hard-max/argmax (or high-$\beta$ softmax followed by deadzone snapping) routes column $j$ with zero dispersive cross-talk.

2. **Parabolic Key-Query Matching (Scalar Address Indices):**
   When addresses are represented as scalar integers $i, j \in \mathbb{Z}_{\ge 0}$, exact associative lookup is achieved via 2D parabolic projections:
   $$Q_i = \begin{bmatrix} i \\ 1 \end{bmatrix}, \quad K_j = \begin{bmatrix} 2j \\ -j^2 \end{bmatrix} \implies Q_i^T K_j = 2ij - j^2 = -(i - j)^2 + i^2$$
   For any query index $i$, the quantity $-(i - j)^2 + i^2$ peaks uniquely at $j = i$ with a margin of at least $1$ over adjacent addresses $j = i \pm 1$.

3. **Unnormalized Causal `SumAttn` (Dynamic Stack Depth Tracking):**
   For stack-based bytecode execution, removing the softmax denominator over a causal mask $M_{tk} = \mathbb{I}(k \le t)$ computes an exact prefix sum of stack pointer deltas $\Delta_k \in \{-1, 0, +1\}$:
   $$\text{SP}_t = \sum_{k=1}^{t} M_{tk} \Delta_k$$

---

## 4. Memory Sub-Architectures: Bipolar State Columns and Banked Gated DeltaNet-2 RAM

To combine zero-latency register/scratchpad access with scalable 64 KB–1 MB random-access memory, the Unified NeuroVM pairs a **2D Bipolar State Tensor** (L1 Scratchpad & Register File) with **Multi-Head Banked Gated DeltaNet-2 Fast Weights** (L2 Main RAM & Firmware ROM).

| Memory Characteristic | Standard Softmax KV Cache | Loom 2D Bipolar Tensor ($X \in \{-1, +1\}^{d \times n}$) | Multi-Head Banked Gated DeltaNet-2 ($\alpha=1, b=1, w=1$) |
| :--- | :--- | :--- | :--- |
| **Spatial Footprint** | $O(L \cdot d)$ (Grows unboundedly with trace length $L$) | $O(d \times n)$ (Fixed SRAM-resident matrix) | $O(H \cdot d_v \cdot d_k)$ (Fixed fast-weight tensor across $H$ heads) |
| **Lookup Complexity** | $O(L)$ per step / $O(L^2)$ total | $O(n)$ hard-max column match | $O(d_v \cdot d_k)$ matrix-vector product per active bank |
| **Write Mechanism** | Append-only sequence concatenation | Single-cycle column overwrite (Layer 5) | Exact in-place rank-one row erase & overwrite |
| **Drift & Retention** | Degrades as context window overflows | Infinite retention via Layer-3 deadzone snapping | Infinite retention ($\alpha_t = 1.0$) with orthonormal basis keys |
| **Optimal Role in NeuroVM** | Outer LLM natural language conversation only | CPU Registers, ALU Staging, L1 Scratchpad, MMIO Ports | Scalable 64 KB–1 MB Main RAM, Heap, and `qcc` Compiler ROM |

### 4.1 L1 Spatial Partitioning in the Bipolar State Tensor $X$
Within the 8-layer looped execution core, the working state tensor $X \in \{-1, +1\}^{d \times n}$ is partitioned across spatial columns and feature rows:
* **Columns $0 \le j < s$ ($s = 32$):** L1 Scratchpad Workspace, Architecturally Visible Registers (`R0`–`R15`), Program Counter (`PC`), and Hardware MMIO Trap Registers (`REG_MMIO_*`).
* **Columns $s \le j < s + m$:** L1 Tightly Coupled Data Memory (fast local stack/data words).
* **Columns $s + m \le j < n$:** L1 Instruction Cache / Micro-Op ROM storing 4-tuple instructions $I_k = [\text{Op}, a, b, c]$.
* **Row Allocation ($d$ features):** Row $0$ holds the immutable scratchpad mask ($+1$ for $j < s$, $0$ otherwise); Rows $1 \dots \ell$ store the immutable $\ell$-bit binary column coordinates $\text{pos}_j$; Rows $\ell+1 \dots \ell+N$ store the $N$-bit signed bipolar data word; and Rows $\ell+N+1 \dots d-1$ provide transient instruction decode, pointer dereference, and ALU staging buses.

### 4.2 Resolving the Orthonormality Bottleneck: Multi-Head Banked Gated DeltaNet-2 RAM
In a single Gated DeltaNet head, the recurrent fast-weight state $S_t \in \mathbb{R}^{d_v \times d_k}$ updates via:

$$S_t = S_{t-1} \left(\alpha_t I - \beta_t k_t k_t^T\right) + \beta_t v_t k_t^T$$

Setting $\alpha_t = 1.0$, $\beta_t = 1.0$, and restricting keys/queries to standard basis vectors $e_i \in \{0, 1\}^{d_k}$ yields exact row erasure and overwrite: $S_t = S_{t-1}(I - e_i e_i^T) + v_t e_i^T$ and readout $y_t = S_t e_i = v_t$. However, because $e_i$ must be strictly orthonormal ($e_i^T e_j = \delta_{ij}$) to prevent rank-one erasure cross-talk across addresses, a single head with key dimension $d_k = 256$ can address **at most 256 memory words**.

To scale deterministic RAM to 128 KB (or more) without inflating $d_k$ to $16,384$, we incorporate **Gated DeltaNet-2's decoupled channel-wise erase ($b_t$) and write ($w_t$) gates** across **$H = 64$ parallel RAM heads** ($d_k = 256$ rows per head, $d_v = 64$ bits per word $\implies 64 \times 256 \times 8\text{ B} = 128\text{ KB}$ per RAM layer):

$$S_t^{(h)} = S_{t-1}^{(h)} \Big(I - k_t \left(b_t^{(h)} \odot k_t\right)^T\Big) + \left(w_t^{(h)} \odot v_t\right) k_t^T$$

A 14-bit word address $A \in [0, 16383]$ (covering 128 KB at 64-bit word granularity) is split into:
* **6-Bit Bank Selector:** $h_{\text{target}} = (A \gg 8) \in \{0, \dots, 63\}$
* **8-Bit Intra-Bank Row Key:** $i = (A \mathbin{\&} \text{0xFF}) \in \{0, \dots, 255\} \implies k_t = e_i \in \{0, 1\}^{256}$

A SwiGLU bank-decoder checks whether head $h$ matches $h_{\text{target}}$ during a `STORE` operation, setting the erase and write gates to:

$$b_t^{(h)} = w_t^{(h)} = \mathbb{I}\left(h = h_{\text{target}} \land \text{Op} = \text{STORE}\right) \cdot \mathbf{1}$$

Thus, on any memory write cycle, only the targeted head $h_{\text{target}}$ erases and overwrites row $i$, while all other 63 heads preserve their state identically ($S_t^{(h)} = S_{t-1}^{(h)}$). On a `LOAD` cycle, all gates remain $0$ and the output projection selects $y_t = \sum_{h=0}^{63} \mathbb{I}(h = h_{\text{target}}) S_t^{(h)} e_i$.

---

## 5. Control Flow, Indirect Addressing, and Folded Invariant Preservation

### 5.1 Single-Cycle Indirect Pointer Dereferencing (`LOAD` / `STORE` / `FIND`)
Direct-addressing OISC machines (`SUBLEQ`) require self-modifying code to index arrays, inflating program lengths by $4\times$. In our 8-layer looped core:
* **`LOAD` (Indirect Read):** Layer 2 fetches the pointer address from memory column $a$; Layer 3's attention head uses that fetched value as the query coordinate to read the target data word in the same instruction cycle.
* **`STORE` (Indirect Write):** Layer 2 fetches the destination pointer from column $b$ and overwrites the scratchpad's write-address register (`scr_write_addr`). When Layer 5's writeback attention executes, it routes the ALU output directly to the dynamically dereferenced column. This reduces compiled instruction counts on pointer-heavy benchmarks (such as a $9\times 9$ Sudoku solver) by **73.8%** (from 1,085 instructions down to 284).

### 5.2 Folded Layer-3 Deadzone Snapping (Zero Drift in 8 Layers)
In trained looped transformers (e.g., Universal Transformers, Ouro), non-unit Jacobian eigenvalues cause continuous activations to explode or collapse over long horizons ($T \gg T_{\text{train}}$), even when mitigated by contrastive latent regularizers like `LoopCD`. Conversely, allocating a standalone 9th or 8th FFN layer solely to error clamping wastes depth.

Following `Loom`'s verified circuit optimization, **folded scratchpad correction** merges the bipolar deadzone snapping filter directly into **Layer 3's FFN** alongside indirect `LOAD`/`FIND` resolution. For every persistent bipolar bit $x \in \{-1, +1\}$ and tolerance $\epsilon = 0.1$, a 6-threshold ReLU filter applies:

$$\text{Snap}_\epsilon(x) = \begin{cases} +1.0 & \text{if } x \in [1 - \epsilon, 1 + \epsilon] \\ -1.0 & \text{if } x \in [-1 - \epsilon, -1 + \epsilon] \\ x & \text{otherwise} \end{cases}$$

Any floating-point rounding drift $|\delta| < 0.1$ accumulated during `fp32`/`bf16` matrix multiplications is algebraically erased on every single clock cycle, enabling millions of recurrent iterations with **zero bit divergence**.

### 5.3 Merged Layer-6 Branch Evaluation, PC+1 Carry Chain, and Coordinate-Zero Halting
In Layer 6 (an FFN-only layer with $Q = K = V = 0$), two spatially disjoint FFN sub-blocks execute in parallel:
1. **Sub-block 1 (Condition Flag Evaluation):** Inspects the $N$-bit subtraction result $R$ produced by Layer 4. Under signed two's-complement bipolar encoding, $R \le 0$ holds if and only if either the MSB sign bit $R_0 = +1$ (negative) or all bits $R_0 = R_1 = \dots = R_{N-1} = -1$ (zero). A single threshold gate combines these conditions to set $\text{Flag}_{\text{branch}} \in \{-1, +1\}$.
2. **Sub-block 2 ($\text{PC} + 1$ Carry-Chain Increment):** Simultaneously increments the $\ell$-bit program counter $\text{PC}^{(t)}$ using the direct carry-chain ReLU pattern.

In Layer 7, a multiplexer updates the program counter:

$$\text{PC}^{(t+1)} = \begin{cases} C & \text{if } \text{Flag}_{\text{branch}} = +1 \\ \text{PC}^{(t)} + 1 & \text{if } \text{Flag}_{\text{branch}} = -1 \end{cases}$$

Instead of stochastic Adaptive Computation Time (ACT)—which blends intermediate states and corrupts discrete registers—halting is triggered deterministically when control branches to **Address 0** ($\text{PC}^{(t+1)} = 0$) or when Layer 6 asserts an MMIO hardware trap (`REG_MMIO_CMD != 0`).

---

## 6. Intermediate Representation and Dual-Tier Compiler Architecture

### 6.1 ISA & IR Trade-Off Analysis

| Architectural Criterion | RISC-V (`RV32IM`) | WebAssembly (`WASM`) | 64-bit `eBPF` Bytecode | NeuroVM Quad-Tuple ISA (Loom-22 + Mul/MMIO) |
| :--- | :--- | :--- | :--- | :--- |
| **Execution Paradigm** | Load-Store Register Machine (32 regs) | Stack Machine with Structured Frames | Bounded Register Machine (11 regs) | Memory-to-Memory + Scratchpad Register Hybrid |
| **Instruction Encoding** | 32-bit packed bitfields | Variable-length (1–5 bytes) | Fixed 64-bit bitfields | Pre-unpacked 4-tuple $[\text{Op}, a, b, c]$ |
| **Neural Decode Cost** | High (bitfield extraction layers) | Medium (requires `SumAttn` stack head) | Low-Medium (fixed 64-bit fields) | Minimal (1-layer symmetric coordinate fetch) |
| **Pointer & Array Efficiency** | High (native base+offset load/store) | Medium (linear memory index calc) | High (register offset dereference) | High (single-cycle indirect `LOAD`/`STORE`) |
| **Loop Depth Footprint** | 11–13 Layers (`Qwenputer`) | 7–8 Layers (`Percepta` / `WASM`) | 8–10 Layers | **Strictly 8 Layers** |

To minimize neural circuit depth while supporting full C and Rust toolchains, NeuroVM uses a two-layer IR strategy: high-level languages compile either to **eBPF / RV32IM** (for external toolchains) or directly to **unpacked 4-tuple micro-instructions** $I_k = [\text{Op}_k, a, b, c]$ (22 core Loom opcodes plus limb-chunked `MUL` and `MMIO_TRAP`), which execute in exactly one 8-layer pass.

### 6.2 Dual-Tier Compilation Pipeline
To satisfy both self-contained in-weights autonomy and industrial compiler optimization, NeuroVM provides two compilation tiers:
* **Tier 1: In-Weights Compiler Firmware (`qcc`-Class Autonomous Compilation):**
  Following `Qwenx86`, a compact recursive-descent C/DSL compiler binary is pre-compiled into quad-tuple machine code and permanently baked into the read-only Gated DeltaNet ROM banks. When the host LLM generates C source code inside a `<compile_and_run>` tag, the 8-layer looped processor boots directly into the in-weights `qcc` firmware, parses the source tokens from the input buffer, emits compiled quad-tuple instructions into the writeable RAM columns, and branches to the newly compiled entry point—executing the entire edit-compile-run cycle inside the model's weights.
* **Tier 2: Host Clang/LLVM Lowering & State Serializer:**
  For large multi-file Rust or C++ workloads, an external Clang/LLVM pass lowers source code to eBPF/RV32I, legalizes control flow, translates instructions into quad-tuple vectors $I_k = [\text{Op}_k, a, b, c]$, and injects them directly into the ROM columns of $X^{(0)}$ and Gated DeltaNet RAM banks via zero-copy memory mapping.

---

## 7. Unified Master Blueprint: The Orthogonal Looped NeuroVM

```
+-----------------------------------------------------------------------------------+
|                        HOST / HYBRID FOUNDATION MODEL                             |
|  Residual Stream Width: d_total = d_llm (2560) + d_vm (164) = 2724 Channels       |
+-----------------------------------------------------------------------------------+
         │ (Normal Text Tokens: d_vm == 0, 0.0 KL Divergence)
         ▼
+-----------------------------------------------------------------------------------+
|  FUSED IN-SRAM TRITON LOOP (8 Compiled Layers, 4.7M-14M Params Pinned in L2)      |
|                                                                                   |
|  Layer 1: Instruction Fetch       [1 Argmax Head (Q=K) + FFN]                     |
|           Matches PC to column pos_j; loads [Op, a, b, c] into scratchpad.        |
|                                                                                   |
|  Layer 2: Operand Fetch & Route   [2 Argmax Heads + Banked GDN-2 Read + FFN]      |
|           Fetches operands A, B from L1 columns or 64-head GDN-2 RAM;             |
|           FFN performs opcode-as-operand staging (-B, AND/OR/XOR masks, shifts).  |
|                                                                                   |
|  Layer 3: Indirect Dereference    [2 Argmax Heads + Folded Deadzone Snap FFN]     |
|           Executes LOAD/FIND pointer dereference; FFN folds 6-threshold ReLU      |
|           deadzone snap filter (eps = 0.1) over persistent state rows.            |
|                                                                                   |
|  Layer 4: Combinational ALU Core  [0 Heads (FFN Only)]                            |
|           Sub-block A: 10N-unit 6-threshold ReLU direct borrow-chain subtractor.  |
|           Sub-block B: 8-bit DAC -> SwiGLU partial products -> ADC multiplier.    |
|                                                                                   |
|  Layer 5: Memory Writeback        [1 Argmax Head + 64-Head GDN-2 Write + FFN]     |
|           Writes ALU result to direct column a, dereferenced STORE pointer,       |
|           or targeted Gated DeltaNet-2 bank h_target via rank-one update.         |
|                                                                                   |
|  Layer 6: Branch, PC+1 & Trap     [0 Heads (FFN Only)]                            |
|           Disjoint FFN sub-blocks compute Flag_branch (R <= 0), PC + 1 carry      |
|           chain, and inspect REG_MMIO_CMD != 0 for host hardware traps.           |
|                                                                                   |
|  Layer 7: PC Multiplexer          [1 Argmax Head + FFN]                           |
|           Updates PC^{(t+1)} = C (if Flag_branch == +1) else PC^{(t)} + 1.        |
|                                                                                   |
|  Layer 8: Loop Boundary & Bridge  [0 Heads (FFN Only)]                            |
|           Finalizes scratchpad output buffers; checks Halt (PC == 0) or Trap.     |
+-----------------------------------------------------------------------------------+
         │                                                   │
         │ (REG_MMIO_CMD != 0)                               │ (PC == 0: Halt)
         ▼                                                   ▼
+-----------------------------------------+   +-------------------------------------+
| ZERO-COPY PINNED MMIO HARDWARE BRIDGE   |   | RESUME DOWNSTREAM LLM LAYERS        |
| cudaHostAllocMapped Shared Flag & Args  |   | Project VM output register into     |
| -> Host CPU POSIX Syscalls              |   | LLM residual stream & continue      |
| -> Native CUDA / cuBLAS / NPU Dispatch  |   | natural language generation.        |
+-----------------------------------------+   +-------------------------------------+
```

### 7.1 Complete 8-Layer Mathematical Specification

| Layer | Active Heads | Functional Domain | Exact Mathematical Operation |
| :--- | :--- | :--- | :--- |
| **Layer 1** | 1 Head (Argmax $Q=K$) | **Instruction Fetch** | $\text{Score}(0, j) = p_{\text{PC}}^T \text{pos}_j$; transfers $I_{\text{PC}} = [\text{Op}, a, b, c]$ into scratchpad column $X_0$. |
| **Layer 2** | 2 Argmax Heads + $H=64$ GDN-2 Read | **Operand Fetch & Opcode-as-Operand Staging** | Reads $[a], [b]$ from $X$ or $S_t^{(h)} e_i$; FFN stages `scr_min`, `scr_sub`, and `STORE` pointer override. |
| **Layer 3** | 2 Argmax Heads | **Indirect Dereference & Folded Snap Filter** | Dereferences `LOAD` (`[[b]]`) and `FIND`; FFN applies $\text{Snap}_{0.1}(x)$ across all persistent bipolar rows. |
| **Layer 4** | 0 Heads ($Q=K=V=0$) | **Hybrid Combinational ALU** | Evaluates $R_i = 2\sum_{k=1}^3 (\text{ReLU}(z_{i,2k-1}) - \text{ReLU}(z_{i,2k})) - 3$ and limb-chunked SwiGLU product. |
| **Layer 5** | 1 Argmax Head + $H=64$ GDN-2 Write | **Direct/Indirect Memory Writeback** | Updates column $X_{\text{target}}$ or GDN-2 bank $S_t^{(h)} = S_{t-1}^{(h)}(I - \mathbb{I}_{h} e_i e_i^T) + \mathbb{I}_{h} v_t e_i^T$. |
| **Layer 6** | 0 Heads ($Q=K=V=0$) | **Branch Flag, $\text{PC}+1$ & Trap Detect** | Parallel FFN sub-blocks compute $\text{Flag}_{\text{branch}} = \mathbb{I}(R \le 0)$, $\text{PC}+1$, and $\text{Trap} = \mathbb{I}(\text{CMD} \neq 0)$. |
| **Layer 7** | 1 Head (Argmax) | **Program Counter Update** | Multiplexes $\text{PC}^{(t+1)} = C$ if $\text{Flag}_{\text{branch}} = +1$ else $\text{PC}^{(t)} + 1$. |
| **Layer 8** | 0 Heads ($Q=K=V=0$) | **Terminal Snap & Host Bus Interface** | Cleans scratchpad staging buses and exposes `REG_MMIO_*` and halt status to the Triton loop controller. |

### 7.2 Zero-Copy Pinned MMIO Hardware Bridge
To bridge the neural execution core with physical host CPUs, CUDA Tensor Cores, and NPUs without device-to-host tensor serialization overhead, rows within Scratchpad Column $0$ (mirrored to Bank 0, Rows 250–255 of Gated DeltaNet-2) are mapped to page-locked Unified Memory (`cudaHostAllocMapped`):
* `REG_MMIO_STATUS` (`Row 250`): `0x0 = RUNNING`, `0x1 = TRAP_PENDING`, `0x2 = HALTED`
* `REG_MMIO_CMD` (`Row 251`): `0x10 = POSIX_READ`, `0x11 = POSIX_WRITE`, `0x20 = CUBLAS_GEMM`, `0x30 = LLVM_JIT_COMPILE`, `0x40 = NPU_DISPATCH`
* `REG_MMIO_ADDR` (`Row 252`): 64-bit virtual/physical buffer descriptor
* `REG_MMIO_LEN` (`Row 253`): Payload byte length or matrix dimension tuple
* `REG_MMIO_ARG0..1` (`Rows 254–255`): Scalar arguments and return status code

When Layer 6 detects `REG_MMIO_CMD != 0`, the fused Triton loop sets `REG_MMIO_STATUS = 0x1` and enters a lightweight GPU spin-wait (`__nanosleep`) or yields the stream. A host C++/Rust worker thread polling the cache-coherent pinned memory pointer immediately executes the native OS syscall or dispatches the asynchronous cuBLAS/NPU kernel on the physical hardware, writes the resulting payload directly into the mapped memory columns, clears `REG_MMIO_CMD = 0x0`, and sets `REG_MMIO_STATUS = 0x0`—allowing the SRAM-resident Triton loop to resume on the very next microsecond cycle.

---

## 8. References

1. **Qwenputer:** *A fully functioning, deterministic RISC-V CPU compiled into Qwen3.5-4B weights*, `divagr.com/blog/qwenputer` & `github.com/divagr18/qwenputer`.
2. **Qwenx86:** *Qwenx86-4B-compact: 64-bit x86 CPU and in-weights C compiler (`qcc`) in 13 prepended layers*, `huggingface.co/Quazim0t0/Qwenx86-4B-compact`.
3. **qwenGPU:** *4-lane SIMT PTX GPU compiled into 45 layers ahead of Qwen3.5-2B*, `emergent.sh/news/qwengpu-qwen3-5-2b-runs-cuda-45-compiled-gpu`.
4. **Loom:** *Loom: A Scalable Analytical Neural Computer Architecture*, Mehmet Kerem Turkcan et al., `arXiv:2604.08816v1` & `github.com/mkturkcan/Loom`.
5. **WASM Interpreter Transformer:** *Analytical WebAssembly bytecode execution in 8 heterogeneous layers*, `huggingface.co/eastlondoner/wasm-interpreter-transformer`.
6. **Chipformer:** *Framework for compiling digital circuits into transformer weights*, `huggingface.co/divagr1925/chipformer`.
7. **Looped Transformers as Programmable Computers:** Angeliki Giannou et al., *ICML 2023*, `proceedings.mlr.press/v202/giannou23a/giannou23a.pdf`.
8. **Gated Delta Networks:** *Improving Mamba2 with Delta Rule*, Songlin Yang et al., `arXiv:2412.06464`.
9. **Gated DeltaNet-2:** *Decoupling Erase and Write in Linear Attention*, `researchgate.net/publication/405132211`.
10. **LoopKV & LoopSpec:** *Non-uniform KV cache compression and pipelined self-speculative decoding for looped transformers*, `alphaxiv.org/abs/2611.04137` & `researchgate.net/publication/414357704`.
11. **LoopCD:** *Contrastive latent decoding for recurrent depth models*, `arXiv:2610.02185v1`.
