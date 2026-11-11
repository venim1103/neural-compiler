# Neural Compiler (`NeuroVM`)

**Architecture and Implementation of Weight-Compiled Looped Neural Virtual Machines, In-Weights Compilers, and Zero-Copy Hardware Bridges.**

## Overview

This repository houses the technical specification and implementation of **NeuroVM**, a hybrid neurosymbolic execution architecture that compiles deterministic virtual machines, arithmetic logic units (ALUs), random-access memory banks, and compiler toolchains directly into transformer weights.

See [**`WHITEPAPER.md`**](./WHITEPAPER.md) for the complete mathematical synthesis, architectural audit corrections, and system blueprint unifying:
- **Orthogonal Looped Execution (`Loom` + `Qwenx86` synthesis):** 8 analytically compiled transformer layers embedded in orthogonal residual channels, executed via an early-exit in-SRAM Triton loop ($X^{(t+1)} = \mathcal{F}_\theta(X^{(t)})$) with zero token emission overhead.
- **Mathematically Consistent Hybrid ALU:** 6-threshold ReLU direct borrow-chain subtraction ($10N$ hidden units), opcode-as-operand routing, and an 8-bit limb-chunked **DAC $\rightarrow$ SwiGLU $\rightarrow$ ADC** hardware multiplier.
- **Folded Layer-3 Deadzone Snapping:** Continuous floating-point rounding erasure ($\epsilon = 0.1$) folded directly into Layer 3's FFN to guarantee zero bit drift across millions of clock cycles.
- **Multi-Head Banked Gated DeltaNet-2 RAM:** 128 KB+ of zero-drift associative fast-weight RAM partitioned across $H = 64$ heads ($d_k = 256, d_v = 64$) using 6-bit bank-select gates and 8-bit orthonormal basis keys $e_i$.
- **Dual-Tier Compiler & Zero-Copy MMIO Bridge:** In-weights `qcc`-class C/DSL compiler firmware in ROM paired with host Clang/LLVM lowering and a `cudaHostAllocMapped` trap interface for native CPU syscalls, CUDA Tensor Core kernels, and NPU acceleration.
