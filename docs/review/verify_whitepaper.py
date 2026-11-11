"""Numerical audit of WHITEPAPER.md claims (evidence for WHITEPAPER_REVIEW.md, round 1).

Run: python docs/review/verify_whitepaper.py  (CPU, ~1-2 minutes)
"""

import itertools
import math
import random
import torch

relu = torch.relu


def to_bip(v, N):  # MSB-first bipolar
    return torch.tensor(
        [1.0 if (v >> (N - 1 - i)) & 1 else -1.0 for i in range(N)], dtype=torch.float64
    )


def from_bip(b):
    N = len(b)
    return sum((1 << (N - 1 - i)) for i in range(N) if b[i] > 0)


def result_bits(D, P):
    S = (
        (relu(D + P + 1) - relu(D + P))
        + (relu(-D) - relu(-D - 1))
        + (relu(D - P + 1) - relu(D - P))
    )
    return 2 * S - 3


def sub_whitepaper(a, b, N):  # D_i = sum_{j<i} 2^{N-2-j}(A_j-B_j) + 1/2 (A_i-B_i)
    out = []
    for i in range(N):
        D = sum(2.0 ** (N - 2 - j) * (a[j] - b[j]) for j in range(i)) + 0.5 * (
            a[i] - b[i]
        )
        out.append(result_bits(torch.as_tensor(D), 2.0 ** (N - 1 - i)))
    return torch.stack(out)


def W_loom(N, dtype):  # D = W @ [a; b], Loom: D_i = sum_{j>=i} 2^{N-2-j}(a_j-b_j)
    W = torch.zeros(N, 2 * N, dtype=dtype)
    for i in range(N):
        for j in range(i, N):
            W[i, j] = 2.0 ** (N - 2 - j)
            W[i, N + j] = -(2.0 ** (N - 2 - j))
    return W


def sub_loom(a, b, N, dtype=torch.float64):
    D = W_loom(N, dtype) @ torch.cat([a, b]).to(dtype)
    P = torch.tensor([2.0 ** (N - 1 - i) for i in range(N)], dtype=dtype)
    return result_bits(D, P)


print("== 1. Borrow-chain subtractor, exhaustive N=8 ==")
N = 8
bad_wp = bad_loom = 0
for x, y in itertools.product(range(256), repeat=2):
    a, b = to_bip(x, N), to_bip(y, N)
    want = (x - y) & 0xFF
    r = sub_whitepaper(a, b, N)
    if not all(v in (-1.0, 1.0) for v in r.tolist()) or from_bip(r) != want:
        bad_wp += 1
    if from_bip(sub_loom(a, b, N)) != want:
        bad_loom += 1
print(
    f"whitepaper formula wrong on {bad_wp}/65536 pairs; Loom formula wrong on {bad_loom}/65536"
)

print("\n== 2. Precision of the single-layer subtractor at N=32 ==")


def fails(N, dtype, trials=20000, seed=0):
    rnd = random.Random(seed)
    n = 0
    cases = [
        ((1 << (N - 1)) - 1, 1 << (N - 1)),
        ((1 << N) - 1, 0),
        (0, 1),
        (1 << (N - 1), (1 << (N - 1)) - 1),
    ]
    cases += [(rnd.getrandbits(N), rnd.getrandbits(N)) for _ in range(trials)]
    for x, y in cases:
        r = sub_loom(to_bip(x, N), to_bip(y, N), N, dtype).double()
        if from_bip(r) != (x - y) % (1 << N) or not torch.all(r.abs() == 1):
            n += 1
    return n, len(cases)


for N_, dt in [
    (16, torch.float32),
    (24, torch.float32),
    (32, torch.float32),
    (32, torch.float64),
    (64, torch.float64),
    (8, torch.bfloat16),
    (16, torch.bfloat16),
]:
    f, t = fails(N_, dt, trials=3000)
    print(f"N={N_:2d} {str(dt):15s}: {f}/{t} wrong")

print("\n== 3. Snap_eps as specified is discontinuous (not ReLU-representable) ==")


def snap_spec(x, eps=0.1):
    return torch.where(
        (x - 1).abs() <= eps,
        torch.ones_like(x),
        torch.where((x + 1).abs() <= eps, -torch.ones_like(x), x),
    )


xs = torch.tensor([0.8999, 0.9, 0.9001])
print(
    "spec values around 1-eps:",
    snap_spec(xs).tolist(),
    "-> jump of ~0.1 => no finite ReLU net can equal it",
)
k = 1 / 0.9  # continuous 2-ReLU restoring clamp used instead


def clamp_snap(x):
    return relu(k * x + 1) - relu(k * x - 1) - 1


print(
    "continuous clamp-snap at [-1.09,-0.95,0,0.95,1.09]:",
    clamp_snap(torch.tensor([-1.09, -0.95, 0.0, 0.95, 1.09])).tolist(),
)

print("\n== 4. Parabolic keys q_i.k_j = 2ij - j^2: margin 1 survives? ==")
for dt in (torch.bfloat16, torch.float32):
    worst = None
    for i in [10, 100, 1000, 4096, 5000, 16383]:
        js = torch.arange(max(0, i - 3), i + 4, dtype=torch.float64)
        s = (2 * i * js - js**2).to(dt)
        ok = int(js[s.double().argmax()]) == i and (s == s.max()).sum() == 1
        if not ok and worst is None:
            worst = i
    print(f"{dt}: first failing address = {worst}")

print("\n== 5. 'SwiGLU' product: SiLU(u)*v vs ReLU(u)*v, and bf16 ==")
u = torch.tensor([1.0, 2.0, 3.0, 255.0])
v = torch.tensor([7.0, 7.0, 7.0, 255.0])
print(
    "SiLU gate:",
    (torch.nn.functional.silu(u) * v).tolist(),
    " exact:",
    (u * v).tolist(),
)
print(
    "bf16 255*255 =",
    (
        torch.tensor(255.0, dtype=torch.bfloat16)
        * torch.tensor(255.0, dtype=torch.bfloat16)
    ).item(),
    "(exact 65025)",
)
print(
    "bf16 exact-integer limit 256; fp32 2^24. Column sums of 4 byte products need",
    math.ceil(math.log2(4 * 65025)),
    "bits",
)
# neurons for a single-layer ADC: bit j of x in [0, 2^K) is a periodic step fn with 2^(K-j-1) rising edges, 2 ReLU each
K = 18
print(
    f"single-FFN ADC for an {K}-bit scalar needs ~{sum(2 * 2 * 2 ** (K - j - 1) for j in range(K)):,} ReLU units"
)

print("\n== 6. Softmax lambda=10 leakage on Hamming-margin-2 argmax ==")
for n, lam in [(1024, 10), (2048, 10), (16384, 10), (1024, 5)]:
    ell = int(math.log2(n))
    leak = (n - 1) * math.exp(-2 * lam) / (1 + (n - 1) * math.exp(-2 * lam))
    print(f"n={n:5d} lam={lam}: worst-case mass leaked off target <= {leak:.2e}")

print("\n== 7. Banked RAM storage cost ==")
H, dk, dv = 64, 256, 64
for name, bytes_ in [("fp32", 4), ("bf16", 2)]:
    print(
        f"{name}: state = {H * dk * dv * bytes_ / 2**20:.1f} MiB to hold {H * dk * dv / 8 / 1024:.0f} KiB of logical data"
    )
