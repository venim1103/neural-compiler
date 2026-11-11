"""Weight-compiled looped neural virtual machine and compiler toolkit."""

from __future__ import annotations

import torch

__version__ = "0.1.0"


def snap_bipolar(x: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    """Apply deadzone bipolar snapping to [-1, +1] within tolerance eps."""
    pos_mask = (x >= 1.0 - eps) & (x <= 1.0 + eps)
    neg_mask = (x >= -1.0 - eps) & (x <= -1.0 + eps)
    out = torch.where(pos_mask, torch.ones_like(x), x)
    return torch.where(neg_mask, -torch.ones_like(x), out)


def gdn_exact_write(
    state: torch.Tensor,
    address: int,
    value: torch.Tensor,
) -> torch.Tensor:
    """Perform an exact rank-one Gated DeltaNet row overwrite S_t = S_{t-1}(I - e_i e_i^T) + v_t e_i^T."""
    d_v, d_k = state.shape
    if not (0 <= address < d_k):
        raise ValueError(f"Address {address} out of range for d_k={d_k}")
    if value.shape != (d_v,):
        raise ValueError(f"Expected value shape ({d_v},), got {tuple(value.shape)}")
    e_i = torch.zeros(d_k, dtype=state.dtype, device=state.device)
    e_i[address] = 1.0
    proj = torch.eye(d_k, dtype=state.dtype, device=state.device) - torch.outer(e_i, e_i)
    return state @ proj + torch.outer(value, e_i)


def gdn_exact_read(state: torch.Tensor, address: int) -> torch.Tensor:
    """Read an exact memory row y_t = S_t e_i from a Gated DeltaNet state matrix."""
    _, d_k = state.shape
    if not (0 <= address < d_k):
        raise ValueError(f"Address {address} out of range for d_k={d_k}")
    e_i = torch.zeros(d_k, dtype=state.dtype, device=state.device)
    e_i[address] = 1.0
    return state @ e_i
