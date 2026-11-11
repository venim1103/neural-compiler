"""Baseline unit tests for neural_compiler primitives."""

from __future__ import annotations

import torch

from neural_compiler import gdn_exact_read, gdn_exact_write, snap_bipolar
from neural_compiler.cli import main


def test_snap_bipolar_restores_drifted_bits() -> None:
    drifted = torch.tensor([0.94, -0.96, 1.05, -1.04, 0.25])
    snapped = snap_bipolar(drifted, eps=0.1)
    expected = torch.tensor([1.0, -1.0, 1.0, -1.0, 0.25])
    assert torch.allclose(snapped, expected)


def test_gdn_exact_write_and_read_without_crosstalk() -> None:
    state = torch.zeros((8, 16), dtype=torch.float32)
    val_a = torch.tensor([1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
    val_b = torch.tensor([-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0])

    state = gdn_exact_write(state, 3, val_a)
    state = gdn_exact_write(state, 11, val_b)

    assert torch.equal(gdn_exact_read(state, 3), val_a)
    assert torch.equal(gdn_exact_read(state, 11), val_b)
    assert torch.equal(gdn_exact_read(state, 0), torch.zeros(8))


def test_cli_version(capsys) -> None:
    rc = main(["--version"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "0.1.0" in captured.out
