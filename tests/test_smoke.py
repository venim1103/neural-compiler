"""Baseline unit tests for neural_compiler primitives."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from neural_compiler import gdn_exact_read, gdn_exact_write, snap_bipolar
from neural_compiler.bridge import ExecutionResult
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


def test_cli_check_imports_and_native_flags(monkeypatch, tmp_path: Path, capsys) -> None:
    from neural_compiler import bridge

    clean = tmp_path / "clean.py"
    clean.write_text("import math\nprint(math.comb(10, 5))\n", encoding="utf-8")
    assert main(["--check-imports", str(clean)]) == 0
    clean_report = json.loads(capsys.readouterr().out)
    assert clean_report["compliant"] is True

    disallowed = tmp_path / "bad.py"
    disallowed.write_text("import subprocess\n", encoding="utf-8")
    assert main(["--check-imports", str(disallowed)]) == 1
    bad_report = json.loads(capsys.readouterr().out)
    assert bad_report["compliant"] is False
    assert bad_report["disallowed"] == [{"module": "subprocess", "line": 1}]

    recorded: list[tuple[str, str, float]] = []

    def fake_c(code: str, timeout: float = 20) -> ExecutionResult:
        recorded.append(("c", code, timeout))
        return ExecutionResult("ok", stdout="42\n", returncode=0)

    def fake_asm(code: str, timeout: float = 20) -> ExecutionResult:
        recorded.append(("asm", code, timeout))
        return ExecutionResult("compile_error", stderr="bad asm", returncode=1)

    monkeypatch.setattr(bridge, "run_c", fake_c)
    monkeypatch.setattr(bridge, "run_asm", fake_asm)

    c_file = tmp_path / "prog.c"
    c_file.write_text("int main(void){return 0;}", encoding="utf-8")
    assert main(["--timeout", "3.5", "--run-c", str(c_file)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"

    asm_file = tmp_path / "solve.s"
    asm_file.write_text("ret", encoding="utf-8")
    assert main(["--timeout", "2.0", "--run-asm", str(asm_file)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "compile_error"
    assert recorded == [
        ("c", "int main(void){return 0;}", 3.5),
        ("asm", "ret", 2.0),
    ]
