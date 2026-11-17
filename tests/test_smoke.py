"""Baseline unit tests for neural_compiler primitives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
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


def test_cli_check_imports_and_native_flags(
    monkeypatch, tmp_path: Path, capsys
) -> None:
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

    recorded: list[tuple[str, str, float, str]] = []

    def fake_c(
        code: str, timeout: float = 20, *, compiler: str = "gcc"
    ) -> ExecutionResult:
        recorded.append(("c", code, timeout, compiler))
        return ExecutionResult("ok", stdout="42\n", returncode=0)

    def fake_asm(code: str, timeout: float = 20) -> ExecutionResult:
        recorded.append(("asm", code, timeout, ""))
        return ExecutionResult("compile_error", stderr="bad asm", returncode=1)

    monkeypatch.setattr(bridge, "run_c", fake_c)
    monkeypatch.setattr(bridge, "run_asm", fake_asm)

    c_file = tmp_path / "prog.c"
    c_file.write_text("int main(void){return 0;}", encoding="utf-8")
    assert main(["--timeout", "3.5", "--compiler", "tcc", "--run-c", str(c_file)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"

    asm_file = tmp_path / "solve.s"
    asm_file.write_text("ret", encoding="utf-8")
    assert main(["--timeout", "2.0", "--run-asm", str(asm_file)]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "compile_error"
    assert recorded == [
        ("c", "int main(void){return 0;}", 3.5, "tcc"),
        ("asm", "ret", 2.0, ""),
    ]


@pytest.mark.parametrize("with_tasks", [False, True])
def test_cli_replays_failed_cases_without_execution(
    monkeypatch, tmp_path: Path, capsys, with_tasks
):
    from neural_compiler import bridge
    from neural_compiler.bridge.session import LiveSession
    from neural_compiler.eval import (
        CaseResult,
        IntegerChecker,
        Task,
        dump_case_results,
        dump_tasks,
        matched_report,
    )

    def forbid_execution(*args, **kwargs):
        raise AssertionError("Evidence replay must not execute a model or program")

    for name in ("run_python", "run_c", "run_asm"):
        monkeypatch.setattr(bridge, name, forbid_execution)
    monkeypatch.setattr(LiveSession, "__init__", forbid_execution)
    rows = tuple(
        CaseResult(
            "case",
            "synthetic",
            path,
            "wrong_answer",
            "Mismatch",
            "Final answer: 0",
            2,
            3,
            0,
            (),
        )
        for path in ("no_tool", "python_tool", "current_best")
    )
    evidence = tmp_path / "cases.json"
    with evidence.open("w", encoding="utf-8") as destination:
        dump_case_results(rows, destination)
    arguments = ["--eval-report", str(evidence)]
    expected = json.loads(json.dumps(matched_report(rows), allow_nan=False))
    if with_tasks:
        specification = tmp_path / "tasks.json"
        task = Task("case", "synthetic", "A unit fixture.", IntegerChecker(42))
        with specification.open("w", encoding="utf-8") as destination:
            dump_tasks([task], destination)
        arguments.extend(["--eval-tasks", str(specification)])
        expected["reference_validation"] = {
            "tasks": 1,
            "records": 3,
            "answer_checks": 3,
            "task_text_checks": 0,
            "nonfinal_records": 0,
        }
    assert main(arguments) == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == expected
    assert captured.err == ""


def test_cli_rejects_invalid_evidence_without_report(tmp_path: Path, capsys):
    evidence = tmp_path / "invalid.json"
    evidence.write_text('{"schema_version":2,"cases":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported evidence schema_version"):
        main(["--eval-report", str(evidence)])
    assert capsys.readouterr().out == ""


def test_cli_rejects_incomplete_pairing_without_report(tmp_path: Path, capsys):
    from neural_compiler.eval import CaseResult, dump_case_results

    evidence = tmp_path / "incomplete.json"
    record = CaseResult(
        "case", "synthetic", "no_tool", "correct", "", "42", 1, 1, 0, ()
    )
    with evidence.open("w", encoding="utf-8") as destination:
        dump_case_results([record], destination)
    with pytest.raises(ValueError, match="Missing baseline/current-best result"):
        main(["--eval-report", str(evidence)])
    assert capsys.readouterr().out == ""


def test_cli_eval_tasks_requires_report_before_file_access(tmp_path: Path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["--eval-tasks", str(tmp_path / "absent.json")])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == "" and "--eval-tasks requires --eval-report" in captured.err


def test_cli_reference_mismatch_never_prints_a_report(tmp_path: Path, capsys):
    from neural_compiler.eval import (
        CaseResult,
        IntegerChecker,
        Task,
        dump_case_results,
        dump_tasks,
    )

    evidence, specification = tmp_path / "cases.json", tmp_path / "tasks.json"
    rows = [
        CaseResult(
            "case", "synthetic", path, "correct", "", "Final answer: 0", 1, 3, 0, ()
        )
        for path in ("no_tool", "python_tool", "current_best")
    ]
    with evidence.open("w", encoding="utf-8") as destination:
        dump_case_results(rows, destination)
    task = Task("case", "synthetic", "A unit fixture.", IntegerChecker(42))
    with specification.open("w", encoding="utf-8") as destination:
        dump_tasks([task], destination)
    with pytest.raises(ValueError, match="Recorded outcome differs from reference"):
        main(["--eval-report", str(evidence), "--eval-tasks", str(specification)])
    assert capsys.readouterr().out == ""
