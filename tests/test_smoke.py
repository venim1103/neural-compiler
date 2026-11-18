"""Baseline unit tests for neural_compiler primitives."""

from __future__ import annotations

import ctypes
import json
from dataclasses import replace
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


@pytest.mark.parametrize(
    "runner,result",
    [
        ("run_wat", ExecutionResult("ok", "42\n", "", 0)),
        ("run_hex", ExecutionResult("timeout", "", "instruction limit", 3)),
    ],
)
@pytest.mark.parametrize("bounded", [False, True])
def test_cli_standalone_runtime_flags(
    monkeypatch, tmp_path: Path, capsys, runner, result, bounded
):
    from neural_compiler import bridge

    received = []

    def fake_runtime(code, timeout=20, **options):
        received.append((code, timeout, options))
        return result

    monkeypatch.setattr(bridge, runner, fake_runtime)
    source = tmp_path / "program"
    source.write_bytes(b"fixture\r\n")
    action = "--" + runner.replace("_", "-")
    flag, keyword = (
        ("--wat-fuel", "fuel")
        if runner == "run_wat"
        else ("--hex-max-insns", "max_insns")
    )
    arguments = [flag, "17"] if bounded else []
    assert main(["--timeout", "2.5", *arguments, action, str(source)]) == (
        0 if result.status == "ok" else 1
    )
    assert received == [("fixture\n", 2.5, {keyword: 17} if bounded else {})]
    assert json.loads(capsys.readouterr().out) == {
        "status": result.status,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


@pytest.mark.parametrize(
    "flag,action,counter",
    [
        ("--wat-fuel", "--run-wat", ctypes.c_uint64),
        ("--hex-max-insns", "--run-hex", ctypes.c_size_t),
    ],
)
@pytest.mark.parametrize("overflow", [False, True])
def test_cli_runtime_work_limits_fail_before_file_access(
    tmp_path: Path, monkeypatch, flag, action, counter, overflow
):
    value = 1 << (8 * ctypes.sizeof(counter)) if overflow else 0
    monkeypatch.setattr(
        Path,
        "open",
        lambda *args, **kwargs: pytest.fail("Invalid work limit read a file"),
    )
    with pytest.raises(ValueError, match=rf"{flag}.*(positive integer|counter)"):
        main([flag, str(value), action, str(tmp_path / "absent")])


@pytest.mark.parametrize(
    "flag,action,required",
    [
        ("--wat-fuel", "--run-hex", "--run-wat"),
        ("--hex-max-insns", "--run-wat", "--run-hex"),
    ],
)
def test_cli_runtime_work_options_require_matching_action(
    tmp_path: Path, capsys, flag, action, required
):
    with pytest.raises(SystemExit) as error:
        main([flag, "17", action, str(tmp_path / "absent")])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"{flag} requires {required}" in captured.err


@pytest.mark.parametrize(
    "with_tasks,serving,check_logits,with_development",
    [
        (False, False, False, False),
        (True, False, False, False),
        (True, False, False, True),
        (False, True, False, False),
        (True, True, False, False),
        (True, True, True, False),
        (True, True, True, True),
    ],
)
def test_cli_replays_failed_cases_without_execution(
    monkeypatch,
    tmp_path: Path,
    capsys,
    with_tasks,
    serving,
    check_logits,
    with_development,
):
    from neural_compiler import bridge
    from neural_compiler.bridge.session import LiveSession
    from neural_compiler.eval import (
        ALL_SERVING_MODES,
        CaseResult,
        IntegerChecker,
        ServingCaseResult,
        Task,
        dump_case_results,
        dump_serving_results,
        dump_tasks,
        matched_report,
        summarize_serving,
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
    if serving:
        serving_rows = tuple(
            ServingCaseResult(
                mode,
                rows[1],
                (1, 2, 3),
                torch.tensor(
                    [[-float("inf"), 1.0 if check_logits else float("nan"), 1.25]]
                ),
                3 + index,
                64 if index == 1 else 0,
                index,
                1,
                0,
                logit_prefix_tokens=2,
            )
            for index, mode in enumerate(ALL_SERVING_MODES)
        )
        with evidence.open("w", encoding="utf-8") as destination:
            dump_serving_results(serving_rows, destination)
        arguments = ["--serving-report", str(evidence)]
        report = summarize_serving(serving_rows)
    else:
        with evidence.open("w", encoding="utf-8") as destination:
            dump_case_results(rows, destination)
        arguments = ["--eval-report", str(evidence)]
        report = matched_report(rows)
    expected = json.loads(json.dumps(report, allow_nan=False))
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
    if with_development:
        development = tmp_path / "development.json"
        with development.open("w", encoding="utf-8") as destination:
            dump_tasks(
                [
                    Task(
                        "development",
                        "development",
                        "A unit fixture.",
                        IntegerChecker(99),
                    )
                ],
                destination,
            )
        arguments.extend(["--eval-development-tasks", str(development)])
        expected["task_overlap"] = {
            "development_tasks": 1,
            "evaluation_tasks": 1,
            "overlapping_case_ids": [],
            "overlapping_families": [],
            "evaluation_prompt_overlap_ids": ["case"],
        }
    if check_logits:
        arguments.append("--check-serving-logits")
        expected["logit_validation"] = {
            "cases": 1,
            "full_reprefill_comparisons": 2,
            "cache_live_exact_pairs": 1,
            "atol": 1e-6,
            "rtol": 1e-5,
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


def test_cli_rejects_unpaired_serving_without_report(tmp_path: Path, capsys):
    from neural_compiler.eval import (
        CaseResult,
        ServingCaseResult,
        dump_serving_results,
    )

    case = CaseResult("a", "unit", "python_tool", "correct", "", "42", 1, 3, 0, ())
    first = ServingCaseResult("live_pause_resume", case, (), None, 3, 0, 0, 0, 0)
    rows = [
        first,
        replace(first, mode="prefix_cached_resubmission"),
        replace(first, case=replace(case, case_id="b")),
    ]
    evidence = tmp_path / "unpaired.json"
    with evidence.open("w", encoding="utf-8") as destination:
        dump_serving_results(rows, destination)
    with pytest.raises(ValueError, match="Missing serving modes for b"):
        main(["--serving-report", str(evidence)])
    assert capsys.readouterr().out == ""


def test_cli_logit_mismatch_never_prints_a_report(tmp_path: Path, capsys):
    from neural_compiler.eval import (
        CaseResult,
        ServingCaseResult,
        dump_serving_results,
    )

    case = CaseResult("case", "unit", "python_tool", "correct", "", "42", 1, 3, 0, ())
    reference = ServingCaseResult(
        "full_reprefill_resubmission",
        case,
        (1, 2, 3),
        torch.tensor([[0.0, 1.0]]),
        3,
        0,
        0,
        0,
        0,
        logit_prefix_tokens=2,
    )
    live = replace(
        reference,
        mode="live_pause_resume",
        final_logits=torch.tensor([[2e-6, 1.0]]),
    )
    evidence = tmp_path / "mismatched.json"
    with evidence.open("w", encoding="utf-8") as destination:
        dump_serving_results([reference, live], destination)
    with pytest.raises(ValueError, match="Logits differ from full reprefill"):
        main(["--serving-report", str(evidence), "--check-serving-logits"])
    assert capsys.readouterr().out == ""


def test_cli_logit_checks_require_serving_report_before_file_access(
    tmp_path: Path, capsys
):
    absent = str(tmp_path / "absent")
    for action in (
        [],
        ["--version"],
        ["--eval-report", absent],
        ["--run-python", absent],
    ):
        with pytest.raises(SystemExit) as error:
            main([*action, "--check-serving-logits"])
        assert error.value.code == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "--check-serving-logits requires --serving-report" in captured.err


def test_cli_development_tasks_require_references_before_file_access(
    tmp_path: Path, capsys
):
    absent = str(tmp_path / "absent.json")
    for action in (
        [],
        ["--version"],
        ["--eval-report", absent],
        ["--serving-report", absent],
    ):
        with pytest.raises(SystemExit) as error:
            main([*action, "--eval-development-tasks", absent])
        assert error.value.code == 2
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "--eval-development-tasks requires --eval-tasks" in captured.err


def test_cli_invalid_development_tasks_never_print_a_report(tmp_path: Path, capsys):
    from neural_compiler.eval import (
        CaseResult,
        IntegerChecker,
        Task,
        dump_case_results,
        dump_tasks,
    )

    evidence = tmp_path / "cases.json"
    specification = tmp_path / "tasks.json"
    development = tmp_path / "development.json"
    rows = [
        CaseResult("case", "unit", path, "correct", "", "Final answer: 42", 1, 1, 0, ())
        for path in ("no_tool", "python_tool", "current_best")
    ]
    with evidence.open("w", encoding="utf-8") as destination:
        dump_case_results(rows, destination)
    with specification.open("w", encoding="utf-8") as destination:
        dump_tasks(
            [Task("case", "unit", "A fixture.", IntegerChecker(42))], destination
        )
    development.write_text('{"schema_version":2,"tasks":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported.*schema_version"):
        main(
            [
                "--eval-report",
                str(evidence),
                "--eval-tasks",
                str(specification),
                "--eval-development-tasks",
                str(development),
            ]
        )
    assert capsys.readouterr().out == ""
