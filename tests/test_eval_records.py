"""Offline evidence replay, not accepted study data or model measurements."""

from __future__ import annotations

import io
import json
import sys
from dataclasses import asdict, replace
from fractions import Fraction

import pytest
import torch

from neural_compiler.eval import (
    ALL_SERVING_MODES,
    Budgets,
    CallRecord,
    CaseResult,
    IntegerChecker,
    IntegerSequenceChecker,
    RationalChecker,
    ServingCaseResult,
    Task,
    dump_case_results,
    dump_serving_results,
    dump_tasks,
    load_case_results,
    load_serving_results,
    load_tasks,
    matched_report,
    summarize_serving,
    validate_references,
)
from neural_compiler.eval.harness import _task_text_digest


def case_rows() -> tuple[CaseResult, ...]:
    rows = []
    for path in ("no_tool", "python_tool", "current_best"):
        calls = (
            () if path == "no_tool" else (CallRecord("runtime_error", 0.25, 2, 0.5),)
        )
        first = CaseResult(
            "measured",
            "unit",
            path,
            "correct",
            "",
            "Reasoning: \u00e9\nFinal answer: 42",
            2,
            4,
            sum(call.response_tokens for call in calls),
            calls,
            prompt_tokens=12,
            task_text_sha256="a" * 64,
            budgets=replace(Budgets(), calls=2),
        )
        legacy = replace(
            first,
            case_id="legacy",
            outcome="wrong_answer",
            diagnostic="Reference mismatch",
            answer="Final answer: 0",
            elapsed_seconds=3,
            prompt_tokens=None,
            task_text_sha256=None,
            budgets=None,
        )
        rows.extend((first, legacy))
    return tuple(rows)


def test_case_evidence_round_trip_preserves_failure_inclusive_report(tmp_path):
    rows = case_rows()
    path = tmp_path / "evidence.json"
    with path.open("w", encoding="utf-8") as destination:
        dump_case_results(rows, destination)
    with path.open(encoding="utf-8") as source:
        restored = load_case_results(source)
    assert restored == rows
    assert matched_report(restored) == matched_report(rows)
    assert restored[0].calls == () and restored[2].calls[0].status == "runtime_error"


def test_missing_optional_observations_do_not_impute_current_defaults():
    case = asdict(case_rows()[0])
    for name in ("budgets", "task_text_sha256", "prompt_tokens"):
        del case[name]
    restored = load_case_results(
        io.StringIO(json.dumps({"schema_version": 1, "cases": [case]}))
    )[0]
    assert restored.budgets is None
    assert restored.task_text_sha256 is None and restored.prompt_tokens is None


@pytest.mark.parametrize(
    "change,error",
    [
        ("version", "schema_version"),
        ("boolean_version", "schema_version"),
        ("unknown_field", "unknown"),
        ("missing_field", "missing"),
        ("boolean_count", "generated_tokens"),
        ("negative_time", "nonnegative"),
        ("infinite_time", "finite"),
        ("nonstring_answer", "answer"),
        ("unknown_path", "Unknown evaluation path"),
        ("partial_budgets", "budgets fields"),
        ("unknown_call_status", "Unknown execution status"),
        ("bad_response_count", "response_tokens"),
    ],
)
def test_invalid_case_evidence_fails_explicitly(change, error):
    case = asdict(case_rows()[2])
    document = {"schema_version": 1, "cases": [case]}
    if change == "version":
        document["schema_version"] = 2
    elif change == "boolean_version":
        document["schema_version"] = True
    elif change == "unknown_field":
        case["unexpected"] = "must not be discarded"
    elif change == "missing_field":
        del case["elapsed_seconds"]
    elif change == "boolean_count":
        case["generated_tokens"] = True
    elif change == "negative_time":
        case["elapsed_seconds"] = -1
    elif change == "infinite_time":
        case["elapsed_seconds"] = 1e309
    elif change == "nonstring_answer":
        case["answer"] = 42
    elif change == "unknown_path":
        case["path"] = "unlabelled"
    elif change == "partial_budgets":
        del case["budgets"]["calls"]
    elif change == "unknown_call_status":
        case["calls"][0]["status"] = "success"
    else:
        case["calls"][0]["response_tokens"] = 1.5
    with pytest.raises(ValueError, match=error):
        load_case_results(io.StringIO(json.dumps(document)))


def test_duplicate_json_fields_cannot_shadow_evidence():
    with pytest.raises(ValueError, match="Duplicate JSON field"):
        load_case_results(
            io.StringIO('{"schema_version":1,"schema_version":1,"cases":[]}')
        )


def test_nonfinite_json_constants_are_not_evidence():
    with pytest.raises(ValueError, match="Nonfinite JSON number"):
        load_case_results(io.StringIO('{"schema_version":1,"cases":[NaN]}'))


def test_invalid_record_is_rejected_before_writing():
    destination = io.StringIO()
    with pytest.raises(ValueError, match="answer must be a string"):
        dump_case_results([replace(case_rows()[0], answer=42)], destination)
    assert destination.getvalue() == ""


def test_incomplete_stream_write_is_explicit():
    class PartialWriter(io.StringIO):
        def write(self, text):
            return super().write(text[: len(text) // 2])

    with pytest.raises(OSError, match="Incomplete evaluation evidence write"):
        dump_case_results(case_rows(), PartialWriter())


@pytest.mark.parametrize(
    "checker,answer,answer_format",
    [
        (IntegerChecker(-(10**5000 + 7)), "-1" + "0" * 4999 + "7", "<integer>"),
        (RationalChecker(Fraction(3, 14)), "3/14", "<numerator>/<denominator>"),
        (
            IntegerSequenceChecker((-2, 0, 10**5000 + 7)),
            "-2, 0, 1" + "0" * 4999 + "7",
            "<n1>, <n2>, ...",
        ),
    ],
    ids=["large-negative-integer", "rational", "large-integer-sequence"],
)
def test_exact_task_specifications_round_trip(checker, answer, answer_format):
    task = Task("case", "synthetic", "A unit fixture.", checker, answer_format)
    original_limit = sys.get_int_max_str_digits()
    stream = io.StringIO()
    dump_tasks([task], stream)
    stream.seek(0)
    restored = load_tasks(stream)
    assert restored == (task,)
    assert restored[0].checker("Final answer: " + answer).correct
    assert sys.get_int_max_str_digits() == original_limit


@pytest.mark.parametrize(
    "change,error",
    [
        ("duplicate", "unique case IDs"),
        ("unknown_checker", "Unknown task checker kind"),
        ("boolean_reference", "reference must be a string"),
        ("noncanonical_reference", "canonical decimal"),
        ("zero_denominator", "denominator must be positive"),
        ("empty_sequence", "non-empty tuple"),
        ("unknown_field", "unknown"),
    ],
)
def test_invalid_task_specifications_are_explicit(change, error):
    task = {
        "case_id": "case",
        "family": "synthetic",
        "prompt": "Compute a unit fixture.",
        "answer_format": "<integer>",
        "checker": {"kind": "integer", "expected": "42"},
    }
    tasks = [task]
    if change == "duplicate":
        tasks.append(task)
    elif change == "unknown_checker":
        task["checker"]["kind"] = "module.callable"
    elif change == "boolean_reference":
        task["checker"]["expected"] = True
    elif change == "noncanonical_reference":
        task["checker"]["expected"] = "042"
    elif change == "zero_denominator":
        task["checker"] = {
            "kind": "rational",
            "expected": {"numerator": "1", "denominator": "0"},
        }
    elif change == "empty_sequence":
        task["checker"] = {"kind": "integer_sequence", "expected": []}
    else:
        task["executable"] = "must not be evaluated"
    document = json.dumps({"schema_version": 1, "tasks": tasks})
    with pytest.raises(ValueError, match=error):
        load_tasks(io.StringIO(document))


def test_opaque_checker_is_not_executed_or_written():
    task = Task(
        "case",
        "synthetic",
        "A unit fixture.",
        lambda answer: pytest.fail("A saved checker must never execute"),
    )
    stream = io.StringIO()
    with pytest.raises(ValueError, match="Unsupported task checker"):
        dump_tasks([task], stream)
    assert stream.getvalue() == ""


def test_duplicate_tasks_are_rejected_before_writing():
    task = Task("case", "synthetic", "A unit fixture.", IntegerChecker(42))
    stream = io.StringIO()
    with pytest.raises(ValueError, match="unique case IDs"):
        dump_tasks([task, task], stream)
    assert stream.getvalue() == ""


@pytest.mark.parametrize(
    "checker,answer,wrong",
    [
        (IntegerChecker(42), "42", "0"),
        (RationalChecker(Fraction(3, 14)), "3/14", "1/2"),
        (IntegerSequenceChecker((-2, 0, 9)), "-2, 0, 9", "1, 2, 3"),
    ],
)
def test_reference_validation_checks_all_final_categories(checker, answer, wrong):
    task = Task("case", "synthetic", "A unit fixture: \u00e9.", checker)
    row = CaseResult(
        "case",
        "synthetic",
        "no_tool",
        "correct",
        "",
        "Final answer: " + answer,
        1,
        3,
        0,
        (),
        task_text_sha256=_task_text_digest(task),
    )
    rows = [
        row,
        replace(
            row,
            path="python_tool",
            outcome="wrong_answer",
            answer="Final answer: " + wrong,
            task_text_sha256=None,
        ),
        replace(
            row,
            path="current_best",
            outcome="invalid_answer",
            answer="No final line",
            task_text_sha256=None,
        ),
    ]
    assert validate_references([task], rows) == {
        "tasks": 1,
        "records": 3,
        "answer_checks": 3,
        "task_text_checks": 1,
        "nonfinal_records": 0,
    }


@pytest.mark.parametrize(
    "fault,error",
    [
        ("correct_label", "Recorded outcome differs from reference"),
        ("wrong_label", "Recorded outcome differs from reference"),
        ("invalid_label", "Recorded outcome differs from reference"),
        ("family", "Family differs from specification"),
        ("text", "Task text differs from specification"),
        ("identity", "Task/result IDs differ"),
        ("duplicates", "unique case IDs"),
    ],
)
def test_reference_validation_rejects_inconsistent_evidence(fault, error):
    task = Task("case", "synthetic", "A unit fixture.", IntegerChecker(42))
    tasks = [task]
    row = CaseResult(
        "case", "synthetic", "no_tool", "correct", "", "Final answer: 42", 1, 3, 0, ()
    )
    if fault == "correct_label":
        row = replace(row, answer="Final answer: 0")
    elif fault == "wrong_label":
        row = replace(row, outcome="wrong_answer")
    elif fault == "invalid_label":
        row = replace(row, outcome="invalid_answer")
    elif fault == "family":
        row = replace(row, family="other")
    elif fault == "text":
        row = replace(row, task_text_sha256="b" * 64)
    elif fault == "identity":
        row = replace(row, case_id="other")
    else:
        tasks.append(task)
    with pytest.raises(ValueError, match=error):
        validate_references(tasks, [row])


def test_reference_validation_retains_nonfinal_failures():
    outcomes = ("protocol_error", "budget_exhausted", "timeout", "sandbox_error")
    tasks = [
        Task(str(index), "synthetic", "A unit fixture.", IntegerChecker(42))
        for index in range(len(outcomes))
    ]
    rows = [
        CaseResult(
            task.case_id,
            task.family,
            "no_tool",
            outcome,
            "",
            "Final answer: 0",
            1,
            3,
            0,
            (),
        )
        for task, outcome in zip(tasks, outcomes, strict=True)
    ]
    assert validate_references(tasks, rows) == {
        "tasks": 4,
        "records": 4,
        "answer_checks": 0,
        "task_text_checks": 0,
        "nonfinal_records": 4,
    }


def test_reference_validation_never_calls_an_opaque_checker():
    task = Task(
        "case",
        "synthetic",
        "A unit fixture.",
        lambda answer: pytest.fail("Opaque checker executed"),
    )
    row = CaseResult(
        "case", "synthetic", "no_tool", "correct", "", "Final answer: 42", 1, 3, 0, ()
    )
    with pytest.raises(ValueError, match="built-in exact checkers"):
        validate_references([task], [row])


def test_reference_validation_uses_each_cases_own_reference():
    tasks = [
        Task(str(value), "synthetic", f"Compute {value}.", IntegerChecker(value))
        for value in (41, 42)
    ]
    rows = [
        CaseResult(
            task.case_id,
            task.family,
            "no_tool",
            "correct",
            "",
            "Final answer: " + task.case_id,
            1,
            3,
            0,
            (),
        )
        for task in reversed(tasks)
    ]
    assert validate_references(tasks, rows)["answer_checks"] == 2


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_serving_observations_round_trip_bits_and_report(dtype):
    logits = torch.tensor([[-float("inf"), float("nan"), -0.0, 1.25]], dtype=dtype)
    case = case_rows()[2]
    rows = tuple(
        ServingCaseResult(
            mode,
            case,
            (1, 2, 3),
            logits,
            3 + index,
            64 if index == 1 else 0,
            index,
            1,
            0,
            logit_prefix_tokens=2,
        )
        for index, mode in enumerate(ALL_SERVING_MODES)
    )
    stream = io.StringIO()
    dump_serving_results(rows, stream)
    stream.seek(0)
    restored = load_serving_results(stream)
    for before, after in zip(rows, restored, strict=True):
        assert replace(before, final_logits=None) == replace(after, final_logits=None)
        before_logits, after_logits = before.final_logits, after.final_logits
        assert before_logits is not None and after_logits is not None
        assert after_logits.dtype == dtype
        assert after_logits.device.type == "cpu"
        assert torch.equal(
            before_logits.view(torch.uint8),
            after_logits.view(torch.uint8),
        )
    assert summarize_serving(restored) == summarize_serving(rows)


@pytest.mark.parametrize(
    "fault,error",
    [
        ("mode", "Unknown serving mode"),
        ("token", "final_token"),
        ("byte_order", "byte order"),
        ("shape", "byte length"),
        ("dtype", "Unsupported logit dtype"),
        ("hex", "hexadecimal"),
        ("counter", "prefill_tokens"),
    ],
)
def test_invalid_serving_observations_fail_before_tensor_reconstruction(
    monkeypatch, fault, error
):
    row = ServingCaseResult(
        "live_pause_resume",
        case_rows()[2],
        (1, 2, 3),
        torch.zeros((1, 4)),
        3,
        0,
        0,
        0,
        0,
    )
    stream = io.StringIO()
    dump_serving_results([row], stream)
    data = json.loads(stream.getvalue())
    record = data["serving_cases"][0]
    if fault == "mode":
        record["mode"] = "unknown"
    elif fault == "token":
        record["final_tokens"][0] = True
    elif fault == "byte_order":
        record["final_logits"]["byte_order"] = (
            "big" if sys.byteorder == "little" else "little"
        )
    elif fault == "shape":
        record["final_logits"]["shape"] = [1, 5]
    elif fault == "dtype":
        record["final_logits"]["dtype"] = "unreviewed"
    elif fault == "hex":
        record["final_logits"]["bytes_hex"] = "z" * 32
    else:
        record["prefill_tokens"] = -1
    monkeypatch.setattr(
        torch,
        "frombuffer",
        lambda *args, **kwargs: pytest.fail("Invalid metadata reconstructed a tensor"),
    )
    with pytest.raises(ValueError, match=error):
        load_serving_results(io.StringIO(json.dumps(data)))


@pytest.mark.parametrize("logits", [None, torch.empty((1, 0))])
def test_unmeasured_or_empty_serving_logits_stay_distinct(logits):
    row = ServingCaseResult(
        "live_pause_resume",
        case_rows()[2],
        (),
        logits,
        0,
        0,
        0,
        0,
        0,
    )
    stream = io.StringIO()
    dump_serving_results([row], stream)
    stream.seek(0)
    restored = load_serving_results(stream)[0]
    if logits is None:
        assert restored == row
    else:
        assert restored.final_logits is not None
        assert restored.final_logits.shape == logits.shape
        assert restored.logit_prefix_tokens is None


def test_non_cpu_logits_are_rejected_without_writing():
    row = ServingCaseResult(
        "live_pause_resume",
        case_rows()[2],
        (1, 2, 3),
        torch.empty((1, 4), device="meta"),
        3,
        0,
        0,
        0,
        0,
    )
    stream = io.StringIO()
    with pytest.raises(ValueError, match="detached dense CPU"):
        dump_serving_results([row], stream)
    assert stream.getvalue() == ""
