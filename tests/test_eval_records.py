"""Offline evidence replay, not accepted study data or model measurements."""

from __future__ import annotations

import io
import json
from dataclasses import asdict, replace

import pytest

from neural_compiler.eval import (
    Budgets,
    CallRecord,
    CaseResult,
    dump_case_results,
    load_case_results,
    matched_report,
)


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
