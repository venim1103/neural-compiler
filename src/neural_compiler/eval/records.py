"""Versioned JSON case evidence; loading never executes tasks or model code."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict
from typing import NoReturn, TextIO

from .harness import ALL_OUTCOMES, Budgets, CallRecord, CaseResult, _PATHS


def _unique_object(items: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in items:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"Nonfinite JSON number: {value}")


def _object(
    value: object, label: str, required: set[str], optional: Sequence[str] = ()
) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    record: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} field names must be strings")
        record[key] = item
    missing = required - record.keys()
    unknown = record.keys() - required - set(optional)
    if missing or unknown:
        raise ValueError(
            f"{label} fields invalid (missing: {sorted(missing)}, unknown: {sorted(unknown)})"
        )
    return record


def _string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    return value


def _integer(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not finite:
        raise ValueError(f"{label} must be a finite number")
    return value


def _budgets(value: object) -> Budgets:
    data = _object(
        value,
        "budgets",
        {
            "generated_tokens",
            "turn_tokens",
            "call_body_tokens",
            "response_tokens",
            "calls",
            "prompt_tokens",
            "context_tokens",
            "wall_seconds",
            "execution_seconds",
        },
    )
    return Budgets(
        generated_tokens=_integer(data["generated_tokens"], "generated_tokens"),
        turn_tokens=_integer(data["turn_tokens"], "turn_tokens"),
        call_body_tokens=_integer(data["call_body_tokens"], "call_body_tokens"),
        response_tokens=_integer(data["response_tokens"], "response_tokens"),
        calls=_integer(data["calls"], "calls"),
        prompt_tokens=_integer(data["prompt_tokens"], "prompt_tokens"),
        context_tokens=_integer(data["context_tokens"], "context_tokens"),
        wall_seconds=_number(data["wall_seconds"], "wall_seconds"),
        execution_seconds=_number(data["execution_seconds"], "execution_seconds"),
    )


def _call(value: object) -> CallRecord:
    data = _object(
        value,
        "call",
        {"status", "execution_seconds", "response_tokens"},
        ("response_seconds",),
    )
    response_seconds = data.get("response_seconds")
    return CallRecord(
        status=_string(data["status"], "status"),
        execution_seconds=_number(data["execution_seconds"], "execution_seconds"),
        response_tokens=_integer(data["response_tokens"], "response_tokens"),
        response_seconds=None
        if response_seconds is None
        else _number(response_seconds, "response_seconds"),
    )


def _case(value: object) -> CaseResult:
    data = _object(
        value,
        "case",
        {
            "case_id",
            "family",
            "path",
            "outcome",
            "diagnostic",
            "answer",
            "elapsed_seconds",
            "generated_tokens",
            "injected_tokens",
            "calls",
        },
        ("prompt_tokens", "task_text_sha256", "budgets"),
    )
    path = next((path for path in _PATHS if path == data["path"]), None)
    if path is None:
        raise ValueError(f"Unknown evaluation path: {data['path']!r}")
    outcome = next(
        (outcome for outcome in ALL_OUTCOMES if outcome == data["outcome"]), None
    )
    if outcome is None:
        raise ValueError(f"Unknown evaluation outcome: {data['outcome']!r}")
    calls = data["calls"]
    if not isinstance(calls, (list, tuple)):
        raise ValueError("calls must be a JSON array")
    prompt_tokens = data.get("prompt_tokens")
    digest = data.get("task_text_sha256")
    budgets = data.get("budgets")
    return CaseResult(
        case_id=_string(data["case_id"], "case_id"),
        family=_string(data["family"], "family"),
        path=path,
        outcome=outcome,
        diagnostic=_string(data["diagnostic"], "diagnostic"),
        answer=_string(data["answer"], "answer"),
        elapsed_seconds=_number(data["elapsed_seconds"], "elapsed_seconds"),
        generated_tokens=_integer(data["generated_tokens"], "generated_tokens"),
        injected_tokens=_integer(data["injected_tokens"], "injected_tokens"),
        calls=tuple(_call(call) for call in calls),
        prompt_tokens=None
        if prompt_tokens is None
        else _integer(prompt_tokens, "prompt_tokens"),
        task_text_sha256=None
        if digest is None
        else _string(digest, "task_text_sha256"),
        budgets=None if budgets is None else _budgets(budgets),
    )


def dump_case_results(results: Sequence[CaseResult], destination: TextIO) -> None:
    """Write schema-v1 evidence to a caller-owned text stream, not an accepted dataset."""
    cases = [asdict(result) for result in results]
    for case in cases:
        _case(case)
    document = (
        json.dumps(
            {"schema_version": 1, "cases": cases},
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
    )
    if destination.write(document) != len(document):
        raise OSError("Incomplete evaluation evidence write")


def load_case_results(source: TextIO) -> tuple[CaseResult, ...]:
    """Read strict schema-v1 evidence and reconstruct validated nested records."""
    value: object = json.load(
        source, object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    data = _object(value, "evidence", {"schema_version", "cases"})
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError(
            f"Unsupported evidence schema_version: {data['schema_version']!r}"
        )
    cases = data["cases"]
    if not isinstance(cases, list):
        raise ValueError("cases must be a JSON array")
    return tuple(_case(case) for case in cases)
