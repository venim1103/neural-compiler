"""Versioned JSON task specifications and case evidence, without code execution."""

from __future__ import annotations

import json
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict, replace
from fractions import Fraction
from typing import NoReturn, TextIO

import torch

from .harness import (
    ALL_OUTCOMES,
    Budgets,
    CallRecord,
    CaseResult,
    Checker,
    IntegerChecker,
    IntegerSequenceChecker,
    RationalChecker,
    Task,
    _PATHS,
    _parse_decimal_int,
)
from .serving import ALL_SERVING_MODES, ServingCaseResult

_LOGIT_DTYPES: dict[str, tuple[torch.dtype, int]] = {
    "float16": (torch.float16, 2),
    "bfloat16": (torch.bfloat16, 2),
    "float32": (torch.float32, 4),
    "float64": (torch.float64, 8),
}


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


def _decimal_reference(value: int) -> str:
    magnitude = abs(value)
    if not magnitude:
        return "0"
    chunks: list[int] = []
    while magnitude:
        magnitude, chunk = divmod(magnitude, 10**9)
        chunks.append(chunk)
    return (
        ("-" if value < 0 else "")
        + str(chunks[-1])
        + "".join(f"{chunk:09d}" for chunk in reversed(chunks[:-1]))
    )


def _reference_integer(value: object) -> int:
    raw = _string(value, "reference")
    if re.fullmatch(r"0|-?[1-9][0-9]*", raw) is None:
        raise ValueError("Reference must be a canonical decimal string")
    return _parse_decimal_int(raw)


def _task(value: object) -> Task:
    data = _object(
        value,
        "task",
        {"case_id", "family", "prompt", "answer_format", "checker"},
    )
    reference = _object(data["checker"], "checker", {"kind", "expected"})
    kind = _string(reference["kind"], "checker kind")
    expected = reference["expected"]
    checker: Checker
    if kind == "integer":
        checker = IntegerChecker(_reference_integer(expected))
    elif kind == "rational":
        fraction = _object(expected, "rational reference", {"numerator", "denominator"})
        denominator = _reference_integer(fraction["denominator"])
        if denominator <= 0:
            raise ValueError("Rational reference denominator must be positive")
        checker = RationalChecker(
            Fraction(_reference_integer(fraction["numerator"]), denominator)
        )
    elif kind == "integer_sequence":
        if not isinstance(expected, (list, tuple)):
            raise ValueError("Sequence reference must be a JSON array")
        checker = IntegerSequenceChecker(
            tuple(_reference_integer(item) for item in expected)
        )
    else:
        raise ValueError(f"Unknown task checker kind: {kind!r}")
    return Task(
        case_id=_string(data["case_id"], "case_id"),
        family=_string(data["family"], "family"),
        prompt=_string(data["prompt"], "prompt"),
        answer_format=_string(data["answer_format"], "answer_format"),
        checker=checker,
    )


def _task_data(task: Task) -> dict[str, object]:
    checker = task.checker
    reference: dict[str, object]
    if type(checker) is IntegerChecker:
        reference = {
            "kind": "integer",
            "expected": _decimal_reference(checker.expected),
        }
    elif type(checker) is RationalChecker:
        reference = {
            "kind": "rational",
            "expected": {
                "numerator": _decimal_reference(checker.expected.numerator),
                "denominator": _decimal_reference(checker.expected.denominator),
            },
        }
    elif type(checker) is IntegerSequenceChecker:
        reference = {
            "kind": "integer_sequence",
            "expected": [_decimal_reference(item) for item in checker.expected],
        }
    else:
        raise ValueError(f"Unsupported task checker: {type(checker).__name__}")
    return {
        "case_id": task.case_id,
        "family": task.family,
        "prompt": task.prompt,
        "answer_format": task.answer_format,
        "checker": reference,
    }


def _tasks(values: Sequence[object]) -> tuple[Task, ...]:
    tasks = tuple(_task(value) for value in values)
    if len({task.case_id for task in tasks}) != len(tasks):
        raise ValueError("Task specifications must have unique case IDs")
    return tasks


def _logits_data(value: torch.Tensor | None) -> dict[str, object] | None:
    if value is None:
        return None
    if (
        value.device.type != "cpu"
        or value.layout != torch.strided
        or value.requires_grad
    ):
        raise ValueError("Saved logits must be detached dense CPU tensors")
    dtype = next(
        (name for name, (dtype, _) in _LOGIT_DTYPES.items() if dtype == value.dtype),
        None,
    )
    if dtype is None:
        raise ValueError(f"Unsupported logit dtype: {value.dtype}")
    raw = value.detach().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
    return {
        "dtype": dtype,
        "shape": list(value.shape),
        "byte_order": sys.byteorder,
        "bytes_hex": memoryview(raw.numpy()).hex(),
    }


def _logits(value: object) -> torch.Tensor | None:
    if value is None:
        return None
    data = _object(value, "logits", {"dtype", "shape", "byte_order", "bytes_hex"})
    dtype_name = _string(data["dtype"], "logit dtype")
    dtype_info = _LOGIT_DTYPES.get(dtype_name)
    if dtype_info is None:
        raise ValueError(f"Unsupported logit dtype: {dtype_name}")
    if data["byte_order"] != sys.byteorder:
        raise ValueError("Unsupported logit byte order")
    dimensions = data["shape"]
    if not isinstance(dimensions, list):
        raise ValueError("Logit shape must be a JSON array")
    shape = tuple(_integer(size, "logit dimension") for size in dimensions)
    if any(size < 0 for size in shape):
        raise ValueError("Logit dimensions must be nonnegative")
    encoded = _string(data["bytes_hex"], "logit bytes_hex")
    if re.fullmatch(r"[0-9a-f]*", encoded) is None:
        raise ValueError("Logit bytes_hex must be lowercase hexadecimal")
    dtype, itemsize = dtype_info
    if len(encoded) != 2 * math.prod(shape) * itemsize:
        raise ValueError("Logit byte length differs from shape/dtype")
    raw = bytearray.fromhex(encoded)
    if not raw:
        return torch.empty(shape, dtype=dtype, device="cpu")
    return torch.frombuffer(raw, dtype=dtype).reshape(shape).clone()


def _serving_metadata(value: object) -> tuple[ServingCaseResult, object]:
    data = _object(
        value,
        "serving case",
        {
            "mode",
            "case",
            "final_tokens",
            "final_logits",
            "prefill_tokens",
            "checkpoint_bytes_peak",
            "cache_hits",
            "cache_misses",
            "checkpoints_evicted",
        },
        ("logit_prefix_tokens",),
    )
    mode = next((mode for mode in ALL_SERVING_MODES if mode == data["mode"]), None)
    if mode is None:
        raise ValueError(f"Unknown serving mode: {data['mode']!r}")
    tokens = data["final_tokens"]
    if not isinstance(tokens, (list, tuple)):
        raise ValueError("final_tokens must be a JSON array")
    prefix = data.get("logit_prefix_tokens")
    row = ServingCaseResult(
        mode=mode,
        case=_case(data["case"]),
        final_tokens=tuple(_integer(token, "final_token") for token in tokens),
        final_logits=None,
        prefill_tokens=_integer(data["prefill_tokens"], "prefill_tokens"),
        checkpoint_bytes_peak=_integer(
            data["checkpoint_bytes_peak"], "checkpoint_bytes_peak"
        ),
        cache_hits=_integer(data["cache_hits"], "cache_hits"),
        cache_misses=_integer(data["cache_misses"], "cache_misses"),
        checkpoints_evicted=_integer(
            data["checkpoints_evicted"], "checkpoints_evicted"
        ),
        logit_prefix_tokens=None
        if prefix is None
        else _integer(prefix, "logit_prefix_tokens"),
    )
    return row, data["final_logits"]


def _serving_case(value: object) -> ServingCaseResult:
    row, logits = _serving_metadata(value)
    return replace(row, final_logits=_logits(logits))


def _serving_data(row: ServingCaseResult) -> dict[str, object]:
    return {
        "mode": row.mode,
        "case": asdict(row.case),
        "final_tokens": list(row.final_tokens),
        "final_logits": _logits_data(row.final_logits),
        "prefill_tokens": row.prefill_tokens,
        "checkpoint_bytes_peak": row.checkpoint_bytes_peak,
        "cache_hits": row.cache_hits,
        "cache_misses": row.cache_misses,
        "checkpoints_evicted": row.checkpoints_evicted,
        "logit_prefix_tokens": row.logit_prefix_tokens,
    }


def _write_document(key: str, values: Sequence[object], destination: TextIO) -> None:
    document = (
        json.dumps(
            {"schema_version": 1, key: values},
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
        )
        + "\n"
    )
    if destination.write(document) != len(document):
        raise OSError("Incomplete evaluation evidence write")


def _read_document(key: str, source: TextIO) -> list[object]:
    value: object = json.load(
        source, object_pairs_hook=_unique_object, parse_constant=_reject_constant
    )
    data = _object(value, "evidence", {"schema_version", key})
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise ValueError(
            f"Unsupported evidence schema_version: {data['schema_version']!r}"
        )
    values = data[key]
    if not isinstance(values, list):
        raise ValueError(f"{key} must be a JSON array")
    return values


def dump_case_results(results: Sequence[CaseResult], destination: TextIO) -> None:
    """Write schema-v1 evidence to a caller-owned text stream, not an accepted dataset."""
    cases = [asdict(result) for result in results]
    for case in cases:
        _case(case)
    _write_document("cases", cases, destination)


def load_case_results(source: TextIO) -> tuple[CaseResult, ...]:
    """Read strict schema-v1 evidence and reconstruct validated nested records."""
    return tuple(_case(case) for case in _read_document("cases", source))


def dump_tasks(tasks: Sequence[Task], destination: TextIO) -> None:
    """Write exact built-in task specifications, never serialized callables."""
    records = [_task_data(task) for task in tasks]
    _tasks(records)
    _write_document("tasks", records, destination)


def load_tasks(source: TextIO) -> tuple[Task, ...]:
    """Read exact task specifications without accepting data or running checkers."""
    return _tasks(_read_document("tasks", source))


def dump_serving_results(
    results: Sequence[ServingCaseResult], destination: TextIO
) -> None:
    """Write serving observations and detached CPU logits, not live checkpoints."""
    records = [_serving_data(row) for row in results]
    for record in records:
        _serving_metadata(record)
    _write_document("serving_cases", records, destination)


def load_serving_results(source: TextIO) -> tuple[ServingCaseResult, ...]:
    """Read serving observations; logit bytes must use this host's byte order."""
    return tuple(
        _serving_case(record) for record in _read_document("serving_cases", source)
    )
