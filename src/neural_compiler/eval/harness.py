"""Matched evaluation contracts; this module does not load models or task data."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal, Protocol

from transformers import PreTrainedTokenizerBase

from neural_compiler.bridge import ExecutionResult, run_python
from neural_compiler.bridge.execution import _validate_execution_status
from neural_compiler.bridge.protocol import (
    RUN_TOOL,
    ProtocolError,
    QwenToolProtocol,
    ToolCall,
)
from neural_compiler.bridge.session import (
    CheckpointBudgetError,
    DecodeResult,
    _validate_token_ids,
)

Path = Literal["no_tool", "python_tool", "current_best"]
_PATHS: tuple[Path, ...] = ("no_tool", "python_tool", "current_best")
Outcome = Literal[
    "correct",
    "wrong_answer",
    "invalid_answer",
    "protocol_error",
    "budget_exhausted",
    "timeout",
    "sandbox_error",
]
ALL_OUTCOMES: tuple[Outcome, ...] = (
    "correct",
    "wrong_answer",
    "invalid_answer",
    "protocol_error",
    "budget_exhausted",
    "timeout",
    "sandbox_error",
)
_FINAL = re.compile(r"Final answer:\s*([+-]?[0-9]+)\s*")
_FINAL_RATIONAL = re.compile(
    r"Final answer:\s*([+-]?[0-9]+)(?:\s*/\s*([+-]?[0-9]+))?\s*"
)
_FINAL_SEQUENCE = re.compile(r"Final answer:\s*([+-]?[0-9]+(?:\s*,\s*[+-]?[0-9]+)*)\s*")


def _parse_decimal_int(raw: str) -> int:
    magnitude = raw.lstrip("+-").lstrip("0") or "0"
    parsed = 0
    # Small decimal chunks preserve the interpreter's global conversion limit.
    for offset in range(0, len(magnitude), 9):
        chunk = magnitude[offset : offset + 9]
        parsed = parsed * 10 ** len(chunk) + int(chunk)
    return -parsed if raw.startswith("-") else parsed


@dataclass(frozen=True)
class CheckResult:
    """Semantic correctness and final-answer-format validity."""

    correct: bool
    valid: bool
    diagnostic: str

    def __post_init__(self) -> None:
        if type(self.correct) is not bool or type(self.valid) is not bool:
            raise ValueError("Checker correctness and validity must be booleans")


def _answer_outcome(checked: CheckResult) -> Outcome:
    return (
        "invalid_answer"
        if not checked.valid
        else ("correct" if checked.correct else "wrong_answer")
    )


class Checker(Protocol):
    def __call__(self, answer: str) -> CheckResult: ...


@dataclass(frozen=True)
class IntegerChecker:
    expected: int

    def __post_init__(self) -> None:
        if type(self.expected) is not int:
            raise ValueError(
                "IntegerChecker requires an integer reference, not a boolean"
            )

    def __call__(self, answer: str) -> CheckResult:
        lines = answer.rstrip().splitlines()
        match = _FINAL.fullmatch(lines[-1]) if lines else None
        if match is None:
            return CheckResult(
                False,
                False,
                "Expected a complete final integer answer on the last line",
            )
        parsed = _parse_decimal_int(match.group(1))
        correct = parsed == self.expected
        return CheckResult(
            correct, True, "" if correct else "Final integer differs from the reference"
        )


@dataclass(frozen=True)
class RationalChecker:
    expected: Fraction

    def __init__(self, expected: Fraction | int) -> None:
        if type(expected) is bool or not isinstance(expected, (Fraction, int)):
            raise ValueError(
                "RationalChecker requires a Fraction or integer reference, not a boolean"
            )
        object.__setattr__(self, "expected", Fraction(expected))

    def __call__(self, answer: str) -> CheckResult:
        lines = answer.rstrip().splitlines()
        match = _FINAL_RATIONAL.fullmatch(lines[-1]) if lines else None
        if match is None:
            return CheckResult(
                False,
                False,
                "Expected a complete final rational answer on the last line",
            )
        numerator = _parse_decimal_int(match.group(1))
        denominator = (
            1 if match.group(2) is None else _parse_decimal_int(match.group(2))
        )
        if denominator == 0:
            return CheckResult(
                False,
                False,
                "Rational denominator must not be zero",
            )
        parsed = Fraction(numerator, denominator)
        correct = parsed == self.expected
        return CheckResult(
            correct,
            True,
            "" if correct else "Final rational differs from the reference",
        )


@dataclass(frozen=True)
class IntegerSequenceChecker:
    expected: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.expected, tuple)
            or not self.expected
            or any(type(item) is not int for item in self.expected)
        ):
            raise ValueError(
                "IntegerSequenceChecker requires a non-empty tuple of integers"
            )

    def __call__(self, answer: str) -> CheckResult:
        lines = answer.rstrip().splitlines()
        match = _FINAL_SEQUENCE.fullmatch(lines[-1]) if lines else None
        if match is None:
            return CheckResult(
                False,
                False,
                "Expected a complete final integer sequence on the last line",
            )
        parts = [part.strip() for part in match.group(1).split(",")]
        parsed = tuple(_parse_decimal_int(part) for part in parts)
        if len(parsed) != len(self.expected):
            return CheckResult(
                False,
                False,
                f"Expected {len(self.expected)} integers, got {len(parsed)}",
            )
        correct = parsed == self.expected
        return CheckResult(
            correct,
            True,
            "" if correct else "Final integer sequence differs from the reference",
        )


@dataclass(frozen=True)
class Task:
    case_id: str
    family: str
    prompt: str
    checker: Checker
    answer_format: str = "<integer>"

    def __post_init__(self) -> None:
        if (
            not self.case_id.strip()
            or not self.family.strip()
            or not self.prompt.strip()
        ):
            raise ValueError("Task identity, family and prompt must not be empty")
        if not self.answer_format.strip():
            raise ValueError("Task answer_format must not be empty")


def _task_text_digest(task: Task) -> str:
    return hashlib.sha256(
        json.dumps(
            (task.prompt, task.answer_format),
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True)
class Budgets:
    generated_tokens: int = 2048
    turn_tokens: int = 2048
    call_body_tokens: int = 4096
    response_tokens: int = 2048
    calls: int = 5
    prompt_tokens: int = 8192
    context_tokens: int = 16384
    wall_seconds: float = 60
    execution_seconds: float = 20

    def __post_init__(self) -> None:
        for name in (
            "generated_tokens",
            "turn_tokens",
            "call_body_tokens",
            "response_tokens",
            "calls",
            "prompt_tokens",
            "context_tokens",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("wall_seconds", "execution_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class DecodeSession(Protocol):
    """Invoke a supplied callback once per sampled ID and honor its stop.

    Returned IDs must match that callback stream within the requested limit.
    The stop reason denotes a callback stop; EOS and length are distinct.
    """

    def decode_until(
        self,
        max_new_tokens: int,
        on_token: Callable[[int], bool] | None = None,
    ) -> DecodeResult: ...

    def inject(self, response_ids: Sequence[int]) -> None: ...


class SessionFactory(Protocol):
    def __call__(self, prompt_ids: Sequence[int]) -> DecodeSession: ...


class Executor(Protocol):
    def __call__(self, code: str, timeout: float = 20) -> ExecutionResult: ...


@dataclass(frozen=True)
class CallRecord:
    status: str
    execution_seconds: float
    response_tokens: int
    response_seconds: float | None = None

    def __post_init__(self) -> None:
        _validate_execution_status(self.status)
        if (
            isinstance(self.execution_seconds, bool)
            or not math.isfinite(self.execution_seconds)
            or self.execution_seconds < 0
        ):
            raise ValueError("Call execution time must be finite and nonnegative")
        if type(self.response_tokens) is not int or self.response_tokens < 0:
            raise ValueError("response_tokens must be a nonnegative integer")
        if self.response_seconds is not None and (
            isinstance(self.response_seconds, bool)
            or not math.isfinite(self.response_seconds)
            or self.response_seconds < 0
        ):
            raise ValueError("Call response time must be finite and nonnegative")


@dataclass(frozen=True)
class CaseResult:
    """Observed task text and configured budgets do not certify runtime matching."""

    case_id: str
    family: str
    path: Path
    outcome: Outcome
    diagnostic: str
    answer: str
    elapsed_seconds: float
    generated_tokens: int
    injected_tokens: int
    calls: tuple[CallRecord, ...]
    prompt_tokens: int | None = None
    task_text_sha256: str | None = None
    budgets: Budgets | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.elapsed_seconds, bool)
            or not math.isfinite(self.elapsed_seconds)
            or self.elapsed_seconds < 0
        ):
            raise ValueError("Case elapsed time must be finite and nonnegative")
        if self.path not in _PATHS:
            raise ValueError(f"Unknown evaluation path: {self.path}")
        if self.outcome not in ALL_OUTCOMES:
            raise ValueError(f"Unknown evaluation outcome: {self.outcome}")
        for name in ("generated_tokens", "injected_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.prompt_tokens is not None and (
            type(self.prompt_tokens) is not int or self.prompt_tokens < 0
        ):
            raise ValueError("prompt_tokens must be a nonnegative integer or None")
        if self.task_text_sha256 is not None and (
            not isinstance(self.task_text_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.task_text_sha256) is None
        ):
            raise ValueError(
                "task_text_sha256 must be a lowercase SHA-256 digest or None"
            )
        if self.budgets is not None and not isinstance(self.budgets, Budgets):
            raise ValueError("budgets must be a Budgets instance or None")
        if not isinstance(self.calls, tuple) or any(
            not isinstance(call, CallRecord) for call in self.calls
        ):
            raise ValueError("calls must be a tuple of CallRecord instances")
        if self.path == "no_tool" and (self.calls or self.injected_tokens):
            raise ValueError(
                "no_tool results must not record tool calls or injected tokens"
            )
        if self.injected_tokens != sum(call.response_tokens for call in self.calls):
            raise ValueError(
                "injected_tokens must equal the sum of call response_tokens"
            )


class _RunFailure(Exception):
    def __init__(self, outcome: Outcome, diagnostic: str):
        super().__init__(diagnostic)
        self.outcome = outcome


def _validate_tool_description(description: str | None) -> None:
    if description is not None and (
        not isinstance(description, str) or not description.strip()
    ):
        raise ValueError("tool_description must be a non-empty string")


class EvaluationHarness:
    """Same tasks, prompts, checker and budgets across caller-labelled paths.

    The caller supplies an already-loaded session factory and synchronization
    function (CUDA synchronize for GPU timing). Each attempted result return
    includes framing, context admission, injection and synchronization in its
    response time, including failed attempts. Unattempted returns are unmeasured.
    Model deadlines are cooperative; only the supervised executor supplies a
    hard subprocess deadline. An optional tool description declares a configured
    executor without changing the supported run(code) schema or Python default.
    Invalid decoder providers raise explicitly without producing scored records.
    Answer text excludes only the sampled terminal EOS, not other special tokens.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        session_factory: SessionFactory,
        synchronize: Callable[[], None],
        budgets: Budgets = Budgets(),
        executor: Executor = run_python,
        *,
        tool_description: str | None = None,
    ):
        _validate_tool_description(tool_description)
        self.tokenizer = tokenizer
        self.session_factory = session_factory
        self.synchronize = synchronize
        self.budgets = budgets
        self.executor = executor
        self._run_tool = deepcopy(RUN_TOOL)
        if tool_description is not None:
            self._run_tool["function"]["description"] = tool_description

    def run(self, task: Task, path: Path) -> CaseResult:
        if path not in _PATHS:
            raise ValueError(f"Unknown evaluation path: {path}")
        self.synchronize()
        started = time.perf_counter()
        calls: list[CallRecord] = []
        generated = injected = 0
        prompt_count = 0
        prompt_tokens: int | None = None
        answer = ""
        outcome: Outcome = "invalid_answer"
        diagnostic = ""
        task_text_sha256 = _task_text_digest(task)
        call: ToolCall | None = None
        observed_tokens: list[int] = []
        turn_limit = 0
        protocol = QwenToolProtocol(
            self.tokenizer,
            self.budgets.call_body_tokens,
            self.budgets.response_tokens,
        )

        def remaining_time() -> float:
            remaining = self.budgets.wall_seconds - (time.perf_counter() - started)
            if remaining <= 0:
                raise _RunFailure("timeout", "Evaluation wall-clock budget exhausted")
            return remaining

        def on_token(token: int) -> bool:
            nonlocal generated, call
            _validate_token_ids((token,))
            observed_tokens.append(token)
            generated += 1
            if len(observed_tokens) > turn_limit:
                raise RuntimeError("DecodeSession exceeded the requested turn limit")
            if call is not None:
                raise RuntimeError("DecodeSession continued after a callback stop")
            remaining_time()
            call = protocol.feed(token)
            if path == "no_tool" and call is not None:
                raise ProtocolError("Tool calls are disabled for the no-tool baseline")
            return call is not None

        try:
            system_prompt = f"Solve the task. End your response with Final answer: {task.answer_format}."
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task.prompt},
                ],
                tools=None if path == "no_tool" else [self._run_tool],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if (
                not isinstance(prompt, list)
                or not prompt
                or not all(type(token) is int for token in prompt)
            ):
                raise ProtocolError("Chat template must return one token sequence")
            prompt_count = len(prompt)
            prompt_tokens = prompt_count
            if prompt_count > min(
                self.budgets.prompt_tokens, self.budgets.context_tokens
            ):
                raise _RunFailure(
                    "budget_exhausted",
                    "Prompt-token budget exhausted before model prefill",
                )
            remaining_time()
            if prompt_count == self.budgets.context_tokens:
                raise _RunFailure(
                    "budget_exhausted",
                    "Prompt leaves no context-token capacity for a final answer",
                )
            session = self.session_factory(prompt)
            while generated < self.budgets.generated_tokens:
                remaining_time()
                context_remaining = (
                    self.budgets.context_tokens - prompt_count - generated - injected
                )
                call = None
                observed_tokens = []
                generation_limit = min(
                    self.budgets.turn_tokens,
                    self.budgets.generated_tokens - generated,
                )
                turn_limit = min(generation_limit, context_remaining)
                decoded = session.decode_until(turn_limit, on_token)
                self.synchronize()
                remaining_time()
                if decoded.reason not in ("stop", "eos", "length"):
                    raise RuntimeError(
                        "DecodeSession returned an unsupported finish reason"
                    )
                if len(decoded.token_ids) != len(observed_tokens):
                    raise RuntimeError(
                        "DecodeSession returned IDs differ from callbacks"
                    )
                _validate_token_ids(decoded.token_ids)
                if tuple(decoded.token_ids) != tuple(observed_tokens):
                    raise RuntimeError(
                        "DecodeSession returned IDs differ from callbacks"
                    )
                if (decoded.reason == "stop") != (call is not None):
                    raise RuntimeError(
                        "DecodeSession finish reason disagrees with callback stop"
                    )
                if decoded.reason == "eos" and not decoded.token_ids:
                    raise RuntimeError("DecodeSession reported EOS without a token")
                if decoded.reason == "length":
                    if len(observed_tokens) != turn_limit:
                        raise RuntimeError(
                            "DecodeSession reported length before the requested turn limit"
                        )
                    answer = self.tokenizer.decode(
                        decoded.token_ids, skip_special_tokens=False
                    )
                    raise _RunFailure(
                        "budget_exhausted",
                        "Response turn ended at the context-token limit"
                        if context_remaining < generation_limit
                        else "Response turn ended at the generation-token limit",
                    )
                protocol.finish()
                if call is None:
                    answer = self.tokenizer.decode(
                        decoded.token_ids[:-1], skip_special_tokens=False
                    )
                    checked = task.checker(answer)
                    outcome = _answer_outcome(checked)
                    diagnostic = checked.diagnostic
                    break
                if len(calls) >= self.budgets.calls:
                    raise _RunFailure(
                        "budget_exhausted", "Serial tool-call budget exhausted"
                    )
                if generated >= self.budgets.generated_tokens:
                    raise _RunFailure(
                        "budget_exhausted",
                        "No generation tokens remain for a final answer",
                    )
                if len(observed_tokens) == context_remaining:
                    raise _RunFailure(
                        "budget_exhausted",
                        "No context-token capacity remains for a tool response",
                    )
                exec_start = time.perf_counter()
                result = self.executor(
                    call.code,
                    timeout=min(self.budgets.execution_seconds, remaining_time()),
                )
                execution_time = time.perf_counter() - exec_start
                calls.append(CallRecord(result.status, execution_time, 0))
                remaining_time()
                if result.status == "sandbox_error":
                    raise _RunFailure("sandbox_error", result.stderr)
                response_start = time.perf_counter()
                response_count = 0
                try:
                    response = protocol.response_ids(call, result)
                    reply_capacity = context_remaining - len(observed_tokens)
                    if len(response) > reply_capacity:
                        raise _RunFailure(
                            "budget_exhausted",
                            "Tool response would exceed the context-token budget",
                        )
                    if len(response) == reply_capacity:
                        raise _RunFailure(
                            "budget_exhausted",
                            "Tool response leaves no context-token capacity for a final answer",
                        )
                    remaining_time()
                    session.inject(response)
                    response_count = len(response)
                    injected += response_count
                finally:
                    self.synchronize()
                    calls[-1] = CallRecord(
                        result.status,
                        execution_time,
                        response_count,
                        time.perf_counter() - response_start,
                    )
            else:
                raise _RunFailure(
                    "budget_exhausted", "Total generation-token budget exhausted"
                )
        except ProtocolError as exc:
            outcome, diagnostic = "protocol_error", str(exc)
        except CheckpointBudgetError as exc:
            outcome, diagnostic = "budget_exhausted", str(exc)
        except _RunFailure as exc:
            outcome, diagnostic = exc.outcome, str(exc)
        finally:
            self.synchronize()
        elapsed = time.perf_counter() - started
        if (
            outcome in ("correct", "wrong_answer", "invalid_answer")
            and elapsed >= self.budgets.wall_seconds
        ):
            outcome, diagnostic = "timeout", "Evaluation wall-clock budget exhausted"
        return CaseResult(
            task.case_id,
            task.family,
            path,
            outcome,
            diagnostic,
            answer,
            elapsed,
            generated,
            injected,
            tuple(calls),
            prompt_tokens=prompt_tokens,
            task_text_sha256=task_text_sha256,
            budgets=self.budgets,
        )


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = q * (len(ordered) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _wilson_interval(
    correct: int, total: int, z: float = 1.959963984540054
) -> tuple[float, float] | None:
    if total <= 0:
        return None
    p = correct / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _bootstrap_seconds_per_correct_ci(
    results: Sequence[CaseResult], samples: int = 1000, seed: int = 0
) -> tuple[float, float] | None:
    n = len(results)
    if n == 0 or not any(r.outcome == "correct" for r in results):
        return None
    rng = random.Random(seed)
    elapsed_vals = [r.elapsed_seconds for r in results]
    correct_vals = [1 if r.outcome == "correct" else 0 for r in results]
    ratios: list[float] = []
    for _ in range(samples):
        tot_elapsed = 0.0
        tot_correct = 0
        for _ in range(n):
            idx = rng.randrange(n)
            tot_elapsed += elapsed_vals[idx]
            tot_correct += correct_vals[idx]
        ratios.append(tot_elapsed / tot_correct if tot_correct else math.inf)
    low = _quantile(ratios, 0.025)
    high = _quantile(ratios, 0.975)
    assert low is not None and high is not None
    return low, high


def summarize(results: Sequence[CaseResult]) -> dict[str, object]:
    """Failure-inclusive aggregate with confidence intervals stable under row reordering."""
    if len({result.path for result in results}) > 1:
        raise ValueError("Summarize one evaluation path at a time")
    if len({result.case_id for result in results}) != len(results):
        raise ValueError("Duplicate case IDs cannot be counted as independent attempts")
    # The seeded bootstrap draws positions, so assign them independently of row order.
    results = sorted(results, key=lambda result: result.case_id)
    total = len(results)
    correct = sum(result.outcome == "correct" for result in results)
    elapsed_list = [result.elapsed_seconds for result in results]
    elapsed = sum(elapsed_list)
    generated_tokens = sum(result.generated_tokens for result in results)
    injected_tokens = sum(result.injected_tokens for result in results)
    prompt_list = [
        result.prompt_tokens for result in results if result.prompt_tokens is not None
    ]
    tool_calls = sum(len(result.calls) for result in results)
    exec_list = [call.execution_seconds for result in results for call in result.calls]
    execution_seconds = sum(exec_list)
    response_list = [
        call.response_seconds
        for result in results
        for call in result.calls
        if call.response_seconds is not None
    ]
    outcomes = {
        outcome: sum(result.outcome == outcome for result in results)
        for outcome in ALL_OUTCOMES
    }
    ratio_ci = _bootstrap_seconds_per_correct_ci(results)
    ratio_ci_status = (
        "no_correct_answers"
        if ratio_ci is None
        else ("defined" if math.isfinite(ratio_ci[1]) else "unbounded_upper")
    )
    return {
        "cases": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "accuracy_ci_95": _wilson_interval(correct, total),
        "elapsed_seconds": elapsed,
        "elapsed_p50_seconds": _quantile(elapsed_list, 0.50),
        "elapsed_p95_seconds": _quantile(elapsed_list, 0.95),
        "execution_seconds": execution_seconds,
        "execution_p50_seconds": _quantile(exec_list, 0.50),
        "execution_p95_seconds": _quantile(exec_list, 0.95),
        "response_seconds": sum(response_list),
        "response_timing_samples": len(response_list),
        "response_p50_seconds": _quantile(response_list, 0.50),
        "response_p95_seconds": _quantile(response_list, 0.95),
        "seconds_per_correct": elapsed / correct if correct else None,
        "seconds_per_correct_ci_95": None
        if ratio_ci is None
        else tuple(bound if math.isfinite(bound) else None for bound in ratio_ci),
        "seconds_per_correct_ci_status": ratio_ci_status,
        "ratio_status": "defined" if correct else "no_correct_answers",
        "generated_tokens": generated_tokens,
        "injected_tokens": injected_tokens,
        "prompt_tokens": sum(prompt_list),
        "prompt_token_samples": len(prompt_list),
        "task_text_samples": sum(
            result.task_text_sha256 is not None for result in results
        ),
        "budget_samples": sum(result.budgets is not None for result in results),
        "total_tokens": generated_tokens + injected_tokens,
        "tool_calls": tool_calls,
        "outcomes": outcomes,
    }


def _paired_observation_coverage(
    paired: Mapping[str, Mapping[str, CaseResult]],
) -> dict[str, int]:
    complete_text = complete_budgets = 0
    for case_id, case in paired.items():
        observed = {
            result.task_text_sha256
            for result in case.values()
            if result.task_text_sha256 is not None
        }
        if len(observed) > 1:
            raise ValueError(f"Task text differs between paired results for {case_id}")
        observed_budgets = {
            result.budgets for result in case.values() if result.budgets is not None
        }
        if len(observed_budgets) > 1:
            raise ValueError(
                f"Resource budgets differ between paired results for {case_id}"
            )
        complete_text += all(
            result.task_text_sha256 is not None for result in case.values()
        )
        complete_budgets += all(result.budgets is not None for result in case.values())
    return {
        "task_text_complete_pairs": complete_text,
        "budget_complete_pairs": complete_budgets,
    }


def inspect_task_overlap(
    development: Sequence[Task], evaluation: Sequence[Task]
) -> dict[str, object]:
    """Report exact declared overlaps, not split acceptance or semantic leakage.

    Checkers are never invoked. Prompt comparisons ignore answer-format changes
    but otherwise preserve exact text; empty cohorts remain explicitly counted.
    """
    development_ids = {task.case_id for task in development}
    evaluation_ids = {task.case_id for task in evaluation}
    if len(development_ids) != len(development):
        raise ValueError("Development tasks must have unique case IDs")
    if len(evaluation_ids) != len(evaluation):
        raise ValueError("Evaluation tasks must have unique case IDs")
    development_prompts = {task.prompt for task in development}
    return {
        "development_tasks": len(development),
        "evaluation_tasks": len(evaluation),
        "overlapping_case_ids": sorted(development_ids & evaluation_ids),
        "overlapping_families": sorted(
            {task.family for task in development} & {task.family for task in evaluation}
        ),
        "evaluation_prompt_overlap_ids": sorted(
            task.case_id for task in evaluation if task.prompt in development_prompts
        ),
    }


def run_matched(
    tasks: Sequence[Task],
    runners: Mapping[Path, EvaluationHarness],
) -> tuple[CaseResult, ...]:
    """Preflight common budgets and task identities before any inference."""
    paths: tuple[Path, ...] = ("no_tool", "python_tool", "current_best")
    if set(runners) != set(paths):
        raise ValueError("Matched evaluation requires all three labelled paths")
    if len({task.case_id for task in tasks}) != len(tasks):
        raise ValueError("Matched tasks must have unique case IDs")
    budget = runners[paths[0]].budgets
    if any(runners[path].budgets != budget for path in paths):
        raise ValueError("Matched paths must use identical resource budgets")
    return tuple(runners[path].run(task, path) for task in tasks for path in paths)


def validate_references(
    tasks: Sequence[Task], results: Sequence[CaseResult]
) -> dict[str, int]:
    """Check saved final outcomes against built-in references, without rewriting.

    Pairing is validated by report consumers. Nonfinal failures retain their
    outcomes; absent task-text observations remain unmeasured, not imputed.
    """
    if not tasks or not results:
        raise ValueError("Reference validation requires tasks and case results")
    by_id = {task.case_id: task for task in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("Reference tasks must have unique case IDs")
    if {result.case_id for result in results} != by_id.keys():
        raise ValueError("Task/result IDs differ during reference validation")
    if any(
        type(task.checker)
        not in (IntegerChecker, RationalChecker, IntegerSequenceChecker)
        for task in tasks
    ):
        raise ValueError("Reference validation requires built-in exact checkers")
    digests = {case_id: _task_text_digest(task) for case_id, task in by_id.items()}
    answer_checks = text_checks = 0
    for result in results:
        task = by_id[result.case_id]
        if result.family != task.family:
            raise ValueError(f"Family differs from specification for {result.case_id}")
        if result.task_text_sha256 is not None:
            if result.task_text_sha256 != digests[result.case_id]:
                raise ValueError(
                    f"Task text differs from specification for {result.case_id}"
                )
            text_checks += 1
        if result.outcome in ("correct", "wrong_answer", "invalid_answer"):
            expected = _answer_outcome(task.checker(result.answer))
            if result.outcome != expected:
                raise ValueError(
                    f"Recorded outcome differs from reference for "
                    f"{result.case_id}/{result.path}: {result.outcome} != {expected}"
                )
            answer_checks += 1
    return {
        "tasks": len(tasks),
        "records": len(results),
        "answer_checks": answer_checks,
        "task_text_checks": text_checks,
        "nonfinal_records": len(results) - answer_checks,
    }


def _paired_comparison(
    paired: Mapping[str, Mapping[str, CaseResult]],
    target: str,
    baseline: str,
) -> dict[str, object]:
    n = len(paired)
    both = target_only = baseline_only = neither = 0
    time_deltas: list[float] = []
    for case_id in sorted(paired):
        case = paired[case_id]
        t_res = case[target]
        b_res = case[baseline]
        t_ok = t_res.outcome == "correct"
        b_ok = b_res.outcome == "correct"
        if t_ok and b_ok:
            both += 1
        elif t_ok:
            target_only += 1
        elif b_ok:
            baseline_only += 1
        else:
            neither += 1
        time_deltas.append(t_res.elapsed_seconds - b_res.elapsed_seconds)
    return {
        "pairs": n,
        "accuracy_delta": (target_only - baseline_only) / n,
        "both_correct": both,
        "target_only_correct": target_only,
        "baseline_only_correct": baseline_only,
        "neither_correct": neither,
        "elapsed_delta_mean_seconds": sum(time_deltas) / n,
        "elapsed_delta_p50_seconds": _quantile(time_deltas, 0.50),
    }


def _paired_deltas(
    paired: Mapping[str, Mapping[str, CaseResult]],
) -> dict[str, object]:
    return {
        "python_tool_vs_no_tool": _paired_comparison(paired, "python_tool", "no_tool"),
        "current_best_vs_python_tool": _paired_comparison(
            paired, "current_best", "python_tool"
        ),
    }


def matched_report(results: Sequence[CaseResult]) -> dict[str, object]:
    """Require complete paired coverage; retain per-family costs and failures.

    Observed task-text/budget mismatches are rejected; missing observations are
    counted as incomplete. Configured allowances do not certify enforcement,
    matched references, full model prompts, hardware or statistical qualification.
    """
    if not results:
        raise ValueError("A matched report requires case results")
    paths = _PATHS
    paired: dict[str, dict[str, CaseResult]] = {}
    for result in results:
        if result.path not in paths:
            raise ValueError(f"Unknown report path: {result.path}")
        case = paired.setdefault(result.case_id, {})
        if result.path in case:
            raise ValueError(f"Duplicate {result.path} result for {result.case_id}")
        case[result.path] = result
    for case_id, case in paired.items():
        if set(case) != set(paths):
            raise ValueError(f"Missing baseline/current-best result for {case_id}")
        if len({result.family for result in case.values()}) != 1:
            raise ValueError(f"Family identity differs between paths for {case_id}")
    coverage = _paired_observation_coverage(paired)
    families = sorted({result.family for result in results})
    return {
        "paired_cases": len(paired),
        **coverage,
        "by_path": {
            path: summarize([result for result in results if result.path == path])
            for path in paths
        },
        "by_family": {
            family: {
                path: summarize(
                    [
                        result
                        for result in results
                        if result.path == path and result.family == family
                    ]
                )
                for path in paths
            }
            for family in families
        },
        "by_outcome": {
            outcome: {
                path: sum(
                    result.outcome == outcome
                    for result in results
                    if result.path == path
                )
                for path in paths
            }
            for outcome in ALL_OUTCOMES
        },
        "paired_deltas": _paired_deltas(paired),
        "paired_deltas_by_family": {
            family: _paired_deltas(
                {
                    case_id: case
                    for case_id, case in paired.items()
                    if case[paths[0]].family == family
                }
            )
            for family in families
        },
    }
