"""Matched evaluation contracts; this module does not load models or task data."""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol

from transformers import PreTrainedTokenizerBase

from neural_compiler.bridge import ExecutionResult, run_python
from neural_compiler.bridge.protocol import (
    RUN_TOOL,
    ProtocolError,
    QwenToolProtocol,
    ToolCall,
)
from neural_compiler.bridge.session import DecodeResult

Path = Literal["no_tool", "python_tool", "current_best"]
Outcome = Literal[
    "correct",
    "wrong_answer",
    "invalid_answer",
    "protocol_error",
    "budget_exhausted",
    "timeout",
    "sandbox_error",
]
_SYSTEM = "Solve the task. End your response with Final answer: <integer>."
_FINAL = re.compile(r"Final answer:\s*([+-]?[0-9]+)\s*")


@dataclass(frozen=True)
class CheckResult:
    correct: bool
    valid: bool
    diagnostic: str


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
        value = match.group(1)
        magnitude = value.lstrip("+-").lstrip("0") or "0"
        canonical = (
            "-" if value.startswith("-") and magnitude != "0" else ""
        ) + magnitude
        correct = canonical == str(self.expected)
        return CheckResult(
            correct, True, "" if correct else "Final integer differs from the reference"
        )


@dataclass(frozen=True)
class Task:
    case_id: str
    family: str
    prompt: str
    checker: Checker

    def __post_init__(self) -> None:
        if (
            not self.case_id.strip()
            or not self.family.strip()
            or not self.prompt.strip()
        ):
            raise ValueError("Task identity, family and prompt must not be empty")


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
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


class DecodeSession(Protocol):
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


@dataclass(frozen=True)
class CaseResult:
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

    def __post_init__(self) -> None:
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds < 0:
            raise ValueError("Case elapsed time must be finite and nonnegative")


class _RunFailure(Exception):
    def __init__(self, outcome: Outcome, diagnostic: str):
        super().__init__(diagnostic)
        self.outcome = outcome


class EvaluationHarness:
    """Same tasks, prompts, checker and budgets across caller-labelled paths.

    The caller supplies an already-loaded session factory and synchronization
    function (CUDA synchronize for GPU timing). Model deadlines are cooperative;
    only the supervised executor supplies a hard subprocess deadline.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        session_factory: SessionFactory,
        synchronize: Callable[[], None],
        budgets: Budgets = Budgets(),
        executor: Executor = run_python,
    ):
        self.tokenizer = tokenizer
        self.session_factory = session_factory
        self.synchronize = synchronize
        self.budgets = budgets
        self.executor = executor

    def run(self, task: Task, path: Path) -> CaseResult:
        if path not in ("no_tool", "python_tool", "current_best"):
            raise ValueError(f"Unknown evaluation path: {path}")
        self.synchronize()
        started = time.perf_counter()
        calls: list[CallRecord] = []
        generated = injected = 0
        prompt_count = 0
        answer = ""
        outcome: Outcome = "invalid_answer"
        diagnostic = ""
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
            generated += 1
            remaining_time()
            if prompt_count + generated + injected > self.budgets.context_tokens:
                raise _RunFailure("budget_exhausted", "Context-token budget exhausted")
            call = protocol.feed(token)
            if path == "no_tool" and call is not None:
                raise ProtocolError("Tool calls are disabled for the no-tool baseline")
            return call is not None

        try:
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": task.prompt},
                ],
                tools=None if path == "no_tool" else [RUN_TOOL],
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            if not isinstance(prompt, list) or not all(
                isinstance(token, int) for token in prompt
            ):
                raise ProtocolError("Chat template must return one token sequence")
            prompt_count = len(prompt)
            if prompt_count > min(
                self.budgets.prompt_tokens, self.budgets.context_tokens
            ):
                raise _RunFailure(
                    "budget_exhausted",
                    "Prompt-token budget exhausted before model prefill",
                )
            remaining_time()
            session = self.session_factory(prompt)
            while generated < self.budgets.generated_tokens:
                remaining_time()
                call: ToolCall | None = None
                decoded = session.decode_until(
                    min(
                        self.budgets.turn_tokens,
                        self.budgets.generated_tokens - generated,
                    ),
                    on_token,
                )
                self.synchronize()
                remaining_time()
                protocol.finish()
                if call is None:
                    answer = self.tokenizer.decode(
                        decoded.token_ids, skip_special_tokens=True
                    )
                    if decoded.reason == "length":
                        raise _RunFailure(
                            "budget_exhausted",
                            "Answer turn ended at the generation-token limit",
                        )
                    checked = task.checker(answer)
                    outcome = (
                        "correct"
                        if checked.correct
                        else ("wrong_answer" if checked.valid else "invalid_answer")
                    )
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
                response = protocol.response_ids(call, result)
                if (
                    prompt_count + generated + injected + len(response)
                    > self.budgets.context_tokens
                ):
                    raise _RunFailure(
                        "budget_exhausted",
                        "Tool response would exceed the context-token budget",
                    )
                session.inject(response)
                injected += len(response)
                calls[-1] = CallRecord(result.status, execution_time, len(response))
            else:
                raise _RunFailure(
                    "budget_exhausted", "Total generation-token budget exhausted"
                )
        except ProtocolError as exc:
            outcome, diagnostic = "protocol_error", str(exc)
        except _RunFailure as exc:
            outcome, diagnostic = exc.outcome, str(exc)
        self.synchronize()
        return CaseResult(
            task.case_id,
            task.family,
            path,
            outcome,
            diagnostic,
            answer,
            time.perf_counter() - started,
            generated,
            injected,
            tuple(calls),
        )


def summarize(results: Sequence[CaseResult]) -> dict[str, int | float | None | str]:
    """Failure-inclusive aggregate; zero successes have no finite cost ratio."""
    if len({result.path for result in results}) > 1:
        raise ValueError("Summarize one evaluation path at a time")
    if len({result.case_id for result in results}) != len(results):
        raise ValueError("Duplicate case IDs cannot be counted as independent attempts")
    correct = sum(result.outcome == "correct" for result in results)
    elapsed = sum(result.elapsed_seconds for result in results)
    return {
        "cases": len(results),
        "correct": correct,
        "elapsed_seconds": elapsed,
        "seconds_per_correct": elapsed / correct if correct else None,
        "ratio_status": "defined" if correct else "no_correct_answers",
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


def matched_report(results: Sequence[CaseResult]) -> dict[str, dict]:
    """Require complete paired coverage; retain per-family costs and failures.

    Coverage alone does not establish matched hardware, prompts, accepted splits,
    intrinsic availability, independence or statistical qualification.
    """
    if not results:
        raise ValueError("A matched report requires case results")
    paths = ("no_tool", "python_tool", "current_best")
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
    families = sorted({result.family for result in results})
    return {
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
    }
