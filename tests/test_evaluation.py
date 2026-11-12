"""Synthetic evaluation fixtures, not accepted task data or accuracy evidence."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest

from neural_compiler.bridge import ExecutionResult, run_python
from neural_compiler.bridge.session import LiveSession
from neural_compiler.eval.harness import (
    Budgets,
    CaseResult,
    EvaluationHarness,
    IntegerChecker,
    Task,
    matched_report,
    run_matched,
    summarize,
)


def call_text(code: str) -> str:
    return f"<tool_call>\n<function=run>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>"


def harness_for(tokenizer, hybrid_model, monkeypatch, text, **kwargs):
    scripted = iter(tokenizer.encode(text, add_special_tokens=False))
    monkeypatch.setattr(LiveSession, "_next_token", lambda _: next(scripted))
    return EvaluationHarness(
        tokenizer,
        lambda ids: LiveSession(hybrid_model, ids),
        lambda: None,
        **kwargs,
    )


@pytest.mark.parametrize(
    "answer,correct,valid",
    [
        ("Reasoning.\nFinal answer: +00042", True, True),
        ("Final answer: 41", False, True),
        ("Final answer: 42 trailing text", False, False),
        ("Final answer: 42\nFinal answer: 41", False, True),
        ("Final answer: 42\n42", False, False),
        ("Final answer: 4,2", False, False),
    ],
)
def test_checker_rejects_partial_or_lucky_earlier_answers(answer, correct, valid):
    checked = IntegerChecker(42)(answer)
    assert (checked.correct, checked.valid) == (correct, valid)
    if not correct:
        assert checked.diagnostic


@pytest.mark.parametrize("sign", [1, -1])
def test_checker_handles_large_integer_references(sign):
    digits = "1" + "0" * 4499
    prefix = "-" if sign < 0 else "+000"
    checked = IntegerChecker(sign * 10**4499)(f"Final answer: {prefix}{digits}")
    assert checked.correct and checked.valid


def test_checker_retains_valid_wrong_large_integer_answer():
    checked = IntegerChecker(10**4499)("Final answer: " + "2" + "0" * 4499)
    assert checked.valid and not checked.correct
    assert checked.diagnostic


def test_checker_handles_large_mixed_decimal_blocks():
    expected = ((10**4500 - 1) // 9 * 7) * 10 + 3
    checked = IntegerChecker(expected)("Final answer: " + "7" * 4500 + "3")
    assert checked.correct and checked.valid


def test_matched_paths_share_tasks_and_budgets(tokenizer, hybrid_model, monkeypatch):
    rng = random.Random(7)
    for index in range(3):
        a, b = rng.randrange(2, 9), rng.randrange(2, 9)
        task = Task(
            f"synthetic-{index}",
            "unit-multiply",
            f"Compute {a} * {b}.",
            IntegerChecker(a * b),
        )
        for path in ("no_tool", "python_tool", "current_best"):
            runner = harness_for(
                tokenizer, hybrid_model, monkeypatch, f"Final answer: {a * b}<|im_end|>"
            )
            result = runner.run(task, path)
            assert result.outcome == "correct"
            assert result.generated_tokens > 0 and result.calls == ()
            assert result.family == task.family and result.elapsed_seconds > 0


def test_python_path_retains_failed_repairs_and_does_not_leak_reference(
    tokenizer, hybrid_model, monkeypatch
):
    probe = run_python("print(0)")
    if probe.status == "sandbox_error":
        pytest.skip(f"Namespaces unavailable: {probe.stderr}")
    assert probe.status == "ok", probe
    task = Task("repair", "unit-repair", "Compute six times seven.", IntegerChecker(42))
    text = (
        call_text("raise ValueError('repair')")
        + call_text("print(6 * 7)")
        + "Final answer: 42<|im_end|>"
    )
    prompts = []
    scripted = iter(tokenizer.encode(text, add_special_tokens=False))
    monkeypatch.setattr(LiveSession, "_next_token", lambda _: next(scripted))

    def factory(ids):
        prompts.append(tokenizer.decode(ids, skip_special_tokens=False))
        return LiveSession(hybrid_model, ids)

    result = EvaluationHarness(tokenizer, factory, lambda: None).run(
        task, "python_tool"
    )
    assert result.outcome == "correct"
    assert [call.status for call in result.calls] == ["runtime_error", "ok"]
    assert result.injected_tokens == sum(call.response_tokens for call in result.calls)
    assert result.elapsed_seconds >= sum(
        call.execution_seconds for call in result.calls
    )
    assert "42" not in prompts[0]


@pytest.mark.parametrize("path", ["no_tool", "python_tool"])
def test_bad_calls_do_not_execute(tokenizer, hybrid_model, monkeypatch, path):
    text = call_text("print(42)") if path == "no_tool" else "<tool_call>bad</tool_call>"
    runner = harness_for(tokenizer, hybrid_model, monkeypatch, text)
    monkeypatch.setattr(
        runner, "executor", lambda *args, **kwargs: pytest.fail("Must not execute")
    )
    result = runner.run(Task("bad", "unit", "Compute.", IntegerChecker(42)), path)
    assert result.outcome == "protocol_error" and result.calls == ()
    assert result.diagnostic


def test_total_token_budget_does_not_reset_after_a_tool_call(
    tokenizer, hybrid_model, monkeypatch
):
    text = call_text("print(42)") + "Final answer: 42<|im_end|>"
    limit = len(tokenizer.encode(call_text("print(42)"), add_special_tokens=False)) + 2
    runner = harness_for(
        tokenizer,
        hybrid_model,
        monkeypatch,
        text,
        budgets=replace(Budgets(), generated_tokens=limit),
        executor=lambda code, timeout: ExecutionResult("ok", "42\n"),
    )
    result = runner.run(
        Task("bounded", "unit", "Compute.", IntegerChecker(42)), "python_tool"
    )
    assert result.outcome == "budget_exhausted"
    assert result.generated_tokens == limit and len(result.calls) == 1


def test_prompt_limit_prevents_model_prefill(tokenizer):
    runner = EvaluationHarness(
        tokenizer,
        lambda ids: pytest.fail("Model must not run"),
        lambda: None,
        budgets=replace(Budgets(), prompt_tokens=1),
    )
    result = runner.run(
        Task("prompt", "unit", "Compute.", IntegerChecker(42)), "no_tool"
    )
    assert result.outcome == "budget_exhausted" and result.generated_tokens == 0


def test_call_budget_prevents_an_extra_execution(tokenizer, hybrid_model, monkeypatch):
    text = call_text("print(42)") * 2
    runner = harness_for(
        tokenizer,
        hybrid_model,
        monkeypatch,
        text,
        budgets=replace(Budgets(), calls=1),
        executor=lambda code, timeout: ExecutionResult("ok", "42"),
    )
    result = runner.run(
        Task("cap", "unit", "Compute.", IntegerChecker(42)), "python_tool"
    )
    assert result.outcome == "budget_exhausted" and len(result.calls) == 1
    assert "Serial tool-call" in result.diagnostic


def test_context_limit_prevents_oversized_result_prefill(
    tokenizer, hybrid_model, monkeypatch
):
    runner = harness_for(
        tokenizer,
        hybrid_model,
        monkeypatch,
        call_text("print(42)"),
        budgets=replace(Budgets(), context_tokens=200),
        executor=lambda code, timeout: ExecutionResult("ok", "x" * 100),
    )
    result = runner.run(
        Task("context", "unit", "Compute.", IntegerChecker(42)), "python_tool"
    )
    assert result.outcome == "budget_exhausted" and result.injected_tokens == 0
    assert "response would exceed" in result.diagnostic
    assert len(result.calls) == 1


def test_isolation_failure_is_not_a_model_repair_or_success(
    tokenizer, hybrid_model, monkeypatch
):
    runner = harness_for(
        tokenizer,
        hybrid_model,
        monkeypatch,
        call_text("print(42)"),
        executor=lambda code, timeout: ExecutionResult(
            "sandbox_error", stderr="Namespace unavailable"
        ),
    )
    result = runner.run(
        Task("isolation", "unit", "Compute.", IntegerChecker(42)), "python_tool"
    )
    assert result.outcome == "sandbox_error" and result.injected_tokens == 0
    assert result.diagnostic == "Namespace unavailable" and len(result.calls) == 1


def test_complete_answer_at_generation_limit_is_not_accepted(
    tokenizer, hybrid_model, monkeypatch
):
    text = "Final answer: 42"
    count = len(tokenizer.encode(text, add_special_tokens=False))
    runner = harness_for(
        tokenizer,
        hybrid_model,
        monkeypatch,
        text,
        budgets=replace(Budgets(), generated_tokens=count),
    )
    result = runner.run(Task("cut", "unit", "Compute.", IntegerChecker(42)), "no_tool")
    assert result.outcome == "budget_exhausted"


def test_wall_budget_is_counted_across_execution(tokenizer, hybrid_model, monkeypatch):
    from neural_compiler.eval import harness

    now = [0.0]
    monkeypatch.setattr(harness.time, "perf_counter", lambda: now[0])

    def execute(code, timeout):
        assert timeout == 1
        now[0] = 2
        return ExecutionResult("timeout", stderr="Deadline exceeded")

    runner = harness_for(
        tokenizer,
        hybrid_model,
        monkeypatch,
        call_text("print(42)"),
        budgets=replace(Budgets(), wall_seconds=1),
        executor=execute,
    )
    result = runner.run(
        Task("timeout", "unit", "Compute.", IntegerChecker(42)), "python_tool"
    )
    assert result.outcome == "timeout" and result.elapsed_seconds == 2
    assert len(result.calls) == 1 and result.calls[0].status == "timeout"


@pytest.mark.parametrize("stage", ["checker", "synchronize"])
def test_wall_budget_includes_terminal_work(
    tokenizer, hybrid_model, monkeypatch, stage
):
    from neural_compiler.eval import harness

    now = [0.0]
    sync_calls = 0
    monkeypatch.setattr(harness.time, "perf_counter", lambda: now[0])

    def synchronize():
        nonlocal sync_calls
        sync_calls += 1
        if stage == "synchronize" and sync_calls == 3:
            now[0] = 2

    def check(answer):
        checked = IntegerChecker(42)(answer)
        if stage == "checker":
            now[0] = 2
        return checked

    runner = harness_for(
        tokenizer,
        hybrid_model,
        monkeypatch,
        "Final answer: 42<|im_end|>",
        budgets=replace(Budgets(), wall_seconds=1),
    )
    runner.synchronize = synchronize
    result = runner.run(Task("terminal", "unit", "Compute.", check), "no_tool")
    assert result.outcome == "timeout"
    assert result.elapsed_seconds == 2 and result.answer == "Final answer: 42"
    assert result.diagnostic


def test_terminal_wall_check_preserves_explicit_protocol_failure(
    tokenizer, hybrid_model, monkeypatch
):
    from neural_compiler.eval import harness

    now = [0.0]
    sync_calls = 0
    monkeypatch.setattr(harness.time, "perf_counter", lambda: now[0])

    def synchronize():
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 2:
            now[0] = 2

    runner = harness_for(
        tokenizer,
        hybrid_model,
        monkeypatch,
        "<tool_call>bad</tool_call>",
        budgets=replace(Budgets(), wall_seconds=1),
    )
    runner.synchronize = synchronize
    result = runner.run(
        Task("protocol", "unit", "Compute.", IntegerChecker(42)), "python_tool"
    )
    assert result.outcome == "protocol_error" and result.diagnostic
    assert result.elapsed_seconds == 2


def test_zero_success_ratio_is_explicit_and_failed_time_counts():
    good = CaseResult("a", "unit", "no_tool", "correct", "", "42", 2, 3, 0, ())
    bad = replace(good, case_id="b", outcome="wrong_answer", elapsed_seconds=6)
    assert summarize([good, bad])["seconds_per_correct"] == 8
    assert summarize([bad])["seconds_per_correct"] is None
    assert summarize([bad])["ratio_status"] == "no_correct_answers"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"calls": 0},
        {"generated_tokens": True},
        {"wall_seconds": float("nan")},
    ],
)
def test_invalid_budgets_are_explicit(kwargs):
    with pytest.raises(ValueError):
        Budgets(**kwargs)


def test_matched_report_preserves_family_failures_and_rejects_missing_pairs():
    rows = []
    for path in ("no_tool", "python_tool", "current_best"):
        good = CaseResult("a", "arithmetic", path, "correct", "", "42", 2, 3, 0, ())
        bad = replace(
            good,
            case_id="b",
            family="sequences",
            outcome="wrong_answer",
            elapsed_seconds=6,
        )
        rows.extend([good, bad])
    report = matched_report(rows)
    assert report["by_path"]["python_tool"]["seconds_per_correct"] == 8
    assert (
        report["by_family"]["sequences"]["python_tool"]["seconds_per_correct"] is None
    )
    with pytest.raises(ValueError, match="Missing"):
        matched_report(rows[:-1])
    with pytest.raises(ValueError, match="Duplicate"):
        matched_report([*rows, rows[0]])
    with pytest.raises(ValueError, match="Family identity"):
        matched_report([*rows[:-1], replace(rows[-1], family="arithmetic")])
    with pytest.raises(ValueError, match="one evaluation path"):
        summarize(rows)


def test_matched_preflight_rejects_budget_mismatch_before_inference(tokenizer):
    def factory(ids):
        pytest.fail("Preflight must precede model execution")

    base = EvaluationHarness(tokenizer, factory, lambda: None)
    different = EvaluationHarness(
        tokenizer,
        factory,
        lambda: None,
        budgets=replace(Budgets(), calls=1),
    )
    tasks = [Task("a", "unit", "Compute.", IntegerChecker(42))]
    with pytest.raises(ValueError, match="identical resource"):
        run_matched(
            tasks, {"no_tool": base, "python_tool": base, "current_best": different}
        )
    with pytest.raises(ValueError, match="all three"):
        run_matched(tasks, {"python_tool": base})
    with pytest.raises(ValueError, match="unique case"):
        run_matched(
            tasks * 2, {"no_tool": base, "python_tool": base, "current_best": base}
        )


def test_matched_runner_produces_all_three_report_paths(
    tokenizer, hybrid_model, monkeypatch
):
    task = Task("synthetic", "unit", "Compute six times seven.", IntegerChecker(42))

    def factory(ids):
        scripted = iter(
            tokenizer.encode("Final answer: 42<|im_end|>", add_special_tokens=False)
        )
        session = LiveSession(hybrid_model, ids)
        monkeypatch.setattr(session, "_next_token", lambda: next(scripted))
        return session

    runner = EvaluationHarness(tokenizer, factory, lambda: None)
    rows = run_matched(
        [task], {"no_tool": runner, "python_tool": runner, "current_best": runner}
    )
    assert len(rows) == 3
    report = matched_report(rows)
    assert set(report["by_path"]) == {"no_tool", "python_tool", "current_best"}
    assert all(summary["correct"] == 1 for summary in report["by_path"].values())


@pytest.mark.parametrize("elapsed", [-1, float("nan"), float("inf")])
def test_invalid_time_cannot_produce_a_success_shaped_cost(elapsed):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        CaseResult("a", "unit", "no_tool", "correct", "", "42", elapsed, 3, 0, ())
