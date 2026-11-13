"""Synthetic evaluation fixtures, not accepted task data or accuracy evidence."""

from __future__ import annotations

import random
from dataclasses import replace
from fractions import Fraction

import pytest

from neural_compiler.bridge import ExecutionResult, run_python
from neural_compiler.bridge.session import LiveSession
from neural_compiler.eval import (
    ALL_OUTCOMES,
    SYNTHETIC_FAMILIES,
    generate_family_tasks,
    generate_matched_suite,
)
from neural_compiler.eval.harness import (
    Budgets,
    CallRecord,
    CaseResult,
    CheckResult,
    EvaluationHarness,
    IntegerChecker,
    IntegerSequenceChecker,
    RationalChecker,
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


def test_invalid_format_is_not_scored_as_correct(tokenizer, hybrid_model, monkeypatch):
    diagnostic = "Correct value, but invalid final-answer format"
    task = Task(
        "invalid-format",
        "unit-checker",
        "Compute six times seven.",
        lambda answer: CheckResult(True, False, diagnostic),
    )
    runner = harness_for(
        tokenizer,
        hybrid_model,
        monkeypatch,
        "Final answer: 42 trailing text<|im_end|>",
    )
    result = runner.run(task, "no_tool")
    assert result.outcome == "invalid_answer"
    assert result.diagnostic == diagnostic
    assert result.generated_tokens > 0


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


def test_bootstrap_does_not_discard_zero_success_resamples():
    good = CaseResult("good", "unit", "no_tool", "correct", "", "", 1, 3, 0, ())
    bad = replace(good, case_id="bad", outcome="wrong_answer", elapsed_seconds=10)
    summary = summarize([good, bad])
    assert summary["seconds_per_correct"] == 11
    low, high = summary["seconds_per_correct_ci_95"]
    assert low == 1 and high is None
    assert summary["seconds_per_correct_ci_status"] == "unbounded_upper"


def test_execution_time_cannot_produce_nonfinite_report_metrics():
    with pytest.raises(ValueError, match="finite and nonnegative"):
        CallRecord("ok", float("nan"), 0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("generated_tokens", -1),
        ("generated_tokens", True),
        ("injected_tokens", 0.5),
        ("outcome", "unrecorded_failure"),
        ("path", "unknown_backend"),
    ],
)
def test_impossible_case_records_cannot_enter_aggregates(field, value):
    record = CaseResult("a", "unit", "python_tool", "correct", "", "", 1, 3, 0, ())
    with pytest.raises(ValueError, match=field):
        summarize([replace(record, **{field: value})])


def test_negative_response_count_cannot_enter_execution_metrics():
    with pytest.raises(ValueError, match="response_tokens"):
        call = CallRecord("ok", 0.1, -1)
        summarize(
            [
                CaseResult(
                    "a", "unit", "python_tool", "correct", "", "", 1, 3, 0, (call,)
                )
            ]
        )


@pytest.mark.parametrize(
    "expected,answer,correct,valid",
    [
        (Fraction(11, 6), "Work.\nFinal answer: 11/6", True, True),
        (Fraction(11, 6), "Final answer: +22 / +12", True, True),
        (Fraction(-5, 1), "Final answer: -5", True, True),
        (Fraction(11, 6), "Final answer: 10/6", False, True),
        (Fraction(11, 6), "Final answer: 11/0", False, False),
        (Fraction(11, 6), "Final answer: 11/6\nFinal answer: 10/6", False, True),
        (Fraction(11, 6), "Final answer: 11/6 trailing", False, False),
    ],
)
def test_rational_checker_validates_exact_fractions_on_last_line(
    expected, answer, correct, valid
):
    checked = RationalChecker(expected)(answer)
    assert (checked.correct, checked.valid) == (correct, valid)
    if not correct:
        assert checked.diagnostic


def test_rational_checker_handles_large_numbers_and_rejects_booleans():
    big = 10**4500 + 1
    digits = "1" + "0" * 4499 + "1"
    checked = RationalChecker(Fraction(big, 3))(f"Final answer: {digits}/3")
    assert checked.correct and checked.valid
    with pytest.raises(ValueError, match="boolean"):
        RationalChecker(True)


@pytest.mark.parametrize(
    "expected,answer,correct,valid",
    [
        ((3, -1, 0, 42), "Steps.\nFinal answer: 3, -1, 0, +042", True, True),
        ((3, -1, 0, 42), "Final answer: 3, -1, 42, 0", False, True),
        ((3, -1, 0, 42), "Final answer: 3, -1, 0", False, False),
        ((3, -1, 0, 42), "Final answer: 3, -1, 0, 42\n3, -1, 0, 42", False, False),
        ((3, -1, 0, 42), "Final answer: 3, a, 0, 42", False, False),
    ],
)
def test_integer_sequence_checker_validates_exact_tuples(
    expected, answer, correct, valid
):
    checked = IntegerSequenceChecker(expected)(answer)
    assert (checked.correct, checked.valid) == (correct, valid)
    if not correct:
        assert checked.diagnostic


def test_integer_sequence_checker_rejects_empty_or_boolean_references():
    with pytest.raises(ValueError, match="non-empty tuple"):
        IntegerSequenceChecker(())
    with pytest.raises(ValueError, match="non-empty tuple"):
        IntegerSequenceChecker((1, True))


def test_synthetic_family_generators_are_deterministic_and_self_consistent():
    assert len(SYNTHETIC_FAMILIES) == 5
    for family in SYNTHETIC_FAMILIES:
        batch_a = generate_family_tasks(family, count=4, seed=17)
        batch_b = generate_family_tasks(family, count=4, seed=17)
        batch_c = generate_family_tasks(family, count=4, seed=99)
        assert batch_a == batch_b
        assert [t.prompt for t in batch_a] != [t.prompt for t in batch_c]
        assert len({t.case_id for t in batch_a}) == 4
        assert len({t.prompt for t in batch_a}) == 4
        for task in batch_a:
            assert task.family == family
            checker = task.checker
            if isinstance(checker, IntegerChecker):
                assert checker(f"Final answer: {checker.expected}").correct
                assert not checker(f"Final answer: {checker.expected + 1}").correct
            elif isinstance(checker, RationalChecker):
                num, den = checker.expected.numerator, checker.expected.denominator
                assert checker(f"Final answer: {num}/{den}").correct
                assert not checker(f"Final answer: {num + 1}/{den}").correct
            elif isinstance(checker, IntegerSequenceChecker):
                formatted = ", ".join(str(x) for x in checker.expected)
                wrong = ", ".join(str(x + 1) for x in checker.expected)
                assert checker(f"Final answer: {formatted}").correct
                assert not checker(f"Final answer: {wrong}").correct


@pytest.mark.parametrize(
    "fn,args,kwargs",
    [
        (generate_family_tasks, ("unknown_family", 2), {}),
        (generate_family_tasks, ("binomial_coefficient", 0), {}),
        (generate_family_tasks, ("binomial_coefficient", True), {}),
        (generate_matched_suite, (), {"families": ()}),
        (generate_matched_suite, (), {"seed": True}),
        (generate_matched_suite, (), {"seed": False}),
        (generate_matched_suite, (), {"seed": "0"}),
        (
            generate_matched_suite,
            (),
            {"families": ("binomial_coefficient", "binomial_coefficient")},
        ),
    ],
)
def test_family_generator_validation(fn, args, kwargs):
    with pytest.raises(ValueError):
        fn(*args, **kwargs)


def test_matched_report_aggregates_tokens_calls_and_outcome_matrix():
    c1 = CaseResult(
        "c1",
        "binomial_coefficient",
        "python_tool",
        "correct",
        "",
        "Final answer: 10",
        1.5,
        generated_tokens=20,
        injected_tokens=8,
        calls=(CallRecord("ok", 0.25, 8),),
    )
    c2 = CaseResult(
        "c2",
        "harmonic_partial_sum",
        "python_tool",
        "protocol_error",
        "Malformed tool call",
        "",
        0.5,
        generated_tokens=12,
        injected_tokens=0,
        calls=(),
    )
    summary = summarize([c1, c2])
    assert summary["generated_tokens"] == 32
    assert summary["injected_tokens"] == 8
    assert summary["total_tokens"] == 40
    assert summary["tool_calls"] == 1
    assert summary["execution_seconds"] == pytest.approx(0.25)
    assert summary["outcomes"]["correct"] == 1
    assert summary["outcomes"]["protocol_error"] == 1
    assert set(summary["outcomes"]) == set(ALL_OUTCOMES)

    rows = []
    for path in ("no_tool", "python_tool", "current_best"):
        rows.extend([replace(c1, path=path), replace(c2, path=path)])
    report = matched_report(rows)
    assert report["paired_cases"] == 2
    assert report["by_outcome"]["correct"]["python_tool"] == 1
    assert report["by_outcome"]["protocol_error"]["no_tool"] == 1
    assert (
        report["by_family"]["harmonic_partial_sum"]["current_best"]["tool_calls"] == 0
    )


def test_end_to_end_multi_family_suite_on_tiny_cpu_model(
    tokenizer, hybrid_model, monkeypatch
):
    suite = generate_matched_suite(per_family=1, seed=5)
    seen_system_prompts: dict[str, str] = {}

    def format_expected(task: Task) -> str:
        checker = task.checker
        if isinstance(checker, IntegerChecker):
            return str(checker.expected)
        if isinstance(checker, RationalChecker):
            return f"{checker.expected.numerator}/{checker.expected.denominator}"
        if isinstance(checker, IntegerSequenceChecker):
            return ", ".join(str(x) for x in checker.expected)
        raise AssertionError("Unexpected checker")

    def make_runner(path_name: str) -> EvaluationHarness:
        def factory(ids):
            decoded_prompt = tokenizer.decode(ids, skip_special_tokens=False)
            matching = [t for t in suite if t.prompt in decoded_prompt]
            assert len(matching) == 1
            task = matching[0]
            seen_system_prompts[task.family] = decoded_prompt.split("<|im_end|>")[0]
            expected_str = format_expected(task)
            if path_name == "no_tool":
                reply = f"Final answer: {expected_str}<|im_end|>"
            else:
                reply = (
                    call_text(f"print({expected_str!r})")
                    + f"Final answer: {expected_str}<|im_end|>"
                )
            scripted = iter(tokenizer.encode(reply, add_special_tokens=False))
            session = LiveSession(hybrid_model, ids)
            monkeypatch.setattr(session, "_next_token", lambda: next(scripted))
            return session

        return EvaluationHarness(
            tokenizer,
            factory,
            lambda: None,
            executor=lambda code, timeout: ExecutionResult("ok", "42\n"),
        )

    rows = run_matched(
        suite,
        {
            "no_tool": make_runner("no_tool"),
            "python_tool": make_runner("python_tool"),
            "current_best": make_runner("current_best"),
        },
    )
    report = matched_report(rows)
    assert report["paired_cases"] == len(SYNTHETIC_FAMILIES)
    assert "<numerator>/<denominator>" in seen_system_prompts["harmonic_partial_sum"]
    assert "<n1>, <n2>, ..." in seen_system_prompts["recurrence_prefix"]
    assert "<integer>" in seen_system_prompts["binomial_coefficient"]
    assert report["by_path"]["no_tool"]["tool_calls"] == 0
    assert report["by_path"]["python_tool"]["tool_calls"] == len(SYNTHETIC_FAMILIES)
    assert report["by_path"]["python_tool"]["injected_tokens"] > 0
    assert all(
        report["by_path"][p]["correct"] == len(SYNTHETIC_FAMILIES)
        for p in ("no_tool", "python_tool", "current_best")
    )


def test_summarize_quantiles_wilson_ci_and_paired_deltas():
    rows: list[CaseResult] = []
    # 4 paired cases:
    # - c1: both correct (no_tool 4.0s, python_tool 1.0s)
    # - c2: python_tool only correct (no_tool 5.0s, python_tool 2.0s)
    # - c3: python_tool only correct (no_tool 6.0s, python_tool 3.0s)
    # - c4: neither correct (no_tool 10.0s, python_tool 4.0s)
    specs = [
        ("c1", "correct", 4.0, "correct", 1.0),
        ("c2", "wrong_answer", 5.0, "correct", 2.0),
        ("c3", "wrong_answer", 6.0, "correct", 3.0),
        ("c4", "timeout", 10.0, "wrong_answer", 4.0),
    ]
    for cid, nt_out, nt_t, py_out, py_t in specs:
        rows.append(
            CaseResult(cid, "arithmetic", "no_tool", nt_out, "", "", nt_t, 5, 0, ())
        )
        rows.append(
            CaseResult(
                cid,
                "arithmetic",
                "python_tool",
                py_out,
                "",
                "",
                py_t,
                5,
                2,
                (CallRecord("ok", py_t * 0.5, 2),),
            )
        )
        rows.append(
            CaseResult(
                cid,
                "arithmetic",
                "current_best",
                py_out,
                "",
                "",
                py_t * 0.5,
                5,
                2,
                (CallRecord("ok", py_t * 0.25, 2),),
            )
        )

    report = matched_report(rows)
    py_sum = report["by_path"]["python_tool"]
    assert py_sum["accuracy"] == pytest.approx(0.75)
    lo, hi = py_sum["accuracy_ci_95"]
    assert 0.0 < lo < 0.75 < hi < 1.0
    assert py_sum["elapsed_p50_seconds"] == pytest.approx(2.5)
    assert py_sum["elapsed_p95_seconds"] == pytest.approx(3.85)
    assert py_sum["execution_p50_seconds"] == pytest.approx(1.25)
    spc_lo, spc_hi = py_sum["seconds_per_correct_ci_95"]
    assert spc_lo <= py_sum["seconds_per_correct"] <= spc_hi

    nt_sum = report["by_path"]["no_tool"]
    assert nt_sum["execution_p50_seconds"] is None
    assert nt_sum["execution_p95_seconds"] is None

    delta = report["paired_deltas"]["python_tool_vs_no_tool"]
    assert delta["pairs"] == 4
    assert delta["both_correct"] == 1
    assert delta["target_only_correct"] == 2
    assert delta["baseline_only_correct"] == 0
    assert delta["neither_correct"] == 1
    assert delta["accuracy_delta"] == pytest.approx(0.5)
    assert delta["elapsed_delta_mean_seconds"] == pytest.approx(-3.75)
    assert delta["elapsed_delta_p50_seconds"] == pytest.approx(-3.0)
