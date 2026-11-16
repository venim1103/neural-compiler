"""Offline P0 serving comparison checks on tiny CPU hybrid Qwen fixtures."""

from __future__ import annotations

import shutil
from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from neural_compiler.bridge import ExecutionResult, run_c, run_python
from neural_compiler.bridge.protocol import RUN_TOOL
from neural_compiler.eval import (
    ALL_SERVING_MODES,
    FullReprefillSession,
    IntegerChecker,
    LivePauseResumeSession,
    PrefixCachedSession,
    ServingCaseResult,
    Task,
    run_serving_comparison,
    summarize_serving,
)
from neural_compiler.eval.harness import CaseResult
from neural_compiler.bridge.session import CheckpointStore, LiveSession


def _call_text(code: str) -> str:
    return (
        "<tool_call>\n<function=run>\n<parameter=code>\n"
        f"{code}\n</parameter>\n</function>\n</tool_call>"
    )


@pytest.mark.parametrize("has_tasks", [False, True])
@pytest.mark.parametrize(
    "options,error",
    [
        ({"path": "unsupported"}, "Unknown evaluation path"),
        ({"max_store_bytes": 0}, "max_total_bytes must be a positive integer"),
        ({"max_store_entries": 0}, "max_entries must be a positive integer"),
    ],
)
def test_serving_configuration_precedes_factory_and_prefill(
    tokenizer, hybrid_model, monkeypatch, has_tasks, options, error
):
    tasks = (
        [Task("preflight", "unit", "Compute.", IntegerChecker(42))] if has_tasks else []
    )
    monkeypatch.setattr(
        hybrid_model,
        "forward",
        lambda *args, **kwargs: pytest.fail("Invalid configuration ran model"),
    )
    with pytest.raises(ValueError, match=error):
        run_serving_comparison(
            tasks,
            tokenizer,
            hybrid_model,
            lambda: None,
            on_token_factory=lambda *args: pytest.fail(
                "Invalid configuration called factory"
            ),
            **options,
        )


@pytest.mark.parametrize("sampler", [None, 42, "not-callable"])
def test_invalid_sampler_factory_result_never_prefills(
    tokenizer, hybrid_model, monkeypatch, sampler
):
    monkeypatch.setattr(
        hybrid_model,
        "forward",
        lambda *args, **kwargs: pytest.fail("Invalid sampler prefills model"),
    )
    with pytest.raises(ValueError, match="on_token_factory must return a callable"):
        run_serving_comparison(
            [Task("invalid-sampler", "unit", "Compute.", IntegerChecker(42))],
            tokenizer,
            hybrid_model,
            lambda: None,
            on_token_factory=lambda *args: sampler,
        )


@pytest.mark.parametrize("has_tasks", [False, True])
@pytest.mark.parametrize("factory", [False, 42])
def test_noncallable_sampler_factory_is_rejected_before_iteration(
    tokenizer, hybrid_model, monkeypatch, has_tasks, factory
):
    monkeypatch.setattr(
        hybrid_model,
        "forward",
        lambda *args, **kwargs: pytest.fail("Invalid factory ran model"),
    )
    tasks = (
        [Task("factory", "unit", "Compute.", IntegerChecker(42))] if has_tasks else []
    )
    with pytest.raises(ValueError, match="on_token_factory must be callable"):
        run_serving_comparison(
            tasks,
            tokenizer,
            hybrid_model,
            lambda: None,
            on_token_factory=factory,
        )


@pytest.mark.parametrize("mode", ["prefix", "full"])
def test_invalid_decode_budget_does_not_reconstruct_session(
    hybrid_model, monkeypatch, mode
):
    if mode == "prefix":
        store = CheckpointStore(max_total_bytes=1024 * 1024)
        session = PrefixCachedSession(
            hybrid_model, [1, 2, 3], store, on_token_override=lambda _: 4
        )
    else:
        session = FullReprefillSession(
            hybrid_model, [1, 2, 3], on_token_override=lambda _: 4
        )
    session.decode_until(1, lambda _: True)
    assert session._session is None
    tokens = session.tokens
    prefill_tokens = session.prefill_tokens
    if mode == "prefix":
        monkeypatch.setattr(
            store,
            "resume_prefix",
            lambda *a, **kw: pytest.fail("Invalid budget restored a checkpoint"),
        )
    else:
        monkeypatch.setattr(
            LiveSession,
            "_forward",
            lambda *a, **kw: pytest.fail("Invalid budget re-prefilled the model"),
        )
    with pytest.raises(ValueError, match="positive integer"):
        session.decode_until(0)
    assert session._session is None
    assert session.tokens == tokens
    assert session.prefill_tokens == prefill_tokens


@pytest.mark.parametrize("warm,budget", [(False, 0), (True, float("nan"))])
def test_invalid_snapshot_override_is_rejected_before_prefix_lookup(
    hybrid_model, monkeypatch, warm, budget
):
    store = CheckpointStore(max_total_bytes=1024 * 1024)
    if warm:
        store.put(LiveSession(hybrid_model, [1, 2, 3]).checkpoint())
    stored_bytes = store.total_bytes
    monkeypatch.setattr(
        store,
        "resume_prefix",
        lambda *a, **kw: pytest.fail("Invalid snapshot limit resumed a prefix"),
    )
    with pytest.raises(ValueError, match="max_cache_bytes must be a positive integer"):
        PrefixCachedSession(hybrid_model, [1, 2, 3], store, max_checkpoint_bytes=budget)
    assert store.total_bytes == stored_bytes
    assert store.hits == store.misses == 0


def test_captured_logits_are_independent_of_mutable_sampler_sessions(
    tokenizer, hybrid_model
):
    captured = {}

    def sampler_factory(task, mode):
        scripted = iter(
            tokenizer.encode("Final answer: 42<|im_end|>", add_special_tokens=False)
        )

        def sample(session):
            captured[mode] = session
            return next(scripted)

        return sample

    rows = run_serving_comparison(
        [Task("logit-ownership", "unit", "Compute.", IntegerChecker(42))],
        tokenizer,
        hybrid_model,
        lambda: None,
        executor=lambda *args, **kwargs: pytest.fail("Must not execute"),
        on_token_factory=sampler_factory,
    )
    for row in rows:
        assert row.case.outcome == "correct" and row.final_logits is not None
        expected = row.final_logits.clone()
        source = captured[row.mode]
        source.logits.add_(1)
        torch.testing.assert_close(row.final_logits, expected, atol=0, rtol=0)
        source_after = source.logits.clone()
        row.final_logits.zero_()
        torch.testing.assert_close(source.logits, source_after, atol=0, rtol=0)


@pytest.mark.parametrize("boundary", ["eos", "length", "sandbox_error"])
def test_serving_logit_prefix_matches_exact_forwarded_boundary(
    tokenizer, hybrid_model, boundary
):
    from neural_compiler.eval.harness import Budgets

    text = "Final answer: 42"
    budgets = Budgets()
    if boundary == "length":
        budgets = replace(
            budgets, turn_tokens=len(tokenizer.encode(text, add_special_tokens=False))
        )
    else:
        text += "<|im_end|>"
    if boundary == "sandbox_error":
        text = _call_text("print(6 * 7)") + text

    def sampler_factory(task, mode):
        stream = iter(tokenizer.encode(text, add_special_tokens=False))
        return lambda _: next(stream)

    def execute(code, timeout=20):
        if boundary != "sandbox_error":
            pytest.fail("Must not execute")
        return ExecutionResult("sandbox_error", "", "Fixture declined execution")

    rows = run_serving_comparison(
        [Task("logit-prefix", "unit", "Compute.", IntegerChecker(42))],
        tokenizer,
        hybrid_model,
        lambda: None,
        budgets=budgets,
        executor=execute,
        on_token_factory=sampler_factory,
    )
    outcomes = {
        "eos": "correct",
        "length": "budget_exhausted",
        "sandbox_error": "sandbox_error",
    }
    for row in rows:
        assert row.case.outcome == outcomes[boundary]
        assert row.final_logits is not None
        assert row.logit_prefix_tokens == len(row.final_tokens) - (boundary != "length")
        prefix = row.final_tokens[: row.logit_prefix_tokens]
        with torch.no_grad():
            expected = hybrid_model(
                input_ids=torch.tensor([prefix]), use_cache=True, logits_to_keep=1
            )
        torch.testing.assert_close(
            row.final_logits, expected.logits[:, -1, :], atol=1e-6, rtol=1e-5
        )


@pytest.mark.parametrize("count", [-1, True, 4])
def test_invalid_logit_prefix_counts_are_explicit(count):
    legacy = ServingCaseResult(
        "live_pause_resume",
        CaseResult(
            "prefix", "unit", "python_tool", "wrong_answer", "", "", 1, 0, 0, ()
        ),
        (1, 2, 3),
        torch.zeros(1, 8),
        3,
        0,
        0,
        0,
        0,
    )
    assert legacy.logit_prefix_tokens is None
    with pytest.raises(ValueError, match="logit_prefix_tokens"):
        replace(legacy, logit_prefix_tokens=count)


def test_serving_modes_match_logits_and_reduce_prefill_tokens(tokenizer, hybrid_model):
    task = Task(
        "two-step-calc",
        "unit-serving",
        "Compute (6 * 7) + 8.",
        IntegerChecker(50),
    )
    script_text = (
        _call_text("print(6 * 7)")
        + _call_text("print(42 + 8)")
        + "Final answer: 50<|im_end|>"
    )

    def make_sampler(_task, _mode):
        stream = iter(tokenizer.encode(script_text, add_special_tokens=False))
        return lambda _session: next(stream)

    def fake_executor(code: str, timeout: float = 20) -> ExecutionResult:
        if "6 * 7" in code:
            return ExecutionResult("ok", "42\n")
        return ExecutionResult("ok", "50\n")

    results = run_serving_comparison(
        [task],
        tokenizer,
        hybrid_model,
        lambda: None,
        executor=fake_executor,
        on_token_factory=make_sampler,
    )
    by_mode = {row.mode: row for row in results}
    assert set(by_mode) == set(ALL_SERVING_MODES)

    live = by_mode["live_pause_resume"]
    cached = by_mode["prefix_cached_resubmission"]
    reprefill = by_mode["full_reprefill_resubmission"]

    # All three modes produce identical task outcomes, token streams, and logits.
    for row in (live, cached, reprefill):
        assert row.case.outcome == "correct"
        assert row.case.generated_tokens == live.case.generated_tokens
        assert row.case.injected_tokens == live.case.injected_tokens
        assert row.final_tokens == live.final_tokens

    assert live.final_logits is not None
    assert cached.final_logits is not None
    assert reprefill.final_logits is not None
    # Exact checkpoint resume matches live incremental prefill bit-for-bit.
    torch.testing.assert_close(cached.final_logits, live.final_logits, atol=0, rtol=0)
    # Full re-prefill matches within standard floating-point reduction tolerance.
    torch.testing.assert_close(
        reprefill.final_logits, live.final_logits, atol=1e-6, rtol=1e-5
    )

    # Prefix-cached resubmission matches live incremental prefill token count and
    # avoids re-prefilling the growing conversation history across tool turns.
    assert cached.prefill_tokens == live.prefill_tokens
    assert reprefill.prefill_tokens > cached.prefill_tokens
    assert cached.cache_hits == 2
    assert cached.cache_misses == 1
    assert cached.checkpoint_bytes_peak > 0

    summary = summarize_serving(results)
    assert set(summary["by_mode"]) == set(ALL_SERVING_MODES)
    assert summary["by_mode"]["prefix_cached_resubmission"]["cache_hits"] == 2
    assert (
        summary["by_mode"]["full_reprefill_resubmission"]["prefill_tokens"]
        > summary["by_mode"]["prefix_cached_resubmission"]["prefill_tokens"]
    )


def test_declared_c_executor_preserves_matched_framing_and_logits(
    tokenizer, hybrid_model
):
    if shutil.which("gcc") is None:
        pytest.skip("gcc unavailable; no substitute compiler")
    probe = run_python("print(0)")
    if probe.status == "sandbox_error":
        pytest.skip(f"Namespace support unavailable: {probe.stderr}")
    assert probe.status == "ok", probe
    declaration = deepcopy(RUN_TOOL)
    description = "Compile and run C99 code, returning structured stdout or errors."
    tokenizer.chat_template = (
        "{% if tools %}{{ tools | tojson }}\n{% endif %}" + tokenizer.chat_template
    )
    program = (
        '#include <stdio.h>\nint main(void) { printf("%d\\n", 6 * 7); return 0; }\n'
    )
    script = _call_text(program) + "Final answer: 42<|im_end|>"
    prompts = {}
    prompt_counts = {}
    native_results: list[ExecutionResult] = []
    replayed_sources = []

    def make_sampler(_task, mode):
        stream = iter(tokenizer.encode(script, add_special_tokens=False))

        def sample(session):
            if mode not in prompts:
                prompt_counts[mode] = len(session.tokens)
                prompts[mode] = tokenizer.decode(
                    session.tokens, skip_special_tokens=False
                )
            return next(stream)

        return sample

    def execute(code: str, timeout: float = 20) -> ExecutionResult:
        replayed_sources.append(code)
        assert code == program
        if not native_results:
            result = run_c(code, timeout=timeout)
            assert result.status == "ok", result
            assert result.stdout == "42\n"
            native_results.append(result)
        # Replay one real result to keep timing fields identical in cache controls.
        return native_results[0]

    rows = run_serving_comparison(
        [
            Task(
                "declared-c",
                "unit-serving",
                "Compute six times seven.",
                IntegerChecker(42),
            )
        ],
        tokenizer,
        hybrid_model,
        lambda: None,
        path="current_best",
        executor=execute,
        tool_description=description,
        on_token_factory=make_sampler,
    )
    assert len(native_results) == 1 and replayed_sources == [program] * 3
    assert RUN_TOOL == declaration
    for row in rows:
        assert description in prompts[row.mode]
        assert declaration["function"]["description"] not in prompts[row.mode]
        assert row.case.outcome == "correct"
        assert row.case.calls[0].status == "ok"
        assert row.case.prompt_tokens is not None
        assert row.case.prompt_tokens == prompt_counts[row.mode]
        assert row.prefill_tokens > row.case.prompt_tokens
        assert row.final_tokens == rows[0].final_tokens
        assert row.final_logits is not None and rows[0].final_logits is not None
        torch.testing.assert_close(
            row.final_logits, rows[0].final_logits, atol=1e-6, rtol=1e-5
        )


def test_prefix_cached_resubmission_recovers_after_lru_eviction(
    tokenizer, hybrid_model
):
    task = Task("evict-case", "unit-serving", "Compute 3 * 9.", IntegerChecker(27))
    script_text = _call_text("print(3 * 9)") + "Final answer: 27<|im_end|>"

    def make_sampler(_task, _mode):
        stream = iter(tokenizer.encode(script_text, add_special_tokens=False))
        return lambda _session: next(stream)

    # Tight byte budget (1 byte) forces explicit error on capture, while a budget
    # sized for 1 entry evicts the prompt checkpoint when the tool boundary is saved.
    probe = run_serving_comparison(
        [task],
        tokenizer,
        hybrid_model,
        lambda: None,
        modes=("prefix_cached_resubmission",),
        executor=lambda code, timeout=20: ExecutionResult("ok", "27\n"),
        on_token_factory=make_sampler,
    )[0]
    one_slot_budget = probe.checkpoint_bytes_peak

    evicted_run = run_serving_comparison(
        [task],
        tokenizer,
        hybrid_model,
        lambda: None,
        modes=("prefix_cached_resubmission",),
        max_store_bytes=one_slot_budget,
        max_store_entries=1,
        executor=lambda code, timeout=20: ExecutionResult("ok", "27\n"),
        on_token_factory=make_sampler,
    )[0]
    assert evicted_run.case.outcome == "correct"
    assert evicted_run.checkpoints_evicted >= 1
    assert evicted_run.cache_hits >= 1


def test_serving_comparison_validation(tokenizer, hybrid_model):
    task = Task("val", "unit", "Compute 1 + 1.", IntegerChecker(2))
    with pytest.raises(ValueError, match="tool-enabled"):
        run_serving_comparison(
            [task], tokenizer, hybrid_model, lambda: None, path="no_tool"
        )
    with pytest.raises(ValueError, match="unique serving modes"):
        run_serving_comparison(
            [task],
            tokenizer,
            hybrid_model,
            lambda: None,
            modes=("live_pause_resume", "live_pause_resume"),
        )
    with pytest.raises(ValueError, match="at least one result"):
        summarize_serving([])


def test_invalid_tool_description_precedes_sampler_setup(tokenizer, hybrid_model):
    with pytest.raises(ValueError, match="tool_description"):
        run_serving_comparison(
            [Task("bad-declaration", "unit", "Compute.", IntegerChecker(42))],
            tokenizer,
            hybrid_model,
            lambda: pytest.fail("Must not synchronize"),
            tool_description="",
            on_token_factory=lambda *args: pytest.fail("Must not build a sampler"),
        )


@pytest.mark.parametrize("fault", ["missing", "duplicate", "family", "path"])
def test_serving_summary_rejects_unmatched_results(fault):
    case = CaseResult("a", "unit", "python_tool", "correct", "", "", 1, 3, 0, ())
    first = ServingCaseResult("live_pause_resume", case, (), None, 3, 0, 0, 0, 0)
    second = replace(first, mode="prefix_cached_resubmission")
    rows = [first, second]
    if fault == "missing":
        rows.append(replace(first, case=replace(case, case_id="b")))
    elif fault == "duplicate":
        rows.append(first)
    elif fault == "family":
        rows[1] = replace(second, case=replace(case, family="other"))
    else:
        rows[1] = replace(second, case=replace(case, path="current_best"))
    with pytest.raises(ValueError):
        summarize_serving(rows)


def test_serving_summary_exposes_opposing_paired_family_effects():
    rows = []
    for case_id, family, cached_time, live_correct in (
        ("a", "family-a", 6, True),
        ("b", "family-b", 2, False),
    ):
        case = CaseResult(
            case_id,
            family,
            "python_tool",
            "correct" if live_correct else "wrong_answer",
            "",
            "",
            4,
            3,
            0,
            (),
        )
        live = ServingCaseResult("live_pause_resume", case, (), None, 3, 0, 0, 0, 0)
        rows.extend(
            [
                live,
                replace(
                    live,
                    mode="prefix_cached_resubmission",
                    checkpoint_bytes_peak=64 if live_correct else 128,
                    cache_hits=2 if live_correct else 3,
                    case=replace(
                        case,
                        outcome="wrong_answer" if live_correct else "correct",
                        elapsed_seconds=cached_time,
                    ),
                ),
                replace(
                    live,
                    mode="full_reprefill_resubmission",
                    case=replace(case, elapsed_seconds=5 if live_correct else 7),
                ),
            ]
        )
    summary = summarize_serving(list(reversed(rows)))
    assert summary["paired_cases"] == 2
    cached_key = "prefix_cached_resubmission_vs_live_pause_resume"
    cached = summary["paired_deltas"][cached_key]
    assert cached["accuracy_delta"] == cached["elapsed_delta_mean_seconds"] == 0
    assert cached["target_only_correct"] == cached["baseline_only_correct"] == 1
    family_a = summary["paired_deltas_by_family"]["family-a"][cached_key]
    family_b = summary["paired_deltas_by_family"]["family-b"][cached_key]
    assert family_a["accuracy_delta"] == -1
    assert family_b["accuracy_delta"] == 1
    assert family_a["elapsed_delta_mean_seconds"] == 2
    assert family_b["elapsed_delta_mean_seconds"] == -2
    costs_a = summary["by_family"]["family-a"]
    costs_b = summary["by_family"]["family-b"]
    assert costs_a["live_pause_resume"]["seconds_per_correct"] == 4
    assert costs_a["prefix_cached_resubmission"]["seconds_per_correct"] is None
    assert costs_b["live_pause_resume"]["seconds_per_correct"] is None
    assert costs_b["prefix_cached_resubmission"]["seconds_per_correct"] == 2
    assert costs_a["live_pause_resume"]["prefill_tokens"] == 3
    assert summary["by_mode"]["live_pause_resume"]["prefill_tokens"] == 6
    assert costs_a["prefix_cached_resubmission"]["checkpoint_bytes_peak"] == 64
    assert costs_b["prefix_cached_resubmission"]["checkpoint_bytes_peak"] == 128
    assert (
        summary["by_mode"]["prefix_cached_resubmission"]["checkpoint_bytes_peak"] == 128
    )
    assert costs_a["prefix_cached_resubmission"]["cache_hits"] == 2
    assert costs_b["prefix_cached_resubmission"]["cache_hits"] == 3
    assert summary["by_mode"]["prefix_cached_resubmission"]["cache_hits"] == 5
    full = summary["paired_deltas"]["full_reprefill_resubmission_vs_live_pause_resume"]
    assert full["both_correct"] == full["neither_correct"] == 1
    assert full["elapsed_delta_mean_seconds"] == 2


@pytest.mark.parametrize(
    "modes",
    [
        ("prefix_cached_resubmission", "full_reprefill_resubmission"),
        ("full_reprefill_resubmission",),
    ],
)
def test_serving_paired_deltas_use_only_present_modes(modes):
    case = CaseResult("a", "unit", "python_tool", "correct", "", "", 1, 3, 0, ())
    rows = [
        ServingCaseResult(
            mode,
            case
            if mode == "full_reprefill_resubmission"
            else replace(case, outcome="wrong_answer", elapsed_seconds=2),
            (),
            None,
            3,
            0,
            0,
            0,
            0,
        )
        for mode in reversed(modes)
    ]
    summary = summarize_serving(rows)
    assert summary["paired_cases"] == 1
    assert set(summary["by_mode"]) == set(modes)
    if len(modes) == 1:
        assert summary["paired_deltas"] == {}
        assert summary["paired_deltas_by_family"] == {"unit": {}}
    else:
        key = "full_reprefill_resubmission_vs_prefix_cached_resubmission"
        assert set(summary["paired_deltas"]) == {key}
        delta = summary["paired_deltas"][key]
        assert delta["accuracy_delta"] == 1
        assert delta["elapsed_delta_mean_seconds"] == -1


def test_default_snapshot_limit_is_not_the_aggregate_store_budget(
    hybrid_model, monkeypatch
):
    from neural_compiler.bridge import session

    monkeypatch.setattr(session, "_snapshot_bytes", lambda *_: 64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="exceeding"):
        PrefixCachedSession(hybrid_model, [1, 2, 3], CheckpointStore())


@pytest.mark.parametrize("boundary", ["initial", "turn"])
def test_checkpoint_budget_failure_retains_the_matched_case(
    tokenizer, hybrid_model, monkeypatch, boundary
):
    from neural_compiler.bridge import session

    snapshot_bytes = session._snapshot_bytes
    snapshots = []

    def bounded_size(cache, logits):
        snapshots.append(None)
        if boundary == "initial" or len(snapshots) > 1:
            return session.DEFAULT_CHECKPOINT_BYTES + 1
        return snapshot_bytes(cache, logits)

    monkeypatch.setattr(session, "_snapshot_bytes", bounded_size)
    call = _call_text("print(6 * 7)")
    script = call + "Final answer: 42<|im_end|>"

    def sampler_factory(_task, _mode):
        stream = iter(tokenizer.encode(script, add_special_tokens=False))
        return lambda _: next(stream)

    rows = run_serving_comparison(
        [Task("cache-budget", "unit-serving", "Compute 6 * 7.", IntegerChecker(42))],
        tokenizer,
        hybrid_model,
        lambda: None,
        executor=lambda code, timeout=20: ExecutionResult("ok", "42\n"),
        on_token_factory=sampler_factory,
    )
    assert len(rows) == len(ALL_SERVING_MODES)
    cached = next(row for row in rows if row.mode == "prefix_cached_resubmission")
    assert cached.case.outcome == "budget_exhausted"
    assert "exceeding" in cached.case.diagnostic
    assert cached.case.calls == () and cached.case.injected_tokens == 0
    expected_generated = (
        0
        if boundary == "initial"
        else len(tokenizer.encode(call, add_special_tokens=False))
    )
    assert cached.case.generated_tokens == expected_generated
    assert cached.prefill_tokens > 0
    assert len(cached.final_tokens) == cached.prefill_tokens + expected_generated
    if boundary == "initial":
        assert cached.final_logits is None
        assert cached.logit_prefix_tokens is None
    assert all(row.case.outcome == "correct" for row in rows if row is not cached)
    summary = summarize_serving(rows)["by_mode"]["prefix_cached_resubmission"]
    assert summary["cases"] == 1 and summary["outcomes"]["budget_exhausted"] == 1


def test_unsupported_checkpoint_is_not_a_case_budget_failure(
    tokenizer, hybrid_model, monkeypatch
):
    from neural_compiler.bridge import session

    def unsupported(*_args):
        raise ValueError("Unsupported synthetic cache")

    monkeypatch.setattr(session, "_snapshot_bytes", unsupported)
    with pytest.raises(ValueError, match="Unsupported synthetic cache"):
        run_serving_comparison(
            [Task("unsupported", "unit-serving", "Compute 1 + 1.", IntegerChecker(2))],
            tokenizer,
            hybrid_model,
            lambda: None,
            modes=("prefix_cached_resubmission",),
        )


def test_comparison_forwards_explicit_snapshot_override(
    tokenizer, hybrid_model, monkeypatch
):
    from neural_compiler.bridge import session

    size = 65 * 1024 * 1024
    monkeypatch.setattr(session, "_snapshot_bytes", lambda *_: size)
    task = Task("explicit-cap", "unit-serving", "Compute 1 + 1.", IntegerChecker(2))

    def sampler_factory(_task, _mode):
        stream = iter(
            tokenizer.encode("Final answer: 2<|im_end|>", add_special_tokens=False)
        )
        return lambda _: next(stream)

    default = run_serving_comparison(
        [task],
        tokenizer,
        hybrid_model,
        lambda: None,
        modes=("prefix_cached_resubmission",),
        on_token_factory=sampler_factory,
    )[0]
    assert default.case.outcome == "budget_exhausted"
    explicit = run_serving_comparison(
        [task],
        tokenizer,
        hybrid_model,
        lambda: None,
        modes=("prefix_cached_resubmission",),
        max_checkpoint_bytes=size,
        on_token_factory=sampler_factory,
    )[0]
    assert explicit.case.outcome == "correct"
    assert explicit.checkpoint_bytes_peak == size


@pytest.mark.parametrize("limit", [False, 0])
def test_comparison_snapshot_override_is_validated_before_prefill(
    tokenizer, hybrid_model, limit
):
    with pytest.raises(ValueError, match="max_checkpoint_bytes"):
        run_serving_comparison(
            [Task("bad-cap", "unit-serving", "Compute.", IntegerChecker(2))],
            tokenizer,
            hybrid_model,
            lambda: None,
            max_checkpoint_bytes=limit,
            on_token_factory=lambda *_: pytest.fail("Sampler/prefill started"),
        )


def test_malformed_turn_retains_all_sampled_tokens_in_every_serving_mode(
    tokenizer, hybrid_model
):
    text = "<tool_call>bad</tool_call>"

    def sampler_factory(_task, _mode):
        stream = iter(tokenizer.encode(text, add_special_tokens=False))
        return lambda _session: next(stream)

    rows = run_serving_comparison(
        [Task("malformed", "unit-serving", "Compute 1 + 1.", IntegerChecker(2))],
        tokenizer,
        hybrid_model,
        lambda: None,
        executor=lambda *_args, **_kwargs: pytest.fail("Malformed call executed"),
        on_token_factory=sampler_factory,
    )
    sampled = tuple(tokenizer.encode(text, add_special_tokens=False))
    for row in rows:
        assert row.case.outcome == "protocol_error"
        assert row.case.generated_tokens == len(sampled)
        assert row.final_tokens[-len(sampled) :] == sampled
        assert row.final_tokens == rows[0].final_tokens


def test_full_reprefill_retains_pending_token_after_callback_abort(hybrid_model):
    live = LivePauseResumeSession(hybrid_model, [1, 2, 3], lambda _session: 4)
    reprefill = FullReprefillSession(hybrid_model, [1, 2, 3], lambda _session: 4)

    def abort(_token):
        raise ValueError("fixture callback abort")

    for adapter in (live, reprefill):
        with pytest.raises(ValueError, match="callback abort"):
            adapter.decode_until(1, abort)
        adapter.inject([5, 6])
        assert adapter.tokens == (1, 2, 3, 4, 5, 6)
    torch.testing.assert_close(reprefill.logits, live.logits, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("changed", ["weights", "configuration"])
def test_full_reprefill_rejects_model_drift_across_turns(hybrid_model, changed):
    adapter = FullReprefillSession(hybrid_model, [1, 2, 3], lambda _session: 4)
    adapter.decode_until(1, lambda _token: True)
    boundary = adapter.tokens
    if changed == "weights":
        with torch.no_grad():
            next(hybrid_model.parameters()).add_(0.1)
    else:
        hybrid_model.config.rope_parameters["rope_theta"] = 20000
    with pytest.raises(ValueError, match="weights|configuration"):
        adapter.inject([5, 6])
    with pytest.raises(ValueError, match="weights|configuration"):
        adapter.decode_until(1)
    assert adapter.tokens == boundary


def test_prefix_cache_miss_does_not_relabel_a_changed_model(hybrid_model):
    store = CheckpointStore()
    adapter = PrefixCachedSession(
        hybrid_model, [1, 2, 3], store, on_token_override=lambda _session: 4
    )
    adapter.decode_until(1, lambda _token: True)
    boundary = adapter.tokens
    store.clear()
    with torch.no_grad():
        next(hybrid_model.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="weights|configuration"):
        adapter.inject([5, 6])
    with pytest.raises(ValueError, match="weights|configuration"):
        adapter.decode_until(1)
    assert adapter.tokens == boundary


@pytest.mark.parametrize("mode", ALL_SERVING_MODES)
def test_serving_modes_reject_changed_eos_settings(hybrid_model, mode):
    hybrid_model.generation_config.eos_token_id = 5
    if mode == "live_pause_resume":
        adapter = LivePauseResumeSession(hybrid_model, [1, 2, 3], lambda _: 4)
    elif mode == "prefix_cached_resubmission":
        store = CheckpointStore()
        adapter = PrefixCachedSession(
            hybrid_model, [1, 2, 3], store, on_token_override=lambda _: 4
        )
    else:
        adapter = FullReprefillSession(hybrid_model, [1, 2, 3], lambda _: 4)
    adapter.decode_until(1, lambda _: True)
    boundary = adapter.tokens
    if mode == "prefix_cached_resubmission":
        store.clear()
    hybrid_model.generation_config.eos_token_id = 4
    with pytest.raises(ValueError, match="configuration"):
        adapter.inject([6, 7])
    with pytest.raises(ValueError, match="configuration"):
        adapter.decode_until(1)
    assert adapter.tokens == boundary
