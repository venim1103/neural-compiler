"""Offline P0 serving comparison checks on tiny CPU hybrid Qwen fixtures."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from neural_compiler.bridge import ExecutionResult
from neural_compiler.eval import (
    ALL_SERVING_MODES,
    IntegerChecker,
    PrefixCachedSession,
    ServingCaseResult,
    Task,
    run_serving_comparison,
    summarize_serving,
)
from neural_compiler.eval.harness import CaseResult
from neural_compiler.bridge.session import CheckpointStore


def _call_text(code: str) -> str:
    return (
        "<tool_call>\n<function=run>\n<parameter=code>\n"
        f"{code}\n</parameter>\n</function>\n</tool_call>"
    )


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
        executor=lambda code, timeout: ExecutionResult("ok", "27\n"),
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
        executor=lambda code, timeout: ExecutionResult("ok", "27\n"),
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


def test_default_snapshot_limit_is_not_the_aggregate_store_budget(
    hybrid_model, monkeypatch
):
    from neural_compiler.bridge import session

    monkeypatch.setattr(session, "_snapshot_bytes", lambda *_: 64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="exceeding"):
        PrefixCachedSession(hybrid_model, [1, 2, 3], CheckpointStore())
