"""Offline P0 serving comparison: live pause/resume vs. prefix-cached resubmission.

These adapters wrap `LiveSession` and `CheckpointStore` behind the standard
`DecodeSession` interface so `EvaluationHarness` can compare serving strategies
under identical tasks, prompts, checkers, and token/time budgets without altering
execution contracts or claiming unmeasured hardware speedups.
Direct adapters reject non-callable sampler overrides before model prefill or
checkpoint-store access.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from neural_compiler.bridge import run_python
from neural_compiler.bridge.session import (
    DEFAULT_CHECKPOINT_BYTES,
    CheckpointBudgetError,
    CheckpointStore,
    DecodeResult,
    LiveSession,
    _validate_checkpoint_budget,
    _validate_decode_request,
    _validate_prefix_model,
    _validate_store_limits,
    _validate_token_ids,
)

from .harness import (
    Budgets,
    CaseResult,
    EvaluationHarness,
    Executor,
    Path,
    Task,
    _PATHS,
    _paired_comparison,
    _paired_observation_coverage,
    _validate_tool_description,
    summarize,
)

ServingMode = Literal[
    "live_pause_resume",
    "prefix_cached_resubmission",
    "full_reprefill_resubmission",
]
ALL_SERVING_MODES: tuple[ServingMode, ...] = (
    "live_pause_resume",
    "prefix_cached_resubmission",
    "full_reprefill_resubmission",
)


def _validate_sampler_override(
    on_token_override: Callable[[LiveSession], int] | None,
) -> None:
    if on_token_override is not None and not callable(on_token_override):
        raise ValueError("on_token_override must be callable or None")


class LivePauseResumeSession:
    """Keep one `LiveSession` resident across tool calls with incremental inject."""

    def __init__(
        self,
        model: PreTrainedModel,
        prompt_ids: Sequence[int],
        on_token_override: Callable[[LiveSession], int] | None = None,
    ) -> None:
        _validate_sampler_override(on_token_override)
        self._session = LiveSession(model, prompt_ids)
        if on_token_override is not None:
            self._session._next_token = lambda: on_token_override(self._session)
        self.prefill_tokens = len(prompt_ids)

    @property
    def tokens(self) -> tuple[int, ...]:
        return self._session.tokens

    @property
    def logits(self) -> torch.Tensor:
        return self._session.logits

    def decode_until(
        self,
        max_new_tokens: int,
        on_token: Callable[[int], bool] | None = None,
    ) -> DecodeResult:
        return self._session.decode_until(max_new_tokens, on_token)

    def inject(self, response_ids: Sequence[int]) -> None:
        if not response_ids:
            raise ValueError("Tool response must not be empty")
        pending = 1 if self._session._pending is not None else 0
        self._session.inject(response_ids)
        self.prefill_tokens += len(response_ids) + pending


class PrefixCachedSession:
    """Stateless turn resubmission backed by an exact-boundary `CheckpointStore`."""

    def __init__(
        self,
        model: PreTrainedModel,
        prompt_ids: Sequence[int],
        store: CheckpointStore,
        max_checkpoint_bytes: int | None = None,
        on_token_override: Callable[[LiveSession], int] | None = None,
    ) -> None:
        self.model = model
        self.store = store
        self.max_checkpoint_bytes = (
            min(DEFAULT_CHECKPOINT_BYTES, store.max_total_bytes)
            if max_checkpoint_bytes is None
            else max_checkpoint_bytes
        )
        _validate_checkpoint_budget(self.max_checkpoint_bytes)
        _validate_sampler_override(on_token_override)
        self._on_token_override = on_token_override
        self._session: LiveSession | None
        self._session, self.prefill_tokens = store.resume_prefix(model, prompt_ids)
        self._prefix_tensor_versions = self._session._prefix_tensor_versions
        self._prefix_config = self._session._prefix_config
        self._bind_sampler(self._session)
        self._last_tokens: tuple[int, ...] = self._session.tokens
        self._last_logits: torch.Tensor = self._session.logits.detach().clone()
        self.checkpoint_bytes_peak = store.total_bytes
        if self.prefill_tokens > 0:
            cp = self._session.checkpoint(self.max_checkpoint_bytes)
            store.put(cp)
            self.checkpoint_bytes_peak = max(
                self.checkpoint_bytes_peak, store.total_bytes
            )

    def _bind_sampler(self, session: LiveSession) -> None:
        if self._on_token_override is not None:
            session._next_token = lambda: self._on_token_override(session)

    @property
    def tokens(self) -> tuple[int, ...]:
        return self._last_tokens if self._session is None else self._session.tokens

    @property
    def logits(self) -> torch.Tensor:
        return self._last_logits if self._session is None else self._session.logits

    def decode_until(
        self,
        max_new_tokens: int,
        on_token: Callable[[int], bool] | None = None,
    ) -> DecodeResult:
        _validate_decode_request(max_new_tokens, on_token)
        if self._session is None:
            _validate_prefix_model(
                self.model, self._prefix_tensor_versions, self._prefix_config
            )
            self._session, step_prefilled = self.store.resume_prefix(
                self.model, self._last_tokens
            )
            self._bind_sampler(self._session)
            self.prefill_tokens += step_prefilled
        result = self._session.decode_until(max_new_tokens, on_token)
        self._last_tokens = self._session.tokens
        self._last_logits = self._session.logits.detach().clone()
        if result.reason == "stop":
            cp = self._session.checkpoint(self.max_checkpoint_bytes)
            self.store.put(cp)
            self.checkpoint_bytes_peak = max(
                self.checkpoint_bytes_peak, self.store.total_bytes
            )
            # Release live session to model stateless turn boundary handoff.
            self._session = None
        return result

    def inject(self, response_ids: Sequence[int]) -> None:
        if not response_ids:
            raise ValueError("Tool response must not be empty")
        _validate_prefix_model(
            self.model, self._prefix_tensor_versions, self._prefix_config
        )
        base_tokens = (
            self._last_tokens if self._session is None else self._session.tokens
        )
        resubmitted = (*base_tokens, *response_ids)
        self._session, step_prefilled = self.store.resume_prefix(
            self.model, resubmitted
        )
        self._bind_sampler(self._session)
        self.prefill_tokens += step_prefilled
        self._last_tokens = self._session.tokens
        self._last_logits = self._session.logits.detach().clone()


class FullReprefillSession:
    """Stateless turn resubmission without prefix cache (full context re-prefill)."""

    def __init__(
        self,
        model: PreTrainedModel,
        prompt_ids: Sequence[int],
        on_token_override: Callable[[LiveSession], int] | None = None,
    ) -> None:
        _validate_sampler_override(on_token_override)
        self.model = model
        self._on_token_override = on_token_override
        self._session: LiveSession | None = LiveSession(model, prompt_ids)
        self._prefix_tensor_versions = self._session._prefix_tensor_versions
        self._prefix_config = self._session._prefix_config
        self._bind_sampler(self._session)
        self._tokens = list(prompt_ids)
        self._last_logits: torch.Tensor = self._session.logits.detach().clone()
        self.prefill_tokens = len(prompt_ids)

    def _bind_sampler(self, session: LiveSession) -> None:
        if self._on_token_override is not None:
            session._next_token = lambda: self._on_token_override(session)

    @property
    def tokens(self) -> tuple[int, ...]:
        return tuple(self._tokens) if self._session is None else self._session.tokens

    @property
    def logits(self) -> torch.Tensor:
        return self._last_logits if self._session is None else self._session.logits

    def decode_until(
        self,
        max_new_tokens: int,
        on_token: Callable[[int], bool] | None = None,
    ) -> DecodeResult:
        _validate_decode_request(max_new_tokens, on_token)
        if self._session is None:
            _validate_prefix_model(
                self.model, self._prefix_tensor_versions, self._prefix_config
            )
            self._session = LiveSession(self.model, self._tokens)
            self._bind_sampler(self._session)
            self.prefill_tokens += len(self._tokens)
        result = self._session.decode_until(max_new_tokens, on_token)
        self._tokens = list(self._session.tokens)
        self._last_logits = self._session.logits.detach().clone()
        if result.reason == "stop":
            self._session = None
        return result

    def inject(self, response_ids: Sequence[int]) -> None:
        if not response_ids:
            raise ValueError("Tool response must not be empty")
        _validate_prefix_model(
            self.model, self._prefix_tensor_versions, self._prefix_config
        )
        tokens = [*self.tokens, *response_ids]
        session = LiveSession(self.model, tokens)
        self._bind_sampler(session)
        self._tokens = tokens
        self._session = session
        self.prefill_tokens += len(tokens)
        self._last_logits = session.logits.detach().clone()


@dataclass(frozen=True)
class ServingCaseResult:
    """Captured logits can precede a pending final ID in the token history.

    Token histories are immutable tuples of nonnegative integer IDs.
    logit_prefix_tokens identifies their observed prefix; None is unmeasured.
    """

    mode: ServingMode
    case: CaseResult
    final_tokens: tuple[int, ...]
    final_logits: torch.Tensor | None
    prefill_tokens: int
    checkpoint_bytes_peak: int
    cache_hits: int
    cache_misses: int
    checkpoints_evicted: int
    logit_prefix_tokens: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "prefill_tokens",
            "checkpoint_bytes_peak",
            "cache_hits",
            "cache_misses",
            "checkpoints_evicted",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not isinstance(self.final_tokens, tuple):
            raise ValueError("final_tokens must be an immutable tuple")
        _validate_token_ids(self.final_tokens)
        if any(token < 0 for token in self.final_tokens):
            raise ValueError("final_tokens must contain nonnegative token IDs")
        value = self.logit_prefix_tokens
        if value is not None and (
            type(value) is not int or not 0 <= value <= len(self.final_tokens)
        ):
            raise ValueError(
                "logit_prefix_tokens must be an integer within final_tokens"
            )


def run_serving_comparison(
    tasks: Sequence[Task],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    synchronize: Callable[[], None],
    *,
    path: Path = "python_tool",
    budgets: Budgets = Budgets(),
    executor: Executor = run_python,
    tool_description: str | None = None,
    modes: Sequence[ServingMode] = ALL_SERVING_MODES,
    max_store_bytes: int = 256 * 1024 * 1024,
    max_store_entries: int = 32,
    max_checkpoint_bytes: int | None = None,
    on_token_factory: Callable[[Task, ServingMode], Callable[[LiveSession], int]]
    | None = None,
) -> tuple[ServingCaseResult, ...]:
    """Run matched tasks with shared budgets and an optional cached snapshot override."""
    if path == "no_tool":
        raise ValueError("Serving continuation comparison requires a tool-enabled path")
    if path not in _PATHS:
        raise ValueError(f"Unknown evaluation path: {path}")
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("modes must be a non-empty sequence of unique serving modes")
    if max_checkpoint_bytes is not None and (
        type(max_checkpoint_bytes) is not int or max_checkpoint_bytes <= 0
    ):
        raise ValueError("max_checkpoint_bytes must be a positive integer")
    for mode in modes:
        if mode not in ALL_SERVING_MODES:
            raise ValueError(f"Unknown serving mode: {mode}")
    if len({task.case_id for task in tasks}) != len(tasks):
        raise ValueError("Tasks must have unique case IDs")
    _validate_tool_description(tool_description)
    _validate_store_limits(max_store_bytes, max_store_entries)
    if on_token_factory is not None and not callable(on_token_factory):
        raise ValueError("on_token_factory must be callable")

    results: list[ServingCaseResult] = []
    for task in tasks:
        for mode in modes:
            sampler = None if on_token_factory is None else on_token_factory(task, mode)
            if on_token_factory is not None and not callable(sampler):
                raise ValueError("on_token_factory must return a callable sampler")
            store = CheckpointStore(
                max_total_bytes=max_store_bytes, max_entries=max_store_entries
            )
            active: list[
                LivePauseResumeSession | PrefixCachedSession | FullReprefillSession
            ] = []
            failed_prompt: tuple[int, ...] = ()

            def factory(prompt_ids: Sequence[int]):
                nonlocal failed_prompt
                if mode == "live_pause_resume":
                    session = LivePauseResumeSession(model, prompt_ids, sampler)
                elif mode == "prefix_cached_resubmission":
                    try:
                        session = PrefixCachedSession(
                            model,
                            prompt_ids,
                            store,
                            max_checkpoint_bytes=max_checkpoint_bytes,
                            on_token_override=sampler,
                        )
                    except CheckpointBudgetError:
                        # The fresh per-case store cold-prefills before capture.
                        failed_prompt = tuple(prompt_ids)
                        raise
                else:
                    session = FullReprefillSession(model, prompt_ids, sampler)
                active.append(session)
                return session

            harness = EvaluationHarness(
                tokenizer,
                factory,
                synchronize,
                budgets=budgets,
                executor=executor,
                tool_description=tool_description,
            )
            case_result = harness.run(task, path)
            last_session = active[-1] if active else None
            logit_prefix_tokens = None
            if last_session is not None:
                native = last_session._session
                # Stateless adapters release only at a pending callback-stop boundary.
                pending = native is None or native._pending is not None
                logit_prefix_tokens = len(last_session.tokens) - pending
            results.append(
                ServingCaseResult(
                    mode=mode,
                    case=case_result,
                    final_tokens=failed_prompt
                    if last_session is None
                    else last_session.tokens,
                    final_logits=None
                    if last_session is None
                    else last_session.logits.detach().to("cpu", copy=True),
                    prefill_tokens=len(failed_prompt)
                    if last_session is None
                    else last_session.prefill_tokens,
                    checkpoint_bytes_peak=getattr(
                        last_session, "checkpoint_bytes_peak", 0
                    ),
                    cache_hits=store.hits,
                    cache_misses=store.misses,
                    checkpoints_evicted=store.evictions,
                    logit_prefix_tokens=logit_prefix_tokens,
                )
            )
    return tuple(results)


@dataclass(frozen=True)
class _ServingPairs:
    rows: dict[str, dict[ServingMode, ServingCaseResult]]
    cases: dict[str, dict[str, CaseResult]]
    modes: tuple[ServingMode, ...]
    coverage: dict[str, int]


def _pair_serving_results(results: Sequence[ServingCaseResult]) -> _ServingPairs:
    if not results:
        raise ValueError("Serving summary requires at least one result")
    paired: dict[str, dict[ServingMode, ServingCaseResult]] = {}
    for row in results:
        if row.mode not in ALL_SERVING_MODES:
            raise ValueError(f"Unknown serving mode: {row.mode}")
        case = paired.setdefault(row.case.case_id, {})
        if row.mode in case:
            raise ValueError(f"Duplicate serving pair: {row.case.case_id}/{row.mode}")
        case[row.mode] = row
    modes_present = [
        mode for mode in ALL_SERVING_MODES if any(r.mode == mode for r in results)
    ]
    if len({row.case.path for row in results}) != 1:
        raise ValueError("Serving comparisons require the same evaluation path")
    for case_id, case in paired.items():
        if set(case) != set(modes_present):
            raise ValueError(f"Missing serving modes for {case_id}")
        if len({row.case.family for row in case.values()}) != 1:
            raise ValueError(f"Family identity differs between modes for {case_id}")
    paired_cases: dict[str, dict[str, CaseResult]] = {
        case_id: {mode: row.case for mode, row in case.items()}
        for case_id, case in paired.items()
    }
    coverage = _paired_observation_coverage(paired_cases)
    for case_id, case in paired.items():
        observed_prompts = {
            row.case.prompt_tokens
            for row in case.values()
            if row.case.prompt_tokens is not None
        }
        if len(observed_prompts) > 1:
            raise ValueError(
                f"Prompt token counts differ between serving modes for {case_id}"
            )
    return _ServingPairs(paired, paired_cases, tuple(modes_present), coverage)


def _observed_logits(row: ServingCaseResult) -> torch.Tensor:
    logits = row.final_logits
    if (
        not isinstance(logits, torch.Tensor)
        or logits.device.type != "cpu"
        or logits.layout != torch.strided
        or logits.requires_grad
        or logits.dtype
        not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
        or logits.ndim != 2
        or logits.shape[0] != 1
        or not logits.numel()
    ):
        raise ValueError(
            f"Detached nonempty single-sequence CPU logits required for "
            f"{row.case.case_id}/{row.mode}"
        )
    if not torch.isfinite(logits.amax(dim=-1)).all().item():
        raise ValueError(
            f"Logits require a finite maximum for {row.case.case_id}/{row.mode}"
        )
    return logits


def validate_serving_logits(
    results: Sequence[ServingCaseResult],
) -> dict[str, object]:
    """Check recorded same-boundary CPU logits, not live-engine qualification.

    Live/prefix agreement is exact. Full-reprefill comparisons use the existing
    tiny CPU hybrid fixture tolerance; no task outcomes are rewritten.
    """
    comparison = _pair_serving_results(results)
    reference_mode: ServingMode = "full_reprefill_resubmission"
    if reference_mode not in comparison.modes or len(comparison.modes) < 2:
        raise ValueError(
            "Logit checks require a full-reprefill reference and another mode"
        )
    atol, rtol = 1e-6, 1e-5
    full_comparisons = exact_pairs = 0
    for case_id, rows in comparison.rows.items():
        reference = rows[reference_mode]
        if reference.logit_prefix_tokens is None or reference.logit_prefix_tokens <= 0:
            raise ValueError(f"Measured nonempty logit prefix required for {case_id}")
        _validate_token_ids(reference.final_tokens)
        reference_logits = _observed_logits(reference)
        observed = {reference_mode: reference_logits}
        for mode, row in rows.items():
            if mode == reference_mode:
                continue
            _validate_token_ids(row.final_tokens)
            if row.final_tokens != reference.final_tokens:
                raise ValueError(
                    f"Token history differs from full reprefill for {case_id}/{mode}"
                )
            if row.logit_prefix_tokens != reference.logit_prefix_tokens:
                raise ValueError(
                    f"Logit prefix differs or is unmeasured for {case_id}/{mode}"
                )
            logits = _observed_logits(row)
            if (
                logits.shape != reference_logits.shape
                or logits.dtype != reference_logits.dtype
            ):
                raise ValueError(f"Logit shape/dtype differs for {case_id}/{mode}")
            if not torch.allclose(logits, reference_logits, atol=atol, rtol=rtol):
                raise ValueError(
                    f"Logits differ from full reprefill for {case_id}/{mode}"
                )
            observed[mode] = logits
            full_comparisons += 1
        if "live_pause_resume" in observed and "prefix_cached_resubmission" in observed:
            if not torch.equal(
                observed["live_pause_resume"], observed["prefix_cached_resubmission"]
            ):
                raise ValueError(f"Live/prefix logits differ for {case_id}")
            exact_pairs += 1
    return {
        "cases": len(comparison.rows),
        "full_reprefill_comparisons": full_comparisons,
        "cache_live_exact_pairs": exact_pairs,
        "atol": atol,
        "rtol": rtol,
    }


def summarize_serving(results: Sequence[ServingCaseResult]) -> dict[str, object]:
    """Failure-inclusive aggregate/family metrics and matched paired deltas.

    Comparisons use the first present mode in canonical order as baseline:
    live pause/resume when present, otherwise prefix-cached resubmission.
    A single-mode summary has no paired deltas.
    Observed task-text/budget/prompt-count mismatches are rejected within a case;
    missing observations remain unknown.
    """
    comparison = _pair_serving_results(results)
    paired_cases = comparison.cases
    modes_present = comparison.modes
    coverage = comparison.coverage

    def summarize_mode(subset: Sequence[ServingCaseResult]) -> dict[str, object]:
        base = summarize([r.case for r in subset])
        base["prefill_tokens"] = sum(r.prefill_tokens for r in subset)
        base["checkpoint_bytes_peak"] = max(r.checkpoint_bytes_peak for r in subset)
        base["cache_hits"] = sum(r.cache_hits for r in subset)
        base["cache_misses"] = sum(r.cache_misses for r in subset)
        base["checkpoints_evicted"] = sum(r.checkpoints_evicted for r in subset)
        return base

    by_mode = {
        mode: summarize_mode([r for r in results if r.mode == mode])
        for mode in modes_present
    }
    baseline = modes_present[0]

    def deltas(cases: Mapping[str, Mapping[str, CaseResult]]) -> dict[str, object]:
        return {
            f"{mode}_vs_{baseline}": _paired_comparison(cases, mode, baseline)
            for mode in modes_present[1:]
        }

    families = sorted({row.case.family for row in results})
    return {
        "by_mode": by_mode,
        "by_family": {
            family: {
                mode: summarize_mode(
                    [
                        row
                        for row in results
                        if row.mode == mode and row.case.family == family
                    ]
                )
                for mode in modes_present
            }
            for family in families
        },
        "paired_cases": len(paired_cases),
        **coverage,
        "paired_deltas": deltas(paired_cases),
        "paired_deltas_by_family": {
            family: deltas(
                {
                    case_id: case
                    for case_id, case in paired_cases.items()
                    if case[baseline].family == family
                }
            )
            for family in families
        },
    }
