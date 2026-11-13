"""Offline P0 serving comparison: live pause/resume vs. prefix-cached resubmission.

These adapters wrap `LiveSession` and `CheckpointStore` behind the standard
`DecodeSession` interface so `EvaluationHarness` can compare serving strategies
under identical tasks, prompts, checkers, and token/time budgets without altering
execution contracts or claiming unmeasured hardware speedups.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from neural_compiler.bridge import run_python
from neural_compiler.bridge.session import (
    DEFAULT_CHECKPOINT_BYTES,
    CheckpointStore,
    DecodeResult,
    LiveSession,
    _validate_prefix_model,
)

from .harness import (
    Budgets,
    CaseResult,
    EvaluationHarness,
    Executor,
    Path,
    Task,
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


class LivePauseResumeSession:
    """Keep one `LiveSession` resident across tool calls with incremental inject."""

    def __init__(
        self,
        model: PreTrainedModel,
        prompt_ids: Sequence[int],
        on_token_override: Callable[[LiveSession], int] | None = None,
    ) -> None:
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
    mode: ServingMode
    case: CaseResult
    final_tokens: tuple[int, ...]
    final_logits: torch.Tensor | None
    prefill_tokens: int
    checkpoint_bytes_peak: int
    cache_hits: int
    cache_misses: int
    checkpoints_evicted: int


def run_serving_comparison(
    tasks: Sequence[Task],
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    synchronize: Callable[[], None],
    *,
    path: Path = "python_tool",
    budgets: Budgets = Budgets(),
    executor: Executor = run_python,
    modes: Sequence[ServingMode] = ALL_SERVING_MODES,
    max_store_bytes: int = 256 * 1024 * 1024,
    max_store_entries: int = 32,
    on_token_factory: Callable[[Task, ServingMode], Callable[[LiveSession], int]]
    | None = None,
) -> tuple[ServingCaseResult, ...]:
    """Run matched tasks across serving modes with shared budgets and executor."""
    if path == "no_tool":
        raise ValueError("Serving continuation comparison requires a tool-enabled path")
    if not modes or len(set(modes)) != len(modes):
        raise ValueError("modes must be a non-empty sequence of unique serving modes")
    for mode in modes:
        if mode not in ALL_SERVING_MODES:
            raise ValueError(f"Unknown serving mode: {mode}")
    if len({task.case_id for task in tasks}) != len(tasks):
        raise ValueError("Tasks must have unique case IDs")

    results: list[ServingCaseResult] = []
    for task in tasks:
        for mode in modes:
            sampler = None if on_token_factory is None else on_token_factory(task, mode)
            store = CheckpointStore(
                max_total_bytes=max_store_bytes, max_entries=max_store_entries
            )
            active: list[
                LivePauseResumeSession | PrefixCachedSession | FullReprefillSession
            ] = []

            def factory(prompt_ids: Sequence[int]):
                if mode == "live_pause_resume":
                    session = LivePauseResumeSession(model, prompt_ids, sampler)
                elif mode == "prefix_cached_resubmission":
                    session = PrefixCachedSession(
                        model,
                        prompt_ids,
                        store,
                        on_token_override=sampler,
                    )
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
            )
            case_result = harness.run(task, path)
            last_session = active[-1] if active else None
            results.append(
                ServingCaseResult(
                    mode=mode,
                    case=case_result,
                    final_tokens=() if last_session is None else last_session.tokens,
                    final_logits=None
                    if last_session is None
                    else last_session.logits.detach().cpu(),
                    prefill_tokens=0
                    if last_session is None
                    else last_session.prefill_tokens,
                    checkpoint_bytes_peak=getattr(
                        last_session, "checkpoint_bytes_peak", 0
                    ),
                    cache_hits=store.hits,
                    cache_misses=store.misses,
                    checkpoints_evicted=store.evictions,
                )
            )
    return tuple(results)


def summarize_serving(results: Sequence[ServingCaseResult]) -> dict[str, object]:
    """Aggregate failure-inclusive metrics for complete matched serving cases."""
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
    by_mode: dict[str, object] = {}
    for mode in modes_present:
        subset = [r for r in results if r.mode == mode]
        base = summarize([r.case for r in subset])
        base["prefill_tokens"] = sum(r.prefill_tokens for r in subset)
        base["checkpoint_bytes_peak"] = max(r.checkpoint_bytes_peak for r in subset)
        base["cache_hits"] = sum(r.cache_hits for r in subset)
        base["cache_misses"] = sum(r.cache_misses for r in subset)
        base["checkpoints_evicted"] = sum(r.checkpoints_evicted for r in subset)
        by_mode[mode] = base
    return {"by_mode": by_mode}
