"""Single-sequence, greedy decode with live hybrid-cache continuation."""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    DynamicLayer,
    LinearAttentionLayer,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

DEFAULT_CHECKPOINT_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class DecodeResult:
    token_ids: tuple[int, ...]
    reason: Literal["stop", "eos", "length"]


@dataclass(frozen=True)
class SessionCheckpoint:
    """Process-local CPU snapshot; private tensors are never lent to resumed sessions."""

    model: PreTrainedModel
    tokens: tuple[int, ...]
    pending_token: int | None
    cache_bytes: int
    _cache: DynamicCache
    _logits: torch.Tensor
    _tensor_versions: tuple[tuple[int, int, int], ...]
    _config: str
    _eos: frozenset[int]


def _snapshot_bytes(cache: Cache, logits: torch.Tensor) -> int:
    if type(cache) is not DynamicCache or cache.offloading:
        raise ValueError("Checkpoints support only non-offloaded DynamicCache")
    storages = {}

    def count(value) -> None:
        if isinstance(value, torch.Tensor):
            if value.device.type != "cpu" or value.requires_grad:
                raise ValueError("Checkpoints require CPU inference tensors")
            storage = value.untyped_storage()
            storages[storage.data_ptr()] = storage.nbytes()
        elif isinstance(value, (tuple, list)):
            for item in value:
                count(item)
        elif isinstance(value, dict):
            for item in value.values():
                count(item)

    count(logits)
    for layer in cache.layers:
        if type(layer) not in (DynamicLayer, LinearAttentionLayer):
            raise ValueError(f"Unsupported checkpoint layer: {type(layer).__name__}")
        for value in vars(layer).values():
            count(value)
    return sum(storages.values())


def _tensor_versions(model: PreTrainedModel) -> tuple[tuple[int, int, int], ...]:
    tensors = [*model.parameters(), *model.buffers()]
    return tuple((id(tensor), tensor._version, tensor.data_ptr()) for tensor in tensors)


def _is_eval_mode(model: PreTrainedModel) -> bool:
    return all(not module.training for module in model.modules())


def _validate_prefix_model(
    model: PreTrainedModel,
    versions: tuple[tuple[int, int, int], ...] | None,
    config: str | None,
) -> None:
    if versions is not None and (
        _tensor_versions(model) != versions or model.config.to_json_string() != config
    ):
        raise ValueError("Model weights or configuration changed after initial prefill")


class LiveSession:
    """Own a live cache; sampled stop/EOS tokens are pending until the next feed."""

    def __init__(self, model: PreTrainedModel, prompt_ids: Sequence[int]):
        if not _is_eval_mode(model):
            raise ValueError("LiveSession requires a model in eval mode")
        if model.generation_config is None:
            raise ValueError("LiveSession requires a causal language model")
        self.model = model
        self.cache: Cache | None = None
        self.logits: torch.Tensor
        self._tokens = list(prompt_ids)
        self._pending: int | None = None
        self._prefix_tensor_versions: tuple[tuple[int, int, int], ...] | None = None
        self._prefix_config: str | None = None
        eos = model.generation_config.eos_token_id
        self._eos = (
            set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
        )
        self._forward(self._tokens)
        # Other live-model paths remain usable, but cannot supply CPU checkpoints.
        tensors = [*model.parameters(), *model.buffers()]
        if model.device.type == "cpu" and all(
            not tensor.is_inference() for tensor in tensors
        ):
            self._prefix_tensor_versions = _tensor_versions(model)
            self._prefix_config = model.config.to_json_string()

    @property
    def tokens(self) -> tuple[int, ...]:
        return tuple(self._tokens)

    def _check_prefix_model(self) -> None:
        _validate_prefix_model(
            self.model, self._prefix_tensor_versions, self._prefix_config
        )

    @torch.no_grad()
    def _forward(self, ids: Sequence[int]) -> None:
        if not ids:
            raise ValueError("Cannot prefill an empty token sequence")
        if not _is_eval_mode(self.model):
            raise ValueError("LiveSession requires a model in eval mode")
        self._check_prefix_model()
        output = self.model(
            input_ids=torch.tensor(
                [list(ids)], dtype=torch.long, device=self.model.device
            ),
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        if not isinstance(output, CausalLMOutputWithPast) or not isinstance(
            output.past_key_values, Cache
        ):
            raise RuntimeError(
                "Model must return causal logits and a live Transformers Cache"
            )
        if output.logits is None:
            raise RuntimeError("Model returned no logits")
        self.cache = output.past_key_values
        self.logits = output.logits[:, -1, :]

    def _next_token(self) -> int:
        return int(self.logits.argmax(-1).item())

    def decode_until(
        self,
        max_new_tokens: int,
        on_token: Callable[[int], bool] | None = None,
    ) -> DecodeResult:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if not _is_eval_mode(self.model):
            raise ValueError("LiveSession requires a model in eval mode")
        self._check_prefix_model()
        if self._pending is not None:
            self._forward([self._pending])
            self._pending = None
        sampled = []
        for _ in range(max_new_tokens):
            token = self._next_token()
            sampled.append(token)
            self._tokens.append(token)
            self._pending = token
            if on_token is not None and on_token(token):
                return DecodeResult(tuple(sampled), "stop")
            if token in self._eos:
                return DecodeResult(tuple(sampled), "eos")
            self._forward([token])
            self._pending = None
        return DecodeResult(tuple(sampled), "length")

    def inject(self, response_ids: Sequence[int]) -> None:
        if not response_ids:
            raise ValueError("Tool response must not be empty")
        head = [] if self._pending is None else [self._pending]
        self._forward([*head, *response_ids])
        self._pending = None
        self._tokens.extend(response_ids)

    def checkpoint(
        self, max_cache_bytes: int = DEFAULT_CHECKPOINT_BYTES
    ) -> SessionCheckpoint:
        """Capture this exact boundary without flushing or replaying pending tokens."""
        if type(max_cache_bytes) is not int or max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be a positive integer")
        if self.cache is None:
            raise RuntimeError("Cannot checkpoint an absent cache")
        if not _is_eval_mode(self.model):
            raise ValueError("Checkpoints require a model in eval mode")
        size = _snapshot_bytes(self.cache, self.logits)
        if size > max_cache_bytes:
            raise ValueError(
                f"Checkpoint requires {size} bytes, exceeding {max_cache_bytes}"
            )
        if not isinstance(self.cache, DynamicCache):
            raise ValueError("Checkpoint requires DynamicCache")
        cached_tokens = len(self._tokens) - (self._pending is not None)
        if self.cache.get_seq_length() != cached_tokens:
            raise RuntimeError(
                "Cache length does not match the exact sampled-token boundary"
            )
        if self._prefix_tensor_versions is None or self._prefix_config is None:
            raise RuntimeError("Checkpoints require a version-tracked CPU model")
        self._check_prefix_model()
        return SessionCheckpoint(
            self.model,
            self.tokens,
            self._pending,
            size,
            copy.deepcopy(self.cache),
            self.logits.detach().clone(),
            self._prefix_tensor_versions,
            self._prefix_config,
            frozenset(self._eos),
        )

    @classmethod
    def from_checkpoint(
        cls,
        model: PreTrainedModel,
        checkpoint: SessionCheckpoint,
    ) -> LiveSession:
        """Fork an independent session; no prefill or recurrent-state truncation."""
        if model is not checkpoint.model or not _is_eval_mode(model):
            raise ValueError("Checkpoint requires the same unchanged eval-mode model")
        if (
            _tensor_versions(model) != checkpoint._tensor_versions
            or model.config.to_json_string() != checkpoint._config
        ):
            raise ValueError(
                "Model weights or configuration changed after checkpoint creation"
            )
        resumed = cls.__new__(cls)
        resumed.model = model
        resumed.cache = copy.deepcopy(checkpoint._cache)
        resumed.logits = checkpoint._logits.clone()
        resumed._tokens = list(checkpoint.tokens)
        resumed._pending = checkpoint.pending_token
        resumed._eos = set(checkpoint._eos)
        resumed._prefix_tensor_versions = checkpoint._tensor_versions
        resumed._prefix_config = checkpoint._config
        return resumed


class CheckpointStore:
    """Bounded process-local LRU store for exact-boundary hybrid checkpoints."""

    def __init__(
        self,
        max_total_bytes: int = 256 * 1024 * 1024,
        max_entries: int = 32,
    ) -> None:
        if type(max_total_bytes) is not int or max_total_bytes <= 0:
            raise ValueError("max_total_bytes must be a positive integer")
        if type(max_entries) is not int or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        self.max_total_bytes = max_total_bytes
        self.max_entries = max_entries
        self._entries: OrderedDict[tuple[int, ...], SessionCheckpoint] = OrderedDict()
        self._total_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def entries(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._total_bytes = 0

    def _evict_oldest(self) -> None:
        _, evicted = self._entries.popitem(last=False)
        self._total_bytes -= evicted.cache_bytes
        self.evictions += 1

    def put(self, checkpoint: SessionCheckpoint) -> None:
        if not isinstance(checkpoint, SessionCheckpoint):
            raise TypeError("CheckpointStore accepts only SessionCheckpoint instances")
        if checkpoint.cache_bytes > self.max_total_bytes:
            raise ValueError(
                f"Checkpoint requires {checkpoint.cache_bytes} bytes, "
                f"exceeding store budget {self.max_total_bytes}"
            )
        key = checkpoint.tokens
        existing = self._entries.pop(key, None)
        if existing is not None:
            self._total_bytes -= existing.cache_bytes
        while self._entries and (
            len(self._entries) >= self.max_entries
            or self._total_bytes + checkpoint.cache_bytes > self.max_total_bytes
        ):
            self._evict_oldest()
        self._entries[key] = checkpoint
        self._total_bytes += checkpoint.cache_bytes

    def get(self, tokens: Sequence[int]) -> SessionCheckpoint | None:
        key = tuple(tokens)
        checkpoint = self._entries.get(key)
        if checkpoint is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return checkpoint

    def match_prefix(
        self, tokens: Sequence[int]
    ) -> tuple[SessionCheckpoint, tuple[int, ...]] | None:
        """Return the longest stored prefix checkpoint and uncached suffix, if any."""
        target = tuple(tokens)
        if not target:
            self.misses += 1
            return None
        best_key: tuple[int, ...] | None = None
        for key in self._entries:
            if (
                len(key) <= len(target)
                and target[: len(key)] == key
                and (best_key is None or len(key) > len(best_key))
            ):
                best_key = key
        if best_key is None:
            self.misses += 1
            return None
        self._entries.move_to_end(best_key)
        self.hits += 1
        return self._entries[best_key], target[len(best_key) :]

    def resume_prefix(
        self, model: PreTrainedModel, tokens: Sequence[int]
    ) -> tuple[LiveSession, int]:
        """Restore from the longest prefix checkpoint or cold-prefill on miss.

        Returns `(session, prefilled_tokens)` where `prefilled_tokens` counts
        tokens forwarded through the model to reach `tokens`.
        """
        target = tuple(tokens)
        matched = self.match_prefix(target)
        if matched is None:
            return LiveSession(model, target), len(target)
        checkpoint, suffix = matched
        session = LiveSession.from_checkpoint(model, checkpoint)
        if not suffix:
            return session, 0
        pending_tokens = 1 if checkpoint.pending_token is not None else 0
        session.inject(suffix)
        return session, len(suffix) + pending_tokens
