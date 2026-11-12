"""Single-sequence, greedy decode with live hybrid-cache continuation."""

from __future__ import annotations

import copy
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


class LiveSession:
    """Own a live cache; sampled stop/EOS tokens are pending until the next feed."""

    def __init__(self, model: PreTrainedModel, prompt_ids: Sequence[int]):
        if model.training:
            raise ValueError("LiveSession requires a model in eval mode")
        if model.generation_config is None:
            raise ValueError("LiveSession requires a causal language model")
        self.model = model
        self.cache: Cache | None = None
        self.logits: torch.Tensor
        self._tokens = list(prompt_ids)
        self._pending: int | None = None
        eos = model.generation_config.eos_token_id
        self._eos = (
            set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
        )
        self._forward(self._tokens)

    @property
    def tokens(self) -> tuple[int, ...]:
        return tuple(self._tokens)

    @torch.no_grad()
    def _forward(self, ids: Sequence[int]) -> None:
        if not ids:
            raise ValueError("Cannot prefill an empty token sequence")
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

    def checkpoint(self, max_cache_bytes: int = 64 * 1024 * 1024) -> SessionCheckpoint:
        """Capture this exact boundary without flushing or replaying pending tokens."""
        if type(max_cache_bytes) is not int or max_cache_bytes <= 0:
            raise ValueError("max_cache_bytes must be a positive integer")
        if self.cache is None:
            raise RuntimeError("Cannot checkpoint an absent cache")
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
        return SessionCheckpoint(
            self.model,
            self.tokens,
            self._pending,
            size,
            copy.deepcopy(self.cache),
            self.logits.detach().clone(),
            _tensor_versions(self.model),
            self.model.config.to_json_string(),
            frozenset(self._eos),
        )

    @classmethod
    def from_checkpoint(
        cls,
        model: PreTrainedModel,
        checkpoint: SessionCheckpoint,
    ) -> LiveSession:
        """Fork an independent session; no prefill or recurrent-state truncation."""
        if model is not checkpoint.model or model.training:
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
        return resumed
