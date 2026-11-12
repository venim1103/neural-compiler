"""Single-sequence, greedy decode with live hybrid-cache continuation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass(frozen=True)
class DecodeResult:
    token_ids: tuple[int, ...]
    reason: Literal["stop", "eos", "length"]


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
        self._eos = set(eos if isinstance(eos, list) else [eos]) if eos is not None else set()
        self._forward(self._tokens)

    @property
    def tokens(self) -> tuple[int, ...]:
        return tuple(self._tokens)

    @torch.no_grad()
    def _forward(self, ids: Sequence[int]) -> None:
        if not ids:
            raise ValueError("Cannot prefill an empty token sequence")
        output = self.model(
            input_ids=torch.tensor([list(ids)], dtype=torch.long, device=self.model.device),
            past_key_values=self.cache, use_cache=True, logits_to_keep=1,
        )
        if not isinstance(output, CausalLMOutputWithPast) or not isinstance(output.past_key_values, Cache):
            raise RuntimeError("Model must return causal logits and a live Transformers Cache")
        if output.logits is None:
            raise RuntimeError("Model returned no logits")
        self.cache = output.past_key_values
        self.logits = output.logits[:, -1, :]

    def _next_token(self) -> int:
        return int(self.logits.argmax(-1).item())

    def decode_until(
        self, max_new_tokens: int, on_token: Callable[[int], bool] | None = None,
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
