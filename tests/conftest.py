"""Shared offline tokenizer and tiny hybrid model fixtures."""

from __future__ import annotations

import inspect

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3_5ForCausalLM, Qwen3_5TextConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

MARKERS = [
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<|im_start|>",
    "<|im_end|>",
]
TEMPLATE = (
    "{% for m in messages %}"
    "{% if m.role == 'tool' %}<|im_start|>user\n<tool_response>\n{{ m.content }}\n</tool_response>"
    "{% else %}<|im_start|>{{ m.role }}\n"
    "{% if m.role == 'assistant' %}<think>\n\n</think>\n\n{% endif %}"
    "{% if m.tool_calls %}<tool_call>\n<function=run>\n<parameter=code>\n"
    "{{ m.tool_calls[0].function.arguments.code }}\n</parameter>\n</function>\n</tool_call>"
    "{% else %}{{ m.content }}{% endif %}{% endif %}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
)


@pytest.fixture
def tokenizer():
    vocabulary = {
        char: index
        for index, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    backend = Tokenizer(models.BPE(vocab=vocabulary, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(
        add_prefix_space=False, use_regex=False
    )
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<|im_end|>"
    )
    tokenizer.add_special_tokens({"additional_special_tokens": MARKERS})
    tokenizer.chat_template = TEMPLATE
    return tokenizer


@pytest.fixture
def hybrid_model(tokenizer, monkeypatch):
    # CUDA-profile FLA decorators bind Triton even on CPU; use the same model's
    # Torch reference kernels for these deterministic offline cache checks.
    for name in ("torch_chunk_gated_delta_rule", "torch_recurrent_gated_delta_rule"):
        monkeypatch.setattr(qwen, name, inspect.unwrap(getattr(qwen, name)))
    threads = torch.get_num_threads()
    torch.set_num_threads(2)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        config = Qwen3_5TextConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            layer_types=["linear_attention", "full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
            },
            eos_token_id=tokenizer.eos_token_id,
        )
        config._attn_implementation = "eager"
        model = Qwen3_5ForCausalLM(config).eval()
    try:
        yield model
    finally:
        torch.set_num_threads(threads)
