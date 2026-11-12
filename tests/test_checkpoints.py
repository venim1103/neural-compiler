"""CPU hybrid-prefix correctness; no serving-engine or pretrained-model measurements."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers.cache_utils import LinearAttentionLayer, StaticCache

from neural_compiler.bridge.protocol import QwenToolProtocol, ToolCall
from neural_compiler.bridge.session import LiveSession
from neural_compiler.bridge import ExecutionResult


def assert_reference(session: LiveSession):
    with torch.no_grad():
        expected = session.model(
            input_ids=torch.tensor([session.tokens]), use_cache=True, logits_to_keep=1
        )
    torch.testing.assert_close(
        session.logits, expected.logits[:, -1, :], atol=1e-6, rtol=1e-5
    )
    assert session.cache is not None and session.cache.get_seq_length() == len(
        session.tokens
    )


@pytest.mark.parametrize("boundary", ["stop", "eos", "length"])
def test_checkpoint_preserves_pending_token_exactly_once(
    tokenizer, hybrid_model, monkeypatch, boundary
):
    session = LiveSession(hybrid_model, [1, 2, 3])
    token = tokenizer.eos_token_id if boundary == "eos" else 4
    monkeypatch.setattr(session, "_next_token", lambda: token)
    decoded = session.decode_until(1, (lambda _: True) if boundary == "stop" else None)
    assert decoded.reason == boundary
    checkpoint = session.checkpoint()
    assert checkpoint.pending_token == (None if boundary == "length" else token)
    resumed = LiveSession.from_checkpoint(hybrid_model, checkpoint)
    session.inject([5, 6])
    resumed.inject([5, 6])
    assert resumed.tokens == session.tokens == (1, 2, 3, token, 5, 6)
    assert resumed.cache is not session.cache
    torch.testing.assert_close(resumed.logits, session.logits, atol=0, rtol=0)
    assert_reference(resumed)


def test_two_tool_boundaries_resubmit_without_mutating_original_cache(
    tokenizer, hybrid_model, monkeypatch
):
    session = LiveSession(hybrid_model, [1, 2, 3])
    protocol = QwenToolProtocol(tokenizer)
    for code in ("print(42)", "print(6 * 7)"):
        text = f"<tool_call>\n<function=run>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>"
        scripted = iter(tokenizer.encode(text, add_special_tokens=False))
        monkeypatch.setattr(session, "_next_token", lambda: next(scripted))
        call = None

        def on_token(token):
            nonlocal call
            call = protocol.feed(token)
            return call is not None

        session.decode_until(200, on_token)
        assert call == ToolCall(code)
        checkpoint = session.checkpoint()
        resumed = LiveSession.from_checkpoint(hybrid_model, checkpoint)
        independent = LiveSession.from_checkpoint(hybrid_model, checkpoint)
        response = protocol.response_ids(call, ExecutionResult("ok", "42\n", "", 0))
        resumed.inject(response)
        session.inject(response)
        torch.testing.assert_close(resumed.logits, session.logits, atol=0, rtol=0)
        assert_reference(resumed)
        independent.inject([9, 10])
        assert_reference(independent)
        # Forking the same boundary after both branches advanced must still be pristine.
        again = LiveSession.from_checkpoint(hybrid_model, checkpoint)
        again.inject(response)
        torch.testing.assert_close(again.logits, resumed.logits, atol=0, rtol=0)
        assert_reference(again)
        session = resumed


def test_snapshot_clones_recurrent_conv_kv_and_logits(hybrid_model):
    session = LiveSession(hybrid_model, [1, 2, 3])
    checkpoint = session.checkpoint()
    resumed = LiveSession.from_checkpoint(hybrid_model, checkpoint)
    assert resumed.cache is not None and session.cache is not None
    for original, clone in zip(session.cache.layers, resumed.cache.layers):
        values = (
            ("conv_states", "recurrent_states")
            if isinstance(original, LinearAttentionLayer)
            else ("keys", "values")
        )
        for name in values:
            left, right = getattr(original, name), getattr(clone, name)
            if isinstance(left, (list, dict)):
                left, right = left[0], right[0]
            assert left.data_ptr() != right.data_ptr()
            torch.testing.assert_close(left, right, atol=0, rtol=0)
    assert resumed.logits.data_ptr() != session.logits.data_ptr()
    tensors = [session.logits]
    for layer in session.cache.layers:
        if isinstance(layer, LinearAttentionLayer):
            tensors.extend([layer.conv_states[0], layer.recurrent_states[0]])
        else:
            tensors.extend([layer.keys, layer.values])
    assert checkpoint.cache_bytes == sum(
        tensor.untyped_storage().nbytes() for tensor in tensors
    )
    with pytest.raises(ValueError, match="exceeding"):
        session.checkpoint(max_cache_bytes=checkpoint.cache_bytes - 1)


def test_checkpoint_rejects_resource_overflow_and_unsupported_cache(hybrid_model):
    session = LiveSession(hybrid_model, [1, 2, 3])
    with pytest.raises(ValueError, match="exceeding"):
        session.checkpoint(max_cache_bytes=1)
    session.cache = StaticCache(config=hybrid_model.config, max_cache_len=8)
    with pytest.raises(ValueError, match="DynamicCache"):
        session.checkpoint()


def test_checkpoint_rejects_changed_weights_and_train_mode(hybrid_model):
    session = LiveSession(hybrid_model, [1, 2, 3])
    checkpoint = session.checkpoint()
    hybrid_model.train()
    with pytest.raises(ValueError, match="eval-mode"):
        LiveSession.from_checkpoint(hybrid_model, checkpoint)
    hybrid_model.eval()
    with torch.no_grad():
        next(hybrid_model.parameters()).add_(0.1)
    with pytest.raises(ValueError, match="weights"):
        LiveSession.from_checkpoint(hybrid_model, checkpoint)


@pytest.mark.parametrize("operation", ["create", "capture", "restore"])
def test_sessions_reject_partially_training_model(hybrid_model, operation):
    session = LiveSession(hybrid_model, [1, 2, 3])
    checkpoint = session.checkpoint()
    next(hybrid_model.children()).train()
    assert not hybrid_model.training
    with pytest.raises(ValueError, match="eval"):
        if operation == "create":
            LiveSession(hybrid_model, [1, 2, 3])
        elif operation == "capture":
            session.checkpoint()
        else:
            LiveSession.from_checkpoint(hybrid_model, checkpoint)
    if operation == "capture":
        with pytest.raises(ValueError, match="eval"):
            session.decode_until(1)
        with pytest.raises(ValueError, match="eval"):
            session.inject([4])
        assert session.tokens == (1, 2, 3)


def test_checkpoint_rejects_changed_configuration_and_wrong_model(hybrid_model):
    session = LiveSession(hybrid_model, [1, 2, 3])
    checkpoint = session.checkpoint()
    with pytest.raises(ValueError, match="same unchanged"):
        LiveSession.from_checkpoint(copy.deepcopy(hybrid_model), checkpoint)
    hybrid_model.config.rope_parameters["rope_theta"] = 20000
    with pytest.raises(ValueError, match="configuration"):
        LiveSession.from_checkpoint(hybrid_model, checkpoint)


def test_checkpoint_can_resume_decoding_without_a_response(hybrid_model, monkeypatch):
    session = LiveSession(hybrid_model, [1, 2, 3])
    monkeypatch.setattr(session, "_next_token", lambda: 4)
    session.decode_until(1, lambda _: True)
    checkpoint = session.checkpoint()
    resumed = LiveSession.from_checkpoint(hybrid_model, checkpoint)
    monkeypatch.setattr(resumed, "_next_token", lambda: 5)
    resumed.decode_until(1)
    assert resumed.tokens == (1, 2, 3, 4, 5)
    assert_reference(resumed)
