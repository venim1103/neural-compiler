"""CPU hybrid-prefix correctness; no serving-engine or pretrained-model measurements."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers.cache_utils import LinearAttentionLayer, StaticCache

from neural_compiler.bridge.protocol import QwenToolProtocol, ToolCall
from neural_compiler.bridge.session import CheckpointStore, LiveSession
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


@pytest.mark.parametrize("changed", ["weights", "configuration"])
def test_checkpoint_rejects_model_changes_since_initial_prefill(hybrid_model, changed):
    session = LiveSession(hybrid_model, [1, 2, 3])
    if changed == "weights":
        with torch.no_grad():
            next(hybrid_model.parameters()).add_(0.1)
    else:
        hybrid_model.config.rope_parameters["rope_theta"] = 20000
    with pytest.raises(ValueError, match="weights|configuration"):
        session.checkpoint()
    with pytest.raises(ValueError, match="weights|configuration"):
        session.decode_until(1)
    with pytest.raises(ValueError, match="weights|configuration"):
        session.inject([4])
    assert session.tokens == (1, 2, 3)


def test_unversioned_model_remains_usable_without_checkpoints(hybrid_model):
    with torch.inference_mode():
        model = type(hybrid_model)(hybrid_model.config)
    model.eval()
    session = LiveSession(model, [1, 2, 3])
    with pytest.raises(RuntimeError, match="version"):
        session.checkpoint()
    session.inject([4])
    assert_reference(session)


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


def test_checkpoint_store_enforces_lru_byte_and_entry_budgets(hybrid_model):
    s1 = LiveSession(hybrid_model, [1, 2, 3])
    cp1 = s1.checkpoint()
    s2 = LiveSession(hybrid_model, [1, 2, 3, 4])
    cp2 = s2.checkpoint()
    s3 = LiveSession(hybrid_model, [1, 2, 3, 4, 5])
    cp3 = s3.checkpoint()

    # Budget fits (cp1 + cp2) or (cp1 + cp3), not all three -> inserting cp3 evicts LRU cp2.
    store = CheckpointStore(max_total_bytes=cp1.cache_bytes + cp3.cache_bytes, max_entries=8)
    store.put(cp1)
    store.put(cp2)
    assert store.entries == 2 and store.total_bytes == cp1.cache_bytes + cp2.cache_bytes
    # Touch cp1 so cp2 becomes the oldest entry.
    assert store.get([1, 2, 3]) is cp1
    assert store.hits == 1 and store.misses == 0
    store.put(cp3)
    assert store.evictions == 1
    assert store.entries == 2
    assert store.total_bytes == cp1.cache_bytes + cp3.cache_bytes
    assert store.get([1, 2, 3]) is cp1
    assert store.get([1, 2, 3, 4]) is None
    assert store.get([1, 2, 3, 4, 5]) is cp3

    # Entry cap also triggers eviction even when byte budget has room.
    capped = CheckpointStore(max_total_bytes=10 * 1024 * 1024, max_entries=1)
    capped.put(cp1)
    capped.put(cp2)
    assert capped.entries == 1 and capped.evictions == 1
    assert capped.get([1, 2, 3]) is None
    assert capped.get([1, 2, 3, 4]) is cp2


def test_checkpoint_store_rejects_oversized_or_invalid_inputs(hybrid_model):
    cp = LiveSession(hybrid_model, [1, 2, 3]).checkpoint()
    with pytest.raises(ValueError, match="max_total_bytes"):
        CheckpointStore(max_total_bytes=0)
    with pytest.raises(ValueError, match="max_entries"):
        CheckpointStore(max_entries=False)  # type: ignore[arg-type]
    store = CheckpointStore(max_total_bytes=cp.cache_bytes - 1)
    with pytest.raises(ValueError, match="exceeding store budget"):
        store.put(cp)
    with pytest.raises(TypeError, match="SessionCheckpoint"):
        store.put("not-a-checkpoint")  # type: ignore[arg-type]


def test_checkpoint_store_prefix_resumption_preserves_pending_boundary(
    hybrid_model, monkeypatch
):
    session = LiveSession(hybrid_model, [1, 2, 3])
    monkeypatch.setattr(session, "_next_token", lambda: 4)
    decoded = session.decode_until(1, lambda _: True)
    assert decoded.reason == "stop"
    cp = session.checkpoint()
    assert cp.pending_token == 4

    store = CheckpointStore(max_total_bytes=1024 * 1024, max_entries=4)
    store.put(cp)

    # Exact match restores boundary with 0 forwarded tokens.
    exact, prefilled_exact = store.resume_prefix(hybrid_model, [1, 2, 3, 4])
    assert prefilled_exact == 0
    assert exact.tokens == (1, 2, 3, 4)

    # Longer continuation injects suffix [5, 6] plus pending token 4 (3 tokens forwarded).
    resumed, prefilled_suffix = store.resume_prefix(hybrid_model, [1, 2, 3, 4, 5, 6])
    assert prefilled_suffix == 3
    assert resumed.tokens == (1, 2, 3, 4, 5, 6)
    assert_reference(resumed)

    # Cache miss cold-prefills the entire sequence.
    cold, prefilled_cold = store.resume_prefix(hybrid_model, [9, 8, 7])
    assert prefilled_cold == 3
    assert cold.tokens == (9, 8, 7)
    assert_reference(cold)
