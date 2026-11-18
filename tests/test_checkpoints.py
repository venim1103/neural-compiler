"""CPU hybrid-prefix correctness; no serving-engine or pretrained-model measurements."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers.cache_utils import (
    DynamicCache,
    DynamicLayer,
    LinearAttentionLayer,
    StaticCache,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from neural_compiler.bridge.protocol import QwenToolProtocol, ToolCall
from neural_compiler.bridge.session import (
    CheckpointBudgetError,
    CheckpointStore,
    LiveSession,
)
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


@pytest.mark.parametrize(
    "budget,callback,error",
    [
        (1.5, None, "positive integer"),
        (True, None, "positive integer"),
        (1, False, "on_token must be callable"),
        (1, 42, "on_token must be callable"),
    ],
)
def test_invalid_decode_request_preserves_pending_boundary(
    hybrid_model, monkeypatch, budget, callback, error
):
    session = LiveSession(hybrid_model, [1, 2, 3])
    monkeypatch.setattr(session, "_next_token", lambda: 4)
    session.decode_until(1, lambda _: True)
    tokens = session.tokens
    logits = session.logits.clone()
    monkeypatch.setattr(
        session, "_forward", lambda _: pytest.fail("Invalid request forwarded tokens")
    )
    with pytest.raises(ValueError, match=error):
        session.decode_until(budget, callback)
    assert session.tokens == tokens
    assert session._pending == 4
    torch.testing.assert_close(session.logits, logits, atol=0, rtol=0)


@pytest.mark.parametrize("sampled", [1.5, True, -1, "overflow"])
@pytest.mark.parametrize("pending", [False, True])
def test_invalid_sample_never_enters_history_or_callback(
    hybrid_model, monkeypatch, sampled, pending
):
    session = LiveSession(hybrid_model, [1, 2, 3])
    if sampled == "overflow":
        sampled = session.logits.shape[-1]
    if pending:
        monkeypatch.setattr(session, "_next_token", lambda: 4)
        session.decode_until(1, lambda _: True)
    prefix = session.tokens
    logits = session.logits.clone()
    callbacks = []
    monkeypatch.setattr(session, "_next_token", lambda: sampled)

    def stop(token):
        callbacks.append(token)
        return True

    expected = (
        "Token IDs must be integers"
        if type(sampled) is not int
        else "outside the model logit vocabulary"
    )
    with pytest.raises(ValueError, match=expected):
        session.decode_until(1, stop)
    assert callbacks == []
    assert session.tokens == prefix and session._pending is None
    if not pending:
        torch.testing.assert_close(session.logits, logits, atol=0, rtol=0)
    assert_reference(session)
    monkeypatch.setattr(session, "_next_token", lambda: 5)
    decoded = session.decode_until(1, lambda _: True)
    assert decoded.token_ids == (5,) and decoded.reason == "stop"
    session.inject([6, 7])
    assert session.tokens == (*prefix, 5, 6, 7)
    assert_reference(session)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("pending", [False, True])
def test_nonfinite_greedy_selection_never_enters_history(
    hybrid_model, monkeypatch, value, pending
):
    session = LiveSession(hybrid_model, [1, 2, 3])
    if pending:
        with monkeypatch.context() as patch:
            patch.setattr(session, "_next_token", lambda: 4)
            session.decode_until(1, lambda _: True)
    prefix = session.tokens
    original_forward = hybrid_model.forward

    def corrupt(logits):
        logits.fill_(-float("inf"))
        logits[..., 4] = value

    def invalid_logits(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        corrupt(output.logits)
        return output

    corrupt(session.logits)
    callbacks = []
    with monkeypatch.context() as patch:
        patch.setattr(hybrid_model, "forward", invalid_logits)
        with pytest.raises(RuntimeError, match="finite maximum logit"):
            session.decode_until(1, lambda token: callbacks.append(token) or True)
    assert callbacks == [] and session.tokens == prefix
    assert session._pending is None
    assert session.cache.get_seq_length() == len(prefix)
    session.inject([5, 6])
    assert session.tokens == (*prefix, 5, 6)
    assert_reference(session)


def test_greedy_selection_preserves_masked_logits_and_first_tie(hybrid_model):
    session = LiveSession(hybrid_model, [1, 2, 3])
    session.logits.fill_(-float("inf"))
    session.logits[..., 4:6] = 2
    decoded = session.decode_until(1, lambda _: True)
    assert decoded.token_ids == (4,) and decoded.reason == "stop"
    session.inject([6])
    assert session.tokens == (1, 2, 3, 4, 6)
    assert_reference(session)


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


def test_fractional_prompt_ids_cannot_be_silently_truncated(hybrid_model):
    with pytest.raises(ValueError, match="Token IDs must be integers"):
        LiveSession(hybrid_model, [1.5, 2, 3])


@pytest.mark.parametrize(
    "eos",
    [1.0, True, [1, True], (1, 2)],
    ids=["float", "boolean", "mixed-boolean-list", "tuple"],
)
def test_eos_ids_are_validated_before_model_prefill(hybrid_model, monkeypatch, eos):
    hybrid_model.generation_config.eos_token_id = eos
    monkeypatch.setattr(
        hybrid_model,
        "forward",
        lambda *args, **kwargs: pytest.fail("Invalid EOS configuration prefilled"),
    )
    with pytest.raises(ValueError, match="Token IDs must be integers"):
        LiveSession(hybrid_model, [1, 2, 3])


@pytest.mark.parametrize(
    "eos,reason", [(None, "length"), ([], "length"), ([4, 5, 4], "eos")]
)
def test_valid_eos_configuration_preserves_pending_boundary(
    hybrid_model, monkeypatch, eos, reason
):
    hybrid_model.generation_config.eos_token_id = eos
    session = LiveSession(hybrid_model, [1, 2, 3])
    monkeypatch.setattr(session, "_next_token", lambda: 4)
    result = session.decode_until(1)
    assert result.token_ids == (4,) and result.reason == reason
    assert session.tokens == (1, 2, 3, 4)
    assert session._pending == (4 if reason == "eos" else None)
    assert session.cache is not None
    assert session.cache.get_seq_length() == (3 if reason == "eos" else 4)


def test_invalid_response_ids_preserve_pending_cache_boundary(
    hybrid_model, monkeypatch
):
    session = LiveSession(hybrid_model, [1, 2, 3])
    monkeypatch.setattr(session, "_next_token", lambda: 4)
    session.decode_until(1, lambda _: True)
    checkpoint = session.checkpoint()
    with pytest.raises(ValueError, match="Token IDs must be integers"):
        session.inject([5.5])
    assert session.tokens == checkpoint.tokens
    assert session._pending == checkpoint.pending_token
    torch.testing.assert_close(session.logits, checkpoint._logits, atol=0, rtol=0)
    assert session.cache.get_seq_length() == len(session.tokens) - 1
    session.inject([5, 6])
    assert_reference(session)


@pytest.mark.parametrize("pending", [False, True])
def test_input_materialization_failure_preserves_healthy_cache(
    hybrid_model, monkeypatch, pending
):
    session = LiveSession(hybrid_model, [1, 2, 3])
    if pending:
        monkeypatch.setattr(session, "_next_token", lambda: 4)
        session.decode_until(1, lambda _: True)
    checkpoint = session.checkpoint()
    cache = session.cache
    with monkeypatch.context() as patch:
        patch.setattr(
            hybrid_model,
            "forward",
            lambda *args, **kwargs: pytest.fail("Unrepresentable input ran model"),
        )
        with pytest.raises(ValueError, match="Overflow when unpacking long long"):
            session.inject([2**64])
    assert session.cache is cache
    assert session.tokens == checkpoint.tokens
    assert session._pending == checkpoint.pending_token
    torch.testing.assert_close(session.logits, checkpoint._logits, atol=0, rtol=0)
    retained = session.checkpoint()
    assert retained.tokens == checkpoint.tokens
    assert retained.pending_token == checkpoint.pending_token
    session.inject([5, 6])
    assert session.tokens == (*checkpoint.tokens, 5, 6)
    assert_reference(session)


def test_initial_cache_must_cover_the_prompt(hybrid_model, monkeypatch):
    original_forward = hybrid_model.forward

    def incomplete_cache(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        output.past_key_values = DynamicCache(config=hybrid_model.config)
        return output

    monkeypatch.setattr(hybrid_model, "forward", incomplete_cache)
    with pytest.raises(RuntimeError, match="cache length"):
        LiveSession(hybrid_model, [1, 2, 3])


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("failure", ["exception", "invalid_output", "cache_length"])
def test_failed_forward_cannot_reuse_or_checkpoint_mutated_cache(
    hybrid_model, monkeypatch, restored, failure
):
    session = LiveSession(hybrid_model, [1, 2, 3])
    checkpoint = session.checkpoint()
    if restored:
        session = LiveSession.from_checkpoint(hybrid_model, checkpoint)
    layer = next(
        layer
        for layer in session.cache.layers
        if isinstance(layer, LinearAttentionLayer)
    )
    state = layer.recurrent_states[0]
    assert state is not None
    before = state.clone()
    original_forward = hybrid_model.forward

    def interrupted(*args, **kwargs):
        assert kwargs["past_key_values"] is session.cache
        state.add_(1)
        if failure == "invalid_output":
            return None
        if failure == "cache_length":
            return CausalLMOutputWithPast(
                logits=session.logits.unsqueeze(1), past_key_values=session.cache
            )
        raise RuntimeError("Fixture interrupted a recurrent-state update")

    monkeypatch.setattr(hybrid_model, "forward", interrupted)
    expected = {
        "exception": "Fixture interrupted",
        "invalid_output": "Model must return",
        "cache_length": "cache length",
    }[failure]
    with pytest.raises(RuntimeError, match=expected):
        session.inject([4])
    assert session.tokens == checkpoint.tokens
    assert session.cache.get_seq_length() == len(session.tokens)
    torch.testing.assert_close(state, before + 1, atol=0, rtol=0)
    monkeypatch.setattr(
        hybrid_model,
        "forward",
        lambda *args, **kwargs: pytest.fail("Damaged session called the model"),
    )
    for operation in (
        session.checkpoint,
        lambda: session.decode_until(1),
        lambda: session.inject([5]),
    ):
        with pytest.raises(RuntimeError, match="unusable after a failed forward"):
            operation()
    monkeypatch.setattr(hybrid_model, "forward", original_forward)
    recovered = LiveSession.from_checkpoint(hybrid_model, checkpoint)
    recovered.inject([5, 6])
    assert recovered.tokens == (1, 2, 3, 5, 6)
    assert_reference(recovered)


@pytest.mark.parametrize("shape", ["batch", "rank", "tokens", "vocabulary"])
def test_invalid_logit_shape_cannot_commit_a_healthy_boundary(
    hybrid_model, monkeypatch, shape
):
    session = LiveSession(hybrid_model, [1, 2, 3])
    checkpoint = session.checkpoint()
    original_forward = hybrid_model.forward

    def invalid_shape(*args, **kwargs):
        output = original_forward(*args, **kwargs)
        assert output.logits is not None
        if shape == "batch":
            output.logits = output.logits.expand(2, -1, -1)
        elif shape == "rank":
            output.logits = output.logits.unsqueeze(1)
        elif shape == "tokens":
            output.logits = output.logits[:, :0, :]
        else:
            output.logits = output.logits[:, :, :0]
        return output

    monkeypatch.setattr(hybrid_model, "forward", invalid_shape)
    with pytest.raises(RuntimeError, match="single-sequence"):
        session.inject([4])
    assert session.tokens == checkpoint.tokens
    for operation in (
        session.checkpoint,
        lambda: session.decode_until(1),
        lambda: session.inject([5]),
    ):
        with pytest.raises(RuntimeError, match="unusable after a failed forward"):
            operation()
    monkeypatch.setattr(hybrid_model, "forward", original_forward)
    recovered = LiveSession.from_checkpoint(hybrid_model, checkpoint)
    recovered.inject([5, 6])
    assert_reference(recovered)


def test_invalid_prefix_ids_cannot_alias_a_cached_integer_prompt(hybrid_model):
    store = CheckpointStore()
    checkpoint = LiveSession(hybrid_model, [1, 2, 3]).checkpoint()
    store.put(checkpoint)
    with pytest.raises(ValueError, match="Token IDs must be integers"):
        store.resume_prefix(hybrid_model, [True, 2, 3])
    assert store.hits == store.misses == 0
    assert store.entries == 1
    assert store.total_bytes == checkpoint.cache_bytes


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


@pytest.mark.parametrize("view", ["cache_alias", "broadcast", "strided"])
def test_snapshot_budget_matches_independently_cloned_logit_storage(
    hybrid_model, monkeypatch, view
):
    session = LiveSession(hybrid_model, [1, 2, 3])
    assert session.cache is not None
    vocabulary = session.logits.shape[-1]
    if view == "cache_alias":
        layer = next(
            layer for layer in session.cache.layers if isinstance(layer, DynamicLayer)
        )
        assert layer.keys is not None
        session.logits = layer.keys[0, 0, 0, :].unsqueeze(0)
    elif view == "broadcast":
        session.logits = torch.zeros(1, 1).expand(1, vocabulary)
    else:
        session.logits = torch.zeros(1, vocabulary * 2)[:, ::2]

    checkpoint = session.checkpoint()
    tensors = [checkpoint._logits]
    for layer in checkpoint._cache.layers:
        if isinstance(layer, LinearAttentionLayer):
            tensors.extend([layer.conv_states[0], layer.recurrent_states[0]])
        else:
            tensors.extend([layer.keys, layer.values])
    storages = {
        tensor.untyped_storage().data_ptr(): tensor.untyped_storage().nbytes()
        for tensor in tensors
    }
    actual_bytes = sum(storages.values())
    assert 0 < actual_bytes <= 1024 * 1024
    assert checkpoint.cache_bytes == actual_bytes

    with monkeypatch.context() as patch:
        patch.setattr(
            session.cache,
            "__deepcopy__",
            lambda _: pytest.fail("Over-budget request cloned cache storage"),
            raising=False,
        )
        with pytest.raises(CheckpointBudgetError, match="exceeding"):
            session.checkpoint(max_cache_bytes=actual_bytes - 1)
    assert session.checkpoint(max_cache_bytes=actual_bytes).cache_bytes == actual_bytes
    store = CheckpointStore(max_total_bytes=actual_bytes - 1)
    with pytest.raises(CheckpointBudgetError, match="store budget"):
        store.put(checkpoint)
    assert store.entries == store.total_bytes == 0


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


@pytest.mark.parametrize("operation", ["capture", "restore"])
def test_checkpoint_rejects_changed_eos_settings(hybrid_model, operation):
    hybrid_model.generation_config.eos_token_id = 5
    session = LiveSession(hybrid_model, [1, 2, 3])
    checkpoint = session.checkpoint()
    hybrid_model.generation_config.eos_token_id = [4, 5]
    with pytest.raises(ValueError, match="configuration"):
        if operation == "capture":
            session.checkpoint()
        else:
            LiveSession.from_checkpoint(hybrid_model, checkpoint)
    assert session.tokens == (1, 2, 3)


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


@pytest.mark.parametrize("changed", ["configuration", "eos"])
@pytest.mark.parametrize("operation", ["decode", "inject"])
def test_unversioned_live_configuration_changes_preserve_pending_boundary(
    hybrid_model, monkeypatch, changed, operation
):
    with torch.inference_mode():
        model = type(hybrid_model)(copy.deepcopy(hybrid_model.config))
    model.eval()
    session = LiveSession(model, [1, 2, 3])
    assert session._prefix_tensor_versions is None
    monkeypatch.setattr(session, "_next_token", lambda: 4)
    session.decode_until(1, lambda _: True)
    tokens = session.tokens
    logits = session.logits.clone()
    if changed == "configuration":
        model.config.rope_parameters["rope_theta"] = 20000
    else:
        model.generation_config.eos_token_id = [4, 5]
    monkeypatch.setattr(
        model,
        "forward",
        lambda *args, **kwargs: pytest.fail("Changed configuration called the model"),
    )
    with pytest.raises(ValueError, match="configuration"):
        if operation == "decode":
            session.decode_until(1, lambda _: True)
        else:
            session.inject([5, 6])
    assert session.tokens == tokens and session._pending == 4
    assert session._cache_usable and session.cache.get_seq_length() == 3
    torch.testing.assert_close(session.logits, logits, atol=0, rtol=0)


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
    store = CheckpointStore(
        max_total_bytes=cp1.cache_bytes + cp3.cache_bytes, max_entries=8
    )
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
