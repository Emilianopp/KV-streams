# SPDX-License-Identifier: Apache-2.0
"""Interceptor tests for the Markovian Summary Branch-A extension in
``kv_eviction.env._install_message_padding_interceptor``.

Uses a stub ``orig_create`` (AsyncMock) plus a stub tokenizer to
exercise both summary modes (``markovian``, ``eviction``), the
recursion guard, re-fire prevention, error paths, and stats counters
without touching vLLM or a real tokenizer.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import kv_eviction.env as env
import pytest
from kv_eviction.types import compute_turn_compaction_state_id

INSTR = "Please summarize everything important for the task."


class _StubTok:
    """Minimal tokenizer stub: apply_chat_template returns a stable
    marker string, encode returns a deterministic token list. The
    actual values don't matter for interceptor logic — we just need
    the re-tokenize step to succeed."""

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=False
    ):
        return f"TOK:{len(messages)}"

    def encode(self, s, add_special_tokens=False):
        return [1, 2, 3]


def _make_summary_response(
    text="SUMMARY",
    *,
    prompt_ids=None,
    completion_ids=None,
    logprobs=None,
):
    """Build a ChatCompletion-ish SimpleNamespace that
    ``_generate_summary`` can extract a summary + train-sample payload
    from. ``logprobs`` is a list of floats; ``None`` means no logprobs
    at all."""
    msg = SimpleNamespace(content=text)
    lp = None
    if logprobs is not None:
        lp = SimpleNamespace(
            content=[SimpleNamespace(token=f"t{i}", logprob=x) for i, x in enumerate(logprobs)]
        )
    choice = SimpleNamespace(message=msg, token_ids=completion_ids, logprobs=lp)
    return SimpleNamespace(
        choices=[choice],
        prompt_token_ids=prompt_ids,
    )


def _outer_response():
    return SimpleNamespace(id="outer-0", prompt_token_ids=None)


def _sys(content="sys"):
    return {"role": "system", "content": content}


def _user(content="u"):
    return {"role": "user", "content": content}


def _asst(content="a"):
    return {"role": "assistant", "content": content}


def _turn(i):
    return [_user(f"u{i}"), _asst(f"a{i}")]


@pytest.fixture(autouse=True)
def reset_configs():
    """Clear all interceptor state between tests."""
    env._markovian_config = None
    env._summary_config = None
    env._padding_config = None
    env._SUMMARY_CACHE.set(None)
    env._LOGICAL_EVICTED_TOKENS.set(0)
    env._markovian_stats = {
        "n_truncations": 0,
        "n_messages_dropped": 0,
        "n_summaries": 0,
        "n_summary_failures": 0,
        "summary_prompt_tokens": 0,
        "summary_output_tokens": 0,
        "summary_latency_ms": 0,
    }
    yield
    env._markovian_config = None
    env._summary_config = None
    env._padding_config = None


def _install_mt(max_turns=8, stride=None):
    env.configure_markovian_thinker(
        enabled=True, tokenizer=_StubTok(), max_turns=max_turns, stride=stride
    )


def _install_summary(
    *,
    enabled=True,
    mode="markovian",
    compaction_max_turns=2,
    max_len_summary=128,
    on_error="drop",
    instruction_text=INSTR,
    resume_text="",
    temperature=0.3,
):
    env.configure_markovian_summary(
        enabled=enabled,
        mode=mode,
        compaction_max_turns=compaction_max_turns,
        max_len_summary=max_len_summary,
        instruction_text=instruction_text,
        resume_text=resume_text,
        temperature=temperature,
        top_p=0.95,
        on_error=on_error,
        log_summaries=False,
    )


def _run_with_fake_orig(fake_orig, run_coro_factory):
    """Reinstall the interceptor on top of ``fake_orig`` then run
    ``run_coro_factory()`` once through ``asyncio.run``. Returns
    whatever the coroutine returned."""
    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create

        async def run():
            return await run_coro_factory(patched)

        return asyncio.run(run())


def _is_summary_call(kwargs):
        return kwargs.get("max_tokens") == 128


# ─── Trigger / mode branching ───


def test_markovian_mode_full_reset_shape():
    _install_mt(max_turns=8, stride=8)  # evict the whole window == full reset
    _install_summary(mode="markovian", compaction_max_turns=2)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        if "logprobs" in kwargs and kwargs["logprobs"] is True:
            return _make_summary_response("SUMMARY-TEXT")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    # 2 calls: summary request, then outer rewritten request.
    assert len(calls) == 2
    summary_call, outer_call = calls
    assert summary_call["logprobs"] is True
    assert summary_call["max_tokens"] == 128
    # Summary messages = full history + instruction.
    assert summary_call["messages"][-1] == {"role": "user", "content": INSTR}
    assert summary_call["messages"][:-1] == msgs
    assert "tools" not in summary_call
    assert "tool_choice" not in summary_call

    # Outer call: markovian mode = sys + [I, S] + tail (body dropped).
    # The summary LEADS so it can be persisted as the new conversation
    # base; the tail is already the pending user turn.
    outer_messages = outer_call["messages"]
    assert outer_messages == [
        _sys(),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUMMARY-TEXT"},
        _user("pending"),
    ]


def test_markovian_mode_skips_resume_text_when_tail_is_pending():
    """resume_text is only needed when there is no in-flight observation.
    Here the tail is the pending user turn, so it already serves as the
    generation prompt and no resume message is appended."""
    _install_mt(max_turns=8, stride=8)  # evict the whole window == full reset
    _install_summary(
        mode="markovian",
        compaction_max_turns=2,
        resume_text="Please continue.",
    )

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        if "logprobs" in kwargs and kwargs["logprobs"] is True:
            return _make_summary_response("SUM")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    outer_messages = calls[1]["messages"]
    assert outer_messages == [
        _sys(),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUM"},
        _user("pending"),
    ]


def test_eviction_mode_append_only_shape():
    _install_mt(max_turns=8)
    _install_summary(mode="eviction", compaction_max_turns=2)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        if "logprobs" in kwargs and kwargs["logprobs"] is True:
            return _make_summary_response("S-EVICT")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    assert len(calls) == 2
    outer_messages = calls[1]["messages"]
    # Eviction mode keeps body groups intact: sys + body + I + S + tail.
    assert outer_messages == [
        _sys(),
        *_turn(1),
        *_turn(2),
        *_turn(3),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "S-EVICT"},
        _user("pending"),
    ]


def test_below_trigger_no_summary_fires():
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=4)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return _outer_response()

    # 2 real turns, below the 4-turn trigger.
    msgs = [_sys(), *_turn(1), *_turn(2), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    # Only one call (outer). No summary fired.
    assert len(calls) == 1
    # Message list unchanged from input (Markovian's plain-truncation
    # below max_turns=8 is a no-op).
    assert calls[0]["messages"] == msgs


def test_eviction_mode_refire_prevention_after_single_summary():
    """After the summary fires, the next step's message list contains
    a prior summary exchange + 1 new real turn. With the discount, the
    turn count stays ≤ max_turns so the trigger does NOT re-fire.
    Without the discount it would fire every single subsequent step."""
    _install_mt(max_turns=8)
    # Trigger is now `n_real >= compaction_max_turns` (it fires when the count
    # REACHES the threshold, matching truncation). To assert non-refire the
    # threshold must sit ABOVE the discounted count, so use 3 rather than 2.
    _install_summary(mode="eviction", compaction_max_turns=3)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return _outer_response()

    # 1 real turn + 1 summary exchange + 1 new real turn = 3 groups.
    # n_real = 3 - 1 = 2. 2 >= 3 is False → does NOT fire.
    msgs = [
        _sys(),
        *_turn(1),
        _user(INSTR),
        _asst("old-summary"),
        *_turn(2),
        _user("pending"),
    ]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    # Exactly one (outer) call — no summary re-fired.
    assert len(calls) == 1


def test_eviction_mode_refires_after_enough_new_real_turns():
    """After enough additional real turns beyond the prior summary, the
    trigger re-fires."""
    _install_mt(max_turns=8)
    _install_summary(mode="eviction", compaction_max_turns=2)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        if _is_summary_call(kwargs):
            return _make_summary_response("S2")
        return _outer_response()

    # 1 real turn + 1 summary + 3 new real turns = 5 groups.
    # n_real = 5 - 1 = 4 > 2 → re-fires.
    msgs = [
        _sys(),
        *_turn(1),
        _user(INSTR),
        _asst("s1"),
        *_turn(2),
        *_turn(3),
        *_turn(4),
        _user("pending"),
    ]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)
    assert len(calls) == 2


# ─── Recursion guard ───


def test_recursion_guard_bypasses_interceptor():
    """When _IN_SUMMARY_CALL is True at entry, patched_create must not
    touch messages at all — it just forwards to orig_create."""
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=2)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        token = env._IN_SUMMARY_CALL.set(True)
        try:
            return await patched(self=None, model="m", messages=msgs)
        finally:
            env._IN_SUMMARY_CALL.reset(token)

    _run_with_fake_orig(fake_orig, factory)

    # Only one call and the messages are the raw input — no retokenize
    # or rewrite happened.
    assert len(calls) == 1
    assert calls[0]["messages"] == msgs
    # No summary triggered either.
    assert env._markovian_stats["n_summaries"] == 0


# ─── Error paths ───


def test_on_error_drop_falls_back_to_plain_truncation():
    _install_mt(max_turns=2)
    _install_summary(mode="markovian", compaction_max_turns=2, on_error="drop")

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        if _is_summary_call(kwargs):
            raise RuntimeError("summary backend down")
        calls.append(dict(kwargs))
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    # One outer call (summary failed). Messages are plain-truncated
    # (max_turns=2 → keep last 2 groups + tail).
    assert len(calls) == 1
    outer = calls[0]["messages"]
    # No [I, S] injected.
    assert {"role": "user", "content": INSTR} not in outer
    # Should contain last 2 turns (u2/a2 and u3/a3) + pending.
    assert outer[-1] == _user("pending")
    assert env._markovian_stats["n_summary_failures"] == 1
    assert env._markovian_stats["n_summaries"] == 0


def test_on_error_raise_propagates():
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=2, on_error="raise")

    async def fake_orig(self, *args, **kwargs):
        if _is_summary_call(kwargs):
            raise RuntimeError("boom")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    with pytest.raises(RuntimeError, match="boom"):
        _run_with_fake_orig(fake_orig, factory)


def test_empty_summary_text_treated_as_failure():
    _install_mt(max_turns=2)
    _install_summary(mode="markovian", compaction_max_turns=2)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        if _is_summary_call(kwargs):
            return _make_summary_response("   ")  # whitespace-only
        calls.append(dict(kwargs))
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    # Falls back to plain truncation.
    assert len(calls) == 1
    outer = calls[0]["messages"]
    assert {"role": "user", "content": INSTR} not in outer
    assert env._markovian_stats["n_summary_failures"] == 1
    assert env._markovian_stats["n_summaries"] == 0


# ─── Stats counters ───


def test_stats_counters_increment_on_summary_success():
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=2)

    async def fake_orig(self, *args, **kwargs):
        if _is_summary_call(kwargs):
            return _make_summary_response("SUMMARY")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    assert env._markovian_stats["n_summaries"] == 1
    assert env._markovian_stats["n_summary_failures"] == 0
    # Summary path rewrote messages, so the truncation counter fires too.
    assert env._markovian_stats["n_truncations"] == 1


def test_stats_drain_and_reset():
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=2)

    async def fake_orig(self, *args, **kwargs):
        if _is_summary_call(kwargs):
            return _make_summary_response("SUMMARY")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)
    drained = env.pop_markovian_stats()
    assert drained["n_summaries"] == 1
    # Counters reset after drain.
    assert env._markovian_stats["n_summaries"] == 0
    assert env._markovian_stats["n_summary_failures"] == 0


# ─── Summary-call kwargs hygiene ───


def test_summary_trainsample_attached_to_outer_response():
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=2)

    async def fake_orig(self, *args, **kwargs):
        if _is_summary_call(kwargs):
            return _make_summary_response(
                "S",
                prompt_ids=[11, 12, 13],
                completion_ids=[21, 22],
                logprobs=[-0.1, -0.2],
            )
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    captured = {}

    async def factory(patched):
        r = await patched(self=None, model="test-model", messages=msgs)
        captured["response"] = r
        return r

    _run_with_fake_orig(fake_orig, factory)

    attached = captured["response"].summary_trainsample
    assert attached["prompt_token_ids"] == [11, 12, 13]
    assert attached["completion_token_ids"] == [21, 22]
    assert attached["completion_logprobs"] == pytest.approx([-0.1, -0.2])
    assert attached["completion_temperature"] == pytest.approx(0.3)
    assert attached["model"] == "test-model"
    assert attached == {
        "prompt_token_ids": [11, 12, 13],
        "completion_token_ids": [21, 22],
        "completion_logprobs": pytest.approx([-0.1, -0.2]),
        "model": "test-model",
        "compaction_events": [],
        "completion_temperature": pytest.approx(0.3),
    }


def test_no_summary_no_trainsample_attached():
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=8)  # below trigger

    async def fake_orig(self, *args, **kwargs):
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), _user("pending")]

    captured = {}

    async def factory(patched):
        r = await patched(self=None, model="m", messages=msgs)
        captured["response"] = r
        return r

    _run_with_fake_orig(fake_orig, factory)

    assert getattr(captured["response"], "summary_trainsample", None) is None


def test_summary_failure_does_not_attach_trainsample():
    _install_mt(max_turns=2)
    _install_summary(mode="markovian", compaction_max_turns=2, on_error="drop")

    async def fake_orig(self, *args, **kwargs):
        if _is_summary_call(kwargs):
            raise RuntimeError("summary down")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    captured = {}

    async def factory(patched):
        r = await patched(self=None, model="m", messages=msgs)
        captured["response"] = r
        return r

    _run_with_fake_orig(fake_orig, factory)

    assert getattr(captured["response"], "summary_trainsample", None) is None


def test_summary_call_matches_action_rendering_but_cannot_call_tools():
    """`tools` changes how the chat template renders the SYSTEM block, so the
    summary must carry the SAME tools as the action calls or its prompt
    diverges at token ~0 and every compaction is a full prefix-cache miss.
    It must not be able to actually CALL a tool (tool_choice="none"), and
    response_format / the outer extra_body must not leak (extra_body carries
    min_tokens etc. that would distort the summary generation)."""
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=2)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        if _is_summary_call(kwargs):
            return _make_summary_response("S")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]
    tools = [{"type": "function", "function": {"name": "foo"}}]

    async def factory(patched):
        return await patched(
            self=None,
            model="m",
            messages=msgs,
            tools=tools,
            tool_choice="auto",
            response_format={"type": "json_object"},
            extra_body={"something": True},
            extra_headers={"X-Session-ID": "rollout-7"},
        )

    _run_with_fake_orig(fake_orig, factory)

    # calls[0] = summary; calls[1] = outer
    summary_call = calls[0]
    assert summary_call["logprobs"] is True
    assert summary_call["top_logprobs"] == 0
    # Same template rendering + routing as the action call...
    assert summary_call["tools"] == tools
    assert summary_call["extra_headers"] == {"X-Session-ID": "rollout-7"}
    # ...but no tool calls, no leaked output constraints.
    assert summary_call["tool_choice"] == "none"
    assert "response_format" not in summary_call
    assert summary_call["extra_body"] == {"return_token_ids": True}


def test_summary_greedy_request_records_effective_training_temperature():
    from prime_rl.orchestrator.trajectories import _build_summary_sample
    from prime_rl.trainer.batch import prepare_sample

    _install_mt(max_turns=8)
    _install_summary(
        mode="markovian",
        compaction_max_turns=2,
        temperature=0.0,
    )
    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        if _is_summary_call(kwargs):
            return _make_summary_response(
                "S",
                prompt_ids=[1, 2],
                completion_ids=[3],
                logprobs=[-0.1],
            )
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    response = _run_with_fake_orig(fake_orig, factory)

    assert calls[0]["temperature"] == 0.0
    assert calls[0]["logprobs"] is True
    assert calls[0]["extra_body"]["return_token_ids"] is True
    assert response.summary_trainsample["completion_temperature"] == 1.0
    sample = _build_summary_sample(
        {"extras": {"summary_trainsample": response.summary_trainsample}},
        temperature=0.0,
        has_error=False,
    )
    assert sample is not None
    assert prepare_sample(sample, seq_len=8).temperatures == [1.0, 1.0, 1.0]


@pytest.mark.parametrize(
    "temperature",
    [-0.1, float("nan"), float("inf"), float("-inf")],
)
def test_summary_rejects_invalid_training_temperature(temperature):
    _install_mt(max_turns=8)

    with pytest.raises(ValueError, match="training temperature"):
        _install_summary(temperature=temperature)


# ─── _extract_summary_trainsample / attach_summary_trainsample_from_response ───


def test_extract_summary_trainsample_attribute():
    sample = {"prompt_token_ids": [1], "completion_token_ids": [2]}
    resp = SimpleNamespace(summary_trainsample=sample)
    assert env._extract_summary_trainsample(resp) == sample


def test_extract_summary_trainsample_none_response():
    assert env._extract_summary_trainsample(None) is None


def test_extract_summary_trainsample_absent_field():
    resp = SimpleNamespace()
    assert env._extract_summary_trainsample(resp) is None


def test_extract_summary_trainsample_non_dict_ignored():
    resp = SimpleNamespace(summary_trainsample="not a dict")
    assert env._extract_summary_trainsample(resp) is None


def test_attach_summary_trainsample_from_response_roundtrip():
    sample = {"prompt_token_ids": [1, 2], "completion_token_ids": [3]}
    resp = SimpleNamespace(summary_trainsample=sample)
    step: dict = {"extras": None}
    env.attach_summary_trainsample_from_response(step, resp)  # type: ignore[arg-type]
    assert step["extras"]["summary_trainsample"] == sample


def test_attach_summary_trainsample_noop_when_absent():
    resp = SimpleNamespace()
    step: dict = {"extras": None}
    env.attach_summary_trainsample_from_response(step, resp)  # type: ignore[arg-type]
    assert step["extras"] is None


# ─── Eviction-mode padding + compaction_events capture ───


def _install_padding(enabled=True, block_size=16, phase4_enabled=False):
    """Install a MessagePaddingConfig. Tokenizer is the stub from above;
    the real render_padded_prompt is patched per-test so its internals
    don't care about tokenizer behavior."""
    env.configure_message_padding(
        enabled=enabled,
        tokenizer=_StubTok(),
        block_size=block_size,
        filler_token_id=198,
        im_end_token_id=151645,
        phase4_enabled=phase4_enabled,
    )


def _prefill_trim_summary_event(
    submitted_prompt_ids,
    *,
    kept_token_ids=None,
):
    kept = (
        list(kept_token_ids)
        if kept_token_ids is not None
        else list(submitted_prompt_ids[:2]) + list(submitted_prompt_ids[4:])
    )
    return {
        "num_output_tokens_at_compaction": 0,
        "tokens_evicted": 2,
        "position_offset_after": 2,
        "num_prompt_tokens": len(kept),
        "evict_start": 2,
        "new_user_fragment_len": 1,
        "kept_indices": [0, 1, *range(4, len(submitted_prompt_ids))],
        "kept_token_ids": kept,
        "last_turn_evicted": 0,
        "num_turns_evicted_after": 1,
    }


def test_eviction_mode_padding_enabled_pads_summary_call():
    """When scfg.mode=='eviction' and _padding_config is enabled,
    _generate_summary should render the summary call's prompt via
    render_padded_prompt and forward the padded ids via
    extra_body.prompt_token_ids."""
    _install_mt(max_turns=8)
    _install_summary(mode="eviction", compaction_max_turns=2)
    _install_padding(enabled=True, block_size=16)

    calls: list[dict] = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        if _is_summary_call(kwargs):
            return _make_summary_response("S-PAD")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    padded_summary_ids = list(range(32))  # len=32, multiple of block_size=16
    padded_outer_ids = list(range(48))  # len=48, multiple of block_size=16

    call_count = {"n": 0}

    def fake_render(*, tokenizer, messages, tools, block_size, filler_token_id, im_end_token_id):
        call_count["n"] += 1
        # First call is inside _generate_summary (summary_messages), second
        # call is in Branch A for the outer rewritten messages.
        ids = padded_summary_ids if call_count["n"] == 1 else padded_outer_ids
        return f"rendered-{call_count['n']}", ids, []

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    with patch.object(env, "render_padded_prompt", fake_render):
        _run_with_fake_orig(fake_orig, factory)

    # render_padded_prompt was called twice: once for the summary call,
    # once for the outer rewrite.
    assert call_count["n"] == 2

    # Summary call (calls[0]) carries padded prompt_token_ids via extra_body.
    summary_call = calls[0]
    assert summary_call["logprobs"] is True
    assert "extra_body" in summary_call
    assert summary_call["extra_body"]["prompt_token_ids"] == padded_summary_ids
    assert summary_call["extra_body"]["return_token_ids"] is True
    assert len(padded_summary_ids) % 16 == 0

    # Outer call (calls[1]) carries the outer padded ids via extra_body.
    outer_call = calls[1]
    assert outer_call["extra_body"]["prompt_token_ids"] == padded_outer_ids
    assert len(padded_outer_ids) % 16 == 0


def test_eviction_mode_padding_disabled_no_padding():
    """When _padding_config is disabled, even eviction mode falls back to
    the raw apply_chat_template + encode path — no render_padded_prompt
    call."""
    _install_mt(max_turns=8)
    _install_summary(mode="eviction", compaction_max_turns=2)
    _install_padding(enabled=False)

    async def fake_orig(self, *args, **kwargs):
        if _is_summary_call(kwargs):
            return _make_summary_response("S")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    call_count = {"n": 0}

    def fake_render(**kwargs):
        call_count["n"] += 1
        return "rendered", [0], []

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    with patch.object(env, "render_padded_prompt", fake_render):
        _run_with_fake_orig(fake_orig, factory)

    # Padding is disabled: render_padded_prompt never called.
    assert call_count["n"] == 0


def test_markovian_mode_ignores_padding_config():
    """Markovian mode must be byte-identical regardless of padding
    config — no render_padded_prompt invocation even when padding is
    enabled. This is the parity-regression guard."""
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=2)
    _install_padding(enabled=True, block_size=16)

    async def fake_orig(self, *args, **kwargs):
        if _is_summary_call(kwargs):
            return _make_summary_response("S")
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    call_count = {"n": 0}

    def fake_render(**kwargs):
        call_count["n"] += 1
        return "rendered", [0], []

    async def factory(patched):
        return await patched(self=None, model="m", messages=msgs)

    with patch.object(env, "render_padded_prompt", fake_render):
        _run_with_fake_orig(fake_orig, factory)

    assert call_count["n"] == 0


def test_eviction_mode_captures_compaction_events_on_summary():
    """In eviction mode, compaction_events emitted by vLLM during the
    summary call's prefill/decode must land on the
    summary_trainsample dict. The trainer uses these events to compute
    prompt_aligned_len correctly."""
    _install_mt(max_turns=8)
    _install_summary(mode="eviction", compaction_max_turns=2)

    events = [
        {
            "num_output_tokens_at_compaction": 128,
            "tokens_evicted": 512,
            "position_offset_after": 4096,
            "num_prompt_tokens": 2048,
            "evict_start": 0,
        },
        {},
    ]

    async def fake_orig(self, *args, **kwargs):
        if _is_summary_call(kwargs):
            resp = _make_summary_response(
                "S",
                prompt_ids=[1, 2, 3],
                completion_ids=[9, 10],
                logprobs=[-0.1, -0.2],
            )
            resp.compaction_events = events
            return resp
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    captured = {}

    async def factory(patched):
        r = await patched(self=None, model="test-model", messages=msgs)
        captured["response"] = r
        return r

    _run_with_fake_orig(fake_orig, factory)

    attached = captured["response"].summary_trainsample
    event = attached["compaction_events"][0]
    for key, value in events[0].items():
        assert event[key] == value
    assert event["compaction_strategy"] == "fifo"
    assert event["attention_matching_pre_sample"] is False
    # The env.py extractor adds defaults for the Phase A fields
    # (new_user_fragment_len, kept_indices, kept_token_ids,
    # last_turn_evicted, num_turns_evicted_after) when the response
    # doesn't carry them. The original 5 fields must still match.
    assert len(attached["compaction_events"]) == 1
    captured_event = attached["compaction_events"][0]
    assert {
        key: captured_event[key]
        for key in (
            "num_output_tokens_at_compaction",
            "tokens_evicted",
            "position_offset_after",
            "num_prompt_tokens",
            "evict_start",
        )
    } == events[0]
    assert captured_event["new_user_fragment_len"] == 0
    assert captured_event["kept_indices"] == []
    assert captured_event["kept_token_ids"] == []
    assert captured_event["last_turn_evicted"] == -1
    assert captured_event["num_turns_evicted_after"] == 0


def test_markovian_mode_ignores_compaction_events_on_summary():
    """Markovian mode must not capture compaction_events even if a test
    response has them. The summary sample's compaction_events must be
    an empty list (not None), so the orchestrator sees a no-event
    sample and builds a plain TrainingSample."""
    _install_mt(max_turns=8)
    _install_summary(mode="markovian", compaction_max_turns=2)

    events = [
        {
            "num_output_tokens_at_compaction": 64,
            "tokens_evicted": 256,
            "position_offset_after": 2048,
            "num_prompt_tokens": 1024,
            "evict_start": 0,
        },
    ]

    async def fake_orig(self, *args, **kwargs):
        if _is_summary_call(kwargs):
            resp = _make_summary_response(
                "S",
                prompt_ids=[1, 2],
                completion_ids=[3, 4],
                logprobs=[-0.1, -0.2],
            )
            resp.compaction_events = events
            return resp
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    captured = {}

    async def factory(patched):
        r = await patched(self=None, model="test-model", messages=msgs)
        captured["response"] = r
        return r

    _run_with_fake_orig(fake_orig, factory)

    attached = captured["response"].summary_trainsample
    assert attached["compaction_events"] == []


def test_prefill_trim_summary_carries_exact_replay_and_latches_phase4():
    _install_mt(max_turns=8)
    _install_summary(mode="eviction", compaction_max_turns=2)
    _install_padding(enabled=True, block_size=4, phase4_enabled=True)

    submitted = list(range(8))
    event = _prefill_trim_summary_event(submitted)
    captured = {}

    def fake_render(**kwargs):
        return "rendered", submitted, []

    async def fake_orig(self, *args, **kwargs):
        if _is_summary_call(kwargs):
            response = _make_summary_response(
                "S",
                prompt_ids=submitted,
                completion_ids=[20, 21],
                logprobs=[-0.1, -0.2],
            )
            response.compaction_events = [event]
            response.compaction_replay_mode = "prefill_trim"
            return response
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        state = env._get_or_create_phase4_state()
        state["prev_state_tokens"] = [90, 91]
        response = await patched(self=None, model="test-model", messages=msgs)
        captured["response"] = response
        captured["state"] = state
        captured["next_prompt"] = env._build_phase4_incremental_prompt(
            [_user("future")],
            env._padding_config,
        )
        return response

    with patch.object(env, "render_padded_prompt", fake_render):
        _run_with_fake_orig(fake_orig, factory)

    payload = captured["response"].summary_trainsample
    assert payload["compaction_replay_mode"] == "prefill_trim"
    assert payload["submitted_prompt_token_ids"] == submitted
    assert payload["native_prompt_token_ids"] == submitted
    assert payload["prompt_token_ids"] == event["kept_token_ids"]
    assert payload["completion_token_ids"] == [20, 21]
    assert payload["completion_logprobs"] == pytest.approx([-0.1, -0.2])
    assert payload["completion_temperature"] == pytest.approx(0.3)
    assert len(payload["compaction_events"]) == 1
    assert captured["state"]["prefill_trim_replay"] is True
    assert "prev_state_tokens" not in captured["state"]
    assert captured["next_prompt"] is None


def test_prefill_trim_summary_carries_eventless_turn_compaction_state():
    _install_mt(max_turns=8)
    _install_summary(mode="eviction", compaction_max_turns=2)
    _install_padding(enabled=True, block_size=4)

    submitted = list(range(8))
    carried_prefix_len = 6
    state = {
        "version": 1,
        "position_offset": 8,
        "protected_prefix_len": 2,
        "num_turns_evicted": 2,
        "carried_prefix_num_live_turns": 1,
        "carried_prefix_len": carried_prefix_len,
    }
    state["state_id"] = compute_turn_compaction_state_id(
        **state,
        carried_prefix_token_ids=submitted[:carried_prefix_len],
    )
    captured = {}

    def fake_render(**kwargs):
        return "rendered", submitted, []

    async def fake_orig(self, *args, **kwargs):
        if _is_summary_call(kwargs):
            response = _make_summary_response(
                "S",
                prompt_ids=submitted,
                completion_ids=[20, 21],
                logprobs=[-0.1, -0.2],
            )
            response.compaction_replay_mode = "prefill_trim"
            response.turn_compaction_state = state
            return response
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        response = await patched(self=None, model="test-model", messages=msgs)
        captured["response"] = response
        return response

    with patch.object(env, "render_padded_prompt", fake_render):
        _run_with_fake_orig(fake_orig, factory)

    payload = captured["response"].summary_trainsample
    assert payload["prompt_token_ids"] == submitted
    assert payload["compaction_events"] == []
    assert payload["turn_compaction_state"] == state


@pytest.mark.parametrize(
    ("history", "error"),
    [
        pytest.param(
            "valid-plus-malformed",
            "exactly one raw compaction event",
        ),
        pytest.param(
            "two-valid",
            "exactly one raw compaction event",
        ),
        pytest.param(
            "malformed-sole",
            "exactly one valid compaction event",
        ),
        pytest.param(
            "missing",
            "concrete list or tuple",
        ),
    ],
)
def test_prefill_trim_summary_rejects_invalid_event_history_after_latching(
    history, error
):
    _install_mt(max_turns=8)
    _install_summary(mode="eviction", compaction_max_turns=2)
    _install_padding(enabled=True, block_size=4, phase4_enabled=True)

    submitted = list(range(8))
    event = _prefill_trim_summary_event(submitted)
    captured = {}

    def fake_render(**kwargs):
        return "rendered", submitted, []

    async def fake_orig(self, *args, **kwargs):
        if _is_summary_call(kwargs):
            response = _make_summary_response(
                "S",
                prompt_ids=submitted,
                completion_ids=[20],
                logprobs=[-0.1],
            )
            response.compaction_replay_mode = "prefill_trim"
            if history == "valid-plus-malformed":
                response.compaction_events = [event, {}]
            elif history == "two-valid":
                response.compaction_events = [event, dict(event)]
            elif history == "malformed-sole":
                response.compaction_events = [{}]
            return response
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        state = env._get_or_create_phase4_state()
        state["prev_state_tokens"] = [90, 91]
        captured["state"] = state
        with pytest.raises(ValueError, match=error):
            await patched(self=None, model="m", messages=msgs)

    with patch.object(env, "render_padded_prompt", fake_render):
        _run_with_fake_orig(fake_orig, factory)

    assert captured["state"]["prefill_trim_replay"] is True
    assert "prev_state_tokens" not in captured["state"]


@pytest.mark.parametrize("failure", ["empty_text", "malformed_survivors"])
def test_prefill_trim_summary_latches_phase4_before_later_rejection(failure):
    _install_mt(max_turns=8)
    _install_summary(mode="eviction", compaction_max_turns=2)
    _install_padding(enabled=True, block_size=4, phase4_enabled=True)

    submitted = list(range(8))
    kept = None if failure == "empty_text" else [0, 1, 99, 5, 6, 7]
    event = _prefill_trim_summary_event(submitted, kept_token_ids=kept)
    captured = {}

    def fake_render(**kwargs):
        return "rendered", submitted, []

    async def fake_orig(self, *args, **kwargs):
        if _is_summary_call(kwargs):
            response = _make_summary_response(
                "" if failure == "empty_text" else "S",
                prompt_ids=submitted,
                completion_ids=[20],
                logprobs=[-0.1],
            )
            response.compaction_events = [event]
            response.compaction_replay_mode = "prefill_trim"
            return response
        return _outer_response()

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        state = env._get_or_create_phase4_state()
        state["prev_state_tokens"] = [90, 91]
        captured["state"] = state
        if failure == "malformed_survivors":
            with pytest.raises(ValueError, match="survivor tokens"):
                await patched(self=None, model="m", messages=msgs)
            return None
        return await patched(self=None, model="m", messages=msgs)

    with patch.object(env, "render_padded_prompt", fake_render):
        _run_with_fake_orig(fake_orig, factory)

    assert captured["state"]["prefill_trim_replay"] is True
    assert "prev_state_tokens" not in captured["state"]


def test_prefill_trim_summary_without_exact_submitted_ids_is_unsupported():
    _install_mt(max_turns=8)
    _install_summary(mode="eviction", compaction_max_turns=2)

    submitted = list(range(8))
    event = _prefill_trim_summary_event(submitted)

    async def fake_orig(self, *args, **kwargs):
        response = _make_summary_response(
            "S",
            prompt_ids=None,
            completion_ids=[20],
            logprobs=[-0.1],
        )
        response.compaction_events = [event]
        response.compaction_replay_mode = "prefill_trim"
        return response

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        with pytest.raises(ValueError, match="requires exact submitted"):
            await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)


def test_summary_unknown_replay_mode_fails_closed():
    _install_mt(max_turns=8)
    _install_summary(mode="eviction", compaction_max_turns=2)

    async def fake_orig(self, *args, **kwargs):
        response = _make_summary_response("S")
        response.compaction_replay_mode = "future_mode"
        return response

    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        with pytest.raises(ValueError, match="unsupported compaction_replay_mode"):
            await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)


def test_summary_is_reused_between_triggers_in_eviction_mode():
    """Eviction mode does not persist the splice -- the engine compresses KV
    while the client-visible history keeps growing -- so `n_real` never falls
    back below the threshold and every turn past the trigger would otherwise
    generate a fresh summary, each prefilling the FULL history. The
    per-rollout cache restores the intended "every N real turns" cadence.

    Markovian mode instead persists the splice, which resets `n_real` and
    makes the trigger periodic on its own; it bypasses this cache (see
    test_markovian_persistence_regenerates_instead_of_reusing_a_stale_summary).
    """
    _install_mt(max_turns=99)
    _install_summary(mode="eviction", compaction_max_turns=2, resume_text="go on")

    summary_calls = 0
    outer_calls = 0

    async def fake_orig(self, *args, **kwargs):
        nonlocal summary_calls, outer_calls
        if env._IN_SUMMARY_CALL.get():
            summary_calls += 1
            return SimpleNamespace(
                id=f"sum-{summary_calls}",
                prompt_token_ids=[1, 2],
                choices=[
                    SimpleNamespace(
                        token_ids=[9, 9, 9],
                        message=SimpleNamespace(content=f"SUMMARY-{summary_calls}"),
                        # healthy stack: logprobs present (their absence now
                        # triggers the rescore fallback, an extra call this
                        # test's counters must not see)
                        logprobs=SimpleNamespace(
                            content=[SimpleNamespace(logprob=-0.1)] * 3
                        ),
                    )
                ],
            )
        outer_calls += 1
        return _outer_response()

    def _convo(n_real_turns):
        msgs = [_sys()]
        for i in range(1, n_real_turns + 1):
            msgs.extend(_turn(i))
        msgs.append(_user("pending"))
        return msgs

    async def factory(patched):
        # One asyncio task == one rollout == one ContextVar context.
        for n in (3, 4, 5, 6):
            await patched(self=None, model="m", messages=_convo(n))

    _run_with_fake_orig(fake_orig, factory)

    assert outer_calls == 4, outer_calls
    # n_real=3 generate (gen@3) | 4 reuse (4-3<2) | 5 generate (5-3==2) | 6 reuse
    assert summary_calls == 2, (
        f"expected 2 generations across 4 turns, got {summary_calls} "
        "(pre-cache behaviour regenerates every turn)"
    )
    assert env._markovian_stats.get("n_summary_cache_hits") == 2


def test_markovian_persistence_regenerates_instead_of_reusing_a_stale_summary():
    """Markovian mode persists the splice, so `n_real` RESETS after every
    compaction instead of climbing.

    That breaks the cache's staleness test, which is
    ``n_real - n_real_at_gen < compaction_max_turns``. Post-reset the next
    trigger sees n_real back at exactly compaction_max_turns -- the same
    value it had when the summary was generated -- so the delta is 0 at
    every future trigger and the FIRST summary would be reused for the
    rest of the rollout, never reflecting anything that happened after it.
    Markovian mode therefore bypasses the cache.
    """
    _install_mt(max_turns=2, stride=2)
    _install_summary(mode="markovian", compaction_max_turns=2)

    summary_calls = 0

    async def fake_orig(self, *args, **kwargs):
        nonlocal summary_calls
        if env._IN_SUMMARY_CALL.get():
            summary_calls += 1
            return SimpleNamespace(
                id=f"sum-{summary_calls}",
                prompt_token_ids=[1, 2],
                choices=[
                    SimpleNamespace(
                        token_ids=[9],
                        message=SimpleNamespace(content=f"SUMMARY-{summary_calls}"),
                        logprobs=SimpleNamespace(
                            content=[SimpleNamespace(logprob=-0.1)]
                        ),
                    )
                ],
            )
        return _outer_response()

    # Turn A: fresh history, 2 real turns -> fires.
    convo_a = [_sys(), *_turn(1), *_turn(2), _user("o3")]
    # Turn B: what the env now replays after persistence -- the summary is
    # the base, followed by the turns taken since. 3 groups, one of which is
    # the summary exchange, so n_real == 2 -> fires again.
    convo_b = [
        _sys(),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUMMARY-1"},
        *_turn(3),
        *_turn(4),
        _user("o5"),
    ]

    async def factory(patched):
        await patched(self=None, model="m", messages=convo_a)
        await patched(self=None, model="m", messages=convo_b)

    _run_with_fake_orig(fake_orig, factory)

    assert summary_calls == 2, (
        f"expected a fresh summary at each trigger, got {summary_calls} "
        "(the cache pinned the first summary for the whole rollout)"
    )
    assert not env._markovian_stats.get("n_summary_cache_hits")


def test_attach_compacted_prompt_makes_the_summary_the_new_base():
    """The interceptor's rewrite of kwargs["messages"] only affects the
    outbound request. Persisting it into the trajectory step's prompt is
    what makes the next turn continue from [sys][I][S][obs] -- verifiers
    rebuilds each prompt from trajectory[-1]["prompt"] + ["completion"].
    """
    compacted = [
        _sys(),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUM"},
        _user("pending"),
    ]
    response = SimpleNamespace(kv_compacted_prompt=compacted)
    step = {"prompt": [_sys(), *_turn(1), *_turn(2), _user("pending")], "extras": {}}

    env.attach_compacted_prompt_from_response(step, response)
    assert step["prompt"] == compacted

    # No summary this turn -> prompt untouched.
    untouched = {"prompt": [_sys(), _user("pending")], "extras": {}}
    before = list(untouched["prompt"])
    env.attach_compacted_prompt_from_response(untouched, SimpleNamespace())
    assert untouched["prompt"] == before


@pytest.mark.parametrize(
    "stride,expected_body_groups",
    [
        (None, 2),   # None == evict 1  (matches truncation.py:98 default)
        (1, 2),      # evict 1 oldest   -> 2 of 3 groups survive
        (2, 1),      # evict 2 oldest
        (3, 0),      # evict the whole window -> full reset (summary is the context)
        # stride > max_turns is rejected at config time (env.py:7258,
        # "stride must be in [1, max_turns]"), so over-evict is unreachable.
    ],
)
def test_stride_means_turns_evicted_not_turns_preserved(stride, expected_body_groups):
    """`stride` must mean the same thing on every path: how many OLDEST turn
    groups a trigger evicts (truncation.py:98, vLLM
    compaction_eviction_turn_stride).

    It used to be passed straight through to the splice as a PRESERVE count,
    so the same value did opposite things: stride=10 evicted 10 groups on the
    eviction path but kept 10 here, and stride=None evicted 1 there but kept 0
    here -- i.e. the gentlest-looking setting was the most destructive.
    """
    # preserved = max_turns - stride, so max_turns IS the window under test
    env.configure_markovian_thinker(
        enabled=True, tokenizer=_StubTok(), max_turns=3, stride=stride
    )
    _install_summary(mode="markovian", compaction_max_turns=1, resume_text="go")

    async def fake_orig(self, *args, **kwargs):
        if env._IN_SUMMARY_CALL.get():
            return SimpleNamespace(
                id="s", prompt_token_ids=[1],
                choices=[SimpleNamespace(
                    token_ids=[9], message=SimpleNamespace(content="SUM"),
                    logprobs=None)],
            )
        captured["sent"] = kwargs["messages"]
        return _outer_response()

    captured: dict = {}
    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3), _user("pending")]

    async def factory(patched):
        await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    sent = captured["sent"]
    # Shape is sys + [I, S] + surviving groups + tail. The surviving groups
    # are the NEWEST ones, so with 3 groups and k evicted they are
    # turns (k+1)..3.
    surviving = [_turn(i) for i in range(3 - expected_body_groups + 1, 4)]
    expected = [
        _sys(),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUM"},
        *[m for grp in surviving for m in grp],
        _user("pending"),
    ]
    assert sent == expected, (
        f"stride={stride}: expected {expected_body_groups} surviving groups "
        f"-- sent={[m['content'] for m in sent]}"
    )


def test_compacted_prompt_survives_the_native_to_verifiers_boundary():
    """The interceptor stashes the compacted prompt on the NATIVE
    ChatCompletion, but the consumer reads it off the VERIFIERS Response.
    Upstream builds that Response from a hardcoded field list, so anything
    not explicitly forwarded in patched_from_native is silently dropped.

    Regression: kv_compacted_prompt was attached but never forwarded, so
    attach_compacted_prompt_from_response always saw nothing, the step's
    prompt was never rewritten, and the summary was never persisted --
    reverting to a fresh summary every turn, the exact pathology
    persistence exists to prevent.

    Asserting on the real boundary rather than handing a stub straight to
    the attach helper is the whole point: the stub-level test passed
    throughout.
    """
    import asyncio

    from openai.types.chat import ChatCompletion
    from verifiers.clients.openai_chat_completions_client import (
        OpenAIChatCompletionsClient,
    )

    compacted = [
        _sys(),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUM"},
        _user("pending"),
    ]

    native = ChatCompletion.model_validate(
        {
            "id": "c1",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "act"},
                    "finish_reason": "stop",
                }
            ],
        }
    )
    env._attach_compacted_prompt(native, compacted)

    client = object.__new__(OpenAIChatCompletionsClient)
    vf_response = asyncio.run(client.from_native_response(native))

    step = {"prompt": [_sys(), *_turn(1), _user("pending")], "extras": {}}
    env.attach_compacted_prompt_from_response(step, vf_response)

    assert step["prompt"] == compacted, (
        "compacted prompt did not survive native -> verifiers.Response; "
        "it must be forwarded in patched_from_native"
    )


def test_logical_budget_accumulates_across_persisted_splices():
    """The logical (cumulative-episode) budget must keep growing across a
    persisted splice. Persistence shrinks the VISIBLE history to
    [sys][I][SUM][tail]; without the evicted-tokens base the next call would
    measure only that window, max_logical_seq_len would never bind, and the
    fairness cap against the full-context arm silently dies (episodes run to
    max_episode_steps regardless of the 32k budget).
    """
    _install_mt(max_turns=2, stride=2)
    _install_summary(mode="markovian", compaction_max_turns=2)

    async def fake_orig(self, *args, **kwargs):
        if env._IN_SUMMARY_CALL.get():
            return _make_summary_response("SUM", completion_ids=[9] * 5)
        return _outer_response()

    seen = {}

    async def factory(patched):
        # Turn A: 2 real turns -> summary fires and is persisted.
        convo_a = [_sys(), *_turn(1), *_turn(2), _user("o3")]
        r1 = await patched(self=None, model="m", messages=convo_a)
        seen["base_after_splice"] = env._LOGICAL_EVICTED_TOKENS.get()
        seen["logical_1"] = r1.logical_seq_len
        # Turn B: what the env replays after persistence.
        convo_b = [
            _sys(),
            {"role": "user", "content": INSTR},
            {"role": "assistant", "content": "SUM"},
            _user("o3"),
            _asst("a3"),
            _user("o4"),
        ]
        r2 = await patched(self=None, model="m", messages=convo_b)
        seen["logical_2"] = r2.logical_seq_len
        return r2

    _run_with_fake_orig(fake_orig, factory)

    assert seen["base_after_splice"] > 0, (
        "persisted splice must bank the evicted tokens into "
        "_LOGICAL_EVICTED_TOKENS"
    )
    # The second call's logical length must include the banked base, i.e.
    # exceed what tokenizing its own (compacted) messages alone would give.
    # _StubTok.encode always returns 3 ids, so the visible part is 3.
    assert seen["logical_2"] >= seen["base_after_splice"] + 3


def test_summary_call_stats_ride_the_response():
    """markovian_summary/* metrics are aggregated from per-step extras in the
    orchestrator process; the interceptor must attach per-call stats to the
    response (module counters never cross the process boundary)."""
    _install_mt(max_turns=2, stride=2)
    _install_summary(mode="markovian", compaction_max_turns=2)

    async def fake_orig(self, *args, **kwargs):
        if env._IN_SUMMARY_CALL.get():
            return _make_summary_response(
                "SUM", prompt_ids=[1] * 40, completion_ids=[9] * 7,
                logprobs=[-0.1] * 7,
            )
        return _outer_response()

    captured = {}

    async def factory(patched):
        msgs = [_sys(), *_turn(1), *_turn(2), _user("pending")]
        captured["response"] = await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    stats = captured["response"].markovian_summary_call
    assert stats["n_generated"] == 1
    assert stats["n_failed"] == 0
    assert stats["prompt_tokens"] == 40
    assert stats["output_tokens"] == 7
    assert stats["latency_ms"] >= 0
    trunc = captured["response"].markovian_truncation
    assert trunc["n_truncations"] == 1

    # And they survive into a trajectory step's extras.
    step = {"prompt": [], "extras": {}}
    env.attach_summary_call_stats_from_response(step, captured["response"])
    env.attach_markovian_truncation_from_response(step, captured["response"])
    assert step["extras"]["markovian_summary_call"]["output_tokens"] == 7
    assert step["extras"]["markovian_truncation"]["n_truncations"] == 1


def test_summary_logprobs_recovered_via_rescore():
    """When the summary response has echoed ids but no parseable logprobs
    (the production drop), a prompt_logprobs rescore recovers the anchor
    and the TrainingSample is emitted instead of dropped."""
    _install_mt(max_turns=2, stride=2)
    _install_summary(mode="markovian", compaction_max_turns=2)

    prompt_ids = [11, 12, 13]
    completion_ids = [21, 22]
    calls = []

    async def fake_orig(self, *args, **kwargs):
        calls.append(dict(kwargs))
        eb = kwargs.get("extra_body") or {}
        if eb.get("prompt_logprobs") == 0:
            # rescore call: prompt_logprobs aligned to prompt+completion,
            # first entry None, string token-id keys as JSON would give
            full = eb["prompt_token_ids"]
            assert full == prompt_ids + completion_ids
            pl = [None] * len(prompt_ids) + [
                {str(t): {"logprob": -0.25 * (i + 1)}}
                for i, t in enumerate(completion_ids)
            ]
            return SimpleNamespace(
                prompt_logprobs=pl,
                choices=[SimpleNamespace(
                    token_ids=[9], message=SimpleNamespace(content="x"),
                    logprobs=None)],
            )
        if env._IN_SUMMARY_CALL.get():
            # summary generation: ids echoed, logprobs MISSING
            return _make_summary_response(
                "SUM", prompt_ids=prompt_ids,
                completion_ids=completion_ids, logprobs=None,
            )
        return _outer_response()

    captured = {}

    async def factory(patched):
        msgs = [_sys(), *_turn(1), *_turn(2), _user("pending")]
        captured["response"] = await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    sample = captured["response"].summary_trainsample
    assert sample["completion_token_ids"] == completion_ids
    assert sample["completion_logprobs"] == pytest.approx([-0.25, -0.5])
    # rescore reuses the summary's own sampling params
    rescore = [c for c in calls
               if (c.get("extra_body") or {}).get("prompt_logprobs") == 0]
    assert len(rescore) == 1
    assert rescore[0]["temperature"] == pytest.approx(0.3)
    assert env._markovian_stats.get("n_summary_logprob_rescores") == 1


def test_rescore_failure_degrades_to_the_old_drop():
    """If the rescore itself yields nothing usable, behavior matches the
    pre-recovery world: sample carries 0 logprobs and is dropped
    downstream -- no crash, no partial data."""
    _install_mt(max_turns=2, stride=2)
    _install_summary(mode="markovian", compaction_max_turns=2)

    async def fake_orig(self, *args, **kwargs):
        eb = kwargs.get("extra_body") or {}
        if eb.get("prompt_logprobs") == 0:
            return _outer_response()  # no prompt_logprobs field at all
        if env._IN_SUMMARY_CALL.get():
            return _make_summary_response(
                "SUM", prompt_ids=[1, 2], completion_ids=[5, 6],
                logprobs=None,
            )
        return _outer_response()

    captured = {}

    async def factory(patched):
        msgs = [_sys(), *_turn(1), *_turn(2), _user("pending")]
        captured["response"] = await patched(self=None, model="m", messages=msgs)

    _run_with_fake_orig(fake_orig, factory)

    sample = captured["response"].summary_trainsample
    assert sample["completion_token_ids"] == [5, 6]
    assert sample["completion_logprobs"] == []
    assert not env._markovian_stats.get("n_summary_logprob_rescores")
