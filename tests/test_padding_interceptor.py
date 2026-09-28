# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the AsyncOpenAI.chat.completions.create interceptor
installed by `kv_eviction.env._install_message_padding_interceptor`.

Replaces the underlying `orig_create` with an AsyncMock that records the
kwargs it received, then invokes the installed wrapper and asserts the
wrapper has:
  - not touched kwargs when padding is disabled,
  - injected `extra_body={"prompt_token_ids": ...}` when padding is
    enabled,
  - merged into (rather than clobbered) a pre-existing `extra_body`,
  - stashed the padded ids onto the returned response object so
    Patch #1 can forward them to the verifiers Response.

No HTTP, no real model, no tokenizer: a stub tokenizer with a fixed
return value is enough to exercise the branching logic.
"""

from types import SimpleNamespace

import pytest

import kv_eviction.env as env
from kv_eviction.padding import resolve_im_end_token_id


class _StubTok:
    pad_token_id = 100

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=False
    ):
        return "TOK:"

    def encode(self, s, add_special_tokens=False):
        if s == "<|im_start|>assistant\n":
            return [9]
        if s.startswith("<|im_end|>\n<|im_start|>user\n"):
            return [999, 8, 999]
        if (
            '{"retrieve": ["T0001"]}' in s
            and "hidden memory spans are restored" in s
        ):
            return [44, 45, 999, 8, 999]
        if "hidden memory spans are restored" in s:
            return [7, 999, 8, 999, 9]
        # raw: 3 tokens before an <|im_end|>; block_size=4 means 0 padding
        # (body=3, target=3).
        return [1, 2, 3, 999]

    def convert_tokens_to_ids(self, tok):
        return 999 if tok == "<|im_end|>" else -1


def test_turn_terminator_resolver_rejects_unk_fallback_for_gemma_style_token():
    class StubTokenizer:
        unk_token_id = 3

        def convert_tokens_to_ids(self, tok):
            return 106 if tok == "<turn|>" else 3

    assert resolve_im_end_token_id(StubTokenizer()) == 106


def _managed_context_compaction_event(span_id="T0001"):
    return SimpleNamespace(
        num_output_tokens_at_compaction=0,
        tokens_evicted=128,
        position_offset_after=128,
        num_prompt_tokens=512,
        evict_start=64,
        last_turn_evicted=3,
        num_turns_evicted_after=2,
        evicted_token_ids=[8, 9, 10],
        kept_indices=[],
        kept_token_ids=[],
        new_user_fragment_len=0,
        archived_span_ids=[span_id],
    )


def _prefill_trim_event():
    return {
        "num_output_tokens_at_compaction": 0,
        "tokens_evicted": 2,
        "position_offset_after": 2,
        "num_prompt_tokens": 4,
        "evict_start": 2,
        "new_user_fragment_len": 1,
        "kept_indices": [0, 1, 4, 5],
        "kept_token_ids": [10, 11, 14, 15],
        "last_turn_evicted": 0,
        "num_turns_evicted_after": 1,
    }


def _managed_context_turn_messages(n_turns: int):
    messages = [{"role": "system", "content": "system"}]
    for idx in range(n_turns):
        messages.extend(
            [
                {"role": "user", "content": f"obs {idx}"},
                {"role": "assistant", "content": f"<action>act {idx}</action>"},
            ]
        )
    messages.append({"role": "user", "content": "current obs"})
    return messages


@pytest.fixture
def stub_response():
    # Pydantic-style stand-in; we only care that setattr works for
    # `prompt_token_ids` forwarding.
    return SimpleNamespace(id="resp-0", prompt_token_ids=None)


@pytest.fixture(autouse=True)
def reset_padding_config():
    env.reset_managed_context_stats()
    yield
    env._padding_config = None
    env.reset_managed_context_stats()


def _get_patched_create():
    from openai.resources.chat.completions.completions import AsyncCompletions

    return AsyncCompletions.create


def test_integration_disabled_is_exact_passthrough(stub_response):
    import asyncio
    from unittest.mock import patch

    env._padding_config = None
    env._markovian_config = None
    called_with = {}
    request_kwargs = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.6,
        "extra_body": {"custom": {"nested": True}},
    }

    async def fake_orig(self, *args, **kwargs):
        called_with.update(kwargs)
        return stub_response

    # Reinstall interceptor on top of fake_orig.
    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create

        async def run():
            return await patched(self=None, **request_kwargs)

        asyncio.run(run())

    assert called_with == request_kwargs
    assert called_with["messages"] is request_kwargs["messages"]
    assert called_with["extra_body"] is request_kwargs["extra_body"]
    assert stub_response.prompt_token_ids is None


def test_streaming_is_exact_passthrough_when_padding_enabled(stub_response):
    import asyncio
    from unittest.mock import patch

    called_with = {}
    request_kwargs = {
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "extra_body": {"custom": True},
    }

    async def fake_orig(self, *args, **kwargs):
        called_with.update(kwargs)
        return stub_response

    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create
        env.configure_message_padding(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
        )

        async def run():
            return await patched(self=None, **request_kwargs)

        asyncio.run(run())

    assert called_with == request_kwargs
    assert called_with["messages"] is request_kwargs["messages"]
    assert called_with["extra_body"] is request_kwargs["extra_body"]
    assert stub_response.prompt_token_ids is None


def test_padding_enabled_injects_extra_body(stub_response):
    import asyncio
    from unittest.mock import patch

    called_with = {}

    async def fake_orig(self, *args, **kwargs):
        called_with.update(kwargs)
        return stub_response

    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create
        env.configure_message_padding(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
        )

        async def run():
            return await patched(
                self=None,
                model="m",
                messages=[{"role": "user", "content": "hi"}],
            )

        asyncio.run(run())

    assert "extra_body" in called_with
    # raw=[1,2,3,999] body=3 tokens before <|im_end|> at block_size=4
    # -> no padding needed, prompt_token_ids == raw.
    assert called_with["extra_body"]["prompt_token_ids"] == [1, 2, 3, 999]
    assert called_with["extra_body"]["return_token_ids"] is True
    assert called_with["logprobs"] is True
    assert called_with["top_logprobs"] == 0
    # Forward-stash on response.
    assert stub_response.prompt_token_ids == [1, 2, 3, 999]


def test_padding_preserves_explicit_training_metadata_opt_out():
    import asyncio
    from unittest.mock import patch

    called_with = {}
    mode_one_response = SimpleNamespace(
        prompt_token_ids=[1, 2, 3, 999],
        compaction_replay_mode="prefill_trim",
        choices=[SimpleNamespace(token_ids=[4], logprobs=None)],
    )

    async def fake_orig(self, *args, **kwargs):
        called_with.update(kwargs)
        return mode_one_response

    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create
        env.configure_message_padding(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
        )

        async def run():
            return await patched(
                self=None,
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                logprobs=False,
                extra_body={"return_token_ids": False},
            )

        response = asyncio.run(run())

    assert called_with["logprobs"] is False
    assert "top_logprobs" not in called_with
    assert called_with["extra_body"]["return_token_ids"] is False
    assert called_with["extra_body"]["prompt_token_ids"] == [1, 2, 3, 999]
    with pytest.raises(ValueError, match="standard completion logprobs"):
        env._strict_prefill_trim_native_transport(
            response,
            response.submitted_prompt_token_ids,
            context="prefill_trim response",
        )


def test_padding_merges_preexisting_extra_body(stub_response):
    import asyncio
    from unittest.mock import patch

    called_with = {}

    async def fake_orig(self, *args, **kwargs):
        called_with.update(kwargs)
        return stub_response

    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create
        env.configure_message_padding(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
        )

        async def run():
            return await patched(
                self=None,
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                extra_body={"some_other_flag": True},
            )

        asyncio.run(run())

    assert called_with["extra_body"]["some_other_flag"] is True
    assert called_with["extra_body"]["prompt_token_ids"] == [1, 2, 3, 999]
    assert called_with["extra_body"]["return_token_ids"] is True


def test_padding_overlong_prompt_raises_before_http(stub_response):
    import asyncio
    from unittest.mock import patch

    from verifiers.errors import OverlongPromptError

    called = False

    async def fake_orig(self, *args, **kwargs):
        nonlocal called
        called = True
        return stub_response

    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create
        env.configure_message_padding(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
            max_prompt_len=4,
        )

        async def run():
            return await patched(
                self=None,
                model="m",
                messages=[{"role": "user", "content": "hi"}],
            )

        with pytest.raises(OverlongPromptError):
            asyncio.run(run())

    assert called is False
    assert stub_response.prompt_token_ids is None


def test_padding_allows_server_to_clamp_requested_completion_budget(stub_response):
    import asyncio
    from unittest.mock import patch

    called_with = {}

    async def fake_orig(self, *args, **kwargs):
        called_with.update(kwargs)
        return stub_response

    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create
        env.configure_message_padding(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
            max_prompt_len=8,
        )

        async def run():
            return await patched(
                self=None,
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                max_completion_tokens=8,
            )

        asyncio.run(run())

    assert called_with["max_completion_tokens"] == 8
    assert called_with["extra_body"]["prompt_token_ids"] == [1, 2, 3, 999]


@pytest.mark.parametrize(
    ("state", "submitted_ids", "max_seq_len", "expected_budget"),
    [
        ({}, [1, 2, 3, 4], 8, 1),
        (
            {"prev_state_tokens": [1, 2, 3, 4], "logical_seq_len": 10},
            [1, 2, 3, 4, 5, 6],
            16,
            1,
        ),
    ],
)
def test_phase4_logical_sequence_budget_caps_completion(
    state,
    submitted_ids,
    max_seq_len,
    expected_budget,
):
    cfg = env.MessagePaddingConfig(
        enabled=True,
        tokenizer=_StubTok(),
        block_size=4,
        filler_token_id=100,
        im_end_token_id=999,
        max_logical_seq_len=max_seq_len,
        phase4_enabled=True,
    )
    kwargs = {"max_completion_tokens": 32}

    was_capped = env._apply_phase4_logical_sequence_budget(
        kwargs,
        submitted_ids,
        cfg,
        state,
    )

    assert kwargs["max_completion_tokens"] == expected_budget
    assert was_capped


def test_phase4_logical_sequence_budget_reports_unchanged_request():
    cfg = env.MessagePaddingConfig(
        enabled=True,
        tokenizer=_StubTok(),
        block_size=4,
        filler_token_id=100,
        im_end_token_id=999,
        max_logical_seq_len=32,
        phase4_enabled=True,
    )
    kwargs = {"max_completion_tokens": 4}

    was_capped = env._apply_phase4_logical_sequence_budget(
        kwargs,
        [1, 2, 3, 4],
        cfg,
        {},
    )

    assert kwargs["max_completion_tokens"] == 4
    assert not was_capped


def test_phase4_sequence_budget_excludes_padding_with_physical_backstop():
    cfg = env.MessagePaddingConfig(
        enabled=True,
        tokenizer=_StubTok(),
        block_size=4,
        filler_token_id=100,
        im_end_token_id=999,
        max_logical_seq_len=8,
        count_padding_toward_sequence_limit=False,
        max_padding_tokens=4,
        phase4_enabled=True,
    )
    kwargs = {"max_completion_tokens": 32}
    state = {}

    was_capped = env._apply_phase4_logical_sequence_budget(
        kwargs,
        [1, 100, 2, 100],
        cfg,
        state,
        new_prompt_padding_tokens=2,
    )

    assert was_capped
    assert kwargs["max_completion_tokens"] == 5
    assert state["pending_logical_padding_prompt_len"] == 2


def test_phase4_response_reports_padding_and_non_padding_lengths():
    import asyncio

    async def run():
        env.set_phase4_rollout_metadata(
            env="textworld",
            example_id="padding-counts",
            game_id="padding-counts",
            task="easy-nav",
            current_turn=0,
        )
        cfg = env.MessagePaddingConfig(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
            max_logical_seq_len=8,
            count_padding_toward_sequence_limit=False,
            max_padding_tokens=4,
            phase4_enabled=True,
        )
        state = env._get_phase4_state()
        assert state is not None
        env._apply_phase4_logical_sequence_budget(
            {"max_completion_tokens": 2},
            [1, 100, 2, 100],
            cfg,
            state,
            new_prompt_padding_tokens=2,
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="x"),
                    token_ids=[3, 4],
                )
            ],
            padding_token_ids=[100, 100],
            compaction_events=[],
        )

        env._update_phase4_state_from_response(
            response,
            [1, 100, 2, 100],
            cfg,
        )

        assert response.logical_seq_len == 8
        assert response.logical_padding_seq_len == 4
        assert response.logical_non_padding_seq_len == 4
        assert response.logical_sequence_limit_len == 4
        assert response.context_seq_len == 8
        assert response.context_padding_seq_len == 4
        assert response.context_non_padding_seq_len == 4

    asyncio.run(run())


def test_phase4_padding_allowance_is_a_hard_physical_limit():
    from verifiers.errors import OverlongPromptError

    cfg = env.MessagePaddingConfig(
        enabled=True,
        tokenizer=_StubTok(),
        block_size=4,
        filler_token_id=100,
        im_end_token_id=999,
        max_logical_seq_len=10,
        count_padding_toward_sequence_limit=False,
        max_padding_tokens=4,
        phase4_enabled=True,
    )
    state = {
        "prev_state_tokens": [1, 2, 3, 4],
        "logical_seq_len": 10,
        "logical_padding_seq_len": 5,
    }

    with pytest.raises(OverlongPromptError, match="cumulative logical sequence"):
        env._apply_phase4_logical_sequence_budget(
            {"max_completion_tokens": 32},
            [1, 2, 3, 4, 5, 100],
            cfg,
            state,
            new_prompt_padding_tokens=1,
        )


def test_phase4_logical_sequence_budget_stops_before_overflow():
    from verifiers.errors import OverlongPromptError

    cfg = env.MessagePaddingConfig(
        enabled=True,
        tokenizer=_StubTok(),
        block_size=4,
        filler_token_id=100,
        im_end_token_id=999,
        max_logical_seq_len=12,
        phase4_enabled=True,
    )
    state = {
        "prev_state_tokens": [1, 2, 3, 4],
        "logical_seq_len": 10,
    }

    with pytest.raises(OverlongPromptError, match="cumulative logical sequence"):
        env._apply_phase4_logical_sequence_budget(
            {"max_completion_tokens": 32},
            [1, 2, 3, 4, 5],
            cfg,
            state,
        )


def test_managed_context_retrieve_parser_is_strict():
    assert env._parse_managed_context_retrieve(
        '{"retrieve": ["T0001", "T0001", "T0002"]}', 2
    ) == ["T0001", "T0002"]
    assert (
        env._parse_managed_context_retrieve(
            '{"retrieve": ["T0001", "T0002", "T0003"]}', 2
        )
        is None
    )
    assert env._parse_managed_context_retrieve(
        'prefix {"retrieve": []}', 2
    ) is None
    assert (
        env._parse_managed_context_retrieve(
            '{"retrieve": ["T0004: secret answer for key item_003."]}', 2
        )
        is None
    )


def test_managed_context_retry_adds_restore_xargs():
    import asyncio
    from unittest.mock import patch

    calls = []
    first_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"retrieve": ["T0001"]}'),
                token_ids=[144, 145, 999],
            )
        ],
        compaction_events=[_managed_context_compaction_event("T0001")],
        prompt_token_ids=None,
    )
    final_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="apple"),
                token_ids=[],
            )
        ],
        prompt_token_ids=None,
    )

    async def fake_orig(self, *args, **kwargs):
        calls.append(kwargs)
        return first_response if len(calls) == 1 else final_response

    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create
        env.configure_message_padding(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
            phase4_enabled=True,
            managed_context_enabled=True,
            recall_max_spans=1,
        )

        async def run():
            return await patched(
                self=None,
                model="m",
                messages=[{"role": "user", "content": "what fruit?"}],
            )

        response = asyncio.run(run())

    assert response is final_response
    assert len(calls) == 2
    assert all(call["logprobs"] is True for call in calls)
    assert all(call["top_logprobs"] == 0 for call in calls)
    assert all(call["extra_body"]["return_token_ids"] is True for call in calls)
    assert calls[1]["extra_body"]["vllm_xargs"]["kve_restore_span_ids"] == [
        "T0001"
    ]
    assert calls[0]["extra_body"]["prompt_token_ids"] == [1, 2, 3, 999]
    assert calls[1]["extra_body"]["prompt_token_ids"] == [
        1,
        2,
        3,
        999,
        144,
        145,
        999,
        100,
        7,
        999,
        100,
        100,
        8,
        999,
        100,
        100,
        9,
        100,
        100,
        9,
    ]
    assert calls[1]["extra_body"]["vllm_xargs"][
        "kve_restore_after_visible_tokens"
    ] == len(
        [
            1,
            2,
            3,
            999,
            144,
            145,
            999,
            100,
            7,
            999,
            100,
            100,
            8,
            999,
            100,
            100,
            9,
            100,
            100,
        ]
    )
    assert calls[1]["extra_body"]["vllm_xargs"][
        "kve_restore_defer_until_prefill"
    ] is True
    assert calls[1]["extra_body"]["vllm_xargs"][
        "kve_phase4_expected_cached_tokens"
    ] == 8
    assert final_response.prompt_token_ids == calls[1]["extra_body"][
        "prompt_token_ids"
    ]
    assert (
        calls[1]["extra_body"]["vllm_xargs"]["kve_phase4_call_idx"]
        == calls[0]["extra_body"]["vllm_xargs"]["kve_phase4_call_idx"] + 1
    )
    stats = env.get_managed_context_stats()
    assert stats["retrieve_requests"] == 1
    assert stats["retrieve_spans_requested"] == 1
    assert stats["retrieve_spans_unavailable"] == 0
    assert stats["restore_retries"] == 1
    assert stats["restore_retries_without_spans"] == 0


def test_managed_context_recall_event_records_turn_metadata(monkeypatch):
    import asyncio
    from unittest.mock import patch

    monkeypatch.setenv("KVE_MANAGED_CONTEXT_RECORD_RECALL_EVENTS", "1")
    calls = []
    first_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"retrieve": ["T0001"]}'),
                token_ids=[144, 145, 999],
            )
        ],
        compaction_events=[_managed_context_compaction_event("T0001")],
        prompt_token_ids=None,
    )
    final_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="apple"),
                token_ids=[],
            )
        ],
        prompt_token_ids=None,
    )

    async def fake_orig(self, *args, **kwargs):
        calls.append(kwargs)
        return first_response if len(calls) == 1 else final_response

    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create
        env.configure_message_padding(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
            phase4_enabled=True,
            managed_context_enabled=True,
            managed_context_index_enabled=True,
            recall_max_spans=1,
        )

        async def run():
            env.set_phase4_rollout_metadata(
                env="textworld",
                example_id=17,
                game_id="80",
                task="easy-nav",
                current_turn=7,
            )
            return await patched(
                self=None,
                model="m",
                messages=[{"role": "user", "content": "what fruit?"}],
            )

        asyncio.run(run())

    events = env.get_managed_context_recall_events()
    assert len(events) == 1
    event = events[0]
    assert event["env"] == "textworld"
    assert event["example_id"] == 17
    assert event["game_id"] == "80"
    assert event["current_turn"] == 7
    assert event["requested_span_ids"] == ["T0001"]
    assert event["restored_span_ids"] == ["T0001"]
    assert event["visible_index_span_ids"] == ["T0001"]
    assert event["hidden_tokens"] == 128
    assert event["restored_span_rows"][0]["original_turn_start"] == 0
    assert event["restored_span_rows"][0]["original_turn_end"] == 1
    context_events = env.get_managed_context_context_events()
    assert len(context_events) == 1
    assert context_events[0]["current_turn"] == 7
    assert context_events[0]["visible_index_span_ids"] == []
    assert context_events[0]["available_span_rows"] == []


def test_managed_context_retry_filters_unavailable_restore_span():
    import asyncio
    from unittest.mock import patch

    calls = []
    first_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"retrieve": ["T0099"]}'),
                token_ids=[144, 145, 999],
            )
        ],
        compaction_events=[_managed_context_compaction_event("T0001")],
        prompt_token_ids=None,
    )
    final_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="answer from visible context"),
                token_ids=[],
            )
        ],
        prompt_token_ids=None,
    )

    async def fake_orig(self, *args, **kwargs):
        calls.append(kwargs)
        return first_response if len(calls) == 1 else final_response

    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create
        env.configure_message_padding(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
            phase4_enabled=True,
            managed_context_enabled=True,
            recall_max_spans=1,
        )

        async def run():
            return await patched(
                self=None,
                model="m",
                messages=[{"role": "user", "content": "what fruit?"}],
            )

        response = asyncio.run(run())

    assert response is final_response
    assert len(calls) == 2
    retry_xargs = calls[1]["extra_body"]["vllm_xargs"]
    assert "kve_restore_span_ids" not in retry_xargs
    assert "kve_restore_defer_until_prefill" not in retry_xargs
    assert "kve_restore_after_visible_tokens" not in retry_xargs
    stats = env.get_managed_context_stats()
    assert stats["retrieve_requests"] == 1
    assert stats["retrieve_spans_requested"] == 1
    assert stats["retrieve_spans_unavailable"] == 1
    assert stats["retrieve_requests_unavailable"] == 1
    assert stats["restore_retries"] == 1
    assert stats["restore_retries_without_spans"] == 1


def test_managed_context_summary_select_summarizes_new_span_and_retries(monkeypatch):
    import asyncio
    from unittest.mock import patch

    monkeypatch.setenv("KVE_MANAGED_CONTEXT_RECORD_RECALL_EVENTS", "1")
    calls = []
    manager_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"index_updates": [{"span": "NEW1", "summary": '
                        '"early recipe cues"}], "retrieve": ["NEW1"]}'
                    )
                ),
                token_ids=[144, 145, 999],
            )
        ],
        compaction_events=[_managed_context_compaction_event("T0001")],
        prompt_token_ids=None,
    )
    final_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="<action>go north</action>"),
                token_ids=[],
            )
        ],
        prompt_token_ids=None,
    )

    async def fake_orig(self, *args, **kwargs):
        calls.append(kwargs)
        return manager_response if len(calls) == 1 else final_response

    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create
        env.configure_message_padding(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
            phase4_enabled=True,
            managed_context_enabled=True,
            managed_context_index_enabled=True,
            recall_max_spans=1,
            managed_context_recall_mode="summary_select",
            managed_context_compaction_max_turns=1,
            managed_context_turns_last_kept=1,
        )

        async def run():
            return await patched(
                self=None,
                model="m",
                messages=[
                    {"role": "user", "content": "turn one"},
                    {"role": "assistant", "content": "<action>look</action>"},
                    {"role": "user", "content": "what next?"},
                ],
            )

        response = asyncio.run(run())

    assert response is final_response
    assert len(calls) == 2
    assert calls[0]["temperature"] == 0.0
    assert calls[0]["top_p"] == 1.0
    assert calls[0]["max_tokens"] == 256
    assert "temperature" not in calls[1]
    assert "top_p" not in calls[1]
    assert "max_tokens" not in calls[1]
    assert "kve_restore_span_ids" in calls[1]["extra_body"]["vllm_xargs"]
    assert calls[1]["extra_body"]["vllm_xargs"]["kve_restore_span_ids"] == [
        "T0001"
    ]
    stats = env.get_managed_context_stats()
    assert stats["memory_manager_requests"] == 1
    assert stats["span_summaries_written"] == 1
    assert stats["restore_retries"] == 1
    assert stats["require_retrieve_fallback_requests"] == 0
    recall_events = env.get_managed_context_recall_events()
    assert recall_events[0]["requested_span_ids"] == ["T0001"]
    assert recall_events[0]["restored_span_rows"][0]["summary"] == "early recipe cues"
    context_events = env.get_managed_context_context_events()
    assert context_events[0]["memory_manager_pass"] is True


def test_managed_context_summary_select_repairs_empty_summary(monkeypatch):
    import asyncio
    from unittest.mock import patch

    monkeypatch.setenv("KVE_MANAGED_CONTEXT_RECORD_RECALL_EVENTS", "1")
    calls = []
    empty_manager_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content='{"index_updates": [], "retrieve": []}'
                ),
                token_ids=[144, 145, 999],
            )
        ],
        compaction_events=[_managed_context_compaction_event("T0001")],
        prompt_token_ids=None,
    )
    repair_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=(
                        '{"index_updates": [{"span": "T0001", "summary": '
                        '"recipe and starting room cues"}], '
                        '"retrieve": ["T0001"]}'
                    )
                ),
                token_ids=[146, 147, 999],
            )
        ],
        compaction_events=[],
        prompt_token_ids=None,
    )
    final_response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="<action>go north</action>"),
                token_ids=[],
            )
        ],
        prompt_token_ids=None,
    )

    async def fake_orig(self, *args, **kwargs):
        calls.append(kwargs)
        return [empty_manager_response, repair_response, final_response][
            len(calls) - 1
        ]

    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create
        env.configure_message_padding(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
            phase4_enabled=True,
            managed_context_enabled=True,
            managed_context_index_enabled=True,
            recall_max_spans=1,
            managed_context_recall_mode="summary_select",
            managed_context_compaction_max_turns=1,
            managed_context_turns_last_kept=1,
        )

        async def run():
            return await patched(
                self=None,
                model="m",
                messages=[
                    {"role": "user", "content": "turn one"},
                    {"role": "assistant", "content": "<action>look</action>"},
                    {"role": "user", "content": "what next?"},
                ],
            )

        response = asyncio.run(run())

    assert response is final_response
    assert len(calls) == 3
    assert all(call["logprobs"] is True for call in calls)
    assert all(call["top_logprobs"] == 0 for call in calls)
    assert all(call["extra_body"]["return_token_ids"] is True for call in calls)
    assert calls[0]["temperature"] == 0.0
    assert calls[1]["temperature"] == 0.0
    assert "temperature" not in calls[2]
    assert calls[1]["extra_body"]["vllm_xargs"]["kve_phase4_call_idx"] == 1
    assert calls[2]["extra_body"]["vllm_xargs"]["kve_phase4_call_idx"] == 2
    assert calls[2]["extra_body"]["vllm_xargs"]["kve_restore_span_ids"] == [
        "T0001"
    ]
    stats = env.get_managed_context_stats()
    assert stats["memory_manager_requests"] == 1
    assert stats["memory_manager_repair_requests"] == 1
    assert stats["memory_manager_repair_successes"] == 1
    assert stats["memory_manager_repair_failures"] == 0
    assert stats["span_summaries_written"] == 1
    assert stats["restore_retries"] == 1
    assert stats["require_retrieve_fallback_requests"] == 0
    recall_events = env.get_managed_context_recall_events()
    assert recall_events[0]["requested_span_ids"] == ["T0001"]
    assert (
        recall_events[0]["restored_span_rows"][0]["summary"]
        == "recipe and starting room cues"
    )
    context_events = env.get_managed_context_context_events()
    assert context_events[0]["memory_manager_repair_attempted"] is True
    assert context_events[0]["memory_manager_repair_success"] is True
    assert context_events[0]["memory_manager_repair_missing_summary_span_ids"] == [
        "T0001"
    ]


def test_managed_context_index_is_non_leaking_and_opt_in():
    env.reset_managed_context_stats()
    cfg = env.MessagePaddingConfig(
        enabled=True,
        tokenizer=_StubTok(),
        block_size=4,
        filler_token_id=100,
        im_end_token_id=999,
        phase4_enabled=True,
        managed_context_enabled=True,
        recall_max_spans=2,
        managed_context_index_enabled=True,
        managed_context_index_max_entries=2,
    )
    state = {}
    response = SimpleNamespace(
        compaction_events=[
            _managed_context_compaction_event("T0001")
        ]
    )

    env._record_managed_context_archive_events(response, state)
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "What do you do?"},
    ]
    indexed = env._messages_with_managed_context_index(messages, cfg, state)

    assert indexed is not messages
    assert indexed[-1]["content"].startswith("What do you do?")
    assert "T0001" in indexed[-1]["content"]
    assert "Memory manager task" in indexed[-1]["content"]
    assert "Existing hidden-memory rows:" in indexed[-1]["content"]
    assert (
        "- T0001 | turns t0-t1 | older TextWorld observations/actions"
        in indexed[-1]["content"]
    )
    assert "New hidden-memory rows to summarize on this pass:" in indexed[-1]["content"]
    assert "- none on this pass." in indexed[-1]["content"]
    assert '"index_updates"' in indexed[-1]["content"]
    assert "The orchestrator owns this table." in indexed[-1]["content"]
    assert '"retrieve": ["T0001"]' in indexed[-1]["content"]
    assert "apple" not in indexed[-1]["content"]
    stats = env.get_managed_context_stats()
    assert stats["archived_spans_seen"] == 1
    assert stats["index_injections"] == 1


def test_managed_context_first_compaction_prompt_predicts_first_span_id():
    cfg = env.MessagePaddingConfig(
        enabled=True,
        tokenizer=_StubTok(),
        block_size=4,
        filler_token_id=100,
        im_end_token_id=999,
        phase4_enabled=True,
        managed_context_enabled=True,
        recall_max_spans=2,
        managed_context_index_enabled=True,
        managed_context_index_max_entries=0,
    )

    suffix = env._managed_context_index_suffix(
        cfg,
        {},
        memory_manager_due=True,
    )

    assert "Existing hidden-memory rows:" in suffix
    assert "- none yet." in suffix
    assert "New hidden-memory rows to summarize on this pass:" in suffix
    assert "- T0001 | turns earlier turns | <summarize>" in suffix
    assert "`index_updates`: [] is invalid" in suffix
    assert "Never use empty strings" in suffix
    assert "empty `index_updates` or empty `retrieve` is invalid" in suffix
    assert "Retrieval cardinality is mandatory" in suffix
    assert "too few IDs are invalid" in suffix
    assert '"index_updates": [{"span": "T0001"' in suffix
    assert '"retrieve": ["T0001"]' in suffix


def test_managed_context_prompt_describes_pending_compaction_span():
    cfg = env.MessagePaddingConfig(
        enabled=True,
        tokenizer=_StubTok(),
        block_size=4,
        filler_token_id=100,
        im_end_token_id=999,
        phase4_enabled=True,
        managed_context_enabled=True,
        recall_max_spans=2,
        managed_context_index_enabled=True,
        managed_context_index_max_entries=0,
        managed_context_compaction_max_turns=6,
        managed_context_turns_last_kept=2,
    )

    suffix = env._managed_context_index_suffix(
        cfg,
        {},
        memory_manager_due=True,
        messages=_managed_context_turn_messages(6),
    )

    assert "New hidden-memory rows to summarize on this pass:" in suffix
    assert "- T0001 | turns t0-t1 | <summarize>" in suffix
    assert "Every required summary must be a non-empty factual cue" in suffix
    assert "Do not return only one span when two or more are listed" in suffix
    assert "t2-t5 are expected to remain visible/live" in suffix
    assert "archives the oldest 2 live completed turn(s) per compaction" in suffix


def test_managed_context_index_includes_span_summary():
    env.reset_managed_context_stats()
    cfg = env.MessagePaddingConfig(
        enabled=True,
        tokenizer=_StubTok(),
        block_size=4,
        filler_token_id=100,
        im_end_token_id=999,
        phase4_enabled=True,
        managed_context_enabled=True,
        recall_max_spans=2,
        managed_context_index_enabled=True,
        managed_context_index_max_entries=0,
    )
    state = {}
    response = SimpleNamespace(
        compaction_events=[_managed_context_compaction_event("T0001")]
    )

    new_span_ids = env._record_managed_context_archive_events(response, state)
    env._apply_managed_context_index_updates(
        state=state,
        updates=[{"summary": "early recipe and start-room details"}],
        new_span_ids=new_span_ids,
    )
    indexed = env._messages_with_managed_context_index(
        [{"role": "user", "content": "What now?"}],
        cfg,
        state,
    )

    assert (
        "T0001 | turns t0-t1 | early recipe and start-room details"
        in indexed[-1]["content"]
    )
    assert env.get_managed_context_stats()["span_summaries_written"] == 1


def test_managed_context_new_alias_resolves_cumulative_numbering():
    assert env._resolve_managed_context_retrieve_aliases(
        ["NEW3", "T0001"],
        ["T0003"],
        prior_span_count=2,
    ) == ["T0003", "T0001"]


def test_managed_context_preobs_builds_memory_only_message():
    import asyncio

    async def run():
        env.configure_message_padding(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
            phase4_enabled=True,
            managed_context_enabled=True,
            managed_context_index_enabled=True,
            recall_max_spans=2,
            managed_context_recall_mode="summary_select_preobs",
            managed_context_compaction_max_turns=1,
            managed_context_turns_last_kept=1,
        )
        env.set_phase4_rollout_metadata(
            env="textworld",
            example_id="preobs-test",
            game_id="preobs-test",
            task="easy-nav",
            current_turn=1,
        )

        return env.build_managed_context_pre_observation_message(
            _managed_context_turn_messages(1)[:-1]
        )

    message = asyncio.run(run())

    assert message is not None
    assert "KVE_HIDDEN_MEMORY_MANAGER_PREOBS" in message
    assert "The next TextWorld observation is intentionally not shown yet" in message
    assert "New hidden-memory rows to summarize on this pass:" in message
    assert '"retrieve": ["T0001"]' in message
    assert env.get_managed_context_stats()["index_injections"] == 1


def test_phase4_rollout_metadata_resets_state_when_rollout_changes():
    import asyncio

    async def run():
        env.set_phase4_rollout_metadata(
            env="textworld",
            example_id=1,
            game_id="game-1",
            task="easy-nav",
            current_turn=0,
        )
        state = env._get_or_create_phase4_state()
        trace_id_1 = state["trace_id"]
        state["managed_context_archive_index"] = [{"span_id": "T0001"}]
        state["prev_state_tokens"] = [1, 2, 3]
        state["logical_seq_len"] = 9

        env.set_phase4_rollout_metadata(
            env="textworld",
            example_id=1,
            game_id="game-1",
            task="easy-nav",
            current_turn=1,
        )
        assert state["trace_id"] == trace_id_1
        assert state["managed_context_archive_index"] == [{"span_id": "T0001"}]

        env.set_phase4_rollout_metadata(
            env="textworld",
            example_id=2,
            game_id="game-2",
            task="easy-nav",
            current_turn=0,
        )
        state = env._get_or_create_phase4_state()
        assert state["trace_id"] != trace_id_1
        assert state["call_idx"] == 0
        assert "managed_context_archive_index" not in state
        assert "prev_state_tokens" not in state
        assert "logical_seq_len" not in state
        assert state["rollout_metadata"]["example_id"] == 2

    asyncio.run(run())


def test_phase4_task_completion_schedules_trace_release(monkeypatch):
    import asyncio

    scheduled = []
    monkeypatch.setattr(
        env,
        "_schedule_phase4_trace_release",
        lambda state: scheduled.append(
            (
                state["trace_id"],
                list(state["phase4_trace_release_targets"]),
            )
        ),
    )

    async def rollout():
        state = env._get_or_create_phase4_state()
        state["phase4_trace_release_targets"] = [
            {"base_url": "http://inference/v1", "headers": {}}
        ]
        return state["trace_id"]

    async def run():
        task = asyncio.create_task(rollout())
        trace_id = await task
        await asyncio.sleep(0)
        return trace_id

    trace_id = asyncio.run(run())

    assert scheduled == [
        (
            trace_id,
            [{"base_url": "http://inference/v1", "headers": {}}],
        )
    ]


def test_release_phase4_trace_is_awaited_and_idempotent(monkeypatch):
    import asyncio

    import httpx

    calls = []

    class FakeResponse:
        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *, timeout):
            assert timeout == 5

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, url, *, json, headers):
            calls.append((url, json, headers))
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

    async def run():
        state = env._get_or_create_phase4_state()
        state["trace_id"] = "trace-awaited"
        state["phase4_trace_release_targets"] = [
            {
                "base_url": "http://inference/v1",
                "headers": {"Authorization": "Bearer test"},
            }
        ]
        assert await env.release_phase4_trace()
        assert not await env.release_phase4_trace()

    asyncio.run(run())

    assert calls == [
        (
            "http://inference/v1/phase4/traces/release",
            {"trace_id": "trace-awaited"},
            {"Authorization": "Bearer test"},
        )
    ]


def test_phase4_state_uses_server_returned_auto_padding_tokens():
    import asyncio

    async def run():
        env.set_phase4_rollout_metadata(
            env="textworld",
            example_id="pad-state",
            game_id="pad-state",
            task="easy-nav",
            current_turn=0,
        )
        cfg = env.MessagePaddingConfig(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
            phase4_enabled=True,
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="x"),
                    token_ids=[4],
                )
            ],
            padding_token_ids=[100, 100],
            compaction_events=[],
        )

        env._update_phase4_state_from_response(response, [1, 2, 3], cfg)
        state = env._get_phase4_state()
        assert state is not None
        assert state["prev_state_tokens"] == [1, 2, 3, 4, 100, 100]
        assert state["logical_seq_len"] == 6
        assert response.logical_seq_len == 6

    asyncio.run(run())


def test_phase4_logical_length_survives_multiple_evictions():
    import asyncio

    async def run():
        env.set_phase4_rollout_metadata(
            env="textworld",
            example_id="logical-length",
            game_id="logical-length",
            task="easy-nav",
            current_turn=0,
        )
        cfg = env.MessagePaddingConfig(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
            phase4_enabled=True,
        )

        first_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="x"),
                    token_ids=[4],
                )
            ],
            padding_token_ids=[],
            compaction_events=[],
        )
        env._update_phase4_state_from_response(
            first_response,
            [1, 2, 3],
            cfg,
        )

        second_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="y"),
                    token_ids=[7, 8],
                )
            ],
            padding_token_ids=[],
            compaction_events=[{"kept_token_ids": [1, 2]}],
        )
        env._update_phase4_state_from_response(
            second_response,
            [1, 2, 3, 4, 5, 6],
            cfg,
        )

        third_response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="z"),
                    token_ids=[10, 11],
                )
            ],
            padding_token_ids=[],
            compaction_events=[{"kept_token_ids": [1]}],
        )
        env._update_phase4_state_from_response(
            third_response,
            [1, 2, 7, 8, 9],
            cfg,
        )

        state = env._get_phase4_state()
        assert state is not None
        assert second_response.logical_seq_len == 8
        assert third_response.logical_seq_len == 12
        assert state["logical_seq_len"] == 12
        assert state["prev_state_tokens"] == [1, 10, 11, 100]
        assert state["logical_seq_len"] > len(state["prev_state_tokens"])

    asyncio.run(run())


def test_phase4_logical_length_is_dropped_on_prefix_mismatch():
    import asyncio

    async def run():
        env.set_phase4_rollout_metadata(
            env="textworld",
            example_id="logical-prefix-mismatch",
            game_id="logical-prefix-mismatch",
            task="easy-nav",
            current_turn=0,
        )
        state = env._get_or_create_phase4_state()
        state["prev_state_tokens"] = [1, 2, 3, 4]
        state["logical_seq_len"] = 10
        cfg = env.MessagePaddingConfig(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
            phase4_enabled=True,
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="x"),
                    token_ids=[7],
                )
            ],
            padding_token_ids=[],
            compaction_events=[],
        )

        env._update_phase4_state_from_response(
            response,
            [9, 8, 7],
            cfg,
        )

        assert "logical_seq_len" not in state
        assert state["logical_seq_len_invalid"] is True
        assert not hasattr(response, "logical_seq_len")

    asyncio.run(run())


@pytest.mark.parametrize(
    "response",
    [
        SimpleNamespace(logical_seq_len=17),
        SimpleNamespace(
            logical_seq_len=None,
            model_extra={"logical_seq_len": 17},
        ),
    ],
)
def test_attach_logical_seq_len_from_response(response):
    step = {"extras": {}}

    env.attach_logical_seq_len_from_response(step, response)

    assert step["extras"]["logical_seq_len"] == 17


@pytest.mark.parametrize("use_model_extra", [False, True])
def test_prefill_trim_mode_is_attached_with_compaction_events(use_model_extra):
    payload = {
        "compaction_events": [_prefill_trim_event()],
        "compaction_replay_mode": "prefill_trim",
    }
    if use_model_extra:
        response = SimpleNamespace(
            compaction_events=None,
            compaction_replay_mode=None,
            model_extra=payload,
        )
    else:
        response = SimpleNamespace(**payload)
    step = {"extras": {}}

    env.attach_compaction_events_from_response(step, response)

    assert step["extras"]["compaction_replay_mode"] == "prefill_trim"
    assert step["extras"]["compaction_events"][0]["kept_token_ids"] == [
        10,
        11,
        14,
        15,
    ]


def _invalid_survivor_metadata(valid, malformation):
    if malformation == "scalar-string":
        return "".join(str(value) for value in valid)
    if malformation == "scalar-bytes":
        return bytes(valid)
    if malformation == "scalar-int":
        return valid[0]
    if malformation == "integral-float":
        return [float(valid[0]), *valid[1:]]
    if malformation == "truncatable-float":
        return [valid[0] + 0.9, *valid[1:]]
    if malformation == "bool":
        return [False, *valid[1:]]
    if malformation == "negative":
        return [-1, *valid[1:]]
    if malformation == "mixed-string":
        return [*valid[:-1], str(valid[-1])]
    raise AssertionError(f"unknown malformation: {malformation}")


@pytest.mark.parametrize("field", ["kept_indices", "kept_token_ids"])
@pytest.mark.parametrize(
    "malformation",
    [
        "scalar-string",
        "scalar-bytes",
        "scalar-int",
        "integral-float",
        "truncatable-float",
        "bool",
        "negative",
        "mixed-string",
    ],
)
def test_prefill_trim_attachment_rejects_invalid_survivor_metadata(
    field,
    malformation,
):
    event = _prefill_trim_event()
    event[field] = _invalid_survivor_metadata(event[field], malformation)
    response = SimpleNamespace(
        compaction_events=[event],
        compaction_replay_mode="prefill_trim",
    )

    with pytest.raises(ValueError, match=rf"{field} must"):
        env.attach_compaction_events_from_response({"extras": {}}, response)


def test_prefill_trim_attachment_accepts_tuple_survivor_metadata():
    event = _prefill_trim_event()
    event["kept_indices"] = tuple(event["kept_indices"])
    event["kept_token_ids"] = tuple(event["kept_token_ids"])
    response = SimpleNamespace(
        compaction_events=[event],
        compaction_replay_mode="prefill_trim",
    )
    step = {"extras": {}}

    env.attach_compaction_events_from_response(step, response)

    assert step["extras"]["compaction_events"][0]["kept_indices"] == [0, 1, 4, 5]
    assert step["extras"]["compaction_events"][0]["kept_token_ids"] == [
        10,
        11,
        14,
        15,
    ]


def test_legacy_attachment_keeps_permissive_survivor_coercion():
    event = _prefill_trim_event()
    event["kept_indices"] = [0.0, "1", False]
    event["kept_token_ids"] = [10.0, "11", True]
    response = SimpleNamespace(compaction_events=[event])
    step = {"extras": {}}

    env.attach_compaction_events_from_response(step, response)

    converted = step["extras"]["compaction_events"][0]
    assert converted["kept_indices"] == [0, 1, 0]
    assert converted["kept_token_ids"] == [10, 11, 1]
    assert "compaction_replay_mode" not in step["extras"]


@pytest.mark.parametrize("malformation", ["missing", "empty"])
def test_prefill_trim_attachment_requires_nonempty_kept_indices(malformation):
    event = _prefill_trim_event()
    if malformation == "missing":
        event.pop("kept_indices")
    else:
        event["kept_indices"] = []
    response = SimpleNamespace(
        compaction_events=[event],
        compaction_replay_mode="prefill_trim",
    )

    with pytest.raises(ValueError, match="requires non-empty kept_indices"):
        env.attach_compaction_events_from_response({"extras": {}}, response)


@pytest.mark.parametrize(
    ("raw_events", "error"),
    [
        pytest.param(
            [_prefill_trim_event(), {}],
            "exactly one raw compaction event",
            id="valid-plus-malformed",
        ),
        pytest.param(
            [_prefill_trim_event(), _prefill_trim_event()],
            "exactly one raw compaction event",
            id="two-valid",
        ),
        pytest.param(
            [{}],
            "exactly one valid compaction event",
            id="malformed-sole",
        ),
        pytest.param(
            None,
            "concrete list or tuple",
            id="missing",
        ),
    ],
)
def test_prefill_trim_attachment_rejects_invalid_event_history(raw_events, error):
    response = SimpleNamespace(compaction_replay_mode="prefill_trim")
    if raw_events is not None:
        response.compaction_events = raw_events
    step = {"extras": {}}

    with pytest.raises(ValueError, match=error):
        env.attach_compaction_events_from_response(step, response)

    assert step["extras"] == {}


def test_legacy_attachment_skips_malformed_extra_event():
    event = _prefill_trim_event()
    event.pop("kept_indices")
    response = SimpleNamespace(
        compaction_events=[event, {}],
    )
    step = {"extras": {}}

    env.attach_compaction_events_from_response(step, response)

    assert len(step["extras"]["compaction_events"]) == 1
    assert step["extras"]["compaction_events"][0]["kept_token_ids"] == [
        10,
        11,
        14,
        15,
    ]
    assert step["extras"]["compaction_events"][0]["kept_indices"] == []
    assert "compaction_replay_mode" not in step["extras"]


def test_unknown_compaction_replay_mode_is_rejected():
    response = SimpleNamespace(
        compaction_events=[_prefill_trim_event()],
        compaction_replay_mode="unknown",
    )

    with pytest.raises(ValueError, match="unsupported compaction_replay_mode"):
        env.attach_compaction_events_from_response({"extras": {}}, response)


def test_prefill_trim_disables_phase4_state_rebuild(monkeypatch):
    import asyncio

    async def run():
        state = env._get_or_create_phase4_state()
        assert state is not None
        state["prev_state_tokens"] = [1, 2, 3]
        cfg = env.MessagePaddingConfig(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
            phase4_enabled=True,
        )

        def unexpected_update(*args, **kwargs):
            raise AssertionError("Phase4 kept-state update must not run")

        monkeypatch.setattr(
            env,
            "_update_phase4_state_from_response",
            unexpected_update,
        )
        response = SimpleNamespace(compaction_replay_mode="prefill_trim")
        assert not env._maybe_update_phase4_state_from_response(
            response,
            [1, 2, 3],
            cfg,
        )
        assert "prev_state_tokens" not in state
        assert state["prefill_trim_replay"] is True
        assert (
            env._build_phase4_incremental_prompt(
                [{"role": "user", "content": "next"}],
                cfg,
            )
            is None
        )
        assert not env._maybe_update_phase4_state_from_response(
            SimpleNamespace(),
            [4, 5, 6],
            cfg,
        )

    asyncio.run(run())


@pytest.mark.parametrize("fallback", ["missing_messages", "render_failure"])
def test_prefill_trim_latches_phase4_state_on_passthrough_paths(
    monkeypatch, fallback
):
    import asyncio
    from unittest.mock import patch

    response = SimpleNamespace(compaction_replay_mode="prefill_trim")

    async def fake_orig(self, *args, **kwargs):
        return response

    from openai.resources.chat.completions.completions import AsyncCompletions

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        patched = AsyncCompletions.create
        env.configure_message_padding(
            enabled=True,
            tokenizer=_StubTok(),
            block_size=4,
            filler_token_id=100,
            im_end_token_id=999,
            phase4_enabled=True,
        )

        if fallback == "render_failure":
            def fail_render(*args, **kwargs):
                raise ValueError("bad chat template")

            monkeypatch.setattr(env, "render_padded_prompt", fail_render)

        async def run():
            state = env._get_or_create_phase4_state()
            state["prev_state_tokens"] = [10, 11]
            kwargs = {"model": "m"}
            if fallback == "render_failure":
                kwargs["messages"] = [{"role": "user", "content": "hello"}]
            result = await patched(self=None, **kwargs)
            assert result is response
            assert state["prefill_trim_replay"] is True
            assert "prev_state_tokens" not in state

        asyncio.run(run())


def test_phase4_expected_cached_len_can_backoff_one_block(monkeypatch):
    cfg = env.MessagePaddingConfig(
        enabled=True,
        tokenizer=_StubTok(),
        block_size=16,
        filler_token_id=100,
        im_end_token_id=999,
        phase4_enabled=True,
    )

    assert env._phase4_expected_cached_len(1040, cfg) == 1040
    monkeypatch.setenv("KVE_PHASE4_EXPECTED_CACHED_BACKOFF_BLOCKS", "1")
    assert env._phase4_expected_cached_len(1040, cfg) == 1024


def test_visible_prefill_prompt_inserts_archived_tokens():
    env.reset_managed_context_stats()
    state = {
        "managed_context_archive_index": [
            {
                "span_id": "T0001",
                "evict_start": 2,
                "evicted_token_ids": [8, 9, 10],
            }
        ]
    }

    out = env._visible_prefill_prompt_for_spans([1, 2, 3, 4], ["T0001"], state)

    assert out == [1, 2, 8, 9, 10, 3, 4]
    stats = env.get_managed_context_stats()
    assert stats["visible_prefill_requests"] == 1
    assert stats["visible_prefill_tokens"] == 3
