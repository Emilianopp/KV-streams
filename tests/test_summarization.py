# SPDX-License-Identifier: Apache-2.0
"""Unit tests for kv_eviction.summarization — pure helpers backing the
summary-based eviction feature described in
``plans/markovian_summary.md``.

No tokenizer, no async, no I/O. Mocked-``orig_create`` integration
tests live in ``tests/test_summary_interceptor.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from kv_eviction.summarization import (
    SummaryTrainSample,
    build_exchange,
    build_post_summary_messages,
    content_to_text,
    count_summary_exchanges,
    extract_completion_logprobs,
    extract_completion_token_ids,
    extract_prompt_token_ids,
    partition_messages,
    sanitize_summary,
)

# ─── Fixtures (mirror tests/test_truncation.py shapes) ───


def _sys(content="sys"):
    return {"role": "system", "content": content}


def _user(content="u"):
    return {"role": "user", "content": content}


def _assistant(content="a", tool_calls=None):
    msg = {"role": "assistant", "content": content}
    if tool_calls is not None:
        msg["tool_calls"] = tool_calls
    return msg


def _tool(content="t"):
    return {"role": "tool", "content": content}


def _turn(i):
    return [_user(f"u{i}"), _assistant(f"a{i}")]


# ─── partition_messages ───


def test_partition_empty():
    n, sys_p, groups, tail = partition_messages([])
    assert n == 0
    assert sys_p == []
    assert groups == []
    assert tail == []


def test_partition_system_only():
    msgs = [_sys("a"), _sys("b")]
    n, sys_p, groups, tail = partition_messages(msgs)
    assert n == 0
    assert sys_p == msgs
    assert groups == []
    assert tail == []


def test_partition_no_user_yet():
    n, sys_p, groups, tail = partition_messages([_sys()])
    assert (n, sys_p, groups, tail) == (0, [_sys()], [], [])


def test_partition_pending_user_is_tail():
    msgs = [_sys(), _user("pending")]
    n, sys_p, groups, tail = partition_messages(msgs)
    assert n == 0
    assert sys_p == [_sys()]
    assert groups == []
    assert tail == [_user("pending")]


def test_partition_three_plain_turns():
    msgs = [_sys(), *_turn(1), *_turn(2), *_turn(3)]
    n, sys_p, groups, tail = partition_messages(msgs)
    assert n == 3
    assert sys_p == [_sys()]
    assert [g[0]["content"] for g in groups] == ["u1", "u2", "u3"]
    assert [g[-1]["content"] for g in groups] == ["a1", "a2", "a3"]
    assert tail == []


def test_partition_tool_chain_is_one_group():
    msgs = [
        _sys(),
        _user("u1"),
        _assistant(content=None, tool_calls=[{"id": "a"}]),
        _tool("t_a"),
        _assistant(content=None, tool_calls=[{"id": "b"}]),
        _tool("t_b"),
        _assistant("final1"),
        *_turn(2),
    ]
    n, sys_p, groups, tail = partition_messages(msgs)
    assert n == 2
    assert len(groups[0]) == 6  # u, a(tc), t, a(tc), t, a(final)
    assert groups[0][-1]["content"] == "final1"
    assert groups[1][-1]["content"] == "a2"


def test_partition_inflight_tail_preserved():
    msgs = [_sys(), *_turn(1), _user("pending")]
    n, sys_p, groups, tail = partition_messages(msgs)
    assert n == 1
    assert tail == [_user("pending")]


def test_partition_does_not_mutate_input():
    """partition_messages must not mutate the input list or its dicts.

    It DOES share dict references with the caller (no deep copy), so
    callers must not mutate returned dicts if they need the input
    untouched. That contract matches truncate_messages_to_last_k_turns.
    """
    from copy import deepcopy

    msgs = [_sys(), *_turn(1), *_turn(2)]
    snapshot = deepcopy(msgs)
    _n, _s, _groups, _t = partition_messages(msgs)
    assert msgs == snapshot


# ─── count_summary_exchanges ───


INSTR = "Summarize everything important."


def test_count_empty_no_instruction():
    assert count_summary_exchanges([], INSTR) == 0
    assert count_summary_exchanges([_sys(), *_turn(1)], "") == 0


def test_count_zero_when_no_match():
    msgs = [_sys(), *_turn(1), *_turn(2)]
    assert count_summary_exchanges(msgs, INSTR) == 0


def test_count_one_match():
    msgs = [
        _sys(),
        *_turn(1),
        _user(INSTR),
        _assistant("summary text"),
        *_turn(2),
    ]
    assert count_summary_exchanges(msgs, INSTR) == 1


def test_count_multiple_matches():
    msgs = [
        _sys(),
        *_turn(1),
        _user(INSTR),
        _assistant("sum1"),
        *_turn(2),
        _user(INSTR),
        _assistant("sum2"),
        *_turn(3),
    ]
    assert count_summary_exchanges(msgs, INSTR) == 2


def test_count_ignores_user_only_trailing_instruction():
    # Trailing user=INSTR with no following assistant — does NOT count.
    msgs = [_sys(), *_turn(1), _user(INSTR)]
    assert count_summary_exchanges(msgs, INSTR) == 0


def test_count_requires_exact_content_match():
    msgs = [
        _sys(),
        *_turn(1),
        _user(INSTR + " slight difference"),
        _assistant("close but not a summary"),
    ]
    assert count_summary_exchanges(msgs, INSTR) == 0


def test_count_ignores_assistant_instruction():
    # An assistant message with content == INSTR must not count.
    msgs = [_sys(), _assistant(INSTR), *_turn(1)]
    assert count_summary_exchanges(msgs, INSTR) == 0


def test_count_handles_multimodal_text_part():
    # content as [{"type":"text", "text": INSTR}] counts as match.
    msgs = [
        _sys(),
        *_turn(1),
        {"role": "user", "content": [{"type": "text", "text": INSTR}]},
        _assistant("summary"),
    ]
    assert count_summary_exchanges(msgs, INSTR) == 1


def test_count_rejects_multimodal_mixed_parts():
    msgs = [
        _sys(),
        {
            "role": "user",
            "content": [
                {"type": "text", "text": INSTR},
                {"type": "image_url", "image_url": "..."},
            ],
        },
        _assistant("x"),
    ]
    assert count_summary_exchanges(msgs, INSTR) == 0


# ─── build_exchange ───


def test_build_exchange_shapes():
    I_msg, S_msg = build_exchange(INSTR, "SUMMARY")
    assert I_msg == {"role": "user", "content": INSTR}
    assert S_msg == {"role": "assistant", "content": "SUMMARY"}
    # S_msg must not have tool_calls — it is a terminal assistant so
    # the next partition_messages() call will treat it as a turn boundary.
    assert "tool_calls" not in S_msg


def test_build_exchange_roundtrip_through_partition():
    I_msg, S_msg = build_exchange(INSTR, "SUMMARY")
    msgs = [_sys(), *_turn(1), I_msg, S_msg, *_turn(2)]
    n, _sp, groups, _tail = partition_messages(msgs)
    assert n == 3
    # The summary exchange forms the middle group.
    assert groups[1][0] == I_msg
    assert groups[1][1] == S_msg
    assert count_summary_exchanges(msgs, INSTR) == 1


# ─── build_post_summary_messages ───


def test_build_post_summary_markovian_drops_body():
    sys_p = [_sys()]
    body = [list(_turn(1)), list(_turn(2))]
    tail = [_user("pending")]
    out = build_post_summary_messages(
        mode="markovian",
        sys_prefix=sys_p,
        body_groups=body,
        tail=tail,
        instruction_text=INSTR,
        summary_text="SUMMARY",
    )
    # Shape: sys + [I, S] + preserved(=empty) + tail. The summary leads so
    # it can be persisted as the conversation's new base. The tail is the
    # pending user turn, so no resume message is added.
    assert out == [
        _sys(),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUMMARY"},
        _user("pending"),
    ]


def test_build_post_summary_eviction_keeps_body():
    sys_p = [_sys()]
    body = [list(_turn(1)), list(_turn(2))]
    tail = [_user("pending")]
    out = build_post_summary_messages(
        mode="eviction",
        sys_prefix=sys_p,
        body_groups=body,
        tail=tail,
        instruction_text=INSTR,
        summary_text="SUMMARY",
    )
    assert out == [
        _sys(),
        *_turn(1),
        *_turn(2),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUMMARY"},
        _user("pending"),
    ]


def test_build_post_summary_empty_body_groups():
    # No prior turns (only sys + tail). Markovian output still has [I, S].
    out = build_post_summary_messages(
        mode="markovian",
        sys_prefix=[_sys()],
        body_groups=[],
        tail=[],
        instruction_text=INSTR,
        summary_text="SUMMARY",
    )
    assert out == [
        _sys(),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUMMARY"},
    ]


def test_build_post_summary_markovian_skips_resume_when_tail_is_pending():
    """A non-empty tail IS the pending user turn, so no resume is added.

    Shape: sys + [I, S] + preserved + tail.
    """
    sys_p = [_sys()]
    body = [list(_turn(1)), list(_turn(2))]
    tail = [_user("pending")]
    out = build_post_summary_messages(
        mode="markovian",
        sys_prefix=sys_p,
        body_groups=body,
        tail=tail,
        instruction_text=INSTR,
        summary_text="SUMMARY",
        n_preserved_turns=1,
        resume_text="Please continue.",
    )
    assert out == [
        _sys(),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUMMARY"},
        *_turn(2),
        _user("pending"),
    ]


def test_build_post_summary_markovian_appends_resume_when_tail_is_empty():
    """No in-flight observation: the prompt would end on the assistant
    summary, so resume_text supplies the pending user turn."""
    out = build_post_summary_messages(
        mode="markovian",
        sys_prefix=[_sys()],
        body_groups=[list(_turn(1))],
        tail=[],
        instruction_text=INSTR,
        summary_text="SUMMARY",
        n_preserved_turns=0,
        resume_text="Please continue.",
    )
    assert out == [
        _sys(),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUMMARY"},
        {"role": "user", "content": "Please continue."},
    ]


def test_build_post_summary_markovian_drops_prior_summary_exchange():
    """A new summary supersedes the old one: only one [I, S] survives, and
    it leads. Otherwise a preserve window wider than the trigger period
    would stack summaries forward."""
    prior = [
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "OLD SUMMARY"},
    ]
    out = build_post_summary_messages(
        mode="markovian",
        sys_prefix=[_sys()],
        body_groups=[prior, list(_turn(1)), list(_turn(2))],
        tail=[_user("pending")],
        instruction_text=INSTR,
        summary_text="NEW SUMMARY",
        n_preserved_turns=99,  # wide enough to otherwise retain `prior`
    )
    assert out == [
        _sys(),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "NEW SUMMARY"},
        *_turn(1),
        *_turn(2),
        _user("pending"),
    ]
    assert count_summary_exchanges(out, INSTR) == 1


def test_build_post_summary_eviction_ignores_resume_text():
    """Eviction mode does not append resume_text (tail is already pending)."""
    sys_p = [_sys()]
    body = [list(_turn(1))]
    tail = [_user("pending")]
    out = build_post_summary_messages(
        mode="eviction",
        sys_prefix=sys_p,
        body_groups=body,
        tail=tail,
        instruction_text=INSTR,
        summary_text="S",
        resume_text="Please continue.",
    )
    assert out == [
        _sys(),
        *_turn(1),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "S"},
        _user("pending"),
    ]


def test_build_post_summary_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        build_post_summary_messages(
            mode="nope",
            sys_prefix=[_sys()],
            body_groups=[],
            tail=[],
            instruction_text=INSTR,
            summary_text="s",
        )


def test_build_post_summary_does_not_mutate_inputs():
    from copy import deepcopy

    sys_p = [_sys()]
    body = [list(_turn(1))]
    tail = [_user("pending")]
    sys_snap = deepcopy(sys_p)
    body_snap = deepcopy(body)
    tail_snap = deepcopy(tail)
    _ = build_post_summary_messages(
        mode="eviction",
        sys_prefix=sys_p,
        body_groups=body,
        tail=tail,
        instruction_text=INSTR,
        summary_text="SUMMARY",
    )
    assert sys_p == sys_snap
    assert body == body_snap
    assert tail == tail_snap


def test_build_post_summary_markovian_preserves_n_turns():
    """n_preserved_turns=2: sys + [I, S] + last_2_body_groups + tail."""
    sys_p = [_sys()]
    body = [list(_turn(1)), list(_turn(2)), list(_turn(3)), list(_turn(4))]
    tail = [_user("pending")]
    out = build_post_summary_messages(
        mode="markovian",
        sys_prefix=sys_p,
        body_groups=body,
        tail=tail,
        instruction_text=INSTR,
        summary_text="SUMMARY",
        n_preserved_turns=2,
    )
    assert out == [
        _sys(),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUMMARY"},
        *_turn(3),
        *_turn(4),
        _user("pending"),
    ]


def test_build_post_summary_markovian_preserves_more_than_available():
    """n_preserved_turns > available: keep all body groups (no error)."""
    sys_p = [_sys()]
    body = [list(_turn(1))]
    tail = [_user("pending")]
    out = build_post_summary_messages(
        mode="markovian",
        sys_prefix=sys_p,
        body_groups=body,
        tail=tail,
        instruction_text=INSTR,
        summary_text="SUMMARY",
        n_preserved_turns=5,
    )
    assert out == [
        _sys(),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUMMARY"},
        *_turn(1),
        _user("pending"),
    ]


def test_build_post_summary_markovian_preserve_zero_is_full_reset():
    sys_p = [_sys()]
    body = [list(_turn(1)), list(_turn(2))]
    tail = [_user("pending")]
    out = build_post_summary_messages(
        mode="markovian",
        sys_prefix=sys_p,
        body_groups=body,
        tail=tail,
        instruction_text=INSTR,
        summary_text="SUMMARY",
        n_preserved_turns=0,
    )
    assert out == [
        _sys(),
        {"role": "user", "content": INSTR},
        {"role": "assistant", "content": "SUMMARY"},
        _user("pending"),
    ]


def test_build_post_summary_eviction_ignores_n_preserved_turns():
    """Eviction mode keeps full body regardless of n_preserved_turns."""
    sys_p = [_sys()]
    body = [list(_turn(1)), list(_turn(2)), list(_turn(3))]
    tail = [_user("pending")]
    out_0 = build_post_summary_messages(
        mode="eviction",
        sys_prefix=sys_p,
        body_groups=body,
        tail=tail,
        instruction_text=INSTR,
        summary_text="S",
        n_preserved_turns=0,
    )
    out_99 = build_post_summary_messages(
        mode="eviction",
        sys_prefix=sys_p,
        body_groups=body,
        tail=tail,
        instruction_text=INSTR,
        summary_text="S",
        n_preserved_turns=99,
    )
    assert out_0 == out_99


def test_build_post_summary_eviction_roundtrip_through_partition():
    """Spliced list re-partitions correctly — the summary exchange
    lands as its own group, body groups remain intact."""
    msgs_before = [_sys(), *_turn(1), *_turn(2)]
    _n, sys_p, body, tail = partition_messages(msgs_before)
    out = build_post_summary_messages(
        mode="eviction",
        sys_prefix=sys_p,
        body_groups=body,
        tail=tail,
        instruction_text=INSTR,
        summary_text="SUMMARY",
    )
    n2, _sp2, groups2, _t2 = partition_messages(out)
    # Original 2 body groups + the new summary exchange group = 3.
    assert n2 == 3
    assert count_summary_exchanges(out, INSTR) == 1


# ─── sanitize_summary ───


def test_sanitize_clean_input_identity():
    text = "A perfectly fine summary with no weird tokens."
    out, modified = sanitize_summary(text)
    assert out == text
    assert modified is False


def test_sanitize_strips_im_start():
    text = "Leading text<|im_start|>assistant\nstuff"
    out, modified = sanitize_summary(text)
    assert "<|im_start|>" not in out
    assert modified is True


def test_sanitize_strips_im_end():
    text = "body<|im_end|>more body"
    out, modified = sanitize_summary(text)
    assert "<|im_end|>" not in out
    assert modified is True


def test_sanitize_custom_blocklist():
    text = "Hello <FOO>world<BAR>!"
    out, modified = sanitize_summary(text, blocklist=("<FOO>", "<BAR>"))
    assert out == "Hello world!"
    assert modified is True


def test_sanitize_empty_string():
    out, modified = sanitize_summary("")
    assert out == ""
    assert modified is False


# ─── content_to_text ───


def test_content_to_text_none():
    assert content_to_text(None) == ""


def test_content_to_text_string():
    assert content_to_text("hello") == "hello"


def test_content_to_text_list_text_parts():
    content = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    assert content_to_text(content) == "ab"


def test_content_to_text_non_text_parts():
    content = [
        {"type": "text", "text": "prefix "},
        {"type": "image_url", "image_url": "..."},
    ]
    out = content_to_text(content)
    assert "prefix " in out
    assert "[image_url]" in out


def test_content_to_text_fallback_on_weird_shape():
    # Dicts / objects that aren't in the expected shape go through str().
    assert content_to_text(42) == "42"
    assert content_to_text({"key": "value"}) == str({"key": "value"})


# ─── SummaryTrainSample ───


def test_summary_train_sample_defaults():
    s = SummaryTrainSample()
    assert s.prompt_token_ids == []
    assert s.completion_token_ids == []
    assert s.completion_logprobs == []
    assert s.model == ""
    assert s.compaction_events == []
    assert s.submitted_prompt_token_ids == []
    assert s.native_prompt_token_ids == []
    assert s.completion_temperature is None
    assert s.compaction_replay_mode is None
    assert s.turn_compaction_state is None


def test_summary_train_sample_roundtrip_lossless():
    s = SummaryTrainSample(
        prompt_token_ids=[1, 2, 3],
        completion_token_ids=[4, 5],
        completion_logprobs=[-0.1, -0.2],
        model="Qwen/Qwen2.5-4B",
    )
    d = s.to_dict()
    assert d == {
        "prompt_token_ids": [1, 2, 3],
        "completion_token_ids": [4, 5],
        "completion_logprobs": [-0.1, -0.2],
        "model": "Qwen/Qwen2.5-4B",
        "compaction_events": [],
    }
    s2 = SummaryTrainSample.from_dict(d)
    assert s == s2


def test_summary_train_sample_roundtrip_with_events():
    # Eviction mode: the interceptor stashes vLLM-side compaction events
    # on the sample dict as plain JSON-serializable dicts. Ensure the
    # roundtrip preserves them verbatim.
    events = [
        {
            "num_output_tokens_at_compaction": 128,
            "tokens_evicted": 512,
            "position_offset_after": 4608,
            "num_prompt_tokens": 5120,
            "evict_start": 0,
        }
    ]
    s = SummaryTrainSample(
        prompt_token_ids=[1, 2],
        completion_token_ids=[3, 4, 5],
        completion_logprobs=[-0.1, -0.2, -0.3],
        model="Qwen/Qwen2.5-4B",
        compaction_events=events,
    )
    d = s.to_dict()
    assert d["compaction_events"] == events
    s2 = SummaryTrainSample.from_dict(d)
    assert s2 == s


def test_summary_train_sample_roundtrip_with_prefill_trim_contract():
    event = {
        "num_output_tokens_at_compaction": 0,
        "tokens_evicted": 2,
        "position_offset_after": 2,
        "num_prompt_tokens": 4,
        "evict_start": 2,
        "kept_indices": [0, 1, 4, 5],
        "kept_token_ids": [10, 11, 14, 15],
    }
    state = {
        "version": 1,
        "position_offset": 2,
        "protected_prefix_len": 2,
        "num_turns_evicted": 1,
        "carried_prefix_num_live_turns": 1,
        "carried_prefix_len": 4,
        "state_id": "sha256:" + "a" * 64,
    }
    sample = SummaryTrainSample(
        prompt_token_ids=[10, 11, 14, 15],
        completion_token_ids=[20],
        completion_logprobs=[-0.1],
        model="test-model",
        compaction_events=[event],
        submitted_prompt_token_ids=[10, 11, 12, 13, 14, 15],
        native_prompt_token_ids=[10, 11, 12, 13, 14, 15],
        completion_temperature=0.25,
        compaction_replay_mode="prefill_trim",
        turn_compaction_state=state,
    )

    payload = sample.to_dict()

    assert payload["submitted_prompt_token_ids"] == [10, 11, 12, 13, 14, 15]
    assert payload["native_prompt_token_ids"] == [10, 11, 12, 13, 14, 15]
    assert payload["completion_temperature"] == 0.25
    assert payload["compaction_replay_mode"] == "prefill_trim"
    assert payload["turn_compaction_state"] == state
    assert SummaryTrainSample.from_dict(payload) == sample


def test_summary_train_sample_from_dict_defensive():
    # Missing keys default to empty.
    s = SummaryTrainSample.from_dict({})
    assert s.prompt_token_ids == []
    assert s.completion_token_ids == []
    assert s.completion_logprobs == []
    assert s.model == ""
    assert s.compaction_events == []


def test_summary_train_sample_from_dict_none_fields():
    # Explicit None for list fields is treated as empty.
    s = SummaryTrainSample.from_dict(
        {
            "prompt_token_ids": None,
            "completion_token_ids": None,
            "completion_logprobs": None,
            "model": None,
            "compaction_events": None,
        }
    )
    assert s.prompt_token_ids == []
    assert s.completion_token_ids == []
    assert s.completion_logprobs == []
    assert s.model == ""
    assert s.compaction_events == []


def test_summary_train_sample_from_dict_coerces_types():
    s = SummaryTrainSample.from_dict(
        {
            "prompt_token_ids": ["1", "2"],
            "completion_token_ids": (3.0, 4.0),
            "completion_logprobs": ("-0.1", "-0.2"),
            "model": 123,
        }
    )
    assert s.prompt_token_ids == [1, 2]
    assert s.completion_token_ids == [3, 4]
    assert s.completion_logprobs == [pytest.approx(-0.1), pytest.approx(-0.2)]
    assert s.model == "123"


# ─── extract_prompt_token_ids / extract_completion_token_ids /
#     extract_completion_logprobs ───


def _fake_response(
    *,
    prompt_ids=None,
    completion_ids=None,
    logprobs_content=None,
):
    lp = None
    if logprobs_content is not None:
        lp = SimpleNamespace(content=logprobs_content)
    choice = SimpleNamespace(token_ids=completion_ids, logprobs=lp)
    return SimpleNamespace(
        prompt_token_ids=prompt_ids,
        choices=[choice],
    )


def test_extract_prompt_token_ids_attribute():
    resp = _fake_response(prompt_ids=[1, 2, 3])
    assert extract_prompt_token_ids(resp) == [1, 2, 3]


def test_extract_prompt_token_ids_none_returns_empty():
    resp = _fake_response(prompt_ids=None)
    assert extract_prompt_token_ids(resp) == []


def test_extract_prompt_token_ids_handles_none_response():
    assert extract_prompt_token_ids(None) == []


def test_extract_prompt_token_ids_coerces_from_floats():
    resp = _fake_response(prompt_ids=[1.0, 2.0])
    assert extract_prompt_token_ids(resp) == [1, 2]


def test_extract_completion_token_ids_attribute():
    resp = _fake_response(completion_ids=[10, 11, 12])
    assert extract_completion_token_ids(resp) == [10, 11, 12]


def test_extract_completion_token_ids_missing_choices():
    resp = SimpleNamespace(choices=[])
    assert extract_completion_token_ids(resp) == []


def test_extract_completion_token_ids_no_attribute_on_choice():
    # A choice object that has no .token_ids attribute returns [].
    choice = SimpleNamespace(logprobs=None)
    resp = SimpleNamespace(choices=[choice])
    assert extract_completion_token_ids(resp) == []


def test_extract_completion_logprobs_full_shape():
    entries = [
        SimpleNamespace(token="a", logprob=-0.1),
        SimpleNamespace(token="b", logprob=-0.2),
    ]
    resp = _fake_response(logprobs_content=entries)
    out = extract_completion_logprobs(resp)
    assert out == [pytest.approx(-0.1), pytest.approx(-0.2)]


def test_extract_completion_logprobs_empty_content():
    resp = _fake_response(logprobs_content=[])
    assert extract_completion_logprobs(resp) == []


def test_extract_completion_logprobs_missing_logprobs():
    resp = _fake_response(logprobs_content=None)
    assert extract_completion_logprobs(resp) == []


def test_extract_completion_logprobs_bails_on_non_numeric():
    # If one entry lacks a numeric .logprob, we return [] (partial
    # extractions are worse than empty — the trainer sanity-checks
    # lengths).
    entries = [
        SimpleNamespace(token="a", logprob=-0.1),
        SimpleNamespace(token="b"),  # missing logprob
    ]
    resp = _fake_response(logprobs_content=entries)
    assert extract_completion_logprobs(resp) == []


def test_extract_completion_logprobs_tolerates_dict_shapes():
    """verifiers' parser accepts logprobs as a dict and entries as dicts;
    the extractor must too, or every summary TrainingSample is dropped
    downstream with 'N ids vs 0 logprobs' while the action path works."""
    from types import SimpleNamespace

    from kv_eviction.summarization import extract_completion_logprobs

    obj_shape = SimpleNamespace(
        choices=[
            SimpleNamespace(
                logprobs=SimpleNamespace(
                    content=[SimpleNamespace(logprob=-0.5), SimpleNamespace(logprob=-1.5)]
                )
            )
        ]
    )
    assert extract_completion_logprobs(obj_shape) == [-0.5, -1.5]

    dict_shape = SimpleNamespace(
        choices=[
            SimpleNamespace(
                logprobs={"content": [{"logprob": -0.25}, {"logprob": -2.0}]}
            )
        ]
    )
    assert extract_completion_logprobs(dict_shape) == [-0.25, -2.0]

    none_shape = SimpleNamespace(choices=[SimpleNamespace(logprobs=None)])
    assert extract_completion_logprobs(none_shape) == []
