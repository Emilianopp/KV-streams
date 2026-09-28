# SPDX-License-Identifier: Apache-2.0
"""Tests for Markovian message truncation with KV-eviction parity."""

from copy import deepcopy

import pytest

from kv_eviction.truncation import (
    kv_eviction_live_turns,
    partition_messages_for_kv_eviction,
    truncate_messages_to_anchor_and_recent_turns,
    truncate_messages_to_last_k_turns,
)


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


def _tool_turn(i):
    return [
        _user(f"u{i}"),
        _assistant(
            content=None,
            tool_calls=[{"id": f"tc{i}", "function": {}}],
        ),
        _tool(f"t{i}"),
        _assistant(f"a{i}"),
    ]


def _history(num_turns, *, pending=False):
    messages = [_sys()]
    for turn in range(1, num_turns + 1):
        messages.extend(_turn(turn))
    if pending:
        messages.append(_user("pending"))
    return messages


def _user_contents(messages):
    return [
        message["content"]
        for message in messages
        if message["role"] == "user"
    ]


def test_partition_matches_kv_message_boundary_pairs():
    messages = [
        _sys(),
        _user("task"),
        _assistant(
            content=None,
            tool_calls=[{"id": "one", "function": {}}],
        ),
        _tool("result-one"),
        _assistant(
            content=None,
            tool_calls=[{"id": "two", "function": {}}],
        ),
        _tool("pending-result"),
    ]

    count, prefix, turns, tail = partition_messages_for_kv_eviction(
        messages
    )

    assert count == 2
    assert prefix == [_sys()]
    assert turns == [
        [messages[1], messages[2]],
        [messages[3], messages[4]],
    ]
    assert tail == [messages[5]]


@pytest.mark.parametrize(
    ("num_turns", "expected"),
    [
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 2),
        (5, 3),
        (6, 2),
        (7, 3),
        (8, 2),
    ],
)
def test_live_turn_count_matches_kv_sawtooth(num_turns, expected):
    assert (
        kv_eviction_live_turns(
            num_turns,
            max_turns=4,
            stride=2,
        )
        == expected
    )


def test_empty_messages_are_unchanged():
    assert truncate_messages_to_last_k_turns([], max_turns=4) == []


def test_below_threshold_returns_same_object():
    messages = _history(2)

    output = truncate_messages_to_last_k_turns(
        messages,
        max_turns=3,
        stride=1,
    )

    assert output is messages


def test_equal_threshold_triggers_eviction():
    messages = _history(3)

    output = truncate_messages_to_last_k_turns(
        messages,
        max_turns=3,
        stride=1,
    )

    assert _user_contents(output) == ["u2", "u3"]


@pytest.mark.parametrize(
    ("num_turns", "expected_users"),
    [
        (3, ["u1", "u2", "u3"]),
        (4, ["u3", "u4"]),
        (5, ["u3", "u4", "u5"]),
        (6, ["u5", "u6"]),
        (7, ["u5", "u6", "u7"]),
    ],
)
def test_truncation_reproduces_sawtooth_window(
    num_turns,
    expected_users,
):
    output = truncate_messages_to_last_k_turns(
        _history(num_turns),
        max_turns=4,
        stride=2,
    )

    assert _user_contents(output) == expected_users


def test_default_stride_evicts_one_turn():
    output = truncate_messages_to_last_k_turns(
        _history(3),
        max_turns=3,
    )

    assert _user_contents(output) == ["u2", "u3"]


def test_stride_equal_to_max_turns_can_reset_to_prefix():
    output = truncate_messages_to_last_k_turns(
        _history(3),
        max_turns=3,
        stride=3,
    )

    assert output == [_sys()]


def test_unmatched_tail_is_always_preserved():
    output = truncate_messages_to_last_k_turns(
        _history(4, pending=True),
        max_turns=4,
        stride=2,
    )

    assert [message["content"] for message in output] == [
        "sys",
        "u3",
        "a3",
        "u4",
        "a4",
        "pending",
    ]


def test_tool_calls_count_like_kv_eviction_instead_of_one_chain():
    messages = [_sys(), _user("task")]
    for index in range(1, 5):
        messages.extend(
            [
                _assistant(
                    content=None,
                    tool_calls=[
                        {"id": f"call-{index}", "function": {}}
                    ],
                ),
                _tool(f"result-{index}"),
            ]
        )

    output = truncate_messages_to_last_k_turns(
        messages,
        max_turns=3,
        stride=1,
    )

    assert [
        (message["role"], message.get("content"))
        for message in output
    ] == [
        ("system", "sys"),
        ("tool", "result-2"),
        ("assistant", None),
        ("tool", "result-3"),
        ("assistant", None),
        ("tool", "result-4"),
    ]


def test_multi_tool_chain_is_not_an_atomic_markovian_turn():
    messages = [
        _sys(),
        _user("task"),
        _assistant(
            content=None,
            tool_calls=[{"id": "one", "function": {}}],
        ),
        _tool("result-one"),
        _assistant(
            content=None,
            tool_calls=[{"id": "two", "function": {}}],
        ),
        _tool("result-two"),
        _assistant("final"),
    ]

    output = truncate_messages_to_last_k_turns(
        messages,
        max_turns=3,
        stride=1,
    )

    assert [
        (message["role"], message.get("content"))
        for message in output
    ] == [
        ("system", "sys"),
        ("tool", "result-one"),
        ("assistant", None),
        ("tool", "result-two"),
        ("assistant", "final"),
    ]


def test_first_message_is_protected_without_a_system_role():
    messages = [
        *_turn(1),
        *_turn(2),
        *_turn(3),
    ]

    output = truncate_messages_to_last_k_turns(
        messages,
        max_turns=2,
        stride=1,
    )

    assert [message["content"] for message in output] == [
        "u1",
        "a2",
        "u3",
        "a3",
    ]


def test_truncation_does_not_mutate_input():
    messages = _history(6, pending=True)
    snapshot = deepcopy(messages)

    truncate_messages_to_last_k_turns(
        messages,
        max_turns=4,
        stride=2,
    )

    assert messages == snapshot


def test_truncation_is_idempotent():
    messages = _history(10, pending=True)

    once = truncate_messages_to_last_k_turns(
        messages,
        max_turns=4,
        stride=2,
    )
    twice = truncate_messages_to_last_k_turns(
        once,
        max_turns=4,
        stride=2,
    )

    assert twice is once


def test_output_never_exceeds_input_length():
    messages = _history(10, pending=True)

    for max_turns in range(1, 15):
        output = truncate_messages_to_last_k_turns(
            messages,
            max_turns=max_turns,
            stride=1,
        )
        assert len(output) <= len(messages)


def test_log_reports_evicted_turns():
    messages = _history(5)
    calls = []

    truncate_messages_to_last_k_turns(
        messages,
        max_turns=4,
        stride=2,
        log_fn=calls.append,
    )

    assert len(calls) == 1
    assert "dropped 2 turns" in calls[0]


def test_log_is_silent_below_threshold():
    messages = _history(2)
    calls = []

    truncate_messages_to_last_k_turns(
        messages,
        max_turns=4,
        stride=2,
        log_fn=calls.append,
    )

    assert calls == []


@pytest.mark.parametrize("max_turns", [0, -1, -100])
def test_nonpositive_max_turns_is_a_noop(max_turns):
    messages = _history(2)
    assert (
        truncate_messages_to_last_k_turns(
            messages,
            max_turns=max_turns,
        )
        is messages
    )


@pytest.mark.parametrize("stride", [0, -1, 4])
def test_invalid_stride_fails_loudly(stride):
    with pytest.raises(ValueError, match="stride must be"):
        truncate_messages_to_last_k_turns(
            _history(3),
            max_turns=3,
            stride=stride,
        )


def test_anchor_and_recent_uses_kv_turn_counter():
    messages = _history(6, pending=True)

    output = truncate_messages_to_anchor_and_recent_turns(
        messages,
        max_turns=5,
        anchor_turns=2,
        recent_turns=2,
    )

    assert [message["content"] for message in output] == [
        "sys",
        "u1",
        "a1",
        "u2",
        "a2",
        "u5",
        "a5",
        "u6",
        "a6",
        "pending",
    ]


def test_anchor_and_recent_triggers_at_equal_threshold():
    messages = _history(6)

    output = truncate_messages_to_anchor_and_recent_turns(
        messages,
        max_turns=6,
        anchor_turns=1,
        recent_turns=1,
    )

    assert _user_contents(output) == ["u1", "u6"]


def test_anchor_and_recent_deduplicates_overlap():
    messages = _history(3)

    output = truncate_messages_to_anchor_and_recent_turns(
        messages,
        max_turns=2,
        anchor_turns=2,
        recent_turns=2,
    )

    assert output is messages


def test_anchor_selection_uses_kv_pairs_inside_tool_chain():
    messages = [_sys(), *_tool_turn(1), *_turn(2), *_turn(3)]

    output = truncate_messages_to_anchor_and_recent_turns(
        messages,
        max_turns=3,
        anchor_turns=1,
        recent_turns=1,
    )

    assert [
        (message["role"], message.get("content"))
        for message in output
    ] == [
        ("system", "sys"),
        ("user", "u1"),
        ("assistant", None),
        ("user", "u3"),
        ("assistant", "a3"),
    ]
