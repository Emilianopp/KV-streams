import json

import kv_eviction.env as env


class _Response:
    def __init__(self, compaction_events):
        self.compaction_events = compaction_events


def test_record_managed_context_archive_derives_writer_len() -> None:
    state = {}
    response = _Response(
        [
            {
                "num_output_tokens_at_compaction": 0,
                "tokens_evicted": 2,
                "position_offset_after": 2,
                "evict_start": 1,
                "evicted_token_ids": [20, 21],
                "kept_token_ids": [10, 30, 31],
                "archived_span_ids": ["T0001"],
            }
        ]
    )

    assert env._record_managed_context_archive_events(response, state) == ["T0001"]
    row = state["managed_context_archive_index"][0]
    assert row["writer_len_at_compaction"] == 5


def test_record_managed_context_archive_offsets_visible_writer_len() -> None:
    state = {}
    response = _Response(
        [
            {
                "num_output_tokens_at_compaction": 0,
                "tokens_evicted": 2,
                "position_offset_after": 2,
                "evict_start": 1,
                "evicted_token_ids": [20, 21],
                "kept_token_ids": [10, 30, 31],
                "writer_len_at_compaction": 5,
                "archived_span_ids": ["T0001"],
            },
            {
                "num_output_tokens_at_compaction": 0,
                "tokens_evicted": 3,
                "position_offset_after": 3,
                "evict_start": 1,
                "evicted_token_ids": [40, 41, 42],
                "kept_token_ids": [10, 30],
                "writer_len_at_compaction": 6,
                "archived_span_ids": ["T0002"],
            },
        ]
    )

    assert env._record_managed_context_archive_events(response, state) == [
        "T0001",
        "T0002",
    ]
    rows = state["managed_context_archive_index"]
    assert rows[0]["writer_len_at_compaction"] == 5
    assert rows[1]["writer_len_at_compaction"] == 7


def test_managed_context_replay_spans_require_evicted_token_ids() -> None:
    state = {
        "managed_context_archive_index": [
            {
                "span_id": "T0001",
                "evict_start": 160,
                "tokens_evicted": 2,
                "evicted_token_ids": [11, 12],
                "writer_len_at_compaction": 20,
                "original_turn_start": 0,
                "original_turn_end": 1,
            },
            {
                "span_id": "T0002",
                "evict_start": 160,
                "tokens_evicted": 2,
                "evicted_token_ids": [],
                "writer_len_at_compaction": 22,
                "original_turn_start": 2,
                "original_turn_end": 3,
            },
        ]
    }

    assert env._managed_context_replay_spans_for_restore(["T0002"], state) == []

    assert env._managed_context_replay_spans_for_restore(["T0001"], state) == [
        {
            "span_id": "T0001",
            "evict_start": 160,
            "tokens_evicted": 2,
            "evicted_token_ids": [11, 12],
            "writer_len_at_compaction": 20,
            "original_turn_start": 0,
            "original_turn_end": 1,
        }
    ]


def test_managed_context_replay_spans_include_dependencies_by_default() -> None:
    state = {
        "managed_context_archive_index": [
            {
                "span_id": "T0001",
                "evict_start": 160,
                "tokens_evicted": 672,
                "evicted_token_ids": [1] * 672,
                "writer_len_at_compaction": 2400,
                "original_turn_start": 0,
                "original_turn_end": 1,
            },
            {
                "span_id": "T0002",
                "evict_start": 160,
                "tokens_evicted": 304,
                "evicted_token_ids": [2] * 304,
                "writer_len_at_compaction": 3328,
                "original_turn_start": 2,
                "original_turn_end": 3,
            },
        ]
    }

    replay_rows = env._managed_context_replay_spans_for_restore(
        ["T0002"], state
    )

    assert [row["span_id"] for row in replay_rows] == ["T0001", "T0002"]


def test_managed_context_replay_spans_dependency_opt_out(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "KVE_MANAGED_CONTEXT_REPLAY_XARGS_INCLUDE_DEPENDENCIES", "0"
    )
    state = {
        "managed_context_archive_index": [
            {
                "span_id": "T0001",
                "evict_start": 160,
                "tokens_evicted": 2,
                "evicted_token_ids": [11, 12],
                "writer_len_at_compaction": 20,
                "original_turn_start": 0,
                "original_turn_end": 1,
            },
            {
                "span_id": "T0002",
                "evict_start": 160,
                "tokens_evicted": 2,
                "evicted_token_ids": [13, 14],
                "writer_len_at_compaction": 24,
                "original_turn_start": 2,
                "original_turn_end": 3,
            },
        ]
    }

    replay_rows = env._managed_context_replay_spans_for_restore(
        ["T0002"], state
    )

    assert [row["span_id"] for row in replay_rows] == ["T0002"]


def test_attach_managed_context_replay_spans_sets_xargs() -> None:
    state = {
        "managed_context_archive_index": [
            {
                "span_id": "T0001",
                "evict_start": 160,
                "tokens_evicted": 2,
                "evicted_token_ids": [11, 12],
                "writer_len_at_compaction": 20,
                "original_turn_start": 0,
                "original_turn_end": 1,
            }
        ]
    }
    xargs = {"other": 1}

    env._attach_managed_context_replay_spans(xargs, ["T0001"], state)

    assert xargs["other"] == 1
    replay_spans = json.loads(xargs["kve_compact_replay_spans"])
    assert [row["span_id"] for row in replay_spans] == ["T0001"]


def test_attach_managed_context_replay_spans_clears_xargs_when_missing() -> None:
    xargs = {"kve_compact_replay_spans": "old"}

    env._attach_managed_context_replay_spans(xargs, ["missing"], {})

    assert "kve_compact_replay_spans" not in xargs
