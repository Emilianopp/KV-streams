# SPDX-License-Identifier: Apache-2.0
"""Tests for vLLM KV-eviction event plumbing through verifiers extras."""

from types import SimpleNamespace

from kv_eviction.env import (
    attach_compaction_events_from_response,
    attach_compaction_metrics_to_state,
)


def test_attach_response_events_merges_compaction_and_am_metadata():
    response = SimpleNamespace(
        compaction_events=[
            {
                "num_output_tokens_at_compaction": 10,
                "tokens_evicted": 512,
                "position_offset_after": 512,
                "attention_matching_selected_indices": [[[0, 1], [2, 3]]],
                "attention_matching_replay_steps": [
                    {
                        "source_len": 64,
                        "target_len": 32,
                        "protected_prefix_len": 8,
                        "synthetic_prefix_len": 2,
                        "exact_kept_tokens": 22,
                        "attention_matching_query_seed": 123,
                        "attention_matching_selected_indices": [[[0, 1]]],
                    }
                ],
                "attention_matching_hidden_tail_token_ids": ["151645", 151643],
            }
        ],
        shuffle_events=[
            {
                "num_output_tokens_at_shuffle": "3",
                "chunk_index": 1,
                "chunk_start": 64,
                "chunk_end": 128,
            }
        ],
        noise_events=[
            SimpleNamespace(
                num_output_tokens_at_noise=4,
                chunk_index=2,
                chunk_start=128,
                chunk_end=192,
                target="values",
                std="0.05",
            )
        ],
    )
    step = {}

    attach_compaction_events_from_response(step, response)

    event = step["extras"]["compaction_events"][0]
    expected_base = {
        "num_output_tokens_at_compaction": 10,
        "tokens_evicted": 512,
        "position_offset_after": 512,
        "num_prompt_tokens": 0,
        "evict_start": 0,
    }
    for key, value in expected_base.items():
        assert event[key] == value
    assert event["compaction_strategy"] == "fifo"
    assert event["attention_matching_pre_sample"] is False
    assert event["attention_matching_selected_indices"] == [[[0, 1], [2, 3]]]
    assert event["attention_matching_replay_steps"][0][
        "attention_matching_selected_indices"
    ] == [[[0, 1]]]
    assert event["attention_matching_hidden_tail_token_ids"] == [151645, 151643]
    assert step["extras"]["shuffle_events"] == [
        {
            "num_output_tokens_at_shuffle": 3,
            "chunk_index": 1,
            "chunk_start": 64,
            "chunk_end": 128,
        }
    ]
    assert step["extras"]["noise_events"] == [
        {
            "num_output_tokens_at_noise": 4,
            "chunk_index": 2,
            "chunk_start": 128,
            "chunk_end": 192,
            "target": "values",
            "std": 0.05,
        }
    ]


def test_attach_metrics_counts_all_kv_eviction_event_families():
    state = {
        "trajectory": [
            {
                "extras": {
                    "compaction_events": [[10, 512, 512]],
                    "shuffle_events": [[3, 1, 64, 128]],
                    "noise_events": [[4, 2, 128, 192, "values", "0.05"]],
                }
            }
        ]
    }

    attach_compaction_metrics_to_state(state)

    assert state["num_compaction_events"] == 1
    assert state["num_shuffle_events"] == 1
    assert state["num_noise_events"] == 1
    assert state["metrics"] == {
        "num_compaction_events": 1.0,
        "num_shuffle_events": 1.0,
        "num_noise_events": 1.0,
    }
