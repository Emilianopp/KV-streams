from prime_rl.orchestrator.trajectories import interleave_rollout
from prime_rl.transport.types import CompactionEventWire


def _am_event(
    *,
    source_len: int,
    target_len: int,
    tokens_evicted: int,
    position_offset_after: int,
    replay_steps: list[dict],
    num_prompt_tokens: int | None = None,
    cache_hit_tokens: int = 0,
    hidden_tail_token_ids: list[int] | None = None,
) -> CompactionEventWire:
    return CompactionEventWire(
        num_output_tokens_at_compaction=0,
        tokens_evicted=tokens_evicted,
        position_offset_after=position_offset_after,
        num_prompt_tokens=(
            source_len if num_prompt_tokens is None else num_prompt_tokens
        ),
        compaction_strategy="attention_matching",
        source_len=source_len,
        target_len=target_len,
        protected_prefix_len=800,
        synthetic_prefix_len=16,
        exact_kept_tokens=target_len - 816,
        attention_matching_query_source="random_queries",
        attention_matching_max_queries_per_kv_head=8,
        attention_matching_query_seed=123,
        attention_matching_zerobeta=True,
        attention_matching_pre_sample=True,
        attention_matching_replay_steps=replay_steps,
        attention_matching_cache_hit_tokens=cache_hit_tokens,
        attention_matching_hidden_tail_token_ids=hidden_tail_token_ids,
    )


def _step(
    prompt_ids: list[int],
    completion_ids: list[int],
    events: list[CompactionEventWire],
):
    return {
        "tokens": {
            "prompt_ids": prompt_ids,
            "prompt_mask": [False] * len(prompt_ids),
            "completion_ids": completion_ids,
            "completion_mask": [True] * len(completion_ids),
            "completion_logprobs": [-0.5] * len(completion_ids),
        },
        "extras": {"compaction_events": events},
    }


def test_stateful_am_prompt_replay_splits_prompt_at_vllm_source_boundary():
    step_a = {
        "source_len": 1000,
        "target_len": 900,
        "protected_prefix_len": 800,
        "synthetic_prefix_len": 16,
        "exact_kept_tokens": 84,
        "attention_matching_query_seed": 11,
        "attention_matching_prefix_cache_key": "am-key-a",
        "attention_matching_selected_indices": [[[0] * 16]],
    }
    step_b = {
        "source_len": 970,
        "target_len": 950,
        "protected_prefix_len": 800,
        "synthetic_prefix_len": 16,
        "exact_kept_tokens": 134,
        "attention_matching_query_seed": 12,
        "attention_matching_prefix_cache_key": "am-key-b",
        "attention_matching_selected_indices": [[[0] * 16]],
    }

    prompt0 = list(range(10_000, 11_000))
    completion0 = list(range(20_000, 20_050))
    hidden_tail = [151643] * 10
    # The next rendered prompt extends the visible conversation, but not the
    # hidden tail tokens that vLLM used only to close a prefix-cache block.
    prompt1 = prompt0 + completion0 + list(range(30_000, 30_030))
    completion1 = list(range(40_000, 40_010))

    output = {
        "example_id": "stateful-am-boundary",
        "trajectory": [
            _step(
                prompt0,
                completion0,
                [
                    _am_event(
                        source_len=len(prompt0),
                        target_len=900,
                        tokens_evicted=100,
                        position_offset_after=100,
                        replay_steps=[step_a],
                        hidden_tail_token_ids=hidden_tail,
                    )
                ],
            ),
            _step(
                prompt1,
                completion1,
                [
                    _am_event(
                        source_len=len(prompt1),
                        target_len=980,
                        tokens_evicted=100,
                        position_offset_after=100,
                        replay_steps=[step_a],
                        cache_hit_tokens=900,
                    ),
                    _am_event(
                        source_len=980,
                        target_len=960,
                        tokens_evicted=20,
                        position_offset_after=120,
                        replay_steps=[step_b],
                        num_prompt_tokens=len(prompt1),
                    ),
                ],
            ),
        ],
        "error": None,
        "sampling_args": {"temperature": 1.0},
        "tool_defs": [],
    }

    samples = interleave_rollout(output)
    assert samples is not None
    assert len(samples) == 1
    sample = samples[0]
    events = sample.compaction_events or []
    assert len(events) == 2

    # The second event must fire before the whole new prompt suffix has been
    # appended. Old behavior compacted at 90 and preserved ten hidden-tail
    # tokens through AM; vLLM compacted at 80 and warmed those tokens after AM.
    assert events[1].num_output_tokens_at_compaction == 80
    assert events[1].source_len == 980
    assert events[1].target_len == 960
    assert sum(sample.completion_mask[:80]) == 50
    assert sum(sample.completion_mask[80:90]) == 0
