import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vllm"))
sys.path.insert(0, str(ROOT / "prime-rl" / "src"))

from prime_rl.inference.vllm.cache_salt import apply_prime_rl_policy_cache_salt
from vllm.config.cache import CacheConfig
from vllm.v1.core.kv_cache_utils import generate_block_hash_extra_keys
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.compaction.am_runtime import (
    advance_attention_matching_turn_boundary,
    build_attention_matching_turn_plan,
)
from vllm.v1.core.compaction.am_prefix_cache import (
    build_attention_matching_prefix_cache_key,
    build_attention_matching_turn_prefix_cache_replay,
    build_cross_turn_query_seed,
    hash_attention_matching_tokens,
)

from scripts.am_prefix_cache_collision_demo import run_demo


def test_turn_am_plan_protects_prefix_cached_blocks():
    turn_end = 99
    token_ids = []
    turn_ends = []
    for _ in range(7):
        token_ids.extend([1, 2, 3, turn_end])
        turn_ends.append(len(token_ids))

    min_protected = turn_ends[3]
    plan = build_attention_matching_turn_plan(
        num_computed_tokens=len(token_ids),
        synthetic_prefix_len=2,
        token_ids=token_ids,
        max_turns=2,
        keep_recent_turns=1,
        turn_end_token_id=turn_end,
        protect_first_user=True,
        min_protected_prefix_len=min_protected,
    )

    assert plan is not None
    assert plan.protected_prefix_len >= min_protected


def test_turn_boundary_scanner_includes_newline_and_padding():
    turn_end = 99
    filler = 88
    newline = 198
    token_ids = [
        1,
        turn_end,
        filler,
        newline,
        filler,
        filler,
        2,
        turn_end,
        newline,
        filler,
        3,
    ]

    first = advance_attention_matching_turn_boundary(
        token_ids,
        2,
        len(token_ids),
        filler,
    )
    second = advance_attention_matching_turn_boundary(
        token_ids,
        8,
        len(token_ids),
        filler,
    )

    assert first == 6
    assert second == 10


def test_prime_rl_policy_cache_salt_is_versioned_and_idempotent():
    request = SimpleNamespace(request_id="req-1", cache_salt=None)

    apply_prime_rl_policy_cache_salt(request, policy_version=7)
    apply_prime_rl_policy_cache_salt(request, policy_version=7)

    assert request.cache_salt == "prime-rl-policy-step:7"


def test_am_full_prefix_cache_key_salts_all_compacted_blocks():
    request = SimpleNamespace(
        mm_features=[],
        lora_request=None,
        cache_salt=None,
        prompt_embeds=None,
        _prompt_embeds_per_block_hashes={},
        attention_matching_prefix_cache_key="am-key",
        attention_matching_prefix_cache_key_start=32,
        attention_matching_prefix_cache_hash_start=0,
    )

    before_key, _ = generate_block_hash_extra_keys(request, 0, 16, 0)
    boundary_key, _ = generate_block_hash_extra_keys(request, 16, 32, 0)
    after_key, _ = generate_block_hash_extra_keys(request, 32, 48, 0)

    assert before_key == (("attention_matching_prefix_cache", "am-key"),)
    assert boundary_key == (("attention_matching_prefix_cache", "am-key"),)
    assert after_key == (("attention_matching_prefix_cache", "am-key"),)


def test_am_prefix_cache_collision_demo_separates_synthetic_kv():
    result = run_demo(block_size=16)

    assert result["unsafe"]["protected_block_hash_equal"] is True
    assert result["unsafe"]["synthetic_block_hash_equal"] is True
    assert result["unsafe"]["tail_block_hash_equal"] is True
    assert result["unsafe"]["request_b_lookup"] == "request-A synthetic KV"

    assert result["safe_am_keyed"]["protected_block_hash_equal"] is False
    assert result["safe_am_keyed"]["synthetic_block_hash_equal"] is False
    assert result["safe_am_keyed"]["tail_block_hash_equal"] is False
    assert result["safe_am_keyed"]["request_b_lookup"] == "miss"


def test_am_unsafe_prefix_cache_mode_is_explicitly_allowed():
    cfg = CacheConfig(prefix_caching_mode="am_unsafe")

    assert cfg.prefix_caching_mode == "am_unsafe"


def test_cross_turn_am_cache_miss_retries_shallower_replay_before_restore():
    scheduler = SimpleNamespace(block_size=16)
    activated: list[int] = []
    restored: list[str] = []

    def activate(self, request, replay, step_index, original_prompt):
        activated.append(step_index)
        request.attention_matching_cross_turn_replay_index = step_index

    def restore(self, request):
        restored.append(request.request_id)

    scheduler._activate_attention_matching_cross_turn_candidate = MethodType(
        activate, scheduler
    )
    scheduler._restore_attention_matching_cross_turn_candidate = MethodType(
        restore, scheduler
    )

    request = SimpleNamespace(
        request_id="req",
        attention_matching_cross_turn_candidate=True,
        attention_matching_cross_turn_event=SimpleNamespace(
            protected_prefix_len=32,
            synthetic_prefix_len=16,
        ),
        attention_matching_cross_turn_replay=SimpleNamespace(
            steps=(object(), object(), object())
        ),
        attention_matching_cross_turn_replay_index=2,
        attention_matching_original_prompt_token_ids=[1, 2, 3],
        attention_matching_prefix_cache_key="a" * 32,
        num_tokens=128,
    )

    redo_lookup = Scheduler._maybe_finalize_attention_matching_cross_turn_candidate(
        scheduler, request, num_local_computed_tokens=0
    )

    assert redo_lookup is True
    assert activated == [1]
    assert restored == []


def test_cross_turn_am_key_reuses_synthetic_memory_when_tail_grows():
    turn_end = 99
    # Rendered shape: system, user, assistant, user, assistant, ...
    prompt_a = []
    for message_id in range(7):
        prompt_a.extend([message_id, 10 + message_id, turn_end])
    # A new user message has arrived, but it has not completed a new
    # user/assistant pair. The old compacted turn region is unchanged.
    prompt_b = [*prompt_a, 70, 71, turn_end]

    plan_a = build_attention_matching_turn_plan(
        num_computed_tokens=len(prompt_a),
        synthetic_prefix_len=4,
        token_ids=prompt_a,
        max_turns=2,
        keep_recent_turns=1,
        turn_end_token_id=turn_end,
        protect_first_user=True,
    )
    plan_b = build_attention_matching_turn_plan(
        num_computed_tokens=len(prompt_b),
        synthetic_prefix_len=4,
        token_ids=prompt_b,
        max_turns=2,
        keep_recent_turns=1,
        turn_end_token_id=turn_end,
        protect_first_user=True,
    )

    assert plan_a is not None
    assert plan_b is not None
    assert plan_a.protected_prefix_len == plan_b.protected_prefix_len
    assert (
        prompt_a[
            plan_a.protected_prefix_len : plan_a.source_len
            - plan_a.exact_kept_tokens
        ]
        == prompt_b[
            plan_b.protected_prefix_len : plan_b.source_len
            - plan_b.exact_kept_tokens
        ]
    )

    compacted_hash = hash_attention_matching_tokens(
        prompt_a[
            plan_a.protected_prefix_len : plan_a.source_len
            - plan_a.exact_kept_tokens
        ]
    )
    seed = build_cross_turn_query_seed(
        base_seed=0,
        cache_salt="policy:1",
        compacted_tokens_hash=compacted_hash,
        protected_prefix_len=plan_a.protected_prefix_len,
        synthetic_prefix_len=plan_a.synthetic_prefix_len,
        query_source="random_queries",
        max_queries_per_kv_head=128,
        zerobeta=False,
        parent_key=None,
    )
    key_a = build_attention_matching_prefix_cache_key(
        cache_salt="policy:1",
        protected_prefix_len=plan_a.protected_prefix_len,
        synthetic_prefix_len=plan_a.synthetic_prefix_len,
        compacted_tokens_hash=compacted_hash,
        query_source="random_queries",
        max_queries_per_kv_head=128,
        query_seed=seed,
        zerobeta=False,
        parent_key=None,
        parent_key_start=0,
        tail_signature=None,
    )
    key_b = build_attention_matching_prefix_cache_key(
        cache_salt="policy:1",
        protected_prefix_len=plan_b.protected_prefix_len,
        synthetic_prefix_len=plan_b.synthetic_prefix_len,
        compacted_tokens_hash=compacted_hash,
        query_source="random_queries",
        max_queries_per_kv_head=128,
        query_seed=seed,
        zerobeta=False,
        parent_key=None,
        parent_key_start=0,
        tail_signature=None,
    )

    assert key_a == key_b


def test_am_forget_gate_changes_cache_key_only_when_enabled():
    compacted_hash = hash_attention_matching_tokens([1, 2, 3, 4])
    base_kwargs = dict(
        cache_salt="policy:1",
        protected_prefix_len=0,
        synthetic_prefix_len=4,
        compacted_tokens_hash=compacted_hash,
        query_source="random_queries",
        max_queries_per_kv_head=128,
        query_seed=123,
        zerobeta=True,
        parent_key="parent",
        parent_key_start=0,
        tail_signature=None,
    )

    legacy_key = build_attention_matching_prefix_cache_key(**base_kwargs)
    disabled_key = build_attention_matching_prefix_cache_key(
        **base_kwargs,
        forget_gate_enabled=False,
        forget_gate_alpha=0.5,
    )
    gated_key = build_attention_matching_prefix_cache_key(
        **base_kwargs,
        forget_gate_enabled=True,
        forget_gate_alpha=0.5,
    )
    different_alpha_key = build_attention_matching_prefix_cache_key(
        **base_kwargs,
        forget_gate_enabled=True,
        forget_gate_alpha=0.25,
    )

    assert disabled_key == legacy_key
    assert gated_key != legacy_key
    assert different_alpha_key != gated_key


def test_am_prefix_cache_key_and_query_seed_include_position_offset():
    compacted_hash = hash_attention_matching_tokens([1, 2, 3, 4])
    seed_base = dict(
        base_seed=0,
        cache_salt="policy:1",
        compacted_tokens_hash=compacted_hash,
        protected_prefix_len=16,
        synthetic_prefix_len=4,
        query_source="random_queries",
        max_queries_per_kv_head=128,
        zerobeta=True,
        parent_key="parent",
    )

    seed_a = build_cross_turn_query_seed(
        **seed_base,
        position_offset_before=64,
    )
    seed_b = build_cross_turn_query_seed(
        **seed_base,
        position_offset_before=128,
    )

    key_base = dict(
        cache_salt="policy:1",
        protected_prefix_len=16,
        synthetic_prefix_len=4,
        compacted_tokens_hash=compacted_hash,
        query_source="random_queries",
        max_queries_per_kv_head=128,
        zerobeta=True,
        parent_key="parent",
        parent_key_start=16,
        tail_signature=None,
    )
    key_a = build_attention_matching_prefix_cache_key(
        **key_base,
        query_seed=seed_a,
        position_offset_before=64,
        position_offset_after=128,
    )
    key_b = build_attention_matching_prefix_cache_key(
        **key_base,
        query_seed=seed_b,
        position_offset_before=128,
        position_offset_after=192,
    )

    assert seed_a != seed_b
    assert key_a != key_b


def _render_message(message_id: int, turn_end_token_id: int) -> list[int]:
    return [message_id, 1000 + message_id, turn_end_token_id]


def _render_history(
    *,
    completed_turns: int,
    turn_end_token_id: int,
    trailing_user: int | None = None,
) -> list[int]:
    token_ids = _render_message(0, turn_end_token_id)
    for turn in range(completed_turns):
        token_ids.extend(_render_message(10 + 2 * turn, turn_end_token_id))
        token_ids.extend(_render_message(11 + 2 * turn, turn_end_token_id))
    if trailing_user is not None:
        token_ids.extend(_render_message(trailing_user, turn_end_token_id))
    return token_ids


def _build_replay(token_ids: list[int], *, turn_end_token_id: int):
    return build_attention_matching_turn_prefix_cache_replay(
        token_ids=token_ids,
        base_seed=17,
        cache_salt="policy:1",
        synthetic_prefix_len=4,
        max_turns=2,
        keep_recent_turns=1,
        turn_end_token_id=turn_end_token_id,
        turn_padding_token_id=None,
        protect_first_user=True,
        query_source="random_queries",
        max_queries_per_kv_head=128,
        zerobeta=True,
    )


def test_chained_turn_am_replay_uses_prior_compacted_state_as_parent():
    turn_end = 99
    first_compacted = _build_replay(
        _render_history(completed_turns=3, turn_end_token_id=turn_end),
        turn_end_token_id=turn_end,
    )
    tail_grown = _build_replay(
        _render_history(
            completed_turns=3,
            turn_end_token_id=turn_end,
            trailing_user=90,
        ),
        turn_end_token_id=turn_end,
    )
    second_compacted = _build_replay(
        _render_history(completed_turns=5, turn_end_token_id=turn_end),
        turn_end_token_id=turn_end,
    )

    assert first_compacted is not None
    assert tail_grown is not None
    assert second_compacted is not None
    assert len(first_compacted.steps) == 1
    assert len(tail_grown.steps) == 1
    assert len(second_compacted.steps) == 2

    first_key = first_compacted.final_step.prefix_cache_key
    tail_key = tail_grown.final_step.prefix_cache_key
    second_step = second_compacted.final_step

    assert tail_key == first_key
    assert second_step.parent_key == first_key
    assert second_step.prefix_cache_key != first_key
    assert second_compacted.position_offset > first_compacted.position_offset


def test_chained_turn_am_key_differs_from_raw_one_shot_key():
    turn_end = 99
    token_ids = _render_history(completed_turns=5, turn_end_token_id=turn_end)
    replay = _build_replay(token_ids, turn_end_token_id=turn_end)

    assert replay is not None
    assert len(replay.steps) == 2
    final_step = replay.final_step
    plan = build_attention_matching_turn_plan(
        num_computed_tokens=len(token_ids),
        synthetic_prefix_len=4,
        token_ids=token_ids,
        max_turns=2,
        keep_recent_turns=1,
        turn_end_token_id=turn_end,
        protect_first_user=True,
    )
    assert plan is not None

    raw_hash = hash_attention_matching_tokens(
        token_ids[
            plan.protected_prefix_len : plan.source_len - plan.exact_kept_tokens
        ]
    )
    raw_seed = build_cross_turn_query_seed(
        base_seed=17,
        cache_salt="policy:1",
        compacted_tokens_hash=raw_hash,
        protected_prefix_len=plan.protected_prefix_len,
        synthetic_prefix_len=plan.synthetic_prefix_len,
        query_source="random_queries",
        max_queries_per_kv_head=128,
        zerobeta=True,
        parent_key=None,
    )
    raw_one_shot_key = build_attention_matching_prefix_cache_key(
        cache_salt="policy:1",
        protected_prefix_len=plan.protected_prefix_len,
        synthetic_prefix_len=plan.synthetic_prefix_len,
        compacted_tokens_hash=raw_hash,
        query_source="random_queries",
        max_queries_per_kv_head=128,
        query_seed=raw_seed,
        zerobeta=True,
        parent_key=None,
        parent_key_start=0,
        tail_signature=None,
    )

    assert final_step.parent_key is not None
    assert final_step.prefix_cache_key != raw_one_shot_key


def test_chained_turn_am_replay_can_resume_from_cached_parent_key():
    turn_end = 99
    full_replay = _build_replay(
        _render_history(completed_turns=5, turn_end_token_id=turn_end),
        turn_end_token_id=turn_end,
    )

    assert full_replay is not None
    assert len(full_replay.steps) >= 2

    first_step = full_replay.steps[0]
    resumed_replay = build_attention_matching_turn_prefix_cache_replay(
        token_ids=first_step.physical_token_ids_after,
        base_seed=17,
        cache_salt="policy:1",
        synthetic_prefix_len=4,
        max_turns=2,
        keep_recent_turns=1,
        turn_end_token_id=turn_end,
        turn_padding_token_id=None,
        protect_first_user=True,
        query_source="random_queries",
        max_queries_per_kv_head=128,
        zerobeta=True,
        initial_parent_key=first_step.prefix_cache_key,
        initial_parent_key_start=first_step.plan.protected_prefix_len,
        initial_position_offset=first_step.position_offset_after,
    )

    assert resumed_replay is not None
    assert resumed_replay.final_step is not None
    assert (
        resumed_replay.final_step.prefix_cache_key
        == full_replay.steps[1].prefix_cache_key
    )
    assert resumed_replay.final_step.parent_key == first_step.prefix_cache_key
