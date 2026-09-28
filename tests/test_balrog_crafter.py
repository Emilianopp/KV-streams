import asyncio
import os
from pathlib import Path

from verifiers.utils.message_utils import maybe_normalize_messages
import verifiers as vf

from balrog_crafter.actions import extract_action
from balrog_crafter.paths import model_label, scratch_dir
from balrog_crafter.rewards import (
    achievement_progress_reward,
    action_validity_reward,
    verified_crafter_reward,
)
from balrog_crafter.stats import CRAFTER_METRIC_NAMES, init_behavior_stats, scalar_metrics, update_behavior_stats


def test_extracts_tagged_action() -> None:
    assert extract_action("I will do this.\n<action>Move North</action>") == "Move North"


def test_extracts_aliases() -> None:
    assert extract_action("Action: craft wooden pickaxe") == "Make Wood Pickaxe"
    assert extract_action("left") == "Move West"
    assert extract_action("noop") == "Noop"


def test_rejects_unknown_action() -> None:
    assert extract_action("teleport home") is None


def test_achievement_progress_is_bounded() -> None:
    state = {"achievements": {"collect_wood": 1, "place_table": 1}, "achievement_count": 4}
    assert achievement_progress_reward(state=state) == 0.5


def test_validity_reward_uses_invalid_count() -> None:
    state = {"steps": 10, "invalid_actions": 2}
    assert action_validity_reward(state=state) == 0.8


def test_verified_reward_is_bounded() -> None:
    state = {
        "achievements": {"collect_wood": 1},
        "achievement_count": 2,
        "steps": 2,
        "invalid_actions": 0,
        "native_reward_total": 3.0,
        "done": False,
    }
    value = verified_crafter_reward(state=state)
    assert 0.0 <= value <= 1.0


def test_scratch_env_override_takes_precedence() -> None:
    path = scratch_dir(
        env={"BALROG_CRAFTER_SCRATCH_DIR": "/tmp/crafter", "USER": "alice.smith"},
        home=Path("/home/alice.smith"),
        exists_fn=lambda _path: True,
    )

    assert path == Path("/tmp/crafter")


def test_model_label_sanitizes_paths_and_hf_ids() -> None:
    assert model_label("/network/scratch/d/dane.malenfant/qwen4b_instruct") == "qwen4b_instruct"
    assert model_label("Qwen/Qwen2.5-1.5B-Instruct") == "qwen2.5-1.5b-instruct"


def test_verifiers_can_load_repo_local_environment() -> None:
    env = vf.load_environment("balrog-crafter", num_episodes=2, max_steps=3, seed=7)
    assert env.env_id == "balrog-crafter"
    assert len(env.dataset) == 2
    assert env.dataset[0]["seed"] == 7
    primary_rubric = env.rubric.rubrics[0]
    names = primary_rubric._get_individual_reward_func_names()
    weights = primary_rubric._get_individual_reward_weights()
    assert weights[names.index("verified_crafter_reward")] == 1.0
    assert weights[names.index("crafter/valid_action_rate")] == 0.0


def test_crafter_scalar_metrics_are_finite() -> None:
    state = {
        "steps": 4,
        "valid_actions": 3,
        "invalid_actions": 1,
        "native_reward_total": 2.0,
        "achievements": {"collect_wood": 1},
        "achievement_count": 4,
        "crafter_behavior_stats": {
            "step_count": 4,
            "valid_action_count": 3,
            "invalid_action_count": 1,
            "native_reward_total": 2.0,
            "achievement_count": 1,
            "action_category_counts": {"noop": 1, "move": 2, "do": 1},
            "parse_success_count": 3,
            "output_chars_total": 40,
            "repeated_action_streak_total": 2,
            "repeated_action_streak_max": 2,
        },
    }
    metrics = scalar_metrics(state)
    assert set(CRAFTER_METRIC_NAMES).issubset(metrics)
    assert metrics["crafter/valid_action_rate"] == 0.75
    assert metrics["crafter/achievement_count"] == 1.0


def test_crafter_behavior_stats_accumulate_across_steps() -> None:
    state = {
        "episode_id": "crafter-0",
        "seed": 0,
        "max_steps": 3,
        "steps": 0,
        "valid_actions": 0,
        "invalid_actions": 0,
        "native_reward_total": 0.0,
        "achievements": {},
    }
    init_behavior_stats(state)
    for step, action in enumerate(["Noop", "Move West"], start=1):
        state["steps"] = step
        state["valid_actions"] += 1
        update_behavior_stats(
            state,
            raw_action=f"<action>{action}</action>",
            action=action,
            valid=True,
            native_reward=0.0,
            unlocked=[],
            stats={},
            info={},
            done=False,
        )

    behavior = state["crafter_behavior_stats"]
    assert len(behavior["steps"]) == 2
    assert behavior["action_counts"]["Noop"] == 1
    assert behavior["action_counts"]["Move West"] == 1


def test_env_response_returns_verifiers_messages(tmp_path: Path) -> None:
    old_stats_dir = os.environ.get("BALROG_CRAFTER_STATS_DIR")
    os.environ["BALROG_CRAFTER_STATS_DIR"] = str(tmp_path)
    try:
        asyncio.run(_check_env_response_returns_verifiers_messages())
    finally:
        if old_stats_dir is None:
            os.environ.pop("BALROG_CRAFTER_STATS_DIR", None)
        else:
            os.environ["BALROG_CRAFTER_STATS_DIR"] = old_stats_dir


async def _check_env_response_returns_verifiers_messages() -> None:
    env = vf.load_environment("balrog-crafter", num_episodes=1, max_steps=2, seed=0)
    state = dict(env.dataset[0])
    state = await env.setup_state(state)
    messages = list(state["prompt"])
    messages.append({"role": "assistant", "content": "<action>Noop</action>"})

    response = await env.env_response(messages, state)
    normalized = maybe_normalize_messages(response, field_name="env_response")

    assert isinstance(normalized, list)
    assert normalized[0]["role"] == "user"
    await env.cleanup_crafter(state)
