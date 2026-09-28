from __future__ import annotations

from typing import Any

import verifiers as vf
from datasets import Dataset

from .actions import CRAFTER_ACTIONS, extract_action
from .adapter import BalrogCrafterAdapter, achievement_counts, image_to_data_url, observation_to_text
from .rewards import (
    DEFAULT_ACHIEVEMENT_COUNT,
    action_validity_reward,
    achievement_progress_reward,
    verified_crafter_reward,
)
from .stats import (
    CRAFTER_METRIC_NAMES,
    init_behavior_stats,
    metric_func,
    persist_episode_stats,
    update_behavior_stats,
)

SYSTEM_PROMPT = """You are controlling BALROG Crafter.
Choose exactly one valid action per turn and put it in <action>...</action>.
Do not describe a plan after the action.

Valid actions:
{actions}
"""


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(str(part.get("text", "")))
        return "\n".join(text_parts)
    return str(content)


def _last_assistant_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "assistant":
            return _content_text(message.get("content", ""))
    return ""


def _user_message(text: str, image_url: str | None = None) -> dict[str, Any]:
    if image_url is None:
        return {"role": "user", "content": text}
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    }


def _build_dataset(num_episodes: int, seed: int, max_steps: int) -> Dataset:
    rows = []
    for index in range(num_episodes):
        rows.append(
            {
                "episode_id": f"crafter-{seed + index}",
                "seed": seed + index,
                "max_steps": max_steps,
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT.format(actions="\n".join(CRAFTER_ACTIONS))},
                    {"role": "user", "content": "Start the Crafter episode."},
                ],
            }
        )
    return Dataset.from_list(rows)


class BalrogCrafterVerifierEnv(vf.MultiTurnEnv):
    """Prime-RL compatible Verifiers environment for BALROG Crafter."""

    def __init__(
        self,
        *,
        num_episodes: int = 64,
        seed: int = 0,
        max_steps: int = 100,
        task: str = "crafter",
        use_images: bool = False,
        invalid_action_fallback: str = "Noop",
        env_kwargs: dict[str, Any] | None = None,
        include_auxiliary_rewards: bool = False,
        **kwargs: Any,
    ) -> None:
        self.max_steps = max_steps
        self.task = task
        self.use_images = use_images
        self.invalid_action_fallback = invalid_action_fallback
        self.env_kwargs = env_kwargs or {}

        reward_funcs = [verified_crafter_reward]
        reward_weights = [1.0]
        if include_auxiliary_rewards:
            reward_funcs.extend([achievement_progress_reward, action_validity_reward])
            reward_weights.extend([0.0, 0.0])
        for metric_name in CRAFTER_METRIC_NAMES:
            reward_funcs.append(metric_func(metric_name))
            reward_weights.append(0.0)

        super().__init__(
            dataset=_build_dataset(num_episodes=num_episodes, seed=seed, max_steps=max_steps),
            rubric=vf.Rubric(funcs=reward_funcs, weights=reward_weights),
            **kwargs,
        )

    def _render_observation(self, observation: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        text = observation_to_text(observation)
        stats = state.get("crafter_stats", {})
        achievements = sorted(state.get("achievements", {}).keys())
        steps = int(state.get("steps", 0))
        invalid = int(state.get("invalid_actions", 0))
        lines = [
            f"Step: {steps}/{self.max_steps}",
            f"Invalid actions so far: {invalid}",
            "Observation:",
            text,
        ]
        if achievements:
            lines.extend(["Unlocked achievements:", ", ".join(achievements)])
        if isinstance(stats, dict) and stats.get("score") is not None:
            lines.append(f"BALROG score: {stats['score']}")

        image_url = image_to_data_url(observation.get("image")) if self.use_images else None
        return _user_message("\n".join(lines), image_url=image_url)

    def _start_adapter(self, state: dict[str, Any]) -> BalrogCrafterAdapter:
        return BalrogCrafterAdapter(
            seed=state.get("seed"),
            max_episode_steps=int(state.get("max_steps", self.max_steps)),
            task=self.task,
            env_kwargs=self.env_kwargs,
        )

    async def setup_state(self, state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        maybe_state = await super().setup_state(state, **kwargs)
        if isinstance(maybe_state, dict):
            state = maybe_state

        adapter = self._start_adapter(state)
        observation = adapter.reset()
        stats = adapter.get_stats()
        achievements = achievement_counts(stats)

        state.update(
            {
                "crafter_adapter": adapter,
                "crafter_stats": stats,
                "achievements": achievements,
                "achievement_count": DEFAULT_ACHIEVEMENT_COUNT,
                "steps": 0,
                "invalid_actions": 0,
                "valid_actions": 0,
                "native_reward_total": 0.0,
                "done": False,
                "terminal_info": {},
                "action_trace": [],
            }
        )
        init_behavior_stats(state)
        state["prompt"] = [
            {"role": "system", "content": SYSTEM_PROMPT.format(actions="\n".join(CRAFTER_ACTIONS))},
            self._render_observation(observation, state),
        ]
        return state

    async def env_response(
        self,
        messages: list[dict[str, Any]],
        state: dict[str, Any],
        **_: Any,
    ) -> list[dict[str, Any]]:
        adapter = state["crafter_adapter"]
        raw_action = _last_assistant_text(messages)
        action = extract_action(raw_action)

        valid = action is not None
        if not valid:
            action = self.invalid_action_fallback
            state["invalid_actions"] = int(state.get("invalid_actions", 0)) + 1
        else:
            state["valid_actions"] = int(state.get("valid_actions", 0)) + 1

        observation, native_reward, done, info = adapter.step(action)
        stats = adapter.get_stats()
        achievements = achievement_counts(stats, info)

        previous = set(state.get("achievements", {}))
        current = set(achievements)
        unlocked = sorted(current - previous)

        state["steps"] = int(state.get("steps", 0)) + 1
        state["native_reward_total"] = float(state.get("native_reward_total", 0.0)) + native_reward
        state["crafter_stats"] = stats
        state["achievements"] = achievements
        state["done"] = bool(done or state["steps"] >= self.max_steps)
        if state["done"]:
            state["terminal_info"] = info
        state["action_trace"].append(
            {
                "raw": raw_action,
                "action": action,
                "valid": valid,
                "native_reward": native_reward,
                "unlocked": unlocked,
            }
        )
        update_behavior_stats(
            state,
            raw_action=raw_action,
            action=action,
            valid=valid,
            native_reward=native_reward,
            unlocked=unlocked,
            stats=stats,
            info=info,
            done=state["done"],
        )

        feedback = [
            f"Executed action: {action}",
            f"Action valid: {valid}",
            f"Native reward: {native_reward:g}",
        ]
        if unlocked:
            feedback.append(f"New achievements: {', '.join(unlocked)}")
        if state["done"]:
            feedback.append("Episode finished.")
        else:
            feedback.append("Continue with exactly one <action>...</action>.")

        response = self._render_observation(observation, state)
        if isinstance(response["content"], str):
            response["content"] = "\n".join(feedback + ["", response["content"]])
        elif isinstance(response["content"], list):
            response["content"][0]["text"] = "\n".join(feedback + ["", response["content"][0]["text"]])

        return [vf.UserMessage(content=response["content"])]

    @vf.stop
    async def episode_done(self, state: dict[str, Any], **_: Any) -> bool:
        return bool(state.get("done"))

    @vf.cleanup
    async def cleanup_crafter(self, state: dict[str, Any], **_: Any) -> None:
        persist_episode_stats(state)
        adapter = state.pop("crafter_adapter", None)
        if adapter is not None:
            adapter.close()


def load_environment(**kwargs: Any) -> BalrogCrafterVerifierEnv:
    return BalrogCrafterVerifierEnv(**kwargs)


def load_taskset(**kwargs: Any) -> BalrogCrafterVerifierEnv:
    return load_environment(**kwargs)
