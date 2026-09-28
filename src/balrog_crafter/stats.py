from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .actions import CRAFTER_ACTIONS
from .rewards import (
    action_validity_reward,
    achievement_progress_reward,
    native_env_reward,
    survival_reward,
    verified_crafter_reward,
)

ACTION_CATEGORIES: dict[str, str] = {
    "Noop": "noop",
    "Move West": "move",
    "Move East": "move",
    "Move North": "move",
    "Move South": "move",
    "Do": "do",
    "Sleep": "sleep",
    "Place Stone": "place",
    "Place Table": "place",
    "Place Furnace": "place",
    "Place Plant": "place",
    "Make Wood Pickaxe": "make",
    "Make Stone Pickaxe": "make",
    "Make Iron Pickaxe": "make",
    "Make Wood Sword": "make",
    "Make Stone Sword": "make",
    "Make Iron Sword": "make",
}

DEFAULT_CATEGORIES: tuple[str, ...] = ("noop", "move", "do", "sleep", "place", "make")


def init_behavior_stats(state: dict[str, Any]) -> dict[str, Any]:
    stats = {
        "episode_id": state.get("episode_id"),
        "seed": state.get("seed"),
        "max_steps": state.get("max_steps"),
        "step_count": 0,
        "done": False,
        "done_reason": "",
        "native_reward_total": 0.0,
        "native_reward_per_step": [],
        "achievement_count": 0,
        "achievement_names": [],
        "achievement_unlock_step": {},
        "action_counts": {action: 0 for action in CRAFTER_ACTIONS},
        "action_category_counts": {category: 0 for category in DEFAULT_CATEGORIES},
        "invalid_action_count": 0,
        "valid_action_count": 0,
        "parse_success_count": 0,
        "output_chars_total": 0,
        "repeated_action_streak_total": 0,
        "repeated_action_streak_max": 0,
        "last_action": None,
        "current_action_streak": 0,
        "observation_count": 0,
        "crafter_score": 0.0,
        "dead": False,
        "health": None,
        "food": None,
        "drink": None,
        "energy": None,
        "inventory": {},
        "steps": [],
    }
    state["crafter_behavior_stats"] = stats
    state["crafter_reward_components"] = reward_components(state)
    return stats


def update_behavior_stats(
    state: dict[str, Any],
    *,
    raw_action: str,
    action: str,
    valid: bool,
    native_reward: float,
    unlocked: list[str],
    stats: dict[str, Any],
    info: dict[str, Any],
    done: bool,
) -> None:
    behavior = state.get("crafter_behavior_stats")
    if not isinstance(behavior, dict):
        behavior = init_behavior_stats(state)
    step = int(state.get("steps", 0))
    behavior["step_count"] = step
    behavior["done"] = bool(done)
    behavior["native_reward_total"] = float(state.get("native_reward_total", 0.0))
    behavior["native_reward_per_step"].append(float(native_reward))
    behavior["observation_count"] = int(behavior.get("observation_count", 0)) + 1
    behavior["output_chars_total"] = int(behavior.get("output_chars_total", 0)) + len(raw_action or "")

    if valid:
        behavior["valid_action_count"] = int(behavior.get("valid_action_count", 0)) + 1
        behavior["parse_success_count"] = int(behavior.get("parse_success_count", 0)) + 1
    else:
        behavior["invalid_action_count"] = int(behavior.get("invalid_action_count", 0)) + 1

    action_counts = behavior.setdefault("action_counts", {known: 0 for known in CRAFTER_ACTIONS})
    action_counts[action] = int(action_counts.get(action, 0)) + 1
    category = ACTION_CATEGORIES.get(action, "other")
    category_counts = behavior.setdefault(
        "action_category_counts", {known: 0 for known in DEFAULT_CATEGORIES}
    )
    category_counts[category] = int(category_counts.get(category, 0)) + 1

    if action == behavior.get("last_action"):
        streak = int(behavior.get("current_action_streak", 0)) + 1
    else:
        streak = 1
    behavior["last_action"] = action
    behavior["current_action_streak"] = streak
    behavior["repeated_action_streak_total"] = int(
        behavior.get("repeated_action_streak_total", 0)
    ) + max(0, streak - 1)
    behavior["repeated_action_streak_max"] = max(
        int(behavior.get("repeated_action_streak_max", 0)), streak
    )

    achievements = sorted(state.get("achievements", {}).keys())
    behavior["achievement_count"] = len(achievements)
    behavior["achievement_names"] = achievements
    unlock_step = behavior.setdefault("achievement_unlock_step", {})
    for achievement in unlocked:
        unlock_step.setdefault(achievement, step)

    behavior["crafter_score"] = _first_number(stats, info, keys=("score", "reward", "total_reward"), default=0.0)
    behavior["dead"] = bool(_first_value(stats, info, keys=("dead", "is_dead"), default=False))
    behavior["health"] = _first_number(stats, info, keys=("health", "hp"), default=None)
    behavior["food"] = _first_number(stats, info, keys=("food",), default=None)
    behavior["drink"] = _first_number(stats, info, keys=("drink", "water"), default=None)
    behavior["energy"] = _first_number(stats, info, keys=("energy",), default=None)
    inventory = _first_value(stats, info, keys=("inventory",), default={})
    behavior["inventory"] = dict(inventory) if isinstance(inventory, dict) else {}
    behavior["done_reason"] = _done_reason(done=done, info=info, behavior=behavior, state=state)

    behavior["steps"].append(
        {
            "step": step,
            "raw_action": raw_action,
            "action": action,
            "valid": valid,
            "native_reward": float(native_reward),
            "unlocked": list(unlocked),
            "done": bool(done),
            "score": behavior["crafter_score"],
        }
    )
    state["crafter_reward_components"] = reward_components(state)


def reward_components(state: dict[str, Any]) -> dict[str, float]:
    return {
        "achievement_progress_reward": achievement_progress_reward(state=state),
        "native_env_reward": native_env_reward(state=state),
        "action_validity_reward": action_validity_reward(state=state),
        "survival_reward": survival_reward(state=state),
        "verified_crafter_reward": verified_crafter_reward(state=state),
    }


def scalar_metrics(state: dict[str, Any]) -> dict[str, float]:
    behavior = state.get("crafter_behavior_stats") or {}
    components = reward_components(state)
    steps = max(1, int(behavior.get("step_count") or state.get("steps", 0) or 0))
    valid = float(behavior.get("valid_action_count", state.get("valid_actions", 0)) or 0.0)
    invalid = float(behavior.get("invalid_action_count", state.get("invalid_actions", 0)) or 0.0)
    total_actions = max(1.0, valid + invalid)
    category_counts = behavior.get("action_category_counts") or {}

    metrics = {
        "crafter/native_reward_total": float(
            behavior.get("native_reward_total", state.get("native_reward_total", 0.0)) or 0.0
        ),
        "crafter/verifier_reward": components["verified_crafter_reward"],
        "crafter/achievement_count": float(
            behavior.get("achievement_count", len(state.get("achievements", {}))) or 0.0
        ),
        "crafter/achievement_progress": components["achievement_progress_reward"],
        "crafter/invalid_action_rate": invalid / total_actions,
        "crafter/valid_action_rate": valid / total_actions,
        "crafter/survival_steps": float(behavior.get("step_count", state.get("steps", 0)) or 0.0),
        "crafter/death_rate": 1.0 if behavior.get("dead") else 0.0,
        "crafter/repeated_action_mean_streak": float(
            behavior.get("repeated_action_streak_total", 0) or 0
        )
        / steps,
        "crafter/repeated_action_max_streak": float(
            behavior.get("repeated_action_streak_max", 0) or 0
        ),
        "crafter/output_parse_success_rate": float(
            behavior.get("parse_success_count", valid) or 0.0
        )
        / total_actions,
        "crafter/output_chars_mean": float(behavior.get("output_chars_total", 0) or 0.0)
        / total_actions,
        "crafter/score": float(behavior.get("crafter_score", 0.0) or 0.0),
        **components,
    }
    for category in DEFAULT_CATEGORIES:
        metrics[f"crafter/{category}_rate"] = float(category_counts.get(category, 0) or 0.0) / total_actions
    return {key: _finite_float(value) for key, value in metrics.items()}


def metric_func(metric_name: str):
    def _metric(*_: Any, **kwargs: Any) -> float:
        state = kwargs.get("state")
        if not isinstance(state, dict):
            return 0.0
        return scalar_metrics(state).get(metric_name, 0.0)

    # Verifiers uses function names as metric keys; keep slashes so Prime-RL
    # logs nested W&B paths like metrics/balrog-crafter/crafter/valid_action_rate.
    _metric.__name__ = metric_name
    return _metric


def persist_episode_stats(state: dict[str, Any]) -> None:
    if os.environ.get("BALROG_CRAFTER_DISABLE_LOCAL_STATS", "").lower() in {"1", "true", "yes"}:
        return
    behavior = state.get("crafter_behavior_stats")
    if not isinstance(behavior, dict):
        return
    stats_dir = _stats_dir()
    stats_dir.mkdir(parents=True, exist_ok=True)
    stem = _file_stem()
    episode_row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **_jsonable(behavior),
        "reward_components": reward_components(state),
        "metrics": scalar_metrics(state),
    }
    _append_jsonl(stats_dir / f"episodes-{stem}.jsonl", episode_row)
    for step in behavior.get("steps", []) or []:
        _append_jsonl(
            stats_dir / f"steps-{stem}.jsonl",
            {
                "timestamp": episode_row["timestamp"],
                "episode_id": behavior.get("episode_id"),
                "seed": behavior.get("seed"),
                **_jsonable(step),
            },
        )


def _stats_dir() -> Path:
    override = os.environ.get("BALROG_CRAFTER_STATS_DIR")
    if override:
        return Path(override).expanduser()
    scratch = os.environ.get("BALROG_CRAFTER_SCRATCH_DIR")
    if scratch:
        return Path(scratch).expanduser() / "stats"
    return Path("/network/scratch/d/dane.malenfant/kv-eviction/outputs/balrog_crafter_stats")


def _file_stem() -> str:
    job_id = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID") or "nojid"
    rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "norank"
    return f"job{job_id}-pid{os.getpid()}-rank{rank}"


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _first_value(*sources: dict[str, Any], keys: tuple[str, ...], default: Any) -> Any:
    for source in sources:
        value = _find_key(source, keys)
        if value is not None:
            return value
    return default


def _first_number(
    *sources: dict[str, Any], keys: tuple[str, ...], default: float | None
) -> float | None:
    value = _first_value(*sources, keys=keys, default=None)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _find_key(value: Any, keys: tuple[str, ...]) -> Any:
    if not isinstance(value, dict):
        return None
    key_set = {key.lower() for key in keys}
    stack = [value]
    while stack:
        current = stack.pop()
        for key, item in current.items():
            if str(key).lower() in key_set:
                return item
            if isinstance(item, dict):
                stack.append(item)
    return None


def _done_reason(*, done: bool, info: dict[str, Any], behavior: dict[str, Any], state: dict[str, Any]) -> str:
    if not done:
        return ""
    if behavior.get("dead"):
        return "dead"
    reason = _first_value(info, keys=("done_reason", "termination_reason", "reason"), default=None)
    if reason:
        return str(reason)
    if int(state.get("steps", 0)) >= int(state.get("max_steps", 0)):
        return "max_steps"
    return "env_done"


def _finite_float(value: Any) -> float:
    out = float(value)
    if out != out or out in {float("inf"), float("-inf")}:
        raise ValueError(f"non-finite Crafter metric: {value!r}")
    return out


CRAFTER_METRIC_NAMES: tuple[str, ...] = tuple(scalar_metrics({}).keys())
