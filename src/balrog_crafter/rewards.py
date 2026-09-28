from __future__ import annotations

from typing import Any

DEFAULT_ACHIEVEMENT_COUNT = 22


def _state(kwargs: dict[str, Any]) -> dict[str, Any]:
    state = kwargs.get("state")
    return state if isinstance(state, dict) else {}


def achievement_progress_reward(*_: Any, **kwargs: Any) -> float:
    """Fraction of Crafter achievements unlocked, read from environment state."""

    state = _state(kwargs)
    total = max(1, int(state.get("achievement_count", DEFAULT_ACHIEVEMENT_COUNT)))
    achieved = len(state.get("achievements", {}))
    return max(0.0, min(1.0, achieved / total))


def action_validity_reward(*_: Any, **kwargs: Any) -> float:
    """Valid action rate, useful as a small shaping term during exploration."""

    state = _state(kwargs)
    steps = int(state.get("steps", 0))
    if steps <= 0:
        return 0.0
    invalid = int(state.get("invalid_actions", 0))
    return max(0.0, min(1.0, 1.0 - invalid / steps))


def native_env_reward(*_: Any, **kwargs: Any) -> float:
    """Normalized sum of BALROG/Crafter rewards observed during the episode."""

    state = _state(kwargs)
    total = float(state.get("native_reward_total", 0.0))
    return max(0.0, min(1.0, total / 20.0))


def survival_reward(*_: Any, **kwargs: Any) -> float:
    """Small credit for staying alive until the episode budget is consumed."""

    state = _state(kwargs)
    if not state.get("done"):
        return 1.0
    terminal_info = state.get("terminal_info", {})
    if isinstance(terminal_info, dict) and terminal_info.get("dead"):
        return 0.0
    return 0.5


def verified_crafter_reward(*_: Any, **kwargs: Any) -> float:
    """Single bounded reward used by default for Prime-RL optimization."""

    progress = achievement_progress_reward(**kwargs)
    native = native_env_reward(**kwargs)
    validity = action_validity_reward(**kwargs)
    survival = survival_reward(**kwargs)
    reward = 0.75 * progress + 0.15 * native + 0.08 * validity + 0.02 * survival
    return max(0.0, min(1.0, reward))
