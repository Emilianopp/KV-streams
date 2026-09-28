from __future__ import annotations

import base64
import io
from typing import Any


class BalrogCrafterAdapter:
    """Thin compatibility layer around BALROG's CrafterLanguageWrapper."""

    def __init__(
        self,
        *,
        seed: int | None,
        max_episode_steps: int,
        task: str = "crafter",
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        try:
            import crafter
            from balrog.environments.crafter.env import CrafterLanguageWrapper
        except Exception as exc:  # pragma: no cover - only reached without optional deps.
            raise RuntimeError(
                "BALROG Crafter dependencies are not installed in this environment. "
                "Install BALROG/Crafter before running rollouts, e.g. "
                "`pip install 'balrog @ git+https://github.com/balrog-ai/BALROG.git'`."
            ) from exc

        self._env = crafter.Env(**(env_kwargs or {}))
        if seed is not None and hasattr(self._env, "seed"):
            self._env.seed(seed)
        self._wrapped = CrafterLanguageWrapper(
            self._env,
            task=task,
            max_episode_steps=max_episode_steps,
        )

    def reset(self) -> dict[str, Any]:
        obs = self._wrapped.reset()
        return obs[0] if isinstance(obs, tuple) else obs

    def step(self, action: str) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        result = self._wrapped.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, float(reward), bool(terminated or truncated), dict(info)
        obs, reward, done, info = result
        return obs, float(reward), bool(done), dict(info)

    def get_stats(self) -> dict[str, Any]:
        if hasattr(self._wrapped, "get_stats"):
            return dict(self._wrapped.get_stats())
        return {}

    def close(self) -> None:
        for env in (getattr(self, "_wrapped", None), getattr(self, "_env", None)):
            if hasattr(env, "close"):
                env.close()


def observation_to_text(observation: dict[str, Any]) -> str:
    text_obs = observation.get("text", observation)
    if isinstance(text_obs, str):
        return text_obs
    if not isinstance(text_obs, dict):
        return str(text_obs)

    parts: list[str] = []
    for key in ("long_term_context", "short_term_context"):
        value = text_obs.get(key)
        if value:
            parts.append(str(value).strip())
    if not parts:
        parts.extend(str(value).strip() for value in text_obs.values() if value)
    return "\n\n".join(part for part in parts if part)


def image_to_data_url(image: Any) -> str | None:
    if image is None:
        return None

    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    except Exception:
        return None

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def achievement_counts(stats: dict[str, Any], info: dict[str, Any] | None = None) -> dict[str, int]:
    candidates = []
    if info:
        candidates.extend(
            [
                info.get("achievements"),
                info.get("achievement"),
                info.get("unlocked_achievements"),
            ]
        )
    candidates.extend(
        [
            stats.get("achievements"),
            stats.get("achievement"),
            stats.get("unlocked_achievements"),
        ]
    )

    for candidate in candidates:
        if isinstance(candidate, dict):
            return {str(key): int(value) for key, value in candidate.items() if int(value) > 0}
        if isinstance(candidate, (list, tuple, set)):
            return {str(key): 1 for key in candidate}

    return {}
