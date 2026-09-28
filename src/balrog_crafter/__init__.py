from .actions import CRAFTER_ACTIONS, extract_action
from .rewards import (
    action_validity_reward,
    achievement_progress_reward,
    native_env_reward,
    survival_reward,
    verified_crafter_reward,
)
from .stats import CRAFTER_METRIC_NAMES, scalar_metrics

__all__ = [
    "BalrogCrafterVerifierEnv",
    "CRAFTER_ACTIONS",
    "CRAFTER_METRIC_NAMES",
    "action_validity_reward",
    "achievement_progress_reward",
    "extract_action",
    "load_environment",
    "load_taskset",
    "native_env_reward",
    "scalar_metrics",
    "survival_reward",
    "verified_crafter_reward",
]


def load_environment(**kwargs):
    from .env import load_environment as _load_environment

    return _load_environment(**kwargs)


def load_taskset(**kwargs):
    from .env import load_taskset as _load_taskset

    return _load_taskset(**kwargs)


def __getattr__(name):
    if name == "BalrogCrafterVerifierEnv":
        from .env import BalrogCrafterVerifierEnv

        return BalrogCrafterVerifierEnv
    raise AttributeError(name)
