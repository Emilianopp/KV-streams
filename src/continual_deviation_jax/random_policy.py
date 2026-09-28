"""Helpers for random-action sanity baselines."""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


try:  # pragma: no cover - exercised only when JAX is installed
    import jax
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - numpy fallback is tested
    jax = None
    jnp = None


class RandomPolicyStats(NamedTuple):
    action_space: str
    sample_shape: tuple[int, ...]
    low: float | None
    high: float | None


def sample_random_actions(
    *,
    action_space: str,
    sample_shape: tuple[int, ...],
    rng=None,
    action_dim: int | None = None,
    num_actions: int | None = None,
    low: float = -1.0,
    high: float = 1.0,
):
    """Sample random actions for continuous or discrete benchmarks.

    If `rng` is a JAX PRNGKey and JAX is installed, sampling stays on device.
    Otherwise the function falls back to NumPy sampling.
    """

    if action_space == "continuous":
        if action_dim is None:
            raise ValueError("Continuous random actions require action_dim.")
        shape = tuple(sample_shape) + (int(action_dim),)
        if jax is not None and rng is not None and hasattr(rng, "shape"):
            actions = jax.random.uniform(
                rng, shape=shape, minval=low, maxval=high
            )
        else:
            generator = np.random.default_rng(rng)
            actions = generator.uniform(low, high, size=shape)
        return actions, RandomPolicyStats(
            action_space=action_space,
            sample_shape=shape,
            low=low,
            high=high,
        )

    if action_space == "discrete":
        if num_actions is None:
            raise ValueError("Discrete random actions require num_actions.")
        shape = tuple(sample_shape)
        if jax is not None and rng is not None and hasattr(rng, "shape"):
            actions = jax.random.randint(
                rng, shape=shape, minval=0, maxval=int(num_actions)
            )
        else:
            generator = np.random.default_rng(rng)
            actions = generator.integers(0, int(num_actions), size=shape)
        return actions, RandomPolicyStats(
            action_space=action_space,
            sample_shape=shape,
            low=None,
            high=None,
        )

    raise ValueError(f"Unsupported action_space {action_space!r}")

