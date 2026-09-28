"""Variation-budget utilities for continual deviation experiments.

The World's Edge paper defines a variation budget over changes in the induced
transition kernels and rewards across episodes. In a stationary single-agent
benchmark such as Continuing Swimmer, those environment-side terms should be
approximately zero. To still capture update-driven drift, we also record a
policy-side budget on a fixed anchor observation bank.
"""

from __future__ import annotations

from typing import NamedTuple


try:
    import jax.numpy as xp
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly in tests
    import numpy as xp


class VariationBudgetStats(NamedTuple):
    policy_variation: object
    reward_variation: object
    kernel_variation: object
    total_variation: object
    cumulative_variation: object


def _reduce(values, reduction: str):
    if reduction == "max":
        return xp.max(values)
    if reduction == "mean":
        return xp.mean(values)
    raise ValueError(f"Unsupported reduction {reduction!r}")


def categorical_total_variation(prev_probs, curr_probs, reduction: str = "max"):
    """Empirical total variation for batched categorical distributions."""

    per_anchor = 0.5 * xp.sum(xp.abs(curr_probs - prev_probs), axis=-1)
    return _reduce(per_anchor, reduction)


def _diagonal_gaussian_kl(prev_mean, prev_log_std, curr_mean, curr_log_std):
    prev_var = xp.exp(2.0 * prev_log_std)
    curr_var = xp.exp(2.0 * curr_log_std)
    mean_sq = xp.square(prev_mean - curr_mean)
    term = (prev_var + mean_sq) / curr_var
    return 0.5 * xp.sum(term - 1.0 + 2.0 * (curr_log_std - prev_log_std), axis=-1)


def gaussian_pinsker_tv_proxy(
    prev_mean,
    prev_log_std,
    curr_mean,
    curr_log_std,
    reduction: str = "max",
):
    """A TV-like proxy for diagonal Gaussian policies via symmetrized KL.

    This is not the exact total variation distance. It is a practical proxy for
    continuous-action PPO policies on a fixed observation bank.
    """

    kl_forward = _diagonal_gaussian_kl(
        prev_mean, prev_log_std, curr_mean, curr_log_std
    )
    kl_backward = _diagonal_gaussian_kl(
        curr_mean, curr_log_std, prev_mean, prev_log_std
    )
    sym_kl = 0.5 * (kl_forward + kl_backward)
    per_anchor = xp.sqrt(0.5 * xp.maximum(sym_kl, 0.0))
    return _reduce(per_anchor, reduction)


def reward_variation(prev_rewards, curr_rewards, reduction: str = "max"):
    """Empirical reward-drift term on an anchor batch."""

    deltas = xp.abs(curr_rewards - prev_rewards)
    return _reduce(deltas, reduction)


def deterministic_action_variation(
    prev_actions,
    curr_actions,
    reduction: str = "max",
):
    """Policy-side drift for deterministic actors such as AC-PQN."""

    per_anchor = xp.sqrt(xp.sum(xp.square(curr_actions - prev_actions), axis=-1))
    return _reduce(per_anchor, reduction)


def update_variation_budget(
    cumulative_variation,
    *,
    policy_variation=None,
    reward_variation_term=None,
    kernel_variation=None,
):
    """Combine variation terms into a total and cumulative budget."""

    zero = xp.asarray(0.0)
    policy = zero if policy_variation is None else policy_variation
    reward = zero if reward_variation_term is None else reward_variation_term
    kernel = zero if kernel_variation is None else kernel_variation
    total = policy + reward + kernel
    cumulative = cumulative_variation + total
    return VariationBudgetStats(
        policy_variation=policy,
        reward_variation=reward,
        kernel_variation=kernel,
        total_variation=total,
        cumulative_variation=cumulative,
    )
