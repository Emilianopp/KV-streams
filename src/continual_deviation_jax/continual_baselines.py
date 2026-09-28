"""Continual-learning baseline helpers for the JAX scaffold."""

from __future__ import annotations

from typing import NamedTuple

from .config import CLEARConfig, OnlineEWCConfig, PolicyConsolidationConfig


try:
    import jax.numpy as xp
except ModuleNotFoundError:  # pragma: no cover - exercised indirectly in tests
    import numpy as xp


class OnlineEWCStats(NamedTuple):
    penalty: object


class CLEARStats(NamedTuple):
    total_loss: object
    replay_weight: object
    policy_clone_loss: object
    value_clone_loss: object


class PolicyConsolidationStats(NamedTuple):
    policy_penalty: object
    value_penalty: object
    total_penalty: object


def online_ewc_penalty(
    current_params,
    reference_params,
    fisher_diag,
    config: OnlineEWCConfig,
):
    """Diagonal-Fisher quadratic penalty."""

    if not config.enabled:
        zero = xp.asarray(0.0)
        return zero, OnlineEWCStats(penalty=zero)
    delta = current_params - reference_params
    penalty = 0.5 * config.penalty_weight * xp.sum(fisher_diag * xp.square(delta))
    return penalty, OnlineEWCStats(penalty=penalty)


def clear_loss(
    *,
    on_policy_loss,
    replay_rl_loss,
    policy_clone_loss,
    value_clone_loss,
    config: CLEARConfig,
):
    """CLEAR-style combined loss."""

    if not config.enabled:
        replay_weight = xp.asarray(0.0, dtype=xp.asarray(on_policy_loss).dtype)
        total = xp.asarray(on_policy_loss)
        return total, CLEARStats(total, replay_weight, replay_weight, replay_weight)

    total = (
        config.on_policy_weight * on_policy_loss
        + config.replay_rl_weight * replay_rl_loss
        + config.policy_clone_weight * policy_clone_loss
        + config.value_clone_weight * value_clone_loss
    )
    replay_weight = xp.asarray(config.replay_fraction, dtype=xp.asarray(total).dtype)
    return total, CLEARStats(
        total_loss=total,
        replay_weight=replay_weight,
        policy_clone_loss=policy_clone_loss,
        value_clone_loss=value_clone_loss,
    )


def policy_consolidation_penalty(
    current_policy_outputs,
    current_values,
    teacher_policy_outputs,
    teacher_values,
    config: PolicyConsolidationConfig,
):
    """Penalty to keep current behavior aligned with a teacher cascade."""

    zero = xp.asarray(0.0, dtype=xp.asarray(current_policy_outputs).dtype)
    if not config.enabled:
        return zero, PolicyConsolidationStats(zero, zero, zero)

    if config.reduction == "mean":
        reduce_fn = xp.mean
    elif config.reduction == "max":
        reduce_fn = xp.max
    else:
        raise ValueError(f"Unsupported reduction {config.reduction!r}")

    per_teacher_policy = xp.sum(
        xp.square(current_policy_outputs[None, ...] - teacher_policy_outputs),
        axis=-1,
    )
    per_teacher_value = xp.square(current_values[None, ...] - teacher_values)
    policy_penalty = config.penalty_weight * reduce_fn(per_teacher_policy)
    value_penalty = config.value_weight * reduce_fn(per_teacher_value)
    total = policy_penalty + value_penalty
    return total, PolicyConsolidationStats(
        policy_penalty=policy_penalty,
        value_penalty=value_penalty,
        total_penalty=total,
    )


def baseline_name(project_config) -> str:
    """Human-readable name for the active CL baseline."""

    if project_config.online_ewc.enabled:
        return "online_ewc"
    if project_config.clear.enabled:
        return "clear"
    if project_config.policy_consolidation.enabled:
        return "policy_consolidation"
    return "none"
