"""Actor-critic PQN helpers for continuous-control experiments."""

from __future__ import annotations

from typing import NamedTuple

from .config import CorrectionConfig


def _require_xp():
    try:
        import jax.numpy as xp
        from jax import lax
    except ModuleNotFoundError:
        import numpy as xp  # type: ignore[no-redef]

        class _Lax:
            @staticmethod
            def cond(pred, true_fun, false_fun, operand):
                return true_fun(operand) if bool(pred) else false_fun(operand)

        lax = _Lax()
    return xp, lax


class ACPQNCorrectionStats(NamedTuple):
    active: object
    reference_index: object
    positive_regret: object
    action_distance: object
    penalty: object


def _positive_regret(candidate_score, deviation_score, margin: float, power: float):
    xp, _ = _require_xp()
    gap = xp.maximum(0.0, deviation_score - candidate_score - margin)
    return gap**power


def _select_reference_index(candidate_score, deviation_scores, margin: float):
    xp, _ = _require_xp()
    better = deviation_scores > (candidate_score + margin)
    masked = xp.where(better, deviation_scores, -xp.inf)
    index = xp.argmax(masked)
    active = xp.any(better)
    index = xp.where(active, index, -1)
    return index, active


def ac_pqn_td_target(rewards, discounts, next_q_values):
    """One-step TD target for actor-critic PQN."""

    return rewards + discounts * next_q_values


def ac_pqn_actor_loss(q_values):
    """Deterministic actor objective: maximize Q under the current actor."""

    xp, _ = _require_xp()
    return -xp.mean(q_values)


def deterministic_action_distance(
    candidate_actions,
    reference_actions,
    reduction: str = "mean",
):
    """Distance between two deterministic actor outputs on anchor observations."""

    xp, _ = _require_xp()
    per_anchor = xp.sum(xp.square(candidate_actions - reference_actions), axis=-1)
    if reduction == "mean":
        return xp.mean(per_anchor)
    if reduction == "max":
        return xp.max(per_anchor)
    raise ValueError(f"Unsupported reduction {reduction!r}")


def corrected_ac_pqn_actor_loss(
    *,
    base_actor_loss,
    candidate_score,
    deviation_scores,
    candidate_actions,
    deviation_actions,
    config: CorrectionConfig,
    distance_reduction: str = "mean",
):
    """Deviation correction for deterministic actor-critic PQN."""

    xp, lax = _require_xp()
    zero = xp.asarray(0.0, dtype=base_actor_loss.dtype)
    minus_one = xp.asarray(-1, dtype=xp.int32)

    if not config.enabled:
        return (
            base_actor_loss,
            ACPQNCorrectionStats(False, minus_one, zero, zero, zero),
        )

    reference_index, active = _select_reference_index(
        candidate_score=candidate_score,
        deviation_scores=deviation_scores,
        margin=config.margin,
    )

    def _apply(_):
        reference_actions = deviation_actions[reference_index]
        reference_score = deviation_scores[reference_index]
        regret = _positive_regret(
            candidate_score,
            reference_score,
            margin=config.margin,
            power=config.positive_regret_power,
        ).astype(base_actor_loss.dtype)
        distance = deterministic_action_distance(
            candidate_actions,
            reference_actions,
            reduction=distance_reduction,
        ).astype(base_actor_loss.dtype)
        penalty = xp.asarray(
            config.penalty_weight,
            dtype=base_actor_loss.dtype,
        ) * regret * distance
        return (
            base_actor_loss + penalty,
            ACPQNCorrectionStats(True, reference_index, regret, distance, penalty),
        )

    def _skip(_):
        return (
            base_actor_loss,
            ACPQNCorrectionStats(False, minus_one, zero, zero, zero),
        )

    return lax.cond(active, _apply, _skip, operand=None)
