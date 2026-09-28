"""JAX-native temporal deviation correction."""

from __future__ import annotations

from typing import Any, NamedTuple

from .config import CorrectionConfig


def _require_jax():
    try:
        import jax.numpy as jnp
        from jax import lax
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "JAX is required for the JAX deviation correction path."
        ) from exc
    return jnp, lax


class CorrectionStats(NamedTuple):
    active: Any
    reference_index: Any
    positive_regret: Any
    kl: Any
    penalty: Any


def positive_regret(
    candidate_score,
    deviation_score,
    margin: float = 0.0,
    power: float = 1.0,
):
    """Positive regret in JAX-friendly form."""

    jnp, _ = _require_jax()
    gap = jnp.maximum(0.0, deviation_score - candidate_score - margin)
    return gap**power


def select_reference_index(candidate_score, deviation_scores, margin: float = 0.0):
    """Choose the best deviation that beats the candidate score."""

    jnp, _ = _require_jax()
    better = deviation_scores > (candidate_score + margin)
    masked = jnp.where(better, deviation_scores, -jnp.inf)
    index = jnp.argmax(masked)
    active = jnp.any(better)
    index = jnp.where(active, index, -1)
    return index, active


def _kl_from_log_probs(candidate_log_probs, reference_log_probs, kl_floor: float):
    jnp, _ = _require_jax()
    candidate_probs = jnp.exp(candidate_log_probs)
    kl = jnp.sum(
        candidate_probs * (candidate_log_probs - reference_log_probs),
        axis=-1,
    )
    return jnp.maximum(jnp.mean(kl), kl_floor)


def corrected_policy_loss(
    *,
    base_loss,
    candidate_log_probs,
    candidate_score,
    deviation_scores,
    deviation_log_probs,
    config: CorrectionConfig,
):
    """All-device correction term for JAX PPO updates."""

    jnp, lax = _require_jax()
    zero = jnp.asarray(0.0, dtype=base_loss.dtype)
    minus_one = jnp.asarray(-1, dtype=jnp.int32)

    if not config.enabled:
        return base_loss, CorrectionStats(False, minus_one, zero, zero, zero)

    reference_index, active = select_reference_index(
        candidate_score=candidate_score,
        deviation_scores=deviation_scores,
        margin=config.margin,
    )

    def _apply(_):
        reference_log_probs = deviation_log_probs[reference_index]
        reference_score = deviation_scores[reference_index]
        regret = positive_regret(
            candidate_score,
            reference_score,
            margin=config.margin,
            power=config.positive_regret_power,
        ).astype(base_loss.dtype)
        kl = _kl_from_log_probs(
            candidate_log_probs,
            reference_log_probs,
            config.kl_floor,
        ).astype(base_loss.dtype)
        penalty = jnp.asarray(
            config.penalty_weight,
            dtype=base_loss.dtype,
        ) * regret * kl
        return (
            base_loss + penalty,
            CorrectionStats(True, reference_index, regret, kl, penalty),
        )

    def _skip(_):
        return base_loss, CorrectionStats(False, minus_one, zero, zero, zero)

    return lax.cond(active, _apply, _skip, operand=None)
