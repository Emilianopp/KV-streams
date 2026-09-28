"""Temporal deviation correction utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import CorrectionConfig


@dataclass(frozen=True)
class DeviationCandidate:
    """A past or predicted future policy used as a deviation reference."""

    name: str
    score: float
    log_probs: torch.Tensor
    kind: str = "past"
    step: int | None = None


@dataclass(frozen=True)
class CorrectionResult:
    """Details about the selected reference and resulting penalty."""

    active: bool
    reference_name: str | None
    reference_kind: str | None
    positive_regret: torch.Tensor
    kl: torch.Tensor
    penalty: torch.Tensor

    def summary(self) -> dict[str, float | bool | str | None]:
        """Materialize scalar values only when needed for logging."""

        return {
            "active": self.active,
            "reference_name": self.reference_name,
            "reference_kind": self.reference_kind,
            "positive_regret": float(self.positive_regret.detach().cpu().item()),
            "kl": float(self.kl.detach().cpu().item()),
            "penalty": float(self.penalty.detach().cpu().item()),
        }


def positive_regret(
    candidate_score: float,
    deviation_score: float,
    margin: float = 0.0,
    power: float = 1.0,
) -> float:
    """Positive regret to a deviation policy."""

    gap = max(0.0, deviation_score - candidate_score - margin)
    if gap == 0.0:
        return 0.0
    return gap**power


def select_reference_deviation(
    candidate_score: float,
    deviations: list[DeviationCandidate],
    margin: float = 0.0,
) -> DeviationCandidate | None:
    """Return the highest-scoring deviation that beats the candidate."""

    better = [
        deviation
        for deviation in deviations
        if deviation.score > candidate_score + margin
    ]
    if not better:
        return None
    return max(better, key=lambda deviation: deviation.score)


def _kl_from_log_probs(
    candidate_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    kl_floor: float,
) -> torch.Tensor:
    candidate_probs = candidate_log_probs.exp()
    kl = torch.sum(
        candidate_probs * (candidate_log_probs - reference_log_probs), dim=-1
    )
    return torch.clamp(kl.mean(), min=kl_floor)


def corrected_policy_loss(
    *,
    base_loss: torch.Tensor,
    candidate_log_probs: torch.Tensor,
    candidate_score: float,
    deviations: list[DeviationCandidate],
    config: CorrectionConfig,
) -> tuple[torch.Tensor, CorrectionResult]:
    """Apply the temporal deviation correction to a policy loss."""

    zero = torch.zeros((), device=base_loss.device, dtype=base_loss.dtype)

    if not config.enabled:
        return (
            base_loss,
            CorrectionResult(
                active=False,
                reference_name=None,
                reference_kind=None,
                positive_regret=zero,
                kl=zero,
                penalty=zero,
            ),
        )

    reference = select_reference_deviation(
        candidate_score=candidate_score,
        deviations=deviations,
        margin=config.margin,
    )
    if reference is None:
        return (
            base_loss,
            CorrectionResult(
                active=False,
                reference_name=None,
                reference_kind=None,
                positive_regret=zero,
                kl=zero,
                penalty=zero,
            ),
        )

    regret = torch.as_tensor(
        positive_regret(
            candidate_score=candidate_score,
            deviation_score=reference.score,
            margin=config.margin,
            power=config.positive_regret_power,
        ),
        device=base_loss.device,
        dtype=base_loss.dtype,
    )
    kl = _kl_from_log_probs(
        candidate_log_probs=candidate_log_probs,
        reference_log_probs=reference.log_probs,
        kl_floor=config.kl_floor,
    )
    penalty = torch.as_tensor(
        config.penalty_weight,
        device=base_loss.device,
        dtype=base_loss.dtype,
    ) * regret * kl.to(device=base_loss.device, dtype=base_loss.dtype)
    corrected = base_loss + penalty
    return (
        corrected,
        CorrectionResult(
            active=True,
            reference_name=reference.name,
            reference_kind=reference.kind,
            positive_regret=regret,
            kl=kl.to(device=base_loss.device, dtype=base_loss.dtype),
            penalty=penalty,
        ),
    )
