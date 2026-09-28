"""Representation analysis helpers for continual RL experiments."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch


def _analysis_tensor(features: torch.Tensor) -> torch.Tensor:
    if features.dtype in (torch.float16, torch.bfloat16):
        return features.float()
    return features


def _flatten_samples(features: torch.Tensor) -> torch.Tensor:
    features = _analysis_tensor(features)
    if features.ndim == 1:
        return features[:, None]
    if features.ndim == 2:
        return features
    return features.reshape(features.shape[0], -1)


def _mean_center(features: torch.Tensor) -> torch.Tensor:
    return features - features.mean(dim=0, keepdim=True)


def linear_cka(
    reference: torch.Tensor,
    current: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Linear CKA similarity in [0, 1] for two batches of activations."""

    with torch.no_grad():
        x = _mean_center(_flatten_samples(reference))
        y = _mean_center(_flatten_samples(current))
        cross = torch.linalg.matrix_norm(x.T @ y, ord="fro") ** 2
        x_norm = torch.linalg.matrix_norm(x.T @ x, ord="fro")
        y_norm = torch.linalg.matrix_norm(y.T @ y, ord="fro")
        value = cross / (x_norm * y_norm + eps)
        return float(value.detach().cpu().item())


def cosine_drift(
    reference: torch.Tensor,
    current: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Cosine drift where 0 means aligned and 1 means orthogonal/opposite."""

    with torch.no_grad():
        x = _mean_center(_flatten_samples(reference)).reshape(-1)
        y = _mean_center(_flatten_samples(current)).reshape(-1)
        denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
        similarity = torch.dot(x, y) / (denom + eps)
        return float((1.0 - similarity).detach().cpu().item())


def fit_ridge_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1e-4,
) -> torch.Tensor:
    """Closed-form ridge regression probe with an intercept term."""

    with torch.no_grad():
        x = _flatten_samples(features)
        y = _analysis_tensor(targets)
        if y.ndim == 1:
            y = y[:, None]
        ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        design = torch.cat([x, ones], dim=1)
        identity = torch.eye(design.shape[1], dtype=x.dtype, device=x.device)
        identity[-1, -1] = 0.0  # do not regularize the intercept
        lhs = design.T @ design + alpha * identity
        rhs = design.T @ y
        return torch.linalg.solve(lhs, rhs)


def ridge_probe_r2(
    features: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 1e-4,
) -> float:
    """Fit a ridge probe and return the in-sample R^2 score."""

    with torch.no_grad():
        x = _flatten_samples(features)
        y = _analysis_tensor(targets)
        if y.ndim == 1:
            y = y[:, None]
        weights = fit_ridge_probe(x, y, alpha=alpha)
        ones = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
        design = torch.cat([x, ones], dim=1)
        prediction = design @ weights
        residual = torch.sum((y - prediction) ** 2)
        total = torch.sum((y - y.mean(dim=0, keepdim=True)) ** 2)
        if float(total.detach().cpu().item()) == 0.0:
            return 1.0
        score = 1.0 - residual / total
        return float(score.detach().cpu().item())


@dataclass(frozen=True)
class RepresentationComparison:
    """Summary of how a checkpoint's representation drifted."""

    linear_cka: float
    cosine_drift: float
    ridge_probe_r2: float | None = None


def compare_representations(
    reference: torch.Tensor,
    current: torch.Tensor,
    probe_targets: torch.Tensor | None = None,
    ridge_alpha: float = 1e-4,
) -> RepresentationComparison:
    """Combine the default representation metrics in one call."""

    probe_score = None
    if probe_targets is not None:
        probe_score = ridge_probe_r2(
            current,
            probe_targets,
            alpha=ridge_alpha,
        )
    return RepresentationComparison(
        linear_cka=linear_cka(reference, current),
        cosine_drift=cosine_drift(reference, current),
        ridge_probe_r2=probe_score,
    )


@contextmanager
def capture_layer_outputs(
    model: torch.nn.Module,
    layer_names: tuple[str, ...] | list[str],
) -> Iterator[dict[str, torch.Tensor]]:
    """Capture activations from named submodules during a forward pass."""

    requested = set(layer_names)
    outputs: dict[str, torch.Tensor] = {}
    hooks = []
    available = {name for name, _ in model.named_modules()}
    missing = sorted(requested - available)
    if missing:
        raise KeyError(f"Unknown layer names: {', '.join(missing)}")

    def _hook(name: str):
        def _record(_module, _inputs, output):
            if isinstance(output, tuple):
                output = output[0]
            if isinstance(output, torch.Tensor):
                outputs[name] = output.detach()
        return _record

    for name, module in model.named_modules():
        if name in requested:
            hooks.append(module.register_forward_hook(_hook(name)))
    try:
        yield outputs
    finally:
        for hook in hooks:
            hook.remove()
