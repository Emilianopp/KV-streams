"""Representation analysis that can stay on JAX devices."""

from __future__ import annotations


def _require_jax():
    try:
        import jax.numpy as jnp
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "JAX is required for JAX representation analysis."
        ) from exc
    return jnp


def _flatten_samples(features):
    jnp = _require_jax()
    if features.ndim == 1:
        return features[:, None]
    if features.ndim == 2:
        return features
    return jnp.reshape(features, (features.shape[0], -1))


def _mean_center(features):
    jnp = _require_jax()
    return features - jnp.mean(features, axis=0, keepdims=True)


def linear_cka(reference, current, eps: float = 1e-12):
    jnp = _require_jax()
    x = _mean_center(_flatten_samples(reference).astype(jnp.float32))
    y = _mean_center(_flatten_samples(current).astype(jnp.float32))
    cross = jnp.linalg.norm(x.T @ y, ord="fro") ** 2
    x_norm = jnp.linalg.norm(x.T @ x, ord="fro")
    y_norm = jnp.linalg.norm(y.T @ y, ord="fro")
    return cross / (x_norm * y_norm + eps)


def cosine_drift(reference, current, eps: float = 1e-12):
    jnp = _require_jax()
    x = jnp.reshape(_mean_center(_flatten_samples(reference).astype(jnp.float32)), (-1,))
    y = jnp.reshape(_mean_center(_flatten_samples(current).astype(jnp.float32)), (-1,))
    denom = jnp.linalg.norm(x) * jnp.linalg.norm(y)
    similarity = jnp.dot(x, y) / (denom + eps)
    return 1.0 - similarity


def ridge_probe_r2(features, targets, alpha: float = 1e-4):
    jnp = _require_jax()
    x = _flatten_samples(features).astype(jnp.float32)
    y = targets.astype(jnp.float32)
    if y.ndim == 1:
        y = y[:, None]
    ones = jnp.ones((x.shape[0], 1), dtype=x.dtype)
    design = jnp.concatenate([x, ones], axis=1)
    identity = jnp.eye(design.shape[1], dtype=x.dtype)
    identity = identity.at[-1, -1].set(0.0)
    weights = jnp.linalg.solve(design.T @ design + alpha * identity, design.T @ y)
    prediction = design @ weights
    residual = jnp.sum((y - prediction) ** 2)
    total = jnp.sum((y - jnp.mean(y, axis=0, keepdims=True)) ** 2)
    return jnp.where(total == 0.0, 1.0, 1.0 - residual / total)
