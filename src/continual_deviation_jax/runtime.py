"""JAX runtime helpers for all-GPU continual RL experiments."""

from __future__ import annotations

import os
import sys
from typing import Any

from .config import RuntimeConfig


def recommended_xla_flags(config: RuntimeConfig) -> tuple[str, ...]:
    """Flags recommended by the MJX docs for NVIDIA GPU throughput."""

    return tuple(config.xla_flags)


def configure_runtime_environment(config: RuntimeConfig) -> dict[str, str]:
    """Set environment variables before importing JAX."""

    updates: dict[str, str] = {}
    existing_flags = os.environ.get("XLA_FLAGS", "").split()
    merged_flags = list(existing_flags)
    for flag in recommended_xla_flags(config):
        if flag not in merged_flags:
            merged_flags.append(flag)
    if merged_flags:
        value = " ".join(merged_flags)
        os.environ["XLA_FLAGS"] = value
        updates["XLA_FLAGS"] = value

    if config.require_gpu:
        preferred = "gpu" if config.platform == "auto" else config.platform
        value = f"{preferred},cpu"
        os.environ.setdefault("JAX_PLATFORMS", value)
        updates["JAX_PLATFORMS"] = os.environ["JAX_PLATFORMS"]

    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "true" if config.preallocate else "false",
    )
    updates["XLA_PYTHON_CLIENT_PREALLOCATE"] = os.environ[
        "XLA_PYTHON_CLIENT_PREALLOCATE"
    ]
    return updates


def _require_jax():
    try:
        import jax
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(install_hint()) from exc
    return jax


def install_hint() -> str:
    """Installation guidance grounded in the official JAX/MJX docs."""

    return (
        "JAX/MJX dependencies are not installed. For a Linux NVIDIA GPU box, "
        "install JAX with GPU support and MJX, for example:\n"
        "  pip install --upgrade \"jax[cuda13]\"\n"
        "  pip install --upgrade mujoco-mjx flax optax\n"
        "See the official JAX and MJX installation docs for the exact wheel "
        "matching your CUDA stack."
    )


def device_summary(config: RuntimeConfig) -> dict[str, Any]:
    """Return a compact JAX device summary."""

    configure_runtime_environment(config)
    jax = _require_jax()
    jax.config.update("jax_enable_x64", config.enable_x64)
    jax.config.update("jax_debug_nans", config.jax_debug_nans)
    devices = jax.devices()
    if config.require_gpu and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(
            "GPU backend required, but JAX did not expose any GPU devices."
        )
    return {
        "default_backend": jax.default_backend(),
        "device_count": len(devices),
        "devices": [str(device) for device in devices],
    }


def runtime_summary_text(config: RuntimeConfig) -> str:
    updates = configure_runtime_environment(config)
    lines = [
        f"platform={config.platform}",
        f"require_gpu={config.require_gpu}",
        f"dtype={config.dtype}",
        f"enable_x64={config.enable_x64}",
    ]
    for key, value in updates.items():
        lines.append(f"{key}={value}")
    return "\n".join(lines)
