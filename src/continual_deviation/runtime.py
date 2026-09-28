"""Runtime helpers for preparing the project to run efficiently on GPU."""

from __future__ import annotations

from contextlib import nullcontext

import torch

from .config import RuntimeConfig


_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a runtime device string into a torch device."""

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def resolve_dtype(dtype: str) -> torch.dtype:
    """Resolve a dtype alias used in config files."""

    key = dtype.lower()
    if key not in _DTYPE_MAP:
        supported = ", ".join(sorted(_DTYPE_MAP))
        raise ValueError(f"Unsupported dtype {dtype!r}. Supported: {supported}")
    return _DTYPE_MAP[key]


def configure_torch_runtime(config: RuntimeConfig) -> torch.device:
    """Apply backend flags that matter for GPU training throughput."""

    device = resolve_device(config.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.benchmark = config.cudnn_benchmark
    return device


def prepare_model_for_runtime(
    model: torch.nn.Module,
    config: RuntimeConfig,
) -> tuple[torch.nn.Module, torch.device, torch.dtype]:
    """Move a model to the configured device/dtype and optionally compile it."""

    device = configure_torch_runtime(config)
    dtype = resolve_dtype(config.dtype)
    model = model.to(device=device, dtype=dtype)
    if config.torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode=config.torch_compile_mode)
    return model, device, dtype


def autocast_context(config: RuntimeConfig, device: torch.device):
    """Autocast context tuned to the configured runtime."""

    if not config.amp_enabled:
        return nullcontext()
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=resolve_dtype(config.dtype))
    if device.type == "cpu":
        return torch.autocast(device_type="cpu", dtype=resolve_dtype(config.dtype))
    return nullcontext()


def move_batch_to_device(
    batch,
    device: torch.device,
    non_blocking: bool = True,
):
    """Recursively move a nested batch of tensors to a device."""

    if isinstance(batch, torch.Tensor):
        return batch.to(device=device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {
            key: move_batch_to_device(value, device, non_blocking=non_blocking)
            for key, value in batch.items()
        }
    if isinstance(batch, tuple):
        return tuple(
            move_batch_to_device(value, device, non_blocking=non_blocking)
            for value in batch
        )
    if isinstance(batch, list):
        return [
            move_batch_to_device(value, device, non_blocking=non_blocking)
            for value in batch
        ]
    return batch
