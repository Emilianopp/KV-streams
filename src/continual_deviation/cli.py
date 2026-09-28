"""CLI entrypoints for the continual-deviation scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .benchmarks import benchmark_summary, continuing_swimmer_project
from .config import load_project_config
from .representation import compare_representations
from .runtime import configure_torch_runtime, resolve_dtype
from .update import DeviationCandidate, corrected_policy_loss


def _describe(config_path: str | None) -> int:
    project = (
        load_project_config(config_path)
        if config_path
        else continuing_swimmer_project()
    )
    print(benchmark_summary(project))
    return 0


def _validate(config_path: str) -> int:
    project = load_project_config(config_path)
    print(f"Loaded config from {Path(config_path)}")
    print(benchmark_summary(project))
    return 0


def _smoke_correction() -> int:
    project = continuing_swimmer_project()
    candidate_log_probs = torch.log_softmax(
        torch.tensor([[2.1, 1.2, -0.2]], dtype=torch.float32), dim=-1
    )
    deviations = [
        DeviationCandidate(
            name="pi_t-1",
            score=2.3,
            kind="past",
            step=950_000,
            log_probs=torch.log_softmax(
                torch.tensor([[2.2, 1.0, -0.4]], dtype=torch.float32),
                dim=-1,
            ),
        ),
        DeviationCandidate(
            name="pi_hat_t+1",
            score=2.8,
            kind="future",
            step=1_000_000,
            log_probs=torch.log_softmax(
                torch.tensor([[2.8, 0.7, -1.0]], dtype=torch.float32),
                dim=-1,
            ),
        ),
    ]
    base_loss = torch.tensor(0.37)
    corrected, result = corrected_policy_loss(
        base_loss=base_loss,
        candidate_log_probs=candidate_log_probs,
        candidate_score=2.0,
        deviations=deviations,
        config=project.correction,
    )
    print(f"base_loss={float(base_loss):.6f}")
    print(f"corrected_loss={float(corrected):.6f}")
    print(f"selected_reference={result.reference_name} ({result.reference_kind})")
    summary = result.summary()
    print(f"positive_regret={summary['positive_regret']:.6f}")
    print(f"penalty={summary['penalty']:.6f}")
    return 0


def _smoke_representation() -> int:
    torch.manual_seed(7)
    reference = torch.randn(128, 16)
    current = reference + 0.1 * torch.randn(128, 16)
    targets = 0.4 * current[:, 0] - 0.2 * current[:, 1]
    summary = compare_representations(
        reference=reference,
        current=current,
        probe_targets=targets,
    )
    print(f"linear_cka={summary.linear_cka:.6f}")
    print(f"cosine_drift={summary.cosine_drift:.6f}")
    if summary.ridge_probe_r2 is not None:
        print(f"ridge_probe_r2={summary.ridge_probe_r2:.6f}")
    return 0


def _device_summary(config_path: str | None) -> int:
    project = (
        load_project_config(config_path)
        if config_path
        else continuing_swimmer_project()
    )
    device = configure_torch_runtime(project.runtime)
    dtype = resolve_dtype(project.runtime.dtype)
    print(f"resolved_device={device}")
    print(f"dtype={dtype}")
    print(f"amp_enabled={project.runtime.amp_enabled}")
    print(f"torch_compile={project.runtime.torch_compile}")
    print(f"allow_tf32={project.runtime.allow_tf32}")
    print(f"pin_memory={project.runtime.pin_memory}")
    print(f"dataloader_workers={project.runtime.dataloader_workers}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        index = device.index or 0
        print(f"cuda_device_name={torch.cuda.get_device_name(index)}")
        print(f"cuda_device_capability={torch.cuda.get_device_capability(index)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="continual-deviation",
        description=(
            "Scaffold and diagnostics for deviation-corrected continual RL."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser(
        "describe", help="Show the continuing swimmer benchmark summary."
    )
    describe.add_argument(
        "--config",
        help="Optional YAML config path. If omitted, use the built-in default.",
    )

    validate = subparsers.add_parser(
        "validate-config", help="Load and validate a YAML config."
    )
    validate.add_argument("config", help="Path to YAML config.")

    subparsers.add_parser(
        "smoke-correction",
        help="Run a synthetic deviation-correction example.",
    )
    subparsers.add_parser(
        "smoke-representation",
        help="Run a synthetic representation-analysis example.",
    )
    device_summary = subparsers.add_parser(
        "device-summary",
        help="Show the resolved runtime device and GPU-related settings.",
    )
    device_summary.add_argument(
        "--config",
        help="Optional YAML config path. If omitted, use the built-in default.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "describe":
        return _describe(args.config)
    if args.command == "validate-config":
        return _validate(args.config)
    if args.command == "smoke-correction":
        return _smoke_correction()
    if args.command == "smoke-representation":
        return _smoke_representation()
    if args.command == "device-summary":
        return _device_summary(args.config)
    parser.error(f"Unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
