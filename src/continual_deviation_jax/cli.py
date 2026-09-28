"""CLI for the JAX continual-RL benchmark scaffold."""

from __future__ import annotations

import argparse
from pathlib import Path

from .benchmarks import algorithm_compatibility
from .config import ProjectConfig, load_project_config
from .continual_baselines import baseline_name
from .mjx_env import default_swimmer_mjcf_path
from .runtime import (
    device_summary,
    install_hint,
    runtime_summary_text,
)


def _default_project() -> ProjectConfig:
    return ProjectConfig()


def _describe(config_path: str | None) -> int:
    project = load_project_config(config_path) if config_path else _default_project()
    correction_enabled = (
        project.algorithm.name != "random"
        and project.algorithm.uses_deviation_correction
        and project.correction.enabled
    )
    cl_baseline = baseline_name(project)
    compatible, compatibility_note = algorithm_compatibility(
        project.algorithm.name, project.benchmark.action_space
    )
    print(f"project: {project.project_name}")
    if project.algorithm.name == "random":
        print("base_update: RandomPolicy")
        print("algorithm_variant: random_actions")
        print(f"num_envs: {project.random_policy.num_envs}")
        print(f"rollout_length: {project.random_policy.rollout_length}")
        print(f"evaluation_episodes: {project.random_policy.evaluation_episodes}")
        print(f"discrete_sampling: {project.random_policy.discrete_sampling}")
    elif project.algorithm.name == "ac_pqn":
        print("base_update: AC-PQN")
        print(
            f"algorithm_variant: {'ac_pqn_deviation' if correction_enabled else 'ac_pqn'}"
        )
        print(f"num_envs: {project.ac_pqn.num_envs}")
        print(f"rollout_length: {project.ac_pqn.rollout_length}")
        print(f"batch_size: {project.ac_pqn.batch_size}")
        print(f"actor_update_interval: {project.ac_pqn.actor_update_interval}")
    else:
        print("base_update: PPO")
        print(
            f"algorithm_variant: {'ppo_deviation' if correction_enabled else 'ppo'}"
        )
        print(f"num_envs: {project.ppo.num_envs}")
        print(f"rollout_length: {project.ppo.rollout_length}")
    print(f"deviation_correction: {correction_enabled}")
    print(f"continual_baseline: {cl_baseline}")
    print(f"benchmark: {project.benchmark.name}")
    print(f"benchmark_suite: {project.benchmark.suite}")
    print(f"benchmark_backend: {project.benchmark.backend}")
    print(f"benchmark_adapter: {project.benchmark.adapter}")
    print(f"benchmark_env_id: {project.benchmark.env_id}")
    print(f"benchmark_task: {project.benchmark.task_name}")
    print(f"action_space: {project.benchmark.action_space}")
    print(f"all_gpu_capable: {project.benchmark.all_gpu_capable}")
    print(f"implementation_status: {project.benchmark.implementation_status}")
    print(
        "required_packages: "
        + ",".join(project.benchmark.required_packages)
    )
    if project.benchmark.mjcf_path:
        print(f"mjcf_path: {project.benchmark.mjcf_path}")
    print(f"algorithm_compatible: {compatible}")
    if compatibility_note:
        print(f"compatibility_note: {compatibility_note}")
    print(f"variation_budget_enabled: {project.variation_budget.enabled}")
    print(f"variation_budget_policy_metric: {project.variation_budget.policy_metric}")
    print(f"variation_budget_reduction: {project.variation_budget.reduction}")
    print(f"runtime_platform: {project.runtime.platform}")
    print(f"runtime_dtype: {project.runtime.dtype}")
    print(f"require_gpu: {project.runtime.require_gpu}")
    return 0


def _validate(config_path: str) -> int:
    project = load_project_config(config_path)
    compatible, compatibility_note = algorithm_compatibility(
        project.algorithm.name, project.benchmark.action_space
    )
    print(f"Loaded config from {config_path}")
    print(f"default_asset_exists={default_swimmer_mjcf_path().exists()}")
    if project.benchmark.mjcf_path:
        asset_path = Path(project.benchmark.mjcf_path)
        if not asset_path.is_absolute():
            asset_path = Path(__file__).resolve().parents[2] / asset_path
        print(f"configured_asset_exists={asset_path.exists()}")
    else:
        print("configured_asset_exists=not_applicable")
    print(f"benchmark_backend={project.benchmark.backend}")
    print(f"benchmark_all_gpu_capable={project.benchmark.all_gpu_capable}")
    print(f"benchmark_status={project.benchmark.implementation_status}")
    print(
        "benchmark_required_packages="
        + ",".join(project.benchmark.required_packages)
    )
    print(f"algorithm_compatible={compatible}")
    if compatibility_note:
        print(f"compatibility_note={compatibility_note}")
    if project.runtime.require_gpu and not project.benchmark.all_gpu_capable:
        print(
            "gpu_note=Benchmark env stepping is not GPU-native in this scaffold; "
            "the learner can still stay on GPU."
        )
    print(runtime_summary_text(project.runtime))
    return 0 if compatible else 1


def _install_hint() -> int:
    print(install_hint())
    return 0


def _device_summary(config_path: str | None) -> int:
    project = load_project_config(config_path) if config_path else _default_project()
    try:
        summary = device_summary(project.runtime)
    except ModuleNotFoundError as exc:
        print(str(exc))
        return 1
    for key, value in summary.items():
        print(f"{key}={value}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="continual-deviation-jax",
        description="JAX continual-RL benchmark scaffold.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe = subparsers.add_parser("describe", help="Show project summary.")
    describe.add_argument("--config", help="Optional YAML config path.")

    validate = subparsers.add_parser(
        "validate-config", help="Load and validate a YAML config."
    )
    validate.add_argument("config", help="Path to YAML config.")

    subparsers.add_parser(
        "install-hint", help="Show JAX/MJX installation guidance."
    )

    device = subparsers.add_parser(
        "device-summary", help="Show JAX backend/device summary."
    )
    device.add_argument("--config", help="Optional YAML config path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "describe":
        return _describe(args.config)
    if args.command == "validate-config":
        return _validate(args.config)
    if args.command == "install-hint":
        return _install_hint()
    if args.command == "device-summary":
        return _device_summary(args.config)
    parser.error(f"Unknown command {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
