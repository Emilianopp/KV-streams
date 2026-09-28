"""Benchmark utilities for the continuing swimmer project."""

from __future__ import annotations

from dataclasses import replace

from .config import BenchmarkConfig, PPOConfig, ProjectConfig


def continuing_swimmer_project() -> ProjectConfig:
    """Default project config matching the paper baseline plus our extensions."""

    return ProjectConfig(
        benchmark=BenchmarkConfig(),
        ppo=PPOConfig(),
    )


def build_checkpoint_schedule(config: BenchmarkConfig) -> list[int]:
    """Construct the checkpoint schedule for the full run horizon."""

    if config.checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be positive")
    steps = list(
        range(
            config.checkpoint_interval,
            config.total_steps + 1,
            config.checkpoint_interval,
        )
    )
    if not steps or steps[-1] != config.total_steps:
        steps.append(config.total_steps)
    return steps


def build_representation_schedule(config: BenchmarkConfig) -> list[int]:
    """Schedule periodic representation captures."""

    if config.evaluation_interval <= 0:
        raise ValueError("evaluation_interval must be positive")
    steps = list(
        range(
            config.evaluation_interval,
            config.total_steps + 1,
            config.evaluation_interval,
        )
    )
    if not steps or steps[-1] != config.total_steps:
        steps.append(config.total_steps)
    return steps


def make_continuing_swimmer_env(config: BenchmarkConfig):
    """Create a no-reset Swimmer env once Gymnasium/MuJoCo are installed."""

    try:
        import gymnasium as gym
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Gymnasium is required to instantiate Continuing Swimmer. "
            "Install the optional dependency group with "
            "`pip install .[continual-swimmer]`."
        ) from exc

    kwargs = dict(config.env_kwargs)
    if config.max_episode_steps is None:
        kwargs["max_episode_steps"] = None
    return gym.make(config.env_id, **kwargs)


def benchmark_summary(project: ProjectConfig) -> str:
    """Pretty summary string for the default experiment."""

    benchmark = project.benchmark
    correction = project.correction
    representation = project.representation
    runtime = project.runtime
    lines = [
        f"project: {project.project_name}",
        f"benchmark: {benchmark.name} ({benchmark.env_id})",
        f"total_steps: {benchmark.total_steps:,}",
        f"seeds: {', '.join(str(seed) for seed in benchmark.seeds)}",
        f"checkpoint_interval: {benchmark.checkpoint_interval:,}",
        f"evaluation_interval: {benchmark.evaluation_interval:,}",
        f"correction_enabled: {correction.enabled}",
        f"lookback: {correction.max_lookback}",
        f"predicted_future_steps: {correction.predicted_future_steps}",
        f"runtime_device: {runtime.device}",
        f"runtime_dtype: {runtime.dtype}",
        f"runtime_amp_enabled: {runtime.amp_enabled}",
        f"representation_layers: {', '.join(representation.layers)}",
        f"representation_metrics: {', '.join(representation.metrics)}",
    ]
    return "\n".join(lines)


def with_paper_baseline(project: ProjectConfig) -> ProjectConfig:
    """Disable the correction term for a clean PPO baseline."""

    return replace(
        project,
        correction=replace(project.correction, enabled=False),
    )
