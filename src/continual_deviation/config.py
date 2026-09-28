"""Configuration objects for the continual-deviation research scaffold."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping


def _as_tuple(values: Any, default: tuple[Any, ...]) -> tuple[Any, ...]:
    if values is None:
        return default
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(values)
    return (values,)


@dataclass(frozen=True)
class PPOConfig:
    """PPO baseline from Elelimy et al.'s continuing swimmer experiment."""

    rollout_length: int = 2048
    epochs: int = 4
    minibatch_size: int = 64
    gae_lambda: float = 0.95
    gamma: float = 0.99
    clip_range: float = 0.2
    input_normalization: bool = True
    advantage_normalization: bool = True
    value_loss_clipping: bool = True
    max_grad_norm: float = 0.5
    optimizer: str = "adam"
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    optimizer_eps: float = 1e-5

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "PPOConfig":
        if not data:
            return cls()
        return cls(**dict(data))


@dataclass(frozen=True)
class CorrectionConfig:
    """Controls the temporal deviation correction term."""

    enabled: bool = True
    penalty_weight: float = 0.5
    margin: float = 0.0
    max_lookback: int = 5
    include_best_so_far: bool = True
    predicted_future_steps: int = 1
    positive_regret_power: float = 1.0
    kl_floor: float = 1e-8

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CorrectionConfig":
        if not data:
            return cls()
        return cls(**dict(data))


@dataclass(frozen=True)
class RepresentationConfig:
    """Representation metrics to log while the agent learns online."""

    layers: tuple[str, ...] = ("policy_backbone", "value_backbone")
    metrics: tuple[str, ...] = (
        "linear_cka",
        "cosine_drift",
        "ridge_probe_r2",
    )
    probe_targets: tuple[str, ...] = (
        "forward_velocity",
        "joint_phase",
        "action_norm",
        "return_to_go",
    )
    compare_windows: tuple[str, ...] = ("early", "peak", "post_collapse")
    ridge_alpha: float = 1e-4

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None
    ) -> "RepresentationConfig":
        if not data:
            return cls()
        payload = dict(data)
        if "layers" in payload:
            payload["layers"] = _as_tuple(payload["layers"], cls.layers)
        if "metrics" in payload:
            payload["metrics"] = _as_tuple(payload["metrics"], cls.metrics)
        if "probe_targets" in payload:
            payload["probe_targets"] = _as_tuple(
                payload["probe_targets"], cls.probe_targets
            )
        if "compare_windows" in payload:
            payload["compare_windows"] = _as_tuple(
                payload["compare_windows"], cls.compare_windows
            )
        return cls(**payload)


@dataclass(frozen=True)
class RuntimeConfig:
    """Execution settings for CPU/GPU training and analysis."""

    device: str = "auto"
    dtype: str = "float32"
    amp_enabled: bool = False
    torch_compile: bool = False
    torch_compile_mode: str = "default"
    allow_tf32: bool = True
    cudnn_benchmark: bool = True
    pin_memory: bool = True
    dataloader_workers: int = 4

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RuntimeConfig":
        if not data:
            return cls()
        return cls(**dict(data))


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark settings for the continuing swimmer study."""

    name: str = "continuing_swimmer"
    env_id: str = "Swimmer-v5"
    total_steps: int = 50_000_000
    seeds: tuple[int, ...] = tuple(range(10))
    checkpoint_interval: int = 500_000
    evaluation_interval: int = 1_000_000
    max_episode_steps: int | None = None
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    representation_samples: int = 4096
    metrics: tuple[str, ...] = (
        "online_return",
        "time_to_collapse",
        "end_vs_peak_ratio",
        "deviation_regret",
        "representation_cka",
        "probe_r2",
    )

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None
    ) -> "BenchmarkConfig":
        if not data:
            return cls()
        payload = dict(data)
        if "seeds" in payload:
            payload["seeds"] = tuple(payload["seeds"])
        if "metrics" in payload:
            payload["metrics"] = _as_tuple(payload["metrics"], cls.metrics)
        if "env_kwargs" in payload and payload["env_kwargs"] is None:
            payload["env_kwargs"] = {}
        return cls(**payload)


@dataclass(frozen=True)
class ProjectConfig:
    """Full experiment configuration for the scaffold."""

    project_name: str = "continuing-swimmer-temporal-deviation"
    output_dir: str = "artifacts/continual_swimmer"
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    representation: RepresentationConfig = field(
        default_factory=RepresentationConfig
    )
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProjectConfig":
        payload = dict(data)
        payload["benchmark"] = BenchmarkConfig.from_mapping(
            payload.get("benchmark")
        )
        payload["ppo"] = PPOConfig.from_mapping(payload.get("ppo"))
        payload["correction"] = CorrectionConfig.from_mapping(
            payload.get("correction")
        )
        payload["representation"] = RepresentationConfig.from_mapping(
            payload.get("representation")
        )
        payload["runtime"] = RuntimeConfig.from_mapping(payload.get("runtime"))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_project_config(path: str | Path) -> ProjectConfig:
    """Load a YAML config file into strongly-typed config objects."""

    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to load config files. "
            "Install the optional dependency group: "
            "`pip install .[continual-swimmer]`."
        ) from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected mapping at top of {config_path}, got {type(data)!r}")
    return ProjectConfig.from_mapping(data)
