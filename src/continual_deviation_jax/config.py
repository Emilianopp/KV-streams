"""Configuration objects for the JAX continual-RL benchmark scaffold."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .benchmarks import benchmark_defaults


def _as_tuple(values: Any, default: tuple[Any, ...]) -> tuple[Any, ...]:
    if values is None:
        return default
    if isinstance(values, tuple):
        return values
    if isinstance(values, list):
        return tuple(values)
    return (values,)


def _default_swimmer_mjcf_string() -> str:
    return str(
        Path(__file__).resolve().parents[2]
        / "projects"
        / "continual_swimmer_jax"
        / "assets"
        / "continuing_swimmer.xml"
    )


@dataclass(frozen=True)
class PPOConfig:
    """GPU-batched PPO defaults intended for an MJX setup."""

    num_envs: int = 4096
    rollout_length: int = 32
    num_minibatches: int = 32
    update_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    max_grad_norm: float = 0.5

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "PPOConfig":
        if not data:
            return cls()
        return cls(**dict(data))


@dataclass(frozen=True)
class ACPQNConfig:
    """Actor-critic PQN defaults for continuous-control GPU runs."""

    num_envs: int = 4096
    rollout_length: int = 32
    batch_size: int = 1024
    gamma: float = 0.99
    q_lambda: float = 0.95
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    tau: float = 0.005
    actor_update_interval: int = 2
    action_noise_std: float = 0.1
    max_grad_norm: float = 0.5
    normalization: str = "layernorm"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "ACPQNConfig":
        if not data:
            return cls()
        return cls(**dict(data))


@dataclass(frozen=True)
class AlgorithmConfig:
    """Top-level algorithm selection for the JAX scaffold."""

    name: str = "ppo"
    uses_deviation_correction: bool = True

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None
    ) -> "AlgorithmConfig":
        if not data:
            return cls()
        return cls(**dict(data))


@dataclass(frozen=True)
class RandomPolicyConfig:
    """Pure random-action baseline settings."""

    num_envs: int = 4096
    rollout_length: int = 64
    evaluation_episodes: int = 128
    continuous_low: float = -1.0
    continuous_high: float = 1.0
    discrete_sampling: str = "uniform"

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None
    ) -> "RandomPolicyConfig":
        if not data:
            return cls()
        return cls(**dict(data))


@dataclass(frozen=True)
class CorrectionConfig:
    """Temporal deviation correction on top of PPO."""

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
    """Representation metrics that can remain on accelerator."""

    metrics: tuple[str, ...] = ("linear_cka", "cosine_drift", "ridge_probe_r2")
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
class VariationBudgetConfig:
    """Config for recording World's Edge-style variation budgets."""

    enabled: bool = True
    anchor_observation_count: int = 4096
    reduction: str = "max"
    include_policy_budget: bool = True
    include_reward_budget: bool = True
    include_kernel_budget: bool = False
    policy_metric: str = "gaussian_pinsker_tv"
    accumulate: bool = True

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None
    ) -> "VariationBudgetConfig":
        if not data:
            return cls()
        return cls(**dict(data))


@dataclass(frozen=True)
class OnlineEWCConfig:
    """Online EWC baseline on top of the base RL algorithm."""

    enabled: bool = False
    penalty_weight: float = 10.0
    fisher_decay: float = 0.99
    update_interval: int = 10000
    epsilon: float = 1e-8

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "OnlineEWCConfig":
        if not data:
            return cls()
        return cls(**dict(data))


@dataclass(frozen=True)
class CLEARConfig:
    """CLEAR-style replay and distillation baseline."""

    enabled: bool = False
    replay_capacity: int = 100000
    replay_fraction: float = 0.5
    on_policy_weight: float = 1.0
    replay_rl_weight: float = 1.0
    policy_clone_weight: float = 1.0
    value_clone_weight: float = 0.5
    reservoir_sampling: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "CLEARConfig":
        if not data:
            return cls()
        return cls(**dict(data))


@dataclass(frozen=True)
class PolicyConsolidationConfig:
    """Policy-consolidation baseline for task-agnostic continual RL."""

    enabled: bool = False
    penalty_weight: float = 1.0
    value_weight: float = 0.5
    cascade_decay_rates: tuple[float, ...] = (0.5, 0.9, 0.99)
    reduction: str = "mean"

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None
    ) -> "PolicyConsolidationConfig":
        if not data:
            return cls()
        payload = dict(data)
        if "cascade_decay_rates" in payload:
            payload["cascade_decay_rates"] = _as_tuple(
                payload["cascade_decay_rates"], cls.cascade_decay_rates
            )
        return cls(**payload)


@dataclass(frozen=True)
class RuntimeConfig:
    """JAX runtime settings aimed at keeping everything on GPU."""

    platform: str = "gpu"
    require_gpu: bool = True
    dtype: str = "bfloat16"
    enable_x64: bool = False
    xla_flags: tuple[str, ...] = ("--xla_gpu_triton_gemm_any=true",)
    preallocate: bool = False
    jax_debug_nans: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RuntimeConfig":
        if not data:
            return cls()
        payload = dict(data)
        if "xla_flags" in payload:
            payload["xla_flags"] = _as_tuple(payload["xla_flags"], cls.xla_flags)
        return cls(**payload)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark config shared across all scaffolded environments."""

    name: str = "continuing_swimmer_mjx"
    suite: str = "mujoco"
    backend: str = "mjx"
    adapter: str = "mjx_swimmer"
    env_id: str = "Swimmer-v5"
    task_name: str = "continuing_swimmer"
    action_space: str = "continuous"
    observation_space: str = "state"
    required_packages: tuple[str, ...] = ("jax", "mujoco", "mujoco-mjx")
    all_gpu_capable: bool = True
    implementation_status: str = "ready"
    recommended_algorithm: str = "ac_pqn"
    notes: str = "Current fully scaffolded all-GPU benchmark."
    mjcf_path: str | None = None
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    total_steps: int = 50_000_000
    seeds: tuple[int, ...] = tuple(range(10))
    checkpoint_interval: int = 500_000
    evaluation_interval: int = 1_000_000
    frame_skip: int = 4
    reset_noise_scale: float = 0.1
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 1e-4
    exclude_current_positions_from_observation: bool = True
    solver_iterations: int = 1
    solver_ls_iterations: int = 4
    jacobian: str = "dense"

    def __post_init__(self) -> None:
        if self.mjcf_path is None and self.name == "continuing_swimmer_mjx":
            object.__setattr__(self, "mjcf_path", _default_swimmer_mjcf_string())

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any] | None
    ) -> "BenchmarkConfig":
        if not data:
            return cls()
        name = str(data.get("name", cls.name))
        payload = benchmark_defaults(name)
        payload.update(dict(data))
        if "seeds" in payload:
            payload["seeds"] = tuple(payload["seeds"])
        if "required_packages" in payload:
            payload["required_packages"] = _as_tuple(
                payload["required_packages"], cls.required_packages
            )
        if "env_kwargs" in payload and payload["env_kwargs"] is not None:
            payload["env_kwargs"] = dict(payload["env_kwargs"])
        return cls(**payload)


@dataclass(frozen=True)
class ProjectConfig:
    """Full JAX/MJX project configuration."""

    project_name: str = "continuing-swimmer-jax-all-gpu"
    output_dir: str = "artifacts/continual_swimmer_jax"
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    random_policy: RandomPolicyConfig = field(default_factory=RandomPolicyConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    ac_pqn: ACPQNConfig = field(default_factory=ACPQNConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    representation: RepresentationConfig = field(
        default_factory=RepresentationConfig
    )
    variation_budget: VariationBudgetConfig = field(
        default_factory=VariationBudgetConfig
    )
    online_ewc: OnlineEWCConfig = field(default_factory=OnlineEWCConfig)
    clear: CLEARConfig = field(default_factory=CLEARConfig)
    policy_consolidation: PolicyConsolidationConfig = field(
        default_factory=PolicyConsolidationConfig
    )
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ProjectConfig":
        payload = dict(data)
        payload["algorithm"] = AlgorithmConfig.from_mapping(
            payload.get("algorithm")
        )
        payload["benchmark"] = BenchmarkConfig.from_mapping(
            payload.get("benchmark")
        )
        payload["random_policy"] = RandomPolicyConfig.from_mapping(
            payload.get("random_policy")
        )
        payload["ppo"] = PPOConfig.from_mapping(payload.get("ppo"))
        payload["ac_pqn"] = ACPQNConfig.from_mapping(payload.get("ac_pqn"))
        payload["correction"] = CorrectionConfig.from_mapping(
            payload.get("correction")
        )
        payload["representation"] = RepresentationConfig.from_mapping(
            payload.get("representation")
        )
        payload["variation_budget"] = VariationBudgetConfig.from_mapping(
            payload.get("variation_budget")
        )
        payload["online_ewc"] = OnlineEWCConfig.from_mapping(
            payload.get("online_ewc")
        )
        payload["clear"] = CLEARConfig.from_mapping(payload.get("clear"))
        payload["policy_consolidation"] = PolicyConsolidationConfig.from_mapping(
            payload.get("policy_consolidation")
        )
        payload["runtime"] = RuntimeConfig.from_mapping(payload.get("runtime"))
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_project_config(path: str | Path) -> ProjectConfig:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyYAML is required to load config files."
        ) from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise TypeError(f"Expected mapping at top of {config_path}, got {type(data)!r}")
    return ProjectConfig.from_mapping(data)
