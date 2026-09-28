"""Benchmark registry and compatibility helpers for continual RL configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_swimmer_mjcf_string() -> str:
    return str(
        _repo_root()
        / "projects"
        / "continual_swimmer_jax"
        / "assets"
        / "continuing_swimmer.xml"
    )


@dataclass(frozen=True)
class BenchmarkSpec:
    """Metadata for a known continual-learning benchmark."""

    name: str
    suite: str
    backend: str
    adapter: str
    env_id: str
    task_name: str
    action_space: str
    observation_space: str
    required_packages: tuple[str, ...]
    all_gpu_capable: bool
    implementation_status: str
    recommended_algorithm: str
    notes: str
    default_total_steps: int
    default_mjcf_path: str | None = None


KNOWN_BENCHMARKS: dict[str, BenchmarkSpec] = {
    "continuing_swimmer_mjx": BenchmarkSpec(
        name="continuing_swimmer_mjx",
        suite="mujoco",
        backend="mjx",
        adapter="mjx_swimmer",
        env_id="Swimmer-v5",
        task_name="continuing_swimmer",
        action_space="continuous",
        observation_space="state",
        required_packages=("jax", "mujoco", "mujoco-mjx"),
        all_gpu_capable=True,
        implementation_status="ready",
        recommended_algorithm="ac_pqn",
        notes="Current fully scaffolded all-GPU benchmark.",
        default_total_steps=50_000_000,
        default_mjcf_path=_default_swimmer_mjcf_string(),
    ),
    "craftax_classic_jax": BenchmarkSpec(
        name="craftax_classic_jax",
        suite="craftax",
        backend="jax",
        adapter="craftax_classic",
        env_id="Craftax-Classic",
        task_name="open_ended_survival",
        action_space="discrete",
        observation_space="grid_and_stats",
        required_packages=("jax", "craftax"),
        all_gpu_capable=True,
        implementation_status="planned",
        recommended_algorithm="ppo",
        notes="Best next all-GPU benchmark for representation drift studies.",
        default_total_steps=100_000_000,
    ),
    "continual_world": BenchmarkSpec(
        name="continual_world",
        suite="continual_world",
        backend="gymnasium_cpu",
        adapter="continual_world",
        env_id="ContinualWorld",
        task_name="cw10",
        action_space="continuous",
        observation_space="state",
        required_packages=("gymnasium", "metaworld", "continual-world"),
        all_gpu_capable=False,
        implementation_status="planned",
        recommended_algorithm="ac_pqn",
        notes="Strong continual RL benchmark, but environment stepping is CPU-side.",
        default_total_steps=30_000_000,
    ),
    "jelly_bean_world": BenchmarkSpec(
        name="jelly_bean_world",
        suite="jelly_bean_world",
        backend="python_cpu",
        adapter="jelly_bean_world",
        env_id="JellyBeanWorld",
        task_name="dynamic_foraging",
        action_space="discrete",
        observation_space="grid",
        required_packages=("jelly-bean-world",),
        all_gpu_capable=False,
        implementation_status="planned",
        recommended_algorithm="ppo",
        notes="Interesting continual-learning world benchmark, but not GPU-native.",
        default_total_steps=20_000_000,
    ),
}


def benchmark_defaults(name: str) -> dict[str, Any]:
    """Return registry-backed defaults for a known benchmark name."""

    spec = KNOWN_BENCHMARKS.get(name)
    if spec is None:
        return {}
    payload: dict[str, Any] = {
        "suite": spec.suite,
        "backend": spec.backend,
        "adapter": spec.adapter,
        "env_id": spec.env_id,
        "task_name": spec.task_name,
        "action_space": spec.action_space,
        "observation_space": spec.observation_space,
        "required_packages": spec.required_packages,
        "all_gpu_capable": spec.all_gpu_capable,
        "implementation_status": spec.implementation_status,
        "recommended_algorithm": spec.recommended_algorithm,
        "notes": spec.notes,
        "total_steps": spec.default_total_steps,
    }
    if spec.default_mjcf_path is not None:
        payload["mjcf_path"] = spec.default_mjcf_path
    return payload


def get_benchmark_spec(name: str) -> BenchmarkSpec | None:
    return KNOWN_BENCHMARKS.get(name)


def algorithm_compatibility(
    algorithm_name: str, action_space: str
) -> tuple[bool, str | None]:
    """Return whether the current scaffold can pair this algorithm and action space."""

    if algorithm_name == "random":
        return True, None
    if algorithm_name == "ac_pqn" and action_space != "continuous":
        return (
            False,
            "The current AC-PQN scaffold assumes a continuous-action actor. "
            "Use PPO for discrete-action benchmarks unless we add a discrete AC-PQN branch.",
        )
    return True, None
