"""MJX continuing swimmer environment wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from .config import BenchmarkConfig


def _require_mjx():
    try:
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "MJX environment support requires jax, mujoco, and mujoco-mjx."
        ) from exc
    return jax, jnp, mujoco, mjx


def default_swimmer_mjcf_path() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "projects"
        / "continual_swimmer_jax"
        / "assets"
        / "continuing_swimmer.xml"
    )


def _resolve_mjcf_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / candidate


class EnvState(NamedTuple):
    data: Any
    obs: Any
    reward: Any
    done: Any
    steps: Any
    x_position: Any


@dataclass(frozen=True)
class ContinuingSwimmerFns:
    reset: Any
    step: Any
    action_size: int
    observation_size: int
    dt: float
    mjcf_path: str


def _observation(data, exclude_current_positions: bool):
    qpos = data.qpos
    qvel = data.qvel
    if exclude_current_positions:
        return qpos[2:].tolist(), qvel.tolist()
    return qpos.tolist(), qvel.tolist()


def build_continuing_swimmer(config: BenchmarkConfig) -> ContinuingSwimmerFns:
    """Build a batched, JIT-friendly continuing swimmer wrapper."""

    jax, jnp, mujoco, mjx = _require_mjx()

    if not config.mjcf_path:
        raise ValueError(
            "Continuing Swimmer requires benchmark.mjcf_path to point to an MJCF asset."
        )
    mjcf_path = _resolve_mjcf_path(config.mjcf_path)
    host_model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    host_model.opt.iterations = config.solver_iterations
    host_model.opt.ls_iterations = config.solver_ls_iterations

    mjx_model = mjx.put_model(host_model)
    dt = float(host_model.opt.timestep) * config.frame_skip
    action_size = int(host_model.nu)
    qpos_size = int(host_model.nq)
    qvel_size = int(host_model.nv)
    observation_size = qpos_size + qvel_size - (
        2 if config.exclude_current_positions_from_observation else 0
    )

    def observation_from_data(data):
        qpos = data.qpos
        qvel = data.qvel
        if config.exclude_current_positions_from_observation:
            qpos = qpos[2:]
        return jnp.concatenate([qpos, qvel], axis=-1)

    def single_reset(rng):
        rng_qpos, rng_qvel = jax.random.split(rng)
        data = mjx.make_data(mjx_model)
        qpos_noise = jax.random.uniform(
            rng_qpos,
            shape=data.qpos.shape,
            minval=-config.reset_noise_scale,
            maxval=config.reset_noise_scale,
        )
        qvel_noise = jax.random.uniform(
            rng_qvel,
            shape=data.qvel.shape,
            minval=-config.reset_noise_scale,
            maxval=config.reset_noise_scale,
        )
        data = data.replace(qpos=data.qpos + qpos_noise, qvel=data.qvel + qvel_noise)
        data = mjx.forward(mjx_model, data)
        obs = observation_from_data(data)
        return EnvState(
            data=data,
            obs=obs,
            reward=jnp.asarray(0.0, dtype=obs.dtype),
            done=jnp.asarray(False),
            steps=jnp.asarray(0, dtype=jnp.int32),
            x_position=data.qpos[0],
        )

    def single_step(state: EnvState, action):
        clipped_action = jnp.clip(action, -1.0, 1.0)

        def body_fn(_, data):
            data = data.replace(ctrl=clipped_action)
            return mjx.step(mjx_model, data)

        data = jax.lax.fori_loop(0, config.frame_skip, body_fn, state.data)
        x_before = state.x_position
        x_after = data.qpos[0]
        forward_reward = config.forward_reward_weight * (x_after - x_before) / dt
        ctrl_cost = config.ctrl_cost_weight * jnp.sum(jnp.square(clipped_action))
        reward = forward_reward - ctrl_cost
        obs = observation_from_data(data)
        return EnvState(
            data=data,
            obs=obs,
            reward=reward,
            done=jnp.asarray(False),
            steps=state.steps + jnp.asarray(1, dtype=jnp.int32),
            x_position=x_after,
        )

    reset = jax.jit(jax.vmap(single_reset))
    step = jax.jit(jax.vmap(single_step), donate_argnums=(0,))
    return ContinuingSwimmerFns(
        reset=reset,
        step=step,
        action_size=action_size,
        observation_size=observation_size,
        dt=dt,
        mjcf_path=str(mjcf_path),
    )
