"""Config-level tests for the JAX/MJX scaffold."""

from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from continual_deviation_jax.config import load_project_config
from continual_deviation_jax.ac_pqn import (
    ac_pqn_actor_loss,
    ac_pqn_td_target,
    corrected_ac_pqn_actor_loss,
    deterministic_action_distance,
)
from continual_deviation_jax.benchmarks import (
    algorithm_compatibility,
    get_benchmark_spec,
)
from continual_deviation_jax.continual_baselines import (
    baseline_name,
    clear_loss,
    online_ewc_penalty,
    policy_consolidation_penalty,
)
from continual_deviation_jax.mjx_env import default_swimmer_mjcf_path
from continual_deviation_jax.random_policy import sample_random_actions
from continual_deviation_jax.runtime import (
    install_hint,
    recommended_xla_flags,
)
from continual_deviation_jax.variation_budget import (
    categorical_total_variation,
    deterministic_action_variation,
    gaussian_pinsker_tv_proxy,
    reward_variation,
    update_variation_budget,
)


class JaxConfigTest(unittest.TestCase):
    def test_default_swimmer_asset_exists(self) -> None:
        self.assertTrue(default_swimmer_mjcf_path().exists())

    def test_load_project_config_round_trips_yaml(self) -> None:
        yaml_text = textwrap.dedent(
            """
            project_name: jax-test
            ppo:
              num_envs: 1024
            runtime:
              platform: gpu
              require_gpu: true
              xla_flags:
                - --xla_gpu_triton_gemm_any=true
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            config = load_project_config(path)

        self.assertEqual(config.project_name, "jax-test")
        self.assertEqual(config.ppo.num_envs, 1024)
        self.assertTrue(config.runtime.require_gpu)

    def test_recommended_xla_flags_include_triton(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/continuing_swimmer_jax.yaml"
        )
        self.assertIn(
            "--xla_gpu_triton_gemm_any=true",
            recommended_xla_flags(config.runtime),
        )

    def test_install_hint_mentions_mjx(self) -> None:
        hint = install_hint()
        self.assertIn("mujoco-mjx", hint)
        self.assertIn("jax", hint)

    def test_variation_budget_config_defaults_load(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/continuing_swimmer_jax.yaml"
        )
        self.assertTrue(config.variation_budget.enabled)
        self.assertEqual(config.variation_budget.reduction, "max")

    def test_ac_pqn_config_loads(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/continuing_swimmer_ac_pqn.yaml"
        )
        self.assertEqual(config.algorithm.name, "ac_pqn")
        self.assertFalse(config.algorithm.uses_deviation_correction)
        self.assertEqual(config.ac_pqn.normalization, "layernorm")

    def test_ac_pqn_deviation_config_loads(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/continuing_swimmer_ac_pqn_deviation.yaml"
        )
        self.assertEqual(config.algorithm.name, "ac_pqn")
        self.assertTrue(config.algorithm.uses_deviation_correction)
        self.assertEqual(config.variation_budget.policy_metric, "action_l2")

    def test_online_ewc_config_loads(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/continuing_swimmer_ac_pqn_online_ewc.yaml"
        )
        self.assertTrue(config.online_ewc.enabled)
        self.assertEqual(baseline_name(config), "online_ewc")

    def test_clear_config_loads(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/continuing_swimmer_ac_pqn_clear.yaml"
        )
        self.assertTrue(config.clear.enabled)
        self.assertEqual(baseline_name(config), "clear")

    def test_policy_consolidation_config_loads(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/continuing_swimmer_ac_pqn_policy_consolidation.yaml"
        )
        self.assertTrue(config.policy_consolidation.enabled)
        self.assertEqual(baseline_name(config), "policy_consolidation")

    def test_craftax_config_loads_as_discrete_ppo(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/craftax_classic_ppo_deviation.yaml"
        )
        self.assertEqual(config.algorithm.name, "ppo")
        self.assertEqual(config.benchmark.action_space, "discrete")
        self.assertTrue(config.benchmark.all_gpu_capable)
        self.assertIsNone(config.benchmark.mjcf_path)

    def test_continual_world_config_loads_as_continuous_ac_pqn(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/continual_world_ac_pqn_deviation.yaml"
        )
        self.assertEqual(config.algorithm.name, "ac_pqn")
        self.assertEqual(config.benchmark.action_space, "continuous")
        self.assertFalse(config.benchmark.all_gpu_capable)
        self.assertEqual(config.benchmark.backend, "gymnasium_cpu")

    def test_jelly_bean_world_config_loads_as_discrete_ppo(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/jelly_bean_world_ppo_deviation.yaml"
        )
        self.assertEqual(config.algorithm.name, "ppo")
        self.assertEqual(config.benchmark.action_space, "discrete")
        self.assertFalse(config.benchmark.all_gpu_capable)
        self.assertEqual(config.variation_budget.policy_metric, "categorical_tv")

    def test_benchmark_registry_exposes_expected_metadata(self) -> None:
        spec = get_benchmark_spec("craftax_classic_jax")
        self.assertIsNotNone(spec)
        assert spec is not None
        self.assertEqual(spec.backend, "jax")
        self.assertEqual(spec.recommended_algorithm, "ppo")

    def test_ac_pqn_is_rejected_for_discrete_benchmarks(self) -> None:
        compatible, note = algorithm_compatibility("ac_pqn", "discrete")
        self.assertFalse(compatible)
        self.assertIn("Use PPO", note or "")

    def test_random_swimmer_config_loads(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/continuing_swimmer_random.yaml"
        )
        self.assertEqual(config.algorithm.name, "random")
        self.assertFalse(config.correction.enabled)
        self.assertFalse(config.variation_budget.enabled)
        self.assertEqual(config.random_policy.rollout_length, 64)

    def test_random_craftax_config_loads(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/craftax_classic_random.yaml"
        )
        self.assertEqual(config.algorithm.name, "random")
        self.assertEqual(config.benchmark.action_space, "discrete")
        self.assertEqual(config.random_policy.discrete_sampling, "uniform")

    def test_random_continual_world_config_loads(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/continual_world_random.yaml"
        )
        self.assertEqual(config.algorithm.name, "random")
        self.assertEqual(config.benchmark.action_space, "continuous")
        self.assertFalse(config.benchmark.all_gpu_capable)

    def test_random_jelly_bean_world_config_loads(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/jelly_bean_world_random.yaml"
        )
        self.assertEqual(config.algorithm.name, "random")
        self.assertEqual(config.benchmark.action_space, "discrete")
        self.assertFalse(config.variation_budget.enabled)

    def test_random_policy_is_compatible_with_discrete_benchmarks(self) -> None:
        compatible, note = algorithm_compatibility("random", "discrete")
        self.assertTrue(compatible)
        self.assertIsNone(note)


class VariationBudgetMathTest(unittest.TestCase):
    def test_categorical_total_variation_zero_for_identical_probs(self) -> None:
        import numpy as np

        probs = np.array([[0.2, 0.8], [0.5, 0.5]])
        value = categorical_total_variation(probs, probs, reduction="max")
        self.assertAlmostEqual(float(value), 0.0, places=7)

    def test_gaussian_policy_proxy_zero_for_identical_gaussians(self) -> None:
        import numpy as np

        mean = np.zeros((4, 2))
        log_std = np.zeros((4, 2))
        value = gaussian_pinsker_tv_proxy(mean, log_std, mean, log_std)
        self.assertAlmostEqual(float(value), 0.0, places=4)

    def test_reward_variation_tracks_anchor_reward_change(self) -> None:
        import numpy as np

        prev_rewards = np.array([0.1, 0.2, 0.3])
        curr_rewards = np.array([0.1, 0.5, 0.3])
        value = reward_variation(prev_rewards, curr_rewards, reduction="max")
        self.assertAlmostEqual(float(value), 0.3, places=7)

    def test_update_variation_budget_accumulates_terms(self) -> None:
        stats = update_variation_budget(
            1.5,
            policy_variation=0.2,
            reward_variation_term=0.1,
            kernel_variation=0.3,
        )
        self.assertAlmostEqual(float(stats.total_variation), 0.6, places=7)
        self.assertAlmostEqual(float(stats.cumulative_variation), 2.1, places=7)

    def test_deterministic_action_variation_zero_for_identical_actions(self) -> None:
        import numpy as np

        actions = np.array([[0.1, -0.1], [0.2, -0.2]])
        value = deterministic_action_variation(actions, actions, reduction="max")
        self.assertAlmostEqual(float(value), 0.0, places=7)


class ACPQNMathTest(unittest.TestCase):
    def test_actor_loss_is_negative_mean_q(self) -> None:
        import numpy as np

        q_values = np.array([1.0, 3.0, 5.0])
        value = ac_pqn_actor_loss(q_values)
        self.assertAlmostEqual(float(value), -3.0, places=7)

    def test_td_target_matches_reward_plus_discounted_q(self) -> None:
        import numpy as np

        target = ac_pqn_td_target(
            np.array([1.0, 2.0]),
            np.array([0.9, 0.5]),
            np.array([10.0, 4.0]),
        )
        self.assertEqual(target.tolist(), [10.0, 4.0])

    def test_deterministic_action_distance_zero_when_actions_match(self) -> None:
        import numpy as np

        actions = np.zeros((8, 2))
        value = deterministic_action_distance(actions, actions)
        self.assertAlmostEqual(float(value), 0.0, places=7)

    def test_corrected_ac_pqn_actor_loss_activates(self) -> None:
        import numpy as np

        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/continuing_swimmer_ac_pqn_deviation.yaml"
        )
        base_loss = np.asarray(-1.0)
        candidate_actions = np.array([[0.0, 0.0], [0.5, 0.5]])
        deviation_actions = np.array(
            [
                [[0.0, 0.0], [0.5, 0.5]],
                [[1.0, 0.0], [0.0, 1.0]],
            ]
        )
        corrected, stats = corrected_ac_pqn_actor_loss(
            base_actor_loss=base_loss,
            candidate_score=1.0,
            deviation_scores=np.array([0.9, 1.8]),
            candidate_actions=candidate_actions,
            deviation_actions=deviation_actions,
            config=config.correction,
        )
        self.assertGreater(float(corrected), float(base_loss))
        self.assertTrue(bool(stats.active))


class ContinualBaselineMathTest(unittest.TestCase):
    def test_online_ewc_penalty_zero_at_reference(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/continuing_swimmer_ac_pqn_online_ewc.yaml"
        )
        import numpy as np

        penalty, stats = online_ewc_penalty(
            np.array([1.0, 2.0]),
            np.array([1.0, 2.0]),
            np.array([10.0, 20.0]),
            config.online_ewc,
        )
        self.assertAlmostEqual(float(penalty), 0.0, places=7)
        self.assertAlmostEqual(float(stats.penalty), 0.0, places=7)

    def test_clear_loss_combines_terms(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/continuing_swimmer_ac_pqn_clear.yaml"
        )
        total, _ = clear_loss(
            on_policy_loss=1.0,
            replay_rl_loss=2.0,
            policy_clone_loss=3.0,
            value_clone_loss=4.0,
            config=config.clear,
        )
        self.assertAlmostEqual(float(total), 8.0, places=7)

    def test_policy_consolidation_penalty_zero_when_matching_teachers(self) -> None:
        config = load_project_config(
            ROOT
            / "projects/continual_swimmer_jax/configs/continuing_swimmer_ac_pqn_policy_consolidation.yaml"
        )
        import numpy as np

        current_policy = np.array([[0.1, 0.2], [0.3, 0.4]])
        current_values = np.array([1.0, 2.0])
        teacher_policy = np.stack([current_policy, current_policy], axis=0)
        teacher_values = np.stack([current_values, current_values], axis=0)
        penalty, stats = policy_consolidation_penalty(
            current_policy,
            current_values,
            teacher_policy,
            teacher_values,
            config.policy_consolidation,
        )
        self.assertAlmostEqual(float(penalty), 0.0, places=7)
        self.assertAlmostEqual(float(stats.total_penalty), 0.0, places=7)


class RandomPolicyMathTest(unittest.TestCase):
    def test_continuous_random_actions_respect_shape_and_bounds(self) -> None:
        actions, stats = sample_random_actions(
            action_space="continuous",
            sample_shape=(8,),
            rng=0,
            action_dim=3,
            low=-0.5,
            high=0.5,
        )
        self.assertEqual(actions.shape, (8, 3))
        self.assertTrue((actions >= -0.5).all())
        self.assertTrue((actions <= 0.5).all())
        self.assertEqual(stats.sample_shape, (8, 3))

    def test_discrete_random_actions_respect_cardinality(self) -> None:
        actions, stats = sample_random_actions(
            action_space="discrete",
            sample_shape=(32,),
            rng=1,
            num_actions=5,
        )
        self.assertEqual(actions.shape, (32,))
        self.assertTrue(((actions >= 0) & (actions < 5)).all())
        self.assertEqual(stats.sample_shape, (32,))


if __name__ == "__main__":
    unittest.main()
