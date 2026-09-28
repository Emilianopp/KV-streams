"""Unit tests for the temporal deviation scaffold."""

from __future__ import annotations

import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from continual_deviation.benchmarks import build_checkpoint_schedule
from continual_deviation.config import BenchmarkConfig, load_project_config
from continual_deviation.representation import (
    capture_layer_outputs,
    compare_representations,
    linear_cka,
)
from continual_deviation.runtime import (
    configure_torch_runtime,
    move_batch_to_device,
    resolve_device,
    resolve_dtype,
)
from continual_deviation.update import (
    DeviationCandidate,
    corrected_policy_loss,
    select_reference_deviation,
)


class DeviationSelectionTest(unittest.TestCase):
    def test_select_reference_chooses_best_better_policy(self) -> None:
        current_score = 3.0
        deviations = [
            DeviationCandidate(
                name="older",
                score=3.1,
                log_probs=torch.zeros(1, 2),
            ),
            DeviationCandidate(
                name="best",
                score=3.6,
                kind="future",
                log_probs=torch.zeros(1, 2),
            ),
        ]

        selected = select_reference_deviation(current_score, deviations)

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.name, "best")
        self.assertEqual(selected.kind, "future")

    def test_corrected_loss_is_inactive_without_better_deviation(self) -> None:
        base_loss = torch.tensor(0.4)
        candidate_log_probs = torch.log_softmax(torch.randn(3, 4), dim=-1)
        corrected, result = corrected_policy_loss(
            base_loss=base_loss,
            candidate_log_probs=candidate_log_probs,
            candidate_score=5.0,
            deviations=[
                DeviationCandidate(
                    name="worse",
                    score=4.9,
                    log_probs=candidate_log_probs.clone(),
                )
            ],
            config=load_project_config(
                ROOT / "projects/continual_swimmer/configs/continuing_swimmer.yaml"
            ).correction,
        )

        self.assertFalse(result.active)
        self.assertEqual(float(corrected), float(base_loss))
        self.assertEqual(float(result.penalty), 0.0)


class RepresentationMetricsTest(unittest.TestCase):
    def test_linear_cka_is_one_for_identical_features(self) -> None:
        features = torch.randn(32, 8)
        score = linear_cka(features, features)
        self.assertAlmostEqual(score, 1.0, places=5)

    def test_compare_representations_reports_probe_score(self) -> None:
        torch.manual_seed(0)
        reference = torch.randn(64, 6)
        current = reference + 0.05 * torch.randn(64, 6)
        targets = 0.5 * current[:, 0] - 0.3 * current[:, 1]
        report = compare_representations(reference, current, targets)
        self.assertGreater(report.linear_cka, 0.9)
        self.assertLess(report.cosine_drift, 0.2)
        self.assertIsNotNone(report.ridge_probe_r2)
        assert report.ridge_probe_r2 is not None
        self.assertGreater(report.ridge_probe_r2, 0.9)

    def test_capture_layer_outputs_collects_requested_layers(self) -> None:
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 2),
        )
        with capture_layer_outputs(model, ["0", "2"]) as outputs:
            _ = model(torch.randn(3, 4))
        self.assertEqual(set(outputs), {"0", "2"})
        self.assertEqual(outputs["0"].shape, (3, 8))
        self.assertEqual(outputs["2"].shape, (3, 2))


class ConfigAndBenchmarkTest(unittest.TestCase):
    def test_load_project_config_round_trips_yaml(self) -> None:
        yaml_text = textwrap.dedent(
            """
            project_name: test-project
            benchmark:
              total_steps: 1000
              checkpoint_interval: 250
              seeds: [1, 3]
            correction:
              enabled: false
            """
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(yaml_text, encoding="utf-8")
            config = load_project_config(path)

        self.assertEqual(config.project_name, "test-project")
        self.assertEqual(config.benchmark.total_steps, 1000)
        self.assertEqual(config.benchmark.seeds, (1, 3))
        self.assertFalse(config.correction.enabled)

    def test_checkpoint_schedule_covers_final_step(self) -> None:
        schedule = build_checkpoint_schedule(
            BenchmarkConfig(total_steps=1100, checkpoint_interval=500)
        )
        self.assertEqual(schedule, [500, 1000, 1100])


class RuntimeTest(unittest.TestCase):
    def test_resolve_dtype_aliases(self) -> None:
        self.assertEqual(resolve_dtype("fp32"), torch.float32)
        self.assertEqual(resolve_dtype("bf16"), torch.bfloat16)

    def test_resolve_device_auto_returns_available_backend(self) -> None:
        device = resolve_device("auto")
        self.assertIn(device.type, {"cpu", "cuda", "mps"})

    def test_configure_runtime_returns_device(self) -> None:
        config = load_project_config(
            ROOT / "projects/continual_swimmer/configs/continuing_swimmer.yaml"
        ).runtime
        device = configure_torch_runtime(config)
        self.assertIn(device.type, {"cpu", "cuda", "mps"})

    def test_move_batch_to_device_handles_nested_structures(self) -> None:
        device = torch.device("cpu")
        batch = {
            "obs": torch.randn(2, 3),
            "meta": [torch.randn(1), {"reward": torch.randn(1)}],
        }
        moved = move_batch_to_device(batch, device)
        self.assertEqual(moved["obs"].device.type, "cpu")
        self.assertEqual(moved["meta"][0].device.type, "cpu")
        self.assertEqual(moved["meta"][1]["reward"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
