from __future__ import annotations

import asyncio
import math
from pathlib import Path

import pytest

from airs_bench_env import load_environment
from airs_bench_env.env import _safe_relative_path


def _write_fake_airs_repo(root: Path) -> Path:
    task_dir = root / "airsbench" / "tasks" / "rad" / "FakeMetricTask"
    task_dir.mkdir(parents=True)
    (task_dir / "metadata.yaml").write_text(
        """
logging_info:
  category: Unit Test
  research_problem: Fake task
  dataset: FakeData
  metric: FakeMetric
  estimated_worst_score: 0.0
  optimal_score: 1.0
  sota:
    - sota_score: 0.9
metric_lower_is_better: false
""",
        encoding="utf-8",
    )
    (task_dir / "project_description.md").write_text(
        "Create a submission.csv file and submit it.",
        encoding="utf-8",
    )
    (task_dir / "prepare.py").write_text(
        """
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--global-shared-data-dir", required=True)
parser.add_argument("--agent-data-mount-dir", required=True)
parser.add_argument("--agent-log-dir", required=False)
args = parser.parse_args()
data_dir = Path(args.agent_data_mount_dir)
data_dir.mkdir(parents=True, exist_ok=True)
(data_dir / "train.txt").write_text("train")
(data_dir / "test.txt").write_text("test")
print("prepared")
""",
        encoding="utf-8",
    )
    (task_dir / "evaluate_prepare.py").write_text(
        """
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--global-shared-data-dir", required=True)
parser.add_argument("--agent-data-mount-dir", required=True)
parser.add_argument("--agent-log-dir", required=False)
parser.parse_args()
print("evaluate prepared")
""",
        encoding="utf-8",
    )
    (task_dir / "evaluate.py").write_text(
        """
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--submission-file", required=True)
args = parser.parse_args()
open(args.submission_file).read()
print("--- EVALUATION RESULT ---")
print(json.dumps({"FakeMetric": 0.5}))
""",
        encoding="utf-8",
    )
    return task_dir


def test_safe_relative_path_rejects_escape() -> None:
    assert _safe_relative_path("subdir/file.py") == Path("subdir/file.py")
    with pytest.raises(ValueError):
        _safe_relative_path("../escape.py")
    with pytest.raises(ValueError):
        _safe_relative_path("/tmp/escape.py")


def test_fake_airs_task_submit_scores_official_scripts(tmp_path: Path) -> None:
    async def run() -> None:
        airs_repo = tmp_path / "airs-repo"
        data_root = tmp_path / "global-data"
        work_root = tmp_path / "work"
        data_root.mkdir()
        _write_fake_airs_repo(airs_repo)

        env = load_environment(
            airs_repo_path=str(airs_repo),
            global_shared_data_dir=str(data_root),
            require_data=True,
            task_names="FakeMetricTask",
            num_train_examples=1,
            max_turns=2,
            work_dir=str(work_root),
            cleanup_workdirs=False,
            prepare_timeout_seconds=10,
            evaluate_timeout_seconds=10,
            command_timeout_seconds=10,
        )

        dataset = env.get_dataset()
        assert len(dataset) == 1
        assert dataset[0]["task_name"] == "FakeMetricTask"

        state = await env.setup_state({"task_name": "FakeMetricTask"})
        workspace = Path(state["airs_workspace"])
        assert (workspace / "metadata.yaml").exists()
        assert (workspace / "data" / "train.txt").read_text(encoding="utf-8") == "train"

        feedback = await env._handle_actions(
            '<write path="submission.csv">\nprediction\n</write><submit/>',
            workspace,
            state,
        )
        assert "Submission accepted" in feedback
        assert state["airs_valid_submission"] == 1.0
        assert state["airs_raw_score"] == 0.5
        assert math.isfinite(state["airs_reward"])
        assert state["airs_reward"] == state["airs_normalized_score"]

    asyncio.run(run())


def test_fake_airs_task_setup_accepts_nested_input_state(tmp_path: Path) -> None:
    async def run() -> None:
        airs_repo = tmp_path / "airs-repo"
        data_root = tmp_path / "global-data"
        work_root = tmp_path / "work"
        data_root.mkdir()
        _write_fake_airs_repo(airs_repo)

        env = load_environment(
            airs_repo_path=str(airs_repo),
            global_shared_data_dir=str(data_root),
            require_data=False,
            task_names="FakeMetricTask",
            num_train_examples=1,
            work_dir=str(work_root),
        )

        state = await env.setup_state({"input": {"task_name": "FakeMetricTask"}})
        assert state["task_name"] == "FakeMetricTask"
        assert state["airs_spec"].name == "FakeMetricTask"
        assert Path(state["airs_workspace"]).exists()

    asyncio.run(run())


def test_fake_airs_command_action_runs_in_workspace(tmp_path: Path) -> None:
    async def run() -> None:
        airs_repo = tmp_path / "airs-repo"
        data_root = tmp_path / "global-data"
        work_root = tmp_path / "work"
        data_root.mkdir()
        _write_fake_airs_repo(airs_repo)

        env = load_environment(
            airs_repo_path=str(airs_repo),
            global_shared_data_dir=str(data_root),
            require_data=False,
            task_names="FakeMetricTask",
            num_train_examples=1,
            work_dir=str(work_root),
            command_timeout_seconds=10,
        )
        state = await env.setup_state({"task_name": "FakeMetricTask"})
        workspace = Path(state["airs_workspace"])
        feedback = await env._handle_actions(
            '<write path="hello.py">\nprint("hello airs")\n</write><cmd>python hello.py</cmd>',
            workspace,
            state,
        )
        assert "hello airs" in feedback
        assert (workspace / "hello.py").exists()
        assert state["airs_commands"] == 1
        assert state["airs_successful_commands"] == 1

    asyncio.run(run())
