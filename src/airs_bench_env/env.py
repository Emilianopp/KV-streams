from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import verifiers as vf
import yaml
from datasets import Dataset
from verifiers.types import Messages, State

logger = logging.getLogger(__name__)

TASK_FILES = (
    "metadata.yaml",
    "project_description.md",
    "prepare.py",
    "evaluate_prepare.py",
    "evaluate.py",
    "custom_labels.py",
    "utils.py",
)

SYSTEM_PROMPT = """\
You are an autonomous machine-learning research agent running inside an AIRS-Bench task workspace.

You must solve the task by creating files and running commands. Work only inside the current workspace.
Use exactly one or more of these action tags in each response:

<write path="relative/path.py">
file contents
</write>

<cmd>
shell command to run from the workspace
</cmd>

<submit/>

Rules:
- The official AIRS-Bench task files are already in the workspace.
- The prepared data is mounted at ./data after setup.
- Produce the required submission.csv in the workspace before submitting.
- Do not invent labels or access test labels; the evaluator will expose labels only during official scoring.
- Keep commands bounded and reproducible. Prefer simple Python scripts over long shell pipelines.
"""


@dataclass(frozen=True)
class AirsTaskSpec:
    name: str
    path: Path
    project_description: str
    metadata: dict[str, Any]

    @property
    def logging_info(self) -> dict[str, Any]:
        info = self.metadata.get("logging_info", {})
        return info if isinstance(info, dict) else {}

    @property
    def metric_name(self) -> str:
        return str(self.logging_info.get("metric") or "score")

    @property
    def lower_is_better(self) -> bool:
        return bool(self.metadata.get("metric_lower_is_better", False))

    @property
    def estimated_worst_score(self) -> float | None:
        return _maybe_float(self.logging_info.get("estimated_worst_score"))

    @property
    def optimal_score(self) -> float | None:
        return _maybe_float(self.logging_info.get("optimal_score"))

    @property
    def sota_score(self) -> float | None:
        sota = self.logging_info.get("sota")
        if isinstance(sota, list) and sota:
            return _maybe_float(sota[0].get("sota_score"))
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _state_value(state: State | dict[str, Any], key: str, default: Any = None) -> Any:
    """Read rollout metadata from either top-level state or nested input row."""
    if hasattr(state, "get"):
        value = state.get(key, None)  # type: ignore[attr-defined]
        if value is not None:
            return value
        input_value = state.get("input", None)  # type: ignore[attr-defined]
        if isinstance(input_value, dict):
            return input_value.get(key, default)
    return default


def _state_debug_keys(state: State | dict[str, Any]) -> str:
    state_keys = sorted(str(key) for key in state.keys()) if hasattr(state, "keys") else []
    input_value = state.get("input", None) if hasattr(state, "get") else None  # type: ignore[attr-defined]
    input_keys = sorted(str(key) for key in input_value.keys()) if isinstance(input_value, dict) else []
    return f"state_keys={state_keys}, input_keys={input_keys}"


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(parts)
    return str(content)


def _last_assistant_text(messages: Messages) -> str:
    for message in reversed(messages):
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content", "")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", "")
        if role == "assistant":
            return _content_text(content)
    return ""


def _safe_relative_path(raw_path: str) -> Path:
    path = Path(raw_path.strip())
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path outside workspace: {raw_path!r}")
    if not path.parts:
        raise ValueError("Empty write path")
    return path


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    keep = max(0, max_chars - 120)
    return text[:keep] + f"\n\n[truncated {len(text) - keep} chars]\n"


def _find_airs_repo(airs_repo_path: str | None) -> Path:
    if airs_repo_path:
        root = Path(airs_repo_path).expanduser().resolve()
        if (root / "airsbench" / "tasks").is_dir():
            return root
        raise FileNotFoundError(f"AIRS repo path does not contain airsbench/tasks: {root}")

    try:
        import airsbench  # type: ignore
    except ImportError as exc:
        raise FileNotFoundError(
            "Could not import airsbench. Install AIRS-Bench or pass airs_repo_path. "
            "Example: uv pip install -e '.[airs-bench]'"
        ) from exc

    package_paths = list(getattr(airsbench, "__path__", []))
    for package_path in package_paths:
        candidate = Path(package_path).resolve().parent
        if (candidate / "airsbench" / "tasks").is_dir():
            return candidate
    raise FileNotFoundError("Imported airsbench but could not locate airsbench/tasks")


def _load_task_specs(
    *,
    airs_repo_path: str | None,
    suite: str,
    task_names: str | list[str] | None,
) -> list[AirsTaskSpec]:
    root = _find_airs_repo(airs_repo_path)
    tasks_root = root / "airsbench" / "tasks" / suite
    if not tasks_root.is_dir():
        raise FileNotFoundError(f"AIRS task suite not found: {tasks_root}")

    if task_names is None or task_names == "all":
        names = sorted(path.name for path in tasks_root.iterdir() if path.is_dir())
    elif isinstance(task_names, str):
        names = [name.strip() for name in task_names.split(",") if name.strip()]
    else:
        names = list(task_names)

    specs: list[AirsTaskSpec] = []
    missing: list[str] = []
    for name in names:
        task_path = tasks_root / name
        metadata_path = task_path / "metadata.yaml"
        description_path = task_path / "project_description.md"
        if not task_path.is_dir() or not metadata_path.exists() or not description_path.exists():
            missing.append(name)
            continue
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"metadata.yaml did not parse to a dict for {name}")
        specs.append(
            AirsTaskSpec(
                name=name,
                path=task_path,
                project_description=description_path.read_text(encoding="utf-8"),
                metadata=metadata,
            )
        )
    if missing:
        raise FileNotFoundError(f"Missing AIRS tasks under {tasks_root}: {', '.join(missing)}")
    if not specs:
        raise ValueError(f"No AIRS tasks selected from {tasks_root}")
    return specs


def _task_prompt(spec: AirsTaskSpec) -> list[dict[str, str]]:
    info = spec.logging_info
    summary = {
        "task": spec.name,
        "category": info.get("category"),
        "research_problem": info.get("research_problem"),
        "dataset": info.get("dataset"),
        "metric": spec.metric_name,
        "metric_lower_is_better": spec.lower_is_better,
        "sota_score": spec.sota_score,
        "optimal_score": spec.optimal_score,
    }
    user_content = (
        "AIRS-Bench task metadata:\n"
        f"{json.dumps(summary, indent=2, sort_keys=True)}\n\n"
        "Official project description:\n"
        f"{spec.project_description}\n\n"
        "Workspace contract:\n"
        "- AIRS task files are copied into the workspace root.\n"
        "- Prepared train/validation/test data is at ./data when setup succeeds.\n"
        "- Create ./submission.csv with the required schema, then respond with <submit/>.\n"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_dataset(specs: list[AirsTaskSpec], num_examples: int | None) -> Dataset:
    rows = []
    selected = specs if num_examples is None else specs[: max(0, min(num_examples, len(specs)))]
    for index, spec in enumerate(selected):
        rows.append(
            {
                "task": "airs-bench",
                "task_name": spec.name,
                "answer": spec.name,
                "prompt": _task_prompt(spec),
                "info": json.dumps(
                    {
                        "task_name": spec.name,
                        "metric": spec.metric_name,
                        "category": spec.logging_info.get("category"),
                    }
                ),
                "task_index": index,
            }
        )
    if not rows:
        raise ValueError("AIRS dataset selection is empty")
    return Dataset.from_list(rows)


class AirsBenchEnv(vf.MultiTurnEnv):
    """PrimeRL/verifiers wrapper for official AIRS-Bench RAD task specs."""

    def __init__(
        self,
        *,
        airs_repo_path: str | None = None,
        global_shared_data_dir: str,
        suite: str = "rad",
        task_names: str | list[str] | None = None,
        num_train_examples: int | None = None,
        num_eval_examples: int = 0,
        max_turns: int = 12,
        work_dir: str = "/network/scratch/d/dane.malenfant/kv-eviction/outputs/airs_bench_workdirs",
        command_timeout_seconds: int = 120,
        prepare_timeout_seconds: int = 600,
        evaluate_timeout_seconds: int = 600,
        max_command_output_chars: int = 12000,
        cleanup_workdirs: bool = True,
        require_data: bool = True,
        reuse_prepared_data: bool = False,
        prepared_data_cache_dir: str | None = None,
        prepare_cache_poll_seconds: float = 2.0,
        reward_mode: str = "normalized",
        allow_shell: bool = True,
        python_executable: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.specs = _load_task_specs(airs_repo_path=airs_repo_path, suite=suite, task_names=task_names)
        self.spec_by_name = {spec.name: spec for spec in self.specs}
        self.global_shared_data_dir = Path(global_shared_data_dir).expanduser()
        if require_data and not self.global_shared_data_dir.is_dir():
            raise FileNotFoundError(f"AIRS global_shared_data_dir does not exist: {self.global_shared_data_dir}")

        self.work_dir = Path(work_dir).expanduser()
        self.command_timeout_seconds = command_timeout_seconds
        self.prepare_timeout_seconds = prepare_timeout_seconds
        self.evaluate_timeout_seconds = evaluate_timeout_seconds
        self.max_command_output_chars = max_command_output_chars
        self.cleanup_workdirs = cleanup_workdirs
        self.require_data = require_data
        self.reuse_prepared_data = reuse_prepared_data
        self.prepared_data_cache_dir = (
            Path(prepared_data_cache_dir).expanduser()
            if prepared_data_cache_dir is not None
            else self.work_dir / "_prepared_data_cache"
        )
        self.prepare_cache_poll_seconds = prepare_cache_poll_seconds
        self.reward_mode = reward_mode
        self.allow_shell = allow_shell
        self.python_executable = python_executable or sys.executable

        train_dataset = _build_dataset(self.specs, num_train_examples)
        eval_dataset = _build_dataset(self.specs, num_eval_examples) if num_eval_examples > 0 else None

        rubric = vf.Rubric()
        rubric.add_reward_func(self.airs_reward, weight=1.0)
        rubric.add_reward_func(self.airs_valid_submission, weight=0.0)
        rubric.add_reward_func(self.airs_raw_score, weight=0.0)
        rubric.add_reward_func(self.airs_normalized_score, weight=0.0)
        rubric.add_reward_func(self.airs_command_success_rate, weight=0.0)
        rubric.add_reward_func(self.airs_turns, weight=0.0)

        super().__init__(
            dataset=train_dataset,
            eval_dataset=eval_dataset,
            max_turns=max_turns,
            rubric=rubric,
            parser=vf.Parser(),
            message_type="chat",
            **kwargs,
        )

    async def setup_state(self, state: State) -> State:
        maybe_state = await super().setup_state(state)
        if isinstance(maybe_state, dict):
            state = maybe_state

        task_name = _state_value(state, "task_name") or _state_value(state, "answer")
        if task_name is None:
            raise KeyError(f"Missing AIRS task_name in rollout state ({_state_debug_keys(state)})")
        if str(task_name) not in self.spec_by_name:
            raise KeyError(
                f"Unknown AIRS task_name {task_name!r}; available={sorted(self.spec_by_name)} "
                f"({_state_debug_keys(state)})"
            )

        spec = self.spec_by_name[str(task_name)]
        workspace = self.work_dir / f"{spec.name}-{uuid.uuid4().hex[:12]}"
        data_dir = workspace / "data"
        log_dir = workspace / "logs"
        workspace.mkdir(parents=True, exist_ok=False)
        log_dir.mkdir()

        for filename in TASK_FILES:
            src = spec.path / filename
            if src.exists():
                shutil.copy2(src, workspace / filename)

        if not self.reuse_prepared_data:
            data_dir.mkdir()

        state.update(
            {
                "airs_spec": spec,
                "task_name": spec.name,
                "airs_workspace": str(workspace),
                "airs_done": False,
                "airs_submitted": False,
                "airs_valid_submission": 0.0,
                "airs_raw_score": 0.0,
                "airs_normalized_score": 0.0,
                "airs_reward": 0.0,
                "airs_turns": 0,
                "airs_commands": 0,
                "airs_successful_commands": 0,
                "airs_last_error": "",
            }
        )

        if self.require_data:
            if self.reuse_prepared_data:
                result = self._prepare_or_reuse_cached_data(spec, workspace, data_dir, log_dir)
            else:
                result = await self._run_prepare_script(workspace, data_dir, log_dir)
            state["airs_prepare_output"] = result.output
        return state

    async def _run_prepare_script(self, workspace: Path, data_dir: Path, log_dir: Path) -> "_CommandResult":
        result = await self._run_script(
            workspace,
            [
                self.python_executable,
                "prepare.py",
                "--global-shared-data-dir",
                str(self.global_shared_data_dir),
                "--agent-data-mount-dir",
                str(data_dir),
                "--agent-log-dir",
                str(log_dir),
            ],
            timeout_seconds=self.prepare_timeout_seconds,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"AIRS prepare.py failed for {workspace.name} with exit {result.returncode}\n{result.output}"
            )
        return result

    def _prepare_or_reuse_cached_data(
        self,
        spec: AirsTaskSpec,
        workspace: Path,
        data_dir: Path,
        log_dir: Path,
    ) -> "_CommandResult":
        cache_root = self.prepared_data_cache_dir / spec.name
        cache_data_dir = cache_root / "data"
        ready_path = cache_root / "READY"
        error_path = cache_root / "ERROR"
        lock_path = cache_root.with_suffix(".lock")
        cache_root.parent.mkdir(parents=True, exist_ok=True)

        started = time.monotonic()
        while True:
            if ready_path.exists() and cache_data_dir.is_dir():
                self._attach_prepared_data(cache_data_dir, data_dir)
                return _CommandResult(
                    0,
                    f"Reused cached AIRS prepared data for {spec.name}: {cache_data_dir}",
                    time.monotonic() - started,
                )
            if error_path.exists():
                raise RuntimeError(
                    f"Cached AIRS prepare failed for {spec.name}; remove {cache_root} to retry.\n"
                    f"{error_path.read_text(encoding='utf-8', errors='replace')}"
                )

            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                if time.monotonic() - started > self.prepare_timeout_seconds:
                    raise TimeoutError(
                        f"Timed out waiting for AIRS prepared-data cache for {spec.name}: {cache_root}"
                    )
                time.sleep(self.prepare_cache_poll_seconds)
                continue

            os.close(fd)
            try:
                if ready_path.exists() and cache_data_dir.is_dir():
                    self._attach_prepared_data(cache_data_dir, data_dir)
                    return _CommandResult(
                        0,
                        f"Reused cached AIRS prepared data for {spec.name}: {cache_data_dir}",
                        time.monotonic() - started,
                    )

                shutil.rmtree(cache_root, ignore_errors=True)
                tmp_root = cache_root.with_name(f".{spec.name}-{uuid.uuid4().hex[:12]}.tmp")
                tmp_data_dir = tmp_root / "data"
                tmp_log_dir = tmp_root / "logs"
                tmp_data_dir.mkdir(parents=True, exist_ok=False)
                tmp_log_dir.mkdir(parents=True, exist_ok=False)

                result = self._run_subprocess(
                    workspace,
                    [
                        self.python_executable,
                        "prepare.py",
                        "--global-shared-data-dir",
                        str(self.global_shared_data_dir),
                        "--agent-data-mount-dir",
                        str(tmp_data_dir),
                        "--agent-log-dir",
                        str(tmp_log_dir),
                    ],
                    timeout_seconds=self.prepare_timeout_seconds,
                    shell=False,
                )
                if result.returncode != 0:
                    cache_root.mkdir(parents=True, exist_ok=True)
                    error_path.write_text(result.output, encoding="utf-8")
                    raise RuntimeError(
                        f"AIRS prepare.py failed for {spec.name} with exit {result.returncode}\n{result.output}"
                    )

                cache_root.mkdir(parents=True, exist_ok=True)
                if cache_data_dir.exists():
                    shutil.rmtree(cache_data_dir)
                shutil.move(str(tmp_data_dir), str(cache_data_dir))
                if (cache_root / "logs").exists():
                    shutil.rmtree(cache_root / "logs")
                shutil.move(str(tmp_log_dir), str(cache_root / "logs"))
                shutil.rmtree(tmp_root, ignore_errors=True)
                ready_path.write_text(
                    json.dumps({"task_name": spec.name, "created_at": time.time()}) + "\n",
                    encoding="utf-8",
                )
                self._attach_prepared_data(cache_data_dir, data_dir)
                return result
            finally:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass

    def _attach_prepared_data(self, cache_data_dir: Path, data_dir: Path) -> None:
        if data_dir.is_symlink() or data_dir.exists():
            if data_dir.is_dir() and not data_dir.is_symlink():
                shutil.rmtree(data_dir)
            else:
                data_dir.unlink()
        os.symlink(cache_data_dir, data_dir, target_is_directory=True)

    async def env_response(self, messages: Messages, state: State, **_: Any) -> Messages:
        if state.get("airs_done"):
            return []

        state["airs_turns"] = int(state.get("airs_turns", 0)) + 1
        workspace = Path(str(state["airs_workspace"]))
        assistant_text = _last_assistant_text(messages)

        try:
            feedback = await self._handle_actions(assistant_text, workspace, state)
        except Exception as exc:
            state["airs_last_error"] = repr(exc)
            feedback = f"Action failed: {type(exc).__name__}: {exc}"

        if state.get("airs_done"):
            return [vf.UserMessage(content=feedback)]

        guidance = (
            "\n\nNext action: write/update files, run one command, or submit once "
            "workspace/submission.csv is ready."
        )
        return [vf.UserMessage(content=feedback + guidance)]

    async def _handle_actions(self, text: str, workspace: Path, state: State) -> str:
        write_summaries = self._apply_writes(text, workspace)
        command_summary = ""
        command = self._extract_command(text)
        if command:
            if not self.allow_shell:
                raise RuntimeError("Shell commands are disabled for this AIRS environment")
            state["airs_commands"] = int(state.get("airs_commands", 0)) + 1
            result = await self._run_shell(workspace, command, timeout_seconds=self.command_timeout_seconds)
            if result.returncode == 0:
                state["airs_successful_commands"] = int(state.get("airs_successful_commands", 0)) + 1
            command_summary = (
                f"Command exit code: {result.returncode}\n"
                f"Elapsed seconds: {result.elapsed_seconds:.2f}\n"
                f"{result.output}"
            )

        if self._wants_submit(text):
            return await self._evaluate_submission(workspace, state, write_summaries, command_summary)

        if not write_summaries and not command_summary:
            return (
                "No valid AIRS action was found. Use <write path=\"...\">...</write>, "
                "<cmd>...</cmd>, or <submit/>."
            )

        sections = []
        if write_summaries:
            sections.append("Writes:\n" + "\n".join(write_summaries))
        if command_summary:
            sections.append("Command result:\n" + command_summary)
        sections.append(self._workspace_summary(workspace))
        return "\n\n".join(sections)

    def _apply_writes(self, text: str, workspace: Path) -> list[str]:
        summaries: list[str] = []
        pattern = re.compile(
            r"<write\s+path=[\"']([^\"']+)[\"']\s*>(.*?)</write>",
            flags=re.IGNORECASE | re.DOTALL,
        )
        for match in pattern.finditer(text):
            rel_path = _safe_relative_path(match.group(1))
            dest = workspace / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            content = match.group(2)
            if content.startswith("\n"):
                content = content[1:]
            if content.endswith("\n"):
                content = content[:-1]
            dest.write_text(content, encoding="utf-8")
            summaries.append(f"- wrote {rel_path.as_posix()} ({len(content)} chars)")
        return summaries

    def _extract_command(self, text: str) -> str:
        match = re.search(r"<cmd\s*>(.*?)</cmd>", text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return match.group(1).strip()

    def _wants_submit(self, text: str) -> bool:
        return bool(re.search(r"<submit\s*/\s*>|<submit\s*>", text, flags=re.IGNORECASE))

    async def _evaluate_submission(
        self,
        workspace: Path,
        state: State,
        write_summaries: list[str],
        command_summary: str,
    ) -> str:
        spec = state["airs_spec"]
        submission_path = workspace / "submission.csv"
        if not submission_path.exists():
            state["airs_done"] = True
            state["airs_last_error"] = "submission.csv missing"
            return "Submission failed: workspace/submission.csv does not exist. Reward: 0.0"

        data_dir = workspace / "data"
        prep_result = await self._run_script(
            workspace,
            [
                self.python_executable,
                "evaluate_prepare.py",
                "--global-shared-data-dir",
                str(self.global_shared_data_dir),
                "--agent-data-mount-dir",
                str(data_dir),
                "--agent-log-dir",
                str(workspace),
            ],
            timeout_seconds=self.evaluate_timeout_seconds,
        )
        if prep_result.returncode != 0:
            state["airs_done"] = True
            state["airs_last_error"] = "evaluate_prepare.py failed"
            return (
                "Submission failed during evaluate_prepare.py.\n"
                f"Exit code: {prep_result.returncode}\n{prep_result.output}\nReward: 0.0"
            )

        eval_result = await self._run_script(
            workspace,
            [self.python_executable, "evaluate.py", "--submission-file", str(submission_path)],
            timeout_seconds=self.evaluate_timeout_seconds,
        )
        if eval_result.returncode != 0:
            state["airs_done"] = True
            state["airs_last_error"] = "evaluate.py failed"
            return (
                "Submission failed during evaluate.py.\n"
                f"Exit code: {eval_result.returncode}\n{eval_result.output}\nReward: 0.0"
            )

        scores = self._parse_eval_json(eval_result.output)
        raw_score = self._extract_metric_score(scores, spec)
        normalized = self._normalize_score(raw_score, spec)
        if not math.isfinite(raw_score) or not math.isfinite(normalized):
            raise RuntimeError(
                f"AIRS evaluator returned non-finite score for {spec.name}: "
                f"raw_score={raw_score!r}, normalized={normalized!r}, scores={scores!r}"
            )

        reward = normalized if self.reward_mode == "normalized" else raw_score
        state.update(
            {
                "airs_done": True,
                "airs_submitted": True,
                "airs_valid_submission": 1.0,
                "airs_raw_score": float(raw_score),
                "airs_normalized_score": float(normalized),
                "airs_reward": float(reward),
            }
        )

        sections = []
        if write_summaries:
            sections.append("Writes:\n" + "\n".join(write_summaries))
        if command_summary:
            sections.append("Command result before submit:\n" + command_summary)
        sections.append(
            "Submission accepted.\n"
            f"Metric: {spec.metric_name}\n"
            f"Raw score: {raw_score:.10g}\n"
            f"Normalized AIRS score: {normalized:.10g}\n"
            f"Reward: {reward:.10g}\n"
            f"Evaluator output:\n{eval_result.output}"
        )
        return "\n\n".join(sections)

    def _parse_eval_json(self, output: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        best: dict[str, Any] | None = None
        for index, char in enumerate(output):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(output[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                best = value
        if best is None:
            raise RuntimeError(f"Could not parse evaluator JSON from output:\n{output}")
        return best

    def _extract_metric_score(self, scores: dict[str, Any], spec: AirsTaskSpec) -> float:
        if spec.metric_name in scores:
            score = _maybe_float(scores[spec.metric_name])
        elif len(scores) == 1:
            score = _maybe_float(next(iter(scores.values())))
        else:
            score = None
        if score is None:
            raise RuntimeError(f"Evaluator scores do not contain finite metric {spec.metric_name!r}: {scores!r}")
        return score

    def _normalize_score(self, score: float, spec: AirsTaskSpec) -> float:
        worst = spec.estimated_worst_score
        sota = spec.sota_score
        optimal = spec.optimal_score
        if worst is None or sota is None or optimal is None:
            if spec.lower_is_better:
                return -score
            return score

        eps = 1e-12

        def phi(value: float) -> float:
            return -math.log10(max(abs(value - optimal), eps))

        denom = phi(sota) - phi(worst)
        if abs(denom) < eps:
            raise RuntimeError(f"Degenerate AIRS normalization denominator for task {spec.name}")
        return (phi(score) - phi(worst)) / denom

    def _workspace_summary(self, workspace: Path) -> str:
        files = []
        for path in sorted(workspace.iterdir()):
            if path.name in {"data", "logs", "__pycache__"}:
                continue
            if path.is_file():
                files.append(f"{path.name} ({path.stat().st_size} bytes)")
        if not files:
            return "Workspace files: no files visible."
        return "Workspace files: " + ", ".join(files[:20])

    async def _run_script(self, cwd: Path, args: list[str], timeout_seconds: int) -> "_CommandResult":
        return self._run_subprocess(cwd, args, timeout_seconds, shell=False)

    async def _run_shell(self, cwd: Path, command: str, timeout_seconds: int) -> "_CommandResult":
        return self._run_subprocess(cwd, command, timeout_seconds, shell=True)

    def _run_subprocess(
        self,
        cwd: Path,
        args: list[str] | str,
        timeout_seconds: int,
        shell: bool,
    ) -> "_CommandResult":
        started = time.monotonic()
        env = self._subprocess_env(cwd)
        try:
            completed = subprocess.run(
                args,
                cwd=str(cwd),
                env=env,
                shell=shell,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            output = _truncate(
                f"TIMEOUT after {timeout_seconds}s\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}",
                self.max_command_output_chars,
            )
            return _CommandResult(124, output, time.monotonic() - started)

        output = _truncate(
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}",
            self.max_command_output_chars,
        )
        return _CommandResult(completed.returncode, output, time.monotonic() - started)

    def _subprocess_env(self, cwd: Path) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "AIRS_WORKSPACE": str(cwd),
                "HF_DATASETS_OFFLINE": env.get("HF_DATASETS_OFFLINE", "1"),
                "TRANSFORMERS_OFFLINE": env.get("TRANSFORMERS_OFFLINE", "1"),
                "WANDB_DISABLED": env.get("WANDB_DISABLED", "true"),
                "TOKENIZERS_PARALLELISM": env.get("TOKENIZERS_PARALLELISM", "false"),
                "PYTHONPATH": f"{cwd}{os.pathsep}{env.get('PYTHONPATH', '')}",
            }
        )
        return env

    async def airs_reward(self, state: State, **_: Any) -> float:
        return float(state.get("airs_reward", 0.0))

    async def airs_valid_submission(self, state: State, **_: Any) -> float:
        return float(state.get("airs_valid_submission", 0.0))

    async def airs_raw_score(self, state: State, **_: Any) -> float:
        return float(state.get("airs_raw_score", 0.0))

    async def airs_normalized_score(self, state: State, **_: Any) -> float:
        return float(state.get("airs_normalized_score", 0.0))

    async def airs_command_success_rate(self, state: State, **_: Any) -> float:
        total = int(state.get("airs_commands", 0))
        if total == 0:
            return 0.0
        return float(state.get("airs_successful_commands", 0)) / total

    async def airs_turns(self, state: State, **_: Any) -> float:
        return float(state.get("airs_turns", 0))

    @vf.stop
    async def airs_done(self, state: State, **_: Any) -> bool:
        return bool(state.get("airs_done", False))

    @vf.cleanup
    async def cleanup_workspace(self, state: State, **_: Any) -> None:
        if not self.cleanup_workdirs:
            return
        workspace = state.get("airs_workspace")
        if workspace:
            shutil.rmtree(workspace, ignore_errors=True)


@dataclass(frozen=True)
class _CommandResult:
    returncode: int
    output: str
    elapsed_seconds: float


def load_environment(**kwargs: Any) -> AirsBenchEnv:
    """Entry point for `vf.load_environment("airs-bench-env", ...)`."""
    return AirsBenchEnv(**kwargs)
