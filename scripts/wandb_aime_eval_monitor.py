#!/usr/bin/env python3
"""Lightweight W&B monitor for local AIME eval runs."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import wandb


def _results_path(run_dir: Path) -> Path | None:
    direct = run_dir / "vf_eval" / "results.jsonl"
    if direct.exists():
        return direct
    matches = sorted(
        run_dir.glob("vf_eval/**/results.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return matches[0] if matches else None


def _load_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _correct(row: dict[str, Any]) -> float:
    metrics = row.get("metrics") or {}
    if isinstance(metrics, dict) and "correct_answer" in metrics:
        return float(metrics["correct_answer"])
    if "correct_answer" in row:
        return float(row["correct_answer"])
    return float(row.get("reward", 0.0) or 0.0)


def _num_compactions(row: dict[str, Any]) -> int | None:
    if "num_compaction_events" in row and row["num_compaction_events"] is not None:
        return int(row["num_compaction_events"])
    events = row.get("compaction_events")
    if isinstance(events, list):
        return len(events)
    metrics = row.get("metrics") or {}
    if isinstance(metrics, dict) and "num_compaction_events" in metrics:
        return int(metrics["num_compaction_events"])
    return None


def _num_shuffles(row: dict[str, Any]) -> int | None:
    if "num_shuffle_events" in row and row["num_shuffle_events"] is not None:
        return int(row["num_shuffle_events"])
    events = row.get("shuffle_events")
    if isinstance(events, list):
        return len(events)
    metrics = row.get("metrics") or {}
    if isinstance(metrics, dict) and "num_shuffle_events" in metrics:
        return int(metrics["num_shuffle_events"])
    return None


def _num_noise_events(row: dict[str, Any]) -> int | None:
    if "num_noise_events" in row and row["num_noise_events"] is not None:
        return int(row["num_noise_events"])
    events = row.get("noise_events")
    if isinstance(events, list):
        return len(events)
    metrics = row.get("metrics") or {}
    if isinstance(metrics, dict) and "num_noise_events" in metrics:
        return int(metrics["num_noise_events"])
    return None


def _parse_timing(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.endswith("_seconds"):
            try:
                out[key] = float(value)
            except ValueError:
                continue
    return out


def _parse_inference_log(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics

    latest_prompt_tps: float | None = None
    latest_generation_tps: float | None = None
    latest_gpu_kv_usage: float | None = None
    total_compactions = 0
    compaction_elapsed: list[float] = []

    throughput_re = re.compile(
        r"Avg prompt throughput:\s*([0-9.]+)\s+tokens/s,\s*"
        r"Avg generation throughput:\s*([0-9.]+)\s+tokens/s,.*?"
        r"GPU KV cache usage:\s*([0-9.]+)%"
    )
    compaction_re = re.compile(r"\[AM\] compacted request")
    compaction_elapsed_re = re.compile(r"\[AM\] finished compaction .* elapsed=([0-9.]+)s")

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = throughput_re.search(line)
            if match:
                latest_prompt_tps = float(match.group(1))
                latest_generation_tps = float(match.group(2))
                latest_gpu_kv_usage = float(match.group(3))
            if compaction_re.search(line):
                total_compactions += 1
            elapsed_match = compaction_elapsed_re.search(line)
            if elapsed_match:
                compaction_elapsed.append(float(elapsed_match.group(1)))

    if latest_prompt_tps is not None:
        metrics["prompt_tps"] = latest_prompt_tps
    if latest_generation_tps is not None:
        metrics["generation_tps"] = latest_generation_tps
    if latest_gpu_kv_usage is not None:
        metrics["gpu_kv_cache_usage_pct"] = latest_gpu_kv_usage
    metrics["total_compactions_seen"] = float(total_compactions)
    if compaction_elapsed:
        metrics["avg_compaction_seconds"] = _mean(compaction_elapsed)
        metrics["last_compaction_seconds"] = compaction_elapsed[-1]
    return metrics


def _launcher_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _build_metrics(run_dir: Path) -> tuple[int, list[dict[str, Any]], dict[str, float], dict[str, float]]:
    rows = _load_rows(_results_path(run_dir))
    completed = len(rows)
    timing = _parse_timing(run_dir / "wall_clock_times.txt")
    inference_metrics = _parse_inference_log(run_dir / "inference.log")
    metrics: dict[str, float] = {"completed": float(completed)}

    if rows:
        metrics["accuracy"] = _mean([_correct(r) for r in rows])
        metrics["avg_reward"] = _mean(
            [float(r.get("reward", 0.0) or 0.0) for r in rows]
        )
        metrics["avg_output_tokens"] = _mean(
            [
                float((r.get("token_usage") or {}).get("output_tokens", 0.0) or 0.0)
                for r in rows
            ]
        )
        metrics["truncation_rate"] = _mean(
            [1.0 if r.get("is_truncated") else 0.0 for r in rows]
        )
        compaction_counts = [_num_compactions(r) for r in rows]
        known_counts = [float(c) for c in compaction_counts if c is not None]
        if known_counts:
            metrics["avg_num_compactions"] = _mean(known_counts)
            unique_counts = sorted({int(c) for c in known_counts})
            for n in unique_counts:
                bucket = [
                    _correct(r)
                    for r, c in zip(rows, compaction_counts, strict=False)
                    if c == n
                ]
                if bucket:
                    metrics[f"accuracy_n_compactions/{n}"] = _mean(bucket)
                    metrics[f"count_n_compactions/{n}"] = float(len(bucket))
        shuffle_counts = [_num_shuffles(r) for r in rows]
        known_shuffles = [float(c) for c in shuffle_counts if c is not None]
        if known_shuffles:
            metrics["avg_num_shuffles"] = _mean(known_shuffles)
            unique_shuffles = sorted({int(c) for c in known_shuffles})
            for n in unique_shuffles:
                bucket = [
                    _correct(r)
                    for r, c in zip(rows, shuffle_counts, strict=False)
                    if c == n
                ]
                if bucket:
                    metrics[f"accuracy_n_shuffles/{n}"] = _mean(bucket)
                    metrics[f"count_n_shuffles/{n}"] = float(len(bucket))
        noise_counts = [_num_noise_events(r) for r in rows]
        known_noise = [float(c) for c in noise_counts if c is not None]
        if known_noise:
            metrics["avg_num_noise_events"] = _mean(known_noise)
            unique_noise = sorted({int(c) for c in known_noise})
            for n in unique_noise:
                bucket = [
                    _correct(r)
                    for r, c in zip(rows, noise_counts, strict=False)
                    if c == n
                ]
                if bucket:
                    metrics[f"accuracy_n_noise_events/{n}"] = _mean(bucket)
                    metrics[f"count_n_noise_events/{n}"] = float(len(bucket))

    metrics.update(timing)
    metrics.update(inference_metrics)
    return completed, rows, metrics, timing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--launcher-pid", type=int)
    parser.add_argument("--wandb-run-id")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--one-shot", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("WANDB_API_KEY"):
        return 0

    run_dir = Path(args.run_dir).expanduser()
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", "kv-eviction"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        name=run_dir.name,
        job_type="aime-eval",
        config={
            "mode": args.mode,
            "profile": args.profile,
            "model": args.model,
            "run_dir": str(run_dir),
            "expected_count": args.expected_count,
        },
        tags=["aime", args.mode, args.profile],
        id=args.wandb_run_id,
        resume=args.resume,
        reinit="finish_previous",
    )
    print(f"wandb_run_url={run.url}", flush=True)

    if args.one_shot:
        completed, rows, metrics, _timing = _build_metrics(run_dir)
        wandb.log(metrics, step=completed)
        wandb.summary["status"] = (
            "completed" if completed >= args.expected_count else "ended_early"
        )
        wandb.summary["completed"] = completed
        wandb.summary["expected_count"] = args.expected_count
        if rows:
            wandb.summary["accuracy"] = _mean([_correct(r) for r in rows])
        run.finish()
        return 0

    last_logged_count = -1
    while True:
        completed, rows, metrics, timing = _build_metrics(run_dir)

        if completed != last_logged_count or timing:
            wandb.log(metrics, step=completed)
            last_logged_count = completed

        if completed >= args.expected_count and timing:
            wandb.summary["status"] = "completed"
            break

        if not _launcher_alive(args.launcher_pid):
            wandb.summary["status"] = (
                "completed" if completed >= args.expected_count else "ended_early"
            )
            break

        time.sleep(args.poll_seconds)

    wandb.summary["completed"] = completed
    wandb.summary["expected_count"] = args.expected_count
    if rows:
        wandb.summary["accuracy"] = _mean([_correct(r) for r in rows])
    run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
