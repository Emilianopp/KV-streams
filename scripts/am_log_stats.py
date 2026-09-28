#!/usr/bin/env python3
"""Summarize attention-matching compaction timings from a vLLM inference log."""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


LOCAL_RANK_RE = re.compile(r"\(Worker pid=(\d+)\).*local_rank=(\d+)")
ELAPSED_RE = re.compile(r"\(Worker pid=(\d+)\).*elapsed=([0-9.]+)s")


def format_stats(values: list[float]) -> str:
    return (
        f"count={len(values)} "
        f"mean={mean(values):.3f}s "
        f"min={min(values):.3f}s "
        f"max={max(values):.3f}s "
        f"last={values[-1]:.3f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print per-GPU/worker attention-matching timing stats from an inference.log file.",
    )
    parser.add_argument("log_path", type=Path, help="Path to inference.log")
    args = parser.parse_args()

    lines = args.log_path.read_text(errors="ignore").splitlines()

    local_rank_to_pid: dict[int, int] = {}
    for line in lines:
        match = LOCAL_RANK_RE.search(line)
        if match:
            pid = int(match.group(1))
            local_rank = int(match.group(2))
            local_rank_to_pid[local_rank] = pid

    pid_to_rank = {pid: rank for rank, pid in local_rank_to_pid.items()}
    elapsed_by_pid: dict[int, list[float]] = defaultdict(list)
    for line in lines:
        match = ELAPSED_RE.search(line)
        if match:
            pid = int(match.group(1))
            elapsed_by_pid[pid].append(float(match.group(2)))

    if not elapsed_by_pid:
        print("No AM compaction timing events found.")
        return

    print(f"log: {args.log_path}")
    print(f"workers_with_events: {len(elapsed_by_pid)}")
    print(f"total_events: {sum(len(values) for values in elapsed_by_pid.values())}")
    print()

    for pid in sorted(elapsed_by_pid, key=lambda item: pid_to_rank.get(item, item)):
        local_rank = pid_to_rank.get(pid)
        gpu_label = "unknown" if local_rank is None else str(local_rank)
        print(f"gpu={gpu_label} pid={pid} {format_stats(elapsed_by_pid[pid])}")

    all_values = [value for values in elapsed_by_pid.values() for value in values]
    print()
    print(f"overall {format_stats(all_values)}")


if __name__ == "__main__":
    main()
