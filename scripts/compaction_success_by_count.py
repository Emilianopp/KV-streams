#!/usr/bin/env python3
"""Report success rates grouped by the number of compaction events."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def _results_path(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if path.is_dir():
        direct = path / "results.jsonl"
        if direct.exists():
            return direct
        matches = sorted(
            path.glob("**/results.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if matches:
            return matches[0]
    if not path.exists():
        raise FileNotFoundError(f"missing results file: {path}")
    return path


def _is_success(row: dict[str, Any]) -> bool:
    metrics = row.get("metrics") or {}
    if isinstance(metrics, dict) and "correct_answer" in metrics:
        return bool(float(metrics["correct_answer"]))
    if "correct_answer" in row:
        return bool(float(row["correct_answer"]))
    return float(row.get("reward", 0.0) or 0.0) > 0.0


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


def main() -> int:
    if len(sys.argv) != 2:
        print(
            "usage: scripts/compaction_success_by_count.py "
            "<run-dir|results.jsonl>",
            file=sys.stderr,
        )
        return 2

    results_path = _results_path(sys.argv[1])
    buckets: dict[int | None, list[bool]] = defaultdict(list)
    rows = 0
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows += 1
            buckets[_num_compactions(row)].append(_is_success(row))

    print(f"Compaction-count success rates for {results_path}")
    print("N_compactions  successes  total  accuracy")
    for n in sorted(k for k in buckets if k is not None):
        total = len(buckets[n])
        successes = sum(1 for ok in buckets[n] if ok)
        accuracy = successes / total if total else 0.0
        print(f"{n:13d}  {successes:9d}  {total:5d}  {accuracy:8.3f}")
    if None in buckets:
        total = len(buckets[None])
        successes = sum(1 for ok in buckets[None] if ok)
        accuracy = successes / total if total else 0.0
        print(f"{'unknown':>13}  {successes:9d}  {total:5d}  {accuracy:8.3f}")
        print(
            "WARNING: some rows lack compaction metadata; rerun with "
            "scripts/vf_eval_with_kv_eviction.py to distinguish N=0 from unknown."
        )
    print(f"Overall rows: {rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
