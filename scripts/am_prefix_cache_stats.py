#!/usr/bin/env python3
"""Summarize AM compressed-prefix cache behavior from vLLM logs."""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


def _read_text(path: Path) -> str:
    if not path.exists():
        raise SystemExit(f"log path does not exist: {path}")
    return path.read_text(errors="replace")


def summarize(log_path: Path) -> None:
    text = _read_text(log_path)
    lines = text.splitlines()

    prepared_re = re.compile(
        r"prepared compressed admission candidate for request (\S+) "
        r".* step=(\d+)/(\d+) key=([0-9a-f]+)"
    )
    hit_re = re.compile(
        r"compressed admission hit for request (\S+) hit=(\d+) "
        r"required=(\d+) min_required=(\d+) full_reusable=(\d+) "
        r"allow_partial=(\w+) step=(\d+)/(\d+) key=([0-9a-f]+)"
    )
    miss_re = re.compile(
        r"compressed admission miss for request (\S+) hit=(\d+) "
        r"required=(\d+) min_required=(\d+) full_reusable=(\d+) "
        r"allow_partial=(\w+) step=(\d+)/(\d+) key=([0-9a-f]+); "
        r"(retrying shallower|restoring full prompt)"
    )
    read_hit_re = re.compile(
        r"\[PrefixCache\] read hit request (\S+) cached_tokens=(\d+) "
        r"prompt_tokens=(\d+) mode=([^ ]+) .* am_keyed=(\w+)"
    )
    finalization_re = re.compile(
        r"finalizing stopped request (\S+) with (\d+) hidden tokens"
    )
    engine_re = re.compile(r"\((EngineCore_DP\d+) pid=")
    assign_re = re.compile(
        r"assigned AM prefix-cache key ([0-9a-f]+) for request (\S+)"
    )

    prepared = list(prepared_re.finditer(text))
    hits = list(hit_re.finditer(text))
    misses = list(miss_re.finditer(text))
    read_hits = list(read_hit_re.finditer(text))
    finalizations = list(finalization_re.finditer(text))

    attempts_by_request: dict[str, list[tuple[str, int, int, str]]] = defaultdict(list)
    prepared_by_request: dict[str, set[tuple[int, int, str]]] = defaultdict(set)
    hit_by_depth: Counter[tuple[int, int]] = Counter()
    prepared_by_depth: Counter[tuple[int, int]] = Counter()
    miss_by_depth: Counter[tuple[int, int, str]] = Counter()
    prepared_by_key: Counter[str] = Counter()
    assigned_by_engine_key: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    assigned_by_key: defaultdict[str, list[tuple[int, str]]] = defaultdict(list)
    parsed_attempts: list[tuple[int, str, str, str, str, int]] = []

    for line_no, line in enumerate(lines, start=1):
        engine_match = engine_re.search(line)
        engine = engine_match.group(1) if engine_match else "unknown"
        assign_match = assign_re.search(line)
        if assign_match:
            key, _req = assign_match.groups()
            assigned_by_engine_key[(engine, key)].append(line_no)
            assigned_by_key[key].append((line_no, engine))

        hit_match = hit_re.search(line)
        if hit_match:
            (
                req,
                hit_tokens,
                _required,
                _min_required,
                _full_reusable,
                _allow_partial,
                step,
                depth,
                key,
            ) = hit_match.groups()
            attempts_by_request[req].append(
                ("hit", int(step), int(depth), key)
            )
            parsed_attempts.append(
                (line_no, engine, "hit", req, key, int(hit_tokens))
            )
            continue

        miss_match = miss_re.search(line)
        if miss_match:
            (
                req,
                hit_tokens,
                _required,
                _min_required,
                _full_reusable,
                _allow_partial,
                step,
                depth,
                key,
                action,
            ) = miss_match.groups()
            kind = "retry" if action.startswith("retrying") else "restore"
            attempts_by_request[req].append(
                (kind, int(step), int(depth), key)
            )
            parsed_attempts.append(
                (line_no, engine, "miss", req, key, int(hit_tokens))
            )

    for match in prepared:
        req, step, depth, key = match.groups()
        step_i, depth_i = int(step), int(depth)
        prepared_by_request[req].add((step_i, depth_i, key))
        prepared_by_depth[(step_i, depth_i)] += 1
        prepared_by_key[key] += 1

    for match in hits:
        req, _hit_tokens, _required, _min_required, _full_reusable, _allow_partial, step, depth, key = (
            match.groups()
        )
        step_i, depth_i = int(step), int(depth)
        hit_by_depth[(step_i, depth_i)] += 1

    for match in misses:
        (
            req,
            _hit_tokens,
            _required,
            _min_required,
            _full_reusable,
            _allow_partial,
            step,
            depth,
            key,
            action,
        ) = match.groups()
        step_i, depth_i = int(step), int(depth)
        kind = "retry" if action.startswith("retrying") else "restore"
        miss_by_depth[(step_i, depth_i, kind)] += 1

    final_outcomes: Counter[tuple[str, int, int]] = Counter()
    for events in attempts_by_request.values():
        if not events:
            continue
        outcome, step, depth, _key = events[-1]
        final_outcomes[(outcome, step, depth)] += 1

    final_hit_requests = sum(
        count for (outcome, _step, _depth), count in final_outcomes.items()
        if outcome == "hit"
    )
    final_restore_requests = sum(
        count for (outcome, _step, _depth), count in final_outcomes.items()
        if outcome == "restore"
    )
    final_retry_requests = sum(
        count for (outcome, _step, _depth), count in final_outcomes.items()
        if outcome == "retry"
    )
    final_completed_requests = final_hit_requests + final_restore_requests
    final_completed_hit_rate = (
        final_hit_requests / final_completed_requests
        if final_completed_requests
        else 0.0
    )

    prepared_requests = len(prepared_by_request)
    request_hit_rate = final_hit_requests / prepared_requests if prepared_requests else 0.0
    attempt_hit_rate = len(hits) / len(prepared) if prepared else 0.0
    duplicate_key_ceiling_hits = max(0, len(prepared) - len(prepared_by_key))
    duplicate_key_ceiling_rate = (
        duplicate_key_ceiling_hits / len(prepared) if prepared else 0.0
    )

    miss_causes: Counter[str] = Counter()
    for line_no, engine, kind, _req, key, hit_tokens in parsed_attempts:
        if kind != "miss":
            continue
        earlier_same = [
            n for n in assigned_by_engine_key.get((engine, key), []) if n < line_no
        ]
        earlier_other = [
            (n, e)
            for n, e in assigned_by_key.get(key, [])
            if n < line_no and e != engine
        ]
        later_same = [
            n for n in assigned_by_engine_key.get((engine, key), []) if n > line_no
        ]
        if hit_tokens > 0:
            miss_causes["partial_hit_below_threshold"] += 1
        elif earlier_same:
            miss_causes["unexpected_miss_after_same_engine_assignment"] += 1
        elif earlier_other:
            miss_causes["cross_engine_miss"] += 1
        elif later_same:
            miss_causes["cold_or_concurrent_first_use"] += 1
        elif key in assigned_by_key:
            miss_causes["assigned_later_other_engine"] += 1
        else:
            miss_causes["never_assigned"] += 1

    read_cached_tokens = sum(int(m.group(2)) for m in read_hits)
    read_prompt_tokens = sum(int(m.group(3)) for m in read_hits)
    read_token_reuse = read_cached_tokens / read_prompt_tokens if read_prompt_tokens else 0.0

    print(f"log: {log_path}")
    print()
    print("AM compressed-prefix cache")
    print(f"  prepared candidates:       {len(prepared)}")
    print(f"  candidate requests:        {prepared_requests}")
    print(f"  admission hits:            {len(hits)}")
    print(f"  admission misses:          {len(misses)}")
    print(f"  attempt hit rate:          {attempt_hit_rate:.2%}")
    print(f"  unique AM keys prepared:   {len(prepared_by_key)}")
    print(
        "  duplicate-key hit ceiling: "
        f"{duplicate_key_ceiling_hits}/{len(prepared)} "
        f"({duplicate_key_ceiling_rate:.2%})"
    )
    print(f"  final hit requests:        {final_hit_requests}")
    print(f"  final restore requests:    {final_restore_requests}")
    print(f"  final retry-pending reqs:  {final_retry_requests}")
    print(f"  completed final hit rate:  {final_completed_hit_rate:.2%}")
    print(f"  request hit/prepared rate: {request_hit_rate:.2%}")
    print()
    print("Ordinary/read-hit accounting")
    print(f"  read-hit events:           {len(read_hits)}")
    print(f"  cached tokens on hits:     {read_cached_tokens}")
    print(f"  prompt tokens on hits:     {read_prompt_tokens}")
    print(f"  token reuse on hit events: {read_token_reuse:.2%}")
    print()
    print("AM/runtime counters")
    print(f"  AM compactions finished:   {text.count('[AM] finished compaction request')}")
    print(f"  turn-chain compactions:    {text.count('finished turn-chain compaction')}")
    print(f"  COW privatizations:        {text.count('[PrefixCache][AM][COW]')}")
    print(f"  hidden finalizations:      {len(finalizations)}")
    print(f"  errors:                    {text.count('ERROR')}")
    print(f"  tracebacks:                {text.count('Traceback')}")
    print(f"  runtime errors:            {text.count('RuntimeError')}")
    print(f"  NaN/nan mentions:          {text.count('NaN') + text.count('nan')}")
    print()

    if prepared_by_depth:
        print("Hit rate by requested replay depth")
        for key, prepared_count in prepared_by_depth.most_common():
            hit_count = hit_by_depth[key]
            step, depth = key
            print(
                f"  step {step}/{depth}: prepared={prepared_count} "
                f"hits={hit_count} hit_rate={hit_count / prepared_count:.2%}"
            )
        print()

    if final_outcomes:
        print("Final outcomes")
        for (outcome, step, depth), count in final_outcomes.most_common():
            print(f"  {outcome:7s} step {step}/{depth}: {count}")
        print()

    if miss_by_depth:
        print("Miss breakdown")
        for (step, depth, kind), count in miss_by_depth.most_common():
            print(f"  {kind:7s} step {step}/{depth}: {count}")
        print()

    if miss_causes:
        print("Miss cause classification")
        for cause, count in miss_causes.most_common():
            print(f"  {cause:45s} {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    args = parser.parse_args()
    summarize(args.log_path)


if __name__ == "__main__":
    main()
