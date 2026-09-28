#!/usr/bin/env python3
"""Parallel TextWorld game generation for a balanced, all-difficulty dataset.

Reuses `generate_game()` and `COOKING_DIFFICULTIES` from generate_dataset.py but
distributes work across a process pool so the full difficulty suite (~12k games)
compiles in minutes instead of hours on a many-core node.

Determinism matches the original mix mode: difficulty at index `off` uses
seed_base = seed + off * 10000, game `i` uses seed = seed_base + i (also the
recipe_seed). Seed blocks are 10000 wide per tier so game_XXXXX.z8 filenames
never collide across tiers.

Output layout is byte-for-byte compatible with generate_dataset.py:
    <output>/games/game_XXXXX.z8   pre-compiled games
    <output>/dataset/             HF train dataset
    <output>/eval_dataset/        HF eval dataset
    <output>/metadata.json        game_files (RELATIVE) + max_scores + difficulties

Usage:
    python generate_dataset_parallel.py \
        --output ./data/textworld_cooking_12k \
        --total-train 12000 --eval-per-difficulty 20 \
        --workers 96 --seed 42
"""

import argparse
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from datasets import Dataset

from generate_dataset import COOKING_DIFFICULTIES, generate_game

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

SEED_BLOCK = 10000  # per-difficulty seed stride (matches generate_dataset.py mix mode)


def _worker(args):
    """Generate one game; return locator + metadata (or error)."""
    off, i, name, settings, seed, games_dir = args
    try:
        meta = generate_game(settings, seed, Path(games_dir))
        return (off, i, name, seed, meta, None)
    except Exception as e:  # noqa: BLE001 - report and continue
        return (off, i, name, seed, None, repr(e))


def split_counts(total, n):
    """Split `total` into n as-even-as-possible integers summing to total."""
    base, rem = divmod(total, n)
    return [base + (1 if k < rem else 0) for k in range(n)]


def main():
    p = argparse.ArgumentParser(description="Parallel balanced TextWorld generation")
    p.add_argument("--output", required=True)
    p.add_argument("--total-train", type=int, default=12000,
                   help="Total train games, split evenly across all difficulties")
    p.add_argument("--eval-per-difficulty", type=int, default=20)
    p.add_argument("--workers", type=int, default=96)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--difficulties", nargs="+", default=None,
                   help="Subset of difficulty names (default: all presets)")
    p.add_argument("--retries", type=int, default=3,
                   help="Retry a failed game with an offset seed this many times")
    args = p.parse_args()

    names = args.difficulties or list(COOKING_DIFFICULTIES.keys())
    for name in names:
        assert name in COOKING_DIFFICULTIES, f"Unknown difficulty: {name}"

    output_dir = Path(args.output)
    games_dir = output_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    train_counts = split_counts(args.total_train, len(names))
    n_eval = args.eval_per_difficulty

    # Build the task list deterministically: (off, i, name, settings, seed).
    tasks = []
    per_tier = []  # (off, name, train_count, total_count, seed_base)
    for off, (name, train_count) in enumerate(zip(range(len(names)), train_counts)):
        name = names[off]
        train_count = train_counts[off]
        total = train_count + n_eval
        assert total <= SEED_BLOCK, f"{name}: {total} games exceeds seed block {SEED_BLOCK}"
        seed_base = args.seed + off * SEED_BLOCK
        per_tier.append((off, name, train_count, total, seed_base))
        settings = COOKING_DIFFICULTIES[name]["settings"]
        for i in range(total):
            tasks.append((off, i, name, settings, seed_base + i, str(games_dir)))

    logger.info(
        "Generating %d games across %d difficulties (%d train + %d eval) with %d workers",
        len(tasks), len(names), args.total_train, n_eval * len(names), args.workers,
    )
    for off, name, tc, total, sb in per_tier:
        logger.info("  %-12s %5d train + %d eval  seeds[%d..%d]  — %s",
                    name, tc, n_eval, sb, sb + total - 1,
                    COOKING_DIFFICULTIES[name]["desc"])

    # results[(off, i)] = meta
    results = {}
    failures = []
    done = 0
    total_tasks = len(tasks)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, t): (t[0], t[1]) for t in tasks}
        for fut in as_completed(futs):
            off, i, name, seed, meta, err = fut.result()
            done += 1
            if err is not None:
                failures.append((off, i, name, seed, err))
            else:
                results[(off, i)] = meta
            if done % 500 == 0 or done == total_tasks:
                logger.info("  progress: %d/%d done (%d failures so far)",
                            done, total_tasks, len(failures))

    # Retry failures serially with offset seeds (rare; keeps counts intact).
    for attempt in range(1, args.retries + 1):
        if not failures:
            break
        logger.info("Retry pass %d: %d failed games", attempt, len(failures))
        still = []
        for off, i, name, seed, err in failures:
            settings = COOKING_DIFFICULTIES[name]["settings"]
            new_seed = seed + attempt * SEED_BLOCK * len(names)
            try:
                meta = generate_game(settings, new_seed, games_dir)
                results[(off, i)] = meta
            except Exception as e:  # noqa: BLE001
                still.append((off, i, name, seed, repr(e)))
        failures = still
    if failures:
        logger.warning("%d games still failing after retries: %s",
                       len(failures), failures[:5])

    # Assemble in deterministic (off, i) order -> global game_files index == answer.
    all_game_files, all_max_scores = [], []
    train_rows, eval_rows = [], []
    difficulty_info = {}
    for off, name, train_count, total, seed_base in per_tier:
        n_ok_train = 0
        n_ok_eval = 0
        for i in range(total):
            meta = results.get((off, i))
            if meta is None:
                continue  # dropped game; counts shrink slightly
            idx = len(all_game_files)
            all_game_files.append(meta["game_file"])
            all_max_scores.append(meta["max_score"])
            row = {"question": meta["initial_obs"], "answer": str(idx), "task": name}
            if i < train_count:
                train_rows.append(row)
                n_ok_train += 1
            else:
                eval_rows.append(row)
                n_ok_eval += 1
        difficulty_info[name] = {
            "desc": COOKING_DIFFICULTIES[name]["desc"],
            "num_train": n_ok_train,
            "num_eval": n_ok_eval,
            "settings": COOKING_DIFFICULTIES[name]["settings"],
        }

    logger.info("Generated %d games total (%d train + %d eval)",
                len(all_game_files), len(train_rows), len(eval_rows))
    if not train_rows:
        logger.error("No games generated!")
        return

    Dataset.from_list(train_rows).save_to_disk(str(output_dir / "dataset"))
    if eval_rows:
        Dataset.from_list(eval_rows).save_to_disk(str(output_dir / "eval_dataset"))

    rel_game_files = [f"games/{Path(gf).name}" for gf in all_game_files]
    metadata = {
        "game_files": rel_game_files,
        "max_scores": all_max_scores,
        "num_train": len(train_rows),
        "num_eval": len(eval_rows),
        "seed": args.seed,
        "difficulties": difficulty_info,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Dataset saved to %s", output_dir)
    for name, info in difficulty_info.items():
        logger.info("  %-12s %5d train, %3d eval — %s",
                    name, info["num_train"], info["num_eval"], info["desc"])


if __name__ == "__main__":
    main()
