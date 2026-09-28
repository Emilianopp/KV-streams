#!/usr/bin/env python3
"""Generate a uniform dataset across TextWorld's four bundled challenges."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import textworld
import textworld.challenges
from datasets import Dataset


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
LOGGER = logging.getLogger(__name__)

FAMILIES = (
    "tw-cooking",
    "tw-simple",
    "tw-coin_collector",
    "tw-treasure_hunter",
)
DIFFICULTIES = ("easy", "medium", "hard")
GAME_SIDECAR_SUFFIXES = (".json", ".ni", ".result.json")
SEED_BLOCK = 1_000_000
EVAL_SEED_OFFSET = 500_000
RETRY_SEED_STRIDE = 20_000_000


class GenerationTimeout(TimeoutError):
    pass


def raise_generation_timeout(_signum: int, _frame: Any) -> None:
    raise GenerationTimeout("TextWorld game generation timed out")


COOKING_SETTINGS = {
    "easy": {
        "recipe": 1,
        "take": 1,
        "go": 6,
        "open": False,
        "cook": False,
        "cut": False,
        "drop": False,
    },
    "medium": {
        "recipe": 2,
        "take": 2,
        "go": 6,
        "open": True,
        "cook": True,
        "cut": True,
        "drop": False,
    },
    "hard": {
        "recipe": 3,
        "take": 3,
        "go": 9,
        "open": True,
        "cook": True,
        "cut": True,
        "drop": True,
    },
}

SIMPLE_SETTINGS = {
    "easy": {"rewards": "dense", "goal": "detailed"},
    "medium": {"rewards": "balanced", "goal": "brief"},
    "hard": {"rewards": "sparse", "goal": "none"},
}

COIN_LEVELS = {
    "easy": tuple(range(5, 21)),
    "medium": tuple(range(105, 121)),
    "hard": tuple(range(205, 221)),
}

TREASURE_LEVELS = {
    "easy": tuple(range(1, 11)),
    "medium": tuple(range(11, 21)),
    "hard": tuple(range(21, 31)),
}


def split_counts(total: int, parts: int) -> list[int]:
    base, remainder = divmod(total, parts)
    return [base + (index < remainder) for index in range(parts)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def game_settings(
    family: str,
    difficulty: str,
    split: str,
    ordinal: int,
    seed: int,
) -> dict[str, Any]:
    if family == "tw-cooking":
        settings = dict(COOKING_SETTINGS[difficulty])
        settings["recipe_seed"] = seed
        settings["split"] = "train" if split == "train" else "valid"
        return settings
    if family == "tw-simple":
        settings = dict(SIMPLE_SETTINGS[difficulty])
        settings["test"] = split == "eval"
        return settings
    if family == "tw-coin_collector":
        levels = COIN_LEVELS[difficulty]
        return {"level": levels[ordinal % len(levels)]}
    if family == "tw-treasure_hunter":
        levels = TREASURE_LEVELS[difficulty]
        return {"level": levels[ordinal % len(levels)]}
    raise ValueError(f"Unsupported TextWorld family: {family}")


def game_artifacts(game_file: Path) -> tuple[Path, ...]:
    return (
        game_file,
        *(game_file.with_suffix(suffix) for suffix in GAME_SIDECAR_SUFFIXES),
    )


def cleanup_game_files(game_file: Path) -> None:
    for path in game_artifacts(game_file):
        path.unlink(missing_ok=True)


def generate_one(task: dict[str, Any]) -> dict[str, Any]:
    family = task["family"]
    difficulty = task["difficulty"]
    split = task["split"]
    ordinal = task["ordinal"]
    seed = task["seed"] + task["attempt"] * RETRY_SEED_STRIDE
    output_dir = Path(task["output_dir"])
    game_dir = output_dir / "games" / family / difficulty
    game_dir.mkdir(parents=True, exist_ok=True)
    game_file = game_dir / f"{split}_{ordinal:05d}.z8"
    result_file = game_file.with_suffix(".result.json")

    if task["attempt"] == 0 and all(
        path.is_file() for path in game_artifacts(game_file)
    ):
        result = json.loads(result_file.read_text())
        if result.get("sha256") == sha256_file(game_file):
            return result

    cleanup_game_files(game_file)
    settings = game_settings(family, difficulty, split, ordinal, seed)
    make_game = textworld.challenges.CHALLENGES[family][1]
    options = textworld.GameOptions()
    options.seeds = seed
    options.path = str(game_file)
    game = make_game(settings, options)

    if not game_file.exists():
        compiled_path = Path(textworld.generator.compile_game(game, options=options))
        if compiled_path != game_file:
            game_file = compiled_path
            result_file = game_file.with_suffix(".result.json")

    request_infos = textworld.EnvInfos(
        score=True,
        max_score=True,
        won=True,
        description=True,
        inventory=True,
    )
    env = textworld.start(str(game_file), request_infos)
    try:
        state = env.reset()
        initial_obs = state.feedback
        max_score = int(state.max_score or 1)
    finally:
        env.close()

    result = {
        "key": task["key"],
        "family": family,
        "difficulty": difficulty,
        "split": split,
        "ordinal": ordinal,
        "seed": seed,
        "attempt": task["attempt"],
        "settings": settings,
        "game_file": str(game_file.relative_to(output_dir)),
        "initial_obs": initial_obs,
        "max_score": max_score,
        "uuid": game.metadata.get("uuid"),
        "sha256": sha256_file(game_file),
    }
    temporary = result_file.with_suffix(".result.json.tmp")
    temporary.write_text(json.dumps(result, sort_keys=True))
    os.replace(temporary, result_file)
    return result


def worker(task: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
    previous_handler = signal.signal(signal.SIGALRM, raise_generation_timeout)
    signal.setitimer(signal.ITIMER_REAL, task["timeout_seconds"])
    try:
        return task["key"], generate_one(task), None
    except Exception as error:  # noqa: BLE001 - retry failures in the parent
        return task["key"], None, repr(error)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def duplicate_keys(results: dict[str, dict[str, Any]]) -> set[str]:
    duplicates: set[str] = set()
    for field in ("uuid", "sha256"):
        first_key: dict[str, str] = {}
        for key in sorted(results):
            value = results[key].get(field)
            if not value:
                duplicates.add(key)
                continue
            if value in first_key:
                duplicates.add(key)
            else:
                first_key[value] = key
    return duplicates


def run_generation(
    tasks: list[dict[str, Any]],
    workers: int,
    retries: int,
    timeout_seconds: int,
) -> dict[str, dict[str, Any]]:
    task_by_key = {task["key"]: task for task in tasks}
    attempts = {key: 0 for key in task_by_key}
    results: dict[str, dict[str, Any]] = {}
    pending = set(task_by_key)

    while pending:
        round_tasks = []
        for key in sorted(pending):
            task = dict(task_by_key[key])
            task["attempt"] = attempts[key]
            task["timeout_seconds"] = timeout_seconds
            round_tasks.append(task)

        failures: dict[str, str] = {}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker, task) for task in round_tasks]
            for completed, future in enumerate(as_completed(futures), start=1):
                key, result, error = future.result()
                if error is None and result is not None:
                    results[key] = result
                else:
                    failures[key] = error or "unknown generation error"
                if completed % 250 == 0 or completed == len(futures):
                    LOGGER.info(
                        "Generation round progress: %d/%d (%d failures)",
                        completed,
                        len(futures),
                        len(failures),
                    )

        pending = set(failures)
        if not pending:
            pending = duplicate_keys(results)
            if pending:
                LOGGER.warning(
                    "Regenerating %d duplicate UUID/content slots",
                    len(pending),
                )
                for key in pending:
                    results.pop(key, None)

        exhausted = [key for key in pending if attempts[key] >= retries]
        if exhausted:
            details = {key: failures.get(key, "duplicate") for key in exhausted}
            raise RuntimeError(f"Generation failed after {retries} retries: {details}")
        for key in pending:
            attempts[key] += 1

    if len(results) != len(tasks):
        raise RuntimeError(f"Expected {len(tasks)} results, generated {len(results)}")
    return results


def build_tasks(
    output_dir: Path,
    total_train: int,
    total_eval: int,
    base_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    if total_train % len(FAMILIES) != 0:
        raise ValueError("--total-train must be divisible by four families")
    train_per_family = total_train // len(FAMILIES)
    if total_eval % len(FAMILIES) != 0:
        raise ValueError("--total-eval must be divisible by four families")

    train_counts = split_counts(train_per_family, len(DIFFICULTIES))
    eval_counts = split_counts(total_eval // len(FAMILIES), len(DIFFICULTIES))
    tasks: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}

    for family_index, family in enumerate(FAMILIES):
        counts[family] = {}
        for difficulty_index, difficulty in enumerate(DIFFICULTIES):
            cell_index = family_index * len(DIFFICULTIES) + difficulty_index
            counts[family][difficulty] = {
                "train": train_counts[difficulty_index],
                "eval": eval_counts[difficulty_index],
            }
            for split, count, split_offset in (
                ("train", train_counts[difficulty_index], 0),
                ("eval", eval_counts[difficulty_index], EVAL_SEED_OFFSET),
            ):
                for ordinal in range(count):
                    key = f"{family}:{difficulty}:{split}:{ordinal:05d}"
                    tasks.append(
                        {
                            "key": key,
                            "family": family,
                            "difficulty": difficulty,
                            "split": split,
                            "ordinal": ordinal,
                            "seed": (
                                base_seed
                                + cell_index * SEED_BLOCK
                                + split_offset
                                + ordinal
                            ),
                            "output_dir": str(output_dir),
                        }
                    )
    return tasks, counts


def interleaved_keys(
    counts: dict[str, dict[str, int]],
    split: str,
) -> list[str]:
    keys = []
    maximum = max(
        counts[family][difficulty][split]
        for family in FAMILIES
        for difficulty in DIFFICULTIES
    )
    for ordinal in range(maximum):
        for family in FAMILIES:
            for difficulty in DIFFICULTIES:
                if ordinal < counts[family][difficulty][split]:
                    keys.append(f"{family}:{difficulty}:{split}:{ordinal:05d}")
    return keys


def save_dataset(
    output_dir: Path,
    results: dict[str, dict[str, Any]],
    counts: dict[str, dict[str, int]],
    total_train: int,
    total_eval: int,
    seed: int,
) -> None:
    train_keys = interleaved_keys(counts, "train")
    eval_keys = interleaved_keys(counts, "eval")
    ordered_keys = train_keys + eval_keys
    answer_by_key = {key: index for index, key in enumerate(ordered_keys)}

    def row(key: str) -> dict[str, str]:
        result = results[key]
        return {
            "question": result["initial_obs"],
            "answer": str(answer_by_key[key]),
            "task": f"{result['family']}:{result['difficulty']}",
        }

    Dataset.from_list([row(key) for key in train_keys]).save_to_disk(
        str(output_dir / "dataset")
    )
    if eval_keys:
        Dataset.from_list([row(key) for key in eval_keys]).save_to_disk(
            str(output_dir / "eval_dataset")
        )

    entries = [results[key] for key in ordered_keys]
    metadata = {
        "game_files": [entry["game_file"] for entry in entries],
        "max_scores": [entry["max_score"] for entry in entries],
        "num_train": len(train_keys),
        "num_eval": len(eval_keys),
        "num_total": len(entries),
        "seed": seed,
        "families": counts,
        "family_order": list(FAMILIES),
        "difficulty_order": list(DIFFICULTIES),
        "uniform_train_grid": {
            "families": len(FAMILIES),
            "difficulties_per_family": len(DIFFICULTIES),
            "games_per_cell_min": min(
                counts[family][difficulty]["train"]
                for family in FAMILIES
                for difficulty in DIFFICULTIES
            ),
            "games_per_cell_max": max(
                counts[family][difficulty]["train"]
                for family in FAMILIES
                for difficulty in DIFFICULTIES
            ),
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    with (output_dir / "manifest.jsonl").open("w") as manifest:
        for index, entry in enumerate(entries):
            record = dict(entry)
            record["answer"] = str(index)
            manifest.write(json.dumps(record, sort_keys=True) + "\n")

    if len(train_keys) != total_train or len(eval_keys) != total_eval:
        raise RuntimeError(
            f"Wrong split sizes: train={len(train_keys)} eval={len(eval_keys)}"
        )
    if len({entry["game_file"] for entry in entries}) != len(entries):
        raise RuntimeError("Duplicate game paths in final dataset")
    if len({entry["uuid"] for entry in entries}) != len(entries):
        raise RuntimeError("Duplicate game UUIDs in final dataset")
    if len({entry["sha256"] for entry in entries}) != len(entries):
        raise RuntimeError("Duplicate game content hashes in final dataset")
    for entry in entries:
        game_file = output_dir / entry["game_file"]
        for path in game_artifacts(game_file):
            if not path.is_file():
                raise FileNotFoundError(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--total-train", type=int, default=11744)
    parser.add_argument("--total-eval", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument("--task-timeout", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "dataset").exists() or (
        args.output / "eval_dataset"
    ).exists():
        raise FileExistsError(
            "Final dataset already exists; choose a new output directory"
        )

    tasks, counts = build_tasks(
        args.output,
        args.total_train,
        args.total_eval,
        args.seed,
    )
    LOGGER.info(
        "Generating %d train + %d eval games in a uniform 4x3 grid",
        args.total_train,
        args.total_eval,
    )
    LOGGER.info("Per-cell counts: %s", counts)
    results = run_generation(
        tasks,
        args.workers,
        args.retries,
        args.task_timeout,
    )
    save_dataset(
        args.output,
        results,
        counts,
        args.total_train,
        args.total_eval,
        args.seed,
    )
    LOGGER.info("Dataset complete at %s", args.output)


if __name__ == "__main__":
    main()
