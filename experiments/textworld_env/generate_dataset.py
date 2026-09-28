#!/usr/bin/env python3
"""Pre-generate TextWorld game files and HF dataset for RL training.

Generates .z8 game files using TextWorld's challenges API, captures initial
observations, and creates an HF Dataset + metadata.json for TextWorldEnv.

Supports single-difficulty and mixed-difficulty (curriculum) datasets.

Ported from mkv-rl/experiments/textworld_rl/generate_games.py with one
change: `metadata.json` stores **relative** game file paths
(`games/game_XXXXX.z8`) so the dataset directory is fully relocatable.
TextWorldEnv.load_environment resolves them via `_resolve_game_files()`.

Usage:
    # Single difficulty:
    python generate_dataset.py \
        --output ./data/textworld_cooking \
        --difficulty current \
        --num-train 150 --num-eval 50 --seed 42

    # Mixed curriculum (production hard mix):
    python generate_dataset.py \
        --output ./data/textworld_cooking_mix \
        --mix easy-nav:1250 current:500 hard:1500 hard-12room:1000 hard-drop:750 \
        --eval-per-difficulty 20 --seed 42
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path

import textworld
import textworld.challenges
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Cooking difficulty presets. Each maps to TextWorld tw-cooking challenge settings.
COOKING_DIFFICULTIES = {
    "trivial": {
        "settings": {"recipe": 1, "take": 1, "go": 1, "open": False, "cook": False,
                      "cut": False, "drop": False, "recipe_seed": 0, "split": "train"},
        "desc": "1 ingredient, 1 room, no skills",
    },
    "easy": {
        "settings": {"recipe": 1, "take": 1, "go": 1, "open": True, "cook": True,
                      "cut": False, "drop": False, "recipe_seed": 0, "split": "train"},
        "desc": "1 ingredient, 1 room, open+cook",
    },
    "easy-nav": {
        "settings": {"recipe": 1, "take": 1, "go": 6, "open": False, "cook": False,
                      "cut": False, "drop": False, "recipe_seed": 0, "split": "train"},
        "desc": "1 ingredient, 6 rooms, no skills",
    },
    "medium-easy": {
        "settings": {"recipe": 2, "take": 2, "go": 6, "open": True, "cook": True,
                      "cut": False, "drop": False, "recipe_seed": 0, "split": "train"},
        "desc": "2 ingredients, 6 rooms, open+cook",
    },
    "medium": {
        "settings": {"recipe": 2, "take": 2, "go": 6, "open": True, "cook": True,
                      "cut": True, "drop": False, "recipe_seed": 0, "split": "train"},
        "desc": "2 ingredients, 6 rooms, open+cook+cut",
    },
    "current": {
        "settings": {"recipe": 3, "take": 3, "go": 6, "open": True, "cook": True,
                      "cut": True, "drop": False, "recipe_seed": 0, "split": "train"},
        "desc": "3 ingredients, 6 rooms, open+cook+cut",
    },
    "hard": {
        "settings": {"recipe": 3, "take": 3, "go": 9, "open": True, "cook": True,
                      "cut": True, "drop": False, "recipe_seed": 0, "split": "train"},
        "desc": "3 ingredients, 9 rooms, open+cook+cut",
    },
    "hard-4ingr": {
        "settings": {"recipe": 4, "take": 4, "go": 6, "open": True, "cook": True,
                      "cut": True, "drop": False, "recipe_seed": 0, "split": "train"},
        "desc": "4 ingredients, 6 rooms, open+cook+cut",
    },
    "hard-12room": {
        "settings": {"recipe": 3, "take": 3, "go": 12, "open": True, "cook": True,
                      "cut": True, "drop": False, "recipe_seed": 0, "split": "train"},
        "desc": "3 ingredients, 12 rooms, open+cook+cut",
    },
    "hard-drop": {
        "settings": {"recipe": 3, "take": 3, "go": 9, "open": True, "cook": True,
                      "cut": True, "drop": True, "recipe_seed": 0, "split": "train"},
        "desc": "3 ingredients, 9 rooms, all skills + inventory limit",
    },
    "extreme": {
        "settings": {"recipe": 5, "take": 5, "go": 12, "open": True, "cook": True,
                      "cut": True, "drop": False, "recipe_seed": 0, "split": "train"},
        "desc": "5 ingredients, 12 rooms, open+cook+cut",
    },
}


def generate_game(settings: dict, seed: int, output_dir: Path) -> dict:
    """Generate a single TextWorld cooking game and return metadata."""
    game_file = str(output_dir / f"game_{seed:05d}.z8")

    make_fn = textworld.challenges.CHALLENGES["tw-cooking"][1]

    options = textworld.GameOptions()
    options.seeds = seed
    options.path = game_file

    game_settings = dict(settings)
    if "recipe_seed" in game_settings:
        game_settings["recipe_seed"] = seed

    game = make_fn(game_settings, options)

    if not os.path.exists(game_file):
        game_file = textworld.generator.compile_game(game, options=options)

    # Capture initial observation
    request_infos = textworld.EnvInfos(
        score=True, max_score=True, won=True,
        description=True, inventory=True,
    )
    env = textworld.start(game_file, request_infos)
    game_state = env.reset()
    max_score = game_state.max_score or 1
    env.close()

    return {
        "game_file": game_file,
        "initial_obs": game_state.feedback,
        "max_score": max_score,
        "seed": seed,
    }


def _generate_game_task(args):
    i, difficulty_name, settings, seed, games_dir = args
    try:
        meta = generate_game(settings, seed, games_dir)
    except Exception as e:
        return i, seed, None, f"{type(e).__name__}: {e}"
    return i, seed, meta, None


def generate_difficulty(difficulty_name, settings, num_games, seed_base, games_dir,
                        checkpoint_path=None, existing_count=0, workers=1):
    """Generate games for one difficulty level with checkpoint support."""
    game_files = []
    max_scores = []
    rows = []

    def append_meta(meta):
        local_idx = len(game_files)
        game_files.append(meta["game_file"])
        max_scores.append(meta["max_score"])
        rows.append({
            "question": meta["initial_obs"],
            "answer": str(local_idx),
            "task": difficulty_name,
        })

    if workers > 1:
        tasks = [
            (
                i,
                difficulty_name,
                settings,
                seed_base + i,
                games_dir,
            )
            for i in range(num_games)
        ]
        results = {}
        completed = 0
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_generate_game_task, task) for task in tasks]
            for future in as_completed(futures):
                i, seed, meta, error = future.result()
                completed += 1
                if error:
                    logger.error(f"  Failed {difficulty_name} game {i} (seed={seed}): {error}")
                else:
                    results[i] = meta
                if completed % 50 == 0 or completed == num_games:
                    logger.info(
                        f"    {difficulty_name}: {completed}/{num_games} generated "
                        f"({len(results)} ok)"
                    )

        for i in range(num_games):
            meta = results.get(i)
            if meta is not None:
                append_meta(meta)

        return game_files, max_scores, rows

    for i in range(num_games):
        seed = seed_base + i
        # Skip already-generated games (resume from checkpoint)
        game_file_path = games_dir / f"game_{seed:05d}.z8"
        if game_file_path.exists() and i < existing_count:
            continue

        try:
            meta = generate_game(settings, seed, games_dir)
        except Exception as e:
            logger.error(f"  Failed {difficulty_name} game {i} (seed={seed}): {e}")
            continue

        append_meta(meta)

        # Log progress every 50 games
        if (i + 1) % 50 == 0:
            logger.info(f"    {difficulty_name}: {i+1}/{num_games} generated")

    return game_files, max_scores, rows


def main():
    parser = argparse.ArgumentParser(description="Generate TextWorld games for RL training")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)

    # Single difficulty mode
    parser.add_argument("--difficulty", default=None,
                        choices=list(COOKING_DIFFICULTIES.keys()),
                        help="Single difficulty preset")
    parser.add_argument("--num-train", type=int, default=150)
    parser.add_argument("--num-eval", type=int, default=50)

    # Mixed mode
    parser.add_argument("--mix", nargs="+", default=None,
                        help="Mixed difficulties: 'name:count' pairs (e.g. 'easy-nav:30 current:40 hard:50')")
    parser.add_argument("--eval-per-difficulty", type=int, default=10,
                        help="Eval games per difficulty in mix mode (default: 10)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel game-generation workers per difficulty")

    args = parser.parse_args()

    output_dir = Path(args.output)
    games_dir = output_dir / "games"
    games_dir.mkdir(parents=True, exist_ok=True)

    all_game_files = []
    all_max_scores = []
    all_train_rows = []
    all_eval_rows = []
    difficulty_info = {}

    # Check for existing checkpoint to resume from
    checkpoint_file = output_dir / "checkpoint.json"
    checkpoint = None
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            checkpoint = json.load(f)
        logger.info(f"Resuming from checkpoint: {len(checkpoint['game_files'])} games already generated")
        all_game_files = checkpoint["game_files"]
        all_max_scores = checkpoint["max_scores"]
        all_train_rows = checkpoint.get("train_rows", [])
        all_eval_rows = checkpoint.get("eval_rows", [])
        difficulty_info = checkpoint.get("difficulties", {})
        completed_diffs = set(checkpoint.get("completed_difficulties", []))
    else:
        completed_diffs = set()

    if args.mix:
        # Mixed mode: parse "name:count" pairs
        logger.info("Generating mixed-difficulty dataset")
        seed_offset = 0
        for entry in args.mix:
            name, count_str = entry.split(":")
            count = int(count_str)
            assert name in COOKING_DIFFICULTIES, f"Unknown difficulty: {name}"

            config = COOKING_DIFFICULTIES[name]
            num_eval = args.eval_per_difficulty
            num_train = count
            total = num_train + num_eval

            # Skip completed difficulties on resume
            if name in completed_diffs:
                logger.info(f"  {name}: SKIPPED (already in checkpoint)")
                seed_offset += 1
                continue

            logger.info(f"  {name}: {num_train} train + {num_eval} eval — {config['desc']}")

            gf, ms, rows = generate_difficulty(
                name, config["settings"], total,
                args.seed + seed_offset * 10000, games_dir,
                workers=max(1, args.workers),
            )

            # Reindex answers to global game_files offset
            base_idx = len(all_game_files)
            for r in rows:
                r["answer"] = str(base_idx + int(r["answer"]))

            all_game_files.extend(gf)
            all_max_scores.extend(ms)
            all_train_rows.extend(rows[:num_train])
            all_eval_rows.extend(rows[num_train:num_train + num_eval])

            difficulty_info[name] = {
                "desc": config["desc"],
                "num_train": min(num_train, len(rows)),
                "num_eval": min(num_eval, max(0, len(rows) - num_train)),
                "settings": config["settings"],
            }
            completed_diffs.add(name)
            seed_offset += 1

            # Save checkpoint after each difficulty
            ckpt = {
                "game_files": all_game_files,
                "max_scores": all_max_scores,
                "train_rows": all_train_rows,
                "eval_rows": all_eval_rows,
                "difficulties": difficulty_info,
                "completed_difficulties": list(completed_diffs),
            }
            with open(checkpoint_file, "w") as f:
                json.dump(ckpt, f)
            logger.info(f"  Checkpoint saved: {len(all_game_files)} games, {len(completed_diffs)} difficulties done")

    else:
        # Single difficulty mode
        diff_name = args.difficulty or "current"
        config = COOKING_DIFFICULTIES[diff_name]
        total = args.num_train + args.num_eval

        logger.info(f"Generating {total} {diff_name} games ({args.num_train} train + {args.num_eval} eval)")
        logger.info(f"  Settings: {config['desc']}")

        gf, ms, rows = generate_difficulty(
            diff_name, config["settings"], total, args.seed, games_dir,
            workers=max(1, args.workers),
        )

        all_game_files = gf
        all_max_scores = ms
        all_train_rows = rows[:args.num_train]
        all_eval_rows = rows[args.num_train:]

        difficulty_info[diff_name] = {
            "desc": config["desc"],
            "num_train": len(all_train_rows),
            "num_eval": len(all_eval_rows),
            "settings": config["settings"],
        }

    logger.info(f"Generated {len(all_game_files)} games total "
                f"({len(all_train_rows)} train + {len(all_eval_rows)} eval)")

    if not all_train_rows:
        logger.error("No games generated!")
        return

    # Save datasets
    Dataset.from_list(all_train_rows).save_to_disk(str(output_dir / "dataset"))
    if all_eval_rows:
        Dataset.from_list(all_eval_rows).save_to_disk(str(output_dir / "eval_dataset"))

    # Save metadata with RELATIVE game file paths.
    # TextWorldEnv._resolve_game_files joins them against dataset_path at load time.
    rel_game_files = [
        f"games/{Path(gf).name}" for gf in all_game_files
    ]
    metadata = {
        "game_files": rel_game_files,
        "max_scores": all_max_scores,
        "num_train": len(all_train_rows),
        "num_eval": len(all_eval_rows),
        "seed": args.seed,
        "difficulties": difficulty_info,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Remove checkpoint on success
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        logger.info("Checkpoint removed (generation complete)")

    logger.info(f"Dataset saved to {output_dir}")
    for name, info in difficulty_info.items():
        logger.info(f"  {name}: {info['num_train']} train, {info['num_eval']} eval — {info['desc']}")


if __name__ == "__main__":
    main()
