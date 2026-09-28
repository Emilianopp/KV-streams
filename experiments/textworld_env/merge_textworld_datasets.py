#!/usr/bin/env python3
"""Merge a base TextWorld dataset with additional training games."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

from datasets import Dataset, load_from_disk


FAMILIES = (
    "tw-cooking",
    "tw-simple",
    "tw-coin_collector",
    "tw-treasure_hunter",
)
DIFFICULTIES = ("easy", "medium", "hard")
GAME_SIDECAR_SUFFIXES = (".json", ".ni", ".result.json")


def load_manifest(root: Path) -> list[dict[str, Any]]:
    entries = [
        json.loads(line)
        for line in (root / "manifest.jsonl").read_text().splitlines()
        if line
    ]
    entries.sort(key=lambda entry: int(entry["answer"]))
    expected = [str(index) for index in range(len(entries))]
    actual = [str(entry["answer"]) for entry in entries]
    if actual != expected:
        raise ValueError(f"Manifest answers are not contiguous under {root}")
    return entries


def destination_game_path(raw_path: str, namespace: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute():
        relative = Path(relative.name)
    elif relative.parts and relative.parts[0] == "games":
        relative = Path(*relative.parts[1:])
    return Path("games") / namespace / relative


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_or_copy_game(source: Path, destination: Path) -> None:
    link_or_copy(source, destination)
    for suffix in GAME_SIDECAR_SUFFIXES:
        source_sidecar = source.with_name(f"{source.stem}{suffix}")
        if not source_sidecar.is_file():
            raise FileNotFoundError(source_sidecar)
        destination_sidecar = destination.with_name(f"{destination.stem}{suffix}")
        link_or_copy(source_sidecar, destination_sidecar)


def remap_rows(rows: list[dict[str, Any]], start: int) -> list[dict[str, Any]]:
    remapped = []
    for offset, source in enumerate(rows):
        row = dict(source)
        row["answer"] = str(start + offset)
        remapped.append(row)
    return remapped


def merge_datasets(base: Path, additional: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    temporary = output.with_name(f"{output.name}.building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    base_metadata = json.loads((base / "metadata.json").read_text())
    additional_metadata = json.loads((additional / "metadata.json").read_text())
    base_manifest = load_manifest(base)
    additional_manifest = load_manifest(additional)

    base_train_count = int(base_metadata["num_train"])
    base_eval_count = int(base_metadata["num_eval"])
    additional_train_count = int(additional_metadata["num_train"])
    additional_eval_count = int(additional_metadata["num_eval"])
    if additional_eval_count != 0:
        raise ValueError("The additional dataset must contain training games only")
    if len(base_manifest) != base_train_count + base_eval_count:
        raise ValueError("Base metadata and manifest counts disagree")
    if len(additional_manifest) != additional_train_count:
        raise ValueError("Additional metadata and manifest counts disagree")

    base_train_rows = [dict(row) for row in load_from_disk(str(base / "dataset"))]
    base_eval_rows = [
        dict(row) for row in load_from_disk(str(base / "eval_dataset"))
    ]
    additional_train_rows = [
        dict(row) for row in load_from_disk(str(additional / "dataset"))
    ]
    if len(base_train_rows) != base_train_count:
        raise ValueError("Base training dataset count disagrees with metadata")
    if len(base_eval_rows) != base_eval_count:
        raise ValueError("Base evaluation dataset count disagrees with metadata")
    if len(additional_train_rows) != additional_train_count:
        raise ValueError("Additional training dataset count disagrees with metadata")

    source_entries = [
        *((base, "base", entry) for entry in base_manifest[:base_train_count]),
        *((additional, "additional", entry) for entry in additional_manifest),
        *((base, "base", entry) for entry in base_manifest[base_train_count:]),
    ]

    merged_manifest: list[dict[str, Any]] = []
    for answer, (source_root, namespace, source_entry) in enumerate(source_entries):
        source_game = source_root / source_entry["game_file"]
        if not source_game.is_file():
            raise FileNotFoundError(source_game)
        destination_relative = destination_game_path(
            source_entry["game_file"], namespace
        )
        link_or_copy_game(source_game, temporary / destination_relative)
        entry = dict(source_entry)
        entry["answer"] = str(answer)
        entry["game_file"] = destination_relative.as_posix()
        merged_manifest.append(entry)

    uuids = [entry.get("uuid") for entry in merged_manifest]
    hashes = [entry.get("sha256") for entry in merged_manifest]
    if None in uuids or len(set(uuids)) != len(uuids):
        raise ValueError("Merged dataset contains missing or duplicate UUIDs")
    if None in hashes or len(set(hashes)) != len(hashes):
        raise ValueError("Merged dataset contains missing or duplicate game hashes")

    total_train = base_train_count + additional_train_count
    total_eval = base_eval_count
    train_rows = remap_rows(base_train_rows + additional_train_rows, start=0)
    eval_rows = remap_rows(base_eval_rows, start=total_train)
    Dataset.from_list(train_rows).save_to_disk(str(temporary / "dataset"))
    Dataset.from_list(eval_rows).save_to_disk(str(temporary / "eval_dataset"))

    counts = {
        family: {
            difficulty: {"train": 0, "eval": 0} for difficulty in DIFFICULTIES
        }
        for family in FAMILIES
    }
    for entry in merged_manifest:
        family = entry["family"]
        difficulty = entry["difficulty"]
        split = entry["split"]
        counts[family][difficulty][split] += 1

    metadata = {
        "game_files": [entry["game_file"] for entry in merged_manifest],
        "max_scores": [entry["max_score"] for entry in merged_manifest],
        "num_train": total_train,
        "num_eval": total_eval,
        "num_total": total_train + total_eval,
        "seed": base_metadata.get("seed"),
        "seeds": {
            "base": base_metadata.get("seed"),
            "additional": additional_metadata.get("seed"),
        },
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
        "sources": {
            "base": base.name,
            "additional": additional.name,
        },
    }
    (temporary / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    with (temporary / "manifest.jsonl").open("w") as manifest_file:
        for entry in merged_manifest:
            manifest_file.write(json.dumps(entry, sort_keys=True) + "\n")

    os.replace(temporary, output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--additional", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merge_datasets(args.base, args.additional, args.output)
    print(f"Merged TextWorld dataset written to {args.output}")


if __name__ == "__main__":
    main()
