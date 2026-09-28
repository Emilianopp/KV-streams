#!/bin/bash
# Generate the production hard-mixed TextWorld cooking dataset (5000 games,
# 5 difficulty tiers) deterministically. ~20 min CPU on a single node.
#
# Output: ${KV_EVICTION_DATA_ROOT:-$PWD/data}/textworld_cooking_mix/
#   - dataset/            HF dataset with per-game metadata rows
#   - games/*.z8          pre-compiled Z-machine game files
#   - metadata.json       game_files list (RELATIVE paths) + max_scores + difficulty map
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${KV_EVICTION_DATA_ROOT:-$PWD/data}"
OUT_DIR="$DATA_ROOT/textworld_cooking_mix"
WORKERS="${TEXTWORLD_GEN_WORKERS:-1}"

mkdir -p "$DATA_ROOT"

python "$SCRIPT_DIR/generate_dataset.py" \
    --output "$OUT_DIR" \
    --mix "easy-nav:1250" "current:500" "hard:1500" "hard-12room:1000" "hard-drop:750" \
    --seed 42 \
    --workers "$WORKERS"

echo "=== Dataset ready: $OUT_DIR ==="
