# Launching the TextWorld experiments

The paper RL configs in `configs/textworld/` train on the `textworld-env`
package with the all-family TextWorld dataset (see `reproducing.md`). The
cooking-mix dataset below is the older single-family setup.

## 0. One-time prerequisites

Assumes a fresh clone that has already gone through `bash setup.sh` (see
the top-level `README.md`). After setup:

```bash
source .venv/bin/activate
python -c "import textworld_env; print('env OK')"
```

## 1. Dataset

The 5000-game mix is deterministically regeneratable. On Perlmutter it's
already saved at `/pscratch/sd/s/siddart2/datasets/textworld_cooking_mix`.

On a fresh box:

```bash
# Writes to ${KV_EVICTION_DATA_ROOT:-$PWD/data}/textworld_cooking_mix
# Takes ~20 min on a single CPU node. Deterministic seed=42.
bash experiments/textworld_env/prepare_dataset.sh
```

The resulting directory contains:

```
textworld_cooking_mix/
├── metadata.json        # difficulty map + RELATIVE game_files paths
├── dataset/             # HF save_to_disk format (5000 rows)
├── games/               # .z8 + .json + .ni per game (~3.2 GB)
└── eval_dataset/        # optional held-out eval split
```

`metadata.json` stores **relative** `games/game_XXXXX.z8` paths so the
directory is relocatable across machines.

## 2. Training

The paper configs in `configs/textworld/` are standalone. Substitute the
dataset path and launch one arm:

```bash
sed "s|__TEXTWORLD_DATASET__|/path/to/textworld_all4_total60000_train59744_eval256_seed20260719|g" \
    configs/textworld/kv_streams_markovian_thinker.toml > /tmp/run.toml
uv run rl @ /tmp/run.toml
```

Every config targets one 8-GPU node (4 trainer + 4 inference); `rl` launches
both sides. See the arm table in the top-level `README.md`.

## 3. Inference-only eval

`eval_textworld.py` runs rollouts against an already-running vLLM server
(no training) — see its `--help` and `README.md` in this directory.
