# textworld-env

TextWorld interactive-fiction cooking environment for kv-eviction multi-turn
RL training. Plain `vf.MultiTurnEnv` subclass — compaction and block-aligned
message padding are applied transparently by the kv-eviction monkey-patches
in `src/kv_eviction/env.py`.

## Install

Editable install from the repo root:

```bash
uv pip install -e ./experiments/textworld_env
```

This is done automatically by `setup.sh` Step 7. A fresh clone + `bash setup.sh`
leaves `import textworld_env` working in the venv.

## Dataset

The production mix is 5000 train cooking games across 5 difficulty tiers
(easy-nav 1250, current 500, hard 1500, hard-12room 1000, hard-drop 750),
plus 100 held-out eval games (20 per tier).
Average reward with Qwen3-4B-Instruct-2507 base ≈ 0.24, average rollout
length ≈ 12k tokens.

On NERSC Perlmutter it's already saved at
`/pscratch/sd/s/siddart2/datasets/textworld_cooking_mix`.

On a fresh clone, regenerate deterministically (seed=42, ~20 min CPU):

```bash
bash experiments/textworld_env/prepare_dataset.sh
```

This writes into `${KV_EVICTION_DATA_ROOT:-$PWD/data}/textworld_cooking_mix/`
with `metadata.json` holding **relative** `games/game_XXXXX.z8` paths, so the
directory is fully relocatable.

## Env arguments

```python
import verifiers as vf
env = vf.load_environment(
    "textworld-env",
    dataset_path="/path/to/textworld_cooking_mix",
    max_episode_steps=50,
    num_train_examples=4000,
    num_eval_examples=100,
    seed=42,
)
```

- `dataset_path` — directory containing `metadata.json`, `dataset/` (HF format), and `games/*.z8`.
- `max_episode_steps` — hard cap on `env_response` calls per rollout.
- `num_train_examples` / `num_eval_examples` — select train/eval split sizes. Modern datasets load held-out eval rows from `eval_dataset/`.

## 100-question eval

The standalone evaluator uses the held-out `eval_dataset/` split by default.
It reports:

- `success_rate`: mean normalized TextWorld score over the eval set.
- `hard_success_rate`: binary win rate, 1 iff the final score reaches max score.

```bash
python experiments/textworld_env/eval_textworld.py \
  --dataset /path/to/textworld_cooking_mix \
  --base-url http://localhost:8000/v1 \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --num-examples 100 \
  --eval-source eval \
  --eval-set-json experiments/textworld_env/eval_sets/textworld_eval_100_seed42.json \
  --output-json experiments/textworld_env/results_full_context.json
```

The first run writes the eval-set manifest; later runs load the same 100 game
ids so all configs are compared on identical questions.

## Why `vf.MultiTurnEnv` and not the mkv-rl variants

`mkv-rl/experiments/textworld_rl/` has three env files:

- `textworld_env.py` — plain `vf.MultiTurnEnv` (this is what we port)
- `mkv_textworld_env.py` — wraps the plain env with session-based KV eviction *inside* the env class (manual prompt truncation between turns)
- `markov_thinker_textworld_env.py` — another variant with per-turn history pruning

kv-eviction does compaction at the vLLM scheduler level (turn-based eviction
of whole user+assistant pairs), so **the env must NOT truncate its own
history** — that would double-evict and break the padded-token-stream
contract the trainer depends on. Only the plain env is safe here.

## Notes on concurrency

TextWorld's `tatsu` parser has a module-level singleton with a non-thread-safe
`_rule_stack`. The env file includes a `_tw_start_lock` + post-fork parser
reset — do not remove them. Concurrent `run_in_executor` calls from
`setup_state()` will otherwise corrupt game files.
