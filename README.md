# KV-streams

Native vLLM KV cache compaction for RL training. Fork of vLLM 0.19 with
scheduler-integrated block-level eviction.

When a request's KV length exceeds `--compaction-window-size`, the scheduler
evicts the oldest post-prompt blocks (one `--compaction-stride` at a time).
The request becomes physically shorter -- attention is faster, no KV recompute.

## Setup

### Prerequisites

- A CUDA 12.8 environment with A100 (or newer) GPUs
- Python 3.12
- [`uv`](https://docs.astral.sh/uv/) package manager
- A container runtime is recommended but not required. A suitable base
  image is `docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8`,
  which bundles CUDA 12.8, Python 3.12, and `uv`.

### 1. Clone the repo

```bash
git clone --recursive https://github.com/Emilianopp/KV-streams.git
cd KV-streams
```

`vllm/` ([kv-streams-vllm](https://github.com/Emilianopp/kv-streams-vllm)) and
`prime-rl/` ([kv-streams-prime-rl](https://github.com/Emilianopp/kv-streams-prime-rl))
are submodules. If you cloned without `--recursive`:

```bash
git submodule update --init
```

### 2. Install

```bash
# One-command install (creates venv, installs vLLM editable + all deps)
bash setup.sh
```

Or manually, for development:

```bash
uv venv .venv --python python3.12 --clear
source .venv/bin/activate

# Editable install of the compaction-enabled vLLM fork. VLLM_USE_PRECOMPILED=1
# downloads vLLM's CI-built .so artifacts and symlinks them into the source
# tree so Python edits under vllm/vllm/v1/core/compaction/ take effect
# without a full C++/CUDA rebuild.
VLLM_USE_PRECOMPILED=1 uv pip install -e ./vllm --no-build-isolation

# kv-eviction integration layer
uv pip install -e .
```

### 3. Verify installation

```bash
source .venv/bin/activate
python -c "
from vllm.v1.core.compaction import CompactingKVCacheManager, CompactionEvent
print('Compaction imports OK')
import vllm; print(f'vLLM {vllm.__version__}')
"
```

## Inference

### Quick start -- vLLM server with compaction

```bash
source .venv/bin/activate

# Launch vLLM with compaction enabled.
# window=4096: eviction triggers when KV exceeds 4096 tokens
# stride=512:  evict 512 tokens (32 blocks) per compaction event
# block_size defaults to 16, stride must be a multiple of it
# async_scheduling=False is required (see Constraints below)
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --max-model-len 16384 \
  --enable-prefix-caching false \
  --async-scheduling false \
  --compaction-window-size 4096 \
  --compaction-stride 512 \
  --tensor-parallel-size 1 \
  --port 8000
```

### Query the server

```bash
# Standard OpenAI-compatible API -- compaction is transparent
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-4B",
    "messages": [{"role": "user", "content": "Solve step by step: what is 1234 * 5678?"}],
    "max_tokens": 8192,
    "temperature": 0.7
  }' | python -m json.tool
```

### Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="Qwen/Qwen3-4B",
    messages=[{"role": "user", "content": "Write a long essay about AI safety."}],
    max_tokens=8192,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

### Offline inference (no server)

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen3-4B",
    max_model_len=16384,
    enable_prefix_caching=False,
    async_scheduling=False,  # required with compaction (see Constraints)
    # Compaction args
    compaction_window_size=4096,
    compaction_stride=512,
)

outputs = llm.generate(
    ["Explain quantum computing in detail."],
    SamplingParams(max_tokens=8192, temperature=0.7),
)
print(outputs[0].outputs[0].text)
```

## How it works

```
Request generating tokens...
  [prompt: 200 tokens] [gen: 3800 tokens]  total KV = 4000  (below window)

Next token generated:
  [prompt: 200 tokens] [gen: 3801 tokens]  total KV = 4001  (still below)

...after 96 more tokens:
  [prompt: 200 tokens] [gen: 3897 tokens]  total KV = 4097  (exceeds window!)

Compaction fires:
  1. Evict 512 oldest generation tokens (32 blocks) from KV cache
  2. Trim token IDs to match
  3. Reduce num_computed_tokens by 512
  4. Add position_offset += 512 for correct RoPE
  5. Mark request for model runner rebuild

After compaction:
  [prompt: 200 tokens] [gen: 3385 tokens]  total KV = 3585  (back under window)
  Physical seq_len = 3585  (faster attention!)
  RoPE positions = physical + 512  (correct absolute positions)
```

The request continues generating as if nothing happened. Attention cost stays
bounded by the window size instead of growing linearly with generation length.

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--compaction-window-size` | `0` (off) | KV token count that triggers eviction |
| `--compaction-stride` | `0` | Tokens to evict per event (must be multiple of block_size) |

### Constraints

- `--enable-prefix-caching false` required (compaction splices blocks)
- `--async-scheduling false` required. In async mode vLLM pre-schedules the next
  step before the current step's output is processed, so
  `num_output_placeholders` is always nonzero when `update_from_output` runs.
  The compaction trigger is guarded on `num_output_placeholders == 0`, so with
  async scheduling compaction never fires and the run silently degenerates to
  full context. The scheduler asserts this at init.
- Pipeline parallelism (`--pipeline-parallel-size > 1`) not supported
- `stride` must be a multiple of `block_size` (default 16)
- `window_size` must be greater than `stride`

### Recommended settings

| Use case | Window | Stride | Notes |
|----------|--------|--------|-------|
| Long-form generation | 4096 | 512 | Good balance of speed and context |
| Very long generation | 8192 | 1024 | More context retained |
| Aggressive compaction | 2048 | 512 | Maximum speed, less context |

## RL training (TextWorld)

`configs/textworld/` holds the configs behind the paper's TextWorld results:
Qwen3-4B-Instruct-2507 on one 8-GPU H100 node (4 inference + 4 trainer),
batch 512, 8 rollouts per prompt, 500 steps, 32k budget, compaction every 10
turns. Each file is standalone and was checked against the W&B config of the
run it produced.

| Strategy | KV-streams | Re-prefill compaction |
|---|---|---|
| Full context (reference) | `full_context.toml` | — |
| Sliding window | `kv_streams_sliding_window.toml` | `reprefill_sliding_window.toml` |
| Markovian Thinker | `kv_streams_markovian_thinker.toml` | `reprefill_markovian_thinker.toml` |
| Summary | `kv_streams_summary.toml` | `reprefill_summary.toml` |

`sft/` holds the SFT-warm-start + RL runs (Section 6.1): the same KV-streams
sliding-window and Markovian Thinker configs, initialised from
[`ppEmiliano/qwen3-4b-textworld-sft-eviction-mixed`](https://huggingface.co/ppEmiliano/qwen3-4b-textworld-sft-eviction-mixed).

```bash
# Build the dataset (reproducing.md), then substitute its path:
sed "s|__TEXTWORLD_DATASET__|$PWD/data/textworld_all4_total60000_train59744_eval256_seed20260719|g" \
    configs/textworld/kv_streams_sliding_window.toml > /tmp/run.toml
uv run rl @ /tmp/run.toml
```

KV-streams arms set `kv_mode = "kv-eviction"`; re-prefill arms set
`markovian_thinker.enabled = true` with `kv_eviction` off. Add `--dry-run` to
validate and dump the resolved configs without launching.

### What to watch during training

- `progress/reward/mean` — should trend up.
- `loss/mismatch_kl_mean` — trainer-vs-inference logprob agreement. Sits at
  the kernel floor (~1e-3) when training and inference see the same KV; any
  climb is a correctness bug.
- `progress/entropy/mean` — flat or mild downward drift is fine; collapse
  indicates mode collapse.

## Project structure

```
KV-streams/
  vllm/                  # Submodule: kv-streams-vllm (vLLM 0.19 fork, KV eviction)
    vllm/v1/core/compaction/
  prime-rl/              # Submodule: kv-streams-prime-rl (trainer, eviction plumbing)
  src/kv_eviction/       # Integration layer (env patches, padding, segmented forward)
  configs/textworld/     # Paper RL configs, one file per arm
  plans/                 # Design notes referenced from code comments
  experiments/
    textworld_env/       # TextWorld env package, dataset generation, eval
  scripts/               # Analysis / benchmarking utilities
  tests/
  setup.sh               # One-command install
```
