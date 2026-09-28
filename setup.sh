#!/bin/bash
# Full setup script for KV-streams. Run from the project root on a box
# with CUDA 12.8 + A100/H100 GPUs and `uv` installed.
#   bash setup.sh
set -euo pipefail

# Resolve the project root to the directory containing this script so
# the setup works regardless of where the repo was cloned.
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"

# Optional: pin the uv cache to a scratch path if you're on a shared
# filesystem where $HOME is small (e.g. HPC clusters). Leave unset to
# use uv's default cache under $HOME/.cache/uv.
# export UV_CACHE_DIR=/path/to/fast/uv-cache

echo "=== Creating venv ==="
uv venv .venv --python python3.12 --clear
source .venv/bin/activate

# Install order handles transformers version conflict:
#   vLLM 0.19.0 requires transformers >= 4.56, < 5
#   prime-rl requires transformers >= 5.1.0.dev0 (HF git)
#   prime-rl ships a vLLM plugin (transformers_v5_compat) that patches the gap.

echo "=== Step 1a: vLLM 0.19.0 from PyPI to pull in runtime deps ==="
# This pulls in torch, ray, numpy, transformers, etc. — all the
# transitive deps we need. We'll overwrite the vllm package itself
# with an editable install of our compaction-enabled fork in step 1b
# so Phase 2 + Phase 3.1 changes under vllm/vllm/v1/core/compaction/
# and related files take effect.
uv pip install "vllm==0.19.0"

echo "=== Step 1b: editable install of the compaction fork ==="
# The editable vLLM build runs with --no-build-isolation, so its build
# backend (setuptools + setuptools_scm) must already be present in the venv.
uv pip install setuptools setuptools_scm wheel
# Pin the precompiled-wheel CUDA variant to cu129 (torch 2.10). Without this,
# vLLM's detect_system_cuda_variant() reads the DRIVER's CUDA version from
# nvidia-smi; on hosts with a CUDA 13 driver it selects the cu130 nightly,
# which pulls torch 2.11 and ships a wheel missing vllm/_C.abi3.so (breaking
# `import vllm._C`) and mismatching the cu128/torch2.10 flash-attn wheel below.
export VLLM_MAIN_CUDA_VERSION=12.9
# VLLM_USE_PRECOMPILED=1 downloads vllm's CI-built .so artifacts and
# symlinks them into the source tree so we don't need a full source
# rebuild of C++/CUDA extensions. The vllm_flash_attn symlink-mode
# shim (committed on the compaction branch) registers a virtual
# `flash_attn` package at import time, which the symlinked cute/
# files expect.
VLLM_USE_PRECOMPILED=1 uv pip install -e ./vllm --no-build-isolation

echo "=== Step 2: transformers v5 from HF git ==="
uv pip install "transformers @ git+https://github.com/huggingface/transformers.git@c1c3424"

echo "=== Step 3: flash-attn pre-built wheel ==="
uv pip install "flash-attn @ https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu128torch2.10-cp312-cp312-linux_x86_64.whl"

echo "=== Step 4: prime-rl extra deps ==="
uv pip install \
    "beartype>=0.21.0" "jaxtyping>=0.3.2" "tomli-w>=1.2.0" \
    "prime>=0.5.37" "aiolimiter>=1.2.1" "setproctitle>=1.3.0" \
    "tilelang>=0.1.8" "liger-kernel>=0.5.10" \
    "ring-flash-attn>=0.1.8" "wandb>=0.24.2" \
    "verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@0760204" \
    "protobuf>=6.31.1,<7" \
    "torchtitan @ git+https://github.com/pytorch/torchtitan@a1fdd7e" \
    "pydantic-config @ git+https://github.com/samsja/pydantic_config.git@main" \
    "dion @ git+https://github.com/samsja/dion.git@d891eeb" \
    "flash-linear-attention @ git+https://github.com/fla-org/flash-linear-attention"

echo "=== Step 5: prime-rl editable (no-deps) ==="
uv pip install -e ./prime-rl --no-deps

echo "=== Step 6: kv-eviction editable ==="
uv pip install -e .

echo "=== Step 7: textworld-env editable ==="
# Ships the TextWorld multi-turn env package (a plain vf.MultiTurnEnv
# subclass) and its textworld>=1.6.0 runtime dep. The kv-eviction
# compaction monkey-patches automatically apply to it — no env-side code.
uv pip install -e ./experiments/textworld_env

echo ""
echo "=== Verification ==="
python << 'PYEOF'
import sys, torch
print(f"Python {sys.version.split()[0]}")
print(f"torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")
import vllm; print(f"vllm {vllm.__version__}")
import transformers; print(f"transformers {transformers.__version__}")
import flash_attn; print(f"flash_attn {flash_attn.__version__}")
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.request import Request
print("vLLM v1 core imports OK")
from prime_rl.transport.types import TrainingSample
from prime_rl.trainer.model import forward
print("prime-rl imports OK")
import kv_eviction; print("kv_eviction OK")
import textworld_env; print("textworld_env OK")
print("\n=== SETUP COMPLETE ===")
print(f"Activate with: source {sys.prefix}/bin/activate")
PYEOF
