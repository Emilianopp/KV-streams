#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

GPUS="${GPUS:-0,1}"
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NPROC="${NPROC:-${#GPU_LIST[@]}}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
SEQ_LEN="${SEQ_LEN:-16384}"
TURNS="${TURNS:-40}"
COMPLETION_TOKENS="${COMPLETION_TOKENS:-128}"
STACK_SIZES="${STACK_SIZES:-1,2,4}"
MODES="${MODES:-single,vertical,horizontal}"
WARMUP="${WARMUP:-1}"
ITERS="${ITERS:-3}"
EVICT_TOKENS="${EVICT_TOKENS:-512}"
PROTECTED_PREFIX="${PROTECTED_PREFIX:-1024}"
JITTER_STEP="${JITTER_STEP:-0}"
DEBUG_NUM_LAYERS="${DEBUG_NUM_LAYERS:-}"
RANDOM_INIT="${RANDOM_INIT:-1}"
AC_MODE="${AC_MODE:-prime}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/.local_artifacts/bench/flex_stack_$(date +%Y%m%d_%H%M%S)}"

TORCHRUN="${TORCHRUN:-$ROOT/.venv/bin/torchrun}"
if [[ ! -x "$TORCHRUN" ]]; then
  TORCHRUN="$(command -v torchrun)"
fi

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/run.log"

cmd=(
  "$TORCHRUN"
  --standalone
  --nproc-per-node="$NPROC"
  "$ROOT/scripts/bench_flex_stack_local.py"
  --model "$MODEL"
  --seq-len "$SEQ_LEN"
  --turns "$TURNS"
  --completion-tokens "$COMPLETION_TOKENS"
  --stack-sizes "$STACK_SIZES"
  --modes "$MODES"
  --warmup "$WARMUP"
  --iters "$ITERS"
  --evict-tokens "$EVICT_TOKENS"
  --protected-prefix "$PROTECTED_PREFIX"
  --jitter-step "$JITTER_STEP"
)

if [[ -n "$DEBUG_NUM_LAYERS" ]]; then
  cmd+=(--debug-num-layers "$DEBUG_NUM_LAYERS")
fi
if [[ "$RANDOM_INIT" == "1" ]]; then
  cmd+=(--random-init)
else
  cmd+=(--no-random-init)
fi
cmd+=(--activation-checkpointing "$AC_MODE")

{
  echo "root: $ROOT"
  echo "output: $OUTPUT_DIR"
  echo "gpus: $GPUS nproc=$NPROC"
  echo "model: $MODEL seq_len=$SEQ_LEN synthetic_calls=$TURNS completion_tokens=$COMPLETION_TOKENS"
  echo "stack_sizes: $STACK_SIZES modes=$MODES warmup=$WARMUP iters=$ITERS"
  nvidia-smi --query-gpu=index,name,memory.total,memory.used --format=csv || true
  echo
  CUDA_VISIBLE_DEVICES="$GPUS" \
  PYTHONPATH="$ROOT/prime-rl/src:$ROOT/src:${PYTHONPATH:-}" \
  "${cmd[@]}"
} 2>&1 | tee "$LOG"

echo "log: $LOG"
