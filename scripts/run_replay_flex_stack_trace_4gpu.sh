#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PYTHONPATH="$ROOT/prime-rl/src:$ROOT/src:${PYTHONPATH:-}"
export KVE_FLEX_TIMING="${KVE_FLEX_TIMING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

TRACE_ROOT="${TRACE_ROOT:-/scratch/epp/tw-evict-diag-defaultfastpath-t10-s2-rf7-10step-8g/rollouts}"
STEP="${STEP:-1}"
MODE="${MODE:-single}"
STACK_SIZE="${STACK_SIZE:-1}"
STACK_TOKEN_BUDGET="${STACK_TOKEN_BUDGET:-}"
MAX_GROUPS="${MAX_GROUPS:-0}"
SKIP_GROUPS="${SKIP_GROUPS:-0}"
WARMUP="${WARMUP:-1}"
ITERS="${ITERS:-1}"
MODEL="${MODEL:-Qwen/Qwen3-4B-Instruct-2507}"
DTYPE="${DTYPE:-bfloat16}"
DEBUG_NUM_LAYERS="${DEBUG_NUM_LAYERS:-}"
RANKS="${RANKS:-0,1,2,3}"
RUN_NAME="${RUN_NAME:-trace-replay-${MODE}-step${STEP}}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/epp/${RUN_NAME}}"
RANDOM_INIT="${RANDOM_INIT:-1}"
ACTIVATION_CHECKPOINTING="${ACTIVATION_CHECKPOINTING:-1}"
ACTIVATION_CHECKPOINTING_FREQ="${ACTIVATION_CHECKPOINTING_FREQ:-1}"
ACTIVATION_OFFLOADING="${ACTIVATION_OFFLOADING:-0}"
ACTIVATION_OFFLOAD_PIN_MEMORY="${ACTIVATION_OFFLOAD_PIN_MEMORY:-0}"
ACTIVATION_OFFLOAD_MAX_INFLIGHT="${ACTIVATION_OFFLOAD_MAX_INFLIGHT:-1}"
SELECTED_LOGPROB_HEAD="${SELECTED_LOGPROB_HEAD:-0}"
SELECTED_LOGPROB_CHUNK_SIZE="${SELECTED_LOGPROB_CHUNK_SIZE:-8192}"
RSS_TRACE="${RSS_TRACE:-1}"
USE_GLOBAL_GROUPING="${USE_GLOBAL_GROUPING:-1}"

mkdir -p "$OUTPUT_DIR"

IFS=',' read -r -a rank_ids <<< "$RANKS"
if [ "${#rank_ids[@]}" -gt 4 ]; then
    echo "This wrapper maps one process per local GPU and supports at most 4 ranks." >&2
    exit 2
fi

echo "RUN_NAME=$RUN_NAME"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "TRACE_ROOT=$TRACE_ROOT STEP=$STEP RANKS=$RANKS"
echo "MODE=$MODE STACK_SIZE=$STACK_SIZE STACK_TOKEN_BUDGET=${STACK_TOKEN_BUDGET:-none}"
echo "MAX_GROUPS=$MAX_GROUPS SKIP_GROUPS=$SKIP_GROUPS WARMUP=$WARMUP ITERS=$ITERS"
echo "MODEL=$MODEL DTYPE=$DTYPE RANDOM_INIT=$RANDOM_INIT ACTIVATION_CHECKPOINTING=$ACTIVATION_CHECKPOINTING ACTIVATION_CHECKPOINTING_FREQ=$ACTIVATION_CHECKPOINTING_FREQ"
echo "ACTIVATION_OFFLOADING=$ACTIVATION_OFFLOADING ACTIVATION_OFFLOAD_PIN_MEMORY=$ACTIVATION_OFFLOAD_PIN_MEMORY ACTIVATION_OFFLOAD_MAX_INFLIGHT=$ACTIVATION_OFFLOAD_MAX_INFLIGHT RSS_TRACE=$RSS_TRACE USE_GLOBAL_GROUPING=$USE_GLOBAL_GROUPING"
echo "SELECTED_LOGPROB_HEAD=$SELECTED_LOGPROB_HEAD SELECTED_LOGPROB_CHUNK_SIZE=$SELECTED_LOGPROB_CHUNK_SIZE"
echo "KVE_FLEX_TIMING=$KVE_FLEX_TIMING PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

pids=()
logs=()
ranks=()

for local_gpu in "${!rank_ids[@]}"; do
    rank="${rank_ids[$local_gpu]}"
    rank_bin="${TRACE_ROOT}/step_${STEP}/rank_${rank}.bin"
    log_path="${OUTPUT_DIR}/rank_${rank}.log"
    if [ ! -f "$rank_bin" ]; then
        echo "Missing rank bin: $rank_bin" >&2
        exit 3
    fi

    args=(
        .venv/bin/python
        scripts/replay_flex_stack_trace.py
        --rank-bin "$rank_bin"
        --model "$MODEL"
        --mode "$MODE"
        --stack-size "$STACK_SIZE"
        --max-groups "$MAX_GROUPS"
        --skip-groups "$SKIP_GROUPS"
        --warmup "$WARMUP"
        --iters "$ITERS"
        --dtype "$DTYPE"
        --activation-checkpointing-freq "$ACTIVATION_CHECKPOINTING_FREQ"
    )
    if [ -n "$STACK_TOKEN_BUDGET" ]; then
        args+=(--stack-token-budget "$STACK_TOKEN_BUDGET")
    fi
    if [ -n "$DEBUG_NUM_LAYERS" ]; then
        args+=(--debug-num-layers "$DEBUG_NUM_LAYERS")
    fi
    if [ "$RANDOM_INIT" = "0" ]; then
        args+=(--no-random-init)
    fi
    if [ "$ACTIVATION_CHECKPOINTING" = "0" ]; then
        args+=(--no-activation-checkpointing)
    fi
    if [ "$ACTIVATION_OFFLOADING" = "1" ]; then
        args+=(--activation-offloading)
    else
        args+=(--no-activation-offloading)
    fi
    if [ "$ACTIVATION_OFFLOAD_PIN_MEMORY" = "1" ]; then
        args+=(--activation-offload-pin-memory)
    else
        args+=(--no-activation-offload-pin-memory)
    fi
    args+=(--activation-offload-max-inflight "$ACTIVATION_OFFLOAD_MAX_INFLIGHT")
    if [ "$SELECTED_LOGPROB_HEAD" = "1" ]; then
        args+=(--selected-logprob-head --selected-logprob-chunk-size "$SELECTED_LOGPROB_CHUNK_SIZE")
    fi
    if [ "$RSS_TRACE" = "1" ]; then
        args+=(--rss-trace)
    else
        args+=(--no-rss-trace)
    fi
    if [ "$USE_GLOBAL_GROUPING" = "1" ]; then
        for global_rank in "${rank_ids[@]}"; do
            args+=(--global-rank-bin "${TRACE_ROOT}/step_${STEP}/rank_${global_rank}.bin")
        done
    fi

    (
        export CUDA_VISIBLE_DEVICES="$local_gpu"
        echo "rank=$rank local_gpu=$local_gpu rank_bin=$rank_bin"
        echo "command=${args[*]}"
        "${args[@]}"
    ) > "$log_path" 2>&1 &
    pids+=("$!")
    logs+=("$log_path")
    ranks+=("$rank")
    echo "launched rank=$rank local_gpu=$local_gpu pid=${pids[$(( ${#pids[@]} - 1 ))]} log=$log_path"
done

failures=0
for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
        echo "rank=${ranks[$idx]} failed; see ${logs[$idx]}" >&2
        failures=$((failures + 1))
    fi
done

summary_path="${OUTPUT_DIR}/summary.txt"
: > "$summary_path"
echo "===== replay summaries ====="
for idx in "${!logs[@]}"; do
    log_path="${logs[$idx]}"
    rank="${ranks[$idx]}"
    echo "----- rank $rank: $log_path -----" | tee -a "$summary_path"
    grep -E "^(loaded=|group=|SUMMARY)|FLEX-.*TIMING|Traceback|RuntimeError|CUDA out|OutOfMemory" "$log_path" \
        | tail -120 \
        | tee -a "$summary_path" || true
done
echo "summary_path=$summary_path"

if [ "$failures" -ne 0 ]; then
    exit 1
fi
