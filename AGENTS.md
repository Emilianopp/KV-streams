# KV-streams

## Project Overview

Native vLLM KV cache compaction for RL training. Fork of vLLM 0.19 with
scheduler-integrated block-level eviction + segmented forward training (no
detach) on a matching prime-rl fork.

## Repository layout

- `vllm/` — Submodule: forked vLLM v0.19.0
  (`https://github.com/Emilianopp/kv-streams-vllm`). Compaction additions in
  `vllm/v1/core/compaction/` plus small edits to scheduler, request, and
  gpu_model_runner.
- `prime-rl/` — Submodule: forked prime-rl
  (`https://github.com/Emilianopp/kv-streams-prime-rl`). Contains the
  unified-dispatch trainer, segmented-forward path, compaction events
  plumbing, and config validators.
- `src/kv_eviction/` — Integration layer: `segmented_forward.py` (training
  path), `env.py` (verifiers monkey-patches that plumb compaction_events
  through to rollouts), `types.py` (wire format).
- `configs/textworld/` — the paper's TextWorld RL configs, one standalone
  file per arm (`uv run rl @ configs/textworld/<arm>.toml`).
- `experiments/textworld_env/` — TextWorld env package and dataset tooling.
- `plans/` — Design notes referenced from code comments.

## Current status (as of Phase 3 D5 resolution)

- **Phase 1 repo setup**: complete. Both forks live on GitHub.
- **Phase 2 vLLM compaction**: complete. Scheduler-integrated block-level
  eviction with sliding-window strategy, CompactionEvent metadata, Phase
  2.1 prompt-alignment fix, async_scheduling guard, LMCache guard.
- **Phase 3 training integration**: complete. Segmented forward with
  per-segment backward (bptt_segments=1), entropy aggregation, AC +
  compaction incompatibility rejected at config load, D5 unified
  dispatch (every sample in a compaction run goes through
  segmented_forward regardless of whether inference triggered events).
- **Phase 4 experiments**: smokes #1–#4 pass. Smoke #5 (production-
  config smoke) is the next open work item.

Most recent validated config: 5-step smoke at batch_size=64,
rollouts_per_example=8, window=4096, stride=512, peak memory 45.9 GiB
(34 GiB headroom under 80 GB), Mismatch KL 0.0009 (kernel floor).

## Key design decisions

- **V1 engine** (vLLM 0.19 default). NOT MRV2 (experimental).
- **Block-aligned eviction**: whole PagedAttention blocks. Stride must be
  a multiple of `block_size`.
- **Splice block_table** after eviction (not null_block — kernel would
  read garbage from middle holes).
- **position_offset on Request** for correct RoPE after eviction.
  Physical seq_len decreases; positions = physical + offset restore the
  pre-eviction absolute index used at inference time.
- **Segmented forward, no detach** for training — same flash_attn kernel
  as inference → zero kernel-floor KL mismatch. Per-segment backward
  with bptt_segments=1 (O(1 segment) memory, M3 semantics).
- **NOT FlexAttention** — SDPA 4D mask forces the math backend, a
  different kernel from inference, which produces a systematic KL gap.
- **Unified dispatch** (D5): in a compaction training run
  (`compaction.window_size > 0`), every sample is routed through
  `segmented_forward`, including event-less short rollouts that run as
  a single-segment forward. Eliminates the compaction → text modality
  transition that caused the smoke-#4 OOM cascade.

## Environment

Primary development environment is NERSC Perlmutter (see README.cluster.md
for the Perlmutter-specific launch notes). The tracked README.md is
cluster-neutral: anyone with a CUDA 12.8 + A100 box and `uv` can run
`bash setup.sh` from a fresh clone.
