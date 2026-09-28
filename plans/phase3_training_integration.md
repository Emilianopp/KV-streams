# Phase 3: Training Integration — Segmented Forward (No Detach)

## Status: COMPLETE (2026-04-11)

All Phase 3 components landed and validated end-to-end by smoke #4 v6
(5-step stability run at batch_size=64, rollouts_per_example=8, window=4096,
stride=512). Key metrics from v6:

- Steps 0–4 all pass. Peak memory flat at 45.9 GiB (34 GiB headroom under
  80 GB), Mismatch KL 0.0009–0.0010 (kernel floor), entropy smoothly
  decreasing, reward trending upward.
- D5 (compaction → text modality transition OOM) resolved via unified
  dispatch: every sample in a compaction run is routed through
  `segmented_forward`, including event-less short rollouts that run as
  single-segment forwards. See the "D5 resolution" section at the end of
  this file for the full investigation.
- Published forks:
  - `https://github.com/HyperPotatoNeo/vllm` (branch `compaction`)
  - `https://github.com/HyperPotatoNeo/tba-prime` (branch `kv-eviction`)
  - `https://github.com/HyperPotatoNeo/kv-eviction` (top-level)

Next open item is Phase 4 smoke #5 (production-config run). Phase 3 is
frozen; no further edits should be needed.

## Goal

Implement the training-side components that match the inference-side compaction from Phase 2.
After this phase, prime-rl can train on rollouts from the compaction-enabled vLLM server
using a segmented forward pass that:
1. Uses the same `flash_attn` kernel per segment as vLLM inference (zero KL mismatch)
2. Does NOT detach retained KV between segments (preserves cross-chunk gradients, G_distal)
3. Works correctly with FSDP2 (dummy passes for rank synchronization)

## Prerequisites

- Phase 2 complete (vLLM with compaction returning `compaction_events` in responses)
- prime-rl submodule at `$SCRATCH/kv-eviction/prime-rl/` (clean main, no modifications)
- Understanding of mkv-rl's `window_forward.py` (the base we adapt from)

## Files to Create

```
src/kv_eviction/
├── segmented_forward.py   # ~200 lines: segmented forward, no detach
├── env.py                 # ~80 lines: vLLM response -> prime-rl RolloutOutput
└── train_hooks.py         # ~50 lines: inject segmented forward into prime-rl
```

---

## 3.1: Segmented Forward — No Detach

**File: `src/kv_eviction/segmented_forward.py`**

Adapted from `$SCRATCH/mkv-rl/src/prime_rl/trainer/rl/window_forward.py`. The key change
is removing `.detach()` on retained KV between segments, which preserves cross-chunk
gradient flow (the G_distal term in our methods document).

### Architecture

```
Segment 0: [prompt | gen_tokens_0..boundary_0]
  → forward pass with use_cache=True
  → extract KV from DynamicCache via backbone hook
  → drop first `stride` assistant KV entries (same as inference eviction)
  → DO NOT DETACH retained KV  ← KEY DIFFERENCE from mkv-rl
  → trim boundary token (will be recomputed by next segment)

Segment 1: [boundary_token_0 | gen_tokens_boundary_0..boundary_1]
  → forward pass with past_key_values = evicted (non-detached) KV
  → same KV eviction cycle...

Last segment: [boundary_token_{n-1} | remaining_tokens]
  → forward pass with past_key_values
  → no eviction, just collect logits
```

### Differences from mkv-rl window_forward.py

| Aspect | mkv-rl (window_forward.py) | kv-eviction (segmented_forward.py) |
|--------|---------------------------|-----------------------------------|
| `.detach()` on retained KV | Yes (line 243) | **No** — retained KV stays in autograd graph |
| Gradient flow between segments | Blocked (each segment independent) | **Preserved** (G_distal term) |
| Memory usage | Lower (detach frees intermediate activations) | Higher (retains activations for backprop) |
| Gradient correctness | Approximate (local gradients only) | Exact (full chain rule through retained KV) |
| Everything else | Same | Same |

### Full Implementation

```python
"""Segmented forward pass with KV prefix drop, NO detach.

Cross-chunk gradients are preserved through retained KV entries between
segments, enabling the G_distal gradient term. Uses the same flash_attn
kernel per segment as vLLM inference for zero KL mismatch.

Adapted from mkv-rl window_forward.py with one critical change:
  - REMOVED: .detach() on retained KV (lines 243-244 of window_forward.py)
  - RESULT: gradients flow back through retained KV to earlier segments

FSDP2 compatible: dummy forward passes pad ranks with fewer segments.
"""

import logging

import torch
from torch import Tensor
from transformers import DynamicCache

logger = logging.getLogger(__name__)


def _get_kv_from_cache(cache: DynamicCache) -> tuple[list[Tensor], list[Tensor]]:
    """Extract per-layer key/value tensors from a DynamicCache.

    Handles both old API (cache.key_cache) and new API (cache.layers[i].keys).
    Returns per-layer tensors shaped [seq, heads, dim].
    """
    num_layers = len(cache)

    # Try new API first (transformers >= 4.49)
    if hasattr(cache, "layers") and len(cache.layers) > 0:
        keys = [
            cache.layers[l].keys[0].permute(1, 0, 2).contiguous()
            for l in range(num_layers)
        ]
        values = [
            cache.layers[l].values[0].permute(1, 0, 2).contiguous()
            for l in range(num_layers)
        ]
    # Fall back to old API
    elif hasattr(cache, "key_cache"):
        keys = [
            cache.key_cache[l][0].permute(1, 0, 2).contiguous()
            for l in range(num_layers)
        ]
        values = [
            cache.value_cache[l][0].permute(1, 0, 2).contiguous()
            for l in range(num_layers)
        ]
    else:
        raise RuntimeError(f"Unknown DynamicCache API: {type(cache)}")

    return keys, values


def segmented_forward(
    model: torch.nn.Module,
    input_ids: Tensor,           # [1, seq_len]
    position_ids: Tensor,        # [1, seq_len]
    segment_boundaries: list[int],  # cumulative completion token counts
    prompt_len: int,
    stride: int,                 # tokens to drop per eviction
    temperature: Tensor,         # [1, seq_len] per-token
    max_forward_passes: int | None = None,  # for FSDP padding
) -> dict[str, Tensor]:  # {"logits": [1, seq_len, vocab]}
    """Run segmented forward passes with KV prefix drop, NO DETACH.

    This is the key difference from mkv-rl's segmented_forward_detached():
    retained KV between segments is NOT detached, so gradients flow back
    through the KV to earlier segments. This preserves the G_distal term.

    For each segment:
    1. Forward pass with use_cache=True to get logits and KV cache.
    2. Between segments: extract KV, drop first `stride` assistant tokens.
       DO NOT DETACH the retained KV.
    3. Next segment uses evicted (but still-differentiable) past_key_values.

    Args:
        model: HuggingFace model (e.g. Qwen3ForCausalLM). Must support use_cache=True.
        input_ids: Full input_ids [1, seq_len] (prompt + all completion tokens).
        position_ids: Full position_ids [1, seq_len]. Must include position offsets
            matching inference (position = physical_pos + cumulative_evicted).
        segment_boundaries: Cumulative completion token counts at end of each segment.
            These come from compaction_events[i].completion_tokens_at_compaction.
            E.g. [3584, 7168, 10752] means segment 0 covers the first 3584 completion
            tokens, segment 1 covers the next 3584, etc.
        prompt_len: Number of prompt tokens.
        stride: Number of assistant KV entries to drop per eviction. Must match the
            compaction_stride used during inference.
        temperature: Per-token temperatures [1, seq_len].
        max_forward_passes: Target forward pass count for FSDP synchronization.
            Must be >= actual segment count. If None, no padding is done.

    Returns:
        Dict with "logits" key containing [1, seq_len, vocab] temperature-scaled logits.
    """
    device = input_ids.device
    seq_len = input_ids.shape[1]
    assert input_ids.shape[0] == 1, "Segmented forward only supports batch_size=1"

    # Edge case: empty completion
    completion_len = seq_len - prompt_len
    if completion_len <= 0:
        logger.warning("Empty completion: prompt_len=%d >= seq_len=%d", prompt_len, seq_len)
        out = model(input_ids=input_ids, position_ids=position_ids)
        raw_logits = out["logits"] if isinstance(out, dict) else out.logits
        logits = raw_logits["logits"] if isinstance(raw_logits, dict) else raw_logits
        scaled = logits / temperature.unsqueeze(-1).to(logits.dtype)
        actual_passes = 1
        target_passes = max_forward_passes or actual_passes
        if target_passes > actual_passes:
            scaled = _pad_with_dummy_passes(
                model, input_ids, position_ids, scaled,
                target_passes - actual_passes, device,
            )
        return {"logits": scaled}

    # Capture past_key_values from the backbone via hook.
    # FSDP2 + VanillaOutputLinear may not propagate past_key_values through
    # the top-level model output.
    captured_kv: dict[str, DynamicCache | None] = {}

    def _capture_kv_hook(_module, _input, output):
        if hasattr(output, "past_key_values"):
            captured_kv["past_key_values"] = output.past_key_values
        elif isinstance(output, dict):
            captured_kv["past_key_values"] = output.get("past_key_values")

    backbone = model.model if hasattr(model, "model") else model
    hook_handle = backbone.register_forward_hook(_capture_kv_hook)

    # Build segment token ranges in input_ids space.
    # Segment 0: input_ids[0 : prompt_len + boundary_0]
    # Segment k>0: input_ids[prompt_len + boundary_{k-1} - 1 : prompt_len + boundary_k]
    #   The -1 creates boundary token overlap for logit recomputation.
    seg_input_ranges: list[tuple[int, int]] = []
    prev_boundary = 0
    for i, boundary in enumerate(segment_boundaries):
        if i == 0:
            seg_start = 0
        else:
            seg_start = prompt_len + prev_boundary - 1
        seg_end = min(prompt_len + boundary, seq_len)
        if seg_start < seg_end:
            seg_input_ranges.append((seg_start, seg_end))
        prev_boundary = boundary

    # Handle case where boundaries don't cover full completion
    last_covered = prompt_len + segment_boundaries[-1] if segment_boundaries else prompt_len
    if last_covered < seq_len and seg_input_ranges:
        last_start, _ = seg_input_ranges[-1]
        seg_input_ranges[-1] = (last_start, seq_len)

    all_logits_pieces: list[Tensor] = []
    past_key_values: DynamicCache | None = None

    saved_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True

    try:
        for seg_idx, (seg_start, seg_end) in enumerate(seg_input_ranges):
            seg_ids = input_ids[:, seg_start:seg_end]
            seg_positions = position_ids[:, seg_start:seg_end]
            seg_temps = temperature[:, seg_start:seg_end]

            out = model(
                input_ids=seg_ids,
                position_ids=seg_positions,
                past_key_values=past_key_values,
                use_cache=True,
            )

            raw_logits = out["logits"] if isinstance(out, dict) else out.logits
            seg_logits = raw_logits["logits"] if isinstance(raw_logits, dict) else raw_logits
            scaled_seg_logits = seg_logits / seg_temps.unsqueeze(-1).to(seg_logits.dtype)

            is_last_segment = seg_idx == len(seg_input_ranges) - 1

            if is_last_segment:
                all_logits_pieces.append(scaled_seg_logits)
            else:
                # Drop last logit — recomputed by next segment (boundary overlap)
                all_logits_pieces.append(scaled_seg_logits[:, :-1, :])

            # Between segments: KV prefix drop, NO DETACH
            if not is_last_segment:
                kv_cache = captured_kv.get("past_key_values")
                assert kv_cache is not None, (
                    "Hook did not capture past_key_values. "
                    "Ensure model uses impl='hf' which supports use_cache=True."
                )
                captured_kv.clear()

                keys, values = _get_kv_from_cache(kv_cache)
                num_layers = len(keys)
                kv_seq_len = keys[0].shape[0]

                asst_len = kv_seq_len - prompt_len
                actual_stride = min(stride, asst_len)
                trim = 1  # boundary token

                if actual_stride > 0:
                    evicted_cache = DynamicCache()
                    for l in range(num_layers):
                        new_K = torch.cat(
                            [keys[l][:prompt_len],
                             keys[l][prompt_len + actual_stride:-trim]],
                            dim=0,
                        )
                        new_V = torch.cat(
                            [values[l][:prompt_len],
                             values[l][prompt_len + actual_stride:-trim]],
                            dim=0,
                        )

                        # NO DETACH — this is the key difference from window_forward.py
                        # Retained KV stays in the autograd graph, enabling G_distal.
                        # Permute [seq, heads, dim] -> [1, heads, seq, dim] for DynamicCache
                        new_K = new_K.permute(1, 0, 2).unsqueeze(0)
                        new_V = new_V.permute(1, 0, 2).unsqueeze(0)

                        evicted_cache.update(new_K, new_V, l)
                else:
                    evicted_cache = DynamicCache()
                    for l in range(num_layers):
                        # Still trim boundary token, but no stride eviction
                        new_K = keys[l][:-trim].permute(1, 0, 2).unsqueeze(0)
                        new_V = values[l][:-trim].permute(1, 0, 2).unsqueeze(0)
                        evicted_cache.update(new_K, new_V, l)

                new_kv_len = kv_seq_len - actual_stride - trim
                logger.debug(
                    "KV eviction seg %d: kv_len %d -> %d (dropped %d + %d boundary), "
                    "prompt=%d, retained_asst=%d",
                    seg_idx, kv_seq_len, new_kv_len,
                    actual_stride, trim, prompt_len,
                    asst_len - actual_stride - trim,
                )

                del keys, values, kv_cache
                past_key_values = evicted_cache

    finally:
        hook_handle.remove()
        model.config.use_cache = saved_use_cache

    torch.cuda.empty_cache()
    full_logits = torch.cat(all_logits_pieces, dim=1)
    del all_logits_pieces

    # FSDP dummy passes
    actual_passes = len(seg_input_ranges)
    target_passes = max_forward_passes or actual_passes
    if target_passes > actual_passes:
        full_logits = _pad_with_dummy_passes(
            model, input_ids, position_ids, full_logits,
            target_passes - actual_passes, device,
        )

    assert full_logits.shape[1] == input_ids.shape[1], (
        f"Segmented forward logits shape {full_logits.shape[1]} != input {input_ids.shape[1]}. "
        f"segments={len(seg_input_ranges)}, ranges={seg_input_ranges}"
    )

    return {"logits": full_logits}


def _pad_with_dummy_passes(
    model: torch.nn.Module,
    input_ids: Tensor,
    position_ids: Tensor,
    logits: Tensor,
    num_dummy: int,
    device: torch.device,
) -> Tensor:
    """Run dummy forward passes for FSDP rank synchronization.

    Contributes to autograd graph (so FSDP backward hooks fire) but
    gradient values are multiplied by 0.
    """
    dummy_sum = torch.tensor(0.0, device=device)
    for _ in range(num_dummy):
        d_out = model(
            input_ids=input_ids[:, :1],
            position_ids=position_ids[:, :1],
        )
        d_logits = d_out["logits"] if isinstance(d_out, dict) else d_out.logits
        if isinstance(d_logits, dict):
            d_logits = d_logits["logits"]
        # float().mean() prevents bf16 overflow -> Inf. Inf * 0 = NaN (IEEE 754).
        dummy_sum = dummy_sum + d_logits.float().mean()
    return logits + (dummy_sum * 0).to(logits.dtype)
```

### Memory Considerations

Without `.detach()`, PyTorch retains intermediate activations from all segments for backprop.
For N segments, memory grows linearly with N. With window=4096, stride=512, and max 16k
tokens, the max segment count is:
- (16384 - 4096) / 512 + 1 = ~25 segments

Each segment processes ~4096 tokens. The retained KV per segment is (4096-512) = 3584 tokens
worth of K and V tensors across all layers. For Qwen3-4B (36 layers, 32 heads, 128 dim):
- Per segment retained: 3584 * 36 * 2 * 32 * 128 * 2 bytes (bf16) = ~1.06 GB
- 25 segments: ~26 GB for retained KV alone

This fits on A100-80GB but is tight. If memory becomes an issue, gradient checkpointing
within segments (already supported by HuggingFace) can reduce the cost.

---

## 3.2: Environment Wrapper

**File: `src/kv_eviction/env.py`**

Wraps the compaction-enabled vLLM server's response into prime-rl's expected format.

```python
"""RL environment wrapper for compaction-enabled vLLM.

Converts the standard vLLM /v1/chat/completions response (which now includes
compaction_metadata) into the format expected by prime-rl's training pipeline.

The key output field is `segment_boundaries`: a list of cumulative completion
token counts extracted from compaction events. The training forward pass
(segmented_forward.py) uses these to know where to split segments and evict KV.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompactionRollout:
    """Training sample from a compaction-enabled inference rollout.

    Contains everything needed by the segmented forward pass.
    """
    # Token data
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_logprobs: list[float]

    # Compaction metadata (from vLLM response)
    segment_boundaries: list[int]  # cumulative completion token counts
    stride: int                    # tokens evicted per compaction

    # Reward
    reward: float

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def total_len(self) -> int:
        return len(self.prompt_ids) + len(self.completion_ids)

    @property
    def num_segments(self) -> int:
        return len(self.segment_boundaries)


def extract_compaction_rollout(
    response: dict,
    prompt_ids: list[int],
    reward: float,
    stride: int,
) -> CompactionRollout:
    """Convert a vLLM response with compaction metadata into a CompactionRollout.

    Args:
        response: The vLLM API response dict. Expected to contain:
            - "choices": standard completions
            - "compaction_metadata": {"events": [...]} if compaction occurred
        prompt_ids: The original prompt token IDs.
        reward: The reward from the RL environment.
        stride: The compaction stride (tokens per eviction).

    Returns:
        CompactionRollout with segment_boundaries extracted from events.
    """
    # Extract completion token IDs and logprobs from response
    # The exact extraction depends on the vLLM response format.
    # For /v1/chat/completions with logprobs enabled:
    choice = response["choices"][0]
    completion_ids = choice.get("token_ids", [])
    completion_logprobs = choice.get("logprobs", [])

    # Extract segment boundaries from compaction metadata
    segment_boundaries = []
    compaction_meta = response.get("compaction_metadata")
    if compaction_meta and compaction_meta.get("events"):
        for event in compaction_meta["events"]:
            segment_boundaries.append(event["completion_tokens_at_compaction"])

    # If no compaction happened, the entire completion is one segment
    if not segment_boundaries:
        segment_boundaries = [len(completion_ids)]

    # Ensure the last boundary covers the full completion
    if segment_boundaries[-1] < len(completion_ids):
        segment_boundaries.append(len(completion_ids))

    return CompactionRollout(
        prompt_ids=prompt_ids,
        completion_ids=completion_ids,
        completion_logprobs=completion_logprobs,
        segment_boundaries=segment_boundaries,
        stride=stride,
        reward=reward,
    )


def build_position_ids(
    prompt_len: int,
    segment_boundaries: list[int],
    stride: int,
) -> list[int]:
    """Build position IDs that match inference-side compaction.

    After each compaction event, position continues from where it was
    (not from the physical KV length). This means position = physical_pos + offset
    where offset accumulates evicted tokens.

    Args:
        prompt_len: Number of prompt tokens.
        segment_boundaries: Cumulative completion token counts at each compaction.
        stride: Tokens evicted per compaction.

    Returns:
        List of position IDs for the full sequence [prompt + completion].
    """
    total_len = prompt_len + (segment_boundaries[-1] if segment_boundaries else 0)
    positions = list(range(total_len))
    # Positions are simple 0..N-1 because the input_ids include ALL tokens
    # (including evicted ones). The segmented forward reconstructs the eviction
    # via KV manipulation, not by skipping positions in input_ids.
    return positions
```

### Integration with prime-rl's Data Pipeline

Prime-rl collects rollout results via the `env_worker.py` pipeline. The `extract_result()`
function at `$SCRATCH/kv-eviction/prime-rl/src/prime_rl/orchestrator/env_worker.py:50`
extracts trajectory data from `vf.State` including a per-step `extras` dict.

The compaction metadata should be stored in `extras["segment_boundaries"]` and
`extras["compaction_stride"]` so it flows through the existing pipeline without
modifications to prime-rl itself.

---

## 3.3: Training Hooks

**File: `src/kv_eviction/train_hooks.py`**

Injects the segmented forward pass into prime-rl's training loop.

```python
"""Training hooks: inject segmented forward into prime-rl's training loop.

The hook dispatches to segmented_forward() when the rollout has segment_boundaries
(indicating compaction occurred during inference), otherwise falls through to
the standard prime-rl forward pass.

Integration approach: monkey-patch or wrap the forward function used by prime-rl's
DPPO trainer. The exact hook point depends on prime-rl's architecture.
"""

import logging
from typing import Any

import torch
from torch import Tensor

from kv_eviction.segmented_forward import segmented_forward

logger = logging.getLogger(__name__)


def maybe_segmented_forward(
    model: torch.nn.Module,
    input_ids: Tensor,           # [1, seq_len]
    position_ids: Tensor,        # [1, seq_len]
    temperature: Tensor,         # [1, seq_len]
    extras: dict[str, Any],      # from rollout trajectory step
    max_forward_passes: int | None = None,
) -> dict[str, Tensor]:
    """Dispatch to segmented or standard forward based on rollout metadata.

    If extras contains segment_boundaries with >1 segment, use segmented_forward.
    Otherwise, use standard model forward.

    Args:
        model: HuggingFace model.
        input_ids: [1, seq_len] full sequence.
        position_ids: [1, seq_len] position IDs.
        temperature: [1, seq_len] per-token temperatures.
        extras: Dict from rollout trajectory, may contain:
            - "segment_boundaries": list[int] (cumulative completion token counts)
            - "compaction_stride": int (tokens evicted per compaction)
            - "prompt_len": int
        max_forward_passes: For FSDP synchronization.

    Returns:
        {"logits": [1, seq_len, vocab]} temperature-scaled logits.
    """
    segment_boundaries = extras.get("segment_boundaries")
    stride = extras.get("compaction_stride", 0)
    prompt_len = extras.get("prompt_len", 0)

    # Use segmented forward if compaction happened (>1 segment)
    if segment_boundaries and len(segment_boundaries) > 1 and stride > 0:
        logger.debug(
            "Using segmented forward: %d segments, stride=%d, prompt=%d",
            len(segment_boundaries), stride, prompt_len,
        )
        return segmented_forward(
            model=model,
            input_ids=input_ids,
            position_ids=position_ids,
            segment_boundaries=segment_boundaries,
            prompt_len=prompt_len,
            stride=stride,
            temperature=temperature,
            max_forward_passes=max_forward_passes,
        )

    # Standard forward (no compaction, or single segment)
    logger.debug("Using standard forward (no compaction)")
    out = model(input_ids=input_ids, position_ids=position_ids)
    raw_logits = out["logits"] if isinstance(out, dict) else out.logits
    logits = raw_logits["logits"] if isinstance(raw_logits, dict) else raw_logits
    scaled = logits / temperature.unsqueeze(-1).to(logits.dtype)
    return {"logits": scaled}


def compute_max_forward_passes(batch_extras: list[dict]) -> int:
    """Compute max segment count across a batch for FSDP synchronization.

    All ranks must execute the same number of forward passes. This function
    finds the maximum across the batch so ranks with fewer segments can pad
    with dummy passes.

    Args:
        batch_extras: List of extras dicts from the batch.

    Returns:
        Maximum number of forward passes needed by any sample in the batch.
    """
    max_passes = 1
    for extras in batch_extras:
        boundaries = extras.get("segment_boundaries", [])
        if boundaries:
            max_passes = max(max_passes, len(boundaries))
    return max_passes
```

### Hook Integration Point

Prime-rl's DPPO trainer calls a forward function to compute logits for the policy loss.
The exact integration depends on prime-rl's trainer architecture. The recommended approach:

1. **Find the forward call site** in prime-rl's DPPO trainer (likely in
   `src/prime_rl/trainer/rl/` or `src/prime_rl/trainer/model.py`).

2. **Wrap it** so that when `extras["segment_boundaries"]` is present, the segmented
   forward is used instead of the standard forward.

3. **Do NOT modify prime-rl source**. Use a training config flag (e.g., `use_segmented_forward = true`)
   and a wrapper module in `kv_eviction/train_hooks.py` that patches the forward function
   at runtime.

The exact patching mechanism depends on prime-rl's plugin/hook system. If prime-rl supports
custom forward functions via config, use that. Otherwise, monkey-patch the forward method
of the trainer class.

### Key requirement: `impl = "hf"`

The segmented forward requires `use_cache=True` and `past_key_values` support, which means
the model must use the HuggingFace implementation (`impl = "hf"` in prime-rl config), NOT
the vLLM or custom implementation. This is set in the experiment TOML config:

```toml
[model]
impl = "hf"
```

This is the same requirement as mkv-rl's M4 training config.

---

## 3.4: Position ID Construction for Training

The segmented forward pass uses `position_ids` that match the inference-side RoPE positions.
During inference, after compaction:
- Physical KV length decreases by `stride` tokens
- `position_offset` increases by `stride` tokens
- Next token's RoPE position = (new physical length) + position_offset = same as before compaction

For training, the full input_ids sequence contains ALL tokens (including those that were
evicted during inference). The position_ids are simply `[0, 1, 2, ..., seq_len-1]` because
the segmented forward reconstructs the eviction behavior in the KV domain, not the token domain.

The position_ids for each segment's forward pass are sliced from this full range. Between
segments, the KV eviction drops the first `stride` assistant entries, which effectively shifts
the physical KV positions — but since we pass the correct `position_ids` slice for each
segment, and the model uses these positions for RoPE (not the KV cache indices), the RoPE
values match inference exactly.

---

## Testing Plan

### test_segmented_forward.py

```python
"""Tests for segmented forward pass (no detach)."""
import pytest
import torch


class TestSegmentedForward:

    def test_single_segment_passthrough(self):
        """With one segment boundary, output matches standard forward."""
        # segment_boundaries = [completion_len] means no compaction
        # Result should be identical to model(input_ids, position_ids)
        pass

    def test_logit_shape_matches_input(self):
        """Output logits shape [1, seq_len, vocab] matches input_ids shape."""
        pass

    def test_boundary_token_overlap(self):
        """Segment k>0 starts from boundary token of segment k-1.
        The logit at the boundary position should be computed with
        post-eviction KV context."""
        pass

    def test_kv_drop_count(self):
        """After eviction, retained KV length = previous - stride (+ prompt)."""
        pass

    def test_no_detach_gradient_flow(self):
        """Gradients flow from last segment back to first segment's parameters.

        Key test: compute loss on last segment's logits, backprop, check that
        first segment's input embeddings have non-zero gradients.
        Compare with detached version where gradients should be zero.
        """
        pass

    def test_g_distal_nonzero(self):
        """G_distal = grad(no_detach) - grad(detach) is nonzero.

        Run the same forward pass twice: once with detach, once without.
        The gradient difference should be nonzero, proving cross-chunk
        gradient flow.
        """
        pass

    def test_fsdp_dummy_passes(self):
        """With max_forward_passes > actual segments, dummy passes are added.
        The logit shape should still be correct."""
        pass

    def test_short_segment_clamp(self):
        """When stride > actual assistant tokens, stride is clamped."""
        pass

    def test_empty_completion(self):
        """prompt_len >= seq_len is handled gracefully."""
        pass
```

### Key Verification: Step-0 KL

The most important test (run as part of Phase 4 but designed here):

```python
def test_step0_kl_approximately_zero():
    """Training logits match inference logits at step 0 (before any updates).

    Setup:
    1. Start vLLM with compaction (window=4096, stride=512)
    2. Generate a rollout, collect logprobs
    3. Run segmented_forward on the same input with the same model weights
    4. Compare logprobs: KL should be ~0.0

    This works because both use flash_attn kernel:
    - Inference: vLLM's PagedAttention (flash_attn based)
    - Training: HF model with use_cache=True (flash_attn via sdpa)

    If KL > 0.01, something is wrong (kernel mismatch, position error, etc.)
    """
    pass
```

---

## Completion Criteria

- [ ] `segmented_forward()` produces correct logits matching standard forward for single-segment case
- [ ] Multi-segment case: logit shape matches input, boundary tokens handled correctly
- [ ] **No `.detach()` anywhere** in `segmented_forward.py`
- [ ] G_distal test: gradient difference (no-detach minus detach) is nonzero
- [ ] FSDP dummy passes work correctly
- [ ] `env.py` extracts `segment_boundaries` from vLLM compaction response
- [ ] `train_hooks.py` dispatches correctly based on `extras["segment_boundaries"]`
- [ ] `test_segmented_forward.py` all pass
- [ ] Model uses `impl = "hf"` in config (assertion or warning if not)
- [ ] Memory usage is feasible on A100-80GB for max segment count (~25 segments)

---

## Phase 3.4 Live KL Test Results (2026-04-10)

Final end-to-end numerical validation of segmented_forward vs vLLM's
compaction inference on Qwen3-4B. See `experiments/phase3_kl_test/`
for the test scripts.

**Setup:**
- Allocation: 2 A100-80GB nodes, 4 GPUs each (interactive, 4h walltime)
- Inference node: 1 GPU, vLLM 0.19.1-dev (our fork) with
  `enforce_eager=True`, window=4096, stride=512, block_size=16
- Trainer node: 4 GPUs, DP=4 via torchrun, HF `AutoModelForCausalLM`
  with `attn_implementation="flash_attention_2"`, bf16
- Data: 10 rg-mix problems, `max_tokens=16384`, `ignore_eos=True`,
  `temperature=1.0`, `seed=43`
- Total: 160,821 completion tokens per condition, 240 compaction events
  across 10 samples (exactly 24 per sample, spaced every 512 gen tokens
  after the first event at gen[3893])

**Two conditions compared:**
1. **Baseline** — vLLM without compaction + trainer standard `model()`
   forward. Measures the kernel-numerics noise floor between vLLM's
   eager-mode path and HF's flash_attention_2 path on the same tokens.
2. **Compaction** — vLLM with compaction enabled + trainer
   `segmented_forward()` replaying the 24 KV evictions per sample with
   no detach between segments.

**Metric:** per-token `|trainer_logprob - inference_logprob|`, averaged
over the full completion. This is the same metric mkv-rl used for its
M3 KL check; it approximates per-token KL when the logprobs are
sampled-token logprobs.

**Results:**

| Metric                   | Baseline  | Compaction | Ratio |
|--------------------------|-----------|------------|-------|
| mean abs log-ratio       | 0.00842   | 0.01035    | 1.23x |
| max  abs log-ratio       | 4.8975    | 2.8539     | 0.58x |
| mean signed log-ratio    | -0.00059  | -0.00062   | ~1x   |
| samples                  | 10        | 10         | —     |
| total completion tokens  | 160,821   | 160,821    | —     |
| per-sample elapsed       | 1.4s      | 1.9s       | 1.36x |

**Interpretation.** The compaction mean abs log-ratio is **1.23x the
baseline kernel-noise floor** — indistinguishable from pure bf16
rounding noise. The absolute number (~0.010 per token) is ~200x lower
than the ~2.0 number mkv-rl M3 (SDPA + detached) was hitting on the
same kind of measurement. The mean signed log-ratio is nearly identical
between conditions (-0.00059 vs -0.00062), meaning compaction introduces
no systematic bias — whatever drift exists is due to the base model's
kernel numerics, not to anything segmented_forward does.

**Per-sample details** (from kl_results.json per_sample_rank_local):
- All 10 samples had 24 compaction events firing at the expected
  cumulative gen positions (3893, 4405, 4917, ..., 15669), confirming
  the first-eviction math holds in practice with the tightened
  needs_compaction guard.
- Every sample's per-token mean is within 0.006..0.014 for both
  conditions — tight distribution, no outliers.
- The max-abs outliers are driven by 1-2 tokens per trajectory (a common
  pattern under temperature=1 where the argmax disagreement can be
  dramatic at a single sampling step). Baseline saw a 4.90 outlier,
  compaction saw a 2.85 outlier; both are statistical noise from
  sampling, not systematic.

**DP=4 validation.** All 4 ranks processed their shards in lockstep.
The cross-rank `all_reduce(MAX)` for `max_forwards` and the
RSA-hardened defensive branch-agreement `all_reduce` both worked
without deadlock. The modality-partitioning fix from RSA Round 1 and
the CP rejection fix from RSA Round 2 are both exercised and intact.

**Runtime.** Model load 24s per rank (checkpoint shards cached); total
trainer wall time under 60s for 20 samples. Inference phase took
~5 min per condition (eager mode). Full test from salloc to
kl_results.json: ~15 min of the 4h allocation.

**Conclusion.** Phase 3 is numerically validated for forward-only
correctness. The trainer's segmented_forward replay of vLLM's
compaction is within kernel noise of the no-compaction baseline,
meaning the KV drop arithmetic, position handling, temperature scaling,
and cross-rank sync are all correct. Gradient flow through retained KV
and full FSDP2-sharded training runs are the remaining validation
steps before production training (see "Pre-production checklist"
below).

---

## Pre-production Checklist

Phase 3.4 validated forward-pass numerics. Before kicking off full RL
training runs, the following smoke tests should pass. They are listed
in rough order of risk (highest first) and complexity (lowest first).

1. **Backward pass smoke test (single GPU).** The KL test used
   `torch.no_grad()` and never called `.backward()`. segmented_forward
   claims to preserve gradients through retained KV (the whole point of
   "no detach") but this was never exercised numerically. Run one
   compaction sample through segmented_forward, compute a dummy loss,
   call backward, verify: (a) no NaN gradients, (b) every parameter
   receives a gradient (no dead params), (c) gradient norm is in a
   reasonable range (not 0, not 1e6+). This catches any "gradients
   silently zero" bug that would make training a no-op.

2. **FSDP2-sharded segmented_forward smoke test.** The KL test used DDP
   (replicated weights). The real prime-rl trainer uses FSDP2 (sharded
   weights with per-layer all-gather/reshard). segmented_forward makes
   multiple model() calls per sample, each triggering its own FSDP2
   collectives. Test: load Qwen3-4B with FSDP2 sharding across 4 GPUs,
   run one compaction forward, compare resulting logits to the DDP
   version from Phase 3.4. If they disagree beyond kernel noise, the
   FSDP2 per-layer reshard is confusing segmented_forward's backbone
   hook or its past_key_values handling.

3. **End-to-end single training step.** Run the actual prime-rl RL
   trainer (not the standalone KL test) for exactly one step with
   compaction enabled. Use a minimal config: 1 rg-mix rollout, 1 micro
   batch, small LR, FSDP=4 dp, single node. Watch for: (a) the
   compaction samples hitting the segmented_forward dispatch, (b) the
   config validator firing if attn/impl/cp are wrong, (c) the
   optimizer actually stepping (non-zero update norm), (d) the weight
   broadcast to the inference server completing.

4. **Short stability run (5-10 steps).** Same config as #3, run 5-10
   steps. Loss should decrease (or at least not explode), entropy
   should stay finite, no NaN gradients. Confirms FSDP2 + compaction +
   gradient flow + weight sync are all cooperating for more than one
   iteration.

5. **Production-config smoke (1 node, 1-2 steps).** Use the actual
   production TOML (full seq_len, real batch sizes) to verify memory
   fits and per-step wall time is acceptable. Segmented_forward's
   O(window) memory footprint should keep us under peak GPU memory,
   but better to confirm before committing to a long run.

Of the 4 deferred items from RSA review (#37-#40), only #37 (cross-config
validator between trainer and orchestrator inference args) meaningfully
affects production robustness. The others are perf/hardening nits that
don't block the first real runs.

## Pre-production Smoke Test Results (2026-04-10)

### Smoke #1: single-GPU backward pass — PASS

Single A100 80GB, Qwen3-4B bf16 + flash_attention_2, full 16k-token
sample with all 24 compaction events. Activation checkpointing enabled
via segmented_forward's new `activation_checkpointing=True` flag (added
in this round — see Key finding #2 below for why HF's built-in
`gradient_checkpointing_enable` is unusable).

Results:
- Forward: 2.7s, 32GB peak
- Backward: 5.3s, 55.7GB peak
- All 398 params received non-zero gradients (no dead params from the
  retained-KV `torch.cat` chain)
- Zero NaN, zero Inf
- Global grad norm: 1.49 (healthy)
- Loss: 0.197 (reasonable teacher-forced policy loss)

Headroom on 80GB A100: ~25GB, which means real training with AdamW
state (~32GB for a 4B model) will still fit because FSDP2 shards both
params and optimizer state across ranks in production.

Script: `experiments/phase3_preprod/smoke1_backward.py`

### Smoke #2: FSDP2 segmented_forward — PASS (with caveats)

4-GPU A100 80GB node, torchrun --nproc_per_node=4, each rank loads
Qwen3-4B, `fully_shard` applied per decoder layer, each rank processes
the same 4-event sample (truncated to fit without outer AC under
FSDP2).

Results:
- Forward: 1.2s, backward: 1.3s
- Loss cross-rank spread: **0.00e+00** (bit-identical across all 4 ranks)
- 1592 params (398 × 4 ranks) all received gradients
- Zero NaN, zero Inf, zero missing grads
- FSDP2-reduced global grad norm: 3.09

No deadlock despite 5 per-segment all-gathers feeding into one
backward. The retained-KV `torch.cat` chain survives FSDP2's
reshard-after-forward cadence.

Script: `experiments/phase3_preprod/smoke2_fsdp2.py`

### Key findings

1. **`fully_shard` on the root model breaks RMSNorm.** Applying
   `fully_shard(model)` at the top level wraps `model.model.norm` and
   `model.lm_head` as DTensors, and HF's RMSNorm hits
   "aten.mul.Tensor got mixed Tensor and DTensor" during backward.
   Fix: shard only the decoder layers. Prime-rl's production path
   does this correctly by grouping `[lm_head, norm]` into a separate
   `fully_shard` unit with `reshard_after_forward=False` — see
   `prime_rl/trainer/model.py:400-412`. For the smoke test, sharding
   just the decoder layers is sufficient to exercise FSDP2 all-gather
   cadence on the large parameter groups.

2. **`segmented_forward(activation_checkpointing=True)` is
   incompatible with FSDP2 — important production guidance.**
   `torch.utils.checkpoint.checkpoint` with `use_reentrant=False`
   does NOT trigger FSDP2's pre-forward hooks on backward re-entry,
   causing "aten.mul.Tensor got mixed Tensor and DTensor" in RMSNorm
   when the checkpointed segment's forward is re-run during backward.
   The production path must use a completely different AC mechanism:
   prime-rl applies
   `torch.distributed.algorithms._checkpoint.checkpoint_wrapper`
   to each transformer block BEFORE `fully_shard`, so the FSDP2 hook
   ordering works correctly. In production:

   - Trainer config should call prime-rl's per-block AC as normal
     (`trainer.model.ac.freq = 1`)
   - `segmented_forward` must be called WITHOUT its own
     `activation_checkpointing=True`
   - The per-block AC handles memory inside each segment's forward
     (~1 block's activations retained), and the between-segment
     retained-KV chain happens at the outer torch graph level

   The new `activation_checkpointing=True` flag on segmented_forward
   is still useful for single-GPU debugging and offline tests (like
   smoke #1), just not under FSDP2.

3. **Computing grad norm on DTensor params needs `.to_local()`.**
   FSDP2 stores each parameter's grad as a DTensor whose arithmetic
   dispatch insists both operands be DTensors; naively accumulating
   into a plain local tensor trips an isinstance assert in
   `torch/distributed/tensor/_dispatch.py`. Unwrap via
   `g.to_local()` before computing the squared sum. Relevant for
   any custom grad-norm or diagnostic code under FSDP2.

### Item #37: RLConfig cross-config validator — DONE

New `@model_validator(mode="after")` on `RLConfig` in
`prime-rl/src/prime_rl/configs/rl.py` cross-checks that
`trainer.compaction.{window_size, stride, block_size}` exactly mirrors
`inference.vllm_extra.{compaction_window_size, compaction_stride,
block_size}` when KV cache compaction is in use. Bidirectional: either
side enabling compaction forces the other to match. The three sanity
tests (window mismatch, stride mismatch, block_size mismatch) plus the
three valid cases (mirrored, both-disabled, inference-only-enabled
rejected) all pass.

Sanity-check script:
`experiments/phase3_preprod/test_rlconfig_validator.py`

### Remaining smoke tests

Smokes #3-5 (end-to-end single step, short stability run,
production-config smoke) still pending. These require:

- A 2-node allocation (inference node + trainer node)
- `rg-mix-env` installed into the kv-eviction `.venv` (currently
  only in mkv-rl's venv — install from
  `/pscratch/sd/s/siddart2/mkv-rl/experiments/rg_mix/dist/rg_mix_env-0.1.4-py3-none-any.whl`
  inside the podman container via `uv pip install`)
- A minimal compaction RL TOML config adapted from
  `mkv-rl/experiments/rg_mix/rl.toml`, with matching
  `trainer.compaction` and `inference.vllm_extra.compaction_*` keys
  plus `trainer.model.impl = "hf"` and
  `trainer.model.attn = "flash_attention_2"` (enforced by the new
  config validator).

Best tackled in a fresh session with full context runway.

## Pre-production session round 2: per-segment backward + trainer dispatch

Second pre-prod session (same date) pivoted from M4 BPTT attempt to
M3 per-segment backward after finding that M4 interacts fatally with
prime-rl's per-block checkpoint_wrapper under `use_cache=True`. See
`experiments/phase3_preprod/probe_ac_cache_mutation.py` for the
minimal crash reproduction: `torch.utils.checkpoint` (non-reentrant)
re-runs each block's forward during backward, and HF's
`DynamicCache.update()` takes the "concatenate existing slot" branch
instead of "append new slot", producing 2x-length K/V tensors. Torch
catches this loudly via `CheckpointError: Recomputed values have
different metadata than during the forward pass`
(`[1, 1024, 8, 128]` → `[1, 2048, 8, 128]`).

Design pivot: per-segment forward + backward + detach, with an
optional `bptt_segments` knob for truncated BPTT windows:
  - `bptt_segments=1` (default, M3 semantics): detach between every
    segment, one backward per segment, O(1 segment) memory
  - `bptt_segments=K > 1`: gradients flow through retained KV
    within K consecutive segments before detach, O(K) memory
  - `bptt_segments=None`: full trajectory in one BPTT window,
    M4-equivalent semantics, O(all segments) memory

Validation (smoke #1b, single A100 80GB, full 16k / 24-event sample):
  - k=1:  40.9 GB, 5.5s, grad norm 1.22, PASS
  - k=4:  55.4 GB, 5.2s, grad norm 1.30, PASS
  - k=12: OOM at torch.cat in eviction (expected)
  - Loss identical across k (deterministic forward)
  - k=4 grad norm 6% higher than k=1, confirming intra-window
    BPTT actually flows gradients through retained KV

Trainer dispatch wired in `prime-rl/src/prime_rl/trainer/rl/train.py`:
the `use_segmented` branch now builds a `_segment_loss_fn` closure
that slices `labels`, `advantages`, `loss_mask`,
`inference_logprobs`, `teacher_logprobs` to match each segment's
owned logit range and delegates to prime-rl's existing
`compute_loss` (no duplication of the loss formula, just range
bookkeeping). Alignment detail: the last segment's final logit
predicts a nonexistent token at position `seq_len` and must be
dropped explicitly (matching what `shift_tensor_right` does in the
standard path). Three adversarial review rounds caught and fixed
several bugs before landing; see commit messages for details.

### Deferred items (recorded here so future instances can find them)

**D1. Multi-rank `bptt_segments != 1` under FSDP2.** The dummy-pass
padding inside `segmented_forward` pads forwards with paired
(forward + backward) dummies, which only keeps total backward
counts matched across ranks when every real segment also has
exactly one backward — i.e., `bptt_segments == 1`. With
`bptt_segments > 1` or `None`, the number of real backwards per
rank is `ceil(per_rank_segments / K)`, which varies across ranks
when per-rank segment counts differ. Reduce-scatter counts then
diverge and FSDP2 deadlocks.

Proper fix: compute `max_backwards` via `all_reduce(ReduceOp.MAX)`
across ranks (where `max_backwards = max(ceil(actual_i / K))`),
pass it to `segmented_forward` alongside `max_forwards`, and have
the dummy-pass logic issue a mix of:
  - `Y_local = max_backwards - real_bw_count` dummy (forward+backward)
    pairs
  - `X_local - Y_local` forward-only dummies (where
    `X_local = max_forwards - actual`)

The condition `X_local >= Y_local` is provably satisfied because
`f(x) = x - ceil(x/K)` is non-decreasing in x, so
`max_forwards - max_backwards >= actual - ceil(actual/K)` for every
rank. See the third adversarial review agent's analysis in
`.claude/projects/-pscratch-sd-s-siddart2-mkv-rl/` session
transcript (2026-04-10) for the full derivation.

Current state: blocked at training init via a runtime check in
`train.py` right after `parallel_dims = get_parallel_dims(...)`
that raises `ValueError` if `world_size > 1 and bptt_segments != 1`.
Single-GPU runs can freely set any `bptt_segments` value.

**D2. Entropy for compaction samples.** `segmented_forward` returns
a scalar loss, not per-token logits, so `out["entropy"]` is not
populated for segmented micro-batches. The debug-log f-string at
`train.py` is guarded on `not use_segmented and
len(tensors['entropy']) > 0` and simply omits the entropy field
for compaction samples. `compute_stats` downstream handles
missing per-micro-step entries by emitting NaN, so step-level
entropy averages for compaction-heavy steps will come out as NaN.

To fix: extend the `_segment_loss_fn` closure (or a parallel
callback) to compute `compute_entropy(seg_logits_effective)`,
apply `seg_loss_mask`, and accumulate into a per-micro-batch list.
After `segmented_forward` returns, `torch.cat` the accumulated
entropy values and append to `tensors['entropy']` just like the
standard path does at line 673.

**D3. Token-weighted metric aggregation for compaction.** The
per-segment `compute_loss` call wraps each `_safe_mean`-style
metric in `torch.stack([scalar_0d]) → shape [1]`; the closure
appends these to an `accumulated_loss_tensors` dict and the
post-loop `torch.cat(v)` gives a shape `[n_segments]` 1-D tensor
that's appended to `tensors[metric_name]`. Downstream
`compute_stats` averages these unweighted across segments, which
gives a mean-of-means rather than a token-weighted mean. For a
sample whose segments are token-count-skewed (e.g. 4000/4000/1000
trainable tokens per segment) the logged metric value drifts
from what the standard path would report.

The LOGGED LOSS is still correct because the loss decomposition is
exact: `sum_over_segments(compute_loss(seg).loss) ==
compute_loss(full_seq).loss` when every call uses the same global
`loss_scale`. Only the derived metrics (`mismatch_kl`, `is_masked`,
etc.) drift.

To fix: have the closure collect raw per-token `(values, mask)`
pairs per metric instead of per-segment `_safe_mean` outputs,
concatenate across segments, and recompute `_safe_mean` once at
the end before appending to `tensors`. Or: change
`compute_loss`'s per-sequence metrics contract to expose
`(numerator_sum, denominator_count)` so the closure can
accumulate weighted.

**D4. Smokes #3-5 still pending.** As noted in the "Remaining smoke
tests" section above — requires a 2-node allocation, rg-mix-env
installed into the kv-eviction venv, and a minimal compaction RL
TOML config. With the trainer dispatch now landed, these should
run cleanly end-to-end modulo integration surprises.

## Pre-production session round 3: end-to-end plumbing and smoke #3/#4 (2026-04-10 → 2026-04-11)

This session began as a routine run of smokes #3 (single-step E2E) and #4
(5-step stability) and turned into a deep investigation after smoke #4's
reported `Mismatch KL ≈ 0.04` looked ~50× larger than the offline kernel
floor. The investigation uncovered and fixed a chain of latent bugs in
the compaction events plumbing that had been silently breaking every
prior smoke run — D2's partial success and smoke #3's clean exit were
both misleading because the dispatch to `segmented_forward` **never
actually fired**. The real scientific validation (trainer-inference
logprob agreement at the kernel floor under active compaction) finally
happened on step 0 of smoke #4 v5, reported below.

### Initial observation: anomalous Mismatch KL in smoke #4

Smoke #4 (compaction enabled, 5 steps, batch 64) reported the
following trainer-side metrics across steps 0-4:

| Step | Loss | Entropy | Mismatch KL | Grad Norm |
|---:|---:|---:|---:|---:|
| 0 | 0.0018 | 0.3005 | **0.0382** | 0.1727 |
| 1 | 0.0049 | 0.2827 | **0.0390** | 0.2220 |
| 2 | 0.0051 | 0.2527 | **0.0387** | 0.3689 |
| 3 | 0.0031 | 0.2948 | **0.0466** | 0.2369 |
| 4 | 0.0035 | 0.2873 | **0.0351** | 0.1694 |

Smoke #4b (same config, compaction **disabled** via absent
`[trainer.compaction]`, running vLLM without compaction args) was run
as a control and reported Mismatch KL = **0.0007** dead-flat across all
5 steps — pure bf16 roundoff noise. The **55× gap** between compaction
and no-compaction was the triggering anomaly.

### Investigation steps (all independent, all useful as negative results)

1. **Three parallel RSA agents** audited design, numerics, and data
   pipeline. Agent A (design) correctly identified the boundary re-feed
   + `trim = 1` as a candidate mechanism but couldn't prove it
   dominated. Agent B misread `mkv-rl` instead of `kv-eviction/prime-rl`
   and reached a false "temperature-order" conclusion that was easily
   disproven at `train.py:790-791` where scaling happens before
   `selective_log_softmax`. Agent C's "shift_tensor_right mismatch"
   claim was disproven by reading `batch.py:37` directly — both paths
   are target-position indexed and consistent.

2. **Historical comparison.** `phase3_kl_test` (Apr 10 14:13) had
   reported compaction/baseline = 1.23× for `mean_abs_log_ratio`. The
   file mtime of `segmented_forward.py` (Apr 10 16:37) was AFTER the
   test, confirming that phase3_kl_test validated the PRE-per-segment-
   backward version. But more importantly, phase3_kl_test's trainer
   used `model.eval() + torch.no_grad()` on a bare HF full replica,
   not FSDP2 + training mode + AC — completely different runtime.

3. **Offline reproduction of segmented_forward both modes.** Wrote
   `experiments/phase3_kl_test/compare_segforward_modes.py` that runs
   `segmented_forward` in both legacy (`loss_fn=None`) and per-segment
   backward modes on the same rollout data and compares per-position
   owned logits. Result:

   ```
   raw_logit_diff:              mean=0.000000  max=0.000000
   logsoftmax_diff_max_vocab:   mean=0.000000  max=0.000000
   sampled_token_logprob_diff:  mean=0.000000  max=0.000000
   legacy vs inference:         mean=0.009329  max=0.558362
   psb    vs inference:         mean=0.009329  max=0.558362
   psb/legacy ratio: 1.00x

   mismatch_kl mean (token-weighted, 10 samples, full 24 events each):
     aggregate = 0.000691
   ```

   Legacy and per-segment-backward modes produce **bit-identical**
   logits on the full 16k-token sample. The offline kernel floor for
   the complete 10-sample compaction dataset is **mismatch_kl = 0.0007**,
   72× smaller than smoke #4's reported 0.0395.

   **Conclusion:** neither the boundary re-feed nor the per-segment
   backward mode is responsible for the gap. The gap comes from
   somewhere between the trainer's offline reproduction (which uses
   segmented_forward) and smoke #4's live trainer (which uses something
   else).

4. **Direct inspection of smoke #4 rollouts.bin**, loaded from
   `outputs/smoke4_rl_stability_run/run_default/rollouts/step_*/rollouts.bin`
   via `msgspec.msgpack.Decoder(TrainingBatch)`:

   ```
   step 0: 64 examples, 0 w/ compaction_events, 64 rollouts > 4096
   step 1: 64 examples, 0 w/ compaction_events, 48 rollouts > 4096
   step 2: 64 examples, 0 w/ compaction_events, 54 rollouts > 4096
   step 3: 64 examples, 0 w/ compaction_events, 54 rollouts > 4096
   step 4: 64 examples, 0 w/ compaction_events, 63 rollouts > 4096
   ```

   **ZERO samples had `compaction_events` populated**, despite
   inference mean completion length 5000-7500 tokens and vLLM logs
   showing `[COMPACT] enabled window=4096 stride=512` on all 4 DP
   ranks. vLLM was evicting KV during generation, but the events
   never reached the trainer.

   The trainer then took the **standard full-context forward path**
   for every sample, reforwarding against inference logprobs that
   were produced under post-eviction attention. That full-vs-evicted
   mismatch is exactly what the reported 0.04 Mismatch KL measures —
   **not a bug in segmented_forward; segmented_forward was never called**.

### Root cause chain: the compaction events plumbing was silently broken

Tracing the path vLLM → trainer revealed FIVE consecutive latent bugs,
any one of which was sufficient to drop every event:

**Bug 1 — verifiers client adapter drops the field.**
`verifiers/clients/openai_chat_completions_client.py:from_native_response`
constructs a new `verifiers.types.Response(id, created, model, usage,
message)` from a hardcoded 5-field list. The openai-python SDK's
`ChatCompletion` model has `extra='allow'` and does preserve vLLM's
`compaction_events` field in its `model_extra` dict, but verifiers
never reads it and never forwards it into its own Response.

**Bug 2 — verifiers base env initializes extras empty.**
`verifiers/envs/multiturn_env.py:141` creates each `TrajectoryStep`
with `extras={}` and never populates it from the response. Prime-rl's
`orchestrator/trajectories.py:297 _compaction_events_from_step(step)`
reads `step.get("extras").get("compaction_events")` expecting
something, finds an empty dict, returns None. Every downstream
`TrainingSample.compaction_events` ends up as None.

**Bug 3 — the existing `CompactionEnvMixin` is dead code.**
`kv_eviction/env.py` shipped a `CompactionEnvMixin` intended to be
subclassed by env authors (e.g., `class RGMixEnv(CompactionEnvMixin,
SingleTurnEnv)`). But `RGMixEnv` doesn't subclass it, and the mixin
itself was un-importable because it did `from verifiers.types import
ModelResponse` for a type that had been renamed to `Response` — the
first instantiation would `ImportError`. Nobody had ever exercised
this code path in a real run.

**Bug 4 — env server runs in a spawn subprocess.**
Even after writing module-level monkey-patches in `kv_eviction/env.py`
and triggering them via `kv_eviction/__init__.py: from . import env`,
the patches didn't apply in the process where rollouts actually run.
The env server is spawned via
`mp.get_context("spawn").Process(target=ZMQEnvServer.run_server)`.
Spawn starts a fresh Python interpreter; the subprocess does not
inherit any pre-imported modules from the parent. The subprocess
imports `verifiers.serve.ZMQEnvServer` directly — it never imports
`kv_eviction`.

**Bug 5 — compaction_events as msgspec structs fail verifiers'
state_columns JSON-serializability check.** First attempt at the fix
stored `list[CompactionEventWire]` in `step.extras["compaction_events"]`
and every rollout failed with:

```
ValueError("state_columns value for 'trajectory' is not JSON-serializable:
  list. Only JSON-serializable types are allowed.")
```

verifiers routes trajectory step extras through a JSON-serializability
check that rejects msgspec structs as-is. The fix must store plain
dicts.

### Fixes applied

All of the following are now committed. Each was separately tested
either locally (unit-level) or via a smoke re-run.

| Commit | Repo | Purpose |
|---|---|---|
| `e216849` | kv-eviction | `env.py` module-level monkey-patches on `OpenAIChatCompletionsClient.from_native_response` and `MultiTurnEnv.add_model_response`. Both are idempotent (sentinel attr `__kv_eviction_patched__`). The `from_native_response` patch uses setattr on the verifiers Response (possible because `CustomBaseModel` has `extra="allow"`) after reading the raw events from the openai SDK's `model_extra`. The `add_model_response` patch calls `attach_compaction_events_from_response` on the trajectory step the base class just appended. Fixes Bugs 1, 2, 3. |
| `c7c8563b1` | prime-rl | `orchestrator/envs.py`: adds `import kv_eviction` at the top of the module AND adds `_env_server_subprocess_entrypoint(...)` that imports `kv_eviction` before calling `ZMQEnvServer.run_server`. Swaps `mp.Process(target=...)` to the wrapper. The spawn subprocess now applies the patches in both places: at unpickle time (resolving the target function imports `prime_rl.orchestrator.envs` which imports kv_eviction) and as belt-and-suspenders in the wrapper body. Fixes Bug 4. |
| `f230a58` | kv-eviction | `env.py`: `_extract_compaction_event_dicts` now returns `list[dict]` with the three int fields, and `attach_compaction_events_from_response` stores those dicts directly. Downstream `orchestrator/trajectories.py:_compaction_events_from_step` already had a branch that reconstructs `CompactionEventWire` from dict form, so no reader changes were needed. Fixes Bug 5. |
| `4005a7d14` | prime-rl | `train.py`: inlined the per-segment entropy math in `_segment_loss_fn` instead of calling `compute_entropy` (which has `@torch.compile(dynamic=True)`). Dynamic-shape handling did not generalize across the sizes segmented_forward produces (300-1000 token segments per sample × 24 segments) and triggered a silent Inductor recompile stall that hung all 4 DP ranks for 25+ minutes at step 0. The inlined `logsumexp - sum(softmax * logits)` under `torch.no_grad` runs in eager mode without touching Inductor. Same arithmetic, no compile dependency. |
| `4c851fc92` | prime-rl | `configs/trainer.py`: added a config-level validator that rejects `(trainer.compaction.window_size > 0) AND (trainer.model.ac != None)` at load time with a multi-paragraph explanation of the DynamicCache double-update mechanism. Fixes the "AC + compaction crashes on first compaction sample" footgun permanently. |
| `7b99cfc` | kv-eviction | Bumps prime-rl + updates `experiments/smoke4_rl_stability_run/rl_smoke4.toml` to remove the `[trainer.model.ac]` section (now required by the validator above). |

### Verification: smoke #4 v5 Step 0

With all fixes in place, smoke #4 ran end-to-end through step 0 and
reported:

| Metric | v1 (pre-fix) | v5 (post-fix) | Offline floor |
|---|---:|---:|---:|
| **Mismatch KL** | 0.0382 | **0.0009** | 0.0007 |
| Loss | 0.0018 | 0.0009 | — |
| Entropy | 0.3005 | 0.3083 | — |
| Grad Norm | 0.1727 | 0.1013 | — |
| Peak mem | 59.2 GiB | **43.7 GiB** | — |
| Time | 207s | 216s | — |

**Mismatch KL dropped from 0.0382 to 0.0009** — right at the offline
kernel floor (0.0007–0.0009) measured earlier via
`compare_segforward_modes.py`. Direct inspection of the post-fix
step 0 rollouts.bin confirms `58/64 samples have compaction_events
populated, 467 events total`, matching 1:1 the 58 rollouts that
exceeded the 4096 window. The `use_segmented` dispatch is firing for
the first time in this codebase's history. The trainer's per-segment
logprob reconstruction matches the inference engine's on-the-fly
PagedAttention eviction to bf16 noise. Peak memory is **lower** than
the AC-enabled baseline because segmented_forward's per-segment
backward bounds activation memory to O(1 segment).

**The original scientific question is answered.** Compaction training
produces trainer-inference logprob agreement at the kernel floor.
`segmented_forward` is numerically correct. The legacy and
per-segment-backward modes are bit-identical. The boundary re-feed
is not the bug (neither mode has a real problem with it; both agree
with phase3_kl_test's measurements). The inflated Mismatch KL that
started this investigation was entirely a data-plumbing artifact.

### Still open: D5 — rank-level FSDP collective divergence in mixed batches

**Smoke #4 v5 step 0 succeeded. Step 1 crashed with an NCCL
watchdog timeout on `_ALLGATHER_BASE` at seq 27,393** (default 600s
timeout). Last-enqueued vs last-completed work diverged by 1, meaning
one rank enqueued the 27,393rd all-gather but at least one other rank
never reached it. Classic rank-level divergence where different DP
ranks issue different numbers of FSDP collectives per step.

**Mechanism.** A step's rollouts are partitioned by
`trainer/batch.py:prepare_batch` into compaction micro-batches
(single-sample, not bin-packed) and non-compaction micro-batches
(packed). Each compaction sample triggers ~20+ forward passes in
`segmented_forward` (one per segment) plus its own reduce-scatter /
all-gather chain; each non-compaction sample triggers one forward.
So a compaction sample has many more FSDP collectives than a
non-compaction sample.

When a step's rollouts have uneven (compaction : non-compaction)
distribution across DP ranks — for example, rank 0 gets 14 compaction
+ 1 non-compaction, rank 1 gets 13 + 2 — the total FSDP collective
count per rank DIFFERS even if `pad_micro_batch` equalizes the number
of micro-batches. `pad_micro_batch` pads to the max count across
ranks with empty/dummy micro-batches, but a dummy non-compaction
micro-batch costs 1 collective while a dummy compaction micro-batch
would cost ~20+ collectives, and there's no bookkeeping to reconcile
the two.

Step 0 of v5 worked because all 4 ranks happened to have similar
compaction ratios (64/64 samples exceeded the window in step 0's
particular draw). Step 1 had a different draw, one rank had a
skewed ratio, FSDP collective counts diverged, NCCL timed out.

**This is a separate latent bug from everything else debugged in
this session.** It was undetectable before the events-plumbing fix
because the dispatch never took the compaction path — every sample
went through the standard forward, all ranks had symmetric collective
counts, no divergence. v5 is the first run in which the dispatch
actually fires, so this is the first run to exercise the mixed-batch
path.

### D5 detailed problem description (for the next session)

**Observed failure mode.** With smoke #4 v5 config (4096 window,
512 stride, batch_size=64, rollouts_per_example=8, DP=4, rg-mix-env
with natural eos), step 0 completes cleanly but step 1 hangs on
`WorkNCCL(SeqNum=27393, OpType=_ALLGATHER_BASE, NumelIn=25232704,
NumelOut=100930816)` for 600 seconds before the NCCL watchdog aborts
the process. The log line:

```
[Rank 0] First PG on this rank to signal dumping.
last enqueued work: 27393, last completed work: 27392.
This is most likely caused by incorrect usages of collectives, e.g.,
wrong sizes used across ranks, the order of collectives is not same
for all ranks or the scheduled collective, for some reason, didn't
run.
```

confirms the divergence: rank 0 enqueued the 27,393rd collective
but at least one other rank never issued its matching call.

**Required fix — sketch.** `prepare_batch` in `trainer/batch.py`
already partitions compaction samples (kept single) from
non-compaction samples (packed). The missing piece is
**FSDP-collective-count equalization** across DP ranks when the
compaction:non-compaction ratio differs. Two candidate approaches:

1. **Per-rank collective-aware padding.** Count the expected number
   of FSDP forward-pass collectives per rank under the current batch
   partition (for each compaction micro-batch, this is the max over
   ranks of `compute_num_segments(...)` which is already all-reduce-
   synced inside `segmented_forward`; for each non-compaction
   micro-batch, it's 1). Take the max across ranks. Pad lower ranks
   with dummy compaction OR non-compaction micro-batches until their
   collective count matches. Need to also match the BACKWARD count
   (per-segment backward runs one backward per segment, so both
   forward and backward counts must be padded consistently).

2. **Uniform sample partitioning.** Enforce at the trainer batch-
   build step that every DP rank gets the SAME number of compaction
   samples and the SAME number of non-compaction samples. Requires
   modifying the orchestrator→trainer send path to group samples by
   modality before per-rank distribution. More invasive but avoids
   dummy passes entirely.

Option 1 is closer to the existing `pad_micro_batch` mechanism and
leverages the already-working per-micro-batch segment-count sync.
Option 2 is cleaner architecturally but touches more code.

**Verification for the fix.** Re-run smoke #4 v5 (same TOML) with
the fix applied. Expected: all 5 steps complete with Mismatch KL
staying in the 0.001 range (kernel floor), no NCCL timeouts, reward
trajectory similar to the 4b baseline modulo noise.

**Safety net.** A config-level sanity check in `TrainerConfig`
could also warn if `use_token_client=True` with compaction and
`rollouts_per_example * batch_size > dp_size` — any config where
ranks are likely to see uneven compaction ratios. Weaker than the
real fix, but a decent smoke gate.

### Smoke status

- Smoke #1 (single-GPU backward): **PASS** (2026-04-10)
- Smoke #1b (per-segment backward probe): **PASS** (2026-04-10)
- Smoke #2 (FSDP2 segmented forward): **PASS** (2026-04-10)
- Smoke #3 (single end-to-end RL step): **PASS** cosmetically
  (2026-04-10), but the compaction dispatch never fired because
  of the events-plumbing bugs. Real validation came from
  smoke #4 v5 step 0.
- Smoke #4 (5-step stability): **STEP 0 PASS**, steps 1-4 blocked
  on D5. Partial pass. (2026-04-11)
- Smoke #4b (5-step full-context baseline): **PASS** (2026-04-11)
- Smoke #5 (production-config smoke): **BLOCKED** on D5.

### Updated deferred items summary

| ID | Title | Status |
|---|---|---|
| D1 | Multi-rank `bptt_segments != 1` under FSDP2 | Blocked via runtime ValueError at training init. Recompute max_backwards via all_reduce MAX + mixed-dummy pad. |
| D2 | Entropy for compaction samples | **RESOLVED** this session via inlined per-segment entropy math in `_segment_loss_fn` (commit `4005a7d14`). |
| D3 | Token-weighted metric aggregation | Still open. Empirically confirmed the aggregation artifact is small (offline token-weighted vs per-segment-unweighted differ by < 10% on 10-sample test). Low priority until metrics drive decisions. |
| D4 | Smokes #3-5 | **PARTIALLY DONE**: #3 passes, #4 step 0 passes, #4 steps 1-4 blocked on D5, #5 blocked on D5. |
| **D5** | **Rank-level FSDP collective divergence in mixed compaction batches** | **NEW — blocking prod**. Detailed above. This is the next work item. |

### Useful artifacts from this session

- `experiments/phase3_kl_test/compare_segforward_modes.py` — offline
  tool that runs segmented_forward in both legacy and per-segment
  backward modes on saved rollouts and reports both token-weighted
  and per-segment-unweighted mismatch_kl. Useful for future kernel-
  floor measurements and regression testing. See
  `run_compare_modes.sh` + `_compare_modes_inner.sh` for the srun
  wrapper that puts execution on a real 80GB compute node (a bare
  `salloc ... bash script.sh` runs on the login node's 40GB GPU,
  which surfaced as an initial OOM).

- Saved rollouts.bin from v1 and v3 in
  `outputs/smoke4_rl_stability_run.*` directories can be used to
  empirically measure (compaction_events count, distribution across
  ranks, segment count variance) if needed for D5 triage.

- `probe_ac_cache_mutation.py` (pre-existing) — minimal repro of the
  `torch.utils.checkpoint` + `DynamicCache.update()` double-append
  bug that forced the AC config validator.

### D5 resolution (2026-04-11 session 4)

**Root cause turned out NOT to be rank-level FSDP collective divergence.**
The round-3 plan section's hypothesis was wrong. Actual cause was a
CUDA OOM on step 1 triggered by the **compaction → text modality
transition** inside a single training step. Smoke #4's inference config
(`max_completion_tokens=8192`, `window_size=4096`) produced a mix of
event-bearing rollouts (long completions, trigger compaction) and
event-less rollouts (short completions, never compact). The pre-fix
trainer dispatch routed event-bearing samples through
`segmented_forward` and event-less samples through the **standard
packed-text forward**. In a single step each rank processed its
compaction micro-batches first, then its packed text micro-batches.
The per-segment allocation pattern from compaction left the allocator
holding ~11 GiB of cached blocks sized for per-layer/per-segment
tensors that didn't fit the contiguous shapes the packed text forward
needed, pushing the total above 80 GiB on step 1 once the optimizer
state had allocated.

Evidence from the investigation:

1. **Static audit of `segmented_forward`** — traced every tensor lifetime
   for `bptt_segments=1` + per-segment-backward. All retained-KV storage
   is reset between iterations; no cross-call leak.

2. **Memory probe** (`experiments/phase3_kl_test/memory_probe_segforward.py`)
   — 10 sequential `segmented_forward` calls, both frozen and unfrozen
   variants, real gradient flow. Entry memory is identical on every call
   (8.122 GB), peak is bounded per call (38.3 GB unfrozen), no growth.
   Empty-boundaries single-segment path works end-to-end. Mixed-events
   probe (alternating 24-event and 0-event samples) also stable. The
   fix's structural correctness is validated at the probe level.

3. **FSDP2 4-rank probe** (`experiments/phase3_kl_test/memory_probe_fsdp.py`)
   — wrapped Qwen3-4B in per-block FSDP2 matching prime-rl's setup,
   ran 5 compaction calls + 1 text call at `seq_len=16384`, **reproduced
   the exact smoke #4 OOM**: 5 × compaction peak ~34 GB, then the text
   forward OOMs at 78.9 GB. At `seq_len=8192` where text fits, the
   compaction-then-text sequence left `reserved` at 79.7 GB vs text-only
   at 68.5 GB — an **11.2 GB fragmentation delta** from the modality
   transition.

4. **Smoke #4 v5 crash log** (previous session) shows rank 1 OOMing first
   at 06:40 with 79.04 GiB in use, then ranks 0/2/3 either OOMing
   themselves or hitting NCCL watchdog timeouts while waiting on the
   first collective rank 1 never issued. **The NCCL timeout was a
   downstream symptom of OOM, not a rank-count divergence.**

### The fix: unify dispatch via `segmented_forward`

**Design**. Route every sample in a compaction training run
(`compaction.window_size > 0`) through `segmented_forward`, even samples
whose inference rollout never triggered a compaction event. Samples
with events produce a multi-segment forward; event-less samples produce
a single-segment forward covering `[0, seq_len)`, which is numerically
identical to calling `model(input_ids, position_ids)` under the same
flash_attention_2 kernel. The packer un-packs every non-multimodal
sample into its own micro-batch in a compaction run. Non-compaction
runs (`window_size == 0`) are unaffected — they keep the existing
packed text path.

**Changes across 5 files**:

- `prime-rl/src/prime_rl/trainer/batch.py` — threaded
  `compaction_enabled` through `prepare_sample`, `packed_samples_into_micro_bs`,
  `pad_micro_batch`, and `prepare_batch`. When True, every non-multimodal
  sample becomes its own un-packed micro-batch, gets `prompt_len` set,
  uses the continuous position-id padding, and lands in the compaction
  modality bucket.
- `prime-rl/src/prime_rl/trainer/rl/packer.py` — `BasePacker`,
  `SinglePacker`, `MultiPacker`, and `setup_packer` accept the flag and
  pass it to `prepare_batch` calls.
- `prime-rl/src/prime_rl/trainer/rl/data.py` — `DataLoader.__init__`
  takes `compaction_enabled` and forwards to `setup_packer`.
- `prime-rl/src/prime_rl/trainer/rl/train.py` — `DataLoader` is
  constructed with `compaction_enabled=config.compaction.window_size > 0`.
  The per-micro-batch dispatch now uses
  `use_segmented = config.compaction.window_size > 0` instead of
  checking individual-sample events. The `prompt_len is not None` assert
  error message updated.
- `src/kv_eviction/segmented_forward.py` — relaxed the
  `len(segment_boundaries) > 0` assertion, guarded
  `segment_boundaries[-1]` access, and added a `seq_len > 0`-guarded
  fallback that appends `(0, seq_len)` as a single segment when
  boundaries is empty. `compute_num_segments` already returned 1 for
  the empty case, so the `all_reduce MAX` dummy-pass sync still keeps
  FSDP collectives balanced across ranks with different event counts.

### Smoke #4 v6 validation (2026-04-11 09:XX)

All 5 steps passed. Compare to v5 (pre-fix, same config):

| Step | Time (s) | Loss | Entropy | Mismatch KL | Grad Norm | Peak Mem |
|---|---|---|---|---|---|---|
| 0 | 65.67 | 0.0009 | 0.3086 | 0.0010 | 0.1013 | 38.4 |
| 1 | 38.45 | 0.0019 | 0.2891 | 0.0009 | 0.1199 | **45.9** |
| 2 | 55.48 | 0.0021 | 0.2891 | 0.0009 | 0.0925 | 45.9 |
| 3 | 508.45 | 0.0020 | 0.2930 | 0.0009 | 0.1075 | 45.9 |
| 4 | 179.78 | 0.0026 | 0.2676 | 0.0009 | 0.1171 | 45.9 |

v5 for comparison: step 0 passed at 43.7 GiB / 216s / MKL 0.0009, step
1 OOMed at 79.18 GiB. v6 step 0 is 3.3× faster (one-sample micro-batches
pack better through the allocator than 16k text bins), peak memory is
5 GiB lower on step 0 and stays flat at 45.9 GiB across steps 1-4 (no
leak), Mismatch KL holds at the kernel floor (0.0009-0.0010) confirming
the single-segment fallback is numerically identical to the standard
forward. Orchestrator-side reward trends upward (0.266 → 0.472 → 0.352
on the final step, within run-to-run noise). Step 3's 508 s time is an
orchestrator-side inference stall, not a training issue (memory stayed
flat, training metrics remained normal).

**34 GiB of headroom** to the 80 GB limit on step 1+, vs v5 which was
over the limit by several GiB. Production config (smoke #5) has
significant safety margin.

### Updated smoke status

- Smoke #4 (5-step stability): **PASS** (2026-04-11 v6 with D5 fix)
- Smoke #5 (production-config smoke): **UNBLOCKED**, next work item

### Updated deferred items

| ID | Title | Status |
|---|---|---|
| D1 | Multi-rank `bptt_segments != 1` under FSDP2 | Blocked via runtime ValueError. |
| D2 | Entropy for compaction samples | **RESOLVED** (round 3). |
| D3 | Token-weighted metric aggregation | Still open. Low priority. |
| D4 | Smokes #3-5 | **#3/#4 DONE**, #5 unblocked. |
| D5 | Rank-level FSDP collective divergence | **RESOLVED** (this session): not actually the problem — root cause was CUDA OOM from compaction→text allocator fragmentation. Fix: unified dispatch via `segmented_forward` for all samples in a compaction run. Validated by smoke #4 v6. |
