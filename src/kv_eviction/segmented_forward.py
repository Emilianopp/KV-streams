"""Segmented forward pass with KV prefix drop between segments.

This is the training-side mirror of vLLM's scheduler-integrated KV cache
compaction. For each compaction boundary the inference engine reported,
we run one forward-pass segment, then drop the oldest
stride_blocks*block_size generation KV entries (and trim the boundary
token), exactly as the inference engine did.

Two operating modes:

1. **Per-segment backward mode (`loss_fn` provided).** The production
   path. Between segments we `.detach()` the retained KV, turning each
   segment into an independent backward-pass unit. The caller supplies
   a `loss_fn` closure that computes the policy loss for just that
   segment's owned token range; after each segment's forward we call
   that closure, run `loss.backward()`, then let autograd tear down
   the graph and free the segment's activations before moving on to
   the next segment. Memory is bounded to O(1 segment) regardless of
   how many compaction events fire. This is the M3 semantic (no cross-
   chunk BPTT, no G_distal term) and it's compatible with FSDP2 and
   prime-rl's per-block activation checkpointing.

2. **Legacy / single-GPU mode (`loss_fn=None`).** Returns the full
   concatenated logits tensor for one final backward at the caller.
   Retained KV is NOT detached between segments, so in principle
   gradients flow backward through the cat chain (BPTT). In practice
   this mode only works under two configurations:
     - Single GPU with `activation_checkpointing=True` (wraps each
       segment's model call in torch.utils.checkpoint externally).
       Useful for offline KL debugging; see smoke #1.
     - Single GPU with no AC at all, on sequences small enough that
       all segment activations fit in memory.
   Under FSDP2 + prime-rl's per-block checkpoint_wrapper, this mode
   deadlocks/errors because torch.utils.checkpoint's non-reentrant
   mode does NOT trigger FSDP2's pre-forward hooks on backward
   re-entry (see plans/phase3_training_integration.md §"Key findings").
   The trainer in production MUST use the per-segment backward mode.

Why the segmentation gives zero train-inference KL mismatch:
- Each segment's forward uses flash_attention_2 (the same kernel vLLM
  uses for decode), so per-segment logits are numerically identical up
  to float precision.
- The KV drop between segments replicates vLLM's eviction exactly
  (same offsets, same trim, same retained identities) so attention
  over retained KV produces the same output as inference.
- Boundary token overlap: the last token of segment k is re-run in
  segment k+1 so its logit (which predicts the first token of the new
  segment) is computed under the post-eviction context, matching what
  inference did when it sampled that token.

Drop boundary is `prompt_aligned_len`, not `prompt_len`, matching the
block-level eviction semantics in vLLM's CompactingKVCacheManager.
Callers pass `prompt_aligned_len` directly; we don't derive it here
because that would require `block_size` as another parameter.

Future work: M4 (BPTT through retained KV, with G_distal gradient term)
requires a cache-snapshot mechanism so torch.utils.checkpoint's
backward re-runs can recover the original pre-mutation cache state.
Not attempted here; we ship M3 first.
"""

import logging
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field as dataclass_field
from typing import Callable

import torch
import torch._functorch.config  # noqa: F401  load submodule so the assignment below resolves
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor
from transformers import DynamicCache

# FSDP2's AOT autograd donates intermediate buffers between forward and
# backward, which forbids retain_graph=True. The legacy segmented_forward
# loss_fn=None mode runs one backward across all retained-KV segments
# and needs retain_graph; keep donated buffers off so that path stays
# working. The per-call persistent-cache path doesn't need this (the
# cache is detached between calls) but the setting is harmless there.
torch._functorch.config.donated_buffer = False  # type: ignore[attr-defined]

logger = logging.getLogger(__name__)
_KVE_FLEX_TIMING = os.environ.get("KVE_FLEX_TIMING") == "1"


def _timing_now(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


# Type alias for the per-segment loss callback. Given the segment's
# temperature-scaled and boundary-trimmed logits, along with the global
# sequence positions they occupy in the concatenated full-sequence logit
# tensor, returns a scalar loss that will be backward'd immediately
# (gradients accumulate into .grad before we move to the next segment).
#
# full_logit_start and full_logit_end are the inclusive / exclusive range
# in the FULL-SEQUENCE coordinate system. The segment's logit at local
# index i corresponds to full-sequence position (full_logit_start + i),
# which predicts input_ids[full_logit_start + i + 1]. For RL losses the
# callback typically slices advantages / old_logprobs / loss_mask at
# positions [full_logit_start + 1, full_logit_end + 1).
SegmentLossFn = Callable[[Tensor, int, int], Tensor]
SegmentDistillationFn = Callable[[Tensor, Tensor, int, int], Tensor]
SegmentDistillationPrepareFn = Callable[[Tensor, int, int], None]
SegmentDistillationTeacherFn = Callable[[Tensor, int, int], None]
SegmentDistillationLossFn = Callable[[Tensor, int, int], Tensor]
SegmentSelectedLogprobInputsFn = Callable[[int, int, int], tuple[Tensor, Tensor, int]]
SegmentSelectedLogprobLossFn = Callable[[Tensor, Tensor, int, int, int], Tensor]


@dataclass(frozen=True)
class _FlexMaskWriterTimeline:
    """Packed writer timeline for the masked FlexAttention path."""

    input_ids: list[int]
    position_ids: list[int]
    death_indices: list[int]
    # Tuples are (writer_start, writer_end, full_logit_start, full_logit_end).
    loss_ranges: list[tuple[int, int, int, int]]
    # Managed-context hidden recall: extra visibility windows beyond the
    # base [row, death_indices[row]) interval. Triples are
    # (writer_row, birth, death): row is ALSO visible to queries q with
    # birth <= q < death. Populated from kind-1/2 restore events; empty
    # for eviction-only timelines (mask construction is unchanged then).
    extra_intervals: list[tuple[int, int, int]] = dataclass_field(
        default_factory=list
    )
    # Diagnostic metadata: archived span id -> writer rows recorded at
    # splice time. Consumed by offline replay harnesses; not used by the
    # mask construction itself.
    span_rows: dict = dataclass_field(default_factory=dict)


def _normalize_bptt_segments(bptt_segments: int | None) -> int | None:
    """Map the config-facing -1 sentinel to full-chain BPTT."""
    if bptt_segments == -1:
        return None
    return bptt_segments


def compute_num_segments(
    seq_len: int,
    prompt_len: int,
    segment_boundaries: list[int],
) -> int:
    """Compute how many forward-pass segments `segmented_forward` will run.

    This mirrors the range-computation logic inside segmented_forward. The
    trainer uses this to all-reduce-MAX the segment count across DP ranks
    before calling segmented_forward, so every rank runs the same number of
    forwards and NCCL all-gathers stay synchronized.

    Returns:
        Number of segments (= number of forward passes). Always >= 1 for
        non-empty boundaries.
    """
    if not segment_boundaries:
        return 1
    n = 0
    prev_boundary = 0
    for i, boundary in enumerate(segment_boundaries):
        if i == 0:
            seg_start = 0
        else:
            seg_start = prompt_len + prev_boundary - 1
        seg_end = min(prompt_len + boundary, seq_len)
        if seg_start < seg_end:
            n += 1
        prev_boundary = boundary
    # Tail segment covers tokens past the last boundary.
    last_covered = prompt_len + segment_boundaries[-1]
    if last_covered < seq_len:
        n += 1
    return max(n, 1)


def _get_kv_from_cache(cache: DynamicCache) -> tuple[list[Tensor], list[Tensor]]:
    """Extract per-layer [seq, heads, dim] K/V tensors from a DynamicCache.

    Handles both the new transformers API (>=4.49: cache.layers[l].keys) and
    the old API (cache.key_cache[l]). Internal shape in the cache is
    [1, heads, seq, dim]; we permute to [seq, heads, dim] for easier
    splicing along the sequence dimension.
    """
    num_layers = len(cache)
    if hasattr(cache, "layers") and len(cache.layers) > 0:
        keys = [
            cache.layers[l].keys[0].permute(1, 0, 2).contiguous()
            for l in range(num_layers)
        ]
        values = [
            cache.layers[l].values[0].permute(1, 0, 2).contiguous()
            for l in range(num_layers)
        ]
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


def _run_segment_checkpointed(
    model: torch.nn.Module,
    seg_ids: Tensor,
    seg_positions: Tensor,
    prev_keys: list[Tensor] | None,
    prev_values: list[Tensor] | None,
    *,
    capture_cache: bool = True,
) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    """Run a single segment's forward under torch.utils.checkpoint.

    Returns (logits, new_keys, new_values) where each key/value tensor is
    shape [seq, heads, dim] (the _get_kv_from_cache format). When
    ``capture_cache`` is false, the model runs with ``use_cache=False`` and
    the key/value lists are empty.

    The checkpoint re-materializes segment activations during backward, so
    the multi-segment retained-KV chain can be trained at long sequences
    without holding every segment's activations in memory simultaneously.

    Why this bypasses HF's gradient-checkpointing / use_cache conflict:
    HF's built-in gradient_checkpointing_enable() forcibly disables
    use_cache at the top-level model forward, which strips past_key_values
    from the output and breaks segmented_forward entirely. We instead
    leave HF GC disabled and apply checkpointing OUTSIDE the model call,
    preserving use_cache=True and past_key_values semantics. The wrapped
    function below is re-run during backward; each re-run constructs a
    fresh DynamicCache from the same input tensors, so there's no stale
    cache mutation hazard.
    """
    if not capture_cache:
        assert prev_keys is None and prev_values is None, (
            "cache-free checkpointed forward cannot receive retained KV"
        )

    # Flatten prev KV to positional tensor args so checkpoint tracks them
    # as inputs. checkpoint requires tensor args (or None/lists of tensors
    # via the non_tensor_args machinery) — flat positional list is cleanest.
    if prev_keys is not None and prev_values is not None:
        kv_in_flat: list[Tensor] = []
        for k, v in zip(prev_keys, prev_values):
            # _get_kv_from_cache returned [seq, heads, dim]; the model wants
            # [1, heads, seq, dim] inside its DynamicCache.
            kv_in_flat.append(k.permute(1, 0, 2).unsqueeze(0).contiguous())
            kv_in_flat.append(v.permute(1, 0, 2).unsqueeze(0).contiguous())
    else:
        kv_in_flat = []

    def _seg_fn(_ids: Tensor, _pos: Tensor, *_kv_flat: Tensor):
        if capture_cache and len(_kv_flat) > 0:
            nl = len(_kv_flat) // 2
            _cache: DynamicCache | None = DynamicCache()
            for l in range(nl):
                _cache.update(_kv_flat[2 * l], _kv_flat[2 * l + 1], l)
        else:
            _cache = None

        model_kwargs = {
            "input_ids": _ids,
            "position_ids": _pos,
            "use_cache": capture_cache,
        }
        if capture_cache:
            model_kwargs["past_key_values"] = _cache
        _out = model(**model_kwargs)
        _raw_logits = _out["logits"] if isinstance(_out, dict) else _out.logits
        if isinstance(_raw_logits, dict):
            _raw_logits = _raw_logits["logits"]

        if not capture_cache:
            return (_raw_logits,)

        # Pull the updated cache out of the output; for HF modeling this is
        # returned on the output object; some custom paths may leave it on
        # the input cache we passed in.
        _out_cache = None
        if hasattr(_out, "past_key_values") and _out.past_key_values is not None:
            _out_cache = _out.past_key_values
        elif isinstance(_out, dict) and _out.get("past_key_values") is not None:
            _out_cache = _out["past_key_values"]
        else:
            _out_cache = _cache

        if _out_cache is None:
            raise RuntimeError(
                "Checkpointed segment forward did not produce past_key_values. "
                'Ensure model uses impl="hf".'
            )

        _new_keys, _new_values = _get_kv_from_cache(_out_cache)
        # Flatten for checkpoint tensor-output requirement
        _flat_out: list[Tensor] = [_raw_logits]
        for _k, _v in zip(_new_keys, _new_values):
            _flat_out.append(_k)
            _flat_out.append(_v)
        return tuple(_flat_out)

    ckpt_out = torch.utils.checkpoint.checkpoint(
        _seg_fn,
        seg_ids,
        seg_positions,
        *kv_in_flat,
        use_reentrant=False,
    )
    if isinstance(ckpt_out, Tensor):
        return ckpt_out, [], []
    seg_logits = ckpt_out[0]
    kv_out = ckpt_out[1:]
    num_layers = len(kv_out) // 2
    new_keys = [kv_out[2 * l] for l in range(num_layers)]
    new_values = [kv_out[2 * l + 1] for l in range(num_layers)]
    return seg_logits, new_keys, new_values


def segmented_forward(
    model: torch.nn.Module,
    input_ids: Tensor,  # [1, seq_len]
    position_ids: Tensor,  # [1, seq_len]
    segment_boundaries: list[int],  # cumulative completion token counts
    prompt_len: int,  # raw prompt length
    prompt_aligned_len: int,  # block-aligned eviction boundary
    stride: int,  # tokens to drop per eviction (= stride_blocks * block_size)
    temperature: Tensor,  # [1, seq_len] per-token temperatures
    max_forward_passes: int | None = None,  # FSDP synchronization padding
    activation_checkpointing: bool = False,  # per-segment re-materialization (legacy)
    loss_fn: SegmentLossFn | None = None,  # per-segment backward callback
    bptt_segments: int | None = 1,  # truncation depth for TBPTT
) -> dict[str, Tensor]:
    """Run segmented forward with KV prefix drop between segments, no detach.

    Args:
        model: HF CausalLM model with attn_implementation="flash_attention_2"
            and impl="hf" (custom llama asserts past_key_values is None and is
            not usable here).
        input_ids: Full sequence [1, seq_len] = prompt + completion tokens.
        position_ids: Per-token absolute positions [1, seq_len]. Legacy vLLM
            replay uses arange. SGLang prefill_trim replay passes its explicit
            piecewise positions so the eventless one-segment path preserves
            the inference RoPE frame.
        segment_boundaries: Cumulative completion-token counts at each
            compaction event, in ascending order. An empty list runs one
            segment over the full input with ``use_cache=False`` and no cache
            hook, capture, or KV drop.
        prompt_len: Number of prompt tokens (the offset into input_ids where
            completion tokens begin).
        prompt_aligned_len: Block-aligned eviction boundary. Without
            protected prefix: ceil(prompt_len / block_size) * block_size.
            With protected prefix: ceil(min(protected_prefix, prompt_len)
            / block_size) * block_size — can be LESS than prompt_len
            because old conversation turns between the protected prefix
            and the prompt end are evictable. The trainer-side drop must
            match the inference-side boundary exactly or train/inference
            KL explodes.
        stride: Number of tokens evicted per compaction event. In the vLLM
            config this is compaction_stride, a multiple of block_size.
        temperature: Per-token temperatures [1, seq_len] used at generation
            time. Logits are scaled by 1/temperature per-token before being
            concatenated into the final output.
        max_forward_passes: Target forward-pass count for FSDP rank
            synchronization. If greater than the actual number of segments,
            the difference is padded with dummy forward passes that
            contribute to the autograd graph but carry zero gradient weight.
            Must be >= len(segment_boundaries) + 1 when set.
        activation_checkpointing: (Legacy, single-GPU only.) When True,
            each segment's model() call is wrapped in
            torch.utils.checkpoint.checkpoint(use_reentrant=False). This
            re-materializes segment activations during backward instead
            of holding them in memory across the full retained-KV chain.
            Must NOT be combined with HF's
            model.gradient_checkpointing_enable() (disables use_cache).
            Must NOT be used under FSDP2 — non-reentrant checkpoint's
            backward re-entry does not trigger FSDP2's pre-forward
            hooks, producing Tensor/DTensor mismatches in RMSNorm.
            Incompatible with loss_fn (only one checkpointing strategy
            at a time).
        loss_fn: (Production path for compaction training.) Per-segment
            loss callback. When provided, the function runs in
            per-segment backward mode: after every `bptt_segments`
            segments' forwards we call backward on the accumulated
            segment losses, detach the retained KV, and move to the
            next BPTT window. Activations from completed segments are
            freed by autograd as each window's backward finishes,
            bounding memory to O(bptt_segments segments). Compatible
            with FSDP2 + prime-rl's per-block AC. Cannot be combined
            with activation_checkpointing=True.
        bptt_segments: Truncation depth (in segments) for truncated
            backpropagation through time. Only meaningful when loss_fn
            is provided. Each BPTT "window" is this many consecutive
            segments; gradients flow through retained KV within a
            window, and the KV is .detach()'d between windows.
                1 (default) = M3 semantics — no cross-chunk BPTT, one
                    backward per segment, O(1 segment) memory.
                K > 1 = TBPTT with depth K segments, O(K segments)
                    memory, gradients span K segments via retained KV.
                None = "full trajectory" — forward every segment in a
                    single window, one backward at the end. Equivalent
                    to M4 gradient semantics (G_distal term intact)
                    but requires O(all segments) memory.
            The final window may be shorter than bptt_segments if the
            total segment count isn't a multiple of bptt_segments.

    Returns:
        - Legacy mode (loss_fn=None):
            {"logits": [1, seq_len, vocab]} — temperature-scaled full
            sequence logits. Caller is responsible for the final
            backward. Shape matches input_ids exactly.
        - Per-segment backward mode (loss_fn provided):
            {"loss": <scalar tensor, detached>, "n_segments": int}
            The scalar is the sum of all per-segment losses (already
            backward'd; gradients have already accumulated into
            model.parameters().grad). Loss is detached so the caller
            can log/reduce it without re-entering the graph.
    """
    bptt_segments = _normalize_bptt_segments(bptt_segments)
    assert input_ids.shape[0] == 1, "segmented_forward requires batch_size=1"
    # Empty segment_boundaries is a legal no-event sample: the trainer's
    # unified compaction dispatch (D5 fix) routes every sample through
    # segmented_forward in a compaction run, including short rollouts
    # whose inference never triggered a compaction event. Those samples
    # produce seg_ranges = [(0, seq_len)] below — a single segment
    # covering the whole [prompt + completion] sequence, numerically
    # equivalent to calling model(input_ids, position_ids) directly
    # under flash_attention_2. See plans/phase3_training_integration.md.
    assert prompt_aligned_len > 0, (
        f"prompt_aligned_len must be positive, got {prompt_aligned_len}"
    )
    assert prompt_aligned_len <= input_ids.shape[1], (
        f"prompt_aligned_len ({prompt_aligned_len}) exceeds seq_len "
        f"({input_ids.shape[1]})"
    )
    assert stride > 0, f"stride must be positive, got {stride}"
    assert not (activation_checkpointing and loss_fn is not None), (
        "activation_checkpointing=True and loss_fn are mutually exclusive: "
        "the per-segment backward mode (loss_fn) already frees segment "
        "activations by running .backward() between segments, so the "
        "outer torch.utils.checkpoint wrapping would be redundant and "
        "would also conflict with the per-segment backward cadence."
    )
    per_segment_backward = loss_fn is not None
    if per_segment_backward:
        if bptt_segments is not None:
            assert bptt_segments >= 1, (
                f"bptt_segments must be None or >= 1, got {bptt_segments}"
            )

    device = input_ids.device
    seq_len = input_ids.shape[1]
    cache_replay = bool(segment_boundaries)

    # Build the list of segment ranges in input_ids space.
    #
    # There is one segment per compaction event plus a final tail segment
    # covering everything after the last boundary through seq_len.
    #
    # Segment 0: input_ids[0 : prompt_len + boundaries[0])
    # Segment k (k > 0): input_ids[prompt_len + boundaries[k-1] - 1 :
    #                              prompt_len + boundaries[k])
    #     The -1 is the boundary-token overlap: the last token of segment
    #     k-1 is re-fed so its logit (which predicts the first token of
    #     segment k) is recomputed under the post-eviction context.
    # Final tail: input_ids[prompt_len + boundaries[-1] - 1 : seq_len)
    #
    # Tail is only a distinct segment if boundaries[-1] < completion_len.
    # Otherwise the last "segment" in the segment_boundaries list already
    # covers through seq_len.
    seg_ranges: list[tuple[int, int]] = []
    prev_boundary = 0
    for i, boundary in enumerate(segment_boundaries):
        if i == 0:
            seg_start = 0
        else:
            seg_start = prompt_len + prev_boundary - 1
        seg_end = min(prompt_len + boundary, seq_len)
        if seg_start < seg_end:
            seg_ranges.append((seg_start, seg_end))
        prev_boundary = boundary

    # Add a tail segment to cover generation past the last compaction event.
    #
    # Convention: segment_boundaries has one entry per compaction event (from
    # CompactionEventWire.num_output_tokens_at_compaction). After the last
    # compaction, the model typically continues generating until EOS or
    # max_tokens. Those tail tokens need their own segment that starts from
    # the boundary-overlap token and runs to seq_len.
    #
    # (Note: this convention differs from mkv-rl's window_forward which
    # treats boundaries as segment END POINTS including the final one. In
    # mkv-rl the last boundary equals completion_len and no tail is needed.
    # We mirror CompactionEvent semantics directly instead.)
    if segment_boundaries:
        last_covered = prompt_len + segment_boundaries[-1]
        if last_covered < seq_len:
            seg_ranges.append((last_covered - 1, seq_len))
    elif seq_len > 0:
        # No events: single segment covering the whole sample. We still
        # go through segmented_forward's machinery (per-segment backward
        # mode, FSDP dummy-pass padding, etc.) so every rank in a
        # compaction run issues a uniform collective sequence regardless
        # of whether each rank's sample had any events. Numerically this
        # is a plain text forward on the full sequence under
        # flash_attention_2 — same kernel as the non-compaction path.
        seg_ranges.append((0, seq_len))

    assert seg_ranges, (
        f"No segment ranges produced: seq_len={seq_len}, prompt_len={prompt_len}, "
        f"segment_boundaries={segment_boundaries}"
    )

    # Capture past_key_values from the backbone via a hook (non-checkpointed
    # path only). FSDP2 + custom output-linear wrappers may drop
    # past_key_values from the top-level model output, so a hook on the
    # backbone (model.model) is more reliable.
    #
    # The checkpointed path does NOT use this hook — its inner function
    # reconstructs and returns the cache explicitly to keep everything as
    # tensor-typed outputs that checkpoint can track.
    captured_kv: dict[str, DynamicCache | None] = {}

    def _capture_kv_hook(_module, _input, output):
        if hasattr(output, "past_key_values"):
            captured_kv["past_key_values"] = output.past_key_values
        elif isinstance(output, dict):
            captured_kv["past_key_values"] = output.get("past_key_values")

    backbone = model.model if hasattr(model, "model") else model
    hook_handle = (
        backbone.register_forward_hook(_capture_kv_hook)
        if cache_replay and not activation_checkpointing
        else None
    )

    # Legacy (loss_fn=None) path collects per-segment logit slices for
    # one final backward at the caller. Per-segment-backward path uses
    # accumulated_loss (a Python float) and window_loss (a live graph
    # tensor that accumulates per-segment losses within the current
    # BPTT window), leaving all_logits_pieces empty.
    all_logits_pieces: list[Tensor] = []
    accumulated_loss: float = 0.0
    window_loss: Tensor | None = None
    # Count of real segments processed within the current BPTT window.
    # Resets to 0 at every window boundary.
    segments_in_window: int = 0
    # Non-checkpointed path: DynamicCache passed through segments directly.
    # In per-segment backward mode we build a FRESH DynamicCache from
    # prev_keys/prev_values at the top of each loop iteration (so the
    # cache object itself never crosses segments). Whether those
    # prev_keys still carry autograd edges depends on whether we're
    # inside a BPTT window (edges intact) or just crossed a boundary
    # (detached).
    past_key_values: DynamicCache | None = None
    # Checkpointed path: plain tensor lists (shape [seq, heads, dim]) that
    # serve as positional tensor inputs to the next segment's checkpoint.
    # Per-segment backward path: same shape; detached only at BPTT
    # window boundaries (see the eviction block below).
    prev_keys: list[Tensor] | None = None
    prev_values: list[Tensor] | None = None

    saved_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = cache_replay

    try:
        for seg_idx, (seg_start, seg_end) in enumerate(seg_ranges):
            seg_ids = input_ids[:, seg_start:seg_end]
            seg_positions = position_ids[:, seg_start:seg_end]
            seg_temps = temperature[:, seg_start:seg_end]

            # Per-segment backward mode: rebuild a fresh DynamicCache
            # from the detached prev_keys / prev_values before each
            # forward. This guarantees the cache object we pass into
            # model() has never been touched by any prior segment's
            # graph, so the torn-down activations from earlier segments
            # can't be pulled back in via shared state.
            if per_segment_backward and prev_keys is not None:
                fresh_cache = DynamicCache()
                for l, (k_seq, v_seq) in enumerate(zip(prev_keys, prev_values)):
                    # prev_keys[l] is [seq, heads, dim] and already detached
                    k_4d = k_seq.permute(1, 0, 2).unsqueeze(0).contiguous()
                    v_4d = v_seq.permute(1, 0, 2).unsqueeze(0).contiguous()
                    fresh_cache.update(k_4d, v_4d, l)
                past_key_values = fresh_cache

            if activation_checkpointing:
                seg_logits, new_keys, new_values = _run_segment_checkpointed(
                    model,
                    seg_ids,
                    seg_positions,
                    prev_keys,
                    prev_values,
                    capture_cache=cache_replay,
                )
            else:
                if cache_replay:
                    out = model(
                        input_ids=seg_ids,
                        position_ids=seg_positions,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                else:
                    out = model(
                        input_ids=seg_ids,
                        position_ids=seg_positions,
                        use_cache=False,
                    )
                raw_logits = out["logits"] if isinstance(out, dict) else out.logits
                # Some custom output heads (e.g. VanillaOutputLinear) wrap
                # logits in a dict. Unwrap once if we see that pattern.
                seg_logits = (
                    raw_logits["logits"] if isinstance(raw_logits, dict) else raw_logits
                )

            scaled_seg_logits = seg_logits / seg_temps.unsqueeze(-1).to(
                seg_logits.dtype
            )

            is_last_segment = seg_idx == len(seg_ranges) - 1

            # Determine the segment's "owned" range of global logit
            # positions: the positions in the concatenated full-logits
            # tensor that this segment is responsible for computing.
            #
            # Non-final segments drop the last logit (it's re-run by the
            # next segment under the post-eviction context). Final
            # segment keeps all logits.
            if is_last_segment:
                seg_logits_used = scaled_seg_logits
                owned_start = seg_start
                owned_end = seg_end
            else:
                seg_logits_used = scaled_seg_logits[:, :-1, :]
                owned_start = seg_start
                owned_end = seg_end - 1

            if per_segment_backward:
                # Compute this segment's loss via the caller's callback
                # and accumulate into the current BPTT window's loss
                # tensor. The window's combined loss will be backward'd
                # at the end of the window (see eviction block below).
                seg_loss = loss_fn(seg_logits_used, owned_start, owned_end)
                window_loss = seg_loss if window_loss is None else window_loss + seg_loss
                segments_in_window += 1
                # Release our references to the segment's forward-pass
                # tensors we no longer need. The retained KV that feeds
                # the next segment is extracted separately below.
                del seg_loss, seg_logits_used, scaled_seg_logits, seg_logits
                if not activation_checkpointing:
                    del out, raw_logits
            else:
                all_logits_pieces.append(seg_logits_used)

            # Determine whether this segment ends a BPTT window. If it
            # does, we'll need to backward the accumulated window_loss
            # and detach the retained KV that flows into the next
            # window. Windows close at three events:
            #   1. segments_in_window reaches bptt_segments, OR
            #   2. bptt_segments is None and this is the last segment
            #      (full trajectory as one big window), OR
            #   3. it's the last segment regardless (final flush).
            at_window_end = False
            if per_segment_backward:
                if bptt_segments is None:
                    at_window_end = is_last_segment
                else:
                    at_window_end = (
                        segments_in_window >= bptt_segments or is_last_segment
                    )

            # Between-segment KV eviction + cache rebuild.
            #
            # Legacy mode: retained KV keeps its autograd edges so the
            # final backward can flow through the cat chain (BPTT
            # attempt — only works in specific single-GPU modes).
            #
            # Per-segment backward mode: KV keeps autograd edges WITHIN
            # a BPTT window (so gradients can flow back through the
            # cat chain for up to `bptt_segments` steps). At window
            # boundaries we detach after computing the eviction, so
            # the next window starts with fresh leaves.
            if not is_last_segment:
                if activation_checkpointing:
                    keys = new_keys
                    values = new_values
                else:
                    kv_cache = captured_kv.get("past_key_values")
                    assert kv_cache is not None, (
                        "Hook did not capture past_key_values. Ensure model uses "
                        'impl="hf" (the custom llama path asserts past_key_values '
                        "is None and is incompatible with segmented forward)."
                    )
                    captured_kv.clear()
                    keys, values = _get_kv_from_cache(kv_cache)

                num_layers = len(keys)
                kv_seq_len = keys[0].shape[0]

                # Drop range [prompt_aligned_len, prompt_aligned_len + stride),
                # clamped against the actual assistant content we have.
                asst_len = kv_seq_len - prompt_aligned_len
                actual_stride = min(stride, asst_len)
                # Always trim the boundary token (the last KV entry) because
                # the next segment will re-feed that token and its new KV
                # will be appended. Without this trim, the boundary token
                # would appear twice in the KV cache.
                trim = 1

                # Output containers:
                #   - evicted_cache: DynamicCache consumed by the next
                #     segment's model() call in LEGACY mode
                #     (activation_checkpointing=False, loss_fn=None).
                #   - evicted_keys / evicted_values: list-of-tensor form
                #     consumed by:
                #       * the checkpointed-segment helper
                #         (activation_checkpointing=True), AND
                #       * the per-segment backward path
                #         (loss_fn provided), which rebuilds a fresh
                #         DynamicCache from these lists at the top of
                #         the next iteration.
                use_lists = activation_checkpointing or per_segment_backward
                evicted_cache = DynamicCache() if not use_lists else None
                evicted_keys: list[Tensor] = []
                evicted_values: list[Tensor] = []
                for l in range(num_layers):
                    if actual_stride > 0:
                        new_K = torch.cat(
                            [
                                keys[l][:prompt_aligned_len],
                                keys[l][prompt_aligned_len + actual_stride : -trim],
                            ],
                            dim=0,
                        )
                        new_V = torch.cat(
                            [
                                values[l][:prompt_aligned_len],
                                values[l][prompt_aligned_len + actual_stride : -trim],
                            ],
                            dim=0,
                        )
                    else:
                        # actual_stride==0 shouldn't happen in practice
                        # (needs_compaction guards it) but handle defensively.
                        new_K = keys[l][:-trim]
                        new_V = values[l][:-trim]

                    if use_lists:
                        # Keep [seq, heads, dim] for the list-based
                        # inputs. In per-segment-backward mode the source
                        # `keys[l]`/`values[l]` are already detached
                        # above, so new_K/new_V are also detached leaves.
                        evicted_keys.append(new_K)
                        evicted_values.append(new_V)
                    else:
                        # Legacy BPTT path: permute back to
                        # [1, heads, seq, dim] for DynamicCache and
                        # keep autograd edges intact so the final
                        # backward can flow through the cat chain.
                        new_K = new_K.permute(1, 0, 2).unsqueeze(0)
                        new_V = new_V.permute(1, 0, 2).unsqueeze(0)
                        evicted_cache.update(new_K, new_V, l)

                # Correct new_kv_len accounting: when the stride range
                # reaches the end of the KV, the trim range overlaps with it,
                # so a naive (kv_seq_len - actual_stride - trim) would
                # over-subtract. Compute what the cat actually produced:
                kept_asst = max(
                    0, (kv_seq_len - trim) - (prompt_aligned_len + actual_stride)
                )
                new_kv_len = prompt_aligned_len + kept_asst
                logger.debug(
                    "KV eviction seg %d: kv_len %d -> %d (dropped %d stride + "
                    "%d boundary), prompt_aligned=%d, retained_asst=%d",
                    seg_idx,
                    kv_seq_len,
                    new_kv_len,
                    actual_stride,
                    trim,
                    prompt_aligned_len,
                    kept_asst,
                )

                if use_lists:
                    # Drop local refs to the captured K/V. The cache
                    # object (captured via hook in the
                    # non-activation_checkpointing branch) is released
                    # here too since nothing else holds it.
                    del keys, values
                    if not activation_checkpointing:
                        del kv_cache
                    if per_segment_backward and at_window_end:
                        # End of BPTT window: backward the accumulated
                        # loss, then detach the evicted KV so the next
                        # window starts from fresh leaves. The
                        # backward call tears down the current
                        # window's forward graph and frees its
                        # activations. We grab detached copies BEFORE
                        # backward so the next-segment inputs don't
                        # depend on freed graph state.
                        prev_keys = [k.detach() for k in evicted_keys]
                        prev_values = [v.detach() for v in evicted_values]
                        assert window_loss is not None, (
                            "at_window_end reached with no accumulated "
                            "window_loss; per-segment backward state is "
                            "inconsistent"
                        )
                        window_loss.backward()
                        accumulated_loss += float(window_loss.detach().item())
                        window_loss = None
                        segments_in_window = 0
                        del evicted_keys, evicted_values
                    else:
                        # Mid-window (or activation_checkpointing path):
                        # retain autograd edges by passing the
                        # un-detached cat outputs straight through.
                        prev_keys = evicted_keys
                        prev_values = evicted_values
                    past_key_values = None
                else:
                    del keys, values, kv_cache
                    past_key_values = evicted_cache

            # Final-segment flush for per-segment backward mode: the
            # eviction block above only runs on non-final segments,
            # but the last segment also needs its window backward'd.
            # (For multi-window runs with the final segment landing
            # inside a partial window, this is the only place the
            # window_loss gets consumed; for a run where the final
            # window ended on a non-last segment, window_loss will
            # already have been reset to None by the eviction block.)
            if per_segment_backward and is_last_segment and window_loss is not None:
                window_loss.backward()
                accumulated_loss += float(window_loss.detach().item())
                window_loss = None
                segments_in_window = 0

    finally:
        if hook_handle is not None:
            hook_handle.remove()
        model.config.use_cache = saved_use_cache

    # Deliberately NOT calling torch.cuda.empty_cache() here: it's a hard
    # synchronization point that stalls the stream and returns cached blocks
    # to the CUDA driver right before FSDP2 is about to reallocate them for
    # the next micro-batch's all-gather. On a compaction hot path this
    # measurably regresses throughput and can interact badly with
    # activation-offloading pipelines. The inter-segment `del keys, values,
    # kv_cache` inside the loop releases the old cache promptly; relying on
    # the allocator is the same pattern the rest of the trainer uses.
    actual_passes = len(seg_ranges)
    target_passes = max_forward_passes or actual_passes

    if per_segment_backward:
        # In per-segment backward mode we've already called backward on
        # each real segment. For FSDP2 synchronization across ranks, any
        # remaining dummy passes still need to run a forward + dummy
        # backward so every rank issues the same collective sequence.
        # Each dummy backward contributes exactly zero to .grad because
        # the dummy output is multiplied by 0.
        if target_passes > actual_passes:
            _run_dummy_passes_with_backward(
                model,
                input_ids,
                position_ids,
                target_passes - actual_passes,
            )
        return {
            "loss": torch.tensor(accumulated_loss, device=device),
            "n_segments": actual_passes,
        }

    # Legacy path: concatenate per-segment logits and return them.
    full_logits = torch.cat(all_logits_pieces, dim=1)
    del all_logits_pieces

    if target_passes > actual_passes:
        full_logits = _pad_with_dummy_passes(
            model,
            input_ids,
            position_ids,
            full_logits,
            target_passes - actual_passes,
            device,
        )

    assert full_logits.shape[1] == seq_len, (
        f"segmented_forward produced {full_logits.shape[1]} logits, "
        f"expected {seq_len}. segments={len(seg_ranges)}, ranges={seg_ranges}"
    )

    return {"logits": full_logits}


def _dummy_forward_loss(
    model: torch.nn.Module,
    input_ids: Tensor,
    position_ids: Tensor,
) -> Tensor:
    d_out = model(
        input_ids=input_ids[:, :1],
        position_ids=position_ids[:, :1],
        use_cache=False,
    )
    d_logits = d_out["logits"] if isinstance(d_out, dict) else d_out.logits
    if isinstance(d_logits, dict):
        d_logits = d_logits["logits"]
    # float().mean() first to avoid bf16 sum overflow producing Inf
    # (Inf * 0 = NaN, which would corrupt gradients).
    return d_logits.float().mean() * 0.0


def _run_dummy_passes_with_backward(
    model: torch.nn.Module,
    input_ids: Tensor,
    position_ids: Tensor,
    num_dummy: int,
) -> None:
    """Run `num_dummy` dummy forward+backward pairs for FSDP2 rank sync.

    In per-segment backward mode, every rank needs to execute the same
    number of (forward, backward) pairs per training step so FSDP2's
    all-gather / reduce-scatter counts stay matched. If this rank
    processed K_local real segments and max across ranks is K_global,
    this fills in the gap with (K_global - K_local) dummy passes.

    Each dummy pass:
      1. Runs a real forward on a 1-token slice (FSDP2 all-gather fires).
      2. Constructs a zero-weighted loss (mean * 0) so backward flows
         into FSDP2 hooks without actually modifying any gradient.
      3. Calls .backward() to trigger FSDP2's reduce-scatter.
    """
    for _ in range(num_dummy):
        _dummy_forward_loss(model, input_ids, position_ids).backward()


def _pad_with_dummy_passes(
    model: torch.nn.Module,
    input_ids: Tensor,
    position_ids: Tensor,
    logits: Tensor,
    num_dummy: int,
    device: torch.device,
) -> Tensor:
    """Run `num_dummy` extra forward passes on a 1-token slice and add
    `dummy_sum * 0` to the logits.

    This keeps FSDP2 all-gather/reduce-scatter counts synchronized across
    ranks with different segment counts. Each dummy forward:
    1. Runs a real forward pass so FSDP backward hooks fire.
    2. Extracts a scalar (float().mean() to avoid bf16 sum overflow that
       could produce Inf; Inf*0=NaN would corrupt gradients).
    3. Accumulates into dummy_sum, which is multiplied by 0 at the end.

    The result is a no-op on the forward value (logits unchanged) and
    contributes exactly 0 to gradients, but keeps the autograd graph alive
    so FSDP all-reduces happen.
    """
    dummy_sum = torch.tensor(0.0, device=device)
    for _ in range(num_dummy):
        d_out = model(
            input_ids=input_ids[:, :1],
            position_ids=position_ids[:, :1],
            use_cache=False,
        )
        d_logits = d_out["logits"] if isinstance(d_out, dict) else d_out.logits
        if isinstance(d_logits, dict):
            d_logits = d_logits["logits"]
        dummy_sum = dummy_sum + d_logits.float().mean()
    return logits + (dummy_sum * 0).to(logits.dtype)


# ─── Per-call segmented forward ───────────────────────────────────────
#
# Mirrors vLLM's per-call independence: each chat() call in a rollout is
# treated as an atomic forward unit. We maintain a persistent
# `DynamicCache` across calls within a sample, detaching it between
# calls so the per-segment backward (M3 / bptt_segments=1) memory bound
# is preserved.
#
# Each call's forward processes ONLY the new tokens this call
# contributes to the merged frame, reading prior K/V from the persistent
# cache — structurally identical to vLLM's prefix-cache + single
# kernel-forward-per-call admission path.
#
# Admission events affect position_ids only: tokens in the kept portion
# of an admission call rotate at their pre-eviction logical positions
# (matching what vLLM's K writes were rotated at), and we accumulate
# total_evicted as a running `cum_position_offset` for subsequent
# tokens. No explicit cache splice is needed — the evicted tokens are
# absent from merged_input_ids (the orchestrator's _apply_admission_trim
# already removed them) and the causal mask + RoPE relative positions
# handle the rest.


def _call_has_admission(call) -> bool:
    """A CallWire has admission iff any of its compaction_events has
    num_output_tokens_at_compaction == 0."""
    events = getattr(call, "compaction_events", None) or []
    for e in events:
        if int(getattr(e, "num_output_tokens_at_compaction", 0)) == 0:
            return True
    return False


def _regular_split_prompt_tail_enabled() -> bool:
    """Default trainer execution shape for per-call compaction.

    The regular segmented path prefills newly submitted prompt tokens
    separately from completion/pad writes. This keeps cache semantics close to
    vLLM without enabling the token-by-token mirror-decode diagnostic.
    """
    return os.environ.get("KVE_TRAINER_SPLIT_PROMPT_TAIL", "1") != "0"


def _split_final_pad_enabled() -> bool:
    return os.environ.get("KVE_TRAINER_SPLIT_FINAL_PAD", "0") == "1"


def _per_call_mirror_decode_config() -> tuple[bool, set[int] | None, int]:
    """Mirror vLLM decode shape for per-call completion tails by default.

    Multi-turn environments such as DiscoveryWorld generate each assistant
    completion through vLLM decode iterations. Grouping those completion tokens
    into one trainer forward changes the cache-write chronology and measurably
    raises mismatch KL. The faithful default mirrors generated-token writes for
    every per-call turn; env filters are diagnostics/performance overrides only.

    Explicit env behavior is preserved: setting KVE_TRAINER_MIRROR_VLLM_DECODE=1
    with no filters mirrors every call, and setting it to 0 disables the mode.
    """
    raw_mirror = os.environ.get("KVE_TRAINER_MIRROR_VLLM_DECODE")
    if raw_mirror is None:
        mirror_decode = True
        default_last_n = 0
    else:
        mirror_decode = raw_mirror != "0"
        default_last_n = 0

    mirror_decode_filter_env = os.environ.get(
        "KVE_TRAINER_MIRROR_DECODE_CALL_IDX", ""
    )
    mirror_decode_call_filter = None
    if mirror_decode_filter_env:
        mirror_decode_call_filter = {
            int(x.strip())
            for x in mirror_decode_filter_env.split(",")
            if x.strip()
        }

    mirror_decode_last_n = int(
        os.environ.get(
            "KVE_TRAINER_MIRROR_DECODE_LAST_N", str(default_last_n),
        )
        or "0"
    )
    return mirror_decode, mirror_decode_call_filter, mirror_decode_last_n


def _mirror_decode_call_enabled(
    *,
    call_idx: int,
    num_calls: int,
    mirror_decode: bool,
    mirror_decode_call_filter: set[int] | None,
    mirror_decode_last_n: int,
) -> bool:
    if not mirror_decode:
        return False
    mirror_decode_has_filter = (
        mirror_decode_call_filter is not None or mirror_decode_last_n > 0
    )
    return (
        not mirror_decode_has_filter
        or (
            mirror_decode_call_filter is not None
            and call_idx in mirror_decode_call_filter
        )
        or (
            mirror_decode_last_n > 0
            and call_idx >= max(0, num_calls - mirror_decode_last_n)
        )
    )


def _count_tail_forwards(
    *,
    seq_len: int,
    prompt_len: int,
    pad_len: int,
    mirror_decode: bool,
) -> int:
    """Number of model forwards used by _forward_with_optional_decode_mirror."""
    if seq_len <= 0:
        return 0

    keep_len = seq_len - int(pad_len)
    if keep_len < 0:
        raise ValueError(
            f"pad_len={pad_len} exceeds forwarded seq_len={seq_len}"
        )
    prompt_len = max(0, min(int(prompt_len), keep_len))

    if not mirror_decode:
        if (
            _regular_split_prompt_tail_enabled()
            and 0 < prompt_len < seq_len
        ):
            n = 1  # prompt/new-user prefill
            if _split_final_pad_enabled() and pad_len > 0:
                final_pad_start = max(prompt_len, keep_len - 1)
                if final_pad_start > prompt_len:
                    n += 1
                if seq_len > final_pad_start:
                    n += 1
            else:
                n += 1  # completion plus any trailing pad
            return n

        if _split_final_pad_enabled() and pad_len > 0:
            tail_start = max(0, keep_len - 1)
            n = 0
            if tail_start > 0:
                n += 1
            if seq_len > tail_start:
                n += 1
            return max(n, 1)

        return 1

    if os.environ.get("KVE_TRAINER_MIRROR_PROMPT_DECODE", "0") == "1":
        prompt_len = 0

    n = 0
    if prompt_len > 0:
        n += 1
    if keep_len > prompt_len:
        n += max(0, keep_len - 1 - prompt_len)
        n += 1
    elif keep_len < seq_len:
        n += 1
    return max(n, 1)


def compute_per_call_forward_counts(calls) -> list[int]:
    """Forward-pass count for each non-empty call in per-call dispatch.

    Counts the regular segmented default by replaying the same per-call plan:
    old survivor K/V stays in the persistent cache, new prompt tokens prefill
    as one chunk, and completion/pad writes run as the tail chunk. The
    token-by-token mirror-decode diagnostic is still accounted for when its
    env gate is explicitly enabled.
    """
    if not calls:
        return []
    plans, _ = _build_pre_trim_plan(calls)
    (
        mirror_decode,
        mirror_decode_call_filter,
        mirror_decode_last_n,
    ) = _per_call_mirror_decode_config()
    b2b_warmup_splice = (
        os.environ.get("KVE_TRAINER_B2B_WARMUP_SPLICE", "0") == "1"
    )

    counts: list[int] = []
    for plan in plans:
        call_idx = plan["call_idx"]
        cps = plan["call_pre_start"]
        cpe = plan["call_pre_end"]
        has_admission = plan["has_admission"]
        sub_end_pre = plan.get("sub_end_pre", plan["inherited_end_pre"])
        b2b_warm_end_pre = plan.get("b2b_warm_end_pre", sub_end_pre)
        pad_len = plan.get("pad_len", 0)
        prefix_replay_len = len(plan.get("prefix_replay_tokens", []) or [])
        mirror_decode_this_call = _mirror_decode_call_enabled(
            call_idx=call_idx,
            num_calls=len(plans),
            mirror_decode=mirror_decode,
            mirror_decode_call_filter=mirror_decode_call_filter,
            mirror_decode_last_n=mirror_decode_last_n,
        )

        if cpe <= cps:
            continue

        n = 0
        if prefix_replay_len > 0:
            n += 1
        if has_admission and cps > 0:
            if b2b_warmup_splice or prefix_replay_len > 0:
                warm_end = b2b_warm_end_pre
                if warm_end > cps:
                    n += 1
                main_start = warm_end
                main_prompt_len = max(0, min(sub_end_pre, cpe) - main_start)
                n += _count_tail_forwards(
                    seq_len=cpe - main_start,
                    prompt_len=main_prompt_len,
                    pad_len=pad_len,
                    mirror_decode=mirror_decode_this_call,
                )
            else:
                main_prompt_len = max(0, min(sub_end_pre, cpe) - cps)
                n += _count_tail_forwards(
                    seq_len=cpe - cps,
                    prompt_len=main_prompt_len,
                    pad_len=pad_len,
                    mirror_decode=mirror_decode_this_call,
                )
            counts.append(n)
            continue

        if has_admission:
            warm_end = sub_end_pre
            if warm_end > cps:
                n += 1
            main_start = sub_end_pre
            n += _count_tail_forwards(
                seq_len=cpe - main_start,
                prompt_len=0,
                pad_len=pad_len,
                mirror_decode=mirror_decode_this_call,
            )
            counts.append(n)
            continue

        new_prompt_len = max(0, min(sub_end_pre, cpe) - cps)
        n += _count_tail_forwards(
            seq_len=cpe - cps,
            prompt_len=new_prompt_len,
            pad_len=pad_len,
            mirror_decode=mirror_decode_this_call,
        )
        counts.append(n)
    return counts


def compute_num_per_call_forwards(calls) -> int:
    """How many forward passes per_call_segmented_forward will run for
    this list of calls. Used to all-reduce-MAX across DP ranks so FSDP2
    all-gathers stay synchronized."""
    return max(sum(compute_per_call_forward_counts(calls)), 1)


def compute_per_call_bptt_window_forward_counts(
    calls,
    bptt_segments: int | None,
) -> list[int]:
    """Forward-pass counts grouped by per-call BPTT window."""
    bptt_segments = _normalize_bptt_segments(bptt_segments)
    call_counts = compute_per_call_forward_counts(calls)
    if not call_counts:
        return []
    if bptt_segments is None:
        return [sum(call_counts)]
    if bptt_segments < 1:
        raise ValueError(f"bptt_segments must be None or >= 1, got {bptt_segments}")
    return [
        sum(call_counts[i : i + bptt_segments])
        for i in range(0, len(call_counts), bptt_segments)
    ]


def compute_num_per_call_bptt_windows(
    calls,
    bptt_segments: int | None,
) -> int:
    """Number of real backward windows for per-call BPTT."""
    bptt_segments = _normalize_bptt_segments(bptt_segments)
    counts = compute_per_call_forward_counts(calls)
    if not counts:
        return 0
    if bptt_segments is None:
        return 1
    if bptt_segments < 1:
        raise ValueError(f"bptt_segments must be None or >= 1, got {bptt_segments}")
    return math.ceil(len(counts) / bptt_segments)


def _detach_dynamic_cache(cache: DynamicCache) -> DynamicCache:
    """Return a fresh DynamicCache whose K/V tensors are detached from
    the autograd graph. Called between per-call forwards in
    `per_call_segmented_forward` so each call's backward stops at the
    cache boundary, preserving bptt_segments=1 / M3 semantics."""
    detached = DynamicCache()
    n_layers = len(cache)
    for layer_idx in range(n_layers):
        if hasattr(cache, "layers"):
            k = cache.layers[layer_idx].keys
            v = cache.layers[layer_idx].values
        else:
            k = cache.key_cache[layer_idx]
            v = cache.value_cache[layer_idx]
        detached.update(k.detach(), v.detach(), layer_idx)
    return detached


def _splice_dynamic_cache(
    cache: DynamicCache,
    evict_start: int,
    evict_end: int,
) -> DynamicCache:
    """Drop K/V slices [evict_start, evict_end) from every layer of a
    DynamicCache. Mirrors compaction_debug.py / diagnostic_kl_dummy.py's
    H2 oracle splice (lines 254-271).

    Per-layer K/V shape is [1, n_heads, seq, head_dim]. We slice along
    dim=2. Returns a fresh DynamicCache with the spliced K/V uploaded.
    """
    spliced = DynamicCache()
    n_layers = len(cache)
    for layer_idx in range(n_layers):
        if hasattr(cache, "layers"):
            k = cache.layers[layer_idx].keys
            v = cache.layers[layer_idx].values
        else:
            k = cache.key_cache[layer_idx]
            v = cache.value_cache[layer_idx]
        # [1, n_heads, seq, head_dim]; drop along dim=2.
        new_k = torch.cat(
            [k[:, :, :evict_start, :], k[:, :, evict_end:, :]], dim=2,
        ).contiguous()
        new_v = torch.cat(
            [v[:, :, :evict_start, :], v[:, :, evict_end:, :]], dim=2,
        ).contiguous()
        spliced.update(new_k, new_v, layer_idx)
    return spliced


def _piecewise_positions(
    start: int,
    end: int,
    *,
    offset: int,
    protected_prefix_len: int,
    device: torch.device,
) -> torch.Tensor:
    """vLLM's piecewise RoPE frame for physical positions [start, end)."""
    physical = torch.arange(start, end, device=device, dtype=torch.long)
    if offset != 0:
        if protected_prefix_len > 0:
            physical = physical + (
                (physical >= protected_prefix_len).to(torch.long) * offset
            )
        else:
            physical = physical + offset
    return physical.unsqueeze(0)


def _extract_logits(model_output) -> Tensor:
    logits = (
        model_output["logits"]
        if isinstance(model_output, dict)
        else model_output.logits
    )
    if isinstance(logits, dict):
        logits = logits["logits"]
    return logits


@contextmanager
def _lm_head_logits_forward(model: torch.nn.Module):
    """Temporarily make PrimeRL's selected-logprob fused head return logits.

    FusedOutputLinear is an nn.Linear subclass, but its forward intentionally
    requires labels and returns only selected logprobs/entropy. Logit-based
    segmented paths, including compact distillation, still need full logits.
    Patching the instance forward keeps normal module call hooks in place.
    """
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None or lm_head.__class__.__name__ != "FusedOutputLinear":
        yield
        return

    original_forward = lm_head.forward

    def _forward_logits(
        hidden_states: Tensor,
        labels: Tensor | None = None,
        temperature: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del labels, temperature
        bias = getattr(lm_head, "bias", None)
        return {"logits": F.linear(hidden_states, lm_head.weight, bias)}

    lm_head.forward = _forward_logits
    try:
        yield
    finally:
        lm_head.forward = original_forward


def _model_forward_logits(model: torch.nn.Module, **kwargs) -> Tensor:
    with _lm_head_logits_forward(model):
        logits = _extract_logits(model(**kwargs))
    if logits is None:
        raise RuntimeError("model forward did not return logits")
    return logits


def _preview_1d(t: Tensor, start: int, end: int) -> tuple[list[int], list[int]]:
    vals = t[0, start:end].detach().cpu().tolist()
    return vals[:8], vals[-8:]


def _set_attn_implementation_if_needed(
    model: torch.nn.Module,
    attn_impl: str | None,
) -> None:
    if not attn_impl or not hasattr(model, "set_attn_implementation"):
        return
    cfg = getattr(model, "config", None)
    cur = getattr(cfg, "_attn_implementation", None)
    if cur != attn_impl:
        model.set_attn_implementation(attn_impl)


def _model_forward_with_optional_attn_switch(
    model: torch.nn.Module,
    *,
    input_ids: Tensor,
    position_ids: Tensor,
    past_key_values: DynamicCache,
    base_attn_impl: str | None,
    past_attn_impl: str | None,
    attn_impl_override: str | None = None,
):
    target_attn_impl = attn_impl_override or (
        past_attn_impl
        if past_attn_impl and past_key_values.get_seq_length() > 0
        else base_attn_impl
    )
    _set_attn_implementation_if_needed(model, target_attn_impl)
    # PrimeRL's activation-checkpoint wrappers checkpoint modules only while
    # they are in training mode. Per-call replay must pass a live DynamicCache
    # through the model; checkpointing that mutable object is unsafe because
    # the cache is longer by the time backward recomputes the wrapped block.
    # eval() leaves autograd enabled, so gradients still flow while the replay
    # avoids a checkpoint/cache metadata mismatch.
    was_training = model.training
    if was_training:
        model.eval()
    try:
        with _lm_head_logits_forward(model):
            return model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
    finally:
        if was_training:
            model.train()


def _forward_with_optional_decode_mirror(
    model: torch.nn.Module,
    *,
    input_ids: Tensor,
    position_ids: Tensor,
    past_key_values: DynamicCache,
    prompt_len: int,
    pad_len: int,
    mirror_decode: bool,
    base_attn_impl: str | None,
    past_attn_impl: str | None,
    trace_label: str | None = None,
) -> tuple[Tensor, int]:
    """Forward a request tail, optionally mirroring vLLM decode shape.

    Diagnostic mode (`KVE_TRAINER_MIRROR_VLLM_DECODE=1`) preserves the
    same logical span but changes execution shape:
      - prompt/new-user tail in one prefill forward;
      - generated tokens one at a time;
      - final generated token plus auto-pad fillers in one tail forward.

    Returned logits exclude auto-pad filler rows, matching the normal
    loss/logprob frame. The cache still receives the pad K/V writes.
    """
    seq_len = int(input_ids.shape[1])
    if seq_len <= 0:
        raise ValueError("_forward_with_optional_decode_mirror got empty input")

    keep_len = seq_len - int(pad_len)
    if keep_len < 0:
        raise ValueError(
            f"pad_len={pad_len} exceeds forwarded seq_len={seq_len}"
        )
    prompt_len = max(0, min(int(prompt_len), keep_len))

    if not mirror_decode:
        split_prompt_tail = (
            _regular_split_prompt_tail_enabled()
            and 0 < prompt_len < seq_len
        )
        if split_prompt_tail:
            pieces: list[Tensor] = []
            n_forwards = 0

            cache_before = past_key_values.get_seq_length()
            out = _model_forward_with_optional_attn_switch(
                model,
                input_ids=input_ids[:, :prompt_len],
                position_ids=position_ids[:, :prompt_len],
                past_key_values=past_key_values,
                base_attn_impl=base_attn_impl,
                past_attn_impl=past_attn_impl,
            )
            pieces.append(_extract_logits(out))
            n_forwards += 1
            if trace_label is not None:
                tok_head, tok_tail = _preview_1d(input_ids, 0, prompt_len)
                pos_head, pos_tail = _preview_1d(position_ids, 0, prompt_len)
                logger.warning(
                    "[T-FWD-SIG] label=%s chunk=[%d,%d) len=%d "
                    "cache=%d->%d prompt_len=%d pad_len=%d mirror=%s "
                    "attn_override=%s pos_head=%s pos_tail=%s "
                    "tok_head=%s tok_tail=%s",
                    trace_label, 0, prompt_len, prompt_len,
                    cache_before, past_key_values.get_seq_length(),
                    prompt_len, pad_len, mirror_decode, None,
                    pos_head, pos_tail, tok_head, tok_tail,
                )

            split_final_pad_tail = (
                _split_final_pad_enabled()
                and pad_len > 0
            )
            tail_chunks: list[tuple[int, int]] = []
            if split_final_pad_tail:
                final_pad_start = max(prompt_len, keep_len - 1)
                if final_pad_start > prompt_len:
                    tail_chunks.append((prompt_len, final_pad_start))
                tail_chunks.append((final_pad_start, seq_len))
            else:
                tail_chunks.append((prompt_len, seq_len))

            for start, end in tail_chunks:
                if end <= start:
                    continue
                cache_before = past_key_values.get_seq_length()
                out = _model_forward_with_optional_attn_switch(
                    model,
                    input_ids=input_ids[:, start:end],
                    position_ids=position_ids[:, start:end],
                    past_key_values=past_key_values,
                    base_attn_impl=base_attn_impl,
                    past_attn_impl=past_attn_impl,
                )
                tail_logits = _extract_logits(out)
                n_forwards += 1
                if trace_label is not None:
                    tok_head, tok_tail = _preview_1d(input_ids, start, end)
                    pos_head, pos_tail = _preview_1d(position_ids, start, end)
                    logger.warning(
                        "[T-FWD-SIG] label=%s chunk=[%d,%d) len=%d "
                        "cache=%d->%d prompt_len=%d pad_len=%d mirror=%s "
                        "attn_override=%s pos_head=%s pos_tail=%s "
                        "tok_head=%s tok_tail=%s",
                        trace_label, start, end, end - start,
                        cache_before, past_key_values.get_seq_length(),
                        prompt_len, pad_len, mirror_decode, None,
                        pos_head, pos_tail, tok_head, tok_tail,
                    )
                keep_in_chunk = max(0, min(end, keep_len) - start)
                if keep_in_chunk > 0:
                    pieces.append(tail_logits[:, :keep_in_chunk, :])
            return torch.cat(pieces, dim=1), n_forwards

        split_final_pad = (
            _split_final_pad_enabled()
            and pad_len > 0
        )
        if split_final_pad:
            pieces: list[Tensor] = []
            n_forwards = 0
            tail_start = max(0, keep_len - 1)
            if tail_start > 0:
                cache_before = past_key_values.get_seq_length()
                out = _model_forward_with_optional_attn_switch(
                    model,
                    input_ids=input_ids[:, :tail_start],
                    position_ids=position_ids[:, :tail_start],
                    past_key_values=past_key_values,
                    base_attn_impl=base_attn_impl,
                    past_attn_impl=past_attn_impl,
                )
                pieces.append(_extract_logits(out))
                n_forwards += 1
                if trace_label is not None:
                    tok_head, tok_tail = _preview_1d(input_ids, 0, tail_start)
                    pos_head, pos_tail = _preview_1d(position_ids, 0, tail_start)
                    logger.warning(
                        "[T-FWD-SIG] label=%s chunk=[%d,%d) len=%d "
                        "cache=%d->%d prompt_len=%d pad_len=%d mirror=%s "
                        "attn_override=%s pos_head=%s pos_tail=%s "
                        "tok_head=%s tok_tail=%s",
                        trace_label, 0, tail_start, tail_start,
                        cache_before, past_key_values.get_seq_length(),
                        prompt_len, pad_len, mirror_decode, None,
                        pos_head, pos_tail, tok_head, tok_tail,
                    )
            cache_before = past_key_values.get_seq_length()
            out = _model_forward_with_optional_attn_switch(
                model,
                input_ids=input_ids[:, tail_start:],
                position_ids=position_ids[:, tail_start:],
                past_key_values=past_key_values,
                base_attn_impl=base_attn_impl,
                past_attn_impl=past_attn_impl,
            )
            tail_logits = _extract_logits(out)
            n_forwards += 1
            if trace_label is not None:
                tok_head, tok_tail = _preview_1d(input_ids, tail_start, seq_len)
                pos_head, pos_tail = _preview_1d(position_ids, tail_start, seq_len)
                logger.warning(
                    "[T-FWD-SIG] label=%s chunk=[%d,%d) len=%d "
                    "cache=%d->%d prompt_len=%d pad_len=%d mirror=%s "
                    "attn_override=%s pos_head=%s pos_tail=%s "
                    "tok_head=%s tok_tail=%s",
                    trace_label, tail_start, seq_len, seq_len - tail_start,
                    cache_before, past_key_values.get_seq_length(),
                    prompt_len, pad_len, mirror_decode, None,
                    pos_head, pos_tail, tok_head, tok_tail,
                )
            if keep_len > tail_start:
                pieces.append(tail_logits[:, : keep_len - tail_start, :])
            if pieces:
                return torch.cat(pieces, dim=1), n_forwards
            return tail_logits[:, :0, :], n_forwards

        cache_before = past_key_values.get_seq_length()
        out = _model_forward_with_optional_attn_switch(
            model,
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=past_key_values,
            base_attn_impl=base_attn_impl,
            past_attn_impl=past_attn_impl,
        )
        logits = _extract_logits(out)
        if trace_label is not None:
            tok_head, tok_tail = _preview_1d(input_ids, 0, seq_len)
            pos_head, pos_tail = _preview_1d(position_ids, 0, seq_len)
            logger.warning(
                "[T-FWD-SIG] label=%s chunk=[%d,%d) len=%d "
                "cache=%d->%d prompt_len=%d pad_len=%d mirror=%s "
                "attn_override=%s pos_head=%s pos_tail=%s "
                "tok_head=%s tok_tail=%s",
                trace_label, 0, seq_len, seq_len,
                cache_before, past_key_values.get_seq_length(),
                prompt_len, pad_len, mirror_decode, None,
                pos_head, pos_tail, tok_head, tok_tail,
            )
        return logits[:, :keep_len, :], 1

    if os.environ.get("KVE_TRAINER_MIRROR_PROMPT_DECODE", "0") == "1":
        prompt_len = 0

    pieces: list[Tensor] = []
    n_forwards = 0
    last_logits: Tensor | None = None

    def run_chunk(
        start: int,
        end: int,
        *,
        attn_impl_override: str | None = None,
    ) -> Tensor:
        nonlocal n_forwards, last_logits
        cache_before = past_key_values.get_seq_length()
        out = _model_forward_with_optional_attn_switch(
            model,
            input_ids=input_ids[:, start:end],
            position_ids=position_ids[:, start:end],
            past_key_values=past_key_values,
            base_attn_impl=base_attn_impl,
            past_attn_impl=past_attn_impl,
            attn_impl_override=attn_impl_override,
        )
        logits = _extract_logits(out)
        last_logits = logits
        n_forwards += 1
        if trace_label is not None:
            tok_head, tok_tail = _preview_1d(input_ids, start, end)
            pos_head, pos_tail = _preview_1d(position_ids, start, end)
            logger.warning(
                "[T-FWD-SIG] label=%s chunk=[%d,%d) len=%d "
                "cache=%d->%d prompt_len=%d pad_len=%d mirror=%s "
                "attn_override=%s pos_head=%s pos_tail=%s "
                "tok_head=%s tok_tail=%s",
                trace_label,
                start,
                end,
                end - start,
                cache_before,
                past_key_values.get_seq_length(),
                prompt_len,
                pad_len,
                mirror_decode,
                str(attn_impl_override),
                pos_head,
                pos_tail,
                tok_head,
                tok_tail,
            )
        return logits

    # vLLM prefill: all prompt/new-user tokens in one forward.
    if prompt_len > 0:
        prompt_tail_attn_impl = (
            os.environ.get("KVE_TRAINER_PROMPT_TAIL_ATTN_IMPL", "") or None
        )
        if past_key_values.get_seq_length() <= 0:
            prompt_tail_attn_impl = None
        pieces.append(
            run_chunk(
                0,
                prompt_len,
                attn_impl_override=prompt_tail_attn_impl,
            )
        )

    # vLLM decode: each already-sampled completion token is fed on the
    # next scheduler iteration. The last completion token is fed together
    # with auto-pad fillers when auto-pad is active.
    if keep_len > prompt_len:
        for i in range(prompt_len, keep_len - 1):
            pieces.append(run_chunk(i, i + 1))
        tail_logits = run_chunk(keep_len - 1, seq_len)
        pieces.append(tail_logits[:, :1, :])
    elif keep_len < seq_len:
        # Pad-only tail. This should be rare, but keep cache layout exact.
        run_chunk(keep_len, seq_len)

    if pieces:
        return torch.cat(pieces, dim=1), n_forwards
    assert last_logits is not None
    return last_logits[:, :0, :], n_forwards


def _backward_loss_or_zero(loss_val: Tensor, logits: Tensor) -> None:
    if loss_val.requires_grad:
        loss_val.backward()
    elif logits.requires_grad:
        (logits.float().mean() * 0.0).backward()


def _forward_mirror_decode_stream_loss(
    model: torch.nn.Module,
    *,
    input_ids: Tensor,
    position_ids: Tensor,
    past_key_values: DynamicCache,
    prompt_len: int,
    pad_len: int,
    base_attn_impl: str | None,
    past_attn_impl: str | None,
    loss_fn: SegmentLossFn,
    full_logit_start: int,
    trace_label: str | None = None,
) -> tuple[DynamicCache, int, float, int]:
    """Run vLLM-shaped decode forwards without retaining every graph.

    This is the production-memory version of
    `_forward_with_optional_decode_mirror(..., mirror_decode=True)`.
    Each kept logits chunk is sent through the caller's `loss_fn` and
    backwarded immediately, then the cache is detached before the next
    decode chunk. This keeps the diagnostic execution shape while
    bounding activation lifetime to one chunk.
    """
    seq_len = int(input_ids.shape[1])
    if seq_len <= 0:
        raise ValueError("_forward_mirror_decode_stream_loss got empty input")

    keep_len = seq_len - int(pad_len)
    if keep_len < 0:
        raise ValueError(
            f"pad_len={pad_len} exceeds forwarded seq_len={seq_len}"
        )
    prompt_len = max(0, min(int(prompt_len), keep_len))
    if os.environ.get("KVE_TRAINER_MIRROR_PROMPT_DECODE", "0") == "1":
        prompt_len = 0

    n_forwards = 0
    accumulated_loss = 0.0
    kept_so_far = 0

    def run_chunk(
        start: int,
        end: int,
        *,
        keep_in_chunk: int,
        attn_impl_override: str | None = None,
    ) -> None:
        nonlocal past_key_values, n_forwards, accumulated_loss, kept_so_far
        cache_before = past_key_values.get_seq_length()
        out = _model_forward_with_optional_attn_switch(
            model,
            input_ids=input_ids[:, start:end],
            position_ids=position_ids[:, start:end],
            past_key_values=past_key_values,
            base_attn_impl=base_attn_impl,
            past_attn_impl=past_attn_impl,
            attn_impl_override=attn_impl_override,
        )
        logits = _extract_logits(out)
        n_forwards += 1
        if trace_label is not None:
            tok_head, tok_tail = _preview_1d(input_ids, start, end)
            pos_head, pos_tail = _preview_1d(position_ids, start, end)
            logger.warning(
                "[T-FWD-SIG] label=%s chunk=[%d,%d) len=%d "
                "cache=%d->%d prompt_len=%d pad_len=%d mirror=%s "
                "attn_override=%s pos_head=%s pos_tail=%s "
                "tok_head=%s tok_tail=%s stream_loss=True",
                trace_label,
                start,
                end,
                end - start,
                cache_before,
                past_key_values.get_seq_length(),
                prompt_len,
                pad_len,
                True,
                str(attn_impl_override),
                pos_head,
                pos_tail,
                tok_head,
                tok_tail,
            )

        if keep_in_chunk > 0:
            chunk_logits = logits[:, :keep_in_chunk, :]
            chunk_start = full_logit_start + kept_so_far
            chunk_end = chunk_start + keep_in_chunk
            loss_val = loss_fn(chunk_logits, chunk_start, chunk_end)
            _backward_loss_or_zero(loss_val, chunk_logits)
            accumulated_loss += float(loss_val.detach().item())
            kept_so_far += keep_in_chunk
        else:
            (logits.float().mean() * 0.0).backward()
        past_key_values = _detach_dynamic_cache(past_key_values)

    if prompt_len > 0:
        prompt_tail_attn_impl = (
            os.environ.get("KVE_TRAINER_PROMPT_TAIL_ATTN_IMPL", "") or None
        )
        if past_key_values.get_seq_length() <= 0:
            prompt_tail_attn_impl = None
        run_chunk(
            0,
            prompt_len,
            keep_in_chunk=prompt_len,
            attn_impl_override=prompt_tail_attn_impl,
        )

    if keep_len > prompt_len:
        for i in range(prompt_len, keep_len - 1):
            run_chunk(i, i + 1, keep_in_chunk=1)
        run_chunk(keep_len - 1, seq_len, keep_in_chunk=1)
    elif keep_len < seq_len:
        run_chunk(keep_len, seq_len, keep_in_chunk=0)

    return past_key_values, n_forwards, accumulated_loss, kept_so_far


def _apply_splices_to_submitted(
    submitted: list[int],
    splices: list[tuple[int, int]],
    cached_pre: int,
) -> tuple[list[int], list[int], int]:
    """Apply admission deletions to submitted prompt and cached length.

    Returns:
      - post-trim submitted token ids
      - post-index -> pre-index map
      - cached token count after the same deletions
    """
    post = list(submitted)
    post_to_pre = list(range(len(submitted)))
    cached = int(cached_pre)
    for es, te in splices:
        if te <= 0:
            continue
        overlap = max(0, min(cached, es + te) - es)
        cached -= overlap
        del post[es:es + te]
        del post_to_pre[es:es + te]
    return post, post_to_pre, cached


def _build_pre_trim_plan(calls):
    """Walk a sample's CallWire list and build a per-call execution plan.

    For each call, computes:
      - call_pre_start, call_pre_end: this call's contribution range in
        the sample's pre-trim cumulative coordinate (= persistent_cache
        logical positions this call writes).
      - has_admission: bool — call has at least one admission event
        with tokens_evicted > 0.
      - inherited_end_pre: pre-trim absolute position where this call's
        "new" content begins. For admission calls this is
        (call_pre_start + sub_len - new_user_fragment_len). For non-
        admission calls this is call_pre_start (no inherited region to
        warm up — extension calls inherit via persistent_cache).
      - splices: list of (es, te) tuples per admission event in order.
        Each event's evict_start is in the CURRENT submitted-prompt-iter
        coordinate (matches `_apply_admission_trim` semantics: each
        successive event sees the prior iters' deletions already
        applied). Translates 1:1 to cache-index for `_splice_dynamic_cache`
        because cache content and submitted_prompt are co-indexed up
        through `inherited_end_pre` at the moment of splicing.
      - post_start, post_end: post-trim merged coord (matches
        merged_input_ids / labels / loss_mask / advantages frame).
      - kept_in_call: int64 list of pre-trim positions WITHIN this
        call's contribution that survive admission. Indexed relative to
        call_pre_start. Length = post_end - post_start.
      - new_content_start_in_sub: where this call's NEW pre-trim
        contribution starts within submitted_prompt_ids (0 for first
        call, prior cum_pre for extension calls).

    Asserts no mid-generation events (those are routed to legacy
    segmented_forward upstream).

    Returns (plans, pre_trim_ids) where pre_trim_ids is the full
    pre-trim merged token list for the sample.
    """
    plans = []
    pre_trim_ids: list[int] = []
    cum_pre = 0
    cum_post = 0
    # DIAG: stitch-fidelity trace. Reconstructs V's actual KV-cache token
    # sequence by replaying each call's writes + admission splices, then
    # at entry to every extension call (N>0) compares against the
    # orchestrator-provided `sub[:cum_post]`. A mismatch means
    # Phase 4 stitching has placed V's cache content somewhere different
    # from what the trainer's plan accounting assumes — making every
    # downstream `new_content_start_in_sub = cum_post` slice off by the
    # mismatch length. Gated on KVE_TRACE_STITCH=1.
    stitch_trace = os.environ.get("KVE_TRACE_STITCH", "") == "1"
    expected_v_cache_tokens: list[int] = []
    if stitch_trace:
        import sys as _sys
        print(
            f"[STITCH-START] _build_pre_trim_plan: {len(calls)} calls",
            file=_sys.stderr, flush=True,
        )
    for call_idx, call in enumerate(calls):
        sub = list(call.submitted_prompt_ids)
        comp = list(call.completion_ids)
        # vLLM auto-pad: filler tokens vLLM appended to its KV cache at
        # finish-time so the trailing block lands in the prefix cache.
        # Empty when auto-pad did not fire for this call. They live at
        # the END of this call's K-cache contribution (after comp).
        trailing_pad = list(getattr(call, "trailing_pad_ids", None) or [])
        sub_len = len(sub)
        comp_len = len(comp)
        pad_len = len(trailing_pad)

        all_events = list(call.compaction_events or [])
        # Managed-context restore lifecycle (kind 1 attach / kind 2 release):
        # collected separately as visibility ops, consumed by the interval
        # mask in _build_flex_mask_writer_timeline. They never splice rows,
        # so the mid-gen restriction below does not apply to them (releases
        # legitimately fire mid-decode).
        visibility_ops: list[tuple[int, list[str], int]] = []
        # Recall token-surfacing: span id -> (token_ids, pos_start) carried on
        # a kind-1 ATTACH whose span was evicted before this sample's first
        # captured turn. The timeline builder reconstructs the span's KV by
        # forwarding these tokens at pos_start.. so the recalled keys match
        # what the sampler attended to (see _append_hidden_span).
        recall_span_tokens: dict[str, tuple[list[int], int]] = {}
        _op_positions: list[int] = []
        _evict_positions: list[tuple[int, int, int]] = []
        events = []
        for _ev_idx, e in enumerate(all_events):
            kind = int(getattr(e, "event_kind", 0))
            if kind != 0:
                boundary = int(
                    getattr(e, "visibility_boundary_computed", -1)
                )
                span_ids = [
                    str(x)
                    for x in getattr(e, "restored_span_ids", []) or []
                ]
                if kind not in (1, 2) or boundary < 0 or not span_ids:
                    raise ValueError(
                        f"call {call_idx}: malformed visibility event "
                        f"kind={kind} boundary={boundary} spans={span_ids}"
                    )
                surfaced_tokens = [
                    int(x)
                    for x in getattr(e, "restored_span_token_ids", []) or []
                ]
                surfaced_pos_start = int(
                    getattr(e, "restored_span_pos_start", -1)
                )
                if kind == 1 and surfaced_tokens and surfaced_pos_start >= 0:
                    # Single-span surfacing (the perpetual-anchor recall).
                    if len(span_ids) == 1:
                        recall_span_tokens[span_ids[0]] = (
                            surfaced_tokens, surfaced_pos_start
                        )
                visibility_ops.append((kind, span_ids, boundary))
                _op_positions.append(_ev_idx)
                continue
            assert int(e.num_output_tokens_at_compaction) == 0, (
                "per_call_segmented_forward got a mid-generation event "
                f"(call {call_idx}); upstream dispatcher must route "
                "mid-gen samples to legacy segmented_forward."
            )
            if int(e.tokens_evicted) > 0:
                _evict_positions.append(
                    (_ev_idx, int(e.evict_start), int(e.tokens_evicted))
                )
            events.append(e)
        # Upfront restores attach at admission BEFORE the same call's
        # eviction trims the stream, so their boundaries arrive in the
        # PRE-eviction frame (M2 smoke v7 ledger: attach boundary 3663 on
        # a 3664-token pre-trim prompt, eviction trims to 3280 afterwards).
        # The timeline builder maps ops after applying splices — re-express
        # each boundary in the post-trim frame by subtracting its overlap
        # with every eviction that fired LATER in this call's event order.
        if visibility_ops and _evict_positions:
            adjusted: list[tuple[int, list[str], int]] = []
            for op_pos, (kind, span_ids, boundary) in zip(
                _op_positions, visibility_ops
            ):
                b = boundary
                for ev_pos, es, te in _evict_positions:
                    if ev_pos > op_pos and b > es:
                        b -= min(te, b - es)
                adjusted.append((kind, span_ids, b))
            visibility_ops = adjusted
        admission_events = [
            e for e in events
            if int(e.num_output_tokens_at_compaction) == 0
            and int(e.tokens_evicted) > 0
        ]
        # Synthetic events are emitted whenever vLLM's prefix cache
        # returned a non-empty match. They carry nuf_len =
        # num_prompt - cached_tokens (= where V's prefill begins). The
        # admission event's nuf_len reflects something different (tokens
        # past last turn boundary marker), so we always prefer the
        # synthetic event's nuf_len. position_offset_after may be 0 (for
        # non-inheritor cache hits) or > 0 (inheritor); both forms encode
        # the same cached_tokens info via nuf_len.
        synthetic_events = [
            e for e in events
            if int(e.num_output_tokens_at_compaction) == 0
            and int(e.tokens_evicted) == 0
        ]
        synthetic_cached_tokens = None
        if synthetic_events:
            synthetic_cached_tokens = (
                int(synthetic_events[-1].num_prompt_tokens)
                - int(synthetic_events[-1].new_user_fragment_len)
            )

        # Determine where this call's NEW content starts in sub.
        # Call 0: starts at 0 (whole sub is new).
        # Extension call: this call's submitted_prompt_ids reconstructs
        # the prior calls' POST-TRIM content (= what V sees after
        # admissions) followed by completions and the new_user_fragment.
        # The orchestrator's stitching guarantees that sub[:cum_post]
        # matches the prior calls' POST-TRIM contribution (= cum_post is
        # the post-trim cumulative length we've accumulated so far). The
        # NEW content this call adds (in pre-trim) is sub[cum_post:] +
        # completion_ids. Prior calls' PRE-TRIM content (including any
        # tokens evicted by their admissions) is already in pre_trim_ids
        # from the prior call's append; we do NOT reproduce it here.
        dead_branches: list[tuple[int, int]] = []
        if call_idx == 0:
            new_content_start_in_sub = 0
            call_pre_start = 0
        else:
            # STREAM-FORK detection (managed-context manager/repair
            # protocol): a later prompt may DROP a synthetic exchange and
            # continue from the pre-fork branch — the engine then reuses
            # the surviving branch's prefix-cache blocks and the dropped
            # exchange's rows become a dead branch. Purely extensional
            # protocols (the eviction-only arm) never diverge and take
            # the unchanged fast path below.
            _exp = expected_v_cache_tokens
            _forks = 0
            while True:
                _check_len = min(cum_post, sub_len, len(_exp))
                _fork = -1
                for _i in range(_check_len):
                    if _exp[_i] != sub[_i]:
                        _fork = _i
                        break
                if _fork < 0:
                    break
                _forks += 1
                if _forks > 8:
                    raise ValueError(
                        f"call {call_idx}: more than 8 stream forks"
                    )
                # Smallest drop whose removal realigns the streams; the
                # full-tail equality IS the acceptance criterion (a
                # first-token prefilter keeps the scan cheap). The drop
                # may extend to the end of the expected stream (the whole
                # tail was abandoned).
                _drop = None
                _first = sub[_fork] if _fork < sub_len else None
                for _cand in range(1, len(_exp) - _fork + 1):
                    _tail_ok = (
                        _fork + _cand == len(_exp)
                        or _exp[_fork + _cand] == _first
                    )
                    if _tail_ok and _exp[_fork + _cand : cum_post] == sub[
                        _fork : cum_post - _cand
                    ]:
                        _drop = _cand
                        break
                if _drop is None:
                    # Mismatch that is NOT a dropped exchange: keep the
                    # legacy extension behavior (pre-fork code never
                    # required byte-equality here; synthetic/mock chains
                    # and legacy shapes land here). Only resolvable forks
                    # change anything.
                    import logging as _logging

                    _logging.getLogger(__name__).warning(
                        "call %d: prompt/stream mismatch at %d not "
                        "resolvable as a dropped fork; keeping extension "
                        "semantics",
                        call_idx,
                        _fork,
                    )
                    break
                dead_branches.append((_fork, _drop))
                del _exp[_fork : _fork + _drop]
                cum_post -= _drop
            # In the post-trim frame, this call's sub must start with the
            # orchestrator's accumulated prefix (= V's prev_kept_state).
            # Note: admissions in this call evict tokens from V's PRIOR
            # cache content (the prev_kept region of sub) BEFORE V
            # prefills the new content. After Phase A in the orchestrator,
            # the merged sample's POST-TRIM cumulative arrays already
            # reflect this admission deletion (extend_sample applied it).
            # So `sub[:cum_post]` matches the sample's accumulated
            # POST-TRIM content AT THIS CALL'S ENTRY — the stitching
            # invariant continues to hold call-locally.
            assert sub_len >= cum_post, (
                f"extension call {call_idx}: sub_len={sub_len} < "
                f"cum_post={cum_post} (orchestrator stitching invariant)"
            )
            new_content_start_in_sub = cum_post
            call_pre_start = cum_pre
            prefix_cache_keep_len = None
            prefix_replay_tokens: list[int] = []
            if (
                synthetic_cached_tokens is not None
                and synthetic_cached_tokens < cum_post
            ):
                prefix_cache_keep_len = synthetic_cached_tokens
                prefix_replay_tokens = sub[synthetic_cached_tokens:cum_post]

            # DIAG: token-equality check between orchestrator's sub[:cum_post]
            # (= what stitching SAYS V's cache holds) and trainer's replayed
            # V cache (= what V actually wrote). Run before any append so the
            # log surfaces the precise divergence position.
            if stitch_trace:
                import sys as _sys
                actual = sub[:cum_post]
                expected = expected_v_cache_tokens
                if expected != actual:
                    exp_len = len(expected)
                    act_len = len(actual)
                    min_len = min(exp_len, act_len)
                    first_diff = next(
                        (i for i in range(min_len) if expected[i] != actual[i]),
                        min_len,
                    )
                    lo = max(0, first_diff - 8)
                    hi = min(max(exp_len, act_len), first_diff + 8)
                    print(
                        f"[STITCH-MISMATCH] call={call_idx}  "
                        f"exp_len={exp_len}  act_len={act_len}  "
                        f"first_diff={first_diff}  "
                        f"expected[{lo}:{hi}]={expected[lo:hi]}  "
                        f"actual[{lo}:{hi}]={actual[lo:hi]}",
                        file=_sys.stderr, flush=True,
                    )
                else:
                    print(
                        f"[STITCH-OK] call={call_idx}  cum_post={cum_post}",
                        file=_sys.stderr, flush=True,
                    )
        if call_idx == 0:
            prefix_cache_keep_len = None
            prefix_replay_tokens = []

        # Append this call's new pre-trim content. For extension calls
        # following an admission, sub[cum_post:] is everything past the
        # accumulated POST-TRIM prefix (= new_user_fragment + any other
        # NEW tokens beyond prior completion). Auto-pad fillers (when
        # present) sit at the END of the contribution — V's auto-pad
        # forward wrote them last in its request, so the trainer must
        # too so persistent_cache layout matches V's.
        pre_trim_ids.extend(sub[new_content_start_in_sub:])
        pre_trim_ids.extend(comp)
        pre_trim_ids.extend(trailing_pad)
        call_pre_end = len(pre_trim_ids)
        contrib_pre_len = call_pre_end - call_pre_start

        # Determine inherited_end_pre = boundary where this call's NEW
        # content (= what V's prefill writes, NOT what was inherited
        # via prefix cache) begins.
        #
        # Inheritor (synthetic event present): cached_tokens =
        # sub_len - synthetic_nuf_len. The synthetic event is emitted
        # ONLY when get_computed_blocks returned inherited_offset > 0.
        # Its nuf_len = num_prompt - num_new_local_computed_tokens =
        # exact "new tokens past inheritance".
        #
        # Non-inheritor with admission (no synthetic event): the
        # admission event's nuf_len reflects "tokens past last turn
        # boundary" (= generation prompt marker length, typically ~4),
        # NOT the cached-tokens boundary. For this case V's prefix
        # cache may still match prior content (offset 0 chain). Without
        # the cached_tokens info in the wire, we conservatively place
        # the boundary at sub_len - admission_nuf_len (= just before
        # the generation prompt). This produces a wider warmup region
        # than V's actual cached_tokens, so warmup K at positions
        # [cached_tokens, sub_len - nuf) will differ from V's prefill
        # K at those positions (attention context includes the eventually-
        # evicted region). This is a known limitation; the bulk of
        # admissions (= inheritors) hit the synthetic-event path.
        if synthetic_events:
            nuf_len = int(synthetic_events[-1].new_user_fragment_len)
            inherited_end_in_sub = sub_len - nuf_len
        elif admission_events:
            nuf_len = int(admission_events[-1].new_user_fragment_len)
            inherited_end_in_sub = sub_len - nuf_len
        else:
            nuf_len = 0
            inherited_end_in_sub = sub_len  # no admission, no split
        if admission_events:
            if nuf_len <= 0:
                raise ValueError(
                    f"call {call_idx}: nuf_len must be > 0 on admission, "
                    f"got {nuf_len}"
                )
            if inherited_end_in_sub <= 0:
                raise ValueError(
                    f"call {call_idx}: inherited_end_in_sub="
                    f"{inherited_end_in_sub} (sub_len={sub_len}, "
                    f"nuf={nuf_len})"
                )
            inherited_end_pre = call_pre_start + inherited_end_in_sub
        else:
            inherited_end_pre = call_pre_start  # no warmup needed

        # Build splice list. event.evict_start is in the post-prior-iter
        # sub coord (composition matches `_apply_admission_trim`).
        #
        # Interpretation depends on call_idx:
        # - call 0: admission fires INTRA-call (V evicts mid-prefill).
        #   es is in this call's PRE-TRIM contribution coord; the eviction
        #   removes K rows [es, es+te) from the SAME call's prefill cache.
        # - call N>0: admission fires BETWEEN calls (V evicts from its
        #   live cache before prefilling new content). es is in CURRENT
        #   sub coord, which (for the prev_kept region) equals the cache
        #   index coord (= POST-TRIM cumulative through prior calls).
        #
        # Important (Phase A.3 reverted): the orchestrator does NOT
        # delete admission ranges from the merged sample's arrays, so
        # mb.input_ids is PRE-TRIM cumulative (= pre_trim_ids). Each
        # call's logits cover its full pre-trim contribution; loss is
        # computed in pre-trim cumulative (= cum_pre) coordinates.
        # `cum_post` is still tracked because it's what V uses to compute
        # `new_content_start_in_sub` for the next call's sub layout.
        splices: list[tuple[int, int]] = []
        # Per splice: (span_id, start, end) triples from the event's
        # archived_span_ids + archived_span_bounds, in the SAME pre-event
        # frame as evict_start. The timeline builder records each span's
        # writer rows at splice time so later restore events can re-open
        # visibility windows on exactly those rows.
        splice_spans: list[list[tuple[str, int, int]]] = []
        for ev in admission_events:
            es = int(ev.evict_start)
            te = int(ev.tokens_evicted)
            if es < 0 or te <= 0:
                raise ValueError(
                    f"call {call_idx}: bad event evict_start={es}, "
                    f"tokens_evicted={te}"
                )
            splices.append((es, te))
            sids = [
                str(x) for x in getattr(ev, "archived_span_ids", []) or []
            ]
            bounds = [
                int(x) for x in getattr(ev, "archived_span_bounds", []) or []
            ]
            spans_for_event: list[tuple[str, int, int]] = []
            if sids and len(bounds) == 2 * len(sids):
                for i, sid in enumerate(sids):
                    s, e_b = bounds[2 * i], bounds[2 * i + 1]
                    if 0 <= s < e_b:
                        spans_for_event.append((sid, s, e_b))
            splice_spans.append(spans_for_event)
        total_evicted = sum(te for (_, te) in splices)

        # Loss positions: pre-trim cumulative frame (matches mb.input_ids).
        # Auto-pad fillers were not sampled — exclude them from the loss
        # frame. The K-cache forward still WRITES K at the pad positions
        # (for layout match with V), but the resulting logits at those
        # positions are dropped before loss_fn is called.
        post_start = call_pre_start
        post_end = call_pre_end - pad_len
        # `kept_in_call` is the full pre-trim contribution minus trailing
        # pad; subsetting (for B.2a's two-phase warmup) is handled inside
        # per_call_segmented_forward via dummy padding at the evict gap.
        kept_in_call = list(range(contrib_pre_len - pad_len))

        if call_idx > 0 and splices:
            # Defensive: each event's es is in post-prior-iter cache coord.
            running_cache_len = cum_post
            for (es, te) in splices:
                assert es + te <= running_cache_len, (
                    f"call {call_idx}: admission [{es},{es+te}) extends past "
                    f"running cache length {running_cache_len}"
                )
                running_cache_len -= te

        # writer_offset: the inheritance offset of the WRITER whose K
        # this call inherits via prefix cache. For an inheritor (chain
        # cb_offset > 0), vLLM emits a synthetic event with
        # position_offset_after = inherited_offset. Trainer's warmup
        # must rotate K at the writer's frame (= physical + writer_offset
        # for post-sys slots) to match V's cached K. For first-generation
        # admissions (no synthetic, or synthetic with offset 0) this is 0.
        writer_offset = 0
        if synthetic_events:
            writer_offset = int(synthetic_events[-1].position_offset_after)
        # admission_offset_after: V's final position_offset after this
        # call's admission (= writer_offset + total_evicted, used by the
        # MAIN forward to rotate new prefill K at V's post-admission frame).
        admission_offset_after = (
            int(admission_events[-1].position_offset_after)
            if admission_events else 0
        )
        # evict_start of the admission boundary in sub coord (= block-
        # aligned sys boundary, the piecewise split point for warmup
        # positions when writer_offset > 0). For inheritor admissions
        # this is the same across all admission iterations.
        evict_start_in_sub = (
            int(admission_events[0].evict_start)
            if admission_events else 0
        )
        # The worker gates position_offset application at the protected
        # prefix boundary. Synthetic prefix-cache events carry this same
        # block-aligned boundary even on calls with no admission, so prefer
        # them whenever present.
        protected_prefix_len = (
            int(synthetic_events[-1].evict_start)
            if synthetic_events else evict_start_in_sub
        )

        # sub_end_pre: pre-trim position where this call's NEW sub
        # content ends and comp begins. V re-prefills every token in
        # [new_content_start_in_sub..sub_len) of sub before admission
        # fires, so for the B.2a warmup-then-splice path the warmup
        # must cover the FULL re-prefill region (not just up to
        # inherited_end_pre) — otherwise survivor K rows past
        # inherited_end_pre would be written under post-splice
        # attention context, and the splice itself can run off the
        # end of the warmup output when V's evict range extends
        # into the new_user_fragment.
        sub_end_pre = call_pre_start + (sub_len - new_content_start_in_sub)
        admission_nuf_len = (
            int(admission_events[-1].new_user_fragment_len)
            if admission_events else 0
        )
        b2b_warm_end_pre = sub_end_pre
        if call_idx > 0 and admission_events:
            b2b_warm_end_pre = call_pre_start + max(
                0,
                (sub_len - new_content_start_in_sub) - admission_nuf_len,
            )

        plans.append({
            "call_idx": call_idx,
            "call_pre_start": call_pre_start,
            "call_pre_end": call_pre_end,
            "inherited_end_pre": inherited_end_pre,
            "sub_end_pre": sub_end_pre,
            "b2b_warm_end_pre": b2b_warm_end_pre,
            "has_admission": len(admission_events) > 0,
            "splices": splices,  # [(es, te), ...] in CURRENT sub coord
            "dead_branches": dead_branches,  # [(start, len)] dropped forks
            "splice_spans": splice_spans,  # per splice: archived span bounds
            "visibility_ops": visibility_ops,  # [(kind, span_ids, boundary)]
            "recall_span_tokens": recall_span_tokens,  # sid -> (tokens, pos0)
            # Engine-reported request prompt length per event (diagnostics:
            # compare the engine's visible stream against this builder's).
            "event_prompt_lens": [
                int(getattr(e, "num_prompt_tokens", 0)) for e in all_events
            ],
            "post_start": post_start,
            "post_end": post_end,
            "kept_in_call": kept_in_call,  # pre-trim positions WITHIN
                                            # this call's contribution
                                            # that survive admission
            "new_content_start_in_sub": new_content_start_in_sub,
            "prefix_cache_keep_len": prefix_cache_keep_len,
            "prefix_replay_tokens": prefix_replay_tokens,
            "sub_len": sub_len,
            "comp_len": comp_len,
            "contrib_pre_len": contrib_pre_len,
            "nuf_len": admission_nuf_len,
            "writer_offset": writer_offset,
            "admission_offset_after": admission_offset_after,
            "evict_start_in_sub": evict_start_in_sub,
            "protected_prefix_len": protected_prefix_len,
            "pad_len": pad_len,
            "synthetic_num_prompt": (
                int(synthetic_events[-1].num_prompt_tokens)
                if synthetic_events else None
            ),
            "synthetic_nuf_len": (
                int(synthetic_events[-1].new_user_fragment_len)
                if synthetic_events else None
            ),
            "synthetic_cached_tokens": (
                int(synthetic_events[-1].num_prompt_tokens)
                - int(synthetic_events[-1].new_user_fragment_len)
                if synthetic_events else None
            ),
            "synthetic_offset_after": (
                int(synthetic_events[-1].position_offset_after)
                if synthetic_events else None
            ),
            "admission_num_prompt": (
                int(admission_events[-1].num_prompt_tokens)
                if admission_events else None
            ),
            "admission_nuf_len": (
                int(admission_events[-1].new_user_fragment_len)
                if admission_events else None
            ),
            "admission_total_evicted": total_evicted,
            "admission_event_offset_after": (
                int(admission_events[-1].position_offset_after)
                if admission_events else None
            ),
        })
        # ── DIAG: log per-call cache-state accounting vs V's wire events.
        # Compares trainer's cum_post (= predicted V cache state) against
        # V's reported cached_tokens (= num_prompt - synthetic.nuf_len) and
        # surfaces a per-call delta. The first call where delta != 0 is
        # where trainer and V have fallen out of sync.
        if os.environ.get("KVE_TRACE_CACHE_DELTA", "") == "1":
            v_cached_tokens = None
            if synthetic_events:
                v_np = int(synthetic_events[-1].num_prompt_tokens)
                v_nuf = int(synthetic_events[-1].new_user_fragment_len)
                v_cached_tokens = v_np - v_nuf
            cum_post_in = cum_post
            cum_post_out = cum_post - total_evicted + contrib_pre_len
            delta = (cum_post_in - v_cached_tokens
                     if v_cached_tokens is not None else None)
            logger.warning(
                "[CACHE-TRACE] call=%d  trainer_cum_post_in=%d  "
                "v_cached_tokens=%s  delta=%s  sub_len=%d  pad_len=%d  "
                "total_evicted=%d  contrib_pre_len=%d  cum_post_out=%d",
                call_idx, cum_post_in,
                str(v_cached_tokens), str(delta),
                sub_len, pad_len, total_evicted,
                contrib_pre_len, cum_post_out,
            )

        # DIAG: B.2a-specific trace. Catastrophic L1+ outliers in the
        # textworld testbed concentrate on single-call samples with
        # call 0 admission (B.2a path). Log enough to determine whether
        # `inherited_end_pre` matches V's actual cached_tokens, and
        # whether the splice/offset chain on this call is well-formed.
        if os.environ.get("KVE_TRACE_B2A", "") == "1" and call_idx == 0 and admission_events:
            import sys as _sys
            v_cached_tokens = None
            v_num_prompt = None
            v_synth_nuf = None
            if synthetic_events:
                v_num_prompt = int(synthetic_events[-1].num_prompt_tokens)
                v_synth_nuf = int(synthetic_events[-1].new_user_fragment_len)
                v_cached_tokens = v_num_prompt - v_synth_nuf
            adm_es = int(admission_events[0].evict_start)
            adm_te = int(admission_events[0].tokens_evicted)
            adm_nuf = int(admission_events[-1].new_user_fragment_len)
            adm_offset_after = int(admission_events[-1].position_offset_after)
            trainer_inherited_end = inherited_end_pre
            delta_inh_vs_cached = (
                trainer_inherited_end - v_cached_tokens
                if v_cached_tokens is not None else None
            )
            print(
                f"[B2A] sub_len={sub_len}  comp_len={comp_len}  "
                f"pad_len={pad_len}  "
                f"adm:(es={adm_es},te={adm_te},nuf={adm_nuf},off_aft={adm_offset_after})  "
                f"synth:({'yes' if synthetic_events else 'no'},"
                f"np={v_num_prompt},nuf={v_synth_nuf},cached={v_cached_tokens})  "
                f"trainer:(inh_end_pre={trainer_inherited_end},"
                f"sub_end_pre={sub_end_pre},"
                f"writer_off={writer_offset},"
                f"protected_pref={protected_prefix_len})  "
                f"delta(inh_end - cached)={delta_inh_vs_cached}",
                file=_sys.stderr, flush=True,
            )

        cum_pre = call_pre_end
        # cum_post tracks V's cumulative cache state after this call
        # (used by the NEXT call's plan to compute
        # new_content_start_in_sub). It evolves as:
        #   cum_post[N] = cum_post[N-1] - admission_te[N] + contrib_pre[N]
        cum_post = cum_post - total_evicted + contrib_pre_len

        # Replay this call's writes + splices into the running expected
        # V cache. Mirrors per_call_segmented_forward's actual B.2a /
        # B.2b / extension paths so the running token list stays
        # byte-equal to what V's cache holds at this call's end. Always
        # maintained: the stream-fork detector above diffs against it.
        if True:
            if call_idx == 0 and splices:
                # B.2a: warmup writes full sub_N, then splice removes
                # rows in pre-trim contribution coord, then main writes
                # comp + pad.
                expected_v_cache_tokens.extend(sub)
                for (es, te) in splices:
                    del expected_v_cache_tokens[es : es + te]
                expected_v_cache_tokens.extend(comp)
                expected_v_cache_tokens.extend(trailing_pad)
            elif splices:
                # B.2b: splice removes rows in cumulative cache coord,
                # then main writes sub[new_start:] + comp + pad.
                for (es, te) in splices:
                    del expected_v_cache_tokens[es : es + te]
                expected_v_cache_tokens.extend(
                    sub[new_content_start_in_sub:]
                )
                expected_v_cache_tokens.extend(comp)
                expected_v_cache_tokens.extend(trailing_pad)
            else:
                # Extension (no admission): plain append of new content.
                expected_v_cache_tokens.extend(
                    sub[new_content_start_in_sub:]
                )
                expected_v_cache_tokens.extend(comp)
                expected_v_cache_tokens.extend(trailing_pad)
            assert len(expected_v_cache_tokens) == cum_post, (
                f"[STITCH-INTERNAL] call={call_idx} replay length="
                f"{len(expected_v_cache_tokens)} != cum_post={cum_post}"
            )
    return plans, pre_trim_ids


def _position_id_list(
    start: int,
    end: int,
    *,
    offset: int = 0,
    protected_prefix_len: int = 0,
) -> list[int]:
    """Python-list equivalent of _piecewise_positions."""
    positions = list(range(start, end))
    if offset == 0:
        return positions
    if protected_prefix_len > 0:
        return [
            pos + (offset if pos >= protected_prefix_len else 0)
            for pos in positions
        ]
    return [pos + offset for pos in positions]


def _build_flex_mask_writer_timeline(calls) -> _FlexMaskWriterTimeline:
    """Build a single forward timeline plus per-key liveness metadata.

    Each element of the returned timeline is one K/V write in the order
    vLLM would have produced it. Eviction is represented by death_indices:
    key row k is visible to query q iff k <= q < death_indices[k].
    """
    plans, pre_trim_ids_py = _build_pre_trim_plan(calls)

    writer_input_ids: list[int] = []
    writer_position_ids: list[int] = []
    death_indices: list[int | None] = []
    live_writer_indices: list[int] = []
    loss_ranges: list[tuple[int, int, int, int]] = []

    def _append_write(
        pre_start: int,
        pre_end: int,
        position_ids: list[int],
    ) -> tuple[int, int]:
        if pre_end < pre_start:
            raise ValueError(
                f"bad write range [{pre_start}, {pre_end})"
            )
        n_tokens = pre_end - pre_start
        if n_tokens != len(position_ids):
            raise ValueError(
                "position count does not match write range: "
                f"range=[{pre_start},{pre_end}) positions={len(position_ids)}"
            )
        writer_start = len(writer_input_ids)
        writer_input_ids.extend(pre_trim_ids_py[pre_start:pre_end])
        writer_position_ids.extend(position_ids)
        death_indices.extend([None] * n_tokens)
        live_writer_indices.extend(range(writer_start, writer_start + n_tokens))
        return writer_start, writer_start + n_tokens

    def _append_tokens(
        token_ids: list[int],
        position_ids: list[int],
    ) -> tuple[int, int]:
        n_tokens = len(token_ids)
        if n_tokens != len(position_ids):
            raise ValueError(
                "position count does not match token count: "
                f"tokens={n_tokens} positions={len(position_ids)}"
            )
        writer_start = len(writer_input_ids)
        writer_input_ids.extend(token_ids)
        writer_position_ids.extend(position_ids)
        death_indices.extend([None] * n_tokens)
        live_writer_indices.extend(range(writer_start, writer_start + n_tokens))
        return writer_start, writer_start + n_tokens

    def _append_hidden_span(
        token_ids: list[int],
        position_ids: list[int],
    ) -> tuple[int, int]:
        """Append a recalled span's tokens that were evicted before this
        sample's first captured turn (recall token-surfacing).

        Unlike _append_tokens, these rows are NOT added to live_writer_indices
        — the span is hidden KV, never part of the ongoing physical frame
        (so it doesn't perturb later calls' position math). They're marked
        dead at the row right after the span (so the death-only mask hides
        them from every later query); the restore ATTACH re-opens visibility
        for exactly the queries the sampler let attend (deferred boundary).
        """
        n_tokens = len(token_ids)
        if n_tokens != len(position_ids):
            raise ValueError(
                "hidden-span position count mismatch: "
                f"tokens={n_tokens} positions={len(position_ids)}"
            )
        writer_start = len(writer_input_ids)
        writer_input_ids.extend(token_ids)
        writer_position_ids.extend(position_ids)
        death_indices.extend([None] * n_tokens)
        writer_end = writer_start + n_tokens
        for r in range(writer_start, writer_end):
            death_indices[r] = writer_end
        return writer_start, writer_end

    # Managed-context hidden recall bookkeeping. span_rows maps an archived
    # span id to the writer rows that died when it was archived; restore
    # events re-open visibility windows on exactly those rows.
    span_rows: dict[str, list[int]] = {}
    open_intervals: dict[str, int] = {}
    extra_intervals: list[tuple[int, int, int]] = []

    def _splice_live_cache(
        evict_start: int,
        tokens_evicted: int,
        call_idx: int,
        spans: list[tuple[str, int, int]] = (),
    ) -> None:
        if tokens_evicted <= 0:
            return
        evict_end = evict_start + tokens_evicted
        if evict_start < 0 or evict_end > len(live_writer_indices):
            raise ValueError(
                f"call {call_idx}: flex-mask splice [{evict_start},{evict_end}) "
                f"out of live cache len={len(live_writer_indices)}"
            )
        for sid, s, e in spans:
            if s < evict_start or e > evict_end:
                raise ValueError(
                    f"call {call_idx}: archived span {sid} bounds [{s},{e}) "
                    f"outside its event's evict range "
                    f"[{evict_start},{evict_end})"
                )
            span_rows[sid] = list(live_writer_indices[s:e])
        death_idx = len(writer_input_ids)
        for writer_idx in live_writer_indices[evict_start:evict_end]:
            death_indices[writer_idx] = death_idx
        del live_writer_indices[evict_start:evict_end]

    def _process_visibility_ops(
        ops: list[tuple[int, list[str], int]],
        call_idx: int,
        call_start: int,
    ) -> None:
        """Resolve a call's restore attach/release ops to writer intervals.

        Boundaries are num_computed_tokens in the request's post-trim
        current frame — live_writer_indices IS that frame, so the mapping
        is a direct index. Two scope rules:
        - Boundaries landing in the prefix-cached region clamp UP to the
          call's first appended row (call_start): rows before it had their
          queries run by earlier requests, so visibility cannot apply
          retroactively (an upfront attach has boundary 0 = "everything
          this request computes", which is exactly the call's own rows).
        - Visibility is per-request: any interval still open at the end of
          the call closes at the call boundary (a later call sees the
          spans only if it attaches them itself).
        """

        def _boundary_to_writer(boundary: int) -> int:
            if boundary < len(live_writer_indices):
                w = live_writer_indices[boundary]
            elif boundary == len(live_writer_indices):
                w = len(writer_input_ids)
            else:
                raise ValueError(
                    f"call {call_idx}: visibility boundary {boundary} "
                    f"beyond live cache len={len(live_writer_indices)} "
                    f"(op kind={op_kind} spans={op_span_ids} "
                    f"call_start={call_start} "
                    f"writer_len={len(writer_input_ids)} "
                    f"all_ops={[(k, s, b) for k, s, b in ops]})"
                )
            return max(w, call_start)

        # Spans recalled here but whose ARCHIVE event isn't in this sample
        # (e.g. the first recall handshake's manager-pass archive landed in
        # the manager-pass sample across the recall-point split — see
        # KVE_RECALL_SKIP_UNARCHIVED). We can't reproduce their hidden-KV
        # recall in this sample (the rows were never kept), so skip their
        # visibility instead of hard-crashing. The recalled tokens then go
        # unattended → contributes to an off-floor mismatch KL that correctly
        # flags the unaligned recall (the floor-fix is the flex full-BPTT
        # recall-geometry alignment, not this builder).
        _skip_unarchived = os.environ.get("KVE_RECALL_SKIP_UNARCHIVED", "1") != "0"
        skipped_spans: set[str] = set()
        for kind, span_ids, boundary in ops:
            op_kind, op_span_ids = kind, span_ids
            w = _boundary_to_writer(boundary)
            for sid in span_ids:
                if sid not in span_rows:
                    if not _skip_unarchived:
                        raise ValueError(
                            f"call {call_idx}: restore event references span "
                            f"{sid} with no archived rows in this sample's "
                            "event history"
                        )
                    if sid not in skipped_spans:
                        skipped_spans.add(sid)
                        logger.warning(
                            "[RECALL-SKIP] call %d: span %s recalled but not "
                            "archived in this sample; skipping its visibility "
                            "(off-floor KL expected — flex-alignment fix needed)",
                            call_idx, sid,
                        )
                    continue
                if kind == 1:
                    open_intervals[sid] = w
                else:
                    birth = open_intervals.pop(sid, None)
                    if birth is None:
                        if sid in skipped_spans:
                            continue
                        raise ValueError(
                            f"call {call_idx}: release for span {sid} "
                            "without a preceding attach in the same call "
                            "(cross-call visibility is a session-arm "
                            "shape this builder does not support yet)"
                        )
                    if w > birth:
                        for row in span_rows[sid]:
                            extra_intervals.append((row, birth, w))
        call_end = len(writer_input_ids)
        for sid, birth in open_intervals.items():
            if call_end > birth:
                for row in span_rows[sid]:
                    extra_intervals.append((row, birth, call_end))
        open_intervals.clear()

    _timeline_diag = os.environ.get("KVE_FLEX_TIMELINE_DIAG") == "1"
    for plan in plans:
        call_idx = int(plan["call_idx"])
        call_start = len(writer_input_ids)
        cps = int(plan["call_pre_start"])
        cpe = int(plan["call_pre_end"])
        if _timeline_diag:
            import sys as _sys
            print(
                f"[FLEX-TIMELINE] call={call_idx} pre=[{cps},{cpe}) "
                f"live_before={len(live_writer_indices)} "
                f"writer_before={len(writer_input_ids)} "
                f"admission={plan.get('has_admission')} "
                f"splices={plan.get('splices')} "
                f"ops={plan.get('visibility_ops')} "
                f"engine_prompt_lens={plan.get('event_prompt_lens')}",
                file=_sys.stderr,
                flush=True,
            )
        if cpe <= cps:
            continue

        has_admission = bool(plan["has_admission"])
        splices = list(plan["splices"])
        splice_spans = list(plan.get("splice_spans") or [[] for _ in splices])
        visibility_ops = list(plan.get("visibility_ops") or [])
        recall_span_tokens = dict(plan.get("recall_span_tokens") or {})
        # Abandoned forks (e.g. a dropped manager-repair exchange): the
        # engine's request continued from the pre-fork branch's prefix
        # cache, so the dropped rows stop being served from this call on.
        # Killing them BEFORE this call's appends restores
        # global-live == request-visible, which is what makes the
        # position frames agree (M2 root cause #3).
        for _db_start, _db_len in plan.get("dead_branches") or []:
            _splice_live_cache(int(_db_start), int(_db_len), call_idx)
        post_start = int(plan["post_start"])
        post_end = int(plan["post_end"])
        writer_offset = int(plan["writer_offset"])
        admission_offset_after = int(plan["admission_offset_after"])
        protected_prefix_len = int(
            plan.get("protected_prefix_len", plan["evict_start_in_sub"])
        )
        sub_end_pre = int(plan.get("sub_end_pre", plan["inherited_end_pre"]))
        b2b_warm_end_pre = int(plan.get("b2b_warm_end_pre", sub_end_pre))
        prefix_cache_keep_len = plan.get("prefix_cache_keep_len")
        prefix_replay_tokens = list(plan.get("prefix_replay_tokens", []) or [])

        loss_len = post_end - post_start
        if loss_len < 0:
            raise ValueError(
                f"call {call_idx}: negative flex-mask loss length "
                f"post=[{post_start},{post_end})"
            )

        # Recall token-surfacing reconstruction: a span recalled on this call
        # whose birth turn is not in this sample (it was evicted on a
        # side-channel handshake before this sample's first captured turn).
        # Insert its surfaced tokens right after the protected prefix so the
        # writer order is sys < span < visible — the span's KV is then
        # computed attending only to sys and itself (never the later visible
        # turns), exactly as the sampler computed it. The rows are hidden
        # (not live, so later calls' position math is untouched) and
        # immediately dead; the restore ATTACH re-opens them for the queries
        # the sampler let attend. Self-contained: handles the whole call.
        recon = [
            (sid, recall_span_tokens[sid])
            for (k, sids, b) in visibility_ops
            if k == 1
            for sid in sids
            if sid not in span_rows and sid in recall_span_tokens
        ]
        if recon:
            ppl = protected_prefix_len
            real_splice = any(int(te) > 0 for (_es, te) in splices)
            if (
                prefix_cache_keep_len is not None
                or real_splice
                or not (0 < ppl <= loss_len)
            ):
                logger.warning(
                    "[RECALL-RECON] call %d: cannot reconstruct surfaced "
                    "span(s) %s (prefix_cache=%s real_splice=%s ppl=%d "
                    "loss_len=%d); falling back to skip",
                    call_idx, [sid for sid, _ in recon],
                    prefix_cache_keep_len is not None, real_splice, ppl,
                    loss_len,
                )
            else:
                writer_start = len(writer_input_ids)
                if writer_offset != 0:
                    positions = _position_id_list(
                        cps, cpe, offset=writer_offset,
                        protected_prefix_len=protected_prefix_len,
                    )
                else:
                    positions = _position_id_list(cps, cpe)
                _append_write(cps, cps + ppl, positions[:ppl])
                for sid, (tok, pos0) in recon:
                    hs_start, hs_end = _append_hidden_span(
                        tok, list(range(pos0, pos0 + len(tok)))
                    )
                    span_rows[sid] = list(range(hs_start, hs_end))
                    expected = pos0 + len(tok)
                    if positions[ppl] != expected:
                        logger.warning(
                            "[RECALL-RECON] call %d span %s: position seam "
                            "mismatch (span ends at pos %d, visible resumes "
                            "at pos %d; writer_offset=%d span_len=%d) -- "
                            "recalled KV rotation will not match the sampler; "
                            "KL stays off floor until writer_offset==span_len",
                            call_idx, sid, expected, positions[ppl],
                            writer_offset, len(tok),
                        )
                rest_writer_start = len(writer_input_ids)
                _append_write(cps + ppl, cpe, positions[ppl:])
                loss_ranges.append(
                    (writer_start, writer_start + ppl,
                     post_start, post_start + ppl)
                )
                loss_ranges.append(
                    (rest_writer_start, rest_writer_start + (loss_len - ppl),
                     post_start + ppl, post_end)
                )
                _process_visibility_ops(visibility_ops, call_idx, call_start)
                continue

        if prefix_cache_keep_len is not None:
            keep_len = int(prefix_cache_keep_len)
            if keep_len < 0 or keep_len > len(live_writer_indices):
                raise ValueError(
                    f"call {call_idx}: prefix keep len {keep_len} out of "
                    f"live cache len={len(live_writer_indices)}"
                )
            death_idx = len(writer_input_ids)
            for writer_idx in live_writer_indices[keep_len:]:
                death_indices[writer_idx] = death_idx
            del live_writer_indices[keep_len:]
            if prefix_replay_tokens:
                if writer_offset != 0:
                    positions = _position_id_list(
                        keep_len,
                        keep_len + len(prefix_replay_tokens),
                        offset=writer_offset,
                        protected_prefix_len=protected_prefix_len,
                    )
                else:
                    positions = _position_id_list(
                        keep_len,
                        keep_len + len(prefix_replay_tokens),
                    )
                _append_tokens(prefix_replay_tokens, positions)

        if has_admission and cps > 0:
            if prefix_replay_tokens:
                writer_start = len(writer_input_ids)
                warm_end = b2b_warm_end_pre
                if warm_end > cps:
                    physical_start = len(live_writer_indices)
                    if writer_offset != 0:
                        warm_positions = _position_id_list(
                            physical_start,
                            physical_start + (warm_end - cps),
                            offset=writer_offset,
                            protected_prefix_len=protected_prefix_len,
                        )
                    else:
                        warm_positions = _position_id_list(
                            physical_start,
                            physical_start + (warm_end - cps),
                        )
                    _append_write(cps, warm_end, warm_positions)

                for (es, te), sp in zip(splices, splice_spans):
                    _splice_live_cache(int(es), int(te), call_idx, sp)

                main_start = warm_end
                if cpe > main_start:
                    physical_start = len(live_writer_indices)
                    if admission_offset_after != 0:
                        main_positions = _position_id_list(
                            physical_start,
                            physical_start + (cpe - main_start),
                            offset=admission_offset_after,
                            protected_prefix_len=protected_prefix_len,
                        )
                    else:
                        main_positions = _position_id_list(
                            physical_start,
                            physical_start + (cpe - main_start),
                        )
                    _append_write(main_start, cpe, main_positions)
                loss_ranges.append(
                    (writer_start, writer_start + loss_len, post_start, post_end)
                )
                _process_visibility_ops(visibility_ops, call_idx, call_start)
                continue

            # B.2b default path: admission splices prior live cache first,
            # then this call writes its new pre-trim contribution.
            for (es, te), sp in zip(splices, splice_spans):
                _splice_live_cache(int(es), int(te), call_idx, sp)

            writer_start = len(writer_input_ids)
            if admission_offset_after != 0:
                physical_start = len(live_writer_indices)
                positions = _position_id_list(
                    physical_start,
                    physical_start + (cpe - cps),
                    offset=admission_offset_after,
                    protected_prefix_len=protected_prefix_len,
                )
            else:
                positions = _position_id_list(cps, cpe)
            _append_write(cps, cpe, positions)
            loss_ranges.append(
                (writer_start, writer_start + loss_len, post_start, post_end)
            )
            _process_visibility_ops(visibility_ops, call_idx, call_start)
            continue

        if has_admission:
            # B.2a: write the full submitted prompt under pre-admission
            # context, mark evicted rows dead, then write completion/pad
            # rows under post-admission context.
            writer_start = len(writer_input_ids)
            warm_end = sub_end_pre
            if warm_end > cps:
                if writer_offset != 0:
                    warm_positions = _position_id_list(
                        cps,
                        warm_end,
                        offset=writer_offset,
                        protected_prefix_len=protected_prefix_len,
                    )
                else:
                    warm_positions = _position_id_list(cps, warm_end)
                _append_write(cps, warm_end, warm_positions)

            for (es, te), sp in zip(splices, splice_spans):
                _splice_live_cache(int(es), int(te), call_idx, sp)

            main_start = sub_end_pre
            if cpe > main_start:
                if admission_offset_after != 0:
                    physical_start = len(live_writer_indices)
                    main_positions = _position_id_list(
                        physical_start,
                        physical_start + (cpe - main_start),
                        offset=admission_offset_after,
                        protected_prefix_len=protected_prefix_len,
                    )
                else:
                    main_positions = _position_id_list(main_start, cpe)
                _append_write(main_start, cpe, main_positions)

            loss_ranges.append(
                (writer_start, writer_start + loss_len, post_start, post_end)
            )
            _process_visibility_ops(visibility_ops, call_idx, call_start)
            continue

        # Extension or first call without admission.
        writer_start = len(writer_input_ids)
        if writer_offset != 0:
            physical_start = len(live_writer_indices)
            positions = _position_id_list(
                physical_start,
                physical_start + (cpe - cps),
                offset=writer_offset,
                protected_prefix_len=protected_prefix_len,
            )
        elif prefix_replay_tokens:
            physical_start = len(live_writer_indices)
            positions = _position_id_list(
                physical_start,
                physical_start + (cpe - cps),
            )
        else:
            positions = _position_id_list(cps, cpe)
        _append_write(cps, cpe, positions)
        loss_ranges.append(
            (writer_start, writer_start + loss_len, post_start, post_end)
        )
        _process_visibility_ops(visibility_ops, call_idx, call_start)

    final_len = len(writer_input_ids)
    resolved_deaths: list[int] = [
        final_len if death_idx is None else int(death_idx)
        for death_idx in death_indices
    ]
    for writer_idx, death_idx in enumerate(resolved_deaths):
        if death_idx <= writer_idx:
            raise ValueError(
                f"writer row {writer_idx} has invalid death_idx={death_idx}"
            )
    for row, birth, death in extra_intervals:
        # A re-opened window must start after the row's base interval ended
        # (the span was archived before any restore could attach it).
        if not (row < resolved_deaths[row] <= birth < death <= final_len):
            raise ValueError(
                f"writer row {row}: invalid extra visibility interval "
                f"[{birth},{death}) vs base death {resolved_deaths[row]} "
                f"and timeline len {final_len}"
            )

    return _FlexMaskWriterTimeline(
        input_ids=writer_input_ids,
        position_ids=writer_position_ids,
        death_indices=resolved_deaths,
        loss_ranges=loss_ranges,
        extra_intervals=extra_intervals,
        span_rows=dict(span_rows),
    )


def _build_visibility_mask_mod(
    death_idx: Tensor,
    extra_intervals: list[tuple[int, int, int]],
    seq_len: int,
    device: torch.device,
):
    """Build the flex mask_mod for a writer timeline.

    Eviction-only timelines get the classic death-index mask. Timelines with
    managed-context recall additionally OR in K slot tensors of re-opened
    visibility windows: row kv is visible to query q iff
    (kv <= q < death[kv]) or any_k(birth_k[kv] <= q < death_k[kv]).
    Unused slots encode birth > death so they can never match.
    """
    if not extra_intervals:

        def _death_only_mask(_batch_idx, _head_idx, q_idx, kv_idx):
            return (kv_idx <= q_idx) & (q_idx < death_idx[kv_idx])

        return _death_only_mask

    per_row: dict[int, list[tuple[int, int]]] = {}
    for row, birth, death in extra_intervals:
        per_row.setdefault(row, []).append((birth, death))
    k_slots = max(len(v) for v in per_row.values())
    birth_extra = torch.full(
        (k_slots, seq_len), seq_len + 1, device=device, dtype=torch.long
    )
    death_extra = torch.zeros(
        (k_slots, seq_len), device=device, dtype=torch.long
    )
    for row, intervals in per_row.items():
        for k, (birth, death) in enumerate(sorted(intervals)):
            birth_extra[k, row] = birth
            death_extra[k, row] = death

    def _interval_mask(_batch_idx, _head_idx, q_idx, kv_idx):
        live = (kv_idx <= q_idx) & (q_idx < death_idx[kv_idx])
        for k in range(k_slots):
            live = live | (
                (birth_extra[k][kv_idx] <= q_idx)
                & (q_idx < death_extra[k][kv_idx])
            )
        return live

    return _interval_mask


def _flex_kernel_options_from_env() -> dict | None:
    """Optional torch flex_attention kernel_options for diagnostics.

    Production defaults to PyTorch's choices. These env vars let us probe
    numerical and speed sensitivity without changing config surfaces.
    """
    options: dict[str, int | bool | str] = {}
    for env_name, opt_name in (
        ("KVE_FLEX_BLOCK_M", "BLOCK_M"),
        ("KVE_FLEX_BLOCK_N", "BLOCK_N"),
        ("KVE_FLEX_BLOCK_M1", "BLOCK_M1"),
        ("KVE_FLEX_BLOCK_N1", "BLOCK_N1"),
        ("KVE_FLEX_BLOCK_M2", "BLOCK_M2"),
        ("KVE_FLEX_BLOCK_N2", "BLOCK_N2"),
        ("KVE_FLEX_NUM_WARPS", "num_warps"),
        ("KVE_FLEX_NUM_STAGES", "num_stages"),
    ):
        val = os.environ.get(env_name, "")
        if val:
            options[opt_name] = int(val)
    for env_name, opt_name in (
        ("KVE_FLEX_PRESCALE_QK", "PRESCALE_QK"),
        ("KVE_FLEX_ROWS_GUARANTEED_SAFE", "ROWS_GUARANTEED_SAFE"),
        ("KVE_FLEX_BLOCKS_ARE_CONTIGUOUS", "BLOCKS_ARE_CONTIGUOUS"),
        ("KVE_FLEX_FORCE_USE_FLEX_ATTENTION", "FORCE_USE_FLEX_ATTENTION"),
    ):
        val = os.environ.get(env_name, "")
        if val:
            options[opt_name] = val.lower() not in {"0", "false", "no", "off"}
    backend = os.environ.get("KVE_FLEX_BACKEND", "")
    if backend:
        options["BACKEND"] = backend
    return options or None


def _build_selected_flex_writer_inputs(
    timelines: list[_FlexMaskWriterTimeline],
    *,
    mode: str,
    device: torch.device,
) -> tuple[Tensor, Tensor, object, list[int], list[int]]:
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:  # pragma: no cover - depends on torch build
        raise RuntimeError(
            "selected-logprob flex path requires torch flex_attention"
        ) from exc

    if any(not timeline.input_ids for timeline in timelines):
        raise ValueError("selected-logprob flex writer timeline is empty")

    seq_lens = [len(timeline.input_ids) for timeline in timelines]

    if mode == "single":
        if len(timelines) != 1:
            raise ValueError("single selected-logprob flex path expects one timeline")
        timeline = timelines[0]
        seq_len = seq_lens[0]
        writer_ids = torch.tensor(
            timeline.input_ids, device=device, dtype=torch.long
        ).unsqueeze(0)
        writer_positions = torch.tensor(
            timeline.position_ids, device=device, dtype=torch.long
        ).unsqueeze(0)
        death_idx = torch.tensor(
            timeline.death_indices, device=device, dtype=torch.long
        )
        # Reuse the single-sample interval mask builder: death-only when the
        # timeline has no recall windows, OR-in the re-opened intervals when
        # it does (managed-context recall).
        _mask_mod = _build_visibility_mask_mod(
            death_idx, timeline.extra_intervals, seq_len, device
        )
        block_mask = create_block_mask(
            _mask_mod,
            B=1,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
            _compile=True,
        )
        return writer_ids, writer_positions, block_mask, seq_lens, [0]

    if mode == "vertical":
        batch_size = len(timelines)
        max_seq_len = max(seq_lens)
        writer_ids = torch.full(
            (batch_size, max_seq_len),
            1,
            device=device,
            dtype=torch.long,
        )
        writer_positions = torch.zeros(
            (batch_size, max_seq_len),
            device=device,
            dtype=torch.long,
        )
        death_idx = torch.zeros(
            (batch_size, max_seq_len),
            device=device,
            dtype=torch.long,
        )
        for row, timeline in enumerate(timelines):
            row_len = len(timeline.input_ids)
            writer_ids[row, :row_len] = torch.tensor(
                timeline.input_ids, device=device, dtype=torch.long
            )
            writer_positions[row, :row_len] = torch.tensor(
                timeline.position_ids, device=device, dtype=torch.long
            )
            death_idx[row, :row_len] = torch.tensor(
                timeline.death_indices, device=device, dtype=torch.long
            )
            if row_len < max_seq_len:
                writer_positions[row, row_len:] = torch.arange(
                    max_seq_len - row_len,
                    device=device,
                    dtype=torch.long,
                )

        seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.long)

        # Managed-context recall: per-(sample,row) re-opened visibility windows
        # (row kv visible to query q iff (kv<=q<death) OR any open interval).
        # k_slots=0 (no recall) leaves the loop empty -> identical to the plain
        # death-index mask, so eviction-only / non-recall batches are untouched.
        per_rc: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for _b, _timeline in enumerate(timelines):
            for _r, _birth, _death in _timeline.extra_intervals:
                per_rc.setdefault((_b, _r), []).append((_birth, _death))
        k_slots = max((len(v) for v in per_rc.values()), default=0)
        birth_extra = torch.full(
            (max(k_slots, 1), batch_size, max_seq_len),
            max_seq_len + 1, device=device, dtype=torch.long,
        )
        death_extra = torch.zeros(
            (max(k_slots, 1), batch_size, max_seq_len),
            device=device, dtype=torch.long,
        )
        for (_b, _r), _intervals in per_rc.items():
            for _k, (_birth, _death) in enumerate(sorted(_intervals)):
                birth_extra[_k, _b, _r] = _birth
                death_extra[_k, _b, _r] = _death

        def _live_cache_mask(batch_idx, _head_idx, q_idx, kv_idx):
            row_len = seq_lens_t[batch_idx]
            valid_q = q_idx < row_len
            valid_kv = kv_idx < row_len
            live = (
                valid_q
                & valid_kv
                & (kv_idx <= q_idx)
                & (q_idx < death_idx[batch_idx, kv_idx])
            )
            for _k in range(k_slots):
                live = live | (
                    valid_q
                    & valid_kv
                    & (birth_extra[_k, batch_idx, kv_idx] <= q_idx)
                    & (q_idx < death_extra[_k, batch_idx, kv_idx])
                )
            pad_self = (~valid_q) & (q_idx == kv_idx)
            return live | pad_self

        block_mask = create_block_mask(
            _live_cache_mask,
            B=batch_size,
            H=None,
            Q_LEN=max_seq_len,
            KV_LEN=max_seq_len,
            device=device,
            _compile=True,
        )
        return writer_ids, writer_positions, block_mask, seq_lens, [0] * batch_size

    if mode == "horizontal":
        writer_offsets: list[int] = []
        total_seq_len = 0
        for row_len in seq_lens:
            writer_offsets.append(total_seq_len)
            total_seq_len += row_len

        writer_ids = torch.empty(
            (1, total_seq_len), device=device, dtype=torch.long
        )
        writer_positions = torch.empty(
            (1, total_seq_len), device=device, dtype=torch.long
        )
        sample_id = torch.empty((total_seq_len,), device=device, dtype=torch.long)
        death_idx = torch.empty((total_seq_len,), device=device, dtype=torch.long)

        for row, (offset, timeline) in enumerate(
            zip(writer_offsets, timelines, strict=True)
        ):
            row_len = len(timeline.input_ids)
            row_slice = slice(offset, offset + row_len)
            writer_ids[0, row_slice] = torch.tensor(
                timeline.input_ids, device=device, dtype=torch.long
            )
            writer_positions[0, row_slice] = torch.tensor(
                timeline.position_ids, device=device, dtype=torch.long
            )
            sample_id[row_slice] = row
            death_idx[row_slice] = offset + torch.tensor(
                timeline.death_indices, device=device, dtype=torch.long
            )

        def _live_cache_mask(_batch_idx, _head_idx, q_idx, kv_idx):
            same_sample = sample_id[q_idx] == sample_id[kv_idx]
            return same_sample & (kv_idx <= q_idx) & (q_idx < death_idx[kv_idx])

        block_mask = create_block_mask(
            _live_cache_mask,
            B=1,
            H=None,
            Q_LEN=total_seq_len,
            KV_LEN=total_seq_len,
            device=device,
            _compile=True,
        )
        return writer_ids, writer_positions, block_mask, seq_lens, writer_offsets

    raise ValueError(f"unknown selected-logprob flex mode {mode!r}")


def _selected_logprob_flex_mask_segmented_forward_impl(
    model: torch.nn.Module,
    calls_batch: list[list],
    merged_input_ids: Tensor,
    merged_position_ids: Tensor,
    inputs_fn: SegmentSelectedLogprobInputsFn,
    loss_fn: SegmentSelectedLogprobLossFn,
    *,
    mode: str,
    max_forward_passes: int | None = None,
    device: torch.device,
    backward: bool = True,
) -> dict:
    if not calls_batch:
        raise ValueError("selected-logprob flex path requires calls")
    if any(not calls for calls in calls_batch):
        raise ValueError("each selected-logprob flex sample requires non-empty calls")

    timing_enabled = _KVE_FLEX_TIMING
    timings: dict[str, float] = {}
    timing_last = _timing_now(device) if timing_enabled else 0.0

    def _lap(name: str) -> None:
        nonlocal timing_last
        if not timing_enabled:
            return
        now = _timing_now(device)
        timings[name] = timings.get(name, 0.0) + (now - timing_last)
        timing_last = now

    timelines = [_build_flex_mask_writer_timeline(calls) for calls in calls_batch]
    if mode == "horizontal":
        for _t_idx, _t in enumerate(timelines):
            if _t.extra_intervals:
                raise NotImplementedError(
                    f"sample {_t_idx}: managed-context recall intervals are "
                    "not yet supported on the 'horizontal' (packed) "
                    "selected-logprob flex path (plans/flex_full_bptt_"
                    "alignment.md M1) — it needs per-segment offsets on the "
                    "interval tensors. Use 'single' or 'vertical'."
                )
    _lap("build_timeline")
    writer_ids, writer_positions, block_mask, seq_lens_list, writer_offsets = (
        _build_selected_flex_writer_inputs(
            timelines,
            mode=mode,
            device=device,
        )
    )
    _lap("build_inputs_create_block_mask")

    writer_labels = writer_ids.clone()
    writer_temperatures = torch.ones(
        writer_ids.shape,
        device=device,
        dtype=torch.float32,
    )
    ranges: list[tuple[int, int, int, int, int]] = []
    selected_tokens = 0
    for row, timeline in enumerate(timelines):
        writer_offset = int(writer_offsets[row])
        for writer_start, writer_end, full_start, full_end in timeline.loss_ranges:
            if writer_end < writer_start:
                raise ValueError(
                    f"bad flex-mask writer loss range [{writer_start},{writer_end})"
                )
            if writer_end == writer_start:
                continue
            selected_labels, selected_temperatures, effective_end = inputs_fn(
                int(full_start),
                int(full_end),
                row,
            )
            width = int(effective_end) - int(full_start)
            if width <= 0:
                continue
            if width > writer_end - writer_start:
                raise ValueError(
                    "selected-logprob effective segment is wider than the "
                    f"writer range: width={width}, writer_range="
                    f"[{writer_start},{writer_end})"
                )
            if mode == "vertical":
                writer_labels[
                    row : row + 1,
                    writer_start : writer_start + width,
                ] = selected_labels
                writer_temperatures[
                    row : row + 1,
                    writer_start : writer_start + width,
                ] = selected_temperatures
                ranges.append(
                    (writer_start, writer_start + width, row, full_start, effective_end)
                )
            else:
                packed_start = writer_offset + writer_start
                writer_labels[
                    :,
                    packed_start : packed_start + width,
                ] = selected_labels
                writer_temperatures[
                    :,
                    packed_start : packed_start + width,
                ] = selected_temperatures
                ranges.append(
                    (packed_start, packed_start + width, row, full_start, effective_end)
                )
            selected_tokens += width

    _set_attn_implementation_if_needed(model, "flex_attention")
    out = model(
        input_ids=writer_ids,
        position_ids=writer_positions,
        attention_mask=block_mask,
        labels=writer_labels,
        temperature=writer_temperatures,
        use_cache=False,
        kernel_options=_flex_kernel_options_from_env(),
    )
    if isinstance(out, dict):
        logprobs = out.get("logprobs")
        entropy = out.get("entropy")
    else:
        logprobs = getattr(out, "logprobs", None)
        entropy = getattr(out, "entropy", None)
    if logprobs is None or entropy is None:
        raise RuntimeError(
            "selected-logprob flex path requires the root model to return "
            "logprobs and entropy; set fused_lm_head_token_chunk_size to an "
            "integer or 'auto'."
        )
    _lap("model_forward_logprobs")

    window_loss: Tensor | None = None
    accumulated_loss = 0.0
    for selected_start, selected_end, row, full_start, effective_end in ranges:
        if mode == "vertical":
            seg_logprobs = logprobs[row : row + 1, selected_start:selected_end]
            seg_entropy = entropy[row : row + 1, selected_start:selected_end]
        else:
            seg_logprobs = logprobs[:, selected_start:selected_end]
            seg_entropy = entropy[:, selected_start:selected_end]
        loss_val = loss_fn(
            seg_logprobs,
            seg_entropy,
            full_start,
            effective_end,
            row,
        )
        accumulated_loss += float(loss_val.detach().item())
        window_loss = loss_val if window_loss is None else window_loss + loss_val
    if window_loss is None:
        window_loss = logprobs.float().mean() * 0.0
    _lap("loss_loop")

    if backward:
        window_loss.backward()
    _lap("backward")

    target_passes = 1 if max_forward_passes is None else int(max_forward_passes)
    if target_passes < 1:
        raise ValueError(f"max_forward_passes must be >= 1, got {target_passes}")
    if target_passes > 1:
        if not backward:
            raise ValueError("max_forward_passes > 1 requires backward=True")
        _run_dummy_passes_with_backward(
            model,
            merged_input_ids[:1],
            merged_position_ids[:1],
            target_passes - 1,
        )
        _lap("dummy_passes")

    if timing_enabled:
        useful_tokens = sum(seq_lens_list)
        padded_tokens = (
            useful_tokens
            if mode != "vertical"
            else len(seq_lens_list) * max(seq_lens_list)
        )
        timing_parts = " ".join(
            f"{name}={value:.3f}s" for name, value in timings.items()
        )
        memory_part = ""
        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            memory_part = (
                f" alloc={alloc:.3f}GB peak={peak:.3f}GB "
                f"reserved={reserved:.3f}GB"
            )
        logger.warning(
            "[FLEX-SELECTED-LOGPROB-TIMING] mode=%s batch=%d writer_lens=%s "
            "writer_tokens=%d/%d writer_pad=%.3fx loss_ranges=%d "
            "selected_tokens=%d logprobs_shape=%s logprobs_dtype=%s times=%s%s",
            mode,
            len(timelines),
            seq_lens_list,
            useful_tokens,
            padded_tokens,
            padded_tokens / max(useful_tokens, 1),
            sum(len(timeline.loss_ranges) for timeline in timelines),
            selected_tokens,
            tuple(logprobs.shape),
            str(logprobs.dtype),
            timing_parts,
            memory_part,
        )

    return {
        "loss": torch.tensor(accumulated_loss, device=device) if backward else window_loss,
        "n_segments": 1,
    }


def selected_logprob_flex_mask_segmented_forward(
    model: torch.nn.Module,
    calls: list,
    merged_input_ids: Tensor,
    merged_position_ids: Tensor,
    inputs_fn: SegmentSelectedLogprobInputsFn,
    loss_fn: SegmentSelectedLogprobLossFn,
    *,
    max_forward_passes: int | None = None,
    device: torch.device,
    backward: bool = True,
) -> dict:
    return _selected_logprob_flex_mask_segmented_forward_impl(
        model,
        [calls],
        merged_input_ids,
        merged_position_ids,
        inputs_fn,
        loss_fn,
        mode="single",
        max_forward_passes=max_forward_passes,
        device=device,
        backward=backward,
    )


def selected_logprob_batched_flex_mask_segmented_forward(
    model: torch.nn.Module,
    calls_batch: list[list],
    merged_input_ids: Tensor,
    merged_position_ids: Tensor,
    inputs_fn: SegmentSelectedLogprobInputsFn,
    loss_fn: SegmentSelectedLogprobLossFn,
    *,
    max_forward_passes: int | None = None,
    device: torch.device,
    backward: bool = True,
) -> dict:
    return _selected_logprob_flex_mask_segmented_forward_impl(
        model,
        calls_batch,
        merged_input_ids,
        merged_position_ids,
        inputs_fn,
        loss_fn,
        mode="vertical",
        max_forward_passes=max_forward_passes,
        device=device,
        backward=backward,
    )


def selected_logprob_packed_flex_mask_segmented_forward(
    model: torch.nn.Module,
    calls_batch: list[list],
    merged_input_ids: Tensor,
    merged_position_ids: Tensor,
    inputs_fn: SegmentSelectedLogprobInputsFn,
    loss_fn: SegmentSelectedLogprobLossFn,
    *,
    max_forward_passes: int | None = None,
    device: torch.device,
    backward: bool = True,
) -> dict:
    return _selected_logprob_flex_mask_segmented_forward_impl(
        model,
        calls_batch,
        merged_input_ids,
        merged_position_ids,
        inputs_fn,
        loss_fn,
        mode="horizontal",
        max_forward_passes=max_forward_passes,
        device=device,
        backward=backward,
    )


def compute_flex_mask_writer_len(calls) -> int:
    """Return the sequence length used by the masked FlexAttention writer path."""
    return len(_build_flex_mask_writer_timeline(calls).input_ids)


def flex_mask_segmented_forward(
    model: torch.nn.Module,
    calls: list,
    merged_input_ids: Tensor,
    merged_position_ids: Tensor,
    loss_fn: SegmentLossFn,
    *,
    max_forward_passes: int | None = None,
    distillation_fn: SegmentDistillationFn | None = None,
    distillation_prepare_fn: SegmentDistillationPrepareFn | None = None,
    distillation_teacher_fn: SegmentDistillationTeacherFn | None = None,
    distillation_loss_fn: SegmentDistillationLossFn | None = None,
    device: torch.device,
    backward: bool = True,
) -> dict:
    """Single-forward FlexAttention path for admission KV eviction.

    This path encodes vLLM's KV-cache liveness as a BlockMask and runs one
    full-chain forward. It is intentionally full-BPTT: the accumulated loss
    is backward'd once after all call ranges have been sliced from the packed
    logits.
    """
    if not calls:
        raise ValueError("flex_mask_segmented_forward requires non-empty calls")

    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:  # pragma: no cover - depends on torch build
        raise RuntimeError(
            "flex_mask_segmented_forward requires torch flex_attention"
        ) from exc

    timing_enabled = _KVE_FLEX_TIMING
    timings: dict[str, float] = {}
    timing_last = _timing_now(device) if timing_enabled else 0.0

    def _lap(name: str) -> None:
        nonlocal timing_last
        if not timing_enabled:
            return
        now = _timing_now(device)
        timings[name] = timings.get(name, 0.0) + (now - timing_last)
        timing_last = now

    timeline = _build_flex_mask_writer_timeline(calls)
    if not timeline.input_ids:
        raise ValueError("flex-mask writer timeline is empty")
    seq_len = len(timeline.input_ids)
    _lap("build_timeline")

    writer_ids = torch.tensor(
        timeline.input_ids, device=device, dtype=torch.long,
    ).unsqueeze(0)
    writer_positions = torch.tensor(
        timeline.position_ids, device=device, dtype=torch.long,
    ).unsqueeze(0)
    death_idx = torch.tensor(
        timeline.death_indices, device=device, dtype=torch.long,
    )
    seq_len = int(writer_ids.shape[1])
    _lap("build_inputs")

    _live_cache_mask = _build_visibility_mask_mod(
        death_idx, timeline.extra_intervals, seq_len, device
    )

    block_mask = create_block_mask(
        _live_cache_mask,
        B=1,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        _compile=True,
    )
    _lap("create_block_mask")

    _set_attn_implementation_if_needed(model, "flex_attention")

    if distillation_fn is not None and (
        distillation_prepare_fn is not None
        or distillation_teacher_fn is not None
        or distillation_loss_fn is not None
    ):
        raise ValueError(
            "Use either distillation_fn or the compact distillation hook set, not both."
        )

    if (distillation_prepare_fn is not None or distillation_teacher_fn is not None) and (
        distillation_loss_fn is None
    ):
        raise ValueError(
            "distillation_loss_fn is required when compact distillation hooks are used."
        )

    if distillation_prepare_fn is not None:
        with torch.no_grad():
            student_probe_logits = _model_forward_logits(
                model,
                input_ids=writer_ids,
                position_ids=writer_positions,
                attention_mask=block_mask,
                use_cache=False,
                kernel_options=_flex_kernel_options_from_env(),
            )
            for writer_start, writer_end, full_start, full_end in timeline.loss_ranges:
                if writer_end == writer_start:
                    continue
                distillation_prepare_fn(
                    student_probe_logits[:, writer_start:writer_end, :],
                    full_start,
                    full_end,
                )
            del student_probe_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _lap("distill_prepare")

    teacher_logits: Tensor | None = None
    if distillation_fn is not None:
        with torch.no_grad():
            teacher_logits = _model_forward_logits(
                model,
                input_ids=writer_ids,
                position_ids=writer_positions,
                attention_mask=None,
                use_cache=False,
                kernel_options=_flex_kernel_options_from_env(),
            ).detach().to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _lap("distill_teacher")
    elif distillation_teacher_fn is not None:
        with torch.no_grad():
            teacher_probe_logits = _model_forward_logits(
                model,
                input_ids=writer_ids,
                position_ids=writer_positions,
                attention_mask=None,
                use_cache=False,
                kernel_options=_flex_kernel_options_from_env(),
            )
            for writer_start, writer_end, full_start, full_end in timeline.loss_ranges:
                if writer_end == writer_start:
                    continue
                distillation_teacher_fn(
                    teacher_probe_logits[:, writer_start:writer_end, :],
                    full_start,
                    full_end,
                )
            del teacher_probe_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _lap("distill_teacher")

    logits = _model_forward_logits(
        model,
        input_ids=writer_ids,
        position_ids=writer_positions,
        attention_mask=block_mask,
        use_cache=False,
        kernel_options=_flex_kernel_options_from_env(),
    )
    _lap("model_forward_extract_logits")

    window_loss: Tensor | None = None
    accumulated_loss = 0.0
    for writer_start, writer_end, full_start, full_end in timeline.loss_ranges:
        if writer_end < writer_start:
            raise ValueError(
                f"bad flex-mask writer loss range [{writer_start},{writer_end})"
            )
        if writer_end > logits.shape[1]:
            raise ValueError(
                f"flex-mask writer range [{writer_start},{writer_end}) "
                f"exceeds logits length {logits.shape[1]}"
            )
        if writer_end == writer_start:
            continue
        student_slice = logits[:, writer_start:writer_end, :]
        distill_loss = None
        if distillation_fn is not None:
            assert teacher_logits is not None
            distill_loss = distillation_fn(
                student_slice,
                teacher_logits[:, writer_start:writer_end, :].to(
                    device=student_slice.device,
                    non_blocking=True,
                ),
                full_start,
                full_end,
            )
        elif distillation_loss_fn is not None:
            distill_loss = distillation_loss_fn(
                student_slice,
                full_start,
                full_end,
            )
        loss_val = loss_fn(student_slice, full_start, full_end)
        if distill_loss is not None:
            loss_val = loss_val + distill_loss
        accumulated_loss += float(loss_val.detach().item())
        window_loss = loss_val if window_loss is None else window_loss + loss_val
    _lap("loss_loop")

    if window_loss is None:
        window_loss = logits.float().mean() * 0.0
    if backward:
        window_loss.backward()
    _lap("backward")

    target_passes = 1 if max_forward_passes is None else int(max_forward_passes)
    if target_passes < 1:
        raise ValueError(f"max_forward_passes must be >= 1, got {target_passes}")
    if target_passes > 1:
        if not backward:
            raise ValueError("max_forward_passes > 1 requires backward=True")
        _run_dummy_passes_with_backward(
            model,
            merged_input_ids,
            merged_position_ids,
            target_passes - 1,
        )
        _lap("dummy_passes")

    if os.environ.get("KVE_TRACE_FLEX_MASK", "") == "1":
        logger.warning(
            "[FLEX-MASK] writer_len=%d loss_ranges=%d live_rows=%d",
            seq_len,
            len(timeline.loss_ranges),
            sum(1 for d in timeline.death_indices if d == seq_len),
        )

    if timing_enabled:
        timing_parts = " ".join(
            f"{name}={value:.3f}s" for name, value in timings.items()
        )
        memory_part = ""
        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            memory_part = (
                f" alloc={alloc:.3f}GB peak={peak:.3f}GB "
                f"reserved={reserved:.3f}GB"
            )
        logger.warning(
            "[FLEX-SINGLE-TIMING] batch=1 writer_lens=%s writer_tokens=%d/%d "
            "writer_pad=1.000x loss_ranges=%d logits_shape=%s logits_dtype=%s "
            "times=%s%s",
            [seq_len],
            seq_len,
            seq_len,
            len(timeline.loss_ranges),
            tuple(logits.shape),
            str(logits.dtype),
            timing_parts,
            memory_part,
        )

    return {
        "loss": torch.tensor(accumulated_loss, device=device) if backward else window_loss,
        "n_segments": 1,
    }


def batched_flex_mask_segmented_forward(
    model: torch.nn.Module,
    calls_batch: list[list],
    merged_input_ids: Tensor,
    merged_position_ids: Tensor,
    loss_fn,
    *,
    max_forward_passes: int | None = None,
    distillation_fn=None,
    distillation_prepare_fn=None,
    distillation_teacher_fn=None,
    distillation_loss_fn=None,
    device: torch.device,
    backward: bool = True,
) -> dict:
    """Batched masked FlexAttention path for admission KV eviction.

    Each row has its own writer timeline and KV-liveness mask. Rows are padded
    to the longest writer timeline in the group, and loss/distillation callbacks
    receive the row index so they can slice the matching full-sample tensors.
    """
    if not calls_batch:
        raise ValueError("batched_flex_mask_segmented_forward requires calls")
    if any(not calls for calls in calls_batch):
        raise ValueError("each batched flex-mask sample requires non-empty calls")

    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:  # pragma: no cover - depends on torch build
        raise RuntimeError(
            "batched_flex_mask_segmented_forward requires torch flex_attention"
        ) from exc

    timing_enabled = _KVE_FLEX_TIMING
    timings: dict[str, float] = {}
    timing_last = _timing_now(device) if timing_enabled else 0.0

    def _lap(name: str) -> None:
        nonlocal timing_last
        if not timing_enabled:
            return
        now = _timing_now(device)
        timings[name] = timings.get(name, 0.0) + (now - timing_last)
        timing_last = now

    timelines = [_build_flex_mask_writer_timeline(calls) for calls in calls_batch]
    for _t_idx, _t in enumerate(timelines):
        if _t.extra_intervals:
            raise NotImplementedError(
                f"sample {_t_idx}: managed-context recall intervals are only "
                "supported on the single-sample flex_mask_segmented_forward "
                "path for now (plans/flex_full_bptt_alignment.md M1); the "
                "batched/packed/selected variants need per-segment row "
                "offsets for the interval tensors."
            )
    if any(not timeline.input_ids for timeline in timelines):
        raise ValueError("batched flex-mask writer timeline is empty")

    batch_size = len(timelines)
    seq_lens_list = [len(timeline.input_ids) for timeline in timelines]
    max_seq_len = max(seq_lens_list)
    _lap("build_timeline")

    writer_ids = torch.full(
        (batch_size, max_seq_len),
        1,
        device=device,
        dtype=torch.long,
    )
    writer_positions = torch.zeros(
        (batch_size, max_seq_len),
        device=device,
        dtype=torch.long,
    )
    death_idx = torch.zeros(
        (batch_size, max_seq_len),
        device=device,
        dtype=torch.long,
    )
    for row, timeline in enumerate(timelines):
        row_len = len(timeline.input_ids)
        writer_ids[row, :row_len] = torch.tensor(
            timeline.input_ids,
            device=device,
            dtype=torch.long,
        )
        writer_positions[row, :row_len] = torch.tensor(
            timeline.position_ids,
            device=device,
            dtype=torch.long,
        )
        death_idx[row, :row_len] = torch.tensor(
            timeline.death_indices,
            device=device,
            dtype=torch.long,
        )
        if row_len < max_seq_len:
            writer_positions[row, row_len:] = torch.arange(
                max_seq_len - row_len,
                device=device,
                dtype=torch.long,
            )

    seq_lens = torch.tensor(seq_lens_list, device=device, dtype=torch.long)
    _lap("build_inputs")

    def _live_cache_mask(batch_idx, _head_idx, q_idx, kv_idx):
        row_len = seq_lens[batch_idx]
        valid_q = q_idx < row_len
        valid_kv = kv_idx < row_len
        live = (
            valid_q
            & valid_kv
            & (kv_idx <= q_idx)
            & (q_idx < death_idx[batch_idx, kv_idx])
        )
        pad_self = (~valid_q) & (q_idx == kv_idx)
        return live | pad_self

    block_mask = create_block_mask(
        _live_cache_mask,
        B=batch_size,
        H=None,
        Q_LEN=max_seq_len,
        KV_LEN=max_seq_len,
        device=device,
        _compile=True,
    )
    _lap("create_block_mask")

    _set_attn_implementation_if_needed(model, "flex_attention")

    if distillation_fn is not None and (
        distillation_prepare_fn is not None
        or distillation_teacher_fn is not None
        or distillation_loss_fn is not None
    ):
        raise ValueError(
            "Use either distillation_fn or the compact distillation hook set, not both."
        )

    if (distillation_prepare_fn is not None or distillation_teacher_fn is not None) and (
        distillation_loss_fn is None
    ):
        raise ValueError(
            "distillation_loss_fn is required when compact distillation hooks are used."
        )

    if distillation_prepare_fn is not None:
        with torch.no_grad():
            student_probe_logits = _model_forward_logits(
                model,
                input_ids=writer_ids,
                position_ids=writer_positions,
                attention_mask=block_mask,
                use_cache=False,
                kernel_options=_flex_kernel_options_from_env(),
            )
            for row, timeline in enumerate(timelines):
                for writer_start, writer_end, full_start, full_end in timeline.loss_ranges:
                    if writer_end == writer_start:
                        continue
                    distillation_prepare_fn(
                        student_probe_logits[
                            row : row + 1,
                            writer_start:writer_end,
                            :,
                        ],
                        full_start,
                        full_end,
                        row,
                    )
            del student_probe_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _lap("distill_prepare")

    teacher_logits: Tensor | None = None
    if distillation_fn is not None:
        with torch.no_grad():
            teacher_logits = _model_forward_logits(
                model,
                input_ids=writer_ids,
                position_ids=writer_positions,
                attention_mask=None,
                use_cache=False,
                kernel_options=_flex_kernel_options_from_env(),
            ).detach().to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _lap("distill_teacher")
    elif distillation_teacher_fn is not None:
        with torch.no_grad():
            teacher_probe_logits = _model_forward_logits(
                model,
                input_ids=writer_ids,
                position_ids=writer_positions,
                attention_mask=None,
                use_cache=False,
                kernel_options=_flex_kernel_options_from_env(),
            )
            for row, timeline in enumerate(timelines):
                for writer_start, writer_end, full_start, full_end in timeline.loss_ranges:
                    if writer_end == writer_start:
                        continue
                    distillation_teacher_fn(
                        teacher_probe_logits[
                            row : row + 1,
                            writer_start:writer_end,
                            :,
                        ],
                        full_start,
                        full_end,
                        row,
                    )
            del teacher_probe_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _lap("distill_teacher")

    logits = _model_forward_logits(
        model,
        input_ids=writer_ids,
        position_ids=writer_positions,
        attention_mask=block_mask,
        use_cache=False,
        kernel_options=_flex_kernel_options_from_env(),
    )
    _lap("model_forward_extract_logits")

    window_loss: Tensor | None = None
    accumulated_loss = 0.0
    for row, timeline in enumerate(timelines):
        for writer_start, writer_end, full_start, full_end in timeline.loss_ranges:
            if writer_end < writer_start:
                raise ValueError(
                    f"bad flex-mask writer loss range [{writer_start},{writer_end})"
                )
            if writer_end > logits.shape[1]:
                raise ValueError(
                    f"flex-mask writer range [{writer_start},{writer_end}) "
                    f"exceeds logits length {logits.shape[1]}"
                )
            if writer_end == writer_start:
                continue
            student_slice = logits[row : row + 1, writer_start:writer_end, :]
            distill_loss = None
            if distillation_fn is not None:
                assert teacher_logits is not None
                distill_loss = distillation_fn(
                    student_slice,
                    teacher_logits[
                        row : row + 1,
                        writer_start:writer_end,
                        :,
                    ].to(device=student_slice.device, non_blocking=True),
                    full_start,
                    full_end,
                    row,
                )
            elif distillation_loss_fn is not None:
                distill_loss = distillation_loss_fn(
                    student_slice,
                    full_start,
                    full_end,
                    row,
                )
            loss_val = loss_fn(student_slice, full_start, full_end, row)
            if distill_loss is not None:
                loss_val = loss_val + distill_loss
            accumulated_loss += float(loss_val.detach().item())
            window_loss = loss_val if window_loss is None else window_loss + loss_val
    _lap("loss_loop")

    if window_loss is None:
        window_loss = logits.float().mean() * 0.0
    if backward:
        window_loss.backward()
    _lap("backward")

    target_passes = 1 if max_forward_passes is None else int(max_forward_passes)
    if target_passes < 1:
        raise ValueError(f"max_forward_passes must be >= 1, got {target_passes}")
    if target_passes > 1:
        if not backward:
            raise ValueError("max_forward_passes > 1 requires backward=True")
        _run_dummy_passes_with_backward(
            model,
            merged_input_ids[:1],
            merged_position_ids[:1],
            target_passes - 1,
        )
        _lap("dummy_passes")

    if os.environ.get("KVE_TRACE_FLEX_MASK", "") == "1":
        logger.warning(
            "[FLEX-MASK-BATCH] batch=%d max_writer_len=%d loss_ranges=%d",
            batch_size,
            max_seq_len,
            sum(len(timeline.loss_ranges) for timeline in timelines),
        )

    if timing_enabled:
        useful_tokens = sum(seq_lens_list)
        padded_tokens = batch_size * max_seq_len
        timing_parts = " ".join(
            f"{name}={value:.3f}s" for name, value in timings.items()
        )
        memory_part = ""
        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            memory_part = (
                f" alloc={alloc:.3f}GB peak={peak:.3f}GB "
                f"reserved={reserved:.3f}GB"
            )
        logger.warning(
            "[FLEX-TIMING] batch=%d writer_lens=%s writer_tokens=%d/%d "
            "writer_pad=%.3fx loss_ranges=%d logits_shape=%s logits_dtype=%s "
            "times=%s%s",
            batch_size,
            seq_lens_list,
            useful_tokens,
            padded_tokens,
            padded_tokens / max(useful_tokens, 1),
            sum(len(timeline.loss_ranges) for timeline in timelines),
            tuple(logits.shape),
            str(logits.dtype),
            timing_parts,
            memory_part,
        )

    return {
        "loss": torch.tensor(accumulated_loss, device=device) if backward else window_loss,
        "n_segments": 1,
    }


def packed_flex_mask_segmented_forward(
    model: torch.nn.Module,
    calls_batch: list[list],
    merged_input_ids: Tensor,
    merged_position_ids: Tensor,
    loss_fn,
    *,
    max_forward_passes: int | None = None,
    distillation_fn=None,
    distillation_prepare_fn=None,
    distillation_teacher_fn=None,
    distillation_loss_fn=None,
    device: torch.device,
    backward: bool = True,
) -> dict:
    """Horizontally packed masked FlexAttention path.

    Logical samples are concatenated into one sequence. The Flex mask enforces
    both per-sample causal/liveness rules and sample boundaries, so packed
    samples cannot attend to each other.
    """
    if not calls_batch:
        raise ValueError("packed_flex_mask_segmented_forward requires calls")
    if any(not calls for calls in calls_batch):
        raise ValueError("each packed flex-mask sample requires non-empty calls")

    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:  # pragma: no cover - depends on torch build
        raise RuntimeError(
            "packed_flex_mask_segmented_forward requires torch flex_attention"
        ) from exc

    timing_enabled = _KVE_FLEX_TIMING
    timings: dict[str, float] = {}
    timing_last = _timing_now(device) if timing_enabled else 0.0

    def _lap(name: str) -> None:
        nonlocal timing_last
        if not timing_enabled:
            return
        now = _timing_now(device)
        timings[name] = timings.get(name, 0.0) + (now - timing_last)
        timing_last = now

    timelines = [_build_flex_mask_writer_timeline(calls) for calls in calls_batch]
    for _t_idx, _t in enumerate(timelines):
        if _t.extra_intervals:
            raise NotImplementedError(
                f"sample {_t_idx}: managed-context recall intervals are only "
                "supported on the single-sample flex_mask_segmented_forward "
                "path for now (plans/flex_full_bptt_alignment.md M1); the "
                "batched/packed/selected variants need per-segment row "
                "offsets for the interval tensors."
            )
    if any(not timeline.input_ids for timeline in timelines):
        raise ValueError("packed flex-mask writer timeline is empty")

    seq_lens_list = [len(timeline.input_ids) for timeline in timelines]
    offsets: list[int] = []
    total_seq_len = 0
    for row_len in seq_lens_list:
        offsets.append(total_seq_len)
        total_seq_len += row_len
    _lap("build_timeline")

    writer_ids = torch.empty((1, total_seq_len), device=device, dtype=torch.long)
    writer_positions = torch.empty((1, total_seq_len), device=device, dtype=torch.long)
    sample_id = torch.empty((total_seq_len,), device=device, dtype=torch.long)
    death_idx = torch.empty((total_seq_len,), device=device, dtype=torch.long)

    for row, (offset, timeline) in enumerate(zip(offsets, timelines, strict=True)):
        row_len = len(timeline.input_ids)
        row_slice = slice(offset, offset + row_len)
        writer_ids[0, row_slice] = torch.tensor(
            timeline.input_ids,
            device=device,
            dtype=torch.long,
        )
        writer_positions[0, row_slice] = torch.tensor(
            timeline.position_ids,
            device=device,
            dtype=torch.long,
        )
        sample_id[row_slice] = row
        death_idx[row_slice] = offset + torch.tensor(
            timeline.death_indices,
            device=device,
            dtype=torch.long,
        )
    _lap("build_inputs")

    def _live_cache_mask(_batch_idx, _head_idx, q_idx, kv_idx):
        same_sample = sample_id[q_idx] == sample_id[kv_idx]
        return same_sample & (kv_idx <= q_idx) & (q_idx < death_idx[kv_idx])

    block_mask = create_block_mask(
        _live_cache_mask,
        B=1,
        H=None,
        Q_LEN=total_seq_len,
        KV_LEN=total_seq_len,
        device=device,
        _compile=True,
    )
    _lap("create_block_mask")

    _set_attn_implementation_if_needed(model, "flex_attention")

    if distillation_fn is not None and (
        distillation_prepare_fn is not None
        or distillation_teacher_fn is not None
        or distillation_loss_fn is not None
    ):
        raise ValueError(
            "Use either distillation_fn or the compact distillation hook set, not both."
        )

    if (distillation_prepare_fn is not None or distillation_teacher_fn is not None) and (
        distillation_loss_fn is None
    ):
        raise ValueError(
            "distillation_loss_fn is required when compact distillation hooks are used."
        )

    teacher_block_mask = None
    if distillation_fn is not None or distillation_teacher_fn is not None:
        def _full_causal_same_sample_mask(_batch_idx, _head_idx, q_idx, kv_idx):
            same_sample = sample_id[q_idx] == sample_id[kv_idx]
            return same_sample & (kv_idx <= q_idx)

        teacher_block_mask = create_block_mask(
            _full_causal_same_sample_mask,
            B=1,
            H=None,
            Q_LEN=total_seq_len,
            KV_LEN=total_seq_len,
            device=device,
            _compile=True,
        )
        _lap("create_teacher_block_mask")

    if distillation_prepare_fn is not None:
        with torch.no_grad():
            student_probe_logits = _model_forward_logits(
                model,
                input_ids=writer_ids,
                position_ids=writer_positions,
                attention_mask=block_mask,
                use_cache=False,
                kernel_options=_flex_kernel_options_from_env(),
            )
            for row, (offset, timeline) in enumerate(
                zip(offsets, timelines, strict=True)
            ):
                for writer_start, writer_end, full_start, full_end in timeline.loss_ranges:
                    if writer_end == writer_start:
                        continue
                    packed_start = offset + writer_start
                    packed_end = offset + writer_end
                    distillation_prepare_fn(
                        student_probe_logits[:, packed_start:packed_end, :],
                        full_start,
                        full_end,
                        row,
                    )
            del student_probe_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _lap("distill_prepare")

    teacher_logits: Tensor | None = None
    if distillation_fn is not None:
        assert teacher_block_mask is not None
        with torch.no_grad():
            teacher_logits = _model_forward_logits(
                model,
                input_ids=writer_ids,
                position_ids=writer_positions,
                attention_mask=teacher_block_mask,
                use_cache=False,
                kernel_options=_flex_kernel_options_from_env(),
            ).detach().to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _lap("distill_teacher")
    elif distillation_teacher_fn is not None:
        assert teacher_block_mask is not None
        with torch.no_grad():
            teacher_probe_logits = _model_forward_logits(
                model,
                input_ids=writer_ids,
                position_ids=writer_positions,
                attention_mask=teacher_block_mask,
                use_cache=False,
                kernel_options=_flex_kernel_options_from_env(),
            )
            for row, (offset, timeline) in enumerate(
                zip(offsets, timelines, strict=True)
            ):
                for writer_start, writer_end, full_start, full_end in timeline.loss_ranges:
                    if writer_end == writer_start:
                        continue
                    packed_start = offset + writer_start
                    packed_end = offset + writer_end
                    distillation_teacher_fn(
                        teacher_probe_logits[:, packed_start:packed_end, :],
                        full_start,
                        full_end,
                        row,
                    )
            del teacher_probe_logits
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _lap("distill_teacher")

    logits = _model_forward_logits(
        model,
        input_ids=writer_ids,
        position_ids=writer_positions,
        attention_mask=block_mask,
        use_cache=False,
        kernel_options=_flex_kernel_options_from_env(),
    )
    _lap("model_forward_extract_logits")

    window_loss: Tensor | None = None
    accumulated_loss = 0.0
    for row, (offset, timeline) in enumerate(zip(offsets, timelines, strict=True)):
        for writer_start, writer_end, full_start, full_end in timeline.loss_ranges:
            if writer_end < writer_start:
                raise ValueError(
                    f"bad flex-mask writer loss range [{writer_start},{writer_end})"
                )
            packed_start = offset + writer_start
            packed_end = offset + writer_end
            if packed_end > logits.shape[1]:
                raise ValueError(
                    f"packed flex-mask writer range [{packed_start},{packed_end}) "
                    f"exceeds logits length {logits.shape[1]}"
                )
            if writer_end == writer_start:
                continue
            student_slice = logits[:, packed_start:packed_end, :]
            distill_loss = None
            if distillation_fn is not None:
                assert teacher_logits is not None
                distill_loss = distillation_fn(
                    student_slice,
                    teacher_logits[:, packed_start:packed_end, :].to(
                        device=student_slice.device,
                        non_blocking=True,
                    ),
                    full_start,
                    full_end,
                    row,
                )
            elif distillation_loss_fn is not None:
                distill_loss = distillation_loss_fn(
                    student_slice,
                    full_start,
                    full_end,
                    row,
                )
            loss_val = loss_fn(student_slice, full_start, full_end, row)
            if distill_loss is not None:
                loss_val = loss_val + distill_loss
            accumulated_loss += float(loss_val.detach().item())
            window_loss = loss_val if window_loss is None else window_loss + loss_val
    _lap("loss_loop")

    if window_loss is None:
        window_loss = logits.float().mean() * 0.0
    if backward:
        window_loss.backward()
    _lap("backward")

    target_passes = 1 if max_forward_passes is None else int(max_forward_passes)
    if target_passes < 1:
        raise ValueError(f"max_forward_passes must be >= 1, got {target_passes}")
    if target_passes > 1:
        if not backward:
            raise ValueError("max_forward_passes > 1 requires backward=True")
        _run_dummy_passes_with_backward(
            model,
            merged_input_ids[:1],
            merged_position_ids[:1],
            target_passes - 1,
        )
        _lap("dummy_passes")

    if os.environ.get("KVE_TRACE_FLEX_MASK", "") == "1":
        logger.warning(
            "[FLEX-MASK-PACKED] batch=%d writer_len=%d loss_ranges=%d",
            len(timelines),
            total_seq_len,
            sum(len(timeline.loss_ranges) for timeline in timelines),
        )

    if timing_enabled:
        timing_parts = " ".join(
            f"{name}={value:.3f}s" for name, value in timings.items()
        )
        memory_part = ""
        if device.type == "cuda":
            alloc = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            peak = torch.cuda.max_memory_allocated(device) / 1e9
            memory_part = (
                f" alloc={alloc:.3f}GB peak={peak:.3f}GB "
                f"reserved={reserved:.3f}GB"
            )
        logger.warning(
            "[FLEX-PACK-TIMING] batch=%d writer_lens=%s writer_tokens=%d/%d "
            "writer_pad=1.000x loss_ranges=%d logits_shape=%s logits_dtype=%s "
            "times=%s%s",
            len(timelines),
            seq_lens_list,
            total_seq_len,
            total_seq_len,
            sum(len(timeline.loss_ranges) for timeline in timelines),
            tuple(logits.shape),
            str(logits.dtype),
            timing_parts,
            memory_part,
        )

    return {
        "loss": torch.tensor(accumulated_loss, device=device) if backward else window_loss,
        "n_segments": 1,
    }


def _maybe_dump_trainer_kv(
    persistent_cache: DynamicCache,
    *,
    call_idx: int,
    is_last_call: bool,
    pre_trim_ids: list[int],
    pre_trim_through: int,
    cache_token_ids: list[int] | None,
    cache_position_ids: list[int] | None,
    position_ids_this_call: list[int],
    new_tokens_this_call: list[int],
) -> None:
    """Optional K/V cache dump for V-vs-T cache divergence diagnosis.

    Gated on env vars (see debug/local_kl_diag.py):
      - KVE_DUMP_TRAINER_K=<path>: enable, write to <path>
      - KVE_DUMP_TRAINER_CALL_IDX=<int>[,<int>...]: dump after these calls
        (0-indexed within this per_call invocation); default = last call only
      - KVE_DUMP_LAYERS=<csv ints>: subset of layer indices (default all)
    """
    import os as _os
    dump_path = _os.environ.get("KVE_DUMP_TRAINER_K", "")
    if not dump_path:
        return
    dump_call_idx_s = _os.environ.get("KVE_DUMP_TRAINER_CALL_IDX", "")
    if dump_call_idx_s != "":
        target_idxs = {
            int(x.strip())
            for x in dump_call_idx_s.split(",")
            if x.strip()
        }
        fire = call_idx in target_idxs
    else:
        target_idxs = set()
        fire = is_last_call
    if not fire:
        return
    if len(target_idxs) > 1 or _os.environ.get(
        "KVE_DUMP_TRAINER_K_INCLUDE_CALL", ""
    ) == "1":
        root, ext = _os.path.splitext(dump_path)
        if not ext:
            ext = ".pt"
        dump_path = f"{root}_C{call_idx}{ext}"
    _keys, _values = _get_kv_from_cache(persistent_cache)
    _layers_env = _os.environ.get("KVE_DUMP_LAYERS", "")
    if _layers_env:
        _layer_idxs = [int(x) for x in _layers_env.split(",") if x.strip()]
    else:
        _layer_idxs = list(range(len(_keys)))
    K_per_layer = [
        _keys[l].detach().to(torch.float32).cpu() for l in _layer_idxs
    ]
    V_per_layer = [
        _values[l].detach().to(torch.float32).cpu() for l in _layer_idxs
    ]
    payload = {
        "K": K_per_layer[0],
        "V": V_per_layer[0],
        "K_per_layer": K_per_layer,
        "V_per_layer": V_per_layer,
        "layer_idxs": _layer_idxs,
        "num_layers_total": len(_keys),
        "call_idx": call_idx,
        "position_ids_this_call": position_ids_this_call,
        "new_tokens_this_call": new_tokens_this_call,
        "merged_input_ids_so_far": pre_trim_ids[:pre_trim_through],
        "cache_token_ids": (
            list(cache_token_ids) if cache_token_ids is not None else None
        ),
        "cache_position_ids": (
            list(cache_position_ids) if cache_position_ids is not None else None
        ),
    }
    torch.save(payload, dump_path)
    logger.warning(
        "[T-dump] wrote %d layers (of %d); layer 0 K=%s V=%s "
        "after trainer call %d -> %s",
        len(_layer_idxs), len(_keys),
        tuple(K_per_layer[0].shape), tuple(V_per_layer[0].shape),
        call_idx, dump_path,
    )


def per_call_segmented_forward(
    model: torch.nn.Module,
    calls: list,  # list[CallWire]
    merged_input_ids: Tensor,  # [1, post_trim_full_len], post-trim (used for dummy passes only)
    merged_position_ids: Tensor,  # [1, post_trim_full_len] (dummy passes only)
    loss_fn: SegmentLossFn,
    *,
    max_forward_passes: int,
    max_bptt_window_forward_passes: list[int] | None = None,
    bptt_segments: int | None = 1,
    device: torch.device,
) -> dict:
    """Per-call forward with PRE-TRIM input + cache splice on admission.

    Mirrors vLLM exactly: trainer's K cache at any logical position is
    written under the SAME attention context vLLM's writer used at that
    position. The eviction is realized via splice of the persistent_cache
    (the way vLLM evicts in HBM), not by trimming the input tokens (the
    old approach, which broke L1+ K values at the first post-eviction
    slot — see debug/local_kl_diag.py + compare_kv.py).

    For an ADMISSION call (>=1 admission event with tokens_evicted > 0):
      1. Warmup forward over [call_pre_start, inherited_end_pre) — the
         inherited part of submitted_prompt. K cache fills under full
         pre-admission causal attention.
      2. Splice persistent_cache for each admission event in order
         (sequential composition matches _apply_admission_trim).
      3. Main forward over [inherited_end_pre, call_pre_end) — the
         new_user_fragment + completion. Attention sees the spliced cache
         + freshly-written K, matching vLLM's post-admission state.
      4. Concatenate logits: warmup logits (subset to surviving pre-trim
         positions) ++ main logits → length matches the call's post-trim
         contribution to merged_input_ids.

    For an EXTENSION call (no admission, second+ call in a multi-call
    sample): single forward over the call's NEW pre-trim content (the
    tokens beyond what prior calls' forwards already wrote to cache).

    Positions use plain arange + pre-trim absolute offset while the request
    stays in the local frame. When vLLM reports an inherited or
    post-admission position_offset, the trainer switches to the same
    physical-position + piecewise-offset rule the worker uses so cached K
    and newly-written Q/K stay in the writer's RoPE frame.
    """
    bptt_segments = _normalize_bptt_segments(bptt_segments)
    if not calls:
        raise ValueError("per_call_segmented_forward requires non-empty calls list")
    if bptt_segments is not None and bptt_segments < 1:
        raise ValueError(f"bptt_segments must be None or >= 1, got {bptt_segments}")
    bptt_windowed = bptt_segments != 1

    plans, pre_trim_ids_py = _build_pre_trim_plan(calls)
    # _PLAN_CONTRACT: mb.input_ids must be PRE-TRIM cumulative (see the
    # Phase A.3 note in _build_pre_trim_plan) and the plan's final post_end
    # must land exactly on it. A sample whose FIRST call carries an
    # admission trim arrives POST-trim and silently shifts every replay
    # coordinate by the trim size (measured: mismatch_kl 21-26 for four
    # runs). Fail loud instead.
    _merged_len = int(merged_input_ids.shape[1])
    _plan_end = int(plans[-1]["post_end"]) if plans else 0
    if _plan_end != _merged_len:
        raise ValueError(
            "per-call plan/merged coordinate contract violated: plan post_end="
            f"{_plan_end} vs merged_input_ids len={_merged_len} "
            f"(delta {_plan_end - _merged_len}; calls={len(plans)}). A sample "
            "whose first call carries an admission trim was built POST-trim -- "
            "the rollout chain likely split mid-episode (see kv-race notes "
            "2026-08-21)."
        )
    pre_trim_ids = torch.tensor(
        pre_trim_ids_py, device=device, dtype=torch.long,
    ).unsqueeze(0)

    n_forwards_run = 0
    accumulated_loss = 0.0
    window_loss: Tensor | None = None
    calls_in_window = 0
    forwards_in_window = 0
    bptt_window_idx = 0
    persistent_cache: DynamicCache = DynamicCache()
    # Token id map for the live DynamicCache slots after admission splices.
    # `pre_trim_ids` is useful for loss coordinates, but after an admission
    # eviction it no longer matches physical cache slots.
    cache_token_ids_py: list[int] = []
    cache_position_ids_py: list[int] = []
    (
        mirror_decode,
        mirror_decode_call_filter,
        mirror_decode_last_n,
    ) = _per_call_mirror_decode_config()
    stream_mirror_decode_loss = (
        os.environ.get("KVE_TRAINER_STREAM_MIRROR_LOSS", "1") != "0"
    )
    if bptt_windowed and mirror_decode and stream_mirror_decode_loss:
        logger.warning(
            "KVE_TRAINER_STREAM_MIRROR_LOSS is disabled when "
            "per_call_segmented_forward runs with bptt_segments=%s; "
            "losses must stay live until the BPTT window backward.",
            str(bptt_segments),
        )
        stream_mirror_decode_loss = False
    base_attn_impl = getattr(getattr(model, "config", None), "_attn_implementation", None)
    past_attn_impl = os.environ.get("KVE_TRAINER_PAST_ATTN_IMPL", "") or None
    trace_b2b = os.environ.get("KVE_TRACE_B2B", "") == "1"
    trace_call_filter_env = os.environ.get("KVE_TRACE_CALL_IDX", "")
    trace_call_filter = None
    if trace_call_filter_env:
        trace_call_filter = {
            int(x.strip())
            for x in trace_call_filter_env.split(",")
            if x.strip()
        }
    trace_forward_sig = os.environ.get("KVE_TRACE_FORWARD_SIG", "0") == "1"

    def _add_window_loss(loss_val: Tensor) -> None:
        nonlocal accumulated_loss, window_loss
        accumulated_loss += float(loss_val.detach().item())
        if not bptt_windowed:
            loss_val.backward()
            return
        window_loss = loss_val if window_loss is None else window_loss + loss_val

    def _flush_bptt_window(force: bool = False) -> None:
        nonlocal window_loss, calls_in_window, forwards_in_window
        nonlocal bptt_window_idx, persistent_cache
        if not bptt_windowed:
            return
        if not force and calls_in_window <= 0:
            return
        target_forwards = forwards_in_window
        if max_bptt_window_forward_passes is not None:
            if bptt_window_idx >= len(max_bptt_window_forward_passes):
                raise ValueError(
                    "max_bptt_window_forward_passes is shorter than the "
                    f"local BPTT windows: idx={bptt_window_idx}, "
                    f"len={len(max_bptt_window_forward_passes)}"
                )
            target_forwards = int(max_bptt_window_forward_passes[bptt_window_idx])
        if target_forwards < forwards_in_window:
            raise ValueError(
                "max_bptt_window_forward_passes must be >= local forwards "
                f"for each window, got target={target_forwards}, "
                f"local={forwards_in_window}, window={bptt_window_idx}"
            )
        for _ in range(target_forwards - forwards_in_window):
            dummy_loss = _dummy_forward_loss(
                model,
                merged_input_ids,
                merged_position_ids,
            )
            window_loss = (
                dummy_loss if window_loss is None else window_loss + dummy_loss
            )
        if window_loss is not None:
            window_loss.backward()
            window_loss = None
        persistent_cache = _detach_dynamic_cache(persistent_cache)
        calls_in_window = 0
        forwards_in_window = 0
        bptt_window_idx += 1

    def _finish_call(loss_val: Tensor, n_forwards: int) -> None:
        nonlocal n_forwards_run, calls_in_window, forwards_in_window
        nonlocal persistent_cache
        _add_window_loss(loss_val)
        n_forwards_run += n_forwards
        if bptt_windowed:
            calls_in_window += 1
            forwards_in_window += n_forwards
            if bptt_segments is not None and calls_in_window >= bptt_segments:
                _flush_bptt_window()
        else:
            # Detach the cache so the next call's backward stops at this
            # call boundary (bptt_segments=1 / M3 semantics).
            persistent_cache = _detach_dynamic_cache(persistent_cache)

    for plan in plans:
        call_idx = plan["call_idx"]
        cps = plan["call_pre_start"]
        cpe = plan["call_pre_end"]
        ie_pre = plan["inherited_end_pre"]
        has_admission = plan["has_admission"]
        splices = plan["splices"]
        post_start = plan["post_start"]
        post_end = plan["post_end"]
        writer_offset = plan["writer_offset"]
        admission_offset_after = plan["admission_offset_after"]
        evict_start_in_sub = plan["evict_start_in_sub"]
        protected_prefix_len = plan.get("protected_prefix_len", evict_start_in_sub)
        pad_len = plan.get("pad_len", 0)
        sub_end_pre = plan.get("sub_end_pre", plan["inherited_end_pre"])
        b2b_warm_end_pre = plan.get("b2b_warm_end_pre", sub_end_pre)
        prefix_cache_keep_len = plan.get("prefix_cache_keep_len")
        prefix_replay_tokens_py = list(plan.get("prefix_replay_tokens", []) or [])
        trace_this_call = (
            trace_b2b
            and (trace_call_filter is None or call_idx in trace_call_filter)
        )
        trace_forward_this_call = (
            trace_forward_sig
            and (trace_call_filter is None or call_idx in trace_call_filter)
        )
        mirror_decode_this_call = _mirror_decode_call_enabled(
            call_idx=call_idx,
            num_calls=len(calls),
            mirror_decode=mirror_decode,
            mirror_decode_call_filter=mirror_decode_call_filter,
            mirror_decode_last_n=mirror_decode_last_n,
        )

        if cpe <= cps:
            continue  # empty contribution

        prefix_replay_n_forwards = 0
        if prefix_cache_keep_len is not None:
            keep_len = int(prefix_cache_keep_len)
            cache_len_at_prefix_entry = persistent_cache.get_seq_length()
            if keep_len < 0 or keep_len > cache_len_at_prefix_entry:
                raise ValueError(
                    f"call {call_idx}: prefix cache keep len {keep_len} "
                    f"out of cache len={cache_len_at_prefix_entry}"
                )
            if keep_len < cache_len_at_prefix_entry:
                persistent_cache = _splice_dynamic_cache(
                    persistent_cache,
                    keep_len,
                    cache_len_at_prefix_entry,
                )
                del cache_token_ids_py[keep_len:]
                del cache_position_ids_py[keep_len:]
                if trace_forward_this_call:
                    logger.warning(
                        "[T-PREFIX-REPLAY] label=C%d truncate cache=%d->%d",
                        call_idx,
                        cache_len_at_prefix_entry,
                        persistent_cache.get_seq_length(),
                    )
            if prefix_replay_tokens_py:
                replay_tokens = torch.tensor(
                    prefix_replay_tokens_py,
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(0)
                if writer_offset != 0:
                    replay_positions = _piecewise_positions(
                        keep_len,
                        keep_len + len(prefix_replay_tokens_py),
                        offset=writer_offset,
                        protected_prefix_len=protected_prefix_len,
                        device=device,
                    )
                else:
                    replay_positions = (
                        torch.arange(
                            len(prefix_replay_tokens_py),
                            device=device,
                            dtype=torch.long,
                        )
                        + keep_len
                    ).unsqueeze(0)
                cache_before_replay = persistent_cache.get_seq_length()
                out_replay = _model_forward_with_optional_attn_switch(
                    model,
                    input_ids=replay_tokens,
                    position_ids=replay_positions,
                    past_key_values=persistent_cache,
                    base_attn_impl=base_attn_impl,
                    past_attn_impl=past_attn_impl,
                )
                replay_logits = _extract_logits(out_replay)
                if not bptt_windowed:
                    (replay_logits.float().mean() * 0.0).backward()
                    persistent_cache = _detach_dynamic_cache(persistent_cache)
                prefix_replay_n_forwards = 1
                cache_token_ids_py.extend(prefix_replay_tokens_py)
                cache_position_ids_py.extend(
                    replay_positions[0].detach().cpu().tolist()
                )
                if trace_forward_this_call:
                    tok_head, tok_tail = _preview_1d(
                        replay_tokens,
                        0,
                        len(prefix_replay_tokens_py),
                    )
                    pos_head, pos_tail = _preview_1d(
                        replay_positions,
                        0,
                        len(prefix_replay_tokens_py),
                    )
                    logger.warning(
                        "[T-FWD-SIG] label=C%d:prefix-replay len=%d "
                        "cache=%d->%d writer_offset=%d pos_head=%s "
                        "pos_tail=%s tok_head=%s tok_tail=%s",
                        call_idx,
                        len(prefix_replay_tokens_py),
                        cache_before_replay,
                        persistent_cache.get_seq_length(),
                        writer_offset,
                        pos_head,
                        pos_tail,
                        tok_head,
                        tok_tail,
                    )

        if has_admission and cps > 0:
            # ── B.2b: splice-only path for admission in extension calls.
            #
            # When admission fires in call N>0, V's KV cache at the start
            # of this call already contains K written by prior calls (=
            # by the trainer's own prior call forwards in this sample).
            # V's admission evicts rows from that live cache BEFORE
            # prefilling new content; the trainer mirrors this exactly
            # by splicing `persistent_cache` then running a single main
            # forward over [cps..cpe).
            #
            # No warmup needed — the cache K rows for the surviving
            # prior content were written by THIS sample's prior call
            # forwards under their pre-trim logical positions (matching
            # V's writer). The splice removes the same rows V removed.
            #
            # Multi-event composition: each event's es is in
            # post-prior-iter cache coord (matches `_apply_admission_trim`);
            # successive splices operate on the running cache state.
            # Invariant: trainer's cache length at entry to an extension
            # call equals the prior calls' POST-TRIM cumulative length
            # (since prior admissions already shrunk the cache via the
            # two-phase warmup / splice in their own calls). For the
            # very first multi-call admission case (where prior calls
            # had no admissions) this length equals cps (= cum_pre).
            # B.2b: splice cache first, then single forward over the new
            # contribution (sub[cum_post:] + comp). Reorder attempts
            # (re-prefill new content with pre-admission attention before
            # splicing) measurably regressed production KL — likely
            # because they expose more partial-tail batch-shape FA2
            # numerical drift than they remove. Keep the splice-first
            # design until the drift is measured directly.
            b2b_warmup_splice = (
                os.environ.get("KVE_TRAINER_B2B_WARMUP_SPLICE", "0") == "1"
            ) or bool(prefix_replay_tokens_py)
            cache_len_before_splice = persistent_cache.get_seq_length()
            if b2b_warmup_splice:
                # Diagnostic mirror of vLLM's synthetic prefill + admission:
                # write the newly submitted prompt tail against the pre-splice
                # cache, then splice, then write generated/pad tokens against
                # the post-admission cache. Positions are computed from
                # physical cache slots plus the relevant vLLM offset.
                warm_end = b2b_warm_end_pre
                warm_logits_detached: torch.Tensor | None = None
                warm_n_forwards = 0
                if warm_end > cps:
                    w_toks = pre_trim_ids[:, cps:warm_end]
                    w_pos = _piecewise_positions(
                        cache_len_before_splice,
                        cache_len_before_splice + (warm_end - cps),
                        offset=writer_offset,
                        protected_prefix_len=protected_prefix_len,
                        device=device,
                    )
                    cache_before_warm = persistent_cache.get_seq_length()
                    out_w = _model_forward_with_optional_attn_switch(
                        model,
                        input_ids=w_toks,
                        position_ids=w_pos,
                        past_key_values=persistent_cache,
                        base_attn_impl=base_attn_impl,
                        past_attn_impl=past_attn_impl,
                    )
                    w_log = _extract_logits(out_w)
                    warm_logits_detached = w_log if bptt_windowed else w_log.detach()
                    if not bptt_windowed:
                        (w_log.float().mean() * 0.0).backward()
                    warm_n_forwards = 1
                    if trace_forward_this_call:
                        tok_head, tok_tail = _preview_1d(w_toks, 0, warm_end - cps)
                        pos_head, pos_tail = _preview_1d(w_pos, 0, warm_end - cps)
                        logger.warning(
                            "[T-FWD-SIG] label=C%d:B2B-warm chunk=[%d,%d) "
                            "len=%d cache=%d->%d prompt_len=%d pad_len=%d "
                            "mirror=%s attn_override=%s pos_head=%s "
                            "pos_tail=%s tok_head=%s tok_tail=%s",
                            call_idx, cps, warm_end, warm_end - cps,
                            cache_before_warm, persistent_cache.get_seq_length(),
                            warm_end - cps, 0, False, None,
                            pos_head, pos_tail, tok_head, tok_tail,
                        )
                    cache_token_ids_py.extend(pre_trim_ids_py[cps:warm_end])
                    cache_position_ids_py.extend(
                        w_pos[0].detach().cpu().tolist()
                    )
                if not bptt_windowed:
                    persistent_cache = _detach_dynamic_cache(persistent_cache)

                running_cache_len = persistent_cache.get_seq_length()
                for (es, te) in splices:
                    assert es + te <= running_cache_len, (
                        f"call {call_idx} (B.2b-warm): splice "
                        f"[{es},{es+te}) out of cache (len={running_cache_len})"
                    )
                    persistent_cache = _splice_dynamic_cache(
                        persistent_cache, es, es + te,
                    )
                    del cache_token_ids_py[es : es + te]
                    del cache_position_ids_py[es : es + te]
                    if trace_forward_this_call:
                        logger.warning(
                            "[T-SPLICE-SIG] label=C%d:B2B-warm "
                            "splice=[%d,%d) cache=%d->%d",
                            call_idx, es, es + te,
                            running_cache_len,
                            persistent_cache.get_seq_length(),
                        )
                    running_cache_len -= te

                main_start = warm_end
                main_tokens = pre_trim_ids[:, main_start:cpe]
                physical_start = persistent_cache.get_seq_length()
                if admission_offset_after != 0:
                    main_positions = _piecewise_positions(
                        physical_start,
                        physical_start + (cpe - main_start),
                        offset=admission_offset_after,
                        protected_prefix_len=protected_prefix_len,
                        device=device,
                    )
                else:
                    main_positions = (
                        torch.arange(
                            cpe - main_start,
                            device=device,
                            dtype=torch.long,
                        )
                        + physical_start
                    ).unsqueeze(0)
                if trace_this_call:
                    w_pos_1d = (
                        w_pos[0].detach().cpu().tolist()
                        if warm_end > cps else []
                    )
                    m_pos_1d = main_positions[0].detach().cpu().tolist()
                    logger.warning(
                        "[B2B-WARM] call=%d cache_entry=%d splices=%s "
                        "warm=[%d,%d) main=[%d,%d) cache_after_splice=%d "
                        "writer_offset=%d admission_offset_after=%d "
                        "warm_pos_head=%s warm_pos_tail=%s "
                        "main_pos_head=%s main_pos_tail=%s",
                        call_idx, cache_len_before_splice, splices,
                        cps, warm_end, main_start, cpe,
                        persistent_cache.get_seq_length(),
                        writer_offset, admission_offset_after,
                        w_pos_1d[:8], w_pos_1d[-8:],
                        m_pos_1d[:8], m_pos_1d[-8:],
                    )
                if mirror_decode_this_call and stream_mirror_decode_loss:
                    warm_kept_len = 0
                    if warm_logits_detached is not None:
                        warm_kept_len = int(warm_logits_detached.shape[1])
                        warm_loss = loss_fn(
                            warm_logits_detached,
                            post_start,
                            post_start + warm_kept_len,
                        )
                        _backward_loss_or_zero(warm_loss, warm_logits_detached)
                        accumulated_loss += float(warm_loss.detach().item())
                    (
                        persistent_cache,
                        main_n_forwards,
                        main_loss,
                        main_kept_len,
                    ) = _forward_mirror_decode_stream_loss(
                        model,
                        input_ids=main_tokens,
                        position_ids=main_positions,
                        past_key_values=persistent_cache,
                        prompt_len=max(0, min(sub_end_pre, cpe) - main_start),
                        pad_len=pad_len,
                        base_attn_impl=base_attn_impl,
                        past_attn_impl=past_attn_impl,
                        loss_fn=loss_fn,
                        full_logit_start=post_start + warm_kept_len,
                        trace_label=(
                            f"C{call_idx}:B2B-warm-main"
                            if trace_forward_this_call else None
                        ),
                    )
                    cache_token_ids_py.extend(pre_trim_ids_py[main_start:cpe])
                    cache_position_ids_py.extend(
                        main_positions[0].detach().cpu().tolist()
                    )
                    assert warm_kept_len + main_kept_len == post_end - post_start, (
                        f"call {call_idx} (B.2b-warm): streamed logits len="
                        f"{warm_kept_len + main_kept_len} != "
                        f"post_end-post_start={post_end - post_start}"
                    )

                    _maybe_dump_trainer_kv(
                        persistent_cache,
                        call_idx=call_idx,
                        is_last_call=(call_idx == len(calls) - 1),
                        pre_trim_ids=pre_trim_ids_py,
                        pre_trim_through=cpe,
                        cache_token_ids=cache_token_ids_py,
                        cache_position_ids=cache_position_ids_py,
                        position_ids_this_call=(
                            main_positions[0].detach().cpu().tolist()
                        ),
                        new_tokens_this_call=main_tokens[0].detach().cpu().tolist(),
                    )

                    accumulated_loss += main_loss
                    n_forwards_run += (
                        prefix_replay_n_forwards
                        + warm_n_forwards
                        + main_n_forwards
                    )
                    persistent_cache = _detach_dynamic_cache(persistent_cache)
                    continue

                main_logits, main_n_forwards = _forward_with_optional_decode_mirror(
                    model,
                    input_ids=main_tokens,
                    position_ids=main_positions,
                    past_key_values=persistent_cache,
                    prompt_len=max(0, min(sub_end_pre, cpe) - main_start),
                    pad_len=pad_len,
                    mirror_decode=mirror_decode_this_call,
                    base_attn_impl=base_attn_impl,
                    past_attn_impl=past_attn_impl,
                    trace_label=(
                        f"C{call_idx}:B2B-warm-main"
                        if trace_forward_this_call else None
                    ),
                )
                cache_token_ids_py.extend(pre_trim_ids_py[main_start:cpe])
                cache_position_ids_py.extend(
                    main_positions[0].detach().cpu().tolist()
                )

                pieces: list[torch.Tensor] = []
                if warm_logits_detached is not None:
                    pieces.append(warm_logits_detached)
                pieces.append(main_logits)
                kept_logits = torch.cat(pieces, dim=1)
                assert kept_logits.shape[1] == post_end - post_start, (
                    f"call {call_idx} (B.2b-warm): kept_logits len="
                    f"{kept_logits.shape[1]} != post_end-post_start="
                    f"{post_end - post_start}"
                )

                _maybe_dump_trainer_kv(
                    persistent_cache,
                    call_idx=call_idx,
                    is_last_call=(call_idx == len(calls) - 1),
                    pre_trim_ids=pre_trim_ids_py,
                    pre_trim_through=cpe,
                    cache_token_ids=cache_token_ids_py,
                    cache_position_ids=cache_position_ids_py,
                    position_ids_this_call=main_positions[0].detach().cpu().tolist(),
                    new_tokens_this_call=main_tokens[0].detach().cpu().tolist(),
                )

                loss_val = loss_fn(kept_logits, post_start, post_end)
                _finish_call(
                    loss_val,
                    prefix_replay_n_forwards
                    + warm_n_forwards
                    + main_n_forwards,
                )
                continue

            cache_len_at_entry = cache_len_before_splice
            for (es, te) in splices:
                assert es + te <= cache_len_at_entry, (
                    f"call {call_idx} (B.2b): splice [{es},{es+te}) "
                    f"out of cache (len={cache_len_at_entry})"
                )
                persistent_cache = _splice_dynamic_cache(
                    persistent_cache, es, es + te,
                )
                del cache_token_ids_py[es : es + te]
                del cache_position_ids_py[es : es + te]
                if trace_forward_this_call:
                    logger.warning(
                        "[T-SPLICE-SIG] label=C%d:B2B "
                        "splice=[%d,%d) cache=%d->%d",
                        call_idx, es, es + te,
                        cache_len_at_entry,
                        persistent_cache.get_seq_length(),
                    )
                cache_len_at_entry -= te

            main_tokens = pre_trim_ids[:, cps:cpe]
            if admission_offset_after != 0:
                physical_start = cache_len_at_entry
                main_positions = _piecewise_positions(
                    physical_start,
                    physical_start + (cpe - cps),
                    offset=admission_offset_after,
                    protected_prefix_len=protected_prefix_len,
                    device=device,
                )
            else:
                main_positions = (
                    torch.arange(cpe - cps, device=device, dtype=torch.long)
                    + cps
                ).unsqueeze(0)
            main_prompt_len = max(0, min(sub_end_pre, cpe) - cps)
            if trace_this_call:
                pos_1d = main_positions[0].detach().cpu().tolist()
                logger.warning(
                    "[B2B] call=%d cps=%d cpe=%d cache_entry=%d "
                    "splices=%s cache_after_splice=%d main_tokens=%d "
                    "main_prompt_len=%d pad_len=%d post_range=[%d,%d) "
                    "sub_len=%s comp_len=%s sub_end_pre=%d "
                    "new_content_start_in_sub=%s admission_offset_after=%d "
                    "writer_offset=%d protected_prefix_len=%d "
                    "v_synth=(np=%s,nuf=%s,cached=%s,off=%s) "
                    "v_adm=(np=%s,nuf=%s,evicted=%s,off=%s) "
                    "positions_head=%s positions_tail=%s",
                    call_idx, cps, cpe, cache_len_before_splice,
                    splices, cache_len_at_entry, cpe - cps,
                    main_prompt_len, pad_len, post_start, post_end,
                    str(plan.get("sub_len")), str(plan.get("comp_len")),
                    sub_end_pre, str(plan.get("new_content_start_in_sub")),
                    admission_offset_after, writer_offset, protected_prefix_len,
                    str(plan.get("synthetic_num_prompt")),
                    str(plan.get("synthetic_nuf_len")),
                    str(plan.get("synthetic_cached_tokens")),
                    str(plan.get("synthetic_offset_after")),
                    str(plan.get("admission_num_prompt")),
                    str(plan.get("admission_nuf_len")),
                    str(plan.get("admission_total_evicted")),
                    str(plan.get("admission_event_offset_after")),
                    pos_1d[:8], pos_1d[-8:],
                )
            if mirror_decode_this_call and stream_mirror_decode_loss:
                (
                    persistent_cache,
                    main_n_forwards,
                    main_loss,
                    main_kept_len,
                ) = _forward_mirror_decode_stream_loss(
                    model,
                    input_ids=main_tokens,
                    position_ids=main_positions,
                    past_key_values=persistent_cache,
                    prompt_len=main_prompt_len,
                    pad_len=pad_len,
                    base_attn_impl=base_attn_impl,
                    past_attn_impl=past_attn_impl,
                    loss_fn=loss_fn,
                    full_logit_start=post_start,
                    trace_label=(
                        f"C{call_idx}:B2B-main"
                        if trace_forward_this_call else None
                    ),
                )
                cache_token_ids_py.extend(pre_trim_ids_py[cps:cpe])
                cache_position_ids_py.extend(
                    main_positions[0].detach().cpu().tolist()
                )
                assert main_kept_len == post_end - post_start, (
                    f"call {call_idx} (B.2b): streamed logits len="
                    f"{main_kept_len} != post_end-post_start="
                    f"{post_end - post_start}"
                )

                _maybe_dump_trainer_kv(
                    persistent_cache,
                    call_idx=call_idx,
                    is_last_call=(call_idx == len(calls) - 1),
                    pre_trim_ids=pre_trim_ids_py,
                    pre_trim_through=cpe,
                    cache_token_ids=cache_token_ids_py,
                    cache_position_ids=cache_position_ids_py,
                    position_ids_this_call=main_positions[0].detach().cpu().tolist(),
                    new_tokens_this_call=main_tokens[0].detach().cpu().tolist(),
                )

                accumulated_loss += main_loss
                n_forwards_run += prefix_replay_n_forwards + main_n_forwards
                persistent_cache = _detach_dynamic_cache(persistent_cache)
                continue

            main_logits, main_n_forwards = _forward_with_optional_decode_mirror(
                model,
                input_ids=main_tokens,
                position_ids=main_positions,
                past_key_values=persistent_cache,
                prompt_len=main_prompt_len,
                pad_len=pad_len,
                mirror_decode=mirror_decode_this_call,
                base_attn_impl=base_attn_impl,
                past_attn_impl=past_attn_impl,
                trace_label=(
                    f"C{call_idx}:B2B-main"
                    if trace_forward_this_call else None
                ),
            )
            cache_token_ids_py.extend(pre_trim_ids_py[cps:cpe])
            cache_position_ids_py.extend(
                main_positions[0].detach().cpu().tolist()
            )
            assert main_logits.shape[1] == post_end - post_start, (
                f"call {call_idx} (B.2b): main_logits len="
                f"{main_logits.shape[1]} != post_end-post_start="
                f"{post_end - post_start}"
            )

            _maybe_dump_trainer_kv(
                persistent_cache,
                call_idx=call_idx,
                is_last_call=(call_idx == len(calls) - 1),
                pre_trim_ids=pre_trim_ids_py,
                pre_trim_through=cpe,
                cache_token_ids=cache_token_ids_py,
                cache_position_ids=cache_position_ids_py,
                position_ids_this_call=main_positions[0].detach().cpu().tolist(),
                new_tokens_this_call=main_tokens[0].detach().cpu().tolist(),
            )

            loss_val = loss_fn(main_logits, post_start, post_end)
            _finish_call(loss_val, prefix_replay_n_forwards + main_n_forwards)
            continue

        if has_admission:
            # ── B.2a: Warmup-then-splice for admission in call 0.
            #
            # V's K at survivor positions in call 0's admission was
            # written in ONE prefill forward — V's call 0 (or the
            # writer for cross-rollout inheritor cases) prefills all of
            # sub_0 with full causal attention to every prior position,
            # THEN admission evicts cache rows. The surviving rows'
            # K *values* still encode attention to the (now-evicted)
            # range. They aren't "gap-position" K — that earlier
            # diagnosis was wrong.
            #
            # Single forward over pre-trim [cps..sub_end_pre), then splice
            # cache rows [es..es+te) post-hoc. For cross-writer prefix-cache
            # inheritance, the warmup uses vLLM's piecewise writer-offset
            # positions so reconstructed K lands in the same RoPE frame as
            # the cached writer rows.
            #
            # Warmup covers the FULL sub_N (not just the inherited
            # region), because V re-prefilled every token in sub_N
            # under full causal attention BEFORE admission fired —
            # including the new_user_fragment. Stopping warmup at
            # inherited_end_pre left survivor K past that boundary
            # written under wrong attention context AND caused the
            # splice to run off the end of the warmup output when V's
            # evict range extended into the new_user_fragment region.
            warm_end = sub_end_pre
            warm_logits_detached: torch.Tensor | None = None
            warm_n_forwards = 0
            if warm_end > cps:
                w_toks = pre_trim_ids[:, cps:warm_end]
                if writer_offset != 0:
                    w_pos = _piecewise_positions(
                        cps,
                        warm_end,
                        offset=writer_offset,
                        protected_prefix_len=protected_prefix_len,
                        device=device,
                    )
                else:
                    w_pos = (
                        torch.arange(warm_end - cps, device=device, dtype=torch.long)
                        + cps
                    ).unsqueeze(0)
                cache_before_warm = persistent_cache.get_seq_length()
                out_w = _model_forward_with_optional_attn_switch(
                    model,
                    input_ids=w_toks,
                    position_ids=w_pos,
                    past_key_values=persistent_cache,
                    base_attn_impl=base_attn_impl,
                    past_attn_impl=past_attn_impl,
                )
                w_log = _extract_logits(out_w)
                warm_logits_detached = w_log if bptt_windowed else w_log.detach()
                # FSDP2 fwd/bwd lockstep — keep the existing 2-bwd shape
                # for admission calls so dummy-pass padding stays
                # symmetric across ranks.
                if not bptt_windowed:
                    (w_log.float().mean() * 0.0).backward()
                warm_n_forwards = 1
                if trace_forward_this_call:
                    tok_head, tok_tail = _preview_1d(w_toks, 0, warm_end - cps)
                    pos_head, pos_tail = _preview_1d(w_pos, 0, warm_end - cps)
                    logger.warning(
                        "[T-FWD-SIG] label=C%d:B2A-warm chunk=[%d,%d) "
                        "len=%d cache=%d->%d prompt_len=%d pad_len=%d "
                        "mirror=%s attn_override=%s pos_head=%s "
                        "pos_tail=%s tok_head=%s tok_tail=%s",
                        call_idx, cps, warm_end, warm_end - cps,
                        cache_before_warm, persistent_cache.get_seq_length(),
                        warm_end - cps, 0, False, None,
                        pos_head, pos_tail, tok_head, tok_tail,
                    )
                cache_token_ids_py.extend(pre_trim_ids_py[cps:warm_end])
                cache_position_ids_py.extend(
                    w_pos[0].detach().cpu().tolist()
                )
            if not bptt_windowed:
                persistent_cache = _detach_dynamic_cache(persistent_cache)

            # Splice cache rows [es..es+te) per admission event in
            # post-prior-iter order. For call 0 admission, es is in
            # this call's pre-trim coord and the cache content is
            # co-indexed with pre_trim_ids; for multi-iter the events
            # already compose against the running cache state.
            running_cache_len = persistent_cache.get_seq_length()
            for (es, te) in splices:
                assert es + te <= running_cache_len, (
                    f"call {call_idx} (B.2a): splice [{es},{es+te}) "
                    f"out of cache (len={running_cache_len})"
                )
                persistent_cache = _splice_dynamic_cache(
                    persistent_cache, es, es + te,
                )
                del cache_token_ids_py[es : es + te]
                del cache_position_ids_py[es : es + te]
                if trace_forward_this_call:
                    logger.warning(
                        "[T-SPLICE-SIG] label=C%d:B2A splice=[%d,%d) "
                        "cache=%d->%d",
                        call_idx, es, es + te,
                        running_cache_len,
                        persistent_cache.get_seq_length(),
                    )
                running_cache_len -= te

            # Main forward over the remaining pre-trim contribution =
            # [sub_end_pre..cpe) = comp + pad. The warmup covered the
            # full sub_N region, so the only new tokens this forward
            # needs to process are the decode outputs and the auto-pad
            # fillers (both written by V AFTER admission, under V's
            # post-admission attention context = the spliced cache).
            main_start = sub_end_pre
            main_tokens = pre_trim_ids[:, main_start:cpe]
            if admission_offset_after != 0:
                physical_start = persistent_cache.get_seq_length()
                main_positions = _piecewise_positions(
                    physical_start,
                    physical_start + (cpe - main_start),
                    offset=admission_offset_after,
                    protected_prefix_len=protected_prefix_len,
                    device=device,
                )
            else:
                main_positions = (
                    torch.arange(cpe - main_start, device=device, dtype=torch.long)
                    + main_start
                ).unsqueeze(0)
            if mirror_decode_this_call and stream_mirror_decode_loss:
                warm_kept_len = 0
                if warm_logits_detached is not None:
                    warm_kept_len = int(warm_logits_detached.shape[1])
                    warm_loss = loss_fn(
                        warm_logits_detached,
                        post_start,
                        post_start + warm_kept_len,
                    )
                    _backward_loss_or_zero(warm_loss, warm_logits_detached)
                    accumulated_loss += float(warm_loss.detach().item())
                (
                    persistent_cache,
                    main_n_forwards,
                    main_loss,
                    main_kept_len,
                ) = _forward_mirror_decode_stream_loss(
                    model,
                    input_ids=main_tokens,
                    position_ids=main_positions,
                    past_key_values=persistent_cache,
                    prompt_len=0,
                    pad_len=pad_len,
                    base_attn_impl=base_attn_impl,
                    past_attn_impl=past_attn_impl,
                    loss_fn=loss_fn,
                    full_logit_start=post_start + warm_kept_len,
                    trace_label=(
                        f"C{call_idx}:B2A-main"
                        if trace_forward_this_call else None
                    ),
                )
                cache_token_ids_py.extend(pre_trim_ids_py[main_start:cpe])
                cache_position_ids_py.extend(
                    main_positions[0].detach().cpu().tolist()
                )
                assert warm_kept_len + main_kept_len == post_end - post_start, (
                    f"call {call_idx}: streamed logits len="
                    f"{warm_kept_len + main_kept_len} "
                    f"!= post_end-post_start={post_end - post_start} "
                    f"(ie_pre={ie_pre}, main_start={main_start}, cpe={cpe})"
                )

                _maybe_dump_trainer_kv(
                    persistent_cache,
                    call_idx=call_idx,
                    is_last_call=(call_idx == len(calls) - 1),
                    pre_trim_ids=pre_trim_ids_py,
                    pre_trim_through=cpe,
                    cache_token_ids=cache_token_ids_py,
                    cache_position_ids=cache_position_ids_py,
                    position_ids_this_call=main_positions[0].detach().cpu().tolist(),
                    new_tokens_this_call=main_tokens[0].detach().cpu().tolist(),
                )

                accumulated_loss += main_loss
                n_forwards_run += (
                    prefix_replay_n_forwards
                    + warm_n_forwards
                    + main_n_forwards
                )
                persistent_cache = _detach_dynamic_cache(persistent_cache)
                continue
            else:
                main_logits, main_n_forwards = _forward_with_optional_decode_mirror(
                    model,
                    input_ids=main_tokens,
                    position_ids=main_positions,
                    past_key_values=persistent_cache,
                    prompt_len=0,
                    pad_len=pad_len,
                    mirror_decode=mirror_decode_this_call,
                    base_attn_impl=base_attn_impl,
                    past_attn_impl=past_attn_impl,
                    trace_label=(
                        f"C{call_idx}:B2A-main"
                        if trace_forward_this_call else None
                    ),
                )
                cache_token_ids_py.extend(pre_trim_ids_py[main_start:cpe])
                cache_position_ids_py.extend(
                    main_positions[0].detach().cpu().tolist()
                )

                # Assemble logits at every pre-trim position [cps..cpe).
                # Warmup wrote logits at [cps..ie_pre) in one tensor (no gap);
                # main wrote [main_start..cpe). The main helper already drops
                # auto-pad filler logits while still writing their K/V rows.
                pieces: list[torch.Tensor] = []
                if warm_logits_detached is not None:
                    pieces.append(warm_logits_detached)
                pieces.append(main_logits)
                kept_logits = torch.cat(pieces, dim=1)
                assert kept_logits.shape[1] == post_end - post_start, (
                    f"call {call_idx}: kept_logits len={kept_logits.shape[1]} "
                    f"!= post_end-post_start={post_end - post_start} "
                    f"(ie_pre={ie_pre}, main_start={main_start}, cpe={cpe})"
                )

                _maybe_dump_trainer_kv(
                    persistent_cache,
                    call_idx=call_idx,
                    is_last_call=(call_idx == len(calls) - 1),
                    pre_trim_ids=pre_trim_ids_py,
                    pre_trim_through=cpe,
                    cache_token_ids=cache_token_ids_py,
                    cache_position_ids=cache_position_ids_py,
                    position_ids_this_call=main_positions[0].detach().cpu().tolist(),
                    new_tokens_this_call=main_tokens[0].detach().cpu().tolist(),
                )

                loss_val = loss_fn(kept_logits, post_start, post_end)
                _finish_call(
                    loss_val,
                    prefix_replay_n_forwards
                    + warm_n_forwards
                    + main_n_forwards,
                )
        else:
            # ── Extension (or single first-call without admission): single forward.
            # Plain arange + cps in the normal same-frame case. If this
            # request inherited cached blocks from a nonzero writer frame,
            # mirror vLLM's physical+piecewise-offset RoPE positions for the
            # newly written tail.
            new_tokens = pre_trim_ids[:, cps:cpe]
            if writer_offset != 0:
                physical_start = persistent_cache.get_seq_length()
                new_positions = _piecewise_positions(
                    physical_start,
                    physical_start + (cpe - cps),
                    offset=writer_offset,
                    protected_prefix_len=protected_prefix_len,
                    device=device,
                )
            else:
                position_start = (
                    persistent_cache.get_seq_length()
                    if prefix_replay_tokens_py else cps
                )
                new_positions = (
                    torch.arange(cpe - cps, device=device, dtype=torch.long)
                    + position_start
                ).unsqueeze(0)
            new_prompt_len = max(0, min(sub_end_pre, cpe) - cps)
            if mirror_decode_this_call and stream_mirror_decode_loss:
                (
                    persistent_cache,
                    new_n_forwards,
                    new_loss,
                    new_kept_len,
                ) = _forward_mirror_decode_stream_loss(
                    model,
                    input_ids=new_tokens,
                    position_ids=new_positions,
                    past_key_values=persistent_cache,
                    prompt_len=new_prompt_len,
                    pad_len=pad_len,
                    base_attn_impl=base_attn_impl,
                    past_attn_impl=past_attn_impl,
                    loss_fn=loss_fn,
                    full_logit_start=post_start,
                    trace_label=(
                        f"C{call_idx}:ext"
                        if trace_forward_this_call else None
                    ),
                )
                cache_token_ids_py.extend(pre_trim_ids_py[cps:cpe])
                cache_position_ids_py.extend(
                    new_positions[0].detach().cpu().tolist()
                )
                assert new_kept_len == post_end - post_start, (
                    f"call {call_idx} (no admission): streamed logits len="
                    f"{new_kept_len} != post_end-post_start="
                    f"{post_end - post_start}"
                )

                _maybe_dump_trainer_kv(
                    persistent_cache,
                    call_idx=call_idx,
                    is_last_call=(call_idx == len(calls) - 1),
                    pre_trim_ids=pre_trim_ids_py,
                    pre_trim_through=cpe,
                    cache_token_ids=cache_token_ids_py,
                    cache_position_ids=cache_position_ids_py,
                    position_ids_this_call=new_positions[0].detach().cpu().tolist(),
                    new_tokens_this_call=new_tokens[0].detach().cpu().tolist(),
                )

                accumulated_loss += new_loss
                n_forwards_run += prefix_replay_n_forwards + new_n_forwards
                persistent_cache = _detach_dynamic_cache(persistent_cache)
                continue

            logits, new_n_forwards = _forward_with_optional_decode_mirror(
                model,
                input_ids=new_tokens,
                position_ids=new_positions,
                past_key_values=persistent_cache,
                prompt_len=new_prompt_len,
                pad_len=pad_len,
                mirror_decode=mirror_decode_this_call,
                base_attn_impl=base_attn_impl,
                past_attn_impl=past_attn_impl,
                trace_label=(
                    f"C{call_idx}:ext"
                    if trace_forward_this_call else None
                ),
            )
            cache_token_ids_py.extend(pre_trim_ids_py[cps:cpe])
            cache_position_ids_py.extend(
                new_positions[0].detach().cpu().tolist()
            )
            # No admission → kept_in_call is the full range minus pad
            # tail, so logits now match the post-trim contribution 1:1.
            assert logits.shape[1] == post_end - post_start, (
                f"call {call_idx} (no admission): logits len={logits.shape[1]} "
                f"!= post_end-post_start={post_end - post_start}"
            )

            _maybe_dump_trainer_kv(
                persistent_cache,
                call_idx=call_idx,
                is_last_call=(call_idx == len(calls) - 1),
                pre_trim_ids=pre_trim_ids_py,
                pre_trim_through=cpe,
                cache_token_ids=cache_token_ids_py,
                cache_position_ids=cache_position_ids_py,
                position_ids_this_call=new_positions[0].detach().cpu().tolist(),
                new_tokens_this_call=new_tokens[0].detach().cpu().tolist(),
            )

            loss_val = loss_fn(logits, post_start, post_end)
            _finish_call(loss_val, prefix_replay_n_forwards + new_n_forwards)

    if bptt_windowed:
        _flush_bptt_window()
        if max_bptt_window_forward_passes is not None:
            while bptt_window_idx < len(max_bptt_window_forward_passes):
                _flush_bptt_window(force=True)
    else:
        n_dummy = max(0, max_forward_passes - n_forwards_run)
        if n_dummy > 0:
            _run_dummy_passes_with_backward(
                model,
                merged_input_ids,
                merged_position_ids,
                num_dummy=n_dummy,
            )

    return {
        "loss": torch.tensor(accumulated_loss, device=device),
        "n_segments": n_forwards_run,
    }
