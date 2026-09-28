# Plan: ditch two-phase forward, evict-before-prefill in a single forward

**Status**: Proposed 2026-05-15. Supersedes `plans/markovian_two_phase_prefill.md` for vLLM.
**Scope**: vLLM only. Trainer changes are a follow-up (see §7).

## TL;DR

Move admission-time KV eviction from the worker-side two-phase prefill
into the tail of `Scheduler.schedule()`. The worker becomes ignorant of
compaction — `execute_model` runs a single normal forward over the
already-trimmed prompt against the already-spliced block_table. Deletes
~600-700 LOC of two-phase machinery. Preserves prefix-cache hits across
compaction boundaries (unlike `_maybe_compact_prompt` at submission).

## Why this works (sanity check)

The two-phase forward exists today because the original design wanted to
let `new_user_fragment` (NUF) attend over the kept-context-only K/V
during its own prefill, while still letting kept turns' K vectors be
written under full pre-eviction attention (matching prefix-cache reuse
semantics).

The user's insight: **if you evict `turn_to_evict`'s blocks BEFORE the
prefill kernel runs, NUF's K/V is computed against a block_table that
no longer contains `turn_to_evict` — Markovian semantics fall out for
free, in a single forward, with no attention-mask hackery.**

The kept turns' K vectors:
- Were written in *prior steps* (prefix-cache hits across turns mean
  their K/V is already in pool blocks).
- Stay in place — their blocks are NOT spliced out.
- Their RoPE position encoding matches what they were originally
  written under.
- `position_offset` on the request stays at whatever it was (only
  bumps when this step actually evicts).

The to-be-evicted turn's K/V:
- Was also written in prior steps.
- Gets freed THIS step before the kernel fires.
- Block_table for this request loses those entries.
- `position_offset += total_evicted` so future positions land at
  their original absolute index for RoPE.

The new NUF:
- Prefill kernel writes K/V for NUF positions only (everything else
  is already cached).
- Reads K from `[sys, preserved_turns_kept_in_cache]` blocks only.
- Causal mask prevents NUF→future attention.
- Sample T1 from NUF's last logit. Single forward. ✓

This is strictly *more* Markovian than the current two-phase design,
which preserves kept turns' K vectors under "full pre-eviction
attention" for one step and only retroactively patches them out via
the InputBatch swap. Under the new design, kept turns' K vectors are
the values cached from *prior* steps (when they really were Markovian
w.r.t. the prior turn boundary).

## What survives, what dies

### Keep
- `vllm/v1/core/compaction/manager.py` (`CompactingKVCacheManager`,
  `compact_request`, `_maybe_evict_cached_block`). Unchanged.
- `vllm/v1/core/compaction/types.py` (`CompactionEvent` and all its
  fields including `kept_indices`, `kept_token_ids`, `evict_start`,
  `new_user_fragment_len`). Wire contract with trainer unchanged.
  `new_user_fragment_len` becomes pure trainer metadata (vLLM no
  longer uses it internally).
- `Request.position_offset` and the `position_offsets_cpu/gpu`
  plumbing in `gpu_input_batch.py` / `gpu_model_runner.py`. Required
  for RoPE under any eviction (admission or mid-gen).
- `Scheduler._pending_admission_compaction_ids` (set). Gates which
  requests get in-step eviction at prefill-completion.
- `Scheduler._run_admission_eviction_loop` and
  `_compact_request(post_prefill_admission=True)`. The state-mutation
  primitive. Already loops until under `max_turns`.
- `Scheduler._apply_trim`. Shared trim helper for admission and
  mid-gen.
- `Scheduler._plan_turn_evict_range`, `_scan_new_turn_boundaries`,
  `_num_live_completed_turns`, `_rehash_after_eviction`. Eviction
  geometry helpers.
- Mid-gen branch in `update_from_output` (scheduler.py:2711-2733).
  Unchanged.
- Async-scheduling and LMCache startup refusals
  (scheduler.py:127-140, ~292).
- Preempt-compacted-request → abort (scheduler.py:1077-1089).

### Delete entirely
- `vllm/v1/worker/compaction_forward_controller.py` — entire file
  (354 LOC).
- `VLLM_TWO_PHASE_PREFILL` env var.
- `GPUModelRunner._run_phase2_forward` (~90 LOC at gpu_model_runner.py
  lines 3572-3661).
- `GPUModelRunner._run_two_phase_for_call` (if separate from above).
- Two-phase dispatch block in `execute_model`
  (gpu_model_runner.py:4180-4218).
- Between-phase patches + phase-2 forward + merge block
  (gpu_model_runner.py:4503-4523).
- `CompactionForwardController` import + instantiation
  (gpu_model_runner.py:198-200, 692-695).
- `InlineAdmissionPlan` dataclass (output.py:186-244) — or simplify
  to a small `InlineEvictionDirective` carrying only
  `evict_start`/`evict_end`/`total_evicted` for logging. Default is
  full deletion; only keep if event-emission path needs it.
- `SchedulerOutput.inline_admission_plans` field + its `make_empty`
  default.
- `Scheduler._build_inline_admission_plan` (scheduler.py:1674-1822,
  ~150 LOC).
- `Scheduler._plan_full_admission_eviction` aggregator
  (scheduler.py:1573-1647) IF only called from
  `_build_inline_admission_plan`. Verify with grep before deleting.
- `Scheduler._maybe_compact_prompt` (scheduler.py:2013-2157,
  ~140 LOC). **Confirmed orphaned** — only references are a docstring
  at 1840 and a stale comment at 3064. Dead code.
- `Scheduler._compute_new_user_fragment_len` if only used by the
  plan builders.

### Estimated LOC delta
- Delete: 354 (controller) + 90 (phase2 forward) + 60 (execute_model
  dispatch/merge) + 150 (`_build_inline_admission_plan`) + 75
  (`_plan_full_admission_eviction`) + 140 (`_maybe_compact_prompt`)
  + 60 (`InlineAdmissionPlan` + field) ≈ **930 LOC**.
- Add: ~40 LOC in `Scheduler.schedule()` for the new in-step
  eviction-application loop + `num_scheduled_tokens` fix-up.

Net: **~890 LOC removed**, single new method.

## Design

### Where eviction lands

In `Scheduler.schedule()`, between line 964 (after `scheduled_new_reqs`
is finalized) and line 983 (before `_make_cached_request_data`
snapshots block_ids and computed_tokens for the worker).

```python
# scheduler.py, after line 964, before line 972:

# In-step admission eviction: for any request whose prefill completes
# this step AND that is in the pending-admission set, run the
# eviction loop now so the snapshots below see post-eviction state.
self._apply_inline_admission_eviction(
    scheduled_new_reqs, num_scheduled_tokens
)
```

The new helper:

```python
def _apply_inline_admission_eviction(
    self,
    scheduled_new_reqs: list[Request],
    num_scheduled_tokens: dict[str, int],
) -> None:
    """For each request whose prefill completes this step and which is
    in `_pending_admission_compaction_ids`, run the admission-eviction
    loop (frees blocks, splices block_table, trims tokens, bumps
    position_offset, emits CompactionEvent). Then adjust
    num_scheduled_tokens so the worker prefills the trimmed sequence.
    """
    if not self._compaction_enabled or not self._pending_admission_compaction_ids:
        return

    delta = 0
    for req in scheduled_new_reqs:
        req_id = req.request_id
        if req_id not in self._pending_admission_compaction_ids:
            continue
        num_new = num_scheduled_tokens.get(req_id, 0)
        if req.num_computed_tokens + num_new < req.num_prompt_tokens:
            # Prefill doesn't complete this step (chunked). Defer.
            continue
        if self._compaction_max_turns <= 0:
            self._pending_admission_compaction_ids.discard(req_id)
            continue

        pre_prompt = req.num_prompt_tokens
        self._run_admission_eviction_loop(req)  # mutates req in place
        self._pending_admission_compaction_ids.discard(req_id)
        total_evicted = pre_prompt - req.num_prompt_tokens
        if total_evicted > 0:
            # Worker prefills the trimmed prompt; reduce scheduled count.
            num_scheduled_tokens[req_id] = num_new - total_evicted
            delta -= total_evicted

    if delta:
        # caller updates total_num_scheduled_tokens (see invocation).
        ...  # alternatively return delta and have caller adjust
```

Caller-side adjustment to `total_num_scheduled_tokens` follows the
same accounting pattern the current `restrict_for_phase1` uses.

### What the worker sees

After the new helper runs, when the scheduler builds `SchedulerOutput`
at line 1019:
- `scheduled_new_reqs[i]` is the same Request object, but
  `request._all_token_ids`, `prompt_token_ids`, `num_prompt_tokens`,
  `block_ids_per_group`, and `position_offset` all reflect the
  post-eviction state.
- `num_scheduled_tokens[req_id]` is decremented by `total_evicted`.
- `_make_cached_request_data` (line 983) snapshots block_ids and
  computed_tokens from the post-eviction state — the worker
  receives the right block_table on its first `_update_states` pass.

The worker side of `execute_model`:
- `_update_states` populates InputBatch rows from the (post-eviction)
  `SchedulerOutput`. block_table, num_computed_tokens, position_offset
  all correct.
- `_prepare_inputs` builds cu_seqlens, slot_mapping, position
  tensors. The "scheduled tokens" for this req = NUF length.
  Positions = `[num_computed, num_computed + NUF_len)` in physical
  coordinates; RoPE absolute = physical + position_offset.
- `commit_block_table` uploads the spliced block_table.
- Forward kernel runs once. Sample.
- Output return path unchanged.

### Cascade attention guard

`num_common_prefix_blocks` is computed at scheduler.py:954-963 over
the pre-eviction block tables. If any request gets in-step eviction,
that count is stale (the evicted blocks were possibly part of the
common prefix). Safest fix: when `_apply_inline_admission_eviction`
mutates at least one request, set
`num_common_prefix_blocks = [0] * len(groups)` before building
`SchedulerOutput`. This disables cascade attention for the eviction
step only. RL training workload doesn't rely on cascade attention
across heterogeneous-prefix rollouts.

### Prefix-cache hit math

This is the reason for using the in-step path over re-activating
`_maybe_compact_prompt` at submission:

- **In-step (this plan):** new request submits prompt
  `[sys, T1, T2, T3, asst, T4_user]`. Prefix cache hits on
  `[sys, T1, T2, T3, asst]` (entire previous turn) — only `T4_user`
  needs prefill. At schedule completion, eviction frees `T1`'s
  blocks. Prefill runs over `T4_user` only. **~5x fewer prefill
  tokens than admission-trim.**
- **Submission-trim (Agent 2's path):** trimmed prompt is
  `[sys, T2, T3, asst, T4_user]`. Prefix cache hits on `[sys]` only
  (the rest is at different positions than what's cached). Prefill
  runs over `T2+T3+asst+T4_user`. Loses 75-94% prefix-cache hit rate
  that the project's smoke runs validated.

## Phasing

### Phase 1 — scheduler-side cutover (single commit, ~200 LOC delta)

1. Add `_apply_inline_admission_eviction` method (above sketch).
2. Call it from `schedule()` between lines 964 and 983.
3. Adjust `total_num_scheduled_tokens` accordingly.
4. Set `num_common_prefix_blocks = [0] * len(groups)` when any
   request got in-step evicted this step.
5. Remove the admission-eviction branch in `update_from_output`
   (scheduler.py:2678-2697). Keep the drain-pending logic at 2699-2709
   (handles the "live_turns < max_turns at prefill completion"
   no-op case). Keep mid-gen branch 2711-2733.
6. The `inline_admission_plans` field on `SchedulerOutput` is still
   populated by `_build_inline_admission_plan` (so the worker can
   still see it for log/A/B comparison purposes for one merge).
7. **Success criterion**: Phase-3 smoke (5-step, batch_size=64,
   rollouts_per_example=8, window=4096, stride=512) runs to
   completion with identical mismatch KL ≤ kernel floor 0.001 to
   pre-change baseline. Logs show single forward per step, no
   `[TWO-PHASE/…]` lines.

### Phase 2 — worker-side deletion (single commit, ~500 LOC delta)

1. Delete `vllm/v1/worker/compaction_forward_controller.py`.
2. Strip `gpu_model_runner.py`:
   - Lines 198-200: import.
   - Lines 692-695: instantiation.
   - Lines 4180-4218: `two_phase_active` dispatch block.
   - Lines 4503-4523: between-phase patches + phase-2 + merge.
   - Lines 3572-~3661: `_run_phase2_forward` (and
     `_run_two_phase_for_call` if separate).
3. Re-run the same smoke from Phase 1. Identical KL.
4. **Success criterion**: ripgrep for `two_phase|TWO_PHASE|phase2|
   restrict_for_phase1|apply_eviction_patches|merge_logits|
   CompactionForwardController` under `vllm/v1/worker/` returns zero
   hits.

### Phase 3 — wire-type and dead-code cleanup (single commit, ~300 LOC delta)

1. Delete `InlineAdmissionPlan` dataclass and import in
   `vllm/v1/core/sched/output.py`.
2. Delete `SchedulerOutput.inline_admission_plans` field and any
   `make_empty` reference.
3. Delete `Scheduler._build_inline_admission_plan` (lines 1674-1822).
4. Delete `Scheduler._plan_full_admission_eviction` IF unused after
   step 3 (grep first).
5. Delete `Scheduler._compute_new_user_fragment_len` IF unused.
6. Delete `Scheduler._maybe_compact_prompt` (lines 2013-2157,
   confirmed orphan).
7. Remove stale `[TWO-PHASE/…]` log strings everywhere.
8. **Success criterion**: ripgrep for `inline_admission_plans|
   InlineAdmissionPlan|_maybe_compact_prompt|_build_inline_admission_plan`
   returns zero hits.

### Phase 4 — verification

1. Existing tests: `pytest tests/v1/core/test_scheduler_compaction.py
   -v` should pass after updates to assertions that referenced
   two-phase artifacts.
2. New unit test
   `tests/v1/core/test_inline_admission_single_forward.py`:
   - Build a 5-turn prompt, `max_turns=3`, `stride=1`. Schedule once.
   - Assert: post-`schedule()`, `request.num_prompt_tokens` decremented,
     `request.position_offset > 0`, `request.block_ids_per_group`
     shrank, `SchedulerOutput.num_scheduled_tokens[req]` reflects
     trimmed prompt.
   - Mock the forward; assert it's called exactly once with cu_seqlens
     consistent with the trimmed prompt.
3. End-to-end smoke: 5-step textworld compaction-RL on
   `experiments/textworld-markovian-thinker/rl_cluster_turns2_stride1_seed0.toml`.
   Assert: mismatch KL ≤ 0.001 (kernel floor), peak memory not worse
   than baseline two-phase run, prefix-cache hit rate stays in
   75-94% range.

## Risk register

1. **`scheduled_new_reqs` / `scheduled_cached_reqs` snapshot timing**
   — eviction MUST run before `_make_cached_request_data` at line 983.
   Verified: the insertion point at lines 964-972 sits between
   `scheduled_new_reqs` finalization and the snapshot. Mitigation:
   add an assertion immediately before the snapshot that
   `_pending_admission_compaction_ids` is either empty for
   prefill-completing reqs or that they've already been processed.

2. **`num_scheduled_tokens` desync** — if eviction trims the prompt
   by N tokens, the prefill must process N fewer tokens. The new
   helper decrements `num_scheduled_tokens[req_id]` by exactly
   `total_evicted`. `total_num_scheduled_tokens` decremented by
   the sum. Mitigation: assert pre/post sums match.

3. **`needs_rebuild` flag and worker block_table re-upload** —
   `_run_admission_eviction_loop` already sets
   `request.needs_rebuild = True` (scheduler.py:1295). Verify the
   worker's `_update_states` honors this on the SAME step (not the
   next). The existing rebuild path is used by mid-gen compaction,
   so it works for that case; need to confirm it triggers on the
   first step a request appears with the flag set.

4. **Cascade attention `num_common_prefix_blocks` over-count** —
   computed pre-eviction. Mitigation: zero out the array when any
   request got in-step evicted (see Phase 1 step 4).

5. **GPU upload of patched block_table** — today `_prepare_inputs`
   calls `input_batch.block_table.commit_block_table(...)`. With
   the new design, the patched block_table is installed by
   `_update_states` from the `CachedRequestData.new_block_ids` /
   `rebuild_req_ids` path, which already runs before
   `_prepare_inputs`. So the existing commit covers it. Mitigation:
   trace `_update_states` → `block_table.add_row` to confirm full
   replacement (not append) on rebuild.

6. **Block_pool prefix-cache hash collisions** — when a block in
   the evict range has `ref_cnt > 1` (shared via prefix cache),
   `free_blocks` decrements without returning to pool; the hash
   entry is cleared via `_maybe_evict_cached_block`
   (manager.py:167-169). Existing behavior; unchanged.

7. **Chunked prefill** — if the prefill doesn't complete this step
   (`num_computed + num_new < num_prompt`), the new helper defers
   eviction to the step that does. Same gate as the current
   `_build_inline_admission_plan`. Test case in Phase 4.

8. **Trainer-side mirror (out of scope but blocking)** — vLLM-side
   wire format unchanged (`CompactionEvent`), but the trainer's
   `segmented_forward` currently expects the kept-turn K vectors to
   match a "full pre-eviction context" view (Path 2 semantics). After
   this change, vLLM's kept-turn K vectors come from the prior step
   (Markovian w.r.t. the prior boundary). The trainer's segment-0
   forward over the trimmed prompt produces K vectors under the same
   regime, so a trainer that ALSO does `_apply_trim`-equivalent at
   admission will match. Confirm before running end-to-end RL.

9. **PP / TP** — current project is single-GPU per worker. Add
   `assert pipeline_parallel_size == 1 if compaction_enabled` to
   make the scope-out explicit.

10. **Async scheduling and LMCache** — keep existing startup
    refusals. The new path doesn't lift either constraint.

## Open questions

1. Does `_run_admission_eviction_loop`'s `needs_rebuild = True` flag
   propagate to the worker on the SAME `schedule()` call's
   SchedulerOutput, or does it only take effect on the NEXT call?
   Need to verify by tracing `_make_cached_request_data` →
   `rebuild_req_ids` → worker `_update_states`. If it's
   next-call-only, the new helper must also call the appropriate
   block_table refresh path inline, or move the eviction one step
   earlier in `schedule()` to land before any snapshot happens.

2. Is `_plan_full_admission_eviction` called from anywhere besides
   `_build_inline_admission_plan`? Grep before Phase 3 deletion.

3. Should `InlineAdmissionPlan` be fully deleted or simplified to a
   debug-only `InlineEvictionDirective` for logging? Default: full
   deletion (Phase 3). Re-add only if tests/logs need it.

4. Async scheduling: now that the in-step admission compaction path
   no longer runs in `update_from_output`, the startup refusal could
   be re-litigated. Mid-gen still uses `update_from_output`, so for
   safety keep the refusal until mid-gen is also migrated. Separate
   followup.

## Trainer-side mirror (Phase 5) — collapse to single forward per call

### Status

**Implemented 2026-05-15.** See `src/kv_eviction/segmented_forward.py`'s
`per_call_segmented_forward` and `tests/test_per_call_segmented_forward.py`.

### TL;DR

vLLM achieves single-forward-per-admission via three coupled pieces:
prefix cache holds K from prior turns, `block_table.splice()` drops
evicted K before the kernel runs, and `position_offset` bumps Q's RoPE
frame to match the kept K's pre-eviction rotations. The trainer's
analog is structurally identical with one simplification: **the evicted
tokens are absent from `merged_input_ids` (the orchestrator's
`_apply_admission_trim` already removed them), so no explicit cache
splice is needed.** A single HF forward over the post-trim sequence
with eviction-aware `position_ids` reproduces inference exactly.

The key insight: when input_ids = post-trim and position_ids carry an
offset bump at the admission boundary, the causal mask + RoPE relative
positions over the kept tokens produce identical K vectors and
identical attention to what vLLM had after splicing.

### Current state — what was redundant (now deleted)

The pre-refactor `per_call_segmented_forward` did one of two things per
call:
- **Non-admission**: forward `merged_input_ids[:, :owned_end]` — the
  ENTIRE merged prefix through this call's end. Slice owned logits
  at `[cum_offset, owned_end)`. Backward. O(N²) total FLOPs across N
  calls.
- **Admission**: `_run_two_phase_for_call` — phase 1 forwarded
  `submitted[:phase1_token_count]` (= the full pre-fragment prompt
  INCLUDING the about-to-be-evicted turn) with NO inherited cache;
  spliced K at `[evict_start, evict_end)`; phase 2 forwarded
  `[new_user_fragment + completion]` against the spliced cache. ~2x
  the FLOPs of a single forward.

Both branches recomputed K for tokens that prior calls already
produced K for — the trainer was effectively doing "no prefix cache"
inference.

### What we changed — persistent DynamicCache + eviction-aware position_ids

The trainer's analog of vLLM's prefix cache is a Python-side
`DynamicCache` (HF's class) that survives across iterations of the
call loop. The cache is detached between calls (preserves the
per-segment-backward / bptt_segments=1 / M3 semantics — gradients do
not flow across call boundaries).

Per-call protocol (replaces both former branches):

```python
persistent_cache = DynamicCache()
cum_offset = 0
cum_position_offset = 0  # accumulated total_evicted across prior admissions
for call in calls:
    if has admission:
        desc = _aggregate_admission_descriptor(call)
        L1_kept = (sub_len - nuf_len) - desc.total_evicted
        P2 = desc.new_user_fragment_len + len(call.completion_ids)
        # Pre-admission positions: arange(L1_kept) + cum_offset + cum_position_offset
        # Post-admission positions: arange(P2) + cum_offset + L1_kept + cum_position_offset + total_evicted
        call_position_ids = concat([pre_positions, post_positions])
        new_position_offset = cum_position_offset + desc.total_evicted
        call_merged_len = L1_kept + P2
    else:
        call_input_len = len(call.submitted_prompt_ids) + len(call.completion_ids)
        call_merged_len = call_input_len - cum_offset
        call_position_ids = arange(call_merged_len) + cum_offset + cum_position_offset
        new_position_offset = cum_position_offset
    new_tokens = merged_input_ids[:, cum_offset:cum_offset + call_merged_len]
    out = model(input_ids=new_tokens, position_ids=call_position_ids,
                past_key_values=persistent_cache, use_cache=True)
    loss = loss_fn(out.logits, cum_offset, cum_offset + call_merged_len)
    loss.backward()
    cum_offset += call_merged_len
    cum_position_offset = new_position_offset
    persistent_cache = _detach_dynamic_cache(persistent_cache)
```

One HF forward per call, regardless of admission. No cache splice
needed — the evicted tokens are already absent from
`merged_input_ids` (orchestrator's `_apply_admission_trim` removed
them), so the causal mask + RoPE relative positions over the kept
tokens reproduce vLLM's post-eviction state automatically.

### Why the math works (the parallel to vLLM)

| What vLLM does                                                   | What the trainer does                                                |
|------------------------------------------------------------------|----------------------------------------------------------------------|
| Prior turn's prefill wrote K for [sys, T_kept, asst_prev]        | Prior call's forward wrote K for [sys, T_kept, asst_prev]            |
| K vectors have RoPE rotations frozen at their original positions | Same — DynamicCache's K tensors retain their rotations               |
| Prefix-cache HIT inherits cached K at the same positions         | Persistent cache is THE same Python object — no lookup needed        |
| `block_table.splice(evict_start, evict_end)` drops evicted K     | Evicted tokens already absent from `merged_input_ids` — no splice    |
| Single kernel forward over `new_user_fragment + completion`      | Single `model(...)` forward over the post-trim new-token range       |
| Q rotated at `physical + position_offset` (= absolute frame)     | Q rotated at `cum_offset + cum_position_offset + i` (= absolute)     |
| New K vectors written at `physical + position_offset`            | New K written at the eviction-aware position_ids                     |
| Causal mask + RoPE relative positions reproduce attention        | Same — relative offsets match exactly                                |

Critically: the position_ids passed to the trainer's `model(...)` call
must match vLLM's absolute RoPE frame. **The orchestrator's
`merged_position_ids` is a plain arange over the post-trim sequence —
WRONG for samples with admission.** Instead, the trainer constructs
position_ids on-the-fly from `cum_offset + cum_position_offset`,
bumping `cum_position_offset` by `total_evicted` at each admission
boundary. For non-admission samples (cum_position_offset == 0
throughout), this reduces to `arange(merged_seq_len)`, consistent
with the legacy behavior.

The original Phase 5 sketch was hand-wavy about position_ids — it
assumed `merged_position_ids = arange` worked because the
"pre-eviction merged frame" placed evicted tokens at their original
positions. But the orchestrator delivers POST-trim merged with no
gaps, so the trainer must reconstruct the absolute positions itself.

### What got deleted

- `_run_two_phase_for_call` (~185 LOC). Replaced by an inline ~30-line
  branch that constructs eviction-aware position_ids and runs one
  forward. The aggregation helper `_aggregate_admission_descriptor`
  stays — still needed for `evict_start`, `evict_end`,
  `total_evicted`, `new_user_fragment_len`.
- `_PerCallAdmissionUnsupported` exception class (no longer raised).
- The `KV_EVICTION_DEBUG_TWOPHASE` dump block.
- The phase-1-keeps-pre-and-post-of-evict logit slicing — single
  forward produces only the kept tokens' logits directly.
- Phase 1's whole submitted-prompt forward (including evicted tokens
  for K reconstruction). Not needed anymore — the persistent cache
  carries K from prior calls (when extension holds), and for
  admission-at-start-of-sample, the causal mask over the post-trim
  sequence produces the same K vectors that vLLM's spliced cache had.

### What stays

- `_aggregate_admission_descriptor` (collapses multi-iteration vLLM
  eviction loops into one descriptor).
- `_splice_dynamic_cache` (kept for diagnostic tooling and as a
  building block; not used by the production per-call path).
- `_detach_dynamic_cache` (called between calls to enforce
  bptt_segments=1 / M3 semantics).
- The dummy-pass padding for FSDP2 sync.
- The legacy `segmented_forward` block-FIFO path (still handles
  mid-gen eviction; out of scope here because mid-gen is not used
  in the project's RL training config today).

### Boundary-token re-feed: not needed for inter-call boundaries

The legacy `segmented_forward` re-feeds the last token of each
segment in the next segment so its logit (which predicts the first
token of the next segment) is recomputed under post-eviction
context. That re-feed exists for MID-GEN events where eviction fires
DURING a single call's decode — the logit at the boundary genuinely
needs recomputation under the new K state.

For ADMISSION events (between calls): the boundary between calls is
NOT a "logit predicting the next token under different context"
boundary. The last token of call k-1 is an assistant response token;
its logit was sampled by vLLM during call k-1's decode under call
k-1's K state — and that's exactly what we want for loss. The first
token of call k's new range is part of call k's `new_user_fragment`
(= user input, not predicted-from-loss), so we don't need any
boundary logit. No re-feed required for the per-call path.

### Position_ids handling

`merged_position_ids` is a plain `arange(merged_seq_len)` (line
1267 of segmented_forward.py confirms this). That's correct because
the merged trajectory keeps pre-eviction layout: tokens that get
evicted still occupy their original positions in `merged_input_ids`
even though the K cache later drops them. So position_ids[i] = i
matches vLLM's absolute RoPE positions naturally.

After a splice, the persistent_cache's K tensors are dense in memory
(no positional gap in storage), but each K's frozen RoPE rotation
reflects its ORIGINAL position. Q for new tokens at the next-call's
position_ids attends those rotated K vectors with the correct
relative offsets — exactly as in vLLM.

### Phasing

#### Phase 5A — refactor `per_call_segmented_forward` to persistent-cache

1. Replace the non-admission branch (lines 1308-1366) with the
   persistent-cache forward described above.
2. Replace the admission branch (lines 1287-1307) with: splice
   persistent_cache + single forward over new tokens + loss +
   backward + detach cache.
3. Delete `_run_two_phase_for_call` entirely.
4. Plumb the persistent_cache through the loop. Detach between
   iterations (already-used helper `_detach_dynamic_cache`).
5. Keep `_aggregate_admission_descriptor` and `_splice_dynamic_cache`
   intact.

#### Phase 5B — adjust call_end semantics for admission

For non-admission calls, `call_end = call_input_len` (the merged
prefix end). For admission calls, `call_end` must reflect the
merged-frame end AFTER accounting for the eviction's
position-offset effect. Two viable conventions:

- **Pre-eviction merged frame** (recommended): `merged_input_ids`
  includes all tokens vLLM ever processed in their original
  positions. Evicted tokens stay in the trajectory but their K
  gets spliced out of the persistent cache when the admission event
  fires. `call_end = end of this call's contribution to merged in
  the pre-eviction frame`. The trainer NEVER re-forwards evicted
  tokens (their K from prior call's forward is what gets spliced).
- **Post-eviction merged frame**: trim evicted tokens from
  `merged_input_ids`. Position_ids develop gaps. The cache splice
  becomes trivial (cache is already aligned with merged).

Phase 5A picks pre-eviction merged (the simpler change — matches
what segmented_forward.py already assumes per the docstring at
line 285-288). The orchestrator at
`prime-rl/src/prime_rl/orchestrator/trajectories.py` must produce
pre-eviction-merged trajectories; audit before refactoring.

#### Phase 5C — naming + docstring sweep

1. Rename `_run_two_phase_for_call` → DELETED (no longer exists
   post-Phase 5A).
2. Variable names in the refactored loop body: `persistent_cache`,
   `cum_offset`, `call_end`, `new_tokens`, `new_positions`. No
   "phase 1 / phase 2 / TWO-PHASE" references anywhere.
3. Update `_aggregate_admission_descriptor` docstring: rationale
   is now "vLLM may iterate eviction multiple times within one
   admission call; trainer mirrors the AGGREGATE splice once
   before the call's single forward".
4. Remove all references to "Phase D", "vLLM two-phase",
   "CompactionForwardController" in `src/kv_eviction/` comments.
5. Update `KV_EVICTION_DEBUG_TWOPHASE` → `KV_EVICTION_DEBUG_TRAINER_SPLICE`
   (or remove the env var entirely if no scripts depend on it).
6. **Success criterion**: ripgrep for "two_phase", "TWO-PHASE",
   "phase 1", "phase 2", "CompactionForwardController" in
   `src/kv_eviction/` returns zero hits.

#### Phase 5D — KL parity verification

1. Run `experiments/textworld_env/compaction_debug.py` against the
   already-fixed vLLM. Save the full trajectory + compaction_events.
2. Drive the refactored `per_call_segmented_forward` against that
   trajectory.
3. Compute KL between inference and trainer logprobs at every
   sampled position. Per-call and aggregate.
4. **Success criterion**: max KL ≤ 0.001 (kernel floor). Mean KL
   close to numerical noise. No outliers ≫ 0.005.
5. If KL exceeds the floor, the persistent-cache splice math
   differs from vLLM somewhere — likely position_ids handling for
   admission calls, or the splice range semantics.

#### Phase 5E — end-to-end smoke

Run 5-step textworld compaction smoke
(`experiments/textworld-markovian-thinker/rl_cluster_turns2_stride1_seed0.toml`)
post-refactor:

- Mismatch KL ≤ 0.001.
- Peak GPU memory: should be LOWER than baseline (no redundant
  phase-1 forward, no merged-prefix re-forward).
- Trainer step time per rollout: should be FASTER (single forward
  per call vs. two-forward-or-merged-prefix-forward).
- Reward curve matches pre-refactor baseline within seed variance.

### Risk register

1. **Persistent cache + FSDP2 interaction.** FSDP2 hooks fire on
   each `model(...)` invocation to all-gather params. The
   `past_key_values` argument is a non-parameter tensor input —
   FSDP2 doesn't shard it. The per-segment backward already invokes
   `model(...)` repeatedly under FSDP2; the new per-call pattern is
   structurally identical. Risk: low. Mitigation: existing per-call
   FSDP2 tests cover this.

2. **Cache detach between calls.** `_detach_dynamic_cache` returns a
   fresh `DynamicCache` with detached tensors. The autograd graph
   from call k's forward gets freed when call k's backward
   completes; the cache that feeds call k+1 starts with leaf
   tensors. This is the same pattern as the legacy
   `segmented_forward`'s per-segment-backward path. Risk: low.

3. **Splice semantics for multi-event calls.** vLLM's eviction loop
   may fire 2+ admission events for ONE call (each loop iteration
   evicts one stride's worth of turns). The trainer collapses
   these to a SINGLE splice via `_aggregate_admission_descriptor`.
   Both produce the same final K state because the splice is
   idempotent under "drop range [a, b) then drop range [b, c)" =
   "drop range [a, c)". Pre-existing logic; no change needed.
   Verify with a 3-event-per-call regression test.

4. **Boundary token at the END of a call.** The last token of call
   k's new range is part of asst_k — its logit was sampled under
   call k's K state. Trainer's persistent-cache forward produces
   this logit during call k's forward (it's the last position of
   the forward output). No special handling needed.

5. **Mid-gen events on calls that also have admission events.**
   The plan assumes admission events arrive ONLY on calls without
   mid-gen events. Production RL training disables mid-gen anyway
   (compaction_mode = "turn" exclusively). Add an assertion
   forbidding mixed admission + mid-gen on the same call. If the
   project ever enables mid-gen + admission on the same call,
   that's a separate phase to handle.

6. **Numerical drift from cache reuse vs fresh recompute.** This
   is what KL parity verification (Phase 5D) catches. Expected:
   zero drift because the cache K vectors are bit-identical
   between "fresh forward" and "reused from prior call's forward"
   when the model and inputs match.

### Expected wins

- **~50% FLOPs reduction** per admission call (delete phase 1).
- **O(N) total FLOPs reduction** across an N-call rollout (no more
  merged-prefix re-forward in non-admission calls).
- **Cleaner code**: ~200 LOC deleted from segmented_forward.py.
  No "two-phase" terminology anywhere.
- **Memory parity or better**: persistent_cache holds the same
  KV vectors that would have been recomputed; per-segment-backward
  still bounds activations to O(1 segment).
- **Mathematical parity with vLLM**: the trainer is now structurally
  identical to inference — single forward per call, pre-prefill
  splice for admission, persistent KV cache across calls.

### Open questions

1. **Is the orchestrator's `merged_input_ids` already in
   pre-eviction frame?** Need to verify by reading
   `prime-rl/src/prime_rl/orchestrator/trajectories.py`'s sample
   construction. If it's post-eviction (evicted tokens removed),
   Phase 5B is more involved (need to either reconstruct the
   pre-eviction trajectory or change the trainer to handle
   post-eviction-merged with position_id gaps).
2. **Are there any consumers of the deleted `_run_two_phase_for_call`
   outside `per_call_segmented_forward`?** ripgrep before deletion.
3. **Does the dummy-pass padding for FSDP2 (line 1370-1377) need
   recalibration?** Under the new per-call protocol, each call
   counts as 1 forward (not 1 or 2). Set
   `max_forward_passes = num_calls` and pad dummies to that count.

## Out of scope (vLLM-side cleanup)

- Mid-gen compaction migration into `schedule()` (currently fine in
  `update_from_output`).
- Lifting async-scheduling and LMCache refusals.
- FA3-vs-FA2 kernel divergence on H100 SM90.

## Comparison to alternatives

| Design | LOC | Single forward | Prefix-cache hits | Markovian | Trainer impact |
|---|---|---|---|---|---|
| **Two-phase (current)** | +900 | No (2 forwards) | Yes (75-94%) | Yes | Phase-F-2 split |
| **This plan (in-step)** | -890 | Yes | Yes (75-94%) | Yes | Trim segment-0 only |
| **Re-activate `_maybe_compact_prompt`** | -1100 | Yes | No (sys-only ~25%) | Yes | Same as in-step |

The in-step design is the best trade-off for this project's RL
training workload.
