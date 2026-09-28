# Flex full-BPTT trainer/inference alignment — PLAN (2026-06-10, user sign-off; SCOPE CORRECTED same day)

User sign-off 2026-06-10 on decisions D1–D7. **SCOPE CORRECTION
2026-06-10 (user, verbatim): "that is the whole point of everything we
want to do here — we want to make the trainer and the inference,
specifically for the managed context, match."** The target is the
MANAGED-CONTEXT arm (hidden-KV recall/restore, protect-oldest,
CPU-offloaded archives) — not just admission-only window/stride
eviction. The interval-visibility mask and the restore-event wire format
are the core deliverable, not extensions.

## Signed-off decisions

- **D1 — Objective (CORRECTED): the trainer's single full-chain flex
  forward must reproduce the managed-context arm's EXACT per-token KV
  visibility timeline** — evictions (window/stride + turn-based),
  protect-oldest anchor, and hidden recall (spans leaving and
  RE-ENTERING visibility). Visibility is non-monotone: per-token
  INTERVALS, not death indices. "Eviction/recall = attention mask over
  one stream" (the KV-continuity contract) — the flex mask is the
  literal implementation of that contract, and today it only implements
  the eviction half.
- **D2 — Hybrid**: flex trainer + FLASH inference. Flash stays serving
  (flex serving ~1.85x wall). Flex-flex only ever as measurement.
- **D3 — ALWAYS FULL BPTT** (user directive): full-BPTT is the desired
  gradient semantics; M3 `bptt_segments=1` was a memory compromise. The
  flex dispatch validator already forces `bptt_segments=-1`.
- **D4 — Memory**: per-block AC is REQUIRED on the flex path (Gate 0
  result below) and legal (validator exemption, prime-rl
  trainer.py:1169 — use_cache=False ⇒ no DynamicCache double-append).
- **D5 — Numerics kill criterion**: flex-flex floor must be ≥5x below
  the 0.0009 flash-flash floor to claim a numerics win; expected dead
  (bf16 reduction-order noise). Hybrid stands on structural grounds.
- **D6 — Defaults frozen**: dispatch defaults unchanged; adoption is
  explicit in TOMLs; own branch (feat/flex-full-bptt-alignment); the
  speed-opt benchmark footing untouched.
- **D7 — Bit-exactness**: out of scope.
- **D8 — Managed-context coverage (CORRECTED): YES — it is the point.**
  Trainer alignment targets recall-bearing managed-context episodes.
  Death-index-only alignment (rg-mix window/stride) is at most an
  intermediate checkpoint.

## Why the existing assets are NOT sufficient (verified 2026-06-10)

- The 233 captured snapshots (replaycap 2026-06-06) come from the
  compact-replay segmented RE-PREFILL arm — machinery the KV-continuity
  contract has since forbidden — and predate protect-oldest (shipped
  06-09). The dumper only fires at that arm's activation site
  (vllm scheduler.py:4446); it cannot capture the modern arm.
- The snapshot schema and BOTH mask builders are die-once:
  - engine: replay_snapshot.death_indices, single death per token;
  - trainer: `_build_flex_mask_writer_timeline`
    (src/kv_eviction/segmented_forward.py:2326+) — `_splice_live_cache`
    marks rows dead, no resurrection path.
- Restores are not events: `ManagedContextActiveRestore`
  (vllm scheduler.py:194) records span_ids/num_tokens/created_at but NOT
  the output-token index at which the restore became visible. The only
  client-visible restore payload is `managed_context_restore_kind`
  ({kind, spans, resident, h2d} movement verdict — no timing).
  CompactionEvent carries `archived_span_ids` (types.py:105) at eviction
  time only. The trainer consumer (env.py:151-301,
  segmented_forward.py:1865+) is eviction-only.
- Span identity in writer coordinates ALREADY EXISTS engine-side:
  `ManagedContextSpan` (vllm scheduler.py:157) has evict_start/evict_end,
  position_offset_frame, token_ids, recall_count. The missing piece is
  restore TIMING + a client channel.

## Build phases (managed-context core)

### M0 — wire format: restores become events (vLLM owns truth)
Engine-side. Record, at restore-attach time
(vllm scheduler.py:~11290), the request's current writer-timeline
position (prompt+generated length at attach), and emit a restore event
alongside compaction_events:
`{type:"restore", visible_at:<writer_pos>, span_ids:[...],`
`spans:[{span_id, evict_start, evict_end}], num_tokens}`.
Symmetrically, span re-eviction/release must carry its writer position
(re-death). Consumers (env.py → types.py wire format → rollout calls)
plumb it through like CompactionEvents — consumers consume, never
re-derive. Protect-oldest needs no new wire: protected tokens simply
never die.

### M1 — interval mask (trainer + replay harness)
Generalize the death-index timeline to visibility intervals:
per-token list of [birth, death) windows; eviction-only tokens have
exactly one; restored spans have 2+ (birth at original write, death at
evict, re-birth at visible_at, ...). Carrier: K interval slots
(K = 1 + max recalls per token; clamp + assert), tensors
birth[K,N]/death[K,N]; mask_mod:
`(kv<=q) & any_i(birth_i[kv] <= q < death_i[kv])`.
Touch points: `_build_flex_mask_writer_timeline` + mask construction in
`flex_mask_segmented_forward` (+ batched/packed/selected variants), and
the offline harness (replay_reprefill_flex.py successor).
RoPE check rides along: restored K keeps its original absolute
positions (no recompute), so plain-arange writer positions should STILL
hold under recall — the top-1 metric in M2 validates this empirically
(a collapse on post-restore segments means the position story is wrong).

### M1 — STATUS: BUILT + UNIT-VALIDATED 2026-06-10 (commit 7a9888c)
Timeline builder records span rows at splice time (archived_span_bounds
map VERBATIM through live_writer_indices — same pre-event frame as
evict_start, no coordinate transform); kind-1/2 ops resolve per call
with two scope rules: births CLAMP to the call's first appended row
(prefix-cached rows' queries already ran without the restore — an
upfront attach boundary 0 means "everything this request computes"),
and intervals CLOSE at request end (visibility never crosses calls;
the session arm will need cross-call carry — builder raises loudly on
that shape). Mask: `_build_visibility_mask_mod` K-slot birth/death
tensors OR'd over the death mask; eviction-only timelines compile the
identical death-only mask. Batched/packed/selected variants
NotImplementedError on interval timelines (need per-segment row
offsets). 7 unit tests (tests/test_flex_interval_visibility.py); 36
existing tests pass (test_empty_boundaries_raises fails pre-existing).

### M2 — modern capture + offline validation (cluster jobs)
LAUNCH POLICY (user 2026-06-10): GPU validation runs go through cluster
jobs (4 GPUs: 2 trainer + 2 inference, launch via
experiments/debug_balrog/launch_cluster.py runtime contract), NOT the local
box (user debugs there). Keep jobs fast/targeted.
REFINED APPROACH: the RL loop itself is the measurement — a 1-3 step
RL smoke with (a) inference serving the managed-context recall arm,
(b) trainer on flex dispatch + interval mask. In-loop Mismatch KL on
managed-context episodes IS the hybrid floor; the events→wire→
trajectories→interval-mask chain is exercised end-to-end. Template
configs: experiments/compaction_textworld/rl_cluster*.toml +
experiments/local_2gpu_smoke/ shape. OPEN QUESTION to resolve first:
does the RL-training textworld env (verifiers + src/kv_eviction/env.py
phase4 machinery) drive the managed recall handshake (manager pass +
kve_restore xargs) during training rollouts, or is that
eval_textworld-only? If eval-only, either port the handshake into the
training env or fall back to a client-side capture (calls + events +
logprobs) + offline flex replay.
New env-gated dump (request-finish time, managed-context arm):
writer token_ids + per-token visibility intervals + per-token sampled
LOGPROBS (the 06-06 snapshots lacked logprobs; including them makes the
offline hybrid-floor measurement possible without an RL loop).
Recapture from the counting_managed_prod config
(experiments/textworld_env/results/counting_managed_prod_n128_t20_c64_20260610
— 64 games/64c/20 turns/6-2 schedule/16k, ~15 min local run).
Validate: interval-mask flex forward over captured episodes →
(a) teacher-forcing top-1 (the 233-snapshot analogue, now on the modern
arm; watch post-restore segments specifically), (b) per-token logprob
KL vs the captured flash inference logprobs = THE HYBRID FLOOR on the
real managed-context arm, compared against the 0.0009 reference.

### M3 — training-side gates
- Gate 0 (memory, DONE — see below) remains valid: checkpointed memory
  scales with tokens, not mask shape.
- smoke #5-flex (experiments/smoke5_flex_alignment/, Perlmutter) is
  DEMOTED to a mechanics checkpoint: full-BPTT + AC + FSDP + optimizer
  at production shape, in-loop Mismatch KL on rg-mix (admission-only).
  Useful, not the alignment gate.
- The managed-context alignment gate is M2's offline floor + a
  training smoke that consumes managed-context episodes once a trainer
  data path for them exists (this work IS the "trainer mirror" that
  plans/persistent_sessions.md says must exist before anything that
  trains consumes session mode).

### M2 smoke RESULT (2026-06-10, cluster job 332a379c, SUCCEEDED)
**Eviction-half PASS at kernel floor**: turn-mode 6/2 cooking episodes,
flex full-BPTT trainer (interval-capable path, AC on, 39.4 GiB peak,
2x H100 trainer + 2x H100 inference), 14 compactions/sample avg, ~600
tokens evicted avg — **Mismatch KL 0.0005 (step 0) / 0.0008 (step 1)**
vs the 0.0009 flash-flash reference. The hybrid floor holds at
kernel-floor order on real compaction-heavy episodes IN-LOOP.
**Recall handshake did NOT engage**: zero RESTORE lines server-side,
zero retrieve/recall orchestrator-side — the interval mask was never
exercised on real kind-1/2 events (timelines were eviction-only).
NEXT DEBUG TARGET: why build_managed_context_pre_observation_message
stayed dormant in the env-server subprocess despite the managedctx env
block (check the 5 enablement conditions at src/kv_eviction/env.py:2217
— padding cfg autoconfigure preconditions, recall_mode, index flags —
and whether the inference server's scheduler had KVE_MANAGED_CONTEXT=1,
i.e. did archives get created at all under QUIET logs).
Two config-stack fixes landed on the way: turn-mode branch in the
trainer/inference mirror validator (prime-rl; turn-mode RL was
unlaunchable against the engine's window/max_turns mutual-exclusion
assert), and dataset split sizes. Job lineage: 06ccfe13 (assert),
bdd94622 (dataset), 332a379c (SUCCEEDED).

### M2 v2-v6 debug trail (2026-06-10): recall fires; call-stream gap found
After the client-side schedule vars fix (manager pass needs
KVE_MANAGED_CONTEXT_COMPACTION_MAX_TURNS/TURNS_LAST_KEPT — benchmark
script translates these, the cluster env block didn't), the handshake FIRES
in RL rollouts. The trainer's loud guard then caught a real mismatch,
chased through v2-v6 (excesses 527/255/287/271/495 — engine boundary
fixes for physical-vs-visible frames and finish-time releases landed on
the way and are correct hardening). **v6 diagnostics give the verdict:
the engine event is self-consistent (writer_len=4112, computed=4111,
standard deferred attach) but the trainer's live cache for that call is
3616 — the TRAINER'S CALL-STREAM RECONSTRUCTION IS MISSING ~496 VISIBLE
TOKENS (31 blocks).** The boundary was never wrong; call 13's pre-trim
stream is incomplete. Prime suspect: manager-pass exchanges (pre-obs
memory-manager turn + retrieve JSON — synthetic/machinery turns) are
visible to the ENGINE as part of the request but dropped or folded by
the trajectory->CallWire serialization (cf.
KVE_COMPACTION_EXCLUDE_SYNTHETIC_TURNS and the
tw_pending_preobs_env_response replacement mechanics in textworld_env).
NEXT: trace how manager exchanges serialize into trajectory steps vs
how the phase4 client assembles the actual request prompt; the trainer
stream and the engine stream must agree token-for-token on visible
content (either manager turns are calls, or both sides exclude them).
Job lineage: v2 332a379c (no recall, schedule vars) -> v3 87b70948 ->
v4 a168936c -> v5 fa407d99 (boundary-frame hardening) -> v6 60678c1e
(diagnosis). Eviction-only result stands: Mismatch KL 0.0005-0.0008.

### M2 v8 RESULT (job ca5d631f, SUCCEEDED): mechanics PASS, numerics OPEN
The pre-eviction boundary transform (upfront attaches fire BEFORE the
same call's eviction; subtract overlap with later same-call evictions)
cleared the mapping: recall-bearing episodes now run END-TO-END through
the interval mask + full BPTT, both steps, zero guard trips. BUT
**Mismatch KL = 0.144/0.139** vs 0.0009 — the recall reconstruction is
structurally complete and numerically wrong somewhere (suspects:
restored-span RoPE frame in the trainer (rows keep original arange —
does engine hidden-attach match?), visibility semantics
(attach-boundary precision/clamps), or manager-turn content). v9 (job
09347579, kldebug token) localizes per-call/per-token mismatch.

### M2 v9 LOCALIZATION (job 09347579, kldebug): mismatch = recall calls
KL 0.37/0.18 (run-to-run noisy, sample-dependent). [KVE-TOP-KL] top
tokens: **all concentrated in calls 13-15 — the admission calls with
same-call attach+evict (recall-bearing) — at writer positions
~5000-5800, with trainer logprobs catastrophically low (T_lp -15..-33)
where inference was confident (V_lp ~0)**. Per-token KL up to 33. The
trainer's forward sees a different context than inference for queries
in/after the post-attach segments. Suspects, now ranked: (1) the
interval birth/row mapping puts restored-span visibility on the wrong
rows (boundary transform or birth clamp off — wrong rows attending),
(2) restored-span RoPE frame (trainer arange-original vs engine
attach frame), (3) tail pollution: one wrong visibility window corrupts
everything downstream of call 13. NEXT (fresh session): pull ONE
failing sample's calls+events from a v9-style run, replay offline
through flex_mask_segmented_forward vs the engine logprobs per
position, and diff the mask row-by-row around the attach (the offline
harness pattern from the 233-snapshot validation, now with intervals).
### M2 OFFLINE REPRO ESTABLISHED (2026-06-10, replay_recall_sample.py)
Filesystem transport leaves the raw payloads at
outputs/<run>/rollouts/step_N/rank_R.bin — no dump hook needed (the
zmq-side KVE_DUMP_MICRO_BATCHES hook stays for zmq deployments; the
train-loop attempt was reverted: get_batch() is tensorized).
Repro (sample 0, rank0/step0 of v11): mean |dlp|=0.13, p95=0.35,
**max=24.5 at pos 5344 — EXACTLY ONE PAST the last attach birth
(intervals open at 5343)**, with the spike confined to 5344-5347 (ends
at an <|im_end|>) and elevated-but-smaller error through the last-call
region (5584+). Interval sets attach TWO row groups (1120-1167,
1648-1711) at birth 5343 death 5360 (call end).
HYPOTHESES for next loop (test offline in seconds each): (a) boundary
off-by-one/edge — inference's first post-attach query may not yet see
the spans (shift birth +1..+2 and re-measure); (b) wrong span-row SET
at that attach (drop/swap the second group); (c) restored-span RoPE
frame; (d) the answer-control suffix tokens differ. Iterate via:
CUDA_VISIBLE_DEVICES=<free> uv run python experiments/_local_jobs/
debug/scripts/replay_recall_sample.py --micro-batches
outputs/m2_managedctx_flex_smoke/rollouts/step_0/rank_0.bin --sample 0

### M2 OFFLINE SWEEP RESULTS (2026-06-10): suspects eliminated, one left
Perturbation sweep on the pinned sample (replay_recall_sample.py
--sweep): **interval windows are IRRELEVANT to the error** — drop_all
(no restored visibility at all) gives the same mean |dlp| (0.134 vs
0.130) and the same 5344 spike; birth shifts +-1/+2/+8 change nothing.
ELIMINATED: (a) boundary off-by-N, (b) wrong row set (span rows decode
to correct turn content incl. score lines), (e) missing visible
content — submitted_prompt_ids MATCHES event num_prompt_tokens exactly
on all 16 calls (v6's 4112-vs-3616 was a frame artifact of comparing
request frame to live frame, not missing tokens).
The divergent positions are the RECALL MINI-TURN completions (calls
7/9/11/13, comp 4-12 tokens, e.g. 'assistant\n/10' score replies) —
inference confident, trainer lost WITH or WITHOUT the spans visible.
REMAINING PRIME SUSPECT: **RoPE frame of restored K**. The engine
serves hidden K at the span's ARCHIVED positions
(logical_start_by_group + position_offset_frame — per-request frames,
e.g. T0005 at ~1120+offset of call 10) while the trainer's interval
rows sit at writer-arange (1648-1711). If those differ, attention over
the spans cannot match, explaining why visibility windows do not move
the error. NEXT STEP: emit the span position frame on kind-1 events
(per-span logical_starts + position_offset_frame), recapture, compare
to writer rows offline. If frames differ, decide: trainer positions
interval rows at the engine frame (per-row K position override) OR the
engine attaches hidden K at writer-frame positions
(inherit-the-writer's-frame, consistent with the prefix-cache
philosophy in [[feedback-full-prefix-caching-required]]).

### M2 ROOT CAUSE (2026-06-10): piecewise position-frame drift, not the mask
Final sweep round: span_pos_phys (rows at archived physical frame) makes
error 6x WORSE -> engine serves restored K at writer-frame positions
(offset machinery working as designed); RoPE-of-spans hypothesis dead.
Then the positions check found it: **the recall arm's offset chain is
parity-broken across calls** — call 13's final evict bumps offset
1392->1921 (= +tokens_evicted +1), call 14's bumps 1920->2305 (+384+1),
and CONSECUTIVE REQUESTS DISAGREE (call 13 ends 1921, call 14 starts
1920). The trainer faithfully applies offset_after (rows 5280+ sit at
writer+1), but the engine serves the SAME physical rows to different
requests at DIFFERENT frames (piecewise) — a single flex forward with
one position per row cannot represent that. This explains: mean 0.13
spread over post-admission regions, the spike at the frame seam (5344),
and total insensitivity to interval-mask perturbations.
DESIGN DECISION NEEDED (user): (A) fix the engine so the recall arm's
offset bumps stay block/parity-clean and frames agree across calls
(find why smart-bump adds +1 under the manager mini-turns — see
plans/piecewise_position_offset.md), or (B) accept per-call frames and
represent them trainer-side (per-call query position overrides — flex
forward would need per-(row,call) positions, i.e. the piecewise story
in the trainer). (A) preserves the single-stream contract and is the
philosophically consistent fix; (B) is invasive. Repro for either:
replay_recall_sample.py on rollouts/step_0/rank_0.bin sample 0.

### M2 v12 RESULT (job 818c9d2c): block-aligned bump WORKS, one edge left
KVE_SMART_BUMP_BLOCK_ALIGN: bumps now +16 on-grid (74 firings), in-loop
KL 0.14 -> 0.097/0.100; offline mean |dlp| 0.130 -> 0.068; **drop_all
is now WORSE than the real mask (0.093 vs 0.068) — the interval mask
HELPS for the first time** (restored-span attention productive after
the frame fix). REMAINING: one boundary-adjacent seam — the top
mismatch (|dlp| 21) sits at the FIRST QUERY PAST the attach-window
birth (5136 vs birth 5135; same signature as the pre-fix 5344-vs-5343).
NEXT DISSECTION: deferred-attach edge semantics — likely a +-1 between
the engine's "final visible prompt token decodes under restored KV"
(ready_tokens default = boundary - 1?) and the trainer's
birth-at-boundary mapping; test birth-1 ONLY for deferred (non-zero
boundary) attaches on the new capture, and check whether the engine's
visibility_boundary_computed for deferred attaches should be
boundary-1 (the LOGITS at the last prompt position are produced by the
query AT boundary-1). Trajectory: 0.37 -> 0.14 -> 0.10 in-loop; the
fix also repairs prefix-cache inheritance under the recall arm
(off-grid blocks were skipped by the May skip-rule on EVERY recall
episode — likely a production-benchmark win too).

### M2 DISCRIMINATOR RESULT (2026-06-10): residual is INPUT-DATA-level
per_call_segmented_forward on the pinned sample shows the SAME spike
(max 21.2, same position) and is overall WORSE than flex (mean vs
inference 0.038 vs 0.027) -> the residual seam is NOT a flex
representation limit. Away from boundaries flex is at **keep_both mean
0.0071 — kernel-floor territory on recall-bearing samples**.
Shift test localizes the entire spike to ONE token per mini-turn: the
first content token after `<|im_start|>assistant\n` of the manager
mini-turn reply (T=-22.5 vs V=-1.28 at p=5135; T is confident at 5134
and 5136). Both paths share it; all context perturbations irrelevant;
preceding filler+im_start rows are off-loss. CONCLUSION: the recorded
sample's mini-turn boundary differs from what the engine actually
served/conditioned on at that position (length matches, content
suspect — e.g. answer-control suffix or boundary-token variant in the
served prompt vs the serialized one), OR the inference logprob for the
first sampled token of a deferred-attach request reflects a context
neither path models. NEXT: token-level diff of the served prompt tail
vs the serialized stream around the mini-turn boundary (prompt-echo a
traced run, or compare submitted_prompt_ids tail vs merged stream
text), then close the last token class.
TRAJECTORY (in-loop KL): 0.37 -> 0.14 -> 0.097. Offline flex-vs-
inference mean: 0.13 -> 0.027 (boundary-token class excluded: ~0.007).

### M2 FINAL SEAM STATE (2026-06-10): serialization exact; one-token
### distribution disagreement remains
Token-level diff: call 13's served prompt tail == merged stream,
byte-for-byte (contribution equal). Input-data hypothesis DEAD. The
residual is ONE sampled token: the first content token of the
post-recall observation reply — the model sampled 'Human' (an
off-script repair-style continuation at temp 1.0) with V_lp=-1.28,
while BOTH trainer paths rate it -22.5 under the same visible context
and (flex) the same span visibility. A ~e^21 disagreement on one token
with everything else at floor means inference's DISTRIBUTION at that
step differed: remaining suspects (a) the engine's
logprobs_mode='processed_logprobs' (v12 server args) interacting with
sampling at that step, (b) the physical/logical frame at which
H2D-loaded restored spans attach for that specific query (NO-MOVE
re-splice keeps original logicals; H2D loads allocate NEW blocks —
verify their logical_start stamps), (c) some per-step processor state.
NEXT EXPERIMENT (decisive, needs a live server): replay that exact
request (prompt + kve_restore xargs) against a traced server; compare
first-token logprobs with and without the restore directive, and raw
vs processed logprobs. That isolates engine conditioning entirely from
sample data.
STANDING RESULT: flex full-BPTT on recall-bearing managed-context
episodes = **kernel-floor agreement (mean ~0.007) everywhere except
this one token class**; in-loop KL 0.097 and falling with each fix
(0.37 -> 0.14 -> 0.097).

### M2 -> INFERENCE BUG FOUND (2026-06-10): deferred-attach seam corrupts
### the first decode token — THE TRAINER WAS RIGHT
User call: 'if there is a mismatch, the inference engine may be wrong
and the trainer correct.' Confirmed. Completion-head survey across all
captured samples: attach calls 7/9/11 produce clean '<action>...'
replies; **call 13 (the 4th attach call) completions start MID-TEXT in
every sample — continuing the HIDDEN SPAN's tail content** ('0007, I
have been instructed...' continues span id T0007; '5f33ce78b' continues
a trace id; ' required non-empty summaries...' continues manager
instructions). The engine's first-decode query at that seam is
conditioned as if the restored span's tail were its immediate (most
recent RoPE) context — a position/frame misplacement of the
final-prefill chunk after the deferred H2D attach. The trainer's
-22.5 ('Human' impossible) is the CORRECT distribution; the engine's
-1.28 reflects the corrupted context. This bug degrades EVERY managed
rollout's recall turns (off-script repair-loop replies) — plausibly
part of the managed-vs-full-context reward gap (0.078-0.101 vs 0.134).
WHY CALL 13 AND NOT 7/9/11: tbd — candidates: H2D (cold) vs
GPU-resident attach, 2-span attach composition, or offset state by the
4th recall. NEXT: worker-side dive — where deferred restore resumes
the final prefill chunk after attach (gpu_model_runner hidden_kv
attach path: hidden_kv_block_ids/num_tokens, position computation of
the post-attach chunk), find the frame error, fix engine-side. The
trainer is the reference implementation for the fix (its mask/position
semantics produce the sane distribution).

### M2 ROOT CAUSE #2 CONFIRMED (2026-06-10): managed recall is NOT DP-safe
Worker position trace (KVE_TRACE_HIDDEN_POSITIONS): every chunk's
arithmetic is perfect (rope = visible+offset; hidden in physical/
seq_len) — locally at dp=1. Local dp=1 cooking repro: **completely
clean** (45 <action> replies, ZERO off-script markers, reward 0.27).
The cluster smokes ran inference dp=2: **the managed-context archive is
ENGINE-LOCAL; per-call turns round-robin across engines; per-trace span
counters (T0001..) diverge per engine; a recall directive attaches the
WRONG span's KV on the other engine** -> the off-script wrong-span-tail
continuations, accumulating with turns (hence 4th attach). The trainer
was right; the engine context was corrupted. PRODUCTION (dp=1, the
69.0s arm) unaffected. CONSTRAINT recorded: managed-context inference
requires dp=1 until traces are pinned to engines (router affinity by
trace id) or archives are shared. v13 (job 9a4c9701) reruns the smoke
at dp=1: expectation = off-script gone AND in-loop Mismatch KL collapses
toward the 0.0009 floor (the corrupted seams were the dominant term).

### M2 ROOT CAUSE #3 — THE REAL ONE (2026-06-10, v14 worker traces):
### trainer physical frame != request visible frame in the recall arm
Direct position comparison at the seam: trainer row 4927 at RoPE 4943
(global-live-index 3503 + offset 1440); engine served the same query at
RoPE **5375** (request-visible index 3935 + offset 1440). **432-token
disagreement = the request's visible prompt contains ~432 tokens the
trainer's live reconstruction lacks** — the RECALLED TURNS RENDERED
VISIBLY into the per-call prompt by the recall protocol (engine-side
'recalled' turn accounting in _compaction_recalled_turns; the client
re-renders recalled spans as visible text). The trainer's flex timeline
computes physical positions as the global live index — valid for the
window/stride arm where prompt == global live — but the recall arm's
prompt = sys + kept + RECALLED + new. v6's '4112 vs 3616' was this and
was wrongly dismissed as a frame artifact. THE ENGINE IS THE REFERENCE
(its frames are what K/V were written under).
FIX DIRECTION (next session): the trainer's per-call physical frame
must be the REQUEST frame: derive per-call position bases from the
call's own submitted prompt length (events already carry num_prompt_
tokens + offsets), i.e. positions = request_visible_index + offset
instead of len(live_writer_indices) + offset; equivalently account the
visibly-recalled rows in the live view (they ARE in the submitted
prompt, so _build_pre_trim_plan's new_content/cum accounting must treat
them as part of the call's content, not absent). Validate on the
archived v14 capture (/tmp/m2_captures/v14): expect the 17-31 spikes
and the residual 0.027-0.055 KL to collapse to floor.
NOTE: v13/v14 in-loop KL 0.027-0.055 = the floor blocker is THIS frame
gap now; all earlier classes (DP, off-grid bumps, boundary frames,
H2D) are fixed/exonerated.

### M2 ROOT CAUSE #3 REFINED (2026-06-10): the stream FORKS — repair
### exchanges are dead branches; trainer assumes linear extension
Prefix diff (archived v14 capture, call 13): trainer live and submitted
prompt share only 1266 tokens; the trainer's stream carries the manager
REPAIR exchange ('previous hidden-memory manager JSON was invalid...')
that the client DROPPED when re-rendering the next prompt. The engine's
synthetic event for call 13 says nuf=224 => a 3712-token prefix-cache
hit — impossible against the repair stream (diverges at 1266), possible
ONLY against the CALL-11 branch: **the engine reused the pre-repair
branch's physical blocks; the repair request is an abandoned fork.**
The trainer's linear extension model represents the fork as in-stream
content => wrong content AND the 432-position gap (root cause #3's
measured symptom; positions re-coincide once the fork is handled).
FIX DESIGN (eviction-only path BIT-IDENTICAL by construction):
1. Always maintain expected_v_cache_tokens (currently stitch-trace
   DIAG-only) in _build_pre_trim_plan.
2. Per call, verify sub[:cum_post] == expected. NO DIVERGENCE => exact
   existing behavior (eviction-only arm never diverges — its protocol
   is purely extensional; assert-level guarantee + unit test comparing
   plans on eviction-only chains before/after).
3. ON divergence: locate the dead branch — LCP gives the fork point;
   align sub[fork:] against expected[fork+R:] to identify the dropped
   region R (the repair/synthetic exchange = a known prior call's
   contribution rows). In the timeline: kill exactly those rows (death
   at this call = the engine stopped serving those blocks to this
   lineage), then proceed — live set returns to the surviving branch,
   global-live == request-visible again, positions correct without any
   separate position fix.
4. The engine's synthetic_cached_tokens (nuf-derived) cross-checks the
   reconstruction: post-fix, keep == engine's reported hit.
VALIDATION LADDER: (a) 7+36 unit tests + new fork unit test; (b)
offline replay on /tmp/m2_captures/v14 — recall samples to floor,
eviction-only samples byte-identical plans; (c) v15 smoke — in-loop KL
to ~0.001.

### M2 ROOT CAUSE #3 CORRECTED (2026-06-10): warm/replay rows stamped at
### the PRE-admission offset; engine prefills everything POST-admission
The fork theory was WRONG for this seam: with the expected-stream
replay made unconditional, ALL 16 calls are STITCH-OK (sub[:cum_post]
== expected) — the per-call protocol IS extensional; the earlier manual
live-vs-sub diff compared the wrong reconstruction. (The fork-handling
code stays: lenient, no-op on consistent streams, +2 unit tests, 45/46
suite green — harmless robustness.)
THE REAL SEAM: trainer row 4927 at 4943 = 3887 + writer_offset(1056) —
the warm/replay region of the B.2b path is stamped at the
PRE-admission frame — while the engine (single-forward pre-eviction:
admission evicts at schedule time, BEFORE any prefill) wrote the same
rows at offset_after(1440): 3935+1440=5375. The B.2a/B.2b warm model
is a relic of the two-phase era. EVICTION-ONLY IS SAFE because full
prefix-cache hits leave the replay region EMPTY (cached == cum_post);
the recall arm's partial hits (synthetic cached_tokens 3712 < cum_post
3888, the manager/repair turns shorten matches) create a replayed
region written post-admission by the engine but stamped pre-admission
by the trainer.
FIX (next): in the timeline builder, stamp prefix_replay rows and the
B.2b warm region at the POST-admission frame (admission_offset_after)
when the call has admission events — matching in-step pre-eviction.
Verify on /tmp/m2_captures/v14 (seam positions should match HIDDEN-POS
engine values exactly), then the eviction-only plan-equality tests +
v15. NOTE: check the eviction-only validated arm's warm path: with
full hits the warm region is inherited (not re-prefilled), and
inherited K WAS written at the writer frame by previous requests —
positions for warm rows must distinguish INHERITED (writer frame) vs
RE-PREFILLED (post-admission frame); the synthetic cached_tokens
boundary is exactly that split.

## Gate 0 — full-BPTT backward memory probe (DONE 2026-06-10)

Script: experiments/_local_jobs/debug/scripts/
probe_flex_fullbptt_backward_memory.py (GPU 1, expandable_segments,
Qwen3-4B bf16, weight baseline 7.49 GiB), 8 largest 06-06 snapshots
(11.4k–11.8k writer tokens):
- no AC: 8/8 OOM (~75–78 GiB peak mid-forward) — full-BPTT without AC
  does not fit production lengths even at batch-of-one on 80 GB.
- full per-block AC (non-reentrant): 8/8 pass — largest 11,808 tokens:
  peak 20.7 GiB (+13.2 over weights), fwd ~300 ms / bwd ~1.1–1.3 s,
  grads finite. AC mandatory ⇒ D4 amended; smoke TOML has
  [trainer.model.ac] mode="full" freq=1.
Raw: gate0_fullbptt_{noac,ac}_20260610.json alongside the script.
(Memory conclusions are insensitive to mask pattern; interval masks do
not change this verdict.)

## Branch / commit policy

Branch `feat/flex-full-bptt-alignment`. M0 touches
vllm/.../scheduler.py, which currently carries UNCOMMITTED session-arm
work (user keep-vs-shelve decision pending) — resolve tree state before
engine edits to avoid entangling the two efforts. M1 (trainer
segmented_forward.py) is clean to start.

## Status

- 2026-06-10: D1/D8 scope corrected to managed-context core. Engine gap
  analysis complete (restore timing unrecorded; no restore event
  channel; die-once everywhere). Gate 0 done (AC mandatory).
- 2026-06-10 (later): session-arm work committed (vllm 3cc86c67a, main
  4593219); flex work lives on feat/flex-full-bptt-alignment. **M0
  BUILT + committed** (vllm ed544f367, prime-rl ca406df8): restore
  attach/release ride compaction_events as kind-1/2 events with
  visibility_boundary_computed; eviction events carry per-span
  archived_span_bounds; client extraction (env.py) and
  CompactionEventWire mirrors extended (old-payload decode verified);
  trainer death-index path hard-errors on kind!=0 until M1.
- 2026-06-10 (later still): **M0 VALIDATED LIVE** (vllm 217a0e3e3). Two
  traps found on the way: (a) the OpenAI serving layer hand-rolls the
  event→payload conversion in THREE places (CompactionEventPayload +
  chat conversion + session _compaction_event_to_dict) — new event
  fields must be added to all three or they silently drop at the HTTP
  boundary; (b) the COUNTING benchmark never evicts/recalls AT ALL
  (verified: zero archives in the production c64 run — it is a
  request-KV-swap pressure benchmark), so recall validation must use
  the COOKING arm. Proof: experiments/textworld_env/
  m0proof_cooking_launch.sh (4 games/conc4/GPU1, trace on) → 20 kind-1
  attaches + 20 kind-2 releases, 6 archives, 0 errors; upfront attaches
  at computed=0, deferred attaches at computed=writer_len-1 (final
  visible token decodes under restored KV, per the defer contract),
  releases at end-of-known-stream. Next: M1 interval mask in
  segmented_forward → M2 capture with logprobs.

## 2026-06-13 — Recall-floor root cause AIRTIGHT + token-surfacing design

KL-check matrix (committed 7524f94): full-context 0.0021 / kv-eviction
0.0022 / markovian 0.0019 at floor with REAL eviction; **kv-recall 0.1807
+ markovian-recall 0.2881 OFF floor.** Decoding step_0/rollouts.bin proved
the recall floor cause definitively: in the recall arm (e.g. sample 35,
24 calls) span **T0001 is restore-ATTACHed on EVERY call but archived in
NO sample of the 64-sample batch.** T0002/3/4 ARE archived (calls 6/13/20,
bounds [432,~2064]) and reconstruct fine — their rows are born inside the
sample before eviction. T0001 was evicted BEFORE the recall arm's call 0
(first recall handshake, on a side-channel manager-pass call the verifiers
interceptor never attaches) and recalled as HIDDEN KV → its ~1632 tokens
are in NO visible prompt; the paired short arm is an independent re-roll.
**T0001's tokens live only in the engine CPU archive — surfaced to NO
training sample.** trainer span_rows[sid]=live_writer_indices[s:e] has
nothing to map → graceful-skip (KVE_RECALL_SKIP_UNARCHIVED=1 default) lets
it run+measure (0.1807) instead of crash.

### Confirmed-feasible fix: carry archived-span tokens on the restore event
The engine HAS the tokens: `_archive_managed_context_span` stores
`token_ids=request._all_token_ids[evict_start:evict_end]` (scheduler.py
11896) plus `position_offset_frame`, `evict_start`, `evict_end` on the
`ManagedContextSpan`. The restore ATTACH event is emitted per-call at
`_append_managed_context_restore_event` (scheduler.py 11342) and IS
captured on every recall-arm turn — so riding the tokens on it sidesteps
the side-channel-capture + sample-split problem that sank FIX v2. Exact
hook sites (ADDITIVE optional trailing fields, gated KVE_RECALL_SURFACE_
TOKENS default OFF → 3 floor modes untouched; "trailing field is wire-safe"
per types.py:14):
1. vllm `v1/core/compaction/types.py` CompactionEvent (~134): add
   `restored_span_token_ids: list[int]` + `restored_span_positions:
   list[int]` (absolute RoPE positions = evict_start+offset range; needed
   for the seam — root-cause-#3).
2. vllm `v1/core/sched/scheduler.py` `_append_managed_context_restore_event`
   (11342): for kind==1, look up each span in `_managed_context_archive`
   (key=(trace_id,span_id)) and populate the two fields from span.token_ids
   + computed positions. Emit ONLY on first attach per request to bound
   wire bloat (trainer reconstructs once, caches in span_rows).
3. THREE serialization places (the trap): protocol.py CompactionEventPayload
   (~97), chat serving.py (~1634), session serving.py _compaction_event_to_dict
   (~50).
4. prime-rl `transport/types.py` CompactionEventWire (~95): mirror the two
   fields. env.py `_extract_compaction_event_dicts` (151) +
   orchestrator/trajectories.py `_compaction_events_from_step` (607): copy
   them through (verify they're generic, not field-by-field).
5. **trainer `segmented_forward.py` `_process_visibility_ops` (~2599) — the
   geometry-critical step (root-cause-#3):** on a restore (kind 1) whose
   span is not in span_rows but carries restored_span_token_ids, PREPEND
   those tokens to the writer sequence as a span segment, forward them with
   the carried absolute RoPE positions (NOT a fresh arange — this is the
   seam fix), build span_rows[sid] once, then attach. Replace the
   graceful-skip with this reconstruct path (keep skip as the fallback when
   tokens absent).

Validate per step on the batch-4 kv-recall cell: after 1-4, re-decode
rollouts.bin and confirm the recall arm's call-0 CallWire carries
T0001's restored_span_token_ids; after 5, confirm Mismatch KL drops from
0.1807 toward the ~1e-3 floor. markovian-recall (visible re-prefill recall)
is the same seam, different surface.
