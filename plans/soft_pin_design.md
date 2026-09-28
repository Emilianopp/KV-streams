# SOFT-PIN: evictable-but-CPU-backed managed state (K=32 effort)

Date: 2026-06-10. Owner thread: enable managed-recall at concurrency 32 on the
gpu_util-0.20 pool (3,345 blocks) WITHOUT shrinking the context window
(window-tightening explicitly rejected by user) and WITHOUT recompute
(KV-continuity contract).

## Why (measured motivation)
- Per-slot throughput is ~77 tok/s in EVERY mode (managed-drop, managed-recall,
  full-context). Concurrency is the entire performance game.
- Managed-recall call footprint ~250 blocks peak (~190 visible window + ~57
  hidden restore). Pinned-resident design ⇒ all admitted traces' footprints
  must co-fit ⇒ K caps at ~12 (capacity-fit, 2,445 tok/s); pushing past
  deadlocks (hold-and-wait + active-restore no-preemption). recallfinal
  baseline: 12g/12c T=48 = 6.84 s/game. 32/32 attempt = capwedge artifact.
- Full-context runs 32 with BIGGER contexts because its KV is evictable cache
  (ref-0 + hashed), not owned property. Same pool, opposite elasticity.

## The design in one line
Make between-call managed state look exactly like full-context's prefix cache
(ref-0, hashed, instantly reclaimable, re-hittable), with an async CPU mirror
making reclaim always safe: evicted → memcpy back (~ms), never recompute.

## The deep finding (3-agent scope, 2026-06-10): ~90% already exists
- `free_blocks` retains hash + logical_start on free (block_pool.py:561-582);
  reclaim evicts the hash lazily at `get_new_blocks` (461-502); `touch()`
  resurrects ref-0 blocks from the free queue (544-559). Stock semantics.
- The frame-aware prefix walk re-hits OFFSET-SHIFTED blocks: the May-16 "skip"
  guard was rebuilt into multi-candidate frame scoring returning
  `inherited_offset` (single_type_kv_cache_manager.py:577-823;
  get_computed_blocks seeds request.position_offset, kv_cache_manager.py:231-242).
- D2H-then-free-at-store-done already implemented: `_start_phase4_pin_cpu_offload`
  (scheduler.py:6657) → `_complete_phase4_pin_cpu_store` (6726) frees GPU
  entries ONLY after the copy confirms. Unlock-at-store-done by construction.
- Miss recovery already implemented: pin-attach declines unless status
  gpu_pinned (11729) → hash walk adopts residents free → on reclaim,
  `_phase4_try_load_pin_for_prefix_miss` (6946) H2Ds the pin's CPU copy back
  (PHASE4-PREFIX-DEFER park-and-retry, never abort, never recompute).
- Queue-entry prefetch exists: `KVE_PHASE4_PIN_PREFETCH` (6973-7057).
- Admission K-cap self-disables when envs unset (7761-7770) — no bypass code.

## Phase 1 (IMPLEMENTED 2026-06-10, ~40 LOC): policy inversion
`KVE_SOFT_PIN=1` (scheduler.py, helper `_kve_soft_pin_enabled` before
`_pin_phase4_request_blocks`): at every pin publish, immediately call
`_start_phase4_pin_cpu_offload(reason="soft-pin-publish")`. Store-done drops
the GPU refs with hashes retained. Decline (CPU pool full / transfer limit /
archive off) ⇒ pin stays gpu_pinned = today's behavior (graceful fallback).
Config inversion in the launcher (counting_softpin_launch.sh):
- KVE_SOFT_PIN=1, KVE_PHASE4_PIN_PREFETCH=1
- NO K cap / NO auto-budget / NO active-trace watermark (self-disabled)
- KVE_REQUEST_KV_SWAP=1 STAYS as the running-victim pressure valve (vanilla
  preemption of offset>0 requests aborts — scheduler.py:3389-3394; the native
  allocator only reclaims ref-0 blocks, so RUNNING over-commit needs the valve)

### Per-call lifecycle under Phase 1
1. call N finishes → cache_blocks (final block hashed, 13531) → pin publish
   (touch, 7505) → soft-pin offload submit (D2H async, low-pri stream)
2. store-done → free GPU refs; blocks now ref-0 + hashed + resident
3. call N+1 arrives → pin-attach declines (cpu_offloaded) → prefix walk
   re-hits whatever is still resident (FREE, inherited_offset handled);
   reclaimed prefix → pin H2D reload (whole-pin in Phase 1) via prefix-miss
   recovery; prefetch starts it at queue-entry
4. pressure: native allocator reclaims ref-0 soft blocks instantly (no
   victim selection, no copy, infallible); running working sets only are refs

## Verifier verdicts on the 3 scope reports
- Agent 3's "gate _request_kv_swap_enabled() off wholesale" REJECTED in favor
  of agent 2's showstopper: keep the swap valve for RUNNING victims; only the
  between-call economy goes (via env unset, not code).
- Agent 3's new mirror-registry dataclass DEFERRED to Phase 2: Phase4Pin's
  cpu_block_ids_by_group + logical_start_by_group IS the registry;
  reclaimed-vs-never-computed is disambiguated by pin coverage +
  expected_cached_tokens (the existing prefix-miss ladder).
- Agent 2's hybrid recommendation ADOPTED: hash-walk fast path + pin-as-
  registry whole-pin H2D fallback (Phase 1); partial missing-range-only
  reload is Phase 2 (the known crash zone: the sessions reload-cache-hit
  pool corruption, scheduler.py:9088-9095 — stay out until gated).

## Known risks (not stoppers; measure at gates)
- R4 FIFO-free-queue order: a trace's TAIL frees last but the queue is FIFO →
  under heavy over-commit the allocator reclaims oldest-freed first, which is
  fine; but kv_cache_manager.free frees a request's blocks tail-first →
  hot traces may lose tails. If reload-per-call shows up at scale: flip free
  order for soft-pin frees (one line) or trace-aware free ordering.
- Whole-window D2H per call until Phase 2 (no delta-append yet): ~190 blocks
  × 2.25 MiB ≈ 430 MB/call D2H. At 32c/5s-calls ≈ 2.7 GB/s — fine on the link,
  wasteful; Phase 2 trims to ~20-40 blocks/call.
- CPU mirror RAM: pins offload into the existing 216 GiB pinned pool
  (98,304 blocks); 64 traces × ~190 blocks ≈ 27 GiB. Capacity eviction only
  victimizes archive spans, never pins — no live-trace mirror loss.
- PHASE4-PREFIX-ABORT storms if recovery error strings stop matching the
  defer patterns (6882-6889) — watch at smoke.

## Phase 2 (NOT built): bandwidth + granularity
1. Append-only delta mirror (extend pin CPU lists; D2H only new blocks/call).
2. Partial reload: H2D only the reclaimed ranges (generalize the reload-
   cache-hit pattern with commit-or-degrade discipline).
3. Revocable hidden restores under pressure (release-then-swap; bytes in
   archive). Phase 1 leaves restores untouched (per-call restores release at
   finish anyway).
4. Free-order/LRU polish per R4; retire KVE_SWAP_RELOAD_CACHE_HIT (superseded).

## Validation gates
- SMOKE (2g/2c/T=16, diag on): [SOFT-PIN] offload submits at publish,
  [PHASE4-PIN-STORE-DONE] frees, successor adopts via prefix walk (PHASE4-
  PREFIX hit / PIN-HIT absent), zero aborts/500s. → results/
  counting_softpin_smoke_20260610
- PARITY (1g/1c, TEMPERATURE=0, soft on vs off): byte-identical streams +
  identical compaction_events. KV-continuity audit: no COMPACT-REPREFILL,
  forwarded ≈ stream.
- ACCEPT (32g/32c/T=48 recall, quiet): completes, no deadlock, beats 6.84
  s/game (target ≤ ~3.5), per-slot ~77 tok/s, aggregate ≥ ~2,400 tok/s.
- TEXTWORLD no-regression: 64/64/0.20 upfront-KV within noise of 69.0s.

## Status
- 2026-06-10: Phase 1 implemented (2 edits, scheduler.py), compile OK; gate
  launcher experiments/textworld_env/counting_softpin_launch.sh created;
  smoke launched. NOTHING COMMITTED (user rule: validate first).
- SMOKE (2g/2c/T=16, prefetch ON): exit 0, zero aborts; lifecycle correct
  (publish → offload-submit → store-done frees w/ hash retained; misses
  DEFER → whole-pin reload, never abort). BUT 29/32 calls deferred — the
  unconditional pin prefetch (whole-pin H2D fires every call since pins are
  always cpu_offloaded now) races admission: pin load_pending → attach
  declines → walk under-finds (240/1376 at call 6) → defer. CPU ping-pong.
  → results/counting_softpin_smoke_20260610
- WALKDIAG (1g/1c/T=8, prefetch OFF, KVE_TRACE_CACHE_HIT=1): **THE WALK IS
  PERFECT.** Every call's chain hit the ENTIRE published pin resident+free
  (HASH-MISS exactly at the pin boundary each call: idx 26/38/50/62/74/86),
  across the turn-6 eviction too. ZERO defers, ZERO pin loads, zero CPU
  round-trips — the CPU mirror was pure insurance. Soft-pin's free-re-hit
  economics CONFIRMED at 1g. → results/counting_softpin_walkdiag_20260610
- DECISION: KVE_PHASE4_PIN_PREFETCH default 0 in the soft-pin launcher
  (prefetch's blind whole-pin H2D is anti-productive when blocks are
  resident; reclaim-misses self-serve via defer→reload). Root-causing the
  exact prefetch→walk interference mechanism deferred (suspect: attach
  declined on load_pending + walk colliding with in-flight H2D copies; NOT
  hash filters — those default off; NOT load-path hash eviction — none).
- smoke2 (2g, prefetch OFF): defers PERSIST (same 240/1376 @call 6) → NOT
  prefetch. walkdiag2 (2g, trace on) nailed it: both traces' walks hit the
  SAME physical block (block_id=16 @idx 15, n_candidates=1) — **identical
  counting streams dedup onto shared blocks; each trace's eviction rehash
  then pops the single hash registration (one block = one hash) out from
  under siblings** → chain dies at sys boundary → ping-pong. ROOT CAUSE =
  benchmark degeneracy (counting_env start=1 for ALL examples), not an
  engine bug; real workloads diverge at turn 1.
- FIX (harness): counting_env.py per-example starts
  (COUNTING_DISTINCT_START_STRIDE, default 1009; 0 = legacy identical).
  CAVEAT for old numbers: every prior counting baseline ran identical
  streams — fullctx was FLATTERED by cross-game dedup (shared prefix cache
  across games), managed/pin-attach was not. Distinct streams = honest
  benchmark; re-base fullctx/recallfinal later if a clean triangle is wanted.
- Engine hardening option (Phase 2, only if real workloads ever show it):
  scope post-sys cross-trace adoption by writer-trace under soft-pin, or
  preserve sibling discoverability across rehash. NOT needed now.
- smoke3distinct (2g/T=16, distinct starts): call-6 catastrophic break GONE;
  elapsed 15.59s, 0 errors. Remaining: deterministic 8-block shortfall at
  call 12 (cached=1152 expected=1280, offset=2096) — reproduces at 1 GAME
  (walkdiag3) → structural, not cross-trace.
- BUG #2 ROOT CAUSE (walkdiag3 TRACE-HIT): **smart-bump frame seam.** The
  trace's blocks are ALL resident+hashed but two-frame: idx 15-71 at cb=2096,
  idx 72-79 at cb=2112 (call-11 admission bumped offset +16; survivors keep
  old-frame logical_starts BY DESIGN — physical K rotation is history). The
  walk's frame-uniformity rule truncates at the seam (CHAIN-TRUNCATE ...
  available_cbs=[2112]) — the EXACT showstopper #3 the scoping predicted:
  hash-walk-only adoption cannot serve multi-frame chains; hard pin-attach
  never cared (attaches all, seeds from first frame). Bonus finding: defers
  also fired for store_pending (call arrived mid-mirror-store; attach
  declined although blocks are ref'd+intact).
- FIX (the hybrid's registry leg, ~100 LOC): **validated soft attach** —
  Phase4Pin.soft_entries snapshot at publish (blocks + hash + logical_start
  per block); `_phase4_soft_pinned_cache_blocks` re-attaches trace-keyed for
  status in (store_pending, cpu_offloaded) iff EVERY expected block still
  carries its recorded hash+logical_start (get_new_blocks resets both at
  reclaim → stale snapshots decline to the walk + pin CPU reload).
  All-or-nothing; C4 self-cannibalization (restore-load steals validated
  ref-0 blocks pre-touch) degrades to decline-on-retry, correctness intact.
- walkdiag4 (1g/T=16, soft attach): **CLEAN SWEEP — defers 0 (was 18),
  pin loads 0 (was 6), declines 0, aborts 0, errors 0; 17 soft attaches
  served every successor (mostly mid-store status=store_pending; post-bump
  frame-seam calls attach fine).** elapsed 14.27s. The CPU mirror was pure
  insurance. → results/counting_softpin_walkdiag4_20260610
- smoke4 (2g/T=16, distinct+soft-attach confirm): CLEAN (0 defers/loads/
  declines/aborts, 34 soft attaches, 0 errors, 15.41s).
- ACCEPT attempt 1 (32/32 T=48, no brake): **WEDGED at ~90s** — the THIRD
  deadlock leg, as predicted for Phase 2: running=31 each holding visible
  window + HIDDEN restore (~147 blocks avg) ≈ 4,600 > 3,345 pool;
  restore-holders unswappable (active-restore barrier); admission unthrottled
  (caps unset) → hold-and-wait. total_sched_tokens=0, GPU 0%. Killed clean.
  → results/counting_softpin_accept32_20260610 (wedge artifact #2)
- KEY MEASUREMENT (sessions logs): upfront-config restores are FULLY HIDDEN
  (freed_hidden=38 kept_visible=0) — NOT in req_to_blocks → whole-rollout
  swap does NOT carry them → revocable restores REQUIRES re-attach-on-reload
  (full Phase-2 lever, ~2h build: revoke record + reserve spans at swap-out,
  re-activate + pending-load gate at swap-load completion).
- PHYSICS (honesty note): at 0.20 pool (3,345 blocks = 53.5k token-slots),
  32 SIMULTANEOUSLY-RESIDENT recall calls (~2.4-3.9k tok each ≈ 77-125k) are
  impossible for ANY machinery — fullctx "ran 32" by preempt+RECOMPUTE
  time-sharing (pool peaked 38%). Managed must time-share via defer/reload.
  Gate criterion = 32 in flight, no wedge, throughput.
- QUICK LEVER (accept v2): the existing restore-admission brake — defers a
  request BEFORE it attaches/allocates anything (waiting-loop check at
  scheduler.py:1729 precedes allocation; deferred requests hold ZERO).
  KVE_MANAGED_CONTEXT_RESTORE_ADMISSION=1 +
  KVE_MANAGED_CONTEXT_ACTIVE_RESTORE_TARGET_USAGE=0.30 (≈17 concurrent
  restore-holders + decode-growth headroom; yesterday's wedge showed 0.75
  attach-time evaluation is growth-blind → go conservative).
- accept32v2 (brake 0.30): WEDGED — brake bounds restores only; visible
  working sets unbounded. v3 (protected-set narrowing + auto-K): WEDGED —
  auto-K cold-start K=77 (mean pin 26 blocks at first calls). v4 (explicit
  K=18): WEDGED — running=25, cap never gated (same-step admission). THE
  LESSON: admission gating is structurally broken on this workload; relief
  must come from preemption, not prevention.
- REVOCABLE RESTORES built (KVE_SOFT_PIN_REVOCABLE_RESTORES): swap can take
  restore-holders (revoke after last refusal; reserve spans; kind-2 event;
  re-attach GATE at swap reload before schedulable — resident touch or
  pending-load; drop-unavailable fallback). v5 (no caps): NO deadlock but
  THRASH-COLLAPSE — 1,823 pressure-time stores, decode batch 2-4. Relief
  PRIMITIVE wrong: copy-at-pressure.
- STREAMING MIRROR built (KVE_SOFT_PIN_STREAM_MIRROR): KVEStreamMirror per
  running request — filled blocks D2H in background (64 blk/step budget,
  FIFO confirm, orphan-safe); _start_request_kv_swap_out splits at the
  confirmed boundary: prefix FREES INSTANTLY, store event = unconfirmed
  tail only; fully-mirrored victims born "swapped" (no store). v6: running
  14-15 (vs 2-4), balanced rotation, killed at 73% — slow late phase from
  rotation knobs (MIN_PROGRESS=32 → 4 mid-call parks; STOP=0.55 → 1.2k
  blocks over-evicted/burst) + design-B span demotion per swap (601×).
- v7 = knob fixes (MIN_PROGRESS=160, band 0.92/0.85, hot budget 2048) +
  design-B demotion gated off under stream mirror. **RESULT: 32/32 COMPLETES
  — 686.3s, 0 errors, finished 1.0, reward 0.979 (= recallfinal parity →
  re-attach correctness PROVEN over 1,536 calls), 1,061 swaps.** But 3.1×
  slower than the K=12 baseline (6.84 → 21.4 s/game): (a) HOT-ARCHIVE
  CROWD-OUT — 2048-block hot spans hold HARD refs; troughs showed
  running=6, skipped=26, pool 96% used, free=119 — the last unreclaimable
  residency class starves wave re-admission; (b) CONVOY — identical forced
  128-tok calls phase-lock all 32 traces (counting artifact; running
  histogram bimodal 32↔1-9).
- NEXT (user-approved direction, "smart concurrency allocator"):
  L1 archive-soft — span hot copies become ref-0+hashed+CPU-backed
  (delete HOT_BLOCK_BUDGET; native reclaim; recall touches survivors);
  L2 K* governor — EWMA mature-call footprint → admit/backfill target,
  AIMD on rotation-rate, valve as backstop only. DEADLOCK GOAL: MET
  (32 in flight, no caps, completes). THROUGHPUT GOAL: open.
- v8 = L1 SAFE VARIANT via config: ARCHIVE_DEVICE=cpu (immediate) +
  HOT_BLOCK_BUDGET=0 — spans D2H at eviction, GPU entries free at
  store-done, ZERO archive residency (the validated-hot-cache variant
  needs a trust anchor span blocks lack — no hash registry — deferred).
  Mid-run: 7.05 calls/s (~900 tok/s) = K=12-baseline RATE at 32 in
  flight, 24% over v7; crowd-out gone.
- v8 TIME-ANATOMY (trace ts reconstruction): **convoy REFUTED at client
  layer — 32 in flight for 99% of span (347/350s)**; engine queues
  internally: 61% of wall at running≤9, p50 call latency 11.6s vs ~4.5s
  pure decode. Pool free=119-157 with running=11 → ~1.5k blocks ref'd by
  NEITHER running nor archive → **WAVE-END PIN-STORE BURST: 32 finishing
  calls each re-D2H their FULL window (~130 blk ≈ 9GB/turn) although the
  stream mirror already holds confirmed copies; refs held until
  store-done starve next wave's admission.** (Transfer-limit NOT binding:
  CPU_MAX_PENDING_STORE_EVENTS default 0=unlimited; pure volume+ref-hold.)
- FIX BUILT (awaiting v9): **DELTA PUBLISH** — _start_phase4_pin_cpu_offload
  gains stream_mirror_request_id: adopts the finishing request's CONFIRMED
  mirror ids, allocates+stores ONLY the tail; tail==0 → 
  [PHASE4-PIN-OFFLOAD-INSTANT] (release GPU refs synchronously, no store).
  Publish store volume ~130 → ~last-turn blocks; ownership transferred
  (mirror popped; in-flight ids via orphan branch).
- Remaining ranked: restore-H2D prefetch at queue-entry; capacity-gate
  effective-free audit; L2 K* governor; convoy de-phasing (counting-only).
- SOLO TRIANGLE (decisive, 2026-06-11): managed-recall solo T=48 = 45.3s
  (0.94 s/call) BEATS fullctx solo 60.9s (1.27 s/call) — machinery wins
  per-trace; 32c leader pace was 16 s/call → ALL loss is contention policy.
- v9 LESSON: per-request stream mirror re-mirrors the inherited window
  every call (new request_id each call) → delta publish alone moved the
  copy, didn't remove it. FIX = PIN-CHAIN DELTA: publish diffs new pin
  blocks vs old pin soft_entries by id(), inherits old CPU ids
  (ownership: truncate old pin's list pre-release; offload owns on
  success, caller frees on decline). VERIFIED v13: inherited=26
  store_tail=12, declines=0.
- v10-v11: SRPT via native vLLM priority policy (client extra_body
  priority=-call_idx; scheduling_policy="priority" toml-injected; valve
  victim sort + queues already priority-aware). Staircase forms (v11
  spread turn 21↔35).
- SCHED-CENSUS instrument added (KVE_SCHED_DECISION_CENSUS=1, 5s):
  classifies every queued request's exact defer reason + deepest waiters.
  Named serially: (1) swap ready-queue HEAD-OF-LINE (one reload at a
  time) → bypassed under soft-pin (concurrent reloads, bounded by
  admissibility+MAX_PENDING_LOADS) ✓ vanished from census; (2) RECLAIM
  STORM: at free≈74 every next call prefix-misses → whole-pin HARD
  reload (~130 blk, 16 pins = 2,065 blk crowd-out) → more reclaim
  (v12/v13, pin_loads=140, declines=0); (3) v14 = K*-governor watermarks
  (RESTORE_ADMISSION 0.35 + PIN_LOAD 0.80 + RELOAD 0.85): pin_loads=0 ✓,
  staircase STEEP (leaders turn 32 vs avg 16) ✓, but RELOAD=0.85
  strangled refill (running 9-10, leaders parked) + inverted vs valve
  0.92. v15 = ordered ladder: storm gates (0.35/0.80) < reload 0.90 <
  valve 0.94/0.90.
- SWAP-AGE under mirror: room1_store=0.0s (instant swap-out CONFIRMED);
  the whole round-trip cost is room2 pick-wait (was 68s under storms).
- v15 (ordered ladder 0.35/0.80 < reload 0.90 < valve 0.94/0.90):
  REGRESSED to running 4-6. Full accounting at the trough: used 3,134/3,345,
  free=211, running=5 — holders: reload-pins 979 blk (loaded for traces
  whose calls are STILL parked), restores 481 blk (~196 of them attached
  to NON-running requests), running ~700. **FINAL DIAGNOSIS: the resume
  pipeline (pin-load → restore-attach → schedulable) is itself a
  HOLD-AND-WAIT CHAIN — each stage holds GPU blocks while waiting for the
  next stage's watermark, and the watermarks measure the pool those holds
  fill. ~9 slots of blocks parked in half-resumed requests; equilibrium
  locks at running≈5 (one finish releases ~200 blk, one staged resume
  absorbs them, net zero).** Fullctx never has this: its resume is ATOMIC
  (recompute inside the forward pass — no staged holds).
- v16 = ATOMIC RESUME ADMISSION BUILT+RUN (KVE_SOFT_PIN_ATOMIC_RESUME=1,
  ~120 LOC: bundle gate at pin-miss + swap-resume entry; per-stage
  watermarks stand down; grants start window+restore loads same-step;
  SRPT-ordered). **RESULT: 560.4s (best of night; v7 686, v8 692), 0
  errors, reward 0.979, hold-and-wait reason collapsed 21→3-6, bundle
  rarely binds (5 hits), AND THE STAIRCASE IS VISIBLE: 7/32 finished
  early mid-run — first non-end-burst completion all night.** Residual:
  running ~9-10 (not 14-18); census dominant = plain 'queued:WAITING'
  (17-20) — admission-ladder churn the state-probe census can't name
  (suspect: prefix-recovery defer loop with pin reloaded-but-unconsumed).
  NEXT SESSION: census v2 — record the actual defer STRING on the request
  at each prefix-defer site, rerun, fix the named loop. Then TextWorld
  G4 no-regression + THE COMMIT.
- census2 (defer-string stamping) = 548.4s; v17 (headroom 32→16) = 561.9s
  → **STABLE EQUILIBRIUM 557±7s**: queue = one reason (pin reload waiting
  for atomic bundle = honest room wait), grants finisher-paced, bundle ask
  already honest. ARCHITECTURE COMPLETE for this pool; remaining levers:
  pool size (seats = pool ÷ ~170), recall-at-compaction instead of
  per-turn forcing (~7× less recall traffic — the production textworld
  shape), partial reload (Phase 2). NOTE: this benchmark prices the
  WORST-CASE recall tax by design (forced random recall EVERY turn, 6/4).
- CONC16 HONEST-STREAMS HEAD-TO-HEAD (2026-06-11 ~04:00, 32 games T=48):
  **fullctx = 282.3s** (its old 102.8s/3.21-per-game = identical-streams
  cache flattery, now retired); managed forced-per-turn recall = 473.1s /
  467.3s (×2 reproduced; fully seated running=16, zero rotation — the
  cost is the per-turn recall FLOW: ~1,344 extra manager calls + retries,
  not the 57-blk H2D). **At-compaction run was INVALID: the
  --managed-context-force-restore client flag bypasses the
  RECALL_AT_COMPACTION cadence gate (mgr_requests=1344 unchanged).**
  NEXT-SESSION FIRST ITEMS: (1) client fix in env.py force branch — fire
  manager/force flow ONLY when the archive fingerprint changed (i.e., at
  compaction; the _forced_recall_pick fingerprint machinery already
  detects this), plain persistent re-attach otherwise (~20 LOC); rerun
  conc16 → predicted ~190-230s, beating fullctx 282 with full recall
  semantics; (2) THE COMMIT (everything validated, still uncommitted);
  (3) TextWorld 69.0s no-regression gate. Also: solo managed 0.94 s/call
  vs solo fullctx 1.27 stands as the per-trace win; capacity-fit seating
  beats over-commit at this pool (16 seated > 32 churning).
- CONC16 CADENCE-FIX ATTEMPTS (2026-06-11 04:00-05:00): all five managed
  conc16 runs cluster 467-497s (forced=1312/mgr=1344 in EVERY one) vs
  fullctx 282.3s. Attach mechanism ELIMINATED as the cause (spans-resident
  hot-budget run = 497s ≈ H2D runs). THE GAP = the per-call manager/forced
  flow (~1,300 extra serialized client LLM calls). TWO fingerprint fixes
  failed: (a) len(index) grows ~every call (per-turn spans), (b) len//stride
  still mismatches (recalled turn re-archives → rows/fire ≠ stride).
  **NEXT-SESSION FIX (exact, not heuristic): per-trace COMPACTION COUNTER —
  increment where the client consumes a CompactionEvent (eviction kind-0)
  in _update_phase4_state_from_response; fingerprint = that counter.**
  Then rerun conc16 combo (at-compaction + spans-resident): expected
  forced≈350 → wall ~250-300s, at/below fullctx with full recall
  semantics. Env edits so far live in src/kv_eviction/env.py (reuse-branch
  publishes managed_context_current_recall_ids + skips forced flow; the
  //stride fingerprint to be replaced by the event counter).
- **HEAD-TO-HEAD WIN (2026-06-11 ~05:30): combo2 = 276.4s BEATS fullctx
  282.3s** (honest streams, conc16, 32g, T=48; forced=448 ≈ 1/compaction,
  mgr=32 = 1/game, reward 0.979, 0 errors). Root cause of all prior
  identical ~470s runs: counting_softpin_launch.sh HARDCODED
  KVE_MANAGED_CONTEXT_RECALL_AT_COMPACTION=0, silently overriding every
  command-line =1 (now parameterized). Winning config = launcher defaults
  + RECALL_AT_COMPACTION=1 + cpu_deferred/explicit + HOT_BUDGET=1200 +
  ATOMIC_RESUME + SRPT + conc16 (capacity-fit). The env.py reuse-branch
  fix (skip forced flow, publish managed_context_current_recall_ids) +
  //stride fingerprint are part of the win. STILL UNCOMMITTED. — a
  single all-or-nothing admissibility check per returning call covering
  window-reload + restore-attach + decode headroom TOGETHER; start loads
  only when the whole bundle fits; a request holds ZERO GPU blocks until
  its bundle is granted. Implementation sketch: (1) compute bundle size =
  pin/swap reload blocks + restore span blocks + ~2 turns headroom;
  (2) one gate at the front of the resume/admission path (replaces the
  per-stage watermarks for soft-pin mode); (3) grant = start pin/swap
  load + restore load in the same step; stage-completion gates unchanged;
  (4) deny = request stays parked HOLDING NOTHING (no pin load, no
  restore, no partial). Priority-ordered grants (SRPT queue) keep the
  budget flowing to leaders. Est. 1-2h on scheduler.py; everything else
  (instant swap-out, delta publish, soft attach, revocable restores,
  concurrent reloads, SRPT) is in place and verified.

## NEXT-SESSION ENGINE TASK: EVICT-BEFORE-ALLOCATE
On compaction calls, admission allocates the FULL pre-trim prompt
(~190-blk spike) then _compact_request trims to ~75-90. Reorder: the
admission-eviction decision (token math, no allocation needed) runs
BEFORE allocate_slots, allocating only survivors+new fragment.
CAUTION: chained-admission RoPE-fix region (survivor frees beyond
evict_start, smart bump, _rehash_after_eviction ordering) — re-read
those fixes first. Validate: 1g smoke + conc16 combo2 parity
(276.4s / reward 0.979). Win: pool sized by steady state → +1-2 seats.
CONC SWEEP in flight: conc20 (combo2 config) →
results/counting_softpin_conc20combo_20260610 (bar: 276.4 @ conc16).

## CONC SWEEP RESULT (2026-06-11 ~06:00)
conc16=276.4 / conc20=269.4 (KNEE, best) / conc24=275.7 vs fullctx 282.3.
**FINAL HEADLINE: managed-recall 269.4s beats full-context 282.3s by 4.6%**
(honest streams, 32g T=48, reward 0.979 everywhere). Over-commit past
capacity-fit works and the optimum sits just above it — the graceful
time-sharing stack earning its keep. REMAINING QUEUE: repeat-runs to lock
the headline, evict-before-allocate (spec above), THE COMMIT, TextWorld
69.0s G4 gate.

## FINAL SWEEP RESULT: CONC32 WINS (2026-06-11 ~06:30)
conc32+combo = 242.9s — BEST. Sweep: 16=276.4 / 20=269.4 / 24=275.7 /
32=242.9 (single-wave effect; cohort quantization explains 24's dip).
**HEADLINE: managed-recall @ K=32 = 242.9s vs full-context 282.3s = 14%
WIN, on the user's original target concurrency, deadlock-free, reward
0.979, zero recompute.** (Same conc32 was 557s at session start and a
DEADLOCK before soft-pin.) Queue: repeats to lock, evict-before-allocate,
THE COMMIT, TextWorld G4.

## CLOSING NUMBER (2026-06-11 ~06:45): FULLCTX CONC32 = 275.9s
Final identical-settings head-to-head (32g, T=48, honest streams, conc32,
0.20 pool): **managed-recall 242.9s vs full-context 275.9s = 12% WIN**,
zero recompute, reward 0.979. Session complete: capwedge deadlock ->
12%-faster-than-fullctx at the same K=32. Queue unchanged: repeats,
evict-before-allocate, THE COMMIT, TextWorld G4.

## COMPLETE MECHANISM TABLE (2026-06-11, counting honest streams, 32g T=48)
conc32: Markov(client-trim+anchor, RECOMPUTE, reference-only) = 84.6s |
managed-recall KV @compaction = 242.9s | fullctx = 275.9s
conc16: KV @compaction = 276.4s | fullctx = 282.3s | visible-per-turn =
455.7s (1,344 insertions, 2.23M re-prefill tok) | KV-per-turn ≈ 470s
Notes: visible arm pastes span TEXT into the prompt (recompute; engine
eviction unchanged); ran per-turn cadence (bypasses the at-compaction
reuse branch — cadence support for visible mode = follow-up if wanted).
Markov arm = KV_EVICTION_MARKOVIAN_* envs on the fullctx launcher
(client-side trim, anchor_turns=1 as recall analog; 1,312 truncations).
KV-continuity contract premium on this benchmark: 242.9/84.6 ≈ 2.9×
(buys training-legal KV; "Markov = reference only, never a mechanism").
Paste-at-conc32 through the managed engine WEDGES (fresh oversized
prefills; min-progress shields all victims; bundle gate covers resumes
only) — bundle-gate-for-fresh-admissions = queued engine task.

## FINAL c32 LADDER (2026-06-11, all cells, reward 0.979 / 0 errors)
RECOMPUTE-RELIEF (reference): kv-eviction ORIGINAL (KVE_CLIENT_PURE_DROP=1
client toggle added — vanilla requests, no pins/xargs; RECALL_MAX_SPANS=0;
all soft machinery off; miss = re-prefill) = 77.5s; markov-reprefill
anchor1 = 84.6s / no-anchor = 86.4s (anchor free; honest).
NO-RECOMPUTE: managed-recall-kv 242.9s | kv-eviction+CPU-backup 262.2s |
full-context 275.9s. CONTRACT PREMIUM = 77.5→242.9 ≈ 3.1×, of which ~2.4×
is reclaimable seats engineering (10→24 seats: evict-before-allocate,
fresh-admission bundles, partial reload) — irreducible movement premium
≈ 1.3×. Graphs: /tmp/recall_c32_complete.png.

## CORRECTION (2026-06-11): PURE-DROP CELL WITHDRAWN
77.5s confounded: KVE_CLIENT_PURE_DROP trace-id blanking also broke the
client kept-window prompt rebuild (trace max num_messages=96 = full
conversations sent; hybrid behavior). Verified recompute reference =
markov-reprefill 84.6/86.4s. PROPER pure-drop toggle (next session):
suppress ONLY the xargs emission (kve_phase4_* keys) while keeping
phase4_state prompt reconstruction intact; re-verify with max-depth and
prompt-total checks before trusting the wall.

## CORRECTION-OF-CORRECTION: PURE-DROP REINSTATED (verified)
num_messages in trace rows = PRE-rebuild count (both markov and phase4
paths) — wrong metric. Per-call SUBMITTED prompt tokens: markov p50=905
MAX=1,063; pure-drop p50=1,056 MAX=1,440 (= the 6-turn window peak).
Prompts were capped in every call. **kv-eviction ORIGINAL = 77.5s STANDS
as the verified floor.** Final c32 ladder as sealed above is correct.
Verification rule for future cells: judge by per-call usage
prompt_tokens distribution, never num_messages.

## LAST CELL: managed-kv-reprefill @ c32 = 430.9s (per-turn flow,
visible_reqs=1344, max prompt 4,224 tok, MIN_PROGRESS=0 belt — survived
the over-commit that wedged it before). Confirms mechanism-vs-flow:
paste 430.9 < hidden-KV-per-turn 470 at the same cadence; both dominated
by the per-turn recall protocol. SEVEN-CELL c32 LADDER COMPLETE
(graph /tmp/recall_c32_complete.png): 77.5 / 84.6 / 86.4 ‖ 242.9 /
262.2 / 275.9 / 430.9. Benchmark program closed.

## COMMITTED 2026-06-11
vllm @ 52f91dfb1 (compaction-prefix-cache): the full soft-pin stack
(soft attach, pin-chain delta, stream mirror, revocable restores, atomic
resume, concurrent reloads, census). Main repo @ a7b6e59
(feat/flex-full-bptt-alignment): SRPT client, cadence fix, distinct
streams, launchers, this plan, headline figure. Local commits only — not
pushed.

## TEXTWORLD VALIDATION (2026-06-11 ~17:00)
G4 no-regression PASS: flags-off = 69.35s (record band 69.0-69.6), reward
0.122, 64/64, 0 errors — committed engine faithful. Full soft stack ON =
126.0s (correct, 1.8x slow: per-call publish/attach overhead dominates
~10-token calls; knobs tuned for hard pins). BISECTION: KVE_SOFT_PIN
ALONE = 3,736s + 1 error — CATASTROPHIC; the companion flags (atomic
resume esp.) are REQUIRED rescuers, not optional add-ons. RULES: (1)
production textworld keeps flags OFF (record path intact); (2) soft-pin
must imply atomic-resume (add a guard/warning); (3) short-call workloads
need LAZY PUBLISH (offload at idle/pressure, not per call-end) before the
stack transfers — next tuning item. Dirs: ..._g4_20260611 /
..._softpin_20260611 / ..._sp1_20260611.

## TEXTWORLD BISECTION RESOLVED (2026-06-11)
flags-off 69.4s (r.122) | SOFT_PIN+ATOMIC 74.4s (r.068 — repeat needed,
single run) | all-four 126.0s (r.100) | SOFT_PIN alone 3,736s+1err.
VERDICT: atomic-resume is the REQUIRED companion (entire rescue);
mirror+revocable cost ~52s on short-call workloads (over-commit tools,
unneeded at textworld pressure). Soft-pin+atomic is within 7% of record
→ lazy-publish likely closes it. RULE: enforce soft-pin⇒atomic in code.
Dirs: ..._sp1_20260611 / ..._sp_atomic_20260611.

## THINKING-WORKLOAD ANALYSIS (2026-06-11 ~18:30)
Qwen3-4B native thinking on textworld (10/7, 16k): BOTH arms invalid for
walls — turn-counted windows overflow 16k at thinking lengths (base MAX
12,928 = survived by luck; soft drew 16,672 → 400 → dead straggler).
FIXES: client PRE-FLIGHT context guard SHIPPED (env.py: refuse doomed
prompts loudly); token-budget eviction trigger = top engine item; 4/2
schedule as config stopgap. HEALTHY-PHASE tok/s: flags-off 670 vs
flags-on 246 decode tok/s (2.7x deficit; p50 latency 3.3 vs 6.7s).
REFINED LAW: the soft toll is per-turn — what matters is the turn-length
FLOOR (thinking p50=40 tok), not the mean (149). Counting won because
every turn ≥128. NAMED FIX: LAZY PUBLISH (toll on pressure/idle only).

## FINAL THINKING-PAIR VERDICT (2026-06-11 ~21:00, T=32, 4/2, Qwen3-4B)
baseline flags-off: healthy-phase 528 decode tok/s (63/64 salvage; one
straggler lost to a hung HTTP call + the legacy CLIENT_TIMEOUT=7200 —
now 600). SOFT full stack: 4,827s, 33 timeout errors, reward 0.016 —
catastrophic on the bimodal turn mix (median 12 tok). FINAL OPERATING
GUIDANCE: soft stack ON only for uniformly-long-turn workloads
(counting-class, where it BEATS fullctx at K=32); OFF for short/bimodal
turn workloads (textworld-class) until LAZY PUBLISH ships. Ops lessons
banked: verified-clean relaunch (a dying server can spoof the ready
check); per-call usage prompt_tokens as the verification metric;
pre-flight context guard shipped.

## ROOT-CAUSED: THE SILENT-HANG BUG (2026-06-11 ~22:00)
A request aborted during schedule() (e.g. preempt-of-compacted ->
FINISHED_ERROR, scheduler.py:3439-3444) queues its farewell into
_pending_engine_core_outputs (6029) — but the ONLY drain (13387) is
inside update_from_output, which runs only after a MODEL STEP. If the
aborted request was the engine's last, the batch is empty, no step runs,
the farewell never ships: engine idle + client waits to timeout.
Evidence: qwen3t32 baseline starts=1953/dones=1952, GPU 0%, eval in S
state. FIX (next session opener, ~10-30 LOC): flush pending outputs on
empty-batch steps in the engine-core loop. Explains tonight's straggler
+ likely a share of historical mystery tails.

FIX IMPLEMENTED (2026-06-11): `has_finished_requests()`
(scheduler.py:14451) now also returns True while
`_pending_engine_core_outputs` is non-empty. That one predicate gates
BOTH stall points — the busy-loop block (`has_work()`, core.py:1125) and
the `step()` early-return (core.py:389) — so any queued farewell forces
one more (empty-batch) step, whose `update_from_output` drains it via
the existing 13387 path. Rides the same proven empty-batch step that
already runs whenever `finished_req_ids` is non-empty; sole caller of
the predicate is `has_requests()`, so no semantic blast radius. No
busy-spin risk: the pending dict only gains keys via append-at-index
(6029) and is fully `.clear()`ed at drain (13392). Validation = rerun of
the hung shape (qwen3t32 flags-off baseline), which also fills the
missing thinking-T32 wall cell.

## HANG ROUND 2 (2026-06-11 ~23:20): NOT the farewell — an in-engine SPIN
Rerun WITH the farewell fix loaded (verified: PYTHONPATH puts the
submodule ahead of site-packages) hung again at 63/64, same tail.
Live diagnosis before killing it:
- EngineCore main thread at 109% CPU, GPU 0% — SPINNING, not blocked.
- /metrics FROZEN across 10s samples (preemptions stuck at 2233, cache
  byte-identical 78.7%) => the engine never finishes a step: it is stuck
  INSIDE one schedule()/update_from_output call, in a loop that emits no
  ERROR-level logs (VLLM_LOGGING_LEVEL=ERROR hides the WARNING-level
  progress-guard logs, so paths taken are invisible).
- Scheduler state: 1 running, 0 waiting — but the CLIENT has TWO
  outstanding requests (trace starts=1901/dones=1899, idx 1891 & 1895).
  One request is not in the scheduler at all (lost without farewell —
  consistent with being aborted inside the very call that then wedged).
- The original farewell theory was incomplete: engine CPU% was never
  checked in round 1; the same spin likely WAS the round-1 hang.
- Loops audited by eye and NOT the culprit (all have progress guards or
  fail-exits): compaction loop 13801 (breaks on 0 evicted), replay
  segmented refill 4720 (deletes head or returns False), alloc/preempt
  1459 (would climb the frozen preemption counter), admission evict 5194
  (breaks on 0), cpu-archive alloc 10873 (returns None on no victim).
- Stack capture was impossible live (ptrace_scope=1, no sudo, no gdb,
  228 GB RSS rules out a core dump) => SHIPPED a permanent hook:
  run_engine_core now registers faulthandler on SIGUSR1
  (vllm/v1/engine/core.py) — `kill -USR1 <EngineCore pid>` dumps all
  thread stacks to stderr (lands in the inference log).
- Reproducing now with an auto-USR1 stall monitor (fires when the openai
  trace goes quiet >180s mid-sampling). Hang dirs preserved:
  ..._qwen3t32_base_20260611_hung (round 1) and _hung2 (round 2).

## ROOT CAUSE FOUND + FIXED (2026-06-11 ~23:58): THE MODEL-LEN WEDGE
Round 3 (diag run: VLLM_LOGGING_LEVEL=WARNING + census + USR1 trap)
caught it. The USR1 stack proved the engine was healthy-looping in the
no-op-step sleep branch (core.py:1204), and the pre-existing
[SCHED-LIVENESS] instrument (visible only at WARNING level — the
launcher's ERROR quiet-probe had been hiding it all along) printed the
wedged request every 5s:
  computed=14591/14592 prompt=14592 visible_blocks=912 pos=2560
  active_restores=1 active_restore_blocks=112
ARITHMETIC: the running-loop clamp (scheduler.py ~1394) is
  num_new = min(num_new, max_model_len-1 - hidden_kv - computed)
  = min(1, 16383 - 1792 - 14591) = min(1, 0) = 0
-> silent `continue` at the num_new_tokens==0 skip, EVERY step, forever.
The request's visible prompt (14,592 = exactly 912 blocks, full-prompt
prefix-cache hit clamped to prompt-1) plus its hidden recall spans (112
blocks = 1,792 tok) equals EXACTLY max_model_len: structurally
unfinishable, and the scheduler skipped instead of finishing it. Engine
no-op-spins (109-133% CPU, GPU 0%, frozen /metrics because no outputs),
client hangs. The round-1 farewell theory was a red herring; this wedge
was the hang all three rounds. The client pre-flight guard cannot catch
it: it counts visible tokens only (14592+1024 < 16384 passes) — hidden
restore KV lives engine-side.
FIX (scheduler.py running loop): compute model_len_headroom explicitly;
if a request has tokens left but headroom==0, FINISH it as
FINISHED_LENGTH_CAPPED (status -> _free_request ->
_queue_finished_request_output -> pop from running). Client sees
finish_reason=length and ends the rollout — same contract as the
pre-flight guard. Also covers the decode-at-boundary variant (visible
num_tokens never reaches max_model_len when hidden KV exists, so the
stock stop check can never fire). [SCHED-MODEL-LEN-WEDGE] warning marks
each occurrence. Synergy: the farewell-drain fix guarantees the
finish ships even if this was the engine's last request.
Same disease as the overflow class: turn-counted eviction + thinking
turns lets visible+hidden grow to the boundary — token-budget eviction
(which must count hidden restore KV too) dissolves the class.
Census extended with [SCHED-CENSUS-RUNNING-STALLED] (running-but-
unscheduled rows: C/T/P, placeholders, offset, replay/padding flags).

VALIDATION PASS (2026-06-12 00:15): the shape that wedged 3/3 times at
63/64 now completes 64/64, exit 0, ZERO client errors (only 2 benign
[CONTEXT-OVERFLOW] guard rollout-ends), all-200 HTTP, reward 0.0445
(normal for the 4/2 managed thinking cell). Wall 630s launcher-clocked
(00:04:35->00:15:05, ~75s boot) — this is the previously-missing
managed-flags-off thinking-T32 cell; compare fullctx-T32 687.5s
(reward 0.226 — the 4/2 quality gap stands, token-budget eviction is
the cure). Two boundary calls show start-without-done in the openai
trace but their games completed and recorded — trace-logging artifact,
not a correctness issue. Hang class CLOSED:
- farewell drain on empty engine (vllm 99980db47)
- USR1 stack dumps in EngineCore (47f197770)
- model-len wedge -> FINISHED_LENGTH_CAPPED (6f822679c)

## CRAFTER PROD (0.95) FINDINGS — 2026-06-12

Port: managed-context client now shared (src/kv_eviction/eval_client.py),
eval_crafter.py takes the same flags as textworld. Confirmed by smoke
(320 calls, 0 errors, prompts window-bounded, recall loop live).

TWO INCIDENT CLASSES that masqueraded as env/behavior bugs:
1. ONE MANAGED SERVER AT A TIME: a second managed server's pinned CPU
   archive registration (~144 GiB) SIGKILLs the resident engine
   (exitcode=None, no traceback) and fails its own init. Downstream the
   eval aborts every in-flight rollout (ModelError -> APIConnectionError
   'All connection attempts failed') and exits 0 with partial games
   recorded as complete. 2/2 reproductions. Audit rule: grep eval logs
   for "Aborted rollout due to" (any class), never just one error name.
2. The no-soft managed arm at prod pressure loses rollouts by design:
   [PHASE4-PREFIX-ABORT] pin_recovery='Phase4 pin is missing' 500s (22
   by 31% of the run) — pins reclaimed under pressure with no CPU
   backing. The soft stack eliminates the class (0 errors).

OPEN ENGINEERING ITEM — RELOAD BREAKS DELTA-PUBLISH INHERITANCE:
at conc 64 the swap valve reclaims windows; the successor call reloads
them into NEW physical blocks; pin-chain delta publish inherits by
block object identity -> identity miss -> EVERY call-end re-stores the
whole window (store_tail ~430 blocks vs 24 healthy; 5.6k publishes in
21 min; p50 call latency 13.8s for 10-token completions; 2.3 calls/s
vs fullctx 24). FIX: carry the pin's CPU block ids through the
swap-reload path so the re-loaded blocks inherit their CPU copies
(the reload knows exactly which CPU blocks it read from). Until then,
managed@c64 on long-window short-turn workloads is thrash-bound; the
honest cell is managed@c16 (engine logs there showed healthy
inherited=454/store_tail=24).

Healthy solo references: diag4 (conc 4): 99-step games, reward 0.102 >
fullctx 0.085. fullctx c16 cell: 261.7s, 97.6 steps, reward 0.085.

CRAFTER C16 FINAL CELLS (2026-06-12, both clean, zero aborts):
fullctx 262s / 97.6 steps / 214 out-tok/s / reward 0.085 vs managed-soft
1,756s / 92.1 steps / 76 out-tok/s / reward 0.060. VERDICT: on a
10-tok-turn, 100-turn, fits-in-pool workload the soft stack is strictly
worse (6.7x wall) — maximal toll, zero pressure upside. Quiet-logging +
WANDB_MODE=disabled bought 2.2x (1/3 of engine main thread was INFO
chatter through wandb's console hook — launchers must always set
VLLM_LOGGING_LEVEL); the rest is the per-call mirror toll + preempt/
reload cycles (kv usage hit 90%, 503 preemptions even at c16) amplified
by the reload-inheritance break. The two levers that change the verdict:
lazy publish (kills the short-turn toll) and reload-inheritance carry
(unlocks c64 where fullctx oversubscribes 4x). Turn-length floor law
extended: crafter (10 tok) << textworld (12) << counting (128) — the
soft stack's win region starts where per-turn decode amortizes the
publish, OR where concurrency oversubscribes the pool.

## CONCURRENCY SWEEP + LEVERS (2026-06-12, crafter thinking keepthink+YaRN, 25-min caps)
Levers shipped (each env-gated, committed separately for rollback):
- Stage 1 main efa3bb2: min_tokens=2 (1-token empty samples aborted
  whole rollouts; managed-prompt-correlated), keepthink/YaRN configs.
- Stage 2 vllm 054a1bf54: KVE_SOFT_PIN_LAZY_PUBLISH (skip pin mirror
  below 0.80 usage). VERIFIED: 0 offload submits @c16 (vs 5,617
  pre-lazy), zero D2H toll, no tok/s regression.
- Stage 3 vllm e0afa6cbf: KVE_PIN_RELOAD_KEEP_CPU (reload keeps CPU copy
  + re-stamps soft_entries -> delta inheritance survives reloads).

SWEEP (cumulative out-tok/s, zero errors all cells, min_tokens fixed
the empty-abort class):
- c16+lazy:          ~255  (usage 68%, preempt 0)
- c24+lazy+keepcpu:  ~318-340 steady (1.33x, no pressure events)
- c32+lazy+keepcpu:  444 peak pre-pressure (1.74x) -> COLLAPSE at
  pressure onset (~min 6: preempt 0->9->83->149->211, last-min tok/s
  334->119->92->194, cum 314 at kill). Auto-killed per rule.
KNEE CONFIRMED between c24 and c32 even with lazy+keepcpu: the cost is
the preempt/resume cycle at >0.92 pool (atomic resume bundles each need
room), not the re-store volume alone. NOTE: quiet logging
(VLLM_LOGGING_LEVEL=ERROR) hides the store_tail instrument, so
keep-cpu's inheritance effect is UNVERIFIED at pressure — a 10-min diag
rerun at WARNING level would disambiguate resume-cost vs re-store-cost.
NEXT ENGINE LEVERS for the knee: lazy-publish threshold tuning (mirror
EARLIER so preempt victims are already CPU-backed: at 0.80 the first
victims at 0.92 still need full stores), watermark band tuning, and the
async-scheduling redesign (decode bookkeeping = the per-stream floor).
Best prod operating point today: c24 (1.33x c16, stable, no pressure).

## HEADLINE 2026-06-12: MANAGED BEATS FULLCTX 2x AT IDENTICAL QUALITY
Cell D (crafter thinking keepthink+YaRN c16): eviction schedule 40/30
(~3 evictions/100-turn game) + KVE_MANAGED_CONTEXT_RECALL_AT_COMPACTION
+ TERSE_MANAGER (the counting cadence winners, finally ported) + all
night levers (lazy publish, keep-cpu, eager archive, min_tokens):
**wall 864s vs fullctx 1,727s — 2.0x faster — reward 0.0852 vs 0.0852,
ach 1.88 vs 1.88, IDENTICAL to 4 decimals, 0 aborts.**
The window-2 collapse (0.41 turns/s) was the 15/5 schedule: 17
compactions/game put the memory-manager duty on every post-compaction
turn; a thinking model deliberates to the 1024 cap on 32% of those
calls -> seats clog. At 40/30+cadence the machinery engages 3x/game,
turns stay ordinary (43 tok mean), sustained 15-20 turns/s through
every compaction wave. SCHEDULE >> all engine micro-levers.
RECIPE (the prod managed config): toml
inference_managed_qwen3think_yarn131k_40_30.toml + RECALL_AT_COMPACTION=1
TERSE_MANAGER=1 LAZY_PUBLISH=1 PIN_RELOAD_KEEP_CPU=1 ARCHIVE_DEVICE=cpu
POLICY=immediate SOFT_STACK=1 MIN_TOKENS=2, client 40/10.
NEXT: concurrency ladder on THIS schedule (c24/c32 — window peak 20k/trace
means c24 ~99% pool at lockstep peak; watch the knee), then textworld
re-validation with the same recipe.

## EVICTION-COUNT SWEEP (2026-06-12, crafter thinking c16, full 64x100 runs)
n=2 (53/43) 413s r=0.0732 | n=3 (40/30) 864s* r=0.0852 | n=4 (32/22)
352s r=0.0824 | n=5 (27/17) 335s r=0.0767 | n=6 (24/14) 318s r=0.0767 |
n=7 (22/12) 352s r=0.0753. fullctx ref 1,727s r=0.0852.
(*n=3 wall = 2-straggler tail; core ~420s.)
THE SCHEDULE LAW: with at-compaction cadence, wall is FLAT across 2-7
evictions (~320-420s, 4-5x faster than fullctx) — eviction count is
free. QUALITY declines monotonically with eviction count (each cut
loses context) AND with cut size at the extreme (n=2's single 43-turn
dump is worst, 0.0732). Sweet spot n=3-4: reward == fullctx within
noise (SE/cell ~0.008). mgr=64 in EVERY cell = manager fires once per
game regardless of n (first compaction; recall persists).
Driver: experiments/crafter_env/run_evict_sweep.sh; summary
results/evict_sweep_summary_20260612.txt.

## MODEL-PICKED RECALL STUDY (2026-06-12, crafter thinking c16, 35/30 keep-5)
Parse-fix chain: think-strip (130a69e) -> nonthinking manager (6989c6d,
didn't help: model emits dangling UNCLOSED "<think> " even with
enable_thinking=false) -> strip dangling opener (d960acf) -> manager
JSON parses: 4-game verify 3/4 success; full 64-game 34 genuine + 18
fallback. Diagnosis tooling: KVE_MANAGED_CONTEXT_RECORD_RECALL_EVENTS=1
+ eval context_events_tail dump (the previews showed VALID JSON behind
the dangling think in every sampled failure).
RESULTS (keep-5 + pick-5 family): fallback-picked 344s/0.0831 |
model-picked 363s/0.0739 | 22/17-more-compactions 411s/0.0767 — all
within ~1 SE of each other and of recency-only n=4 (352s/0.0824).
KEY FACT: the model, choosing freely, picks the 5 MOST RECENT archived
turns every time — identical to the fallback policy. On crafter, recall
CONTENT is immaterial; the pick machinery is now fully operational for
workloads where old context actually matters (textworld counting = the
discriminating env for model-vs-recency picks).

## MARKOVIAN-RECALL TIMED (2026-06-12, the 4th mode's missing cell)
35/30 keep-5 + pick-5, visible_prefill restores, c16, 64x100:
**wall 359s, reward 0.1001, ach 2.20 — BEST reward of the study**,
beating fullctx (0.0852/1.88) at 4.8x its speed and hidden-KV recall
(0.0739) by ~3 SE. The model benefits from SEEING recalled text;
curated visible context > raw full history on crafter. Cost: recalled
spans are recomputed (reference semantics — kv-recall remains the
never-recompute production path; the quality gap prices what visible
recall buys). Single run, SE ~0.008.
Mode flag shipped: kv_mode in RL toml (prime-rl 8c085169, modes.py
0cc7445, EXPERIMENTS.md at repo root).
