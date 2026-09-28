# Experiments: the four KV-context modes

One flag in the RL toml selects how multi-turn context is managed:

```toml
kv_mode = "kv-recall"   # kv-eviction | markovian | kv-recall | markovian-recall

[orchestrator.markovian_thinker]
max_turns = 40          # eviction triggers when live turns exceed this
stride   = 30           # turns evicted per trigger (window keeps max_turns - stride)
```

Everything else (trainer compaction, padding, vLLM args, engine/client
env) expands automatically from the mode; any field you set explicitly
wins. Definitions live in `src/kv_eviction/modes.py`.

The turn policy is shared by all modes: with `max_turns=40, stride=30`
a 100-turn rollout compacts ~3 times (turns ~40/70/100) and keeps the
last 10 turns visible after each cut.

### All flags

User-facing (set these):

| flag | where | default | meaning |
|------|-------|---------|---------|
| `kv_mode` | top level | None (off) | selects one of the 4 modes below |
| `markovian_thinker.max_turns` | `[orchestrator]` | 6 | eviction/truncation trigger: fires when live turns exceed this |
| `markovian_thinker.stride` | `[orchestrator]` | — | turns evicted per trigger; window keeps `max_turns - stride` |
| `kv_recall_max_spans` | `[orchestrator]` + `[inference]` | 5 | recall modes: max archived turns the model may pick per request |

Expanded automatically per mode (override any of them explicitly to win):

| field | set to | used by |
|-------|--------|---------|
| `orchestrator.compaction_padding.enabled` + `phase4_enabled` | true | block-aligned padding + stable per-rollout trace ids (all engine-eviction modes) |
| `orchestrator.compaction_padding.block_size` | 16 | must match vLLM + trainer block size |
| `inference.vllm_extra.compaction_max_turns` / `compaction_eviction_turn_stride` | from `max_turns`/`stride` | engine turn-mode eviction |
| `inference.vllm_extra.compaction_window_size` / `compaction_stride` | 0 | token-window mode disabled (mutually exclusive with turn mode) |
| `inference.vllm_extra.compaction_block_aligned_finish` + `compaction_assume_aligned_turn_boundaries` | true | exact-edge eviction on padded turn boundaries |
| `trainer.compaction.window_size` / `stride` | 4096 / 512 | segmented-forward enable + capacity hint (geometry comes from mirrored events) |
| `trainer.model.attn` | flex_attention | compaction training forward |
| engine env (recall modes) | soft-pin stack, lazy publish, reload keep-cpu, eager CPU archive | the validated production engine recipe (`engine_env_for_mode`) |
| client env (recall modes) | manager at compaction only, terse + non-thinking manager, per-turn spans | the validated recall flow (`client_env_for_mode`) |
| restore mode (recall modes) | `kv` (kv-recall) / `visible_prefill` (markovian-recall) | hidden KV splice vs visible re-prefill |
| recall index | enabled, 12 entries | span summaries the model picks from |

---

## kv-eviction — drop spans, keep the window

KV of evicted turns is freed. The kept window's KV is *spliced*, never
recomputed (positions preserved via RoPE offset).

```
turns:   [1 ........ 30][31 ...... 40] --> trigger at 40
            evicted          kept
GPU KV:    (freed)        (spliced in place, decode continues)
prompt:                   sys + last 10 turns
```

## markovian — client truncation + re-prefill (reference)

The client trims `messages` to the last `max_turns` groups; vLLM sees
normal full-context requests. KV for the window is *recomputed* each
time the window slides. Reference arm: recompute allowed.

```
turn N:    client sends [sys][turns N-39 .. N]  --> vLLM prefills it all
turn N+1:  window slid  --> prefix changed --> re-prefill again
```

## kv-recall — eviction + hidden-KV recall (the production mode)

Like kv-eviction, but evicted spans are archived to CPU. At each
compaction the model reads an index of span summaries and picks up to
`kv_recall_max_spans` archived turns; their KV is copied back and
attached *hidden* (attention-only — the prompt text does not change).
Never recomputed.

```
evict:    [1 ........ 30] --> CPU archive (per-turn spans + summaries)
                |
manager:  model sees index --> {"retrieve": ["T0019","T0020",...]}
                |
restore:  span KV  CPU --> GPU, attached hidden behind the window
attend:   [hidden recalled KV][sys + last 10 turns][new turn]
```

Validated stack rides along automatically: soft-pin (CPU-backed pins),
lazy publish, reload keep-cpu, eager archive offload, at-compaction
manager cadence, terse + non-thinking manager, per-turn spans.

## markovian-recall — eviction + recall by visible re-prefill (reference)

Same archive + model picks as kv-recall, but the chosen spans' *text*
is re-inserted into the prompt and re-prefilled (recompute allowed).
Reference arm for pricing the KV-splice advantage.

```
restore:  span TEXT --> prompt --> vLLM re-prefills it
attend:   [sys][recalled turn text][last 10 turns][new turn]
```

---

## Measured (crafter, Qwen3-4B thinking, 64 games x 100 turns, conc 16, gpu_util 0.95)

| mode             | wall   | notes                                  |
|------------------|--------|-----------------------------------------|
| full-context     | 1,727s | no management; KV grows unbounded       |
| kv-eviction (40/30, no recall) | ~350-420s | "n=3-4" sweet spot        |
| kv-recall        | 363s   | hidden KV splice; never recomputed      |
| markovian-recall | 359s   | best quality of the study (single run)  |

Schedule law: wall is flat for 2-7 evictions/rollout; quality peaks at
~3-4 evictions and erodes with more (and with one giant cut).

Notable: markovian-recall (the model SEES the recalled text) beat
hidden-KV recall by ~3 SE and even full-context — visible recalled
turns act as a curated context. It pays for this by recomputing the
recalled spans (reference semantics; kv-recall is the never-recompute
production path). Single-run numbers; SE/cell ~0.008.

---

## SFT: synthetic recall (planned)

Collect 10k full-context crafter traces, then retrospectively overlay
the eviction/recall structure offline: synthesize the index + a
manager turn whose retrieve JSON "picks" randomly-sampled archived
turns, and have the trainer mask each segment to
[recalled spans at original positions][kept window]. Trains both
acting-from-recall and the retrieve format itself, with a fully
controlled pick distribution (no recency bias, no parse failures in
the data).
