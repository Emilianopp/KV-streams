# Plan: Summarization-based Eviction (Markovian Summarization)

## Context

**Problem.** Both current eviction paths — the Markovian Thinker
client-side turn truncation and the vLLM scheduler's KV-block
eviction — silently drop context. In long-horizon RL envs (BALROG
BabyAI, TextWorld) this silent amnesia is a common failure mode:
agent re-explores mapped rooms, retries solved subgoals, loses the
thread of a multi-step plan.

**Proposal.** When a turn-count trigger fires, synthesize a
model-authored summary of the prior trajectory and splice a regular
`{user: instruction} + {assistant: summary}` exchange into the
conversation. The summary is a first-class model turn: full-credit,
loss-active (trained), with its own `TrainingSample` emitted via
`extras["summary_trainsample"]` through the existing orchestrator
plumbing (PR 6).

### Two modes — one for each eviction path

The summarization feature supports both existing eviction backends:

| `summary.mode`  | Client-side behavior on trigger                                  | vLLM requirements                                                           | Post-trigger shape (client-visible)                  |
|-----------------|------------------------------------------------------------------|-----------------------------------------------------------------------------|------------------------------------------------------|
| `markovian`     | **Full reset** — truncate to `sys + [I, S] + tail`               | Block & turn compaction OFF (client handles all truncation)                 | `sys_prefix + {user:I} + {asst:S} + tail`            |
| `eviction`      | **Append-only** — `messages ← messages + [I, S]`                 | vLLM compaction ON (block OR turn) — handles KV state                        | `... + old turns + {user:I} + {asst:S} + new turns`  |

- **Markovian mode** layers on top of the existing Markovian Thinker
  interceptor. "Full reset" = keep only `sys_prefix + tail`, drop all
  body turns, splice the summary exchange in between. This is the
  strict-Markovian-summarization regime — the model conditions on
  `sys + summary + current turn` and nothing else.

- **Eviction mode** layers on top of vLLM's scheduler KV compaction.
  The interceptor does NOT truncate the client-visible message list;
  it just appends the summary exchange. vLLM's existing block-level
  eviction (window/stride) or turn-level eviction
  (max_turns/stride) handles KV compression on the server. The
  prompt token stream that the trainer sees comes from
  `extras["prompt_token_ids"]` (admission-trim-KL-fix machinery) —
  whatever vLLM actually processed post-compaction.

Both modes share everything else — trigger logic, summary generation
call, recursion guard, `extras["summary_trainsample"]` emission,
trainer behavior. The mode flag only switches the message-list
transform.

### Why this is "fairly straightforward"

Because summaries are already proper model turns (with `loss_mask=True`
on the completion via the `SummaryTrainSample` emission path), the
only difference between modes is whether the client truncates the
outgoing message list. Training, logprobs, advantage assignment, and
segmented-forward behavior are identical across modes.

### Prerequisite status

Markovian Thinker is landed (commit `e2ce590`).
`src/kv_eviction/env.py` already has `_markovian_config`,
`configure_markovian_thinker`, `_autoconfigure_markovian_from_env`,
`pop_markovian_stats`, and the Branch-A truncation path. vLLM
block-level (Phase 2) and turn-level compaction are both landed. This
plan extends both paths.

---

## Desired semantics

### Trigger (both modes)

Inside `patched_create`, after computing group partitioning:

```python
n_real_groups = n_groups - _count_summary_exchanges(messages, scfg.instruction_text)
fires_summary = (
    scfg is not None
    and scfg.enabled
    and scfg.compaction_max_turns > 0
    and n_real_groups > scfg.compaction_max_turns
)
```

`_count_summary_exchanges(messages, instruction_text)` returns the
number of `(user={role:"user", content:instruction_text}, assistant:*)`
pairs in the messages list. Counting *real* groups matters for
eviction mode: the client-visible message list grows monotonically
(append-only), so without discounting prior summary exchanges the
trigger would re-fire on the very next turn. In markovian mode the
client-side truncation naturally resets `n_groups` to 1 (the summary
exchange + tail), so the discount is a no-op — same code path, both
modes.

### Post-trigger message list

```python
I_msg = {"role": "user",      "content": scfg.instruction_text}
S_msg = {"role": "assistant", "content": summary_text}

if scfg.mode == "markovian":
    # Full reset: sys + [I, S] + tail.
    new_messages = sys_prefix + [I_msg, S_msg] + tail
else:  # "eviction"
    # Append-only: leave the full body intact, splice summary before tail.
    # Concretely: insert at the "end of completed body" boundary, so the
    # tail (in-flight messages) still comes last.
    new_messages = sys_prefix + _flatten(body_groups) + [I_msg, S_msg] + tail
```

In eviction mode the message list grows by 2 messages per trigger.
The trainer's view of the post-compaction tokens comes from
`extras["prompt_token_ids"]` on subsequent responses (whatever vLLM
actually processed after its own block/turn eviction). The client
does not need to track what vLLM trimmed.

### Summary generation (identical both modes)

One extra `orig_create` call from inside the interceptor, using
**full pre-truncation history + instruction**:

```python
summary_messages = messages + [I_msg]
```

Giving the model the full client-visible history (not just a dropped
slice) yields higher-quality summaries. The summary call's prompt is
a prefix-extension of what vLLM has cached → perfect prefix-cache hit
on the summary call's prefill.

Summary kwargs are built from scratch. Strip `tools`, `tool_choice`,
`response_format`, `seed`, `extra_body`. **Request logprobs** so the
trainer has them as `old_logprobs`:

```python
summary_kwargs = {
    "model":         outer_kwargs["model"],
    "messages":      summary_messages,
    "max_tokens":    scfg.max_len_summary,   # independent of rollout's max_completion_tokens
    "temperature":   scfg.temperature,
    "top_p":         scfg.top_p,
    "logprobs":      True,
    "top_logprobs":  0,
}
```

`scfg.max_len_summary` is **independent** of the rollout's
`sampling.max_completion_tokens`. Rationale: the summary call has
different length needs than a regular action turn (usually longer).
Decoupling them avoids forcing operators to raise
`max_completion_tokens` globally just to get longer summaries.

### Recursion guard

`contextvars.ContextVar[bool]` named `_IN_SUMMARY_CALL`. Set `True`
before the inner `orig_create`, reset in `finally`. The interceptor
short-circuits on entry when the flag is set. `contextvars`, not
`threading.local` — the interceptor is async and tasks may migrate
between threads.

---

## Training plumbing (summaries as trainable turns — identical for both modes)

### Response-side capture

After `orig_create` returns the summary response, the interceptor
builds a `SummaryTrainSample`:

```python
@dataclass
class SummaryTrainSample:
    prompt_token_ids:     list[int]        # tokens for (messages + [I_msg]) as vLLM processed
    completion_token_ids: list[int]        # summary assistant tokens + <|im_end|>
    completion_logprobs:  list[float]      # scalar logprobs per completion token
    model:                str              # trainer sanity check
```

Sources (all exposed by our vLLM fork + existing admission-trim stash
machinery):
- `prompt_token_ids` ← `resp.prompt_token_ids` (admission-trim patch)
- `completion_token_ids` ← `resp.choices[0].token_ids` with
  `tokenizer.encode(message.content, add_special_tokens=False)` + end-
  of-turn token as a fallback (unit-tested both branches)
- `completion_logprobs` ← `resp.choices[0].logprobs.content[i].logprob`

Attach to the *main* (outer) response via a monkey-attribute (same
pattern as `compaction_events` in env.py:130–146):

```python
outer_response.extras = {
    **(getattr(outer_response, "extras", {}) or {}),
    "summary_trainsample": summary_train_sample.to_dict(),
}
```

### Trajectory forwarding

`src/kv_eviction/env.py:_extract_compaction_event_dicts` demonstrates
the pattern. Add `_extract_summary_trainsample(response) -> dict |
None`, and in the patched verifiers `from_native_response` path
attach to `TrajectoryStep.extras["summary_trainsample"]`.

### Orchestrator-side emission
(`prime-rl/src/prime_rl/orchestrator/trajectories.py:249–400`,
`interleave_rollout`)

- After building each step's regular `TrainingSample`, check
  `step.extras.get("summary_trainsample")`. If present, construct a
  second, independent `TrainingSample`:
  - `prompt_ids` ← `summary.prompt_token_ids`
  - `completion_ids` ← `summary.completion_token_ids`
  - `completion_mask` ← `[True] * len(completion_ids)` — **trained**
  - `prompt_mask` ← `[False] * len(prompt_ids)`
  - `old_logprobs` ← `summary.completion_logprobs`
  - `advantage` ← terminal step's advantage (inherit; see Risks)
  - `env_id`, `rollout_id` copied from owning step
  - `step_index` slotted between the trigger step and the next real
    step (same numbering rule as existing extension-break samples)

Emit summary sample first, regular sample second — matches temporal
generation order.

### Trainer-side

No changes. Summary samples have the same shape as any other; the
existing `loss_mask = prompt_mask + completion_mask` path
(`trainer/batch.py:48`) activates loss on summary completion tokens;
`segmented_forward` sees the sample as a normal full-context forward
(no compaction events → single segment), matching the unified-dispatch
D5 pattern.

---

## Config surface

### New Pydantic model (`prime-rl/src/prime_rl/configs/orchestrator.py`)

Nest inside existing `MarkovianThinkerConfig` (orchestrator.py:865):

```python
class MarkovianSummaryConfig(BaseConfig):
    enabled: bool = False

    mode: Literal["markovian", "eviction"] = "markovian"

    # Turn-count trigger (both modes).
    compaction_max_turns: Annotated[int, Field(ge=0)] = 0

    # Summary generation length — INDEPENDENT of sampling.max_completion_tokens.
    max_len_summary: Annotated[int, Field(ge=16, le=8192)] = 512

    instruction_text: str = (
        "You will lose context of the prior turns. Please summarize "
        "everything important for the task — the current goal, the "
        "state you've observed, what you've already tried, and any "
        "facts you'll need to continue acting effectively."
    )

    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = 0.3
    top_p: Annotated[float, Field(gt=0.0, le=1.0)] = 0.95

    on_error: Literal["drop", "raise"] = "drop"
    log_summaries: bool = False

class MarkovianThinkerConfig(BaseConfig):
    ...  # existing fields
    summary: MarkovianSummaryConfig = MarkovianSummaryConfig()
```

Notes on the parameter surface vs the v3 design:
- **Removed** `compaction_eviction_turn_stride`. Markovian summary mode uses
  `markovian_thinker.stride` as its semantic-group preservation count;
  eviction mode's kept-count is determined by vLLM's own compaction config.
- **Renamed** `max_tokens` → `max_len_summary` for clarity that it's
  independent of the rollout's `max_completion_tokens`.

### Validators (`prime-rl/src/prime_rl/configs/rl.py`, extend `validate_markovian_thinker` at rl.py:515)

Common:
- `summary.enabled=true` requires `markovian_thinker.enabled=true`.
- `summary.enabled=true` requires `compaction_max_turns > 0`.
- `summary.enabled=true` with a separate `teacher_rollout_model` →
  raise (v1 requires same model for logprob validity).
- `inference.model.max_model_len < max_len_summary + 2048` → raise.

Mode-specific:
- `mode="markovian"`: vLLM block compaction (`window_size`) AND turn
  compaction (`compaction_max_turns` in vllm_extra) MUST both be 0
  (client handles all truncation).
- `mode="eviction"`: at least one of (`inference.vllm_extra.compaction_window_size
  > 0`) or (`inference.vllm_extra.compaction_max_turns > 0`) must be
  true (vLLM has to actually do eviction).
- `mode="eviction"`: further, `trainer.compaction` must match the
  vLLM-side block-compaction params (identical pre-existing rule for
  block compaction — inherited, not a new check).

---

## Interceptor changes (`src/kv_eviction/env.py`)

### Module-level additions (mirror Markovian pattern at lines 427–816)

```python
@dataclass
class MarkovianSummaryRuntimeConfig:
    enabled: bool
    mode: str                          # "markovian" | "eviction"
    compaction_max_turns: int
    max_len_summary: int
    instruction_text: str
    temperature: float
    top_p: float
    on_error: str                      # "drop" | "raise"
    log_summaries: bool

_summary_config: MarkovianSummaryRuntimeConfig | None = None
_IN_SUMMARY_CALL: ContextVar[bool] = ContextVar(
    "_IN_SUMMARY_CALL", default=False)

def configure_markovian_summary(**kwargs) -> None: ...
def _autoconfigure_markovian_summary_from_env() -> None: ...
```

Extend `_markovian_stats` with: `n_summaries`, `n_summary_failures`,
`summary_prompt_tokens`, `summary_output_tokens`,
`summary_latency_ms`. `pop_markovian_stats` drains all of them.

### Env-var autoconfigure for worker subprocesses

Mirror `_autoconfigure_markovian_from_env` (env.py:771). Scalars:
`KV_EVICTION_MARKOVIAN_SUMMARY_{ENABLED,MODE,COMPACTION_MAX_TURNS,
MAX_LEN_SUMMARY,TEMPERATURE,TOP_P,ON_ERROR}`. Long string
(`instruction_text`) via JSON env var:
`KV_EVICTION_MARKOVIAN_SUMMARY_STRINGS_JSON`.

### `patched_create` Branch-A extension

Near existing Markovian block (env.py:527–582):

```python
if _IN_SUMMARY_CALL.get():
    return await orig_create(self, *args, **kwargs)

mcfg = _markovian_config
scfg = _summary_config

if mcfg and mcfg.enabled:
    messages = kwargs.get("messages")
    if messages is not None:
        n_groups, sys_prefix, body_groups, tail = _partition_messages(messages)

        fires_summary = False
        if scfg and scfg.enabled and scfg.compaction_max_turns > 0:
            n_real = n_groups - _count_summary_exchanges(
                messages, scfg.instruction_text)
            fires_summary = n_real > scfg.compaction_max_turns

        summary_train_sample_dict: dict | None = None

        if fires_summary:
            summary_text, summary_train_sample_dict = await _generate_summary(
                self, orig_create, scfg,
                outer_kwargs=kwargs, full_messages=messages,
            )
            if summary_text:
                I_msg = {"role": "user",      "content": scfg.instruction_text}
                S_msg = {"role": "assistant", "content": summary_text}
                if scfg.mode == "markovian":
                    new_messages = sys_prefix + [I_msg, S_msg] + tail
                else:  # eviction
                    new_messages = (sys_prefix + _flatten(body_groups)
                                    + [I_msg, S_msg] + tail)
            else:
                # Summary failed → plain Markovian truncation fallback.
                new_messages = truncate_messages_to_last_k_turns(
                    messages, max_turns=mcfg.max_turns)
                summary_train_sample_dict = None
        else:
            new_messages = truncate_messages_to_last_k_turns(
                messages, max_turns=mcfg.max_turns)

        # Existing rerender + encode + prompt_token_ids stash (unchanged).
        ...
        response = await orig_create(self, *args, **kwargs)
        _stash_prompt_token_ids(response, truncated_ids)

        if summary_train_sample_dict is not None:
            _attach_summary_trainsample(response, summary_train_sample_dict)

        _update_markovian_stats(...)
        return response
```

Key differences from the single-mode design:
- New `mode` branch in the post-summary message-list construction.
- `n_real_groups` discount via `_count_summary_exchanges` so the
  append-only eviction mode doesn't re-fire immediately.
- Plain-truncation fallback uses `mcfg.max_turns` (Markovian Thinker's
  existing knob) — unchanged fallback contract regardless of summary
  mode. Defensive: even with summary turned on, a summary failure
  shouldn't leave messages un-compacted.

### `_generate_summary` (async, in `env.py`)

```python
async def _generate_summary(self, orig_create, scfg, outer_kwargs,
                            full_messages) -> tuple[str | None, dict | None]:
    I_msg = {"role": "user", "content": scfg.instruction_text}
    summary_messages = full_messages + [I_msg]

    summary_kwargs = {
        "model":         outer_kwargs["model"],
        "messages":      summary_messages,
        "max_tokens":    scfg.max_len_summary,
        "temperature":   scfg.temperature,
        "top_p":         scfg.top_p,
        "logprobs":      True,
        "top_logprobs":  0,
    }

    token = _IN_SUMMARY_CALL.set(True)
    t0 = time.perf_counter()
    try:
        resp = await orig_create(self, **summary_kwargs)
    except Exception:
        if scfg.on_error == "raise":
            raise
        logger.warning("markovian summary failed; falling back", exc_info=True)
        _markovian_stats["n_summary_failures"] += 1
        return None, None
    finally:
        _IN_SUMMARY_CALL.reset(token)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    text = (resp.choices[0].message.content or "").strip()
    if not text:
        _markovian_stats["n_summary_failures"] += 1
        return None, None

    prompt_ids           = _extract_prompt_token_ids(resp)
    completion_ids       = _extract_completion_token_ids(resp)
    completion_logprobs  = _extract_completion_logprobs(resp)

    sample_dict = {
        "prompt_token_ids":     prompt_ids,
        "completion_token_ids": completion_ids,
        "completion_logprobs":  completion_logprobs,
        "model":                outer_kwargs["model"],
    }

    _markovian_stats["n_summaries"]            += 1
    _markovian_stats["summary_prompt_tokens"]  += len(prompt_ids)
    _markovian_stats["summary_output_tokens"]  += len(completion_ids)
    _markovian_stats["summary_latency_ms"]     += latency_ms
    return text, sample_dict
```

`orig_create` (not `self.create`) — routing around the patched
method entirely, belt + suspenders with the ContextVar guard.

### New pure module (`src/kv_eviction/summarization.py`)

Testable core out of `env.py`:
- `partition_messages(messages, ...) -> tuple[int, list, list, list]`
  (factored out of `truncation.py`; re-exported there)
- `count_summary_exchanges(messages, instruction_text) -> int` —
  counts `(user:I, assistant:*)` pairs
- `build_exchange(instruction_text, summary_text) -> tuple[dict, dict]`
- `sanitize_summary(text, blocklist=("<|im_start|>", "<|im_end|>"))
   -> tuple[str, bool]`
- `content_to_text(content)` — multimodal flatten for debug logging
- `SummaryTrainSample` dataclass + `to_dict`/`from_dict`

No async, no I/O.

---

## Orchestrator wiring (`prime-rl/src/prime_rl/orchestrator/`)

### orchestrator.py

Inside `if config.orchestrator.markovian_thinker.enabled`:
- `configure_markovian_summary(
    enabled=cfg.markovian_thinker.summary.enabled,
    mode=cfg.markovian_thinker.summary.mode,
    compaction_max_turns=cfg.markovian_thinker.summary.compaction_max_turns,
    max_len_summary=cfg.markovian_thinker.summary.max_len_summary,
    instruction_text=cfg.markovian_thinker.summary.instruction_text,
    temperature=cfg.markovian_thinker.summary.temperature,
    top_p=cfg.markovian_thinker.summary.top_p,
    on_error=cfg.markovian_thinker.summary.on_error,
    log_summaries=cfg.markovian_thinker.summary.log_summaries,
  )`
- Export `KV_EVICTION_MARKOVIAN_SUMMARY_*` env vars for subprocess
  propagation.
- Per-step `pop_markovian_stats()` drain + wandb log extension with
  `markovian_summary/*` keys.

### trajectories.py (interleave_rollout)

In each step's processing (~line 249), check
`step.extras.get("summary_trainsample")`. If present, after building
the regular `TrainingSample`, build and emit a second
`TrainingSample` per the spec in "Training plumbing". Emit summary
sample first.

Order matters for:
- Rollout reproducibility (summary temporally precedes the next
  real turn's generation)
- Batch advantage consistency (both samples share the episode's
  advantage; consecutive ordering makes grouping easier)

---

## Files to change

| File | Change |
|---|---|
| `src/kv_eviction/env.py` | Runtime config, ContextVar, configure/autoconfigure, `_generate_summary`, `_attach_summary_trainsample`, Branch-A extension, `_markovian_stats` additions |
| `src/kv_eviction/summarization.py` | **NEW.** Pure helpers + `SummaryTrainSample` + extract/attach |
| `src/kv_eviction/truncation.py` | Factor `partition_messages` + `count_summary_exchanges` helpers |
| `prime-rl/src/prime_rl/configs/orchestrator.py` | `MarkovianSummaryConfig` (with `mode` + `max_len_summary`) nested in `MarkovianThinkerConfig` |
| `prime-rl/src/prime_rl/configs/rl.py` | Extend `validate_markovian_thinker` — mode-specific validators |
| `prime-rl/src/prime_rl/orchestrator/orchestrator.py` | `configure_markovian_summary` call + env var export + wandb logging |
| `prime-rl/src/prime_rl/orchestrator/trajectories.py` | In `interleave_rollout`: emit second `TrainingSample` from `extras["summary_trainsample"]` |
| `experiments/debug_balrog/rl_markovian_summary.toml` | **NEW.** `mode="markovian"`, full-reset smoke |
| `experiments/debug_balrog/rl_eviction_summary.toml` | **NEW.** `mode="eviction"` + vLLM block compaction on, smoke |
| `tests/test_summarization.py` | **NEW.** Unit tests — pure helpers + `count_summary_exchanges` + `SummaryTrainSample` roundtrip |
| `tests/test_summary_interceptor.py` | **NEW.** Mocked-`orig_create` interceptor tests for both modes |
| `prime-rl/tests/unit/test_markovian_summary_config.py` | **NEW.** Validator tests |
| `prime-rl/tests/unit/test_interleave_summary.py` | **NEW.** trajectories.py emission tests |

**Not touched:** `src/kv_eviction/segmented_forward.py`,
`src/kv_eviction/padding.py`, `prime-rl/src/prime_rl/trainer/*`,
`vllm/*`.

---

## Implementation order (7 incremental PRs)

1. **Pure helpers** — `summarization.py` +
   `partition_messages`/`count_summary_exchanges` factoring in
   `truncation.py` + `test_summarization.py`. Zero runtime impact.
2. **Config surface** — `MarkovianSummaryConfig` (with `mode` +
   `max_len_summary`) + mode-specific validators +
   `test_markovian_summary_config.py`. No-op with summary off.
3. **Runtime config plumbing** — `env.py` dataclass, ContextVar,
   configure/autoconfigure. No interceptor behavior change.
4. **Interceptor extension (summary generation + mode-aware message
   rewrite)** — Branch-A summary path (both modes) without the
   `extras` attachment yet; mocked `orig_create` tests for message
   shape / recursion / error paths / mode branching.
5. **Response extras attachment + trajectory forwarding** —
   `SummaryTrainSample` capture in `env.py`; extract in verifiers
   patch; carry through to `TrajectoryStep.extras["summary_trainsample"]`.
   Roundtrip unit test.
6. **Trajectories emission** — `interleave_rollout` emits the second
   `TrainingSample`. `test_interleave_summary.py` covers advantage
   inheritance, ordering, no-summary passthrough.
7. **Experiment configs + smoke runs** —
   `rl_markovian_summary.toml` (markovian mode) +
   `rl_eviction_summary.toml` (eviction mode + vLLM block compaction
   on), 5-step smokes on BabyAI and TextWorld; verify mode-specific
   prompt shapes in the Gradio viewer; confirm summary samples appear
   in the batch; confirm loss flows through summary completion tokens.

Each PR <350 lines of diff. PRs 1–3 are bit-for-bit no-ops on existing
runs. PR 4 produces the correct message list but drops summaries from
training; PR 5 restores them. PR 6 is the gradient-flow gate.

---

## Test plan

### Unit (`tests/test_summarization.py`)
- `partition_messages` over plain + tool-chain groups
- `count_summary_exchanges` with 0, 1, multiple matches; handles
  tail messages correctly; no false positives on non-exact-match
  user content
- `build_exchange` returns correct roles/contents
- `sanitize_summary` strips chat-template tokens; identity on clean
  input; `was_modified` flag accurate
- `SummaryTrainSample.to_dict`/`from_dict` roundtrip lossless
- Completion-logprobs extraction handles missing/empty/`None`
  defensively

### Interceptor (`tests/test_summary_interceptor.py`, mocked
`orig_create`, parametrized over mode)
- Trigger fires at `n_real > compaction_max_turns` → 2 invocations
- Summary call kwargs: no `tools`, `tool_choice`, `response_format`,
  `extra_body`; has `logprobs=True`; `max_tokens == max_len_summary`
- **markovian mode**: final messages = `sys + I + S + tail`
- **eviction mode**: final messages = `sys + body + I + S + tail`
  (body untouched)
- **eviction mode re-fire**: after one summary injection, on next
  call with only 1 more real turn appended,
  `count_summary_exchanges` discount prevents re-fire
- **eviction mode**: after `compaction_max_turns` MORE real turns
  appended, trigger re-fires (telescoping path)
- Below trigger → passthrough, one invocation
- Recursion guard: nested `_IN_SUMMARY_CALL` bypasses interceptor
- `on_error="drop"` + summary raises → plain truncation, no extras
- `on_error="raise"` propagates
- Empty summary → treated as failure
- Stashed `prompt_token_ids` ==
  `tokenizer.encode(apply_chat_template(new_messages))` for each mode
- `response.extras["summary_trainsample"]` shape + types
- Stats counters increment + drain

### Validator (`prime-rl/tests/unit/test_markovian_summary_config.py`)
- Default loads (summary off)
- `summary.enabled` without `markovian.enabled` raises
- `compaction_max_turns=0` raises
- `max_len_summary` + `max_model_len` headroom check raises with
  concrete numbers
- Rejects `teacher_rollout_model`
- **mode="markovian"**: rejects any vLLM-side compaction on
- **mode="eviction"**: raises if neither `vllm_extra.window_size >
  0` nor `vllm_extra.compaction_max_turns > 0`
- **mode="eviction"** + block-compaction on: inherits
  `trainer.compaction` match validator

### Trajectories (`prime-rl/tests/unit/test_interleave_summary.py`)
- Step with `extras["summary_trainsample"]` → two samples emitted
  (summary first)
- Step without → one sample
- Summary sample: `prompt_mask` False, `completion_mask` True,
  logprobs length matches completion length, advantage inherited
  from terminal sample
- `rollout_id` / `env_id` / `step_index` slotted correctly

### Integration smokes (manual, 5 steps)
- **BabyAI markovian full-reset** (`rl_markovian_summary.toml`,
  `mode=markovian, compaction_max_turns=4, max_len_summary=512`):
  `markovian_summary/n_per_step>0` by step 3; loss finite; mismatch
  KL ~kernel floor; post-trigger prompt in Gradio = `sys + I + S +
  tail`; summary samples in batch.
- **BabyAI eviction** (`rl_eviction_summary.toml`, `mode=eviction`,
  vLLM block compaction: `window_size=4096, stride=512`;
  `compaction_max_turns=4, max_len_summary=512`): same metrics
  + post-trigger Gradio prompt shows `sys + body + I + S + tail` (body
  present), and the trainer's `prompt_token_ids` is shorter than the
  client-visible messages (proof vLLM evicted); summary samples
  appear in the batch.
- **TextWorld markovian**: same on simple-chat env.
- **Parity regression**: `rl_markovian_summary.toml` with
  `summary.enabled=false` → token-identical to existing
  `rl_markovian.toml` at step 0.
- **Loss flow**: dump `loss_per_sample`; summary samples contribute a
  finite, nonzero loss component; no NaN.

---

## Risks / open questions

1. **Advantage assignment for summaries (research).** v1 inherits
   the terminal `TrainingSample`'s advantage. Alternatives: zero
   advantage (imitation-only), step-local propagation,
   separately-learned scalar. Expose a future
   `summary.advantage_strategy` enum if needed.
2. **Prefix cache across trigger.** Markovian mode resets the entire
   prefix → cache miss guaranteed. Eviction mode: vLLM evicts blocks
   but keeps the rest of the KV state → near-perfect preservation
   within vLLM. Within an epoch (between triggers), cache hits are
   perfect in both modes.
3. **Summary-call latency.** ~100–500 ms per trigger on loaded DP=4.
   Mitigated by tight `max_len_summary` and DP parallelism.
4. **Summary quality on small models.** 4B policy may produce weak
   summaries early; gradient on summary tokens should improve them
   over training. May need a curriculum (enable summarization after
   N warm-up steps).
5. **Telescoping summaries.** Both modes: new summaries see prior
   summaries in context → compounding compression. Intentional but
   quality risk. Smoke: 60-turn TextWorld; dump every summary;
   entity retention check.
6. **`enable_prefix_caching`.** Already disabled under Markovian.
   For eviction mode we can leave vLLM's prefix caching at its
   default (it works correctly with turn/block compaction).
7. **Non-ChatML templates.** v1 scoped to Qwen-family ChatML.
8. **`tools` on summary call.** v1 strips them. A/B whether
   `tool_choice="none"` + `tools` yields better summaries on rg-mix.
9. **Training distribution shift.** Policy learns to emit + condition
   on summaries. Deploying without the interceptor becomes OOD.
10. **Logprobs path reliability.** Depends on
    `resp.choices[0].token_ids` and
    `resp.choices[0].logprobs.content`. Unit test both populated and
    missing paths. Non-streaming only.
11. **Eviction mode interacts with two pre-existing compaction
    validators** (trainer vs vLLM sync; AC-incompatibility). The new
    mode doesn't introduce new relaxations — inherit both.

---

## Verification (end-to-end)

1. Fresh clone: `bash setup.sh`.
2. Validator tests: `uv run pytest
   prime-rl/tests/unit/test_markovian_summary_config.py -v`.
3. Helper + interceptor + trajectories tests: `uv run pytest
   tests/test_summarization.py tests/test_summary_interceptor.py
   prime-rl/tests/unit/test_interleave_summary.py -v`.
4. BabyAI markovian smoke (5 steps):
   `bash experiments/debug_balrog/launch.sh rl_markovian_summary.toml`.
   Watch `markovian_summary/n_per_step`, mismatch KL, loss finite.
5. BabyAI eviction smoke (5 steps):
   `bash experiments/debug_balrog/launch.sh rl_eviction_summary.toml`.
   Same checks + verify vLLM-side eviction via `compaction_events`
   metrics (inherit).
6. Gradio viewer: confirm mode-specific prompt shapes; confirm
   summary `TrainingSample`s appear as separate entries with their
   own `completion_ids`.
7. Parity regression: `summary.enabled=false` → step-0 rollouts
   token-identical to existing `rl_markovian.toml`.
8. Loss flow: dump per-sample loss at step 5; summary samples
   contribute finite, nonzero loss; no NaN.

---

## Critical files referenced

- `src/kv_eviction/env.py:427–816` — Markovian runtime config +
  `patched_create` + autoconfigure (extension target)
- `src/kv_eviction/env.py:38–146` — existing `compaction_events`
  extract + attach pattern (mirror for `summary_trainsample`)
- `src/kv_eviction/truncation.py` — `truncate_messages_to_last_k_turns`
  + turn-grouping helper (refactored to share `partition_messages` +
  `count_summary_exchanges`)
- `prime-rl/src/prime_rl/configs/orchestrator.py:865` —
  `MarkovianThinkerConfig` (nest target)
- `prime-rl/src/prime_rl/configs/rl.py:515` —
  `validate_markovian_thinker` (extension target)
- `prime-rl/src/prime_rl/orchestrator/orchestrator.py` — wiring +
  metrics drain
- `prime-rl/src/prime_rl/orchestrator/trajectories.py:249–400` —
  `interleave_rollout` (summary-sample emission site)
- `prime-rl/src/prime_rl/trainer/batch.py:48` — `loss_mask`
  construction (no change; the place summary tokens become loss-active)
- `plans/markovian_thinker_baseline.md` — prior Markovian design
- `plans/phase2_vllm_compaction.md` — vLLM-side block/turn eviction
- `plans/admission_trim_kl_fix.md` — `prompt_token_ids` stash
  mechanism (reused unchanged)
