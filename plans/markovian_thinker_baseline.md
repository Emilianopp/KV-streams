# Plan: Markovian Thinker Baseline (Client-Side Message Truncation)

## Status: Design — Ready for Implementation

## Motivation

The repo currently supports two regimes for multi-turn RL:

1. **Full-context, no-eviction** (`experiments/debug_balrog/rl_no_eviction.toml`):
   vLLM sees the entire growing conversation; capped by env-level
   `max_text_history` or `max_turns` before OOM.
2. **vLLM-side KV compaction** (`compaction_rgmix`, `compaction_textworld`):
   orchestrator sends the full conversation, vLLM's scheduler evicts
   KV blocks (block-FIFO or turn-based), trainer mirrors via
   `segmented_forward`.

**Markovian Thinker** introduces a third regime: **client-side message
truncation**. The orchestrator truncates the `messages` list to the
most recent K turns *before* sending each chat completion request.
vLLM sees every request as a normal full-context completion — no
compaction, no eviction, no `CompactionEvent`s, no `segmented_forward`.

Why we want this baseline:

| Concern                         | vLLM compaction                              | Markovian Thinker                              |
|---------------------------------|----------------------------------------------|------------------------------------------------|
| Information-loss granularity    | Token-level (block/turn boundaries in KV)    | Message-level (whole turns dropped)            |
| Eviction artifacts visible?     | Yes (orphan fragments, RoPE offsets)         | No (clean message sequence)                    |
| Training complexity             | `segmented_forward` + compaction event replay| Standard full-context forward                  |
| Inference-engine coupling       | Requires the vLLM fork                       | Any OpenAI-compatible server works             |
| Prefix caching across turns     | Disrupted by KV eviction within a request    | Different prefix each turn → cache misses      |
| Failure mode                    | Mid-message fragmentation, KL replay drift   | Clean abrupt forgetting                        |

Markovian Thinker is the simplest possible compaction baseline and
provides a lower bound on information-retention quality. If turn-based
vLLM compaction does not beat Markovian Thinker, the added complexity
is not justified.

**Scope: orchestrator + integration layer only.** No vLLM edits.
No `segmented_forward` changes. No changes to `prime-rl/trainer`.

## Key constraints (from the user)

1. **Default behavior unchanged.** With `markovian_thinker.enabled=false`
   (default), the code path is a zero-cost passthrough and every config
   validates exactly as it does today.
2. **Everything new lives under conditional logic** gated on
   `markovian_thinker.enabled` (either a config field or an env var for
   subprocess propagation).
3. **Truncation happens at the messages level**, not the token level.
   It is the simplest hook and requires no vLLM-side changes.

## Desired semantics

### Logical sequence budget and metrics

Markovian re-prefill does not reset the rollout token budget. The top-level
`seq_len` caps the canonical untruncated prompt plus newly generated tokens.
Initial prompt, environment/tool/user additions, and model completions count
once; retained-history copies sent after truncation do not count again.

Each response records `logical_seq_len` for the full rollout and
`context_seq_len` for the physically submitted truncated request. W&B
`seq_len/*` therefore compares one cumulative logical length per rollout
across Markovian and KV-eviction modes, while `context_seq_len/*` exposes the
smaller Markovian re-prefill window.

### KV-eviction-compatible turn grouping

Markovian uses the same observable message-boundary counter as vLLM turn
eviction. The first chat message is the protected prefix. Every subsequent
pair of message-end boundaries is one completed turn, regardless of roles or
`tool_calls`:

```
system                              <- protected first message
user
assistant (tool_calls=[...])        <- completed turn 1
tool
assistant (tool_calls=[...])        <- completed turn 2
tool                                <- unmatched in-flight tail
```

This deliberately does **not** keep a full tool-call chain atomic. vLLM only
sees `<|im_end|>` markers and also pairs boundaries without inspecting roles.
Using terminal assistant messages instead would let tool-heavy Markovian
rollouts retain substantially more context than KV eviction.

### Protected regions (HARD invariants)

1. **Protected prefix**: exactly the first message, matching
   `turn_end_positions[0]` in vLLM.
2. **In-flight tail**: the unmatched final message, if the post-prefix
   message count is odd.

### Truncation policy

Given total completed turns `N`, `max_turns=M`, and eviction stride `S`
(`S=1` when omitted):

1. If `N < M`, return `messages` unchanged.
2. Otherwise the number of due eviction events is
   `floor((N - M) / S) + 1`.
3. Drop that many multiples of `S` oldest turns.
4. Return `protected_prefix + remaining_turn_pairs + tail`.

This reproduces vLLM's sawtooth live window. For `M=4, S=2`, the live-turn
counts for total completed turns 3 through 7 are `3, 2, 3, 2, 3`.

### Edge-case behavior (explicit)

| Case                                                       | Behavior                                              |
|------------------------------------------------------------|-------------------------------------------------------|
| `messages == []`                                           | Return `[]` (unchanged).                              |
| One message                                                | Protected prefix only; unchanged.                     |
| `[system, user]`                                           | No completed post-prefix pair; `user` is the tail.    |
| `[system, user, assistant]`                                | One completed turn.                                   |
| Completed turns `< max_turns`                              | Return unchanged (identity).                          |
| Completed turns `== max_turns`                             | Evict `stride` turns.                                 |
| `max_turns == 0`                                           | Rejected at config validation (must be >= 1).         |
| Tool-call chain spanning multiple messages                 | Counted as paired message boundaries, like vLLM.      |
| Odd post-prefix message count                              | Final unmatched message is the protected tail.       |
| No system role                                             | First message is still protected, matching vLLM.      |
| Multimodal messages (image content parts)                  | Opaque dict handling — no content inspection.         |
| `tools` kwarg changing mid-rollout                         | Passed through unchanged; truncation only touches `messages`. |
| Tools not present on any message                           | Same as chat envs; no special-casing.                 |
| Returned `messages` length <= input length                 | Always (defensive assert in tests).                   |

## Config surface

### New Pydantic model

Location: `prime-rl/src/prime_rl/configs/orchestrator.py`, immediately
after `CompactionPaddingConfig` (line ~785).

```python
class MarkovianThinkerConfig(BaseConfig):
    """Client-side message truncation baseline for multi-turn RL.

    When enabled, the orchestrator applies the same paired-message turn
    counter and max_turns/stride sawtooth eviction schedule as vLLM before
    sending each request. The first message and unmatched final message are
    protected. vLLM sees normal full-context completions with no engine-side
    compaction or `CompactionEvent`s.

    Incompatible with:
    - `inference.vllm_extra.compaction_window_size > 0`
    - `inference.vllm_extra.compaction_max_turns > 0`
    - `trainer.compaction.window_size > 0`
    - `orchestrator.compaction_padding.enabled = true`
      (redundant — Markovian stashes its own `prompt_token_ids`)
    - `orchestrator.use_token_client = true`
      (TITO assumes extension property across turns, which Markovian breaks)

    Enforced at config load by `validate_markovian_thinker`.
    """

    enabled: Annotated[
        bool,
        Field(description="Master switch for Markovian Thinker truncation."),
    ] = False

    max_turns: Annotated[
        int,
        Field(
            ge=1,
            description=(
                "Turn threshold shared with vLLM. Eviction fires when the "
                "number of completed paired-message turns reaches this value. "
                "Default 6: aggressive enough to exercise the truncation "
                "path on typical BabyAI/TextWorld episodes (~15-30 turns) "
                "while leaving enough context for coherent action selection."
            ),
        ),
    ] = 6

    stride: Annotated[
        int | None,
        Field(
            ge=1,
            description=(
                "Number of oldest completed turns evicted per trigger. "
                "None means 1; otherwise must be <= max_turns."
            ),
        ),
    ] = None

    log_truncated_messages: Annotated[
        bool,
        Field(
            description=(
                "Debug-only: log one line per truncation with the number "
                "of paired-message turns dropped and the role of the "
                "first/last dropped message. Off in production."
            ),
        ),
    ] = False
```

### Field on `OrchestratorConfig`

Placed after the existing `compaction_padding` field (around line ~933):

```python
# Client-side message truncation baseline. Default disabled — enabling
# requires a matching TOML with no vLLM compaction, no trainer compaction,
# and no client-side block-aligned padding. See MarkovianThinkerConfig.
markovian_thinker: MarkovianThinkerConfig = MarkovianThinkerConfig()
```

### Cross-config validator in `RLConfig`

Location: `prime-rl/src/prime_rl/configs/rl.py`, as a new
`@model_validator(mode="after")` alongside existing validators like
`validate_compaction_mirrors_inference`.

```python
@model_validator(mode="after")
def validate_markovian_thinker(self):
    """Markovian Thinker assumes no KV compaction anywhere in the stack."""
    mt = self.orchestrator.markovian_thinker
    if not mt.enabled:
        return self

    # vLLM-side compaction off
    if self.inference is not None:
        vllm_extra = self.inference.vllm_extra or {}
        for k in ("compaction_window_size", "compaction_max_turns"):
            v = int(vllm_extra.get(k, 0) or 0)
            if v > 0:
                raise ValueError(
                    f"orchestrator.markovian_thinker.enabled=true is "
                    f"incompatible with inference.vllm_extra.{k}={v}. "
                    "Markovian Thinker performs client-side truncation; "
                    "vLLM-side compaction must be disabled."
                )

    # Trainer-side compaction off
    if self.trainer.compaction is not None and self.trainer.compaction.window_size > 0:
        raise ValueError(
            "orchestrator.markovian_thinker.enabled=true is incompatible "
            "with trainer.compaction.window_size > 0. Markovian samples "
            "have no CompactionEvents; the trainer must use the standard "
            "full-context forward."
        )

    # Client-side block padding redundant (and mutually exclusive with
    # our own prompt_token_ids stash)
    if self.orchestrator.compaction_padding.enabled:
        raise ValueError(
            "orchestrator.markovian_thinker.enabled=true is incompatible "
            "with orchestrator.compaction_padding.enabled=true. "
            "Markovian Thinker stashes its own prompt_token_ids on the "
            "response; block-aligned padding is only meaningful for "
            "block-level KV eviction."
        )

    # TITO extension property breaks under truncation
    if self.orchestrator.use_token_client:
        raise ValueError(
            "orchestrator.markovian_thinker.enabled=true requires "
            "use_token_client=false. TITO assumes extension property "
            "across turns, which client-side truncation breaks."
        )

    return self
```

## Hook point: single interceptor, two branches

The existing `_install_message_padding_interceptor()` in
`src/kv_eviction/env.py` monkey-patches
`openai.resources.chat.completions.completions.AsyncCompletions.create`.
Inside that interceptor, `patched_create` already reads
`kwargs["messages"]` and `kwargs["tools"]`.

**Decision: extend the existing `patched_create` with a Markovian
truncation preamble.** Do NOT install a second monkey-patch.

Rationale:
- Stacked monkey-patches depend on source-file ordering of `_install_*`
  calls, which is fragile. A single interceptor with two `if` branches
  makes the ordering explicit and impossible to break by reordering.
- The config validator forbids padding + markovian together, so in
  practice only one branch fires at a time. The branches are independent
  and compose safely even if that changed.

### Order inside the (extended) interceptor

```python
async def patched_create(self, *args, **kwargs):
    # --- Branch A: Markovian truncation (new) ---
    mcfg = _markovian_config
    if mcfg is not None and mcfg.enabled:
        messages = kwargs.get("messages")
        if messages is not None:
            truncated = truncate_messages_to_last_k_turns(
                messages, max_turns=mcfg.max_turns,
                log_fn=(lambda m: logger.info("[MARKOVIAN] %s", m))
                       if mcfg.log_truncated_messages else None,
            )
            kwargs["messages"] = truncated
            # Pre-tokenize the truncated messages so the trainer uses the
            # EXACT token stream vLLM will run on. See "Training path:
            # the prompt_token_ids divergence" below for why this matters.
            rendered = mcfg.tokenizer.apply_chat_template(
                truncated,
                tools=kwargs.get("tools"),
                add_generation_prompt=True,
                tokenize=False,
            )
            truncated_ids = mcfg.tokenizer.encode(
                rendered, add_special_tokens=False
            )
            response = await orig_create(self, *args, **kwargs)
            _stash_prompt_token_ids(response, truncated_ids)
            return response

    # --- Branch B: Block-aligned padding (existing, unchanged) ---
    pcfg = _padding_config
    if pcfg is None or not pcfg.enabled:
        return await orig_create(self, *args, **kwargs)
    # ... existing render_padded_prompt + extra_body injection ...
```

`_stash_prompt_token_ids(response, ids)` is a shared helper that sets
the attribute via `setattr` / `model_extra`, factored out of the
existing padding path (currently inlined at the bottom of
`patched_create`).

### Module-level runtime config

Mirror the `_padding_config` / `MessagePaddingConfig` pattern in
`src/kv_eviction/env.py`:

```python
@dataclass
class MarkovianThinkerRuntimeConfig:
    enabled: bool
    tokenizer: Any
    max_turns: int
    log_truncated_messages: bool

_markovian_config: MarkovianThinkerRuntimeConfig | None = None

def configure_markovian_thinker(
    *,
    enabled: bool,
    tokenizer: Any,
    max_turns: int,
    log_truncated_messages: bool = False,
) -> None:
    global _markovian_config
    _markovian_config = MarkovianThinkerRuntimeConfig(
        enabled=enabled,
        tokenizer=tokenizer,
        max_turns=max_turns,
        log_truncated_messages=log_truncated_messages,
    )
    if enabled:
        logger.info(
            "kv_eviction: Markovian Thinker ENABLED (max_turns=%d, log=%s)",
            max_turns, log_truncated_messages,
        )
```

### Observability counter: `n_messages_truncated`

In addition to the optional `log_truncated_messages` debug log, the
interceptor maintains a process-local counter of truncations:

```python
# Module-level in env.py
_markovian_stats = {"n_truncations": 0, "n_messages_dropped": 0}

def pop_markovian_stats() -> dict:
    """Drain-and-reset the Markovian counters. Called once per step."""
    global _markovian_stats
    out = dict(_markovian_stats)
    _markovian_stats = {"n_truncations": 0, "n_messages_dropped": 0}
    return out
```

Inside `patched_create` Branch A, after truncation:
```python
if len(truncated) < len(messages):
    _markovian_stats["n_truncations"] += 1
    _markovian_stats["n_messages_dropped"] += len(messages) - len(truncated)
```

The orchestrator's per-step metrics block calls `pop_markovian_stats()`
and logs the two counters to wandb under
`markovian/n_truncations_per_step` and
`markovian/n_messages_dropped_per_step`. This is the main
"did-it-fire-at-the-expected-rate" check in production. Absent from
wandb when Markovian is disabled (no keys emitted).

### Env-var autoconfigure for env-worker subprocesses

Env workers are `mp.spawn`'d by the orchestrator and do NOT inherit the
parent's `configure_markovian_thinker(...)` call. Mirror the existing
`_autoconfigure_padding_from_env()` pattern with three env vars:

```
KV_EVICTION_MARKOVIAN_ENABLED    = "1"            (presence = enabled)
KV_EVICTION_MARKOVIAN_MAX_TURNS  = "<int>"
KV_EVICTION_MARKOVIAN_MODEL      = "<tokenizer name>"
```

`_autoconfigure_markovian_from_env()` is called at module-load time
after `_autoconfigure_padding_from_env()`. It reads the vars, loads the
tokenizer via `AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)`,
and calls `configure_markovian_thinker(...)`.

No-op idempotent behavior when the vars are absent.

### Orchestrator startup wiring

In `prime-rl/src/prime_rl/orchestrator/orchestrator.py`, immediately
after the existing `compaction_padding` block (around line ~180):

```python
if config.markovian_thinker.enabled:
    from kv_eviction.env import configure_markovian_thinker
    configure_markovian_thinker(
        enabled=True,
        tokenizer=tokenizer,
        max_turns=config.markovian_thinker.max_turns,
        log_truncated_messages=config.markovian_thinker.log_truncated_messages,
    )
    # Propagate to env-worker subprocesses
    os.environ["KV_EVICTION_MARKOVIAN_ENABLED"] = "1"
    os.environ["KV_EVICTION_MARKOVIAN_MAX_TURNS"] = str(
        config.markovian_thinker.max_turns
    )
    os.environ["KV_EVICTION_MARKOVIAN_MODEL"] = config.model.name
    logger.info(
        f"Markovian Thinker enabled (max_turns="
        f"{config.markovian_thinker.max_turns})"
    )
```

## Pure truncation function

Location: `src/kv_eviction/truncation.py` (new file — a pure
message-list transform distinct from token-level padding).

```python
def truncate_messages_to_last_k_turns(
    messages: list[dict],
    *,
    max_turns: int,
    stride: int | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> list[dict]:
    """Apply vLLM's paired-message sawtooth turn schedule."""
```

### Algorithm (matches "Desired semantics / Truncation policy" above)

```python
def truncate_messages_to_last_k_turns(
    messages,
    *,
    max_turns,
    stride=None,
    log_fn=None,
):
    if not messages or max_turns < 1:
        return messages

    protected = messages[:1]
    body = messages[1:]
    complete = len(body) - len(body) % 2
    turns = [body[i : i + 2] for i in range(0, complete, 2)]
    tail = body[complete:]

    eviction_stride = 1 if stride is None else stride
    if len(turns) < max_turns:
        return messages
    events = ((len(turns) - max_turns) // eviction_stride) + 1
    keep = len(turns) - events * eviction_stride
    kept = turns[len(turns) - keep :] if keep else []
    return protected + flatten(kept) + tail
```

### Invariants (must all hold; each is covered by a unit test)

1. **Identity no-op**: if no truncation is needed, `out is messages`.
2. **No mutation**: `messages` is untouched; a new list is constructed
   only when truncation fires.
3. **Protected prefix**: `out[0] == messages[0]`.
4. **In-flight tail**: `out[-len(tail):] == messages[-len(tail):]` when
   `tail` is non-empty.
5. **Boundary parity**: each turn is exactly two post-prefix messages.
6. **Monotonic ordering**: retained messages appear in input order.
7. **Idempotence**: `truncate(truncate(M, K), K) == truncate(M, K)`.
8. **Length**: `len(out) <= len(messages)`.

## Training path: why nothing changes, except for the `prompt_token_ids` divergence

Once truncation is wired correctly, the trainer requires **no changes**:

1. **No `CompactionEvent`s**: vLLM does no eviction, so
   `TrajectoryStep.extras["compaction_events"]` is `None` on every step.
   `interleave_rollout` passes `compaction_events=None` to every
   `TrainingSample`.
2. **No `segmented_forward`**: the trainer dispatch
   `use_segmented = config.compaction.window_size > 0` is False
   (enforced by `validate_markovian_thinker`). Every sample takes the
   standard full-context forward path.
3. **`interleave_rollout` splits per turn**: each time truncation drops
   a turn, step N's tokenized prompt stops being a prefix of step N-1's
   `(prompt + completion)`. The extension check at
   `trajectories.py:741` fails → a new `TrainingSample` starts. Each
   truncated window thus becomes its own independent
   `(prompt_ids, completion_ids)` pair. This is exactly correct: the
   trainer sees what inference saw.

### The `prompt_token_ids` divergence (critical correctness issue)

Without mitigation, there is a real train/inference divergence:

- verifiers tokenizes `TrajectoryStep.tokens["prompt_ids"]` from the
  messages passed to `get_model_response`, which is the
  **full pre-truncation messages**.
- The Markovian interceptor runs *inside* `AsyncCompletions.create`,
  downstream of verifiers, and truncates the messages just before vLLM
  sees them. vLLM therefore runs on **truncated messages**, and
  `completion_logprobs` are computed against that shorter context.
- If the trainer used verifiers' `prompt_ids`, it would forward against
  the **full** prompt + truncated-context completion. Logprobs would
  diverge from inference → KL mismatch → training goes off the rails.

**Fix** (already wired into the interceptor sketch above): after
truncation, re-tokenize the truncated messages via
`tokenizer.apply_chat_template(..., tokenize=False)` +
`tokenizer.encode(..., add_special_tokens=False)` and **stash the result
as `prompt_token_ids` on the response object**. The existing plumbing
in `env.py` (Patch #1 `from_native_response` and Patch #2
`add_model_response`, both already installed) forwards
`prompt_token_ids` from the response to the trajectory step's
`extras`. `interleave_rollout.prepare_step_tokens` at
`trajectories.py:299-302` already overrides `prompt_ids` with
`extras["prompt_token_ids"]` when present. **No changes needed in
`trajectories.py` or the trainer** — we just populate the same extras
channel the padding path already uses.

Determinism: the tokenizer used by the interceptor is the same one
vLLM's server uses (same `model.name`), and we render with the same
chat template. Tokens are byte-identical to what vLLM produces
internally, matching the behavior the padding path already relies on.

### `interleave_rollout` extension-break arithmetic

For an episode with T complete turns and `max_turns=K`:
- T <= K: no truncation fires. Extension property holds across all
  steps → 1 merged `TrainingSample`.
- T > K: truncation fires at step K+1. After that, every step drops
  an oldest turn → extension breaks → new sample per step. Total
  samples ~= T - K + 1.

This is identical in shape to what vLLM-compaction runs produce when
eviction fires mid-episode, and the existing orchestrator / advantage
pipeline already handles it.

### Sanity assert on post-truncation samples

When Markovian is enabled and `interleave_rollout` starts a new
`TrainingSample` (i.e., extension-break fired), add a cheap assert
that the new sample's `prompt_ids` begins with the tokenization of
`<|im_start|>system`. This catches silent bugs where the truncation
function accidentally drops the system prefix (e.g., a future edit
to `truncate_messages_to_last_k_turns` breaks the `sys_end`
computation and the model stops seeing its instructions).

Implementation: guarded by a module-level flag set only when
`KV_EVICTION_MARKOVIAN_ENABLED=1` is in the env (zero cost
otherwise). The system-prefix token ids are computed once at trainer
startup from the tokenizer (no per-step tokenization). ~5 lines in
`prime-rl/src/prime_rl/orchestrator/trajectories.py` near the
extension-break branch.

## Files touched

| File                                                                         | Change                                                                                                        |
|------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| `src/kv_eviction/truncation.py`                                              | **NEW**. Pure `truncate_messages_to_last_k_turns`.                                                            |
| `src/kv_eviction/env.py`                                                     | Add `MarkovianThinkerRuntimeConfig`, `_markovian_config`, `configure_markovian_thinker`, `_autoconfigure_markovian_from_env`. Extend `patched_create` with Branch A preamble (truncate + stash `prompt_token_ids`). Factor `_stash_prompt_token_ids` helper. |
| `prime-rl/src/prime_rl/configs/orchestrator.py`                              | Add `MarkovianThinkerConfig` + field on `OrchestratorConfig`.                                                  |
| `prime-rl/src/prime_rl/configs/rl.py`                                        | Add `validate_markovian_thinker` cross-config validator.                                                       |
| `prime-rl/src/prime_rl/orchestrator/orchestrator.py`                         | Startup block (mirror the `compaction_padding` block): call `configure_markovian_thinker(...)`, set env vars. Per-step: drain `pop_markovian_stats()` and log to wandb under `markovian/*` keys when enabled. |
| `prime-rl/src/prime_rl/orchestrator/trajectories.py`                         | Add cheap system-prefix assert on extension-break samples when `KV_EVICTION_MARKOVIAN_ENABLED=1`. ~5 lines.    |
| `experiments/debug_balrog/rl_markovian.toml`                                 | **NEW**. Clone of `rl_no_eviction.toml` with `[orchestrator.markovian_thinker]` section and `compaction_padding.enabled = false`. |
| `experiments/compaction_textworld/rl_markovian.toml`                         | **NEW**. TextWorld variant (simple chat, no tool calls).                                                       |
| `tests/test_truncation.py`                                                   | **NEW**. Unit tests for the pure function.                                                                     |
| `tests/test_markovian_interceptor.py`                                        | **NEW**. Interceptor wiring + passthrough + stashing tests.                                                    |

Explicitly **NOT** touched:
- `src/kv_eviction/padding.py` — no changes.
- `src/kv_eviction/segmented_forward.py` — not used under Markovian.
- `prime-rl/src/prime_rl/trainer/` — no changes.
- `vllm/` — no changes. vLLM sees normal chat completion requests.

The single change to `trajectories.py` is an assert only; the existing
`prompt_token_ids` override in `prepare_step_tokens` already routes the
truncated ids through without modification.

## Experiment config: `rl_markovian.toml`

Cloned from `experiments/debug_balrog/rl_no_eviction.toml` with the
following surgical diffs:

```toml
# ... identical header (max_steps, seq_len, deployment, [model], [trainer.*]) ...

[wandb]
project = "kv-eviction"
name = "debug-balrog-markovian"

[orchestrator]
batch_size = 64
rollouts_per_example = 8
oversampling_factor = 1.0
# Required by validate_markovian_thinker.
use_token_client = false

# Padding is redundant under Markovian (markovian stashes its own
# prompt_token_ids); validator forbids enabling it.
[orchestrator.compaction_padding]
enabled = false

# New section.
[orchestrator.markovian_thinker]
enabled = true
max_turns = 6
log_truncated_messages = false

[orchestrator.train.sampling]
# Main point of Markovian: free up context for deeper reasoning.
max_completion_tokens = 1024
temperature = 1.0

[[orchestrator.train.env]]
id = "balrog-bench"
args = { environments = ["babyai"], max_text_history = 16, max_turns = 30 }

[inference]
seed = 0
enable_prefix_caching = false
gpu_memory_utilization = 0.85

[inference.server]
host = "0.0.0.0"
port = 8000

[inference.model]
name = "Qwen/Qwen3-4B-Instruct-2507"
max_model_len = 16384
enforce_eager = true

[inference.parallel]
dp = 4
tp = 1

# No compaction_* fields in [inference.vllm_extra]; validator would
# reject them.
[inference.vllm_extra]
async_scheduling = false
```

The TextWorld variant
(`experiments/compaction_textworld/rl_markovian.toml`) follows the same
pattern, cloned from `rl_no_eviction.toml`-equivalent in that
directory.

## Testing plan

### Unit tests: `tests/test_truncation.py`

Test the pure `truncate_messages_to_last_k_turns` in isolation.

1. First-message protection and paired post-prefix boundaries.
2. Exact `>= max_turns` trigger behavior.
3. Sawtooth live counts across repeated `stride` eviction events.
4. Default `stride=None` parity with vLLM's stride of one.
5. Unmatched final-message tail preservation.
6. Tool-call chains count as multiple paired-message turns rather than one
   terminal-assistant group.
7. No-system prompts still protect their first message, matching vLLM.
8. Identity below threshold, no mutation, monotonic order, idempotence, and
   output-length invariants.
9. Invalid stride values fail loudly.
10. Anchor mode uses the same boundary counter and trigger before applying its
    intentionally different selection policy.

### Interceptor tests: `tests/test_markovian_interceptor.py`

22. `test_interceptor_passthrough_when_disabled` — with
    `configure_markovian_thinker(enabled=False, ...)`, messages flow
    unchanged and no `prompt_token_ids` attribute is set on the
    returned response.
23. `test_interceptor_truncates_when_enabled` — mock
    `orig_create`; verify `kwargs["messages"]` was shortened and that
    the mocked create received the shortened list.
24. `test_interceptor_stashes_prompt_token_ids` — verify the
    response object has `prompt_token_ids` set to the tokenization of
    the truncated messages.
25. `test_interceptor_idempotent_install` — calling
    `_install_message_padding_interceptor()` twice does not stack
    wrappers.
26. `test_interceptor_updates_stats_counters` — after N truncations,
    `pop_markovian_stats()` returns `n_truncations=N` and a
    drained state on the second call.
27. `test_interceptor_no_stats_when_disabled` — when
    `enabled=False`, counters remain at zero.
28. `test_trainer_system_prefix_assert_fires_on_missing_prefix` —
    synthesize a `TrainingSample` whose `prompt_ids` do not start with
    the system-prefix tokens; assert fires with
    `KV_EVICTION_MARKOVIAN_ENABLED=1` set and is a no-op otherwise.

### Config validator tests (`tests/test_rl_config.py` or equivalent)

29. `test_markovian_off_does_not_validate` — default config with
    `enabled=false` loads and validates (no false positives).
30. `test_markovian_on_plus_vllm_compaction_rejected` — config with
    `markovian_thinker.enabled=true` and
    `inference.vllm_extra.compaction_window_size=1024` raises
    `ValidationError`.
31. `test_markovian_on_plus_trainer_compaction_rejected` — config with
    `markovian_thinker.enabled=true` and
    `trainer.compaction.window_size=512` raises.
32. `test_markovian_on_plus_padding_rejected` — config with
    `markovian_thinker.enabled=true` and
    `compaction_padding.enabled=true` raises.
33. `test_markovian_on_plus_tito_rejected` — config with
    `markovian_thinker.enabled=true` and `use_token_client=true`
    raises.
34. `test_markovian_on_valid_config_accepted` — all constraints
    satisfied → config loads.

### Integration smoke tests (manual)

35. **BabyAI smoke** (`rl_markovian.toml`, 5 steps, `max_turns=6`):
    - vLLM logs show no `compaction_events`.
    - Trainer dispatch log shows standard forward (no
      `segmented_forward`).
    - `interleave_rollout` produces approximately T - max_turns + 1
      samples per rollout (sanity-check via debug log).
    - Loss finite and non-NaN across all steps.
    - Mismatch KL at kernel floor (~1e-3), matching the full-context
      smoke.
    - wandb shows nonzero `markovian/n_truncations_per_step` after
      step ~`max_turns`, zero before.
36. **TextWorld smoke** — same checks on the simple-chat env.
37. **Parity test with flag off**:
    Run `rl_markovian.toml` with
    `markovian_thinker.enabled=false` AND
    `compaction_padding.enabled=false`. Compare against an identical
    `rl_no_eviction.toml` run: rollout tokens, loss, and wandb
    metrics at step 0 should be bit-identical. This is the
    regression test for "default behavior unchanged".
38. **Parity test with high `max_turns`**:
    Run with `max_turns=100` (effectively no truncation on 30-turn
    episodes). Verify `len(TrainingSamples) == num_rollouts` (single
    sample per rollout) and token-level equivalence to the flag-off
    run.
39. **Output quality inspection**: Run a 10-step rollout on BabyAI
    with `max_turns=6` and compare to the block-FIFO compaction
    smoke (`experiments/debug_balrog/compaction_test.ipynb`). Verify
    no mid-word fragmentation artifacts (`"I new observation"`,
    `"TheThe"`, etc.) — Markovian truncation is lossy but clean, so
    these should be absent.

## Risk / open questions

1. **Prefix-cache efficiency**. Each truncation shifts the prefix, so
   vLLM's automatic prefix caching misses on nearly every turn after
   the first `max_turns`. In experiments this may look slower than
   either full-context or vLLM compaction on the same box. Mitigation:
   set `inference.enable_prefix_caching = false` in the Markovian
   config to avoid wasted bookkeeping. Document the tradeoff in the
   config comment.

2. **Tokenization cost in the interceptor**. Each
   `AsyncCompletions.create` now runs `apply_chat_template + encode`
   on the truncated messages. For typical `max_turns=6` BabyAI
   prompts (~800 tokens), this adds <1ms per turn on CPU. Acceptable.
   If it becomes a hotspot, switch to computing the truncated ids by
   re-tokenizing only the dropped-vs-kept boundary instead of the
   whole prompt.

3. **Non-ChatML templates**. The system-prefix rule ("leading
   non-user messages") and the terminal-assistant grouping rule are
   chat-template-agnostic at the role level. But the attribute name
   `tool_calls` is OpenAI's (matches the incoming OAI API dict shape,
   which is what verifiers hands us). If a custom env produced a
   different tool-signaling field, the grouping would silently
   consider those assistants as terminal. Mitigation: documented
   scope is OAI-style `tool_calls`. For non-OAI envs, override via a
   small plugin later.

4. **`max_text_history` interaction**. BabyAI's `balrog-bench` already
   trims observations via `max_text_history=16`. Markovian
   `max_turns=6` is tighter than that and wins. No conflict; worth
   one line of documentation in the experiment README so users don't
   double-configure.

5. **Advantage attribution across per-step samples**. Because
   `interleave_rollout` produces one sample per post-truncation step,
   the final episode reward is attributed to each sample via the
   existing `compute_advantages` path — the same behavior that
   vLLM-compaction already exhibits when eviction fires. Not a new
   concern, but worth verifying in a smoke run that advantage
   distributions look reasonable (matching the vLLM-compaction
   baseline).

6. **`max_turns` per-env vs global** — **locked: global for v1.**
   `max_turns` lives on `OrchestratorConfig` only. A per-env override
   (optional `max_turns` field on the env spec) is a ~5-line addition
   and will be added *only* when a second experiment empirically
   needs a different window than the first. Do not pre-plumb it.

7. **Should we also support a token budget as a secondary cap?**
   ("after group truncation, if the rendered prompt still exceeds N
   tokens, drop one more group"). Not needed for the initial
   baseline — `max_turns` with a conservative value makes token
   overflow very unlikely in practice, and the vLLM server already
   rejects over-long prompts cleanly. Can be added later as
   `max_prompt_tokens: int | None = None`.

8. **Logging evicted message bodies for debugging**. The config
   includes `log_truncated_messages: bool`, which logs one
   summary line per truncation (group count + role/content-prefix of
   first+last dropped message). If a deeper inspection is needed
   (e.g., decode each dropped message's full content, matching
   `log_evicted_text` in `CompactionPaddingConfig`), extend later.

## Event flow (for reference)

```
Orchestrator
  │
  │  build messages (full history up to current turn)
  ▼
verifiers rollout loop
  │  (tokenizes full messages into TrajectoryStep.tokens["prompt_ids"]
  │   but this is later overridden — see below)
  │
  ▼
openai-python client.chat.completions.create(messages=...)
  │
  ▼
AsyncCompletions.create  (monkey-patched interceptor in env.py)
  │
  │  Branch A (Markovian enabled):
  │    1. truncate_messages_to_last_k_turns(messages, max_turns=K)
  │    2. kwargs["messages"] = truncated
  │    3. apply_chat_template + encode -> truncated_prompt_ids
  │    4. orig_create(...) -> vLLM runs on truncated prompt
  │    5. stash truncated_prompt_ids on response.prompt_token_ids
  │
  ▼
vLLM server
  │  (no compaction, normal full-context completion)
  ▼
ChatCompletion response (with extra prompt_token_ids attr)
  │
  ▼
verifiers client.from_native_response  (patched: forwards prompt_token_ids)
  ▼
TrajectoryStep.extras["prompt_token_ids"] = truncated_prompt_ids
  │
  ▼
interleave_rollout.prepare_step_tokens
  │  reads extras["prompt_token_ids"] -> overrides step.tokens["prompt_ids"]
  │  with the TRUNCATED ids (not verifiers' full re-tokenization)
  ▼
TrainingSample
  │  prompt_ids      = truncated_prompt_ids       (what vLLM saw)
  │  completion_ids  = generated against truncated context
  │  compaction_events = None
  ▼
Trainer (standard full-context forward; no segmented_forward)
```
