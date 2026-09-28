# Plan: Block-Aligned Message Padding via prime-rl Patch (Option A)

## Status: Steps 1–4 done, Steps 5–7 pending

**Implementation progress (as of 2026-04-13):**

| Step | Scope | Status |
|---|---|---|
| 1 | vLLM patch — `prompt_token_ids` field + `render_chat` bypass + tests | ✅ done |
| 2 | `src/kv_eviction/padding.py` + unit tests | ✅ done |
| 3 | Orchestrator-side monkey-patch wrapping `AsyncOpenAI.chat.completions.create` (Changes 6 + 7) + tests | ✅ done |
| 4 | prime-rl config + trajectory plumbing (Changes 8, 9, 10) | ✅ done |
| 5 | Trainer loss mask + segmented_forward defensive assertion (Changes 11, 12, 13) | ⏸ deferred (user constraint: "not doing the trainer implementations yet") |
| 6 | Backward-compat gate (padding off, end-to-end smoke parity) | ⏸ pending |
| 7 | Deprecate `plans/block_aligned_message_padding.md` | ⏸ pending |

**What landed (concrete):**

- **vLLM fork** (`vllm/` submodule, branch `eviction-turn-based`):
  - `vllm/entrypoints/openai/chat_completion/protocol.py` — `prompt_token_ids: list[int] | None` added inside the `chat-completion-extra-params` doc block on `ChatCompletionRequest`.
  - `vllm/entrypoints/serve/render/serving.py::render_chat` — bypass branch at top: when `request.prompt_token_ids is not None`, returns `(list(messages), [tokens_input(prompt_token_ids, cache_salt=...)])` and skips `preprocess_chat`. Empty-list → `ErrorResponse`.
  - `tests/entrypoints/openai/chat_completion/test_prompt_token_ids_bypass.py` — 3 unit tests (verbatim passthrough, empty-list error, `cache_salt` threading) using mocked `OpenAIServingRender` + `AsyncMock` trip-wire on `preprocess_chat`. Driven via `asyncio.run()` from sync test bodies (no pytest-asyncio dependency).

- **kv-eviction integration** (parent repo):
  - `src/kv_eviction/padding.py` (new) — `render_padded_prompt(...)` + `resolve_filler_token_id(tokenizer, override)` + `resolve_im_end_token_id(tokenizer)`. Pure functions, no side effects.
  - `tests/test_padding.py` (new) — 7 unit tests covering single/multi-turn boundaries, zero-pad-when-aligned, generation-prompt suffix not padded, filler resolution chain.
  - `src/kv_eviction/env.py` — added `MessagePaddingConfig` dataclass, module-level `_padding_config`, `configure_message_padding(...)`, `_install_message_padding_interceptor()` (class-level monkey-patch on `openai.resources.chat.completions.completions.AsyncCompletions.create`), plus `attach_prompt_token_ids_from_response(step, response)` and `padded_ids_from_step_extras(extras)` read-side helpers. Existing Patches #1 (`patched_from_native`) and #2 (`patched_add_model_response`) extended to forward `prompt_token_ids` alongside `compaction_events`.
  - `tests/test_padding_interceptor.py` (new) — 3 tests: passthrough when disabled, injection of `extra_body["prompt_token_ids"]` when enabled, merge with preexisting `extra_body`.

- **prime-rl** (`prime-rl/` submodule, branch `kv-eviction`):
  - `src/prime_rl/configs/orchestrator.py` — `CompactionPaddingConfig` (`enabled`, `block_size=16`, `filler_token_id`, `im_end_token_id`) on `OrchestratorConfig`.
  - `src/prime_rl/configs/rl.py` — `validate_compaction_padding` `@model_validator(mode="after")` asserts `block_size` matches across trainer/orchestrator/inference and requires `trainer.compaction.window_size > 0`.
  - `src/prime_rl/orchestrator/orchestrator.py` — startup block resolves filler/im_end ids from the tokenizer and calls `configure_message_padding(...)` when enabled.
  - `src/prime_rl/orchestrator/trajectories.py::prepare_step_tokens` — when `step["extras"]["prompt_token_ids"]` is present, overrides `prompt_ids` with the padded ids and rebuilds `prompt_mask = [False] * len(prompt_ids)`.

- **Smoke verified:** `OrchestratorConfig()` parses cleanly with both default (disabled) and `compaction_padding.enabled=True` configs.

**Remaining work — see "Order of operations" below for the original step list.** Steps 5–7 still need to land. Step 5 is gated on explicit user permission to touch trainer code. Step 7 is docs-only and can proceed independently.

---

## Status: Draft / Follow-up to `block_aligned_message_padding.md`

**Starting commit:** `2cd0bac` on branch `eviction-turn-based`
(parent repo) with `vllm/` submodule at `5bfe16642` on branch
`eviction-turn-based` ("Turn-based compaction: evict whole
user+assistant turns"). Implementation branches from here.

Supersedes the "Options A/B/C transport" discussion in the prior
draft — **Option A is committed**. See "Decision record" below for
why B and C are rejected.

## Motivation

The prototype in `experiments/debug_balrog/compaction_test.ipynb`
demonstrates that padding each chat message so `<|im_end|>` lands on
a block boundary produces exact-edge eviction under turn mode (no
orphan fragments, no mid-message cuts). The prototype runs in-process
via `vllm.LLM(...).generate(prompts=[{"prompt_token_ids": …}])`, which
sidesteps the OpenAI server entirely.

Prime-rl's rollout path, however, talks to vLLM over HTTP via
`verifiers.clients.openai_chat_completions_client.OpenAIChatCompletionsClient`
which sends `messages=[…]` and lets the server apply the chat
template. This plan ports the notebook's padding recipe to that
client-server path with minimal surgery:

1. **Tiny vLLM-fork patch** (~30 lines, 2 files) adds a
   `prompt_token_ids` field to `ChatCompletionRequest`. When set,
   vLLM skips `apply_chat_template` and feeds the provided tokens
   directly to the engine.
2. **kv-eviction monkey-patch** in `src/kv_eviction/env.py` wraps
   the verifiers client: tokenize + pad messages client-side, send
   as `extra_body={"prompt_token_ids": padded}` alongside the
   regular `messages=…` (the latter kept for tool-call metadata
   downstream).
3. **prime-rl config + trainer** gain a `message_padding_aware`
   flag that masks filler tokens out of the loss.

This eliminates Bug 2 (`prompt_aligned_len` mismatch) by construction:
with padded prompts, the trainer's block-alignment and vLLM's
block-alignment produce the same boundary because the prompt length
is a multiple of `block_size`.

## Invariants

1. **Inference and training use identical token streams.** The
   trainer's `TrainingSample.prompt_token_ids` is set from the
   padded ids vLLM actually ran on, not re-rendered from `messages`.
2. **Padding tokens contribute zero gradient.** The trainer-side loss
   mask is ANDed with `(input_ids != filler_id)` at positions inside
   the prompt that came from padding. (Positions in the completion
   are never padded; filler collisions there are genuine model
   output and must not be masked.)
3. **Padding never biases generation.** Filler tokens are inserted
   only *before* `<|im_end|>` tokens that are already committed (in
   the prompt). The sampler is never constrained toward fillers.
4. **Deterministic padding rule.** Given
   `(messages, tools, block_size, filler_id)`, the padded id list is
   unique. Enables prefix-cache hits across requests with identical
   prompts.

## Flow diagram

```
                   rollout worker (prime-rl)
                   ┌──────────────────────────────────────────┐
                   │ verifiers.MultiTurnEnv.get_prompt_messages│
                   │   returns messages=[sys,u,a,u,a,...,u]    │
                   └─────────────────────┬────────────────────┘
                                         │ messages, tools
                                         ▼
                   ┌──────────────────────────────────────────┐
                   │ OpenAIChatCompletionsClient.get_response  │
                   │   (monkey-patched in src/kv_eviction/env) │
                   ├──────────────────────────────────────────┤
                   │ if cfg.pad_messages_to_block:             │
                   │   raw  = tokenizer.apply_chat_template(   │
                   │           messages, tools, tokenize=True, │
                   │           add_generation_prompt=True)     │
                   │   pad  = _render_padded(raw, im_end_id,   │
                   │           block_size, filler_id)          │
                   │   body = {"prompt_token_ids": pad}        │
                   │ else:                                     │
                   │   body = {}                               │
                   └─────────────────────┬────────────────────┘
                                         │ POST /v1/chat/completions
                                         │ { messages, tools,         }
                                         │ { extra_body: {            }
                                         │ {   prompt_token_ids: pad  }
                                         │ { }                         }
                                         ▼
  vllm/entrypoints/serve/render/serving.py::render_chat
   ┌───────────────────────────────────────────────────────┐
   │ if request.prompt_token_ids is not None:              │
   │     token_ids = request.prompt_token_ids              │
   │     conversation = request.messages  # metadata-only  │
   │     engine_input = TokensPrompt(                      │
   │         prompt_token_ids=token_ids, prompt=None)      │
   │     return conversation, [engine_input]               │
   │ else:                                                 │
   │     # existing apply_chat_template path               │
   └─────────────────────┬─────────────────────────────────┘
                         │ engine_input (TokensPrompt)
                         ▼
              vllm engine runs inference on padded tokens
                         │
                         │ ChatCompletionResponse with
                         │   prompt_token_ids=padded,
                         │   compaction_events=[…]
                         ▼
                   ┌──────────────────────────────────────────┐
                   │ OpenAIChatCompletionsClient.from_native_ │
                   │   response  (monkey-patched, existing)    │
                   │   + new: forward padded ids to step.extras│
                   └─────────────────────┬────────────────────┘
                                         ▼
                     TrajectoryStep.extras
                       ├── compaction_events
                       └── prompt_token_ids   (new)
                                         │
                                         ▼
         orchestrator/trajectories.py::interleave_rollout
           TrainingSample.prompt_token_ids = extras[prompt_token_ids]
                                         │
                                         ▼
                   ┌──────────────────────────────────────────┐
                   │ trainer/rl/data.py::_micro_batch_to_tensor│
                   │   loss_mask &= (input_ids != filler_id)   │
                   │              (over prompt positions only) │
                   └─────────────────────┬────────────────────┘
                                         ▼
                 segmented_forward — unchanged, same padded ids
                 as vLLM ran, so prompt_aligned_len matches exactly
```

## Changes — vLLM fork

### Change 1: accept `prompt_token_ids` on `ChatCompletionRequest`

**File:** `vllm/entrypoints/openai/chat_completion/protocol.py`

Insert near the other vLLM-specific fields of `ChatCompletionRequest`
(class starts at line 171; put the new field alongside
`kv_transfer_params` / `compaction_events` extensions):

```python
# vLLM extension — kv-eviction message-padding patch.
# When set, the server SKIPS apply_chat_template and uses these
# token ids verbatim. `messages` is still consulted for tool-call
# metadata (reasoning parser, tool_choice validation) but is not
# re-rendered. Intended for clients that need byte-exact control
# over the prompt token stream (e.g. block-aligned padding for
# turn-based KV compaction).
prompt_token_ids: list[int] | None = None
```

Pydantic `OpenAIBaseModel` already allows extras, but making this
an explicit typed field gives us validation + an OpenAPI entry.

### Change 2: bypass branch in `render_chat`

**File:** `vllm/entrypoints/serve/render/serving.py`

Current `render_chat` starts at line 177. Its contract:
`(ChatCompletionRequest) -> (conversation, [engine_input]) | ErrorResponse`.

Add a bypass at the top of the method — before any tokenizer /
tool-parser work that assumes we're going to re-render. Sketch:

```python
async def render_chat(
    self,
    request: ChatCompletionRequest,
) -> tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse:
    # Fast path: caller supplied pre-tokenized prompt. Skip chat
    # template application, but still surface `messages` as the
    # `conversation` so downstream tool-call parsing / reasoning
    # handlers have their metadata.
    if request.prompt_token_ids is not None:
        if not request.prompt_token_ids:
            return self.create_error_response(
                "prompt_token_ids is set but empty"
            )
        # Use request.messages directly as the ConversationMessage
        # list. The openai message typed-dict is structurally a
        # superset of ConversationMessage for our uses.
        conversation = list(request.messages)  # type: ignore[arg-type]
        engine_input = TokensPrompt(
            prompt_token_ids=list(request.prompt_token_ids),
        )
        return conversation, [engine_input]

    # --- existing path below unchanged ---
    tokenizer = self.renderer.tokenizer
    ...
```

**Add import** at top of file:
```python
from vllm.inputs import TokensPrompt
```
(it may already be imported transitively — check before adding.)

### Change 3: downstream input-length / truncation checks

**File:** `vllm/entrypoints/serve/render/serving.py` (around line
145-151 in `render_chat_request`).

`extract_prompt_components` and `extract_prompt_len` already handle
both `TokensPrompt` and `TextPrompt` shapes — no change expected.
Verify by running the existing pytest for chat completions with
token-only prompts (`tests/entrypoints/openai/...`).

### Change 4: test

**File:** `vllm/tests/entrypoints/openai/test_chat_prompt_token_ids.py`
(new)

One test case: POST `/v1/chat/completions` with `messages=[{role:
system, content: "s"}]` plus `prompt_token_ids=[1,2,3,4,5]`. Assert
the response's echoed `prompt_token_ids` is `[1,2,3,4,5]` (not the
result of applying the chat template to `messages`). No model run
needed — can mock at the `render_chat` boundary.

### Total vLLM diff: ~40 lines across 3 files (2 source + 1 test).

## Changes — kv-eviction integration layer

### Change 5: `src/kv_eviction/padding.py` (new)

Ports `_render_padded` + `_filler_token_id` from
`experiments/debug_balrog/compaction_test.ipynb` into a reusable
module. Pure functions, no side effects. API:

```python
def render_padded_prompt(
    tokenizer,
    messages: list[dict],
    tools: list[dict] | None,
    block_size: int,
    filler_token_id: int,
    im_end_token_id: int,
    *,
    add_generation_prompt: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    """
    Returns (raw_ids, padded_ids, per_im_end_pads).
    - raw_ids:     tokenizer.apply_chat_template output
    - padded_ids:  filler tokens inserted so each <|im_end|> lands
                   at the LAST slot of a block_size-sized block
    - per_im_end_pads: pad count inserted before each <|im_end|>,
                   in order of appearance. For debugging / trainer
                   mask reconstruction.
    """

def resolve_filler_token_id(tokenizer, override: int | None) -> int:
    """override -> tokenizer.pad_token_id -> encode(' ')[-1] -> endoftext."""
```

### Change 6: `src/kv_eviction/env.py` — third monkey-patch

Mirrors the existing patches at `env.py:249-309`. Wraps
`OpenAIChatCompletionsClient.get_response` so that when the
orchestrator has enabled padding, the wrapper:

1. Builds `padded_ids` via `render_padded_prompt`.
2. Calls the original `get_response` with an extra kwarg
   `extra_body={"prompt_token_ids": padded_ids}` merged into the
   underlying `client.chat.completions.create` call.
3. Stashes `padded_ids` on the returned OpenAI `ChatCompletion`
   object as an attribute (pydantic `extra="allow"` supports this),
   so patch #1 (`from_native_response`) can read it.
4. Extends patch #1 (`patched_from_native`): alongside
   `compaction_events`, also forward `prompt_token_ids` onto the
   verifiers `Response` extras.
5. Extends patch #2 (`patched_add_model_response`): alongside
   `attach_compaction_events_from_response`, also attach
   `prompt_token_ids` to the trajectory step's extras.

Sketch of the new third patch:

```python
base_client_cls = _vf_client.OpenAIChatCompletionsClient
orig_get_response = base_client_cls.get_response
if not getattr(orig_get_response, "__kv_eviction_padding_patched__", False):
    async def patched_get_response(self, *args, **kwargs):
        cfg = _padding_config()  # module-level, set by orchestrator at startup
        if cfg is None or not cfg.enabled:
            return await orig_get_response(self, *args, **kwargs)

        # Verifiers calls get_response(prompt=messages, tools=tools, ...).
        messages = kwargs.get("prompt") or (args[0] if args else None)
        tools = kwargs.get("tools")
        padded = render_padded_prompt(
            tokenizer=cfg.tokenizer,
            messages=messages,
            tools=tools,
            block_size=cfg.block_size,
            filler_token_id=cfg.filler_token_id,
            im_end_token_id=cfg.im_end_token_id,
        )[1]

        # Thread into the underlying openai client.chat.completions.create.
        # Verifiers' get_response forwards **kwargs to create(), so we can
        # piggyback extra_body through there.
        extra_body = kwargs.pop("extra_body", {}) or {}
        extra_body["prompt_token_ids"] = padded
        kwargs["extra_body"] = extra_body

        response = await orig_get_response(self, *args, **kwargs)
        # Stash padded ids for the from_native patch to forward.
        try:
            setattr(response, "prompt_token_ids", padded)
        except Exception:
            if hasattr(response, "model_extra"):
                if response.model_extra is None:
                    response.__pydantic_extra__ = {}
                response.model_extra["prompt_token_ids"] = padded
        return response

    patched_get_response.__kv_eviction_padding_patched__ = True
    base_client_cls.get_response = patched_get_response
```

**Hook point (Q1 resolved):** Wrap the underlying
`AsyncOpenAI.chat.completions.create` method directly rather than
verifiers' `get_response`. This is one level deeper but bypasses
verifiers' `normalize_sampling_args` filtering entirely and means
we don't need to patch verifiers at all.

Concretely: at orchestrator startup, after the verifiers env is
built, walk each registered `OpenAIChatCompletionsClient`'s
`.client` attribute (an `AsyncOpenAI`) and wrap
`client.chat.completions.create` in-place with a closure that:
1. Reads `messages` and `tools` from the kwargs.
2. Renders padded ids via `render_padded_prompt`.
3. Merges `{"prompt_token_ids": padded}` into `extra_body` (kwarg
   on `AsyncOpenAI`'s `create`, officially supported).
4. Calls the original `create`.
5. Attaches `padded` to the response object before returning.

This keeps the wrapper self-contained on the AsyncOpenAI client and
leaves all existing verifiers patches (Patches #1 and #2 in
`env.py`) untouched.

### Change 7: `src/kv_eviction/env.py` — padding-config registration

Add a small module-level function the orchestrator calls once at
startup:

```python
def configure_message_padding(
    *,
    enabled: bool,
    tokenizer,
    block_size: int,
    filler_token_id: int,
    im_end_token_id: int,
) -> None:
    """Called once by the orchestrator before rollouts start. Stores
    the config in a module-level variable read by the patched
    get_response wrapper. No-op when enabled=False."""
```

This keeps the orchestrator → monkey-patch coupling explicit (not
env-var based).

## Changes — prime-rl

### Change 8: `prime-rl/src/prime_rl/orchestrator/config.py`

Add to the orchestrator config schema:

```python
class CompactionPaddingConfig(BaseModel):
    enabled: bool = False
    block_size: int = 16           # must match inference.vllm_extra.block_size
    filler_token_id: int | None = None  # None = auto via tokenizer.pad_token_id
    im_end_token_id: int | None = None  # None = auto from tokenizer

class OrchestratorConfig(BaseModel):
    ...
    compaction_padding: CompactionPaddingConfig = CompactionPaddingConfig()
```

Cross-validator (in the top-level RLConfig): when
`orchestrator.compaction_padding.enabled is True`,
require `trainer.compaction.message_padding_aware is True` AND
require `inference.vllm_extra.block_size == orchestrator.compaction_padding.block_size`.

### Change 9: `prime-rl/src/prime_rl/orchestrator/orchestrator.py` (startup)

After the orchestrator loads its tokenizer and before kicking off
rollouts, call `kv_eviction.env.configure_message_padding(...)`
with the resolved tokenizer + config values.

### Change 10: `prime-rl/src/prime_rl/orchestrator/trajectories.py`

In the rollout-to-TrainingSample conversion path (near the existing
`compaction_events_from_step_extras` call), add:

```python
from kv_eviction.env import padded_ids_from_step_extras

padded_ids = padded_ids_from_step_extras(first_step.extras)
if padded_ids is not None:
    sample.prompt_token_ids = padded_ids
    # Do NOT re-tokenize from messages — would lose the padding.
```

Requires adding the read-side helper `padded_ids_from_step_extras`
to `src/kv_eviction/env.py` (mirrors the existing
`compaction_events_from_step_extras`).

### Change 11: `prime-rl/src/prime_rl/trainer/rl/config.py`

```python
class CompactionConfig(BaseModel):
    ...
    message_padding_aware: bool = False
    # When True, positions where input_ids == filler_token_id are
    # masked out of the loss. filler_token_id is read from the
    # orchestrator's compaction_padding section via a cross-config
    # validator (must match).
    filler_token_id: int | None = None
```

### Change 12: `prime-rl/src/prime_rl/trainer/rl/data.py`

In `_micro_batch_to_tensor` (where `loss_mask` is built for the
batch): when `config.compaction.message_padding_aware is True`,
AND the existing loss mask with `(input_ids != filler_token_id)`
restricted to prompt positions. Completion positions are never
masked — filler collisions there are genuine model output and must
count.

### Change 13: no changes to `src/kv_eviction/segmented_forward.py`

Eviction-path math is already correct once the prompt length is
block-aligned: `prompt_aligned_len = ceil(mb_prompt_len / bs) * bs`
equals `mb_prompt_len` when padding is on, which equals what vLLM
used. Bug 2 (`explanation_bug_2_*.md`) is resolved by construction.

Add one defensive assertion: if compaction events are present and
padding is on, assert `mb_prompt_len % bs == 0`. Fail loud if not
(indicates config drift between orchestrator and trainer).

## Validation plan

### Gate 1 — vLLM unit test
`pytest vllm/tests/entrypoints/openai/test_chat_prompt_token_ids.py`
passes. New branch in `render_chat` returns `TokensPrompt` when
`prompt_token_ids` is set, and the echoed response carries the same
ids.

### Gate 2 — notebook-to-server parity
Rerun the loop in `compaction_test.ipynb` against the OpenAI-
compatible server (not the local `LLM`). With padding on, the server
logs' `evict=[…,…)` ranges should be identical to the notebook's
local-LLM logs on a fixed seed. Any divergence means the
tokenize-server roundtrip isn't byte-exact.

### Gate 3 — inference smoke
Run `experiments/debug_balrog/rl.toml` with
`orchestrator.compaction_padding.enabled = True`,
`trainer.compaction.message_padding_aware = True`, and
`inference.vllm_extra.compaction_max_turns = 4`. Inference-only (no
gradient step). Assistant output quality must match the notebook's
padded run (no `"I new observation"` fragments, no `"TheThe"`).

### Gate 4 — end-to-end RL
Same config, 5-step RL. Mismatch KL must stay at kernel floor
(~0.001). Loss-curve logging hook: per-batch, count positions where
`is_padding` mask fires; assert `loss.masked_select(is_padding).sum()
== 0`. This is the gate that confirms the whole pipeline — if Bug 2
is fixed by padding, KL should drop from 1.27 to ~0.001 immediately.

### Gate 5 — backward compat
Same config but `compaction_padding.enabled = False`. Bit-for-bit
identical behavior to the pre-patch state. Run the existing
compaction_rgmix smoke and diff wandb traces.

## Interaction with Bug 2

Bug 2 (`explanation_bug_2_prompt_aligned_len_mismatch_with_protected_prefix.md`)
is the `prompt_aligned_len` mismatch between trainer (400) and vLLM
(368) in protected-prefix auto-detect mode — 32-token positional
offset → KL 1.27.

With padding on:
- `<|im_end|>` after the system prompt lands on a block boundary.
- vLLM's auto-detect: `boundary = pos(sys_<|im_end|>) + 1 = 16*k`
  (already block-aligned), so `evict_start = 16*k`.
- Trainer's fallback: `ceil(mb_prompt_len / 16) * 16`. With padding,
  `mb_prompt_len = 16 * m` exactly (every message ends on a
  boundary), so the `ceil(...)` is a no-op and
  `prompt_aligned_len = 16 * m`.

Both compute the same physical boundary. **Bug 2 cannot manifest
when padding is on**, regardless of whether we also ship the
`eviction_start` wire-format fix.

## Decision record

### Why Option A (vLLM schema extension) and not B or C

| Option | Transport | vLLM change | Client change | Tool-call parsing |
|---|---|---|---|---|
| **A (picked)** | `extra_body={"prompt_token_ids": …}` on `/v1/chat/completions` | ~40 LOC, 1 new optional field + 1 bypass branch | monkey-patch only | preserved (conversation still built from `messages`) |
| B | route to `/v1/completions` with `prompt=…` | none | replicate hermes tool-call parser in client | **must re-implement** client-side |
| C | decode padded tokens → text → `messages` | none | lossy roundtrip | preserved but fragile |

B's cost (re-implementing hermes tool-call parsing for BALROG) is
larger than A's total cost. C's fragility (`<|endoftext|>` as filler
may not roundtrip cleanly through detokenize → tokenize) rules it
out.

### Why filler = `tokenizer.pad_token_id` (Qwen3: `<|endoftext|>` = 151643)

- The notebook prototype validated it produces clean eviction on
  BALROG BabyAI.
- It's not a chat-structural token (unlike `<|im_end|>` /
  `<|im_start|>`), so chat-template round-trip is unaffected.
- Open risk: if real `<|endoftext|>` appears in data, masking-by-id
  would incorrectly mask it. See Q3.

## Resolved design questions

**Q1 — monkey-patch hook point.** Wrap the underlying
`AsyncOpenAI` client's `chat.completions.create` method directly
(option (a) in the earlier draft). Verifiers' `get_response` wraps
this but does not forward arbitrary kwargs; patching one level
deeper is bulletproof. No touching of verifiers code.

**Q2 — `block_size` source of truth.** Plumb `block_size` explicitly
in the prime-rl config. It must be identical across
`inference.vllm_extra.block_size`,
`orchestrator.compaction_padding.block_size`, and
`trainer.compaction.block_size`; add a cross-config validator that
fails loud on drift. (Three declarations but single asserted value.)

**Q3 — filler token identity.** Use `tokenizer.pad_token_id`
(= Qwen3 `<|endoftext|>` = 151643) and mask by id in the trainer.
Collision risk accepted: real `<|endoftext|>` in chat data is
vanishingly rare, and the trainer mask only applies to prompt
positions (filler is never in completion). No wire-format change.

**Q4 — padding the trailing assistant-generation prompt.** NOT
padded. The trailing `<|im_start|>assistant\n` belongs to an
in-progress turn, which by the turn-mode eviction contract can never
be evicted (turn-mode only considers *completed* turns with both
`<|im_end|>`s committed). When this assistant turn later becomes a
prior turn — because a follow-up user message is appended —
`_render_padded` runs again on the full message list and block-aligns
its now-closed `<|im_end|>` before it ever becomes evictable.
Invariant: at the moment any turn becomes eligible for eviction, its
`<|im_end|>` is already block-aligned.

**Q5 — upstream vs fork.** Land on the `HyperPotatoNeo/vllm` fork,
branch `eviction-turn-based`. Not upstreaming. Upstream's AGENTS.md
duplicate-work / human-accountability rules do not apply.

**Q6 — notebook regression test.** Skip notebook migration. Validate
directly against real prime-rl rollouts (Gates 3–5 below).

## Order of operations

1. Land the vLLM patch (Changes 1–4). Pytest gate 1 passes.
2. Land `src/kv_eviction/padding.py` (Change 5). Unit-test against
   the notebook's `_render_padded` output for a fixed fixture.
3. Land the orchestrator-side monkey-patch (Changes 6–7, resolving
   Q1). Gate 2 passes (notebook-to-server parity).
4. Land the prime-rl config + trajectory plumbing (Changes 8–10).
   Gate 3 passes (inference smoke with padding on).
5. Land the trainer loss mask (Changes 11–13). Gate 4 passes
   (KL drops to kernel floor in a padded RL smoke).
6. Gate 5 (backward compat with padding off).
7. Deprecate `plans/block_aligned_message_padding.md` (server-side
   design) — mark as superseded by this plan.

## Files touched — summary

**New:**
- `src/kv_eviction/padding.py`
- `vllm/tests/entrypoints/openai/test_chat_prompt_token_ids.py`

**Modified — vLLM fork:**
- `vllm/vllm/entrypoints/openai/chat_completion/protocol.py` — add
  `prompt_token_ids` field to `ChatCompletionRequest`.
- `vllm/vllm/entrypoints/serve/render/serving.py::render_chat` —
  add bypass branch when `request.prompt_token_ids is not None`.

**Modified — kv-eviction integration:**
- `src/kv_eviction/env.py` — third monkey-patch wrapping chat
  completions; `configure_message_padding` helper;
  `padded_ids_from_step_extras` helper. Extend existing patches 1 &
  2 to forward `prompt_token_ids` alongside `compaction_events`.

**Modified — prime-rl:**
- `src/prime_rl/orchestrator/config.py` — `CompactionPaddingConfig`.
- `src/prime_rl/orchestrator/orchestrator.py` — call
  `configure_message_padding` at startup.
- `src/prime_rl/orchestrator/trajectories.py` — carry padded ids
  into `TrainingSample`.
- `src/prime_rl/trainer/rl/config.py` — `message_padding_aware`,
  `filler_token_id`, cross-validator.
- `src/prime_rl/trainer/rl/data.py` — loss-mask AND with
  `is_padding`.

**Not modified:**
- `src/kv_eviction/segmented_forward.py` — eviction math already
  correct under padded prompts; add one defensive assertion only.
- `vllm/v1/core/compaction/*` — untouched. Turn-mode eviction
  already block-aligned; padding just makes it exact.
- `CompactionEvent` wire format — unchanged.

## Total scope

- vLLM fork: ~50 LOC across 3 files (2 source + 1 test).
- kv-eviction: ~150 LOC across 2 files (new padding.py + env.py
  extensions).
- prime-rl: ~100 LOC across 5 files (config, orchestrator
  plumbing, trainer loss mask).

Single coherent feature, gated on a default-off flag, resolves Bug 2
as a side effect.
