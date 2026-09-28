# SPDX-License-Identifier: Apache-2.0
"""
Block-aligned chat message padding.

Ports the `_render_padded` + `_filler_token_id` recipe from
`experiments/debug_balrog/compaction_test.ipynb` into a reusable module
used by the orchestrator-side monkey-patch in `env.py`.

## Why this exists

Turn-based KV eviction (see
`vllm/v1/core/compaction/manager.py::compact_request`) operates on whole
PagedAttention blocks of size `block_size`. When a turn boundary
(`<|im_end|>`) falls inside a block rather than at its last slot, the
scheduler has two options:

1. Inward-snap: evict only the blocks entirely before the last `<|im_end|>`
   of the turn. Wastes capacity; leaves orphan fragments of the just-
   evicted turn in the KV. Also complicates the trainer's
   `prompt_aligned_len` math (Bug 2).
2. Exact-edge: evict up to and including the block that ends the turn.
   Requires the position immediately AFTER the turn's `<|im_end|>` to
   be a multiple of `block_size`.

Padding filler tokens *after* each `<|im_end|>` so that the next turn
starts at a block boundary (i.e. position after `<|im_end|>` is
`k*block_size` for some k) makes option 2 possible by construction.
Both inference and training see the same padded ids, so the
kernel-floor KL gap stays zero.

The AFTER layout also preserves the natural position of `<|im_end|>`
within each message — the model's learned EOS behavior is untouched,
and `<|im_end|>`'s absolute position is stable across re-renders
(filler counts float in the inter-turn gap, not inside the message).

## Contract

- Input: `messages`, `tools`, `block_size`, `filler_id`, `im_end_id`,
  tokenizer. Optionally `add_generation_prompt` (default True).
- Output: `(raw_ids, padded_ids, per_im_end_pads)`.
- `padded_ids` has filler tokens inserted AFTER each `<|im_end|>` so
  that the position immediately following each filler run is a multiple
  of `block_size` (i.e. the next turn's first token starts at a block
  boundary). `<|im_end|>` itself sits at its natural position — the
  model's learned EOS behavior is unchanged. `add_generation_prompt`
  appends `<|im_start|>assistant\\n` AFTER the last `<|im_end|>`'s
  filler run; that trailing region is not padded (the in-flight turn
  is never evictable under turn-mode and will be re-padded on the next
  call once closed).

## Determinism

Given the same `(messages, tools, block_size, filler_id, im_end_id,
tokenizer)`, the output is byte-identical across processes — enables
prefix-cache hits on the vLLM server across rollout calls with the
same prefix.
"""

from typing import Any, Iterable

# `tokenizer` is duck-typed here. The real object is a transformers
# PreTrainedTokenizerBase (the same instance prime-rl already uses for
# training). We avoid importing transformers at module load time.
Tokenizer = Any


def _turn_separator_token_ids(tokenizer: Tokenizer) -> list[int]:
    """Return chat-template separator tokens after ``<|im_end|>``.

    Qwen-style chat templates render messages as ``<|im_end|>\n``.  The
    separator belongs to the completed message boundary; padding before it
    leaves the next turn starting mid-block after the newline is consumed.
    """
    try:
        encoded = tokenizer.encode("\n", add_special_tokens=False)
    except Exception:
        return []
    if not encoded or not all(isinstance(tok, int) for tok in encoded):
        return []
    return [int(tok) for tok in encoded]


def resolve_filler_token_id(
    tokenizer: Tokenizer,
    override: int | None,
    forbidden_token_ids: Iterable[int | None] | None = None,
) -> int:
    """Pick the filler token id used to pad message bodies.

    Resolution order:
        override -> tokenizer.pad_token_id -> <|endoftext|> -> encode(' ')[-1]

    For Qwen3 this resolves to `<|endoftext|>` (151643), matching the
    notebook prototype. See Q3 in
    `plans/prime_rl_message_padding_patch.md` for the collision-risk
    discussion: the trainer's `is_padding` mask uses `(input_ids == filler_id)`
    and would also mask any *real* `<|endoftext|>` in data. Accepted as
    vanishingly rare in chat data. If you need a zero-collision filler,
    override with a token id that cannot appear in real content (e.g. a
    never-used reserved vocab slot).
    """
    forbidden = {
        int(token_id)
        for token_id in (forbidden_token_ids or ())
        if token_id is not None
    }

    def _usable(token_id: Any) -> int | None:
        if token_id is None:
            return None
        if isinstance(token_id, list):
            token_id = token_id[0] if token_id else None
        if token_id is None:
            return None
        token_id = int(token_id)
        if token_id < 0 or token_id in forbidden:
            return None
        return token_id

    if override is not None:
        filler = _usable(override)
        assert filler is not None, (
            f"invalid filler token override {override!r}; it collides with a "
            "reserved boundary token"
        )
        return filler
    filler = _usable(getattr(tokenizer, "pad_token_id", None))
    if filler is not None:
        return filler
    eot = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    filler = _usable(eot)
    if filler is not None:
        return filler
    enc = tokenizer.encode(" ", add_special_tokens=False)
    if enc:
        filler = _usable(enc[-1])
        if filler is not None:
            return filler
    raise AssertionError("no usable filler token")


def resolve_im_end_token_id(tokenizer: Tokenizer) -> int:
    """Look up the chat-template turn terminator token id.

    Qwen-family chat templates use `<|im_end|>`, while Gemma chat
    templates use `<turn|>`. Some tokenizers return `unk_token_id` for
    unknown special-token strings, so explicitly reject that fallback.
    """
    unk_token_id = getattr(tokenizer, "unk_token_id", None)
    candidates = ("<|im_end|>", "<turn|>", "<end_of_turn>")
    for token in candidates:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, list):
            token_id = token_id[0] if token_id else None
        if not isinstance(token_id, int) or token_id < 0:
            continue
        if unk_token_id is not None and token_id == int(unk_token_id):
            continue
        return token_id
    raise AssertionError(
        "no supported chat turn terminator found in tokenizer vocab "
        f"(tried {', '.join(candidates)})"
    )


def render_padded_prompt(
    *,
    tokenizer: Tokenizer,
    messages: list[dict],
    tools: list[dict] | None,
    block_size: int,
    filler_token_id: int,
    im_end_token_id: int,
    add_generation_prompt: bool = True,
) -> tuple[list[int], list[int], list[int]]:
    """Render messages -> token ids -> insert filler tokens AFTER each
    `<|im_end|>` so the next turn's first token starts at a
    block_size-aligned position.

    Position after the filler run is a multiple of `block_size`,
    matching what the turn-mode scheduler tracks in `turn_end_positions`
    (the scan-time snap in vllm/v1/core/sched/scheduler.py yields the
    same position).

    Returns:
        raw_ids: chat-template output BEFORE padding. Needed for
            debugging and for asserting `len(padded_ids) >= len(raw_ids)`.
        padded_ids: chat-template output AFTER inserting filler tokens
            after each `<|im_end|>`. This is what vLLM runs on (via
            `extra_body={"prompt_token_ids": padded_ids}`).
        per_im_end_pads: number of filler tokens inserted after each
            `<|im_end|>`, in order of appearance. Useful for reconstruct-
            ing the padding mask and for sanity checks.

    Implementation notes:
        - We go via string + `tokenizer.encode`, not `tokenize=True`.
          Some chat templates (notably with tools) return the rendered
          string regardless of `tokenize=True`, so the string path is
          the one we trust.
        - `add_special_tokens=False` because the chat template already
          emits any required special tokens.
        - When `add_generation_prompt=True`, the final block-alignment
          pad is inserted BEFORE `<|im_start|>assistant\\n`, not after.
          Putting filler after the generation prefix would prime the
          model with a run of filler tokens (typically <|endoftext|>)
          at the exact position where it starts generating, which
          pushes it off-distribution and causes degenerate output.
          Inserting the pad before the gen prefix keeps the model's
          last pre-generation tokens natural (<|im_start|>assistant\\n)
          while preserving the block-aligned total length the trainer
          relies on.
    """
    # Render messages without the generation prompt so we can isolate the
    # gen-prefix tokens and place the final alignment pad BEFORE them.
    rendered_msgs = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=False,
        tokenize=False,
    )
    raw = tokenizer.encode(rendered_msgs, add_special_tokens=False)
    # Defensive: ensure all ints. Some tokenizers return tokens as
    # np.int64 / tensor scalars which downstream serialization over
    # HTTP chokes on.
    bad = [(i, type(t).__name__) for i, t in enumerate(raw) if not isinstance(t, int)]
    if bad:
        raise TypeError(
            f"non-int tokens in tokenizer.encode output: first 5={bad[:5]}"
        )

    # Gen-prefix tokens = the delta when add_generation_prompt=True. We
    # re-render with the flag set and take the suffix past `raw`. The
    # prefix-preservation assertion catches chat templates that mutate
    # the message region when the gen prompt is added (would break the
    # padding math below).
    if add_generation_prompt:
        rendered_full = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        raw_full = tokenizer.encode(rendered_full, add_special_tokens=False)
        if raw_full[: len(raw)] != raw:
            raise RuntimeError(
                "Chat template is not prefix-preserving when "
                "add_generation_prompt is toggled. Block-aligned padding "
                "requires the gen prompt to be a pure suffix extension."
            )
        gen_prefix = raw_full[len(raw) :]
    else:
        gen_prefix = []

    # Per-turn pad: every <|im_end|> plus its chat-template separator gets
    # filler so the next turn's first token starts at a block_size-aligned
    # position. For Qwen this keeps the natural ``<|im_end|>\n`` boundary
    # intact and moves filler into the inter-turn gap.
    out: list[int] = []
    pads: list[int] = []
    separator_ids = _turn_separator_token_ids(tokenizer)
    i = 0
    while i < len(raw):
        tok = raw[i]
        out.append(tok)
        i += 1
        if tok == im_end_token_id:
            if separator_ids and raw[i : i + len(separator_ids)] == separator_ids:
                out.extend(separator_ids)
                i += len(separator_ids)
            remainder = len(out) % block_size
            n = (block_size - remainder) % block_size
            if n:
                out.extend([filler_token_id] * n)
            pads.append(n)

    # Final alignment: insert filler between the message region and the
    # generation prefix so that `len(out) + len(gen_prefix)` is a multiple
    # of block_size. With no gen prefix, this pads the tail as before.
    total_len = len(out) + len(gen_prefix)
    remainder = total_len % block_size
    if remainder:
        n = block_size - remainder
        out.extend([filler_token_id] * n)
        pads.append(n)

    out.extend(gen_prefix)

    return raw, out, pads
