# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for Markovian summary-based eviction.

Everything here is sync, side-effect free, and tokenizer-free. The
async orchestration (calling `orig_create` to generate a summary, the
`ContextVar` recursion guard, response-extras attachment) lives in
``env.py`` so this module stays testable in isolation.

See ``plans/markovian_summary.md`` for the full design.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


def partition_messages(
    messages: list[dict],
) -> tuple[int, list[dict], list[list[dict]], list[dict]]:
    """Split a message list into (n_groups, sys_prefix, body_groups, tail).

    - ``sys_prefix``: leading messages before the first ``role == "user"``.
      Always preserved by downstream consumers.
    - ``body_groups``: list of complete turn groups. Each group ends at a
      ``role == "assistant"`` message without ``tool_calls`` (the
      "terminal assistant" marker). Tool-call chains stay in one group.
    - ``tail``: trailing messages after the last terminal assistant —
      the in-flight pending exchange.
    - ``n_groups`` = ``len(body_groups)``.

    Does not mutate the input. Safe on empty lists and lists without
    any user message (returns ``(0, messages[:], [], [])``). Matches
    the semantics of ``truncate_messages_to_last_k_turns``'s internal
    partitioning; that function is now a thin wrapper over this helper.
    """
    if not messages:
        return 0, [], [], []

    sys_end = 0
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            break
        sys_end = i + 1

    last_terminal = -1
    for i in range(len(messages) - 1, sys_end - 1, -1):
        m = messages[i]
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            last_terminal = i
            break

    sys_prefix = list(messages[:sys_end])

    if last_terminal < sys_end:
        # No complete turn in the body. Everything after sys_prefix is
        # "in-flight tail".
        return 0, sys_prefix, [], list(messages[sys_end:])

    body = messages[sys_end : last_terminal + 1]
    tail = list(messages[last_terminal + 1 :])

    groups: list[list[dict]] = []
    group_start = 0
    for i, m in enumerate(body):
        if m.get("role") == "assistant" and not m.get("tool_calls"):
            groups.append(list(body[group_start : i + 1]))
            group_start = i + 1

    return len(groups), sys_prefix, groups, tail


def _content_eq(content: object, instruction_text: str) -> bool:
    """Message-content comparison that is tolerant of list-shaped
    multimodal content (``[{"type": "text", "text": "..."}]``).

    Only matches exactly-one text part equal to ``instruction_text``;
    any other list shape is treated as non-match. This is intentionally
    strict to avoid false positives on prompts that happen to contain
    the instruction text as a sub-string.
    """
    if isinstance(content, str):
        return content == instruction_text
    if isinstance(content, list) and len(content) == 1:
        part = content[0]
        if isinstance(part, dict) and part.get("type") == "text":
            return part.get("text") == instruction_text
    return False


def _is_summary_group(group: list[dict], instruction_text: str) -> bool:
    """True when a turn group is exactly a summary exchange ``[I, S]``:
    a user message whose content is ``instruction_text``, followed by an
    assistant message. Mirrors :func:`count_summary_exchanges`' matcher.
    """
    if not instruction_text or len(group) != 2:
        return False
    return (
        group[0].get("role") == "user"
        and group[1].get("role") == "assistant"
        and _content_eq(group[0].get("content"), instruction_text)
    )


def count_summary_exchanges(
    messages: list[dict], instruction_text: str
) -> int:
    """Count how many prior ``(user=I, assistant=*)`` summary exchanges
    are present in ``messages``.

    A summary exchange is a ``role="user"`` message whose content is
    exactly ``instruction_text`` followed immediately by a
    ``role="assistant"`` message (any content). Used to discount past
    summary turns from the trigger count in eviction mode, where the
    client-visible message list grows monotonically across triggers.

    Returns 0 when ``instruction_text`` is empty or no match is found.
    """
    if not instruction_text or not messages:
        return 0
    n = 0
    for i in range(len(messages) - 1):
        cur = messages[i]
        nxt = messages[i + 1]
        if (
            cur.get("role") == "user"
            and nxt.get("role") == "assistant"
            and _content_eq(cur.get("content"), instruction_text)
        ):
            n += 1
    return n


def build_exchange(
    instruction_text: str, summary_text: str
) -> tuple[dict, dict]:
    """Build the ``(I_msg, S_msg)`` pair that gets spliced into messages.

    ``I_msg`` is a plain-string user message carrying the instruction;
    ``S_msg`` is a plain-string assistant message carrying the summary.
    No ``tool_calls`` on ``S_msg`` — it's a terminal assistant, which
    means subsequent calls to ``partition_messages`` will see it as a
    new turn group boundary.
    """
    I_msg = {"role": "user", "content": instruction_text}
    S_msg = {"role": "assistant", "content": summary_text}
    return I_msg, S_msg


def build_post_summary_messages(
    mode: str,
    sys_prefix: list[dict],
    body_groups: list[list[dict]],
    tail: list[dict],
    instruction_text: str,
    summary_text: str,
    n_preserved_turns: int = 0,
    resume_text: str = "",
) -> list[dict]:
    """Build the message list spliced with the summary exchange.

    - ``mode="markovian"``: partial client-side reset. Splice the summary
      exchange ``[I, S]`` directly after the system prefix, then keep the
      last ``n_preserved_turns`` body groups and the in-flight tail (the
      latest observation/tool result). Output is
      ``sys_prefix + [I, S] + last_N_body_groups + tail``. When
      ``n_preserved_turns == 0`` this is a strict full reset:
      ``sys_prefix + [I, S] + tail``.

      The summary leads because the result is *persisted* as the
      conversation's new base (see ``attach_compacted_prompt_from_response``
      in ``env.py``): the next turn continues from this list, so the
      summary has to sit where the dropped history used to be, not after
      the observation that follows it. Leading also makes
      ``sys_prefix + [I, S]`` a stable prefix between triggers, which is
      KV-cache reusable, and makes the next summary recursive — it reads
      the previous ``[I, S]`` as part of its input.

      ``resume_text`` is appended only when ``tail`` is empty; a non-empty
      tail is already the pending user turn vLLM generates against.
    - ``mode="eviction"``: append-only splice. Keep ``body_groups``.
      Output is ``sys_prefix + flatten(body_groups) + [I, S] + tail``.
      ``n_preserved_turns`` and ``resume_text`` are ignored in this mode
      (the full body stays; vLLM-side eviction handles KV compression,
      and the original tail remains the pending user turn).

    Raises ``ValueError`` on unknown mode. Pure function — safe to unit
    test without touching async / tokenizer code.
    """
    I_msg, S_msg = build_exchange(instruction_text, summary_text)
    if mode == "markovian":
        keep = max(0, int(n_preserved_turns))
        preserved: list[dict] = []
        # The new summary supersedes any earlier one: it was generated
        # having read it. Drop prior [I, S] groups so exactly one summary
        # survives, at the front. Without this, a config where the
        # preserve window outruns the trigger period (max_turns much
        # larger than compaction_max_turns) would carry old summaries
        # forward and stack them.
        real_groups = [
            g for g in body_groups if not _is_summary_group(g, instruction_text)
        ]
        if keep > 0 and real_groups:
            for g in real_groups[-keep:]:
                preserved.extend(g)
        out = list(sys_prefix) + [I_msg, S_msg] + preserved + list(tail)
        # `tail` is the pending observation, so it already gives vLLM a
        # user turn to generate against and no resume message is needed.
        # Only when the tail is empty (no in-flight observation) does the
        # prompt end on the assistant summary and need a resume turn.
        if resume_text and not tail:
            out.append({"role": "user", "content": resume_text})
        return out
    if mode == "eviction":
        out2: list[dict] = list(sys_prefix)
        for g in body_groups:
            out2.extend(g)
        out2.extend([I_msg, S_msg])
        out2.extend(tail)
        return out2
    raise ValueError(
        f"build_post_summary_messages: unknown mode={mode!r}; "
        "expected 'markovian' or 'eviction'"
    )


# ChatML markers that must never appear inside an assistant message
# body — the chat template will emit its own. Strip them defensively
# so a hallucinated template token doesn't poison subsequent renders.
_DEFAULT_BLOCKLIST: tuple[str, ...] = ("<|im_start|>", "<|im_end|>")


def sanitize_summary(
    text: str, blocklist: tuple[str, ...] = _DEFAULT_BLOCKLIST
) -> tuple[str, bool]:
    """Remove chat-template tokens from ``text``.

    Returns ``(sanitized, was_modified)``. ``was_modified`` is True iff
    at least one blocklisted token was found. No regex — plain string
    replacement over the blocklist in order.
    """
    was_modified = False
    out = text
    for tok in blocklist:
        if tok in out:
            was_modified = True
            out = out.replace(tok, "")
    # Strip ACTION TAGS. A summary is spliced into the conversation as an
    # assistant turn; if it carries an <action> tag the model reads its own
    # move as already submitted and answers the next observation with an
    # empty turn (measured: P(<|im_end|>) = 0.9997, 3/3 captured empties had
    # a summary ending in <action>...</action>).
    import re as _re
    stripped = _re.sub(r"<\s*action\s*>.*?<\s*/\s*action\s*>", "", out,
                       flags=_re.IGNORECASE | _re.DOTALL)
    stripped = _re.sub(r"<\s*/?\s*action\s*>", "", stripped, flags=_re.IGNORECASE)
    if stripped != out:
        was_modified = True
        out = stripped
    return out, was_modified


def content_to_text(content: object) -> str:
    """Flatten a possibly-multimodal message ``content`` field to plain
    text for debug logging. List-shaped content with ``type="text"``
    parts is concatenated; other types are rendered as ``[<type>]``.

    Never raises; unknown shapes fall through to ``str(content)``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                t = item.get("type")
                if t == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(f"[{t}]")
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def extract_prompt_token_ids(response: object) -> list[int]:
    """Pull ``prompt_token_ids`` off a ChatCompletion-ish response.

    Sourced from the admission-trim-KL patch on the vLLM fork, which
    stashes the post-admission token ids on the response object. Handles
    both attribute access (pydantic ``extra="allow"``) and
    ``model_extra`` fallback. Returns [] when absent.
    """
    if response is None:
        return []
    ids = getattr(response, "prompt_token_ids", None)
    if ids is None and hasattr(response, "model_extra"):
        extra = response.model_extra or {}
        ids = extra.get("prompt_token_ids")
    if ids is None:
        return []
    try:
        return [int(x) for x in ids]
    except (TypeError, ValueError):
        return []


def extract_completion_token_ids(response: object) -> list[int]:
    """Pull the sampled assistant token ids off a ChatCompletion.

    Primary source: ``resp.choices[0].token_ids`` (vLLM fork extension
    — same field the trainer reads for the rollout's completion tokens).
    Returns [] when missing/malformed.
    """
    try:
        choice = response.choices[0]  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError):
        return []
    ids = getattr(choice, "token_ids", None)
    if ids is None and hasattr(choice, "model_extra"):
        extra = choice.model_extra or {}
        ids = extra.get("token_ids")
    if ids is None:
        return []
    try:
        return [int(x) for x in ids]
    except (TypeError, ValueError):
        return []


def extract_completion_logprobs(response: object) -> list[float]:
    """Pull one scalar logprob per completion token from an OpenAI
    ChatCompletion's ``choices[0].logprobs.content``.

    Tolerates every shape verifiers' own parser accepts (see
    ``parse_tokens`` in openai_chat_completions_client.py): ``logprobs``
    as a pydantic object OR a plain dict, and content entries as objects
    with ``.logprob`` OR dicts with ``["logprob"]``. The object-only
    version silently returned [] on dict-shaped responses, which made
    every summary TrainingSample downstream drop with
    "N ids vs 0 logprobs" while the action path (using the tolerant
    parser) kept working. Returns [] when genuinely missing or empty.
    """
    try:
        choice = response.choices[0]  # type: ignore[attr-defined]
    except (AttributeError, IndexError, TypeError):
        return []
    lp = getattr(choice, "logprobs", None)
    if lp is None and isinstance(choice, dict):
        lp = choice.get("logprobs")
    if lp is None:
        return []
    if isinstance(lp, dict):
        content = lp.get("content")
    else:
        content = getattr(lp, "content", None)
    if content is None:
        return []
    out: list[float] = []
    for entry in content:
        try:
            value = entry["logprob"] if isinstance(entry, dict) else entry.logprob
            out.append(float(value))
        except (AttributeError, KeyError, TypeError, ValueError):
            return []
    return out


@dataclass
class SummaryTrainSample:
    """Training-sample payload carried from the interceptor through the
    trajectory to the orchestrator, where it becomes a standalone
    :class:`TrainingSample` in ``interleave_rollout``.

    - ``prompt_token_ids``: the tokens vLLM actually processed for the
      summary request (sourced from the response's ``prompt_token_ids``
      field, stashed by the admission-trim patch on the fork).
    - ``completion_token_ids``: the sampled assistant tokens for the
      summary response (sourced from ``choices[0].token_ids``).
    - ``completion_logprobs``: one scalar logprob per completion token,
      in generation order. Becomes the trainer's ``old_logprobs``.
    - ``model``: model name used — trainer sanity-checks this matches
      the rollout model to avoid cross-model logprob mixups.
    - ``compaction_events``: vLLM-side eviction events emitted during
      the summary call's prefill/decode. Populated in ``mode="eviction"``
      only. Empty in ``mode="markovian"`` (no vLLM compaction). Stored
      as plain JSON-serializable dicts to survive the msgspec roundtrip
      through verifiers' trajectory-state path; converted to
      ``CompactionEventWire`` in ``_build_summary_sample``.
    - ``submitted_prompt_token_ids`` and ``compaction_replay_mode``:
      present only for SGLang ``prefill_trim`` replay. The submitted IDs
      are the exact pre-trim request; ``prompt_token_ids`` contains the
      event's authoritative post-trim survivors, or the unchanged physical
      prompt when no new admission event fired.
    - ``native_prompt_token_ids``: SGLang's echoed pre-compaction prompt,
      retained so prime can independently verify submitted/native equality.
    - ``turn_compaction_state``: validated cumulative SGLang state, retained as
      plain JSON and anchored to the carried prefix of the physical prompt.
    - ``completion_temperature``: the positive effective logit temperature
      used to replay the summary call in training. Greedy requests (0.0) use
      1.0 because inference selects greedily from unscaled logits.
    """

    prompt_token_ids: list[int] = field(default_factory=list)
    completion_token_ids: list[int] = field(default_factory=list)
    completion_logprobs: list[float] = field(default_factory=list)
    model: str = ""
    compaction_events: list[dict] = field(default_factory=list)
    submitted_prompt_token_ids: list[int] = field(default_factory=list)
    native_prompt_token_ids: list[int] = field(default_factory=list)
    completion_temperature: float | None = None
    compaction_replay_mode: str | None = None
    turn_compaction_state: dict | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        if not self.submitted_prompt_token_ids:
            payload.pop("submitted_prompt_token_ids")
        if not self.native_prompt_token_ids:
            payload.pop("native_prompt_token_ids")
        if self.completion_temperature is None:
            payload.pop("completion_temperature")
        if self.compaction_replay_mode is None:
            payload.pop("compaction_replay_mode")
        if self.turn_compaction_state is None:
            payload.pop("turn_compaction_state")
        return payload

    @classmethod
    def from_dict(cls, d: dict) -> "SummaryTrainSample":
        replay_mode = d.get("compaction_replay_mode")
        if replay_mode is not None and replay_mode != "prefill_trim":
            raise ValueError(
                f"unsupported compaction_replay_mode: {replay_mode!r}"
            )
        raw_turn_compaction_state = d.get("turn_compaction_state")
        if raw_turn_compaction_state is not None and not isinstance(
            raw_turn_compaction_state, dict
        ):
            raise ValueError("turn_compaction_state must be a JSON object")
        return cls(
            prompt_token_ids=[int(x) for x in d.get("prompt_token_ids", []) or []],
            completion_token_ids=[
                int(x) for x in d.get("completion_token_ids", []) or []
            ],
            completion_logprobs=[
                float(x) for x in d.get("completion_logprobs", []) or []
            ],
            model=str(d.get("model", "") or ""),
            compaction_events=[
                dict(e) for e in (d.get("compaction_events") or []) if isinstance(e, dict)
            ],
            submitted_prompt_token_ids=[
                int(x)
                for x in (d.get("submitted_prompt_token_ids", []) or [])
            ],
            native_prompt_token_ids=[
                int(x) for x in (d.get("native_prompt_token_ids", []) or [])
            ],
            completion_temperature=(
                float(d["completion_temperature"])
                if d.get("completion_temperature") is not None
                else None
            ),
            compaction_replay_mode=replay_mode,
            turn_compaction_state=(
                dict(raw_turn_compaction_state)
                if raw_turn_compaction_state is not None
                else None
            ),
        )
