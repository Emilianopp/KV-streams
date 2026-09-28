# SPDX-License-Identifier: Apache-2.0
"""
Client-side message truncation for the Markovian Thinker baseline.

Mirrors vLLM's turn-based KV-eviction schedule at the message level. Used by
the AsyncCompletions.create interceptor in ``env.py`` when the orchestrator
enables ``orchestrator.markovian_thinker``.

Contract (see ``plans/markovian_thinker_baseline.md``):

- **Protected prefix**: the first chat message, matching vLLM's first
  ``<|im_end|>`` protected boundary.
- **Completed turn**: each pair of subsequent message-end boundaries,
  independent of message roles or ``tool_calls``.
- **In-flight tail**: an unmatched final message after the completed pairs.
- **Eviction schedule**: trigger at ``num_live_turns >= max_turns`` and evict
  ``stride`` oldest turns. Repeated events produce the same sawtooth live
  window as vLLM.
"""

from collections.abc import Callable

from kv_eviction.summarization import (
    count_summary_exchanges,
    partition_messages,
)

__all__ = [
    "count_summary_exchanges",
    "kv_eviction_live_turns",
    "partition_messages",
    "partition_messages_for_kv_eviction",
    "truncate_messages_to_anchor_and_recent_turns",
    "truncate_messages_to_last_k_turns",
]


def partition_messages_for_kv_eviction(
    messages: list[dict],
) -> tuple[int, list[dict], list[list[dict]], list[dict]]:
    """Partition messages exactly like vLLM's turn-boundary counter.

    vLLM protects the first message-end marker, then counts every two later
    message-end markers as one completed turn. Roles are deliberately ignored:
    in a tool chain, ``[user, assistant(tool)]`` and
    ``[tool, assistant(tool)]`` are separate turns.
    """
    if not messages:
        return 0, [], [], []

    protected_prefix = [messages[0]]
    body = messages[1:]
    complete_message_count = len(body) - len(body) % 2
    groups = [
        list(body[start : start + 2])
        for start in range(0, complete_message_count, 2)
    ]
    tail = list(body[complete_message_count:])
    return len(groups), protected_prefix, groups, tail


def kv_eviction_live_turns(
    num_completed_turns: int,
    *,
    max_turns: int,
    stride: int | None,
) -> int:
    """Return vLLM's live-turn count after all due eviction events."""
    if num_completed_turns < 0:
        raise ValueError("num_completed_turns must be non-negative")
    if max_turns < 1:
        raise ValueError("max_turns must be positive")

    eviction_stride = 1 if stride is None else stride
    if eviction_stride < 1 or eviction_stride > max_turns:
        raise ValueError(
            f"stride must be in [1, max_turns], got {eviction_stride}"
        )
    if num_completed_turns < max_turns:
        return num_completed_turns

    num_evictions = (
        (num_completed_turns - max_turns) // eviction_stride
    ) + 1
    return num_completed_turns - num_evictions * eviction_stride


def truncate_messages_to_last_k_turns(
    messages: list[dict],
    *,
    max_turns: int,
    stride: int | None = None,
    log_fn: Callable[[str], None] | None = None,
) -> list[dict]:
    """Apply vLLM's turn-eviction schedule to a full message history.

    - ``max_turns``: eviction fires when completed live turns reach this.
    - ``stride``: number of oldest turns evicted per event. ``None`` means 1.

    Returns the input unchanged (same identity) when no truncation is
    needed. Never mutates the input list or its dicts.

    The caller supplies the canonical full history each time, so the number of
    prior events is derived analytically to reproduce vLLM's sawtooth window.
    """
    if not messages or max_turns < 1:
        return messages

    n_groups, protected_prefix, groups, tail = (
        partition_messages_for_kv_eviction(messages)
    )
    keep = kv_eviction_live_turns(
        n_groups,
        max_turns=max_turns,
        stride=stride,
    )
    if keep == n_groups:
        return messages

    dropped_count = n_groups - keep
    dropped = groups[:dropped_count]
    kept = groups[dropped_count:]

    if log_fn is not None:
        n_dropped_msgs = sum(len(g) for g in dropped)
        first = dropped[0][0] if dropped and dropped[0] else None
        last = dropped[-1][-1] if dropped and dropped[-1] else None
        log_fn(
            f"dropped {len(dropped)} turns ({n_dropped_msgs} msgs); "
            f"first.role={first.get('role') if first else '?'}, "
            f"last.role={last.get('role') if last else '?'}"
        )

    result: list[dict] = list(protected_prefix)
    for g in kept:
        result.extend(g)
    result.extend(tail)
    return result


def truncate_messages_to_anchor_and_recent_turns(
    messages: list[dict],
    *,
    max_turns: int,
    recent_turns: int,
    anchor_turns: int,
    anchor_policy: str = "earliest",
    log_fn: Callable[[str], None] | None = None,
) -> list[dict]:
    """Truncate at ``max_turns`` while preserving fixed anchor turns.

    This is the stateless visible re-prefill baseline for "recent context
    plus chosen context." It uses the same paired-message counter and
    ``>= max_turns`` trigger as KV eviction, then applies its intentionally
    different anchor-selection policy. For the current TextWorld
    comparison, ``max_turns=6``, ``recent_turns=2``, and
    ``anchor_turns=2`` yields:

        sys + turn1 + turn2 + turn5 + turn6 + pending_user

    No vLLM compaction, Phase4 inherited state, or hidden KV restore is
    involved.
    """
    if not messages or max_turns < 1:
        return messages

    n_groups, protected_prefix, groups, tail = (
        partition_messages_for_kv_eviction(messages)
    )
    if n_groups < max_turns:
        return messages

    recent = max(0, int(recent_turns))
    anchors = max(0, int(anchor_turns))
    if recent < 1 and anchors < 1:
        recent = 1

    policy = str(anchor_policy).strip().lower()
    if policy not in ("earliest", "latest"):
        policy = "earliest"

    selected: set[int] = set()
    if anchors:
        if policy == "latest":
            selected.update(range(max(0, n_groups - anchors), n_groups))
        else:
            selected.update(range(min(anchors, n_groups)))
    selected.update(range(max(0, n_groups - recent), n_groups))

    if len(selected) >= n_groups:
        return messages

    if log_fn is not None:
        dropped = [i for i in range(n_groups) if i not in selected]
        log_fn(
            f"dropped {len(dropped)} turns; kept anchors={anchors} "
            f"policy={policy} recent={recent}"
        )

    result: list[dict] = list(protected_prefix)
    for idx, group in enumerate(groups):
        if idx in selected:
            result.extend(group)
    result.extend(tail)
    return result
