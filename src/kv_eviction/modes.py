"""Single-flag KV-context modes.

One name selects the whole machinery stack for an RL run (or eval):

- ``kv-eviction``      engine turn-eviction, spans drop as they come, the
                       visible window is the last ``max_turns``/``stride``
                       turns. No recall, no CPU backup. KV continuity via
                       block splice (never recomputed).
- ``markovian``        Markovian Thinker baseline: client-side truncation
                       to the last ``max_turns`` turn groups, re-prefilled
                       as normal full-context requests. Reference arm
                       (recompute allowed).
- ``kv-recall``        eviction + hidden-KV recall with the validated
                       production stack (at-compaction cadence, terse +
                       non-thinking manager, soft-pin/lazy-publish/
                       keep-cpu, eager CPU archive).
- ``kv-selection``     eviction + explicit/model-selected complete-turn
                       retention from the would-be evicted turn band. No
                       CPU archive, hidden restore, or arbitrary recall.
- ``markovian-recall`` eviction + recall where restores are VISIBLE
                       re-prefills of the archived text (recompute
                       allowed; reference arm for recall).

``engine_env_for_mode`` belongs in the vLLM inference process environment;
``client_env_for_mode`` belongs in the orchestrator/eval client process
(and any env-server subprocess it spawns); ``padding_kwargs_for_mode``
extends ``kv_eviction.env.configure_message_padding``.
"""

from __future__ import annotations

KV_MODES = (
    "kv-eviction",
    "markovian",
    "kv-recall",
    "kv-selection",
    "markovian-recall",
)

_RECALL_MODES = ("kv-recall", "markovian-recall")

# Engine-side recipe validated 2026-06-12 (plans/soft_pin_design.md):
# soft-pin stack (CPU-backed pins; mandatory under pressure or rollouts
# die to pin-missing aborts), lazy publish (no per-call mirror below
# 0.80 pool), reload keep-cpu (delta inheritance survives reloads),
# eager span archive (lazily-parked spans strangle the pool on
# think-heavy turns).
_RECALL_ENGINE_ENV: dict[str, str] = {
    "KVE_MANAGED_CONTEXT": "1",
    "KVE_MANAGED_CONTEXT_ARCHIVE_DEVICE": "cpu",
    "KVE_MANAGED_CONTEXT_CPU_OFFLOAD_POLICY": "immediate",
    "KVE_MANAGED_CONTEXT_CPU_OFFLOAD_MAX_BLOCKS": "65536",
    "KVE_MANAGED_CONTEXT_SCHEDULER_ACCOUNTED_RESTORE": "1",
    "KVE_MANAGED_CONTEXT_GPU_HOT_BLOCK_BUDGET": "2048",
    "KVE_MANAGED_CONTEXT_CPU_EVICT_ON_CAPACITY": "1",
    "KVE_MANAGED_CONTEXT_EVICT_BY_RECALL": "0",
    "KVE_MANAGED_CONTEXT_DROP_UNAVAILABLE_RESTORE": "1",
    "KVE_SOFT_PIN": "1",
    "KVE_SOFT_PIN_STREAM_MIRROR": "1",
    "KVE_SOFT_PIN_REVOCABLE_RESTORES": "1",
    "KVE_SOFT_PIN_ATOMIC_RESUME": "1",
    "KVE_SOFT_PIN_LAZY_PUBLISH": "1",
    "KVE_PIN_RELOAD_KEEP_CPU": "1",
    "KVE_PHASE4_PIN_PREFETCH": "0",
    "KVE_REQUEST_KV_SWAP": "1",
    "KVE_REQUEST_KV_SWAP_MIN_PROGRESS_TOKENS": "160",
    "KVE_REQUEST_KV_SWAP_MAX_PENDING_STORES": "16",
    "KVE_REQUEST_KV_SWAP_MAX_PENDING_LOADS": "8",
    "KVE_REQUEST_KV_SWAP_OFFLOAD_START_USAGE": "0.92",
    "KVE_REQUEST_KV_SWAP_OFFLOAD_STOP_USAGE": "0.85",
    "KVE_REQUEST_KV_SWAP_RELOAD_TARGET_USAGE": "0.97",
    "KVE_REQUEST_KV_SWAP_RELOAD_STARVATION_SECONDS": "300",
    "KVE_REQUEST_KV_SWAP_EAGER_FILL": "1",
    "KVE_SWAP_RELOAD_CACHE_HIT": "1",
    "KVE_PHASE4_PROACTIVE_CPU_OFFLOAD": "1",
    "KVE_PHASE4_PROACTIVE_OFFLOAD_START_USAGE": "0.90",
    "KVE_PHASE4_PIN_LOAD_TARGET_USAGE": "0.97",
    "KVE_PHASE4_PIN_PROTECT_QUEUED_SUCCESSORS": "1",
}

# Client-side recall flow: manager fires only at compaction events
# (the 470s->276s counting win), terse + non-thinking manager (thinking
# models burn the manager budget mid-<think>), per-turn span granularity
# so the model picks individual turns.
_RECALL_CLIENT_ENV: dict[str, str] = {
    "KVE_MANAGED_CONTEXT_RECALL_AT_COMPACTION": "1",
    "KVE_MANAGED_CONTEXT_TERSE_MANAGER": "1",
    "KVE_MANAGED_CONTEXT_NONTHINKING_MANAGER": "1",
    "KVE_MANAGED_CONTEXT_PER_TURN_SPANS": "1",
}


def validate_kv_mode(mode: str | None) -> str | None:
    if mode is None:
        return None
    if mode not in KV_MODES:
        raise ValueError(f"kv_mode must be one of {KV_MODES}, got {mode!r}")
    return mode


def engine_env_for_mode(mode: str | None, *, recall_max_spans: int = 5) -> dict[str, str]:
    validate_kv_mode(mode)
    if mode not in _RECALL_MODES:
        return {}
    env = dict(_RECALL_ENGINE_ENV)
    env["KVE_MANAGED_CONTEXT_RECALL_MAX_SPANS"] = str(recall_max_spans)
    return env


def client_env_for_mode(mode: str | None) -> dict[str, str]:
    validate_kv_mode(mode)
    if mode not in _RECALL_MODES:
        return {}
    return dict(_RECALL_CLIENT_ENV)


def padding_kwargs_for_mode(
    mode: str | None,
    *,
    max_turns: int,
    stride: int,
    recall_max_spans: int = 5,
    index_max_entries: int = 12,
) -> dict[str, object]:
    """Extra kwargs for configure_message_padding in the recall modes.

    ``turns_last_kept`` (prompt guidance for the manager) = the post-
    eviction remainder ``max_turns - stride``.
    """
    validate_kv_mode(mode)
    if mode not in _RECALL_MODES:
        return {}
    return {
        "managed_context_enabled": True,
        "recall_max_spans": recall_max_spans,
        "managed_context_index_enabled": True,
        "managed_context_index_max_entries": index_max_entries,
        "managed_context_restore_mode": (
            "kv" if mode == "kv-recall" else "visible_prefill"
        ),
        "managed_context_recall_mode": "summary_select",
        "managed_context_compaction_max_turns": int(max_turns),
        "managed_context_turns_last_kept": max(0, int(max_turns) - int(stride)),
    }
