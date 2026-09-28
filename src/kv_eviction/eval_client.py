"""Shared eval-harness client helpers for managed-context runs.

Extracted from experiments/textworld_env/eval_textworld.py so other env
harnesses (crafter, ...) can reuse the OpenAI client trace and the
managed-context configure call without copy-pasting. eval_textworld.py
keeps its original local copies for now (no regression risk on the
validated harness); new harnesses should import from here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def jsonl_append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def _finish_reasons(response: object) -> list[str | None]:
    choices = getattr(response, "choices", None) or []
    reasons: list[str | None] = []
    for choice in choices:
        reasons.append(getattr(choice, "finish_reason", None))
    return reasons


def _field(obj: object, key: str) -> object:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _choice_text_stats(response: object) -> list[dict[str, int]]:
    choices = getattr(response, "choices", None) or []
    out: list[dict[str, int]] = []
    for choice in choices:
        message = _field(choice, "message")
        content = _field(message, "content") if message is not None else None
        reasoning = (
            _field(message, "reasoning_content") if message is not None else None
        )
        if reasoning is None and message is not None:
            reasoning = _field(message, "reasoning")
        out.append(
            {
                "content_chars": len(content) if isinstance(content, str) else 0,
                "reasoning_chars": len(reasoning)
                if isinstance(reasoning, str)
                else 0,
            }
        )
    return out


def _usage_dict(response: object) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    out: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if isinstance(usage, dict):
            value = usage.get(key)
        else:
            value = getattr(usage, key, None)
        if isinstance(value, int):
            out[key] = value
    return out


def install_openai_client_trace(trace_path: Path) -> None:
    """Patch AsyncCompletions.create to append start/done/error rows to a
    jsonl trace. Per-call usage prompt_tokens from this trace is the
    verification metric for prompt construction (never num_messages)."""
    try:
        from openai.resources.chat.completions.completions import (
            AsyncCompletions,
        )
    except ImportError:
        return

    orig_create = AsyncCompletions.create
    if getattr(orig_create, "__kve_eval_trace_patched__", False):
        return

    counter = 0

    async def traced_create(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal counter
        counter += 1
        request_idx = counter
        t0 = time.time()
        messages = kwargs.get("messages")
        jsonl_append(
            trace_path,
            {
                "event": "start",
                "request_idx": request_idx,
                "ts": t0,
                "model": kwargs.get("model"),
                "num_messages": len(messages) if isinstance(messages, list) else None,
                "max_tokens": kwargs.get("max_tokens"),
            },
        )
        try:
            response = await orig_create(self, *args, **kwargs)
        except BaseException as exc:
            jsonl_append(
                trace_path,
                {
                    "event": "error",
                    "request_idx": request_idx,
                    "ts": time.time(),
                    "elapsed_sec": round(time.time() - t0, 3),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                },
            )
            raise
        jsonl_append(
            trace_path,
            {
                "event": "done",
                "request_idx": request_idx,
                "ts": time.time(),
                "elapsed_sec": round(time.time() - t0, 3),
                "finish_reasons": _finish_reasons(response),
                "choice_text_stats": _choice_text_stats(response),
                "usage": _usage_dict(response),
            },
        )
        return response

    traced_create.__kve_eval_trace_patched__ = True  # type: ignore[attr-defined]
    AsyncCompletions.create = traced_create


def add_managed_context_args(p) -> None:
    """argparse args mirroring eval_textworld.py exactly (names and
    defaults), so launchers are interchangeable across env harnesses."""
    p.add_argument(
        "--padding-block-size",
        type=int,
        default=0,
        help="Block size for block-aligned message padding. 0 = disable "
        "padding (full-context baseline). Use 16 for a compaction server.",
    )
    p.add_argument(
        "--phase4-padding",
        action="store_true",
        help="Stable per-rollout trace ids; required for managed-context "
        "restore because span IDs are scoped to the trace.",
    )
    p.add_argument(
        "--managed-context",
        action="store_true",
        help="Enable model-selected managed-context restore on the client side.",
    )
    p.add_argument("--managed-context-recall-max-spans", type=int, default=2)
    p.add_argument("--managed-context-index", action="store_true")
    p.add_argument("--managed-context-index-max-entries", type=int, default=12)
    p.add_argument(
        "--managed-context-restore-mode",
        choices=("kv", "visible_prefill"),
        default="kv",
    )
    p.add_argument("--managed-context-force-restore", action="store_true")
    p.add_argument(
        "--managed-context-force-span-policy",
        choices=("latest", "earliest", "random"),
        default="latest",
    )
    p.add_argument("--managed-context-require-retrieve", action="store_true")
    p.add_argument(
        "--managed-context-recall-mode",
        choices=("summary_select", "summary_select_preobs", "separate"),
        default="summary_select",
    )
    p.add_argument("--managed-context-compaction-max-turns", type=int, default=0)
    p.add_argument("--managed-context-turns-last-kept", type=int, default=0)


def configure_managed_context_from_args(args, model_name: str) -> bool:
    """Validate + install padding/managed-context client machinery from the
    standard args. Returns True when padding is enabled (compaction run)."""
    if args.managed_context and args.padding_block_size <= 0:
        raise ValueError("--managed-context requires --padding-block-size > 0")
    if args.managed_context and not args.phase4_padding:
        raise ValueError("--managed-context requires --phase4-padding")
    if args.padding_block_size <= 0:
        print(
            "[eval] block-aligned padding OFF (full-context baseline)",
            flush=True,
        )
        return False

    from transformers import AutoTokenizer

    from kv_eviction.env import (
        configure_message_padding,
        reset_managed_context_stats,
    )
    from kv_eviction.padding import (
        resolve_filler_token_id,
        resolve_im_end_token_id,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    im_end_id = resolve_im_end_token_id(tokenizer)
    filler_id = resolve_filler_token_id(
        tokenizer,
        override=None,
        forbidden_token_ids=(im_end_id,),
    )
    configure_message_padding(
        enabled=True,
        tokenizer=tokenizer,
        block_size=args.padding_block_size,
        filler_token_id=filler_id,
        im_end_token_id=im_end_id,
        phase4_enabled=args.phase4_padding,
        managed_context_enabled=args.managed_context,
        recall_max_spans=args.managed_context_recall_max_spans,
        managed_context_index_enabled=args.managed_context_index,
        managed_context_index_max_entries=(
            args.managed_context_index_max_entries
        ),
        managed_context_restore_mode=args.managed_context_restore_mode,
        managed_context_force_restore=args.managed_context_force_restore,
        managed_context_force_span_policy=(
            args.managed_context_force_span_policy
        ),
        managed_context_require_retrieve=args.managed_context_require_retrieve,
        managed_context_recall_mode=args.managed_context_recall_mode,
        managed_context_compaction_max_turns=(
            args.managed_context_compaction_max_turns
        ),
        managed_context_turns_last_kept=args.managed_context_turns_last_kept,
    )
    reset_managed_context_stats()
    print(
        f"[eval] block-aligned padding ON: block_size={args.padding_block_size} "
        f"im_end={im_end_id} filler={filler_id} "
        f"phase4={args.phase4_padding} "
        f"managed_context={args.managed_context}",
        flush=True,
    )
    return True
