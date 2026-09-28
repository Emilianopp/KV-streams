"""Run textworld-env rollouts against a live vLLM inference server.

Smoke test for the textworld port: loads N eval samples, rolls out each
one against a running vLLM OpenAI-compatible endpoint, and reports reward,
completion length, compaction event counts, soft success rate (mean
normalized TextWorld score), and hard success rate (1 iff final score reaches
max score / the game is won).

Designed to be run from inside the kv-eviction venv on a node (or
container) that has network access to the vLLM server:

    python eval_textworld.py \
        --dataset /pscratch/.../textworld_cooking_mix \
        --base-url http://localhost:8000/v1 \
        --model Qwen/Qwen3-4B-Instruct-2507 \
        --num-examples 100 \
        --eval-source eval \
        --eval-set-json experiments/textworld_env/eval_sets/textworld_eval_100_seed42.json \
        --max-episode-steps 50 \
        --max-concurrent 32 \
        --output-json /path/to/results.json

When --padding-block-size > 0, this script:
  1. Loads the model's tokenizer
  2. Resolves `<|im_end|>` and filler token ids via
     kv_eviction.padding.resolve_im_end_token_id / resolve_filler_token_id
  3. Calls kv_eviction.env.configure_message_padding(...) which installs
     the openai.AsyncCompletions.create interceptor
Every subsequent chat request is pre-padded so its `<|im_end|>`s land on
block-size boundaries — required for the server-side
`compaction_assume_aligned_turn_boundaries=true` path.

Pass --padding-block-size 0 to disable padding (full-context baseline).
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
import statistics
import sys
import time
from pathlib import Path

from datasets import Dataset, concatenate_datasets


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="Path to textworld_cooking_mix directory")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--num-examples", type=int, default=100)
    p.add_argument(
        "--example-offset",
        type=int,
        default=0,
        help="Start index into the (shuffled) selection; enables sharded/resumable runs.",
    )
    p.add_argument(
        "--eval-source",
        choices=("eval", "train-shuffle", "all-shuffle"),
        default="eval",
        help=(
            "Where to draw examples from. 'eval' uses the held-out "
            "eval_dataset split; '*-shuffle' modes deterministically shuffle "
            "train or train+eval rows with --seed."
        ),
    )
    p.add_argument(
        "--eval-set-json",
        default=None,
        help=(
            "Optional manifest of fixed TextWorld answer/game ids. If the file "
            "exists, those ids are loaded and --eval-source/--seed selection is "
            "ignored. If it does not exist, the selected eval set is written."
        ),
    )
    p.add_argument("--max-episode-steps", type=int, default=50)
    p.add_argument("--max-concurrent", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=512)
    # Synthetic counting benchmark (throughput-under-KV-load): full-context vs
    # managed-context under a controlled per-turn decode volume. See counting_env.py.
    p.add_argument(
        "--env-type",
        choices=("textworld", "counting"),
        default="textworld",
        help="Which env to run. 'counting' = synthetic forced-length counting harness.",
    )
    p.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        help="Force a minimum decode length per turn (with --ignore-eos, pins exact length).",
    )
    p.add_argument(
        "--ignore-eos",
        action="store_true",
        help="Ignore EOS so each turn decodes to max_tokens (forces exact per-turn volume).",
    )
    p.add_argument(
        "--system-prompt-variant",
        choices=("default", "compact", "thinking"),
        default="default",
        help=(
            "TextWorld system prompt variant. 'default' preserves the existing "
            "long prompt; 'compact' is a shorter action-only prompt useful for "
            "models that emit EOS on the long prompt; 'thinking' keeps a compact "
            "action format while explicitly asking the model to reason first."
        ),
    )
    # Default 1.0 + no top_p/top_k = ancestral sampling (matches mkv-rl prod).
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--min-p", type=float, default=None)
    p.add_argument("--presence-penalty", type=float, default=None)
    p.add_argument("--repetition-penalty", type=float, default=None)
    p.add_argument(
        "--chat-template-enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Forward chat_template_kwargs.enable_thinking in OpenAI "
            "extra_body when the served model supports it."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-json", default="textworld_eval_results.json")
    p.add_argument(
        "--save-traces",
        default=None,
        help=(
            "If set, write one JSON line per rollout containing the FULL "
            "conversation (prompt + completion messages, including <think> "
            "blocks), reward, TextWorld scores, and game metadata to this "
            "JSONL path. This is the trace-collection dataset."
        ),
    )
    p.add_argument(
        "--api-key-var",
        default="DUMMY_API_KEY",
        help="Name of env var holding the API key (vLLM ignores but openai client requires one).",
    )
    p.add_argument(
        "--client-timeout",
        type=float,
        default=float(os.environ.get("CLIENT_TIMEOUT", "3600")),
        help="OpenAI client request timeout in seconds.",
    )
    p.add_argument(
        "--client-connect-timeout",
        type=float,
        default=float(os.environ.get("CLIENT_CONNECT_TIMEOUT", "30")),
        help="OpenAI client connect timeout in seconds.",
    )
    p.add_argument(
        "--client-max-retries",
        type=int,
        default=int(os.environ.get("CLIENT_MAX_RETRIES", "3")),
        help="OpenAI client max retries.",
    )
    p.add_argument(
        "--padding-block-size",
        type=int,
        default=16,
        help="Block size for block-aligned message padding. 0 = disable padding "
             "(use for full-context baseline).",
    )
    p.add_argument(
        "--phase4-padding",
        action="store_true",
        help=(
            "Submit Phase4 incremental prompts after the first turn. Required "
            "for managed-context restore because span IDs are scoped to the "
            "per-rollout Phase4 trace."
        ),
    )
    p.add_argument(
        "--managed-context",
        action="store_true",
        help="Enable model-selected managed-context restore on the client side.",
    )
    p.add_argument(
        "--managed-context-recall-max-spans",
        type=int,
        default=2,
        help="Maximum number of hidden memory spans the model may request.",
    )
    p.add_argument(
        "--managed-context-index",
        action="store_true",
        help=(
            "Append a non-leaking hidden-memory index to user turns when "
            "archived span IDs are available."
        ),
    )
    p.add_argument(
        "--managed-context-index-max-entries",
        type=int,
        default=0,
        help=(
            "Maximum archived span IDs to show in the visible index. "
            "0 means show all archived spans."
        ),
    )
    p.add_argument(
        "--managed-context-restore-mode",
        choices=("kv", "visible_prefill"),
        default="kv",
        help=(
            "How to satisfy retrieve requests: hidden KV restore or visible "
            "re-prefill of archived evicted tokens."
        ),
    )
    p.add_argument(
        "--managed-context-force-restore",
        action="store_true",
        help=(
            "When any archived span is available, restore/re-prefill it on "
            "the next model request without waiting for model-selected JSON."
        ),
    )
    p.add_argument(
        "--managed-context-force-span-policy",
        choices=("latest", "earliest", "random"),
        default="latest",
        help="Which archived span(s) to choose in forced-restore mode.",
    )
    p.add_argument(
        "--managed-context-require-retrieve",
        action="store_true",
        help=(
            "When the hidden-memory index is shown, require the model to "
            "reply with retrieve JSON instead of taking a TextWorld action "
            "on the first pass."
        ),
    )
    p.add_argument(
        "--managed-context-recall-mode",
        choices=("summary_select", "summary_select_preobs", "separate"),
        default="summary_select",
        help=(
            "Managed-context first-pass flow. summary_select asks the model "
            "to update span summaries and choose restores in one JSON object; "
            "summary_select_preobs asks for that JSON in a separate "
            "pre-observation control turn; "
            "separate keeps the older retrieve-only prompt."
        ),
    )
    p.add_argument(
        "--managed-context-compaction-max-turns",
        type=int,
        default=0,
        help=(
            "Client-side compaction threshold used to trigger the first "
            "summary_select memory-manager pass before any archived IDs "
            "exist. 0 disables proactive triggering."
        ),
    )
    p.add_argument(
        "--managed-context-turns-last-kept",
        type=int,
        default=0,
        help=(
            "Number of recent turns kept visible after compaction; used in "
            "the summary_select prompt as guidance."
        ),
    )
    return p.parse_args()


def _jsonl_append(path: Path, row: dict) -> None:
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


def _install_openai_client_trace(trace_path: Path) -> None:
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
        _jsonl_append(
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
            _jsonl_append(
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
        _jsonl_append(
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


def _summarize_openai_client_trace(trace_path: Path | None) -> dict:
    if trace_path is None or not trace_path.exists():
        return {}
    starts = 0
    done = 0
    errors = 0
    error_types: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()
    usage_totals: Counter[str] = Counter()
    usage_rows = 0
    content_chars_total = 0
    reasoning_chars_total = 0
    reasoning_chars_max = 0
    reasoning_rows = 0
    started_ids: set[int] = set()
    closed_ids: set[int] = set()
    with trace_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            event = row.get("event")
            req_id = row.get("request_idx")
            if event == "start":
                starts += 1
                if isinstance(req_id, int):
                    started_ids.add(req_id)
            elif event == "done":
                done += 1
                if isinstance(req_id, int):
                    closed_ids.add(req_id)
                for reason in row.get("finish_reasons") or []:
                    finish_reasons[str(reason)] += 1
                for text_stats in row.get("choice_text_stats") or []:
                    if not isinstance(text_stats, dict):
                        continue
                    content_chars = text_stats.get("content_chars")
                    reasoning_chars = text_stats.get("reasoning_chars")
                    if isinstance(content_chars, int):
                        content_chars_total += content_chars
                    if isinstance(reasoning_chars, int):
                        reasoning_chars_total += reasoning_chars
                        reasoning_chars_max = max(reasoning_chars_max, reasoning_chars)
                        if reasoning_chars > 0:
                            reasoning_rows += 1
                usage = row.get("usage") or {}
                if isinstance(usage, dict) and usage:
                    usage_rows += 1
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        value = usage.get(key)
                        if isinstance(value, int):
                            usage_totals[key] += value
            elif event == "error":
                errors += 1
                if isinstance(req_id, int):
                    closed_ids.add(req_id)
                error_types[str(row.get("error_type") or "unknown")] += 1
    return {
        "path": str(trace_path),
        "started": starts,
        "done": done,
        "errors": errors,
        "open": len(started_ids - closed_ids),
        "error_types": dict(error_types),
        "finish_reasons": dict(finish_reasons),
        "usage_rows": usage_rows,
        "usage_totals": dict(usage_totals),
        "choice_text_stats": {
            "content_chars_total": content_chars_total,
            "reasoning_chars_total": reasoning_chars_total,
            "reasoning_chars_max": reasoning_chars_max,
            "reasoning_rows": reasoning_rows,
        },
    }


def _game_seed(game_file: str) -> int | None:
    try:
        return int(Path(game_file).stem.split("_")[-1])
    except (ValueError, IndexError):
        return None


def _manifest_examples(env, rows: list[dict]) -> list[dict]:
    examples = []
    for row in rows:
        answer = int(row["answer"])
        game_file = env._game_files[answer]
        examples.append(
            {
                "answer": answer,
                "task": row.get("task"),
                "game_file": game_file,
                "game_seed": _game_seed(game_file),
            }
        )
    return examples


def _dataset_from_manifest(env, manifest_path: Path) -> Dataset:
    payload = json.loads(manifest_path.read_text())
    answers = payload.get("answers")
    if answers is None:
        answers = [ex["answer"] for ex in payload.get("examples", [])]
    if not answers:
        raise ValueError(f"Eval manifest {manifest_path} has no answers/examples.")

    rows_by_answer: dict[int, dict] = {}
    for ds in [env.dataset, getattr(env, "eval_dataset", None)]:
        if ds is None:
            continue
        for row in ds.to_list():
            rows_by_answer[int(row["answer"])] = row

    missing = [int(a) for a in answers if int(a) not in rows_by_answer]
    if missing:
        raise ValueError(
            f"Eval manifest {manifest_path} references {len(missing)} answer ids "
            f"that are not present in dataset/eval_dataset: {missing[:10]}"
        )

    return Dataset.from_list([rows_by_answer[int(a)] for a in answers])


def _select_eval_dataset(env, args) -> Dataset:
    manifest_path = Path(args.eval_set_json) if args.eval_set_json else None
    loaded_existing_manifest = False
    if manifest_path is not None and manifest_path.exists():
        selected = _dataset_from_manifest(env, manifest_path)
        loaded_existing_manifest = True
        print(
            f"[eval] loaded fixed eval set from {manifest_path} "
            f"({len(selected)} examples)",
            flush=True,
        )
    elif args.eval_source == "eval":
        if env.eval_dataset is None:
            raise ValueError(
                "Requested --eval-source eval, but this dataset has no "
                "eval_dataset split. Regenerate it with "
                "experiments/textworld_env/prepare_dataset.sh."
            )
        selected = env.eval_dataset
    elif args.eval_source == "train-shuffle":
        selected = env.dataset.shuffle(seed=args.seed)
    else:
        datasets = [env.dataset]
        if env.eval_dataset is not None:
            datasets.append(env.eval_dataset)
        selected = concatenate_datasets(datasets).shuffle(seed=args.seed)

    if args.num_examples > 0:
        offset = getattr(args, "example_offset", 0) or 0
        if len(selected) < offset + args.num_examples:
            raise ValueError(
                f"Requested {args.num_examples} eval examples (offset {offset}) "
                f"from {args.eval_source}, but only {len(selected)} are available. "
                "For the 100-question TextWorld eval set, regenerate the "
                "dataset with --eval-per-difficulty 20."
            )
        selected = selected.select(range(offset, offset + args.num_examples))

    if manifest_path is not None and not loaded_existing_manifest:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        rows = selected.to_list()
        payload = {
            "version": 1,
            "dataset": str(args.dataset),
            "source": args.eval_source,
            "seed": args.seed,
            "num_examples": len(rows),
            "answers": [int(row["answer"]) for row in rows],
            "examples": _manifest_examples(env, rows),
        }
        manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[eval] wrote fixed eval set manifest to {manifest_path}", flush=True)

    return selected


def _hard_success(reward: float | None, score: int | None, max_score: int | None, won: bool | None) -> bool:
    if won:
        return True
    if score is not None and max_score is not None:
        return int(score) >= int(max_score)
    return reward is not None and float(reward) >= 1.0


async def main():
    args = _args()
    if args.managed_context and args.padding_block_size <= 0:
        raise ValueError("--managed-context requires --padding-block-size > 0")
    if args.managed_context and not args.phase4_padding:
        raise ValueError("--managed-context requires --phase4-padding")

    # Must import kv_eviction BEFORE verifiers so the module-level
    # monkey-patches (AsyncCompletions interceptor, from_native_response
    # forwarding) are installed before verifiers' client is built.
    import kv_eviction  # noqa: F401 — side-effect import
    from kv_eviction.env import (
        configure_message_padding,
        get_managed_context_context_events,
        get_managed_context_recall_events,
        get_managed_context_stats,
        pop_markovian_stats,
        reset_managed_context_stats,
    )
    from kv_eviction.padding import (
        resolve_filler_token_id,
        resolve_im_end_token_id,
    )

    import verifiers as vf

    # Dummy API key — vLLM accepts any bearer token but the openai-python
    # client insists on one being set.
    os.environ.setdefault(args.api_key_var, "dummy")

    # Install block-aligned message padding so the client's outgoing chat
    # completion requests have filler tokens after each <|im_end|> that
    # land the next turn on a block boundary. Server-side
    # compaction_assume_aligned_turn_boundaries=true relies on this.
    # When block_size=0, padding is disabled (full-context baseline).
    if args.padding_block_size > 0:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
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
    else:
        print("[eval] block-aligned padding OFF (full-context baseline)", flush=True)

    trace_path: Path | None = None
    if os.environ.get("KVE_TRACE_OPENAI_CLIENT", "0") == "1":
        raw_trace_path = os.environ.get("KVE_TRACE_OPENAI_CLIENT_PATH")
        trace_path = (
            Path(raw_trace_path)
            if raw_trace_path
            else Path(args.output_json).with_suffix(".openai_trace.jsonl")
        )
        _install_openai_client_trace(trace_path)
        print(f"[eval] OpenAI client trace ON: {trace_path}", flush=True)

    # Load train + held-out eval splits, then replace env.eval_dataset with
    # the fixed subset used by this run. This keeps TextWorldEnv's game-file
    # and max-score metadata while making the evaluated question set explicit.
    if args.env_type == "counting":
        from counting_env import CountingEnv

        env = CountingEnv(
            num_examples=args.num_examples,
            max_episode_steps=args.max_episode_steps,
        )
        selected_eval = env.eval_dataset
    else:
        env = vf.load_environment(
            "textworld-env",
            dataset_path=args.dataset,
            max_episode_steps=args.max_episode_steps,
            num_train_examples=None,
            num_eval_examples=None,
            seed=args.seed,
            system_prompt_variant=args.system_prompt_variant,
        )
        selected_eval = _select_eval_dataset(env, args)
        env.eval_dataset = selected_eval
    total_rows = len(env.dataset)
    total_eval_rows = len(env.eval_dataset)
    print(
        f"[eval] train_rows={total_rows} selected_eval_rows={total_eval_rows} "
        f"source={args.eval_source} seed={args.seed}",
        flush=True,
    )

    client_config = vf.ClientConfig(
        client_type="openai_chat_completions",
        api_base_url=args.base_url,
        api_key_var=args.api_key_var,
        timeout=args.client_timeout,
        connect_timeout=args.client_connect_timeout,
        max_retries=args.client_max_retries,
    )

    sampling_args = {
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "presence_penalty": args.presence_penalty,
    }
    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if args.min_p is not None:
        extra_body["min_p"] = args.min_p
    if args.repetition_penalty is not None:
        extra_body["repetition_penalty"] = args.repetition_penalty
    if args.min_tokens is not None:
        extra_body["min_tokens"] = args.min_tokens
    if args.ignore_eos:
        extra_body["ignore_eos"] = True
    if args.chat_template_enable_thinking is not None:
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": args.chat_template_enable_thinking
        }
    if extra_body:
        sampling_args["extra_body"] = extra_body

    print(
        f"[eval] dataset={args.dataset} model={args.model} base_url={args.base_url}",
        flush=True,
    )
    print(f"[eval] sampling_args={sampling_args}", flush=True)
    print(
        f"[eval] num_examples={args.num_examples} max_episode_steps={args.max_episode_steps} "
        f"max_concurrent={args.max_concurrent} "
        f"system_prompt_variant={args.system_prompt_variant}",
        flush=True,
    )

    t0 = time.time()
    results = await env.evaluate(
        client=client_config,
        model=args.model,
        sampling_args=sampling_args,
        num_examples=len(selected_eval),
        rollouts_per_example=1,
        max_concurrent=args.max_concurrent,
        state_columns=(
            []
            if args.env_type == "counting"
            else ["tw_score", "tw_max_score", "tw_done", "tw_won"]
        ),
        save_results=False,
    )
    elapsed = time.time() - t0

    # GenerateOutputs is a TypedDict at runtime — access via dict keys.
    outputs = results.get("outputs", []) if isinstance(results, dict) else []

    rewards: list[float] = []
    trace_rows: list[dict] = []
    completion_lens: list[int] = []
    compaction_event_counts: list[int] = []
    trajectory_lens: list[int] = []
    finished: list[bool] = []
    truncated: list[bool] = []
    tw_scores: list[int] = []
    tw_max_scores: list[int] = []
    hard_successes: list[bool] = []
    rollout_rows: list[dict] = []

    for out in outputs:
        r = out.get("reward")
        reward = float(r) if r is not None else None
        if r is not None:
            rewards.append(reward)
        finished.append(bool(out.get("is_completed", False)))
        truncated.append(bool(out.get("is_truncated", False)))
        # state_columns promote per-rollout state entries to top-level output keys.
        score = None
        max_score = None
        if "tw_score" in out:
            score = int(out["tw_score"] or 0)
            tw_scores.append(score)
        if "tw_max_score" in out:
            max_score = int(out["tw_max_score"] or 1)
            tw_max_scores.append(max_score)
        won = bool(out.get("tw_won", False))
        hard = _hard_success(reward, score, max_score, won)
        hard_successes.append(hard)

        traj = out.get("trajectory") or []
        trajectory_lens.append(len(traj))

        total_events = 0
        last_step_completion = 0
        for step in traj:
            if not isinstance(step, dict):
                continue
            extras = step.get("extras") or {}
            events = extras.get("compaction_events") or []
            total_events += len(events)
            cids = step.get("completion_ids") or []
            if cids:
                last_step_completion = len(cids)
        compaction_event_counts.append(total_events)
        completion_lens.append(last_step_completion)
        rollout_rows.append(
            {
                "example_id": out.get("example_id"),
                "task": out.get("task"),
                "answer": out.get("answer"),
                "reward": reward,
                "tw_score": score,
                "tw_max_score": max_score,
                "tw_won": won,
                "hard_success": hard,
                "is_completed": bool(out.get("is_completed", False)),
                "is_truncated": bool(out.get("is_truncated", False)),
                "compaction_events": total_events,
                "trajectory_len": len(traj),
                "completion_len": last_step_completion,
            }
        )

        if args.save_traces:
            trace_rows.append(
                {
                    "example_id": out.get("example_id"),
                    "task": out.get("task"),
                    "answer": out.get("answer"),
                    "reward": reward,
                    "tw_score": score,
                    "tw_max_score": max_score,
                    "tw_won": won,
                    "hard_success": hard,
                    "is_completed": bool(out.get("is_completed", False)),
                    "is_truncated": bool(out.get("is_truncated", False)),
                    "num_env_steps": len(traj),
                    # Full conversation: system+first user prompt and every
                    # assistant/user turn (assistant messages keep <think>).
                    "prompt": out.get("prompt"),
                    "completion": out.get("completion"),
                }
            )

    def _stats(xs: list[float]) -> dict:
        if not xs:
            return {"n": 0}
        return {
            "n": len(xs),
            "mean": statistics.fmean(xs),
            "min": min(xs),
            "max": max(xs),
            "median": statistics.median(xs),
        }

    summary = {
        "dataset": args.dataset,
        "model": args.model,
        "base_url": args.base_url,
        "num_examples": len(selected_eval),
        "eval_source": args.eval_source,
        "eval_set_json": args.eval_set_json,
        "max_episode_steps": args.max_episode_steps,
        "max_concurrent": args.max_concurrent,
        "client_timeout": args.client_timeout,
        "client_connect_timeout": args.client_connect_timeout,
        "client_max_retries": args.client_max_retries,
        "system_prompt_variant": args.system_prompt_variant,
        "sampling_args": sampling_args,
        "elapsed_sec": round(elapsed, 2),
        "openai_trace": _summarize_openai_client_trace(trace_path),
        "reward": _stats(rewards),
        "success_rate": statistics.fmean(rewards) if rewards else 0.0,
        "hard_success_rate": (
            sum(hard_successes) / len(hard_successes) if hard_successes else 0.0
        ),
        "hard_success_count": int(sum(hard_successes)),
        "finished_rate": (sum(finished) / len(finished)) if finished else 0.0,
        "truncated_rate": (sum(truncated) / len(truncated)) if truncated else 0.0,
        "tw_score_mean": (statistics.fmean(tw_scores) if tw_scores else 0.0),
        "tw_max_score_mean": (statistics.fmean(tw_max_scores) if tw_max_scores else 0.0),
        "compaction_events_per_rollout": _stats([float(x) for x in compaction_event_counts]),
        "trajectory_len": _stats([float(x) for x in trajectory_lens]),
        "completion_len": _stats([float(x) for x in completion_lens]),
        "padding_block_size": args.padding_block_size,
        "phase4_padding": args.phase4_padding,
        "managed_context": {
            "enabled": args.managed_context,
            "recall_max_spans": args.managed_context_recall_max_spans,
            "index_enabled": args.managed_context_index,
            "index_max_entries": args.managed_context_index_max_entries,
            "restore_mode": args.managed_context_restore_mode,
            "force_restore": args.managed_context_force_restore,
            "force_span_policy": args.managed_context_force_span_policy,
            "require_retrieve": args.managed_context_require_retrieve,
            "recall_mode": args.managed_context_recall_mode,
            "compaction_max_turns": args.managed_context_compaction_max_turns,
            "turns_last_kept": args.managed_context_turns_last_kept,
            "stats": get_managed_context_stats(),
            "context_events": get_managed_context_context_events(),
            "recall_events": get_managed_context_recall_events(),
        },
        "markovian": {
            "enabled": os.environ.get("KV_EVICTION_MARKOVIAN_ENABLED") == "1",
            "max_turns": os.environ.get("KV_EVICTION_MARKOVIAN_MAX_TURNS"),
            "stride": os.environ.get("KV_EVICTION_MARKOVIAN_STRIDE"),
            "anchor_turns": os.environ.get("KV_EVICTION_MARKOVIAN_ANCHOR_TURNS"),
            "anchor_policy": os.environ.get(
                "KV_EVICTION_MARKOVIAN_ANCHOR_POLICY"
            ),
            "stats": pop_markovian_stats(),
        },
        "rollouts": rollout_rows,
    }
    # Persistent-session client stats (KVE_SESSION_MODE=1 parallel arm).
    try:
        from kv_eviction.env import get_session_client_stats

        session_client_stats = dict(get_session_client_stats())
    except Exception:
        session_client_stats = {}
    summary["session_mode"] = bool(session_client_stats.get("enabled", False))
    summary["session_fallback_count"] = int(
        session_client_stats.get("fallback_count", 0) or 0
    )
    summary["session_client"] = session_client_stats

    trace_summary = summary.get("openai_trace") or {}
    trace_error_types = trace_summary.get("error_types") or {}
    summary["error_count"] = int(trace_summary.get("errors", 0) or 0)
    summary["api_timeout_count"] = int(
        trace_error_types.get("APITimeoutError", 0) or 0
    )
    summary["non_error_count"] = int(trace_summary.get("done", 0) or 0)
    summary["clean_finished_count"] = int(sum(finished))

    per_task: dict[str, dict] = {}
    for task in sorted({r["task"] for r in rollout_rows}):
        task_rows = [r for r in rollout_rows if r["task"] == task]
        task_rewards = [r["reward"] for r in task_rows if r["reward"] is not None]
        per_task[str(task)] = {
            "n": len(task_rows),
            "success_rate": statistics.fmean(task_rewards) if task_rewards else 0.0,
            "hard_success_rate": (
                sum(bool(r["hard_success"]) for r in task_rows) / len(task_rows)
                if task_rows
                else 0.0
            ),
            "hard_success_count": sum(bool(r["hard_success"]) for r in task_rows),
        }
    summary["per_task"] = per_task

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    if args.save_traces:
        traces_path = Path(args.save_traces)
        traces_path.parent.mkdir(parents=True, exist_ok=True)

        def _jsonable(o):
            # verifiers Messages are pydantic models (SystemMessage, etc.).
            if hasattr(o, "model_dump"):
                return o.model_dump()
            if hasattr(o, "dict"):
                return o.dict()
            return str(o)

        with open(traces_path, "w") as f:
            for row in trace_rows:
                f.write(json.dumps(row, default=_jsonable) + "\n")
        summary["traces_path"] = str(traces_path)
        summary["traces_count"] = len(trace_rows)
        print(f"[eval] wrote {len(trace_rows)} full traces to {traces_path}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("[eval] summary:", flush=True)
    for k, v in summary.items():
        if k == "rollouts":
            print(f"  {k}: {len(v)} rows", flush=True)
            continue
        if k == "managed_context" and isinstance(v, dict):
            compact_v = dict(v)
            context_events = compact_v.pop("context_events", [])
            recall_events = compact_v.pop("recall_events", [])
            print(f"  {k}: {compact_v}", flush=True)
            print(f"  managed_context.context_events: {len(context_events)} rows", flush=True)
            print(f"  managed_context.recall_events: {len(recall_events)} rows", flush=True)
            continue
        print(f"  {k}: {v}", flush=True)
    print(f"[eval] wrote {out_path}", flush=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
