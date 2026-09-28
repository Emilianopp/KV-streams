#!/usr/bin/env python3
from __future__ import annotations

import argparse
import statistics
from typing import Any

from transformers import AutoTokenizer

from reasoning_rag_env.env import (
    DEFAULT_SYSTEM_PROMPT,
    build_prompt_messages,
    load_canonical_rows,
)


def _message_token_count(tokenizer: Any, messages: list[dict[str, str]]) -> int:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
            )
            return len(ids)
        except Exception:
            pass
    text = "\n\n".join(f"{message['role']}: {message['content']}" for message in messages)
    return len(tokenizer.encode(text, add_special_tokens=True))


def _quantile(values: list[int], q: float) -> int:
    if not values:
        return 0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whether a reasoning-RAG dataset is long enough and labeled enough for AM + RL."
    )
    parser.add_argument("--source", choices=["jsonl", "hf"], default="jsonl")
    parser.add_argument("--local-jsonl-path")
    parser.add_argument("--hf-dataset-name")
    parser.add_argument("--hf-dataset-config")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--default-benchmark", default="reasoning-rag")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--max-examples", type=int, default=200)
    parser.add_argument("--max-documents", type=int, default=32)
    parser.add_argument("--max-document-chars", type=int, default=4000)
    parser.add_argument("--require-gold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--threshold",
        type=int,
        action="append",
        default=[4096, 8192, 16384, 32768, 65536],
        help="Token-length threshold to report. May be repeated.",
    )
    args = parser.parse_args()

    rows = load_canonical_rows(
        source=args.source,
        local_jsonl_path=args.local_jsonl_path,
        hf_dataset_name=args.hf_dataset_name,
        hf_dataset_config=args.hf_dataset_config,
        hf_split=args.hf_split,
        max_documents=args.max_documents,
        max_document_chars=args.max_document_chars,
        require_gold=args.require_gold,
        default_benchmark=args.default_benchmark,
        max_examples=args.max_examples,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    token_counts: list[int] = []
    doc_counts: list[int] = []
    answer_labeled = 0
    retrieval_labeled = 0
    for row in rows:
        messages = build_prompt_messages(row, system_prompt=DEFAULT_SYSTEM_PROMPT)
        token_counts.append(_message_token_count(tokenizer, messages))
        doc_counts.append(len(row["documents"]))
        if row["answers"]:
            answer_labeled += 1
        if row["gold_doc_ids"]:
            retrieval_labeled += 1

    if not token_counts:
        raise SystemExit("No rows loaded.")

    print(f"rows={len(rows)}")
    print(f"model={args.model}")
    print(f"answer_labeled={answer_labeled}/{len(rows)} ({answer_labeled / len(rows):.1%})")
    print(f"retrieval_labeled={retrieval_labeled}/{len(rows)} ({retrieval_labeled / len(rows):.1%})")
    print(f"documents_per_prompt_mean={statistics.mean(doc_counts):.2f}")
    print(
        "prompt_tokens "
        f"min={min(token_counts)} "
        f"p50={_quantile(token_counts, 0.50)} "
        f"p90={_quantile(token_counts, 0.90)} "
        f"p99={_quantile(token_counts, 0.99)} "
        f"max={max(token_counts)}"
    )
    for threshold in sorted(set(args.threshold)):
        count = sum(value >= threshold for value in token_counts)
        print(f"prompts_at_least_{threshold}_tokens={count}/{len(rows)} ({count / len(rows):.1%})")


if __name__ == "__main__":
    main()
