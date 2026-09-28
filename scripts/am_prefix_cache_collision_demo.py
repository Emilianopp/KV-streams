#!/usr/bin/env python3
"""Demonstrate why AM synthetic KV needs an AM-specific prefix-cache key.

This is a deterministic hash-level reproduction of the failure mode. It uses
the actual vLLM prefix-cache block hasher, but does not require GPUs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vllm"))

from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash


@dataclass
class DemoRequest:
    all_token_ids: list[int]
    attention_matching_prefix_cache_key: str | None
    attention_matching_prefix_cache_key_start: int
    attention_matching_prefix_cache_hash_start: int
    cache_salt: str | None = "prime-rl-policy-step:0"

    def __post_init__(self) -> None:
        self.block_hashes: list[bytes] = []
        self.mm_features: list[Any] = []
        self.lora_request = None
        self.prompt_embeds = None
        self._prompt_embeds_per_block_hashes: dict[tuple[int, int], bytes] = {}

    @property
    def num_tokens(self) -> int:
        return len(self.all_token_ids)


def _hash_tokens(tokens: list[int]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for token_id in tokens:
        digest.update(int(token_id).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _hashes_for(
    token_ids: list[int],
    *,
    block_size: int,
    hash_fn: Any,
    am_key: str | None,
    am_key_start: int,
    am_hash_start: int | None = None,
) -> list[bytes]:
    request = DemoRequest(
        all_token_ids=token_ids,
        attention_matching_prefix_cache_key=am_key,
        attention_matching_prefix_cache_key_start=am_key_start,
        attention_matching_prefix_cache_hash_start=(
            am_key_start if am_hash_start is None else am_hash_start
        ),
    )
    return get_request_block_hasher(block_size, hash_fn)(request)


def run_demo(block_size: int = 16) -> dict[str, Any]:
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)

    # Four full protected blocks. These should be identical and reusable.
    protected = [1000 + i for i in range(4 * block_size)]

    # Two different old histories that AM would summarize.
    old_a = [2000 + i for i in range(5 * block_size)]
    old_b = [3000 + i for i in range(5 * block_size)]

    # Same visible compacted representation after AM. These placeholder IDs are
    # not the identity of the synthetic KV.
    synthetic_placeholders = [0 for _ in range(block_size)]
    exact_recent_tail = [4000 + i for i in range(3 * block_size)]
    compacted_visible_tokens = protected + synthetic_placeholders + exact_recent_tail

    protected_blocks = len(protected) // block_size
    first_synthetic_block = protected_blocks
    first_tail_block = first_synthetic_block + len(synthetic_placeholders) // block_size

    unsafe_a_hashes = _hashes_for(
        compacted_visible_tokens,
        block_size=block_size,
        hash_fn=hash_fn,
        am_key=None,
        am_key_start=len(protected),
        am_hash_start=len(protected),
    )
    unsafe_b_hashes = _hashes_for(
        compacted_visible_tokens,
        block_size=block_size,
        hash_fn=hash_fn,
        am_key=None,
        am_key_start=len(protected),
        am_hash_start=len(protected),
    )

    safe_a_hashes = _hashes_for(
        compacted_visible_tokens,
        block_size=block_size,
        hash_fn=hash_fn,
        am_key=f"am-source:{_hash_tokens(old_a)}",
        am_key_start=len(protected),
        am_hash_start=0,
    )
    safe_b_hashes = _hashes_for(
        compacted_visible_tokens,
        block_size=block_size,
        hash_fn=hash_fn,
        am_key=f"am-source:{_hash_tokens(old_b)}",
        am_key_start=len(protected),
        am_hash_start=0,
    )

    unsafe_cache = {
        unsafe_a_hashes[first_synthetic_block]: "request-A synthetic KV"
    }
    safe_cache = {
        safe_a_hashes[first_synthetic_block]: "request-A synthetic KV"
    }

    return {
        "block_size": block_size,
        "protected_blocks": protected_blocks,
        "unsafe": {
            "protected_block_hash_equal": (
                unsafe_a_hashes[0] == unsafe_b_hashes[0]
            ),
            "synthetic_block_hash_equal": (
                unsafe_a_hashes[first_synthetic_block]
                == unsafe_b_hashes[first_synthetic_block]
            ),
            "tail_block_hash_equal": (
                unsafe_a_hashes[first_tail_block] == unsafe_b_hashes[first_tail_block]
            ),
            "request_b_lookup": unsafe_cache.get(
                unsafe_b_hashes[first_synthetic_block],
                "miss",
            ),
        },
        "safe_am_keyed": {
            "protected_block_hash_equal": safe_a_hashes[0] == safe_b_hashes[0],
            "synthetic_block_hash_equal": (
                safe_a_hashes[first_synthetic_block]
                == safe_b_hashes[first_synthetic_block]
            ),
            "tail_block_hash_equal": (
                safe_a_hashes[first_tail_block] == safe_b_hashes[first_tail_block]
            ),
            "request_b_lookup": safe_cache.get(
                safe_b_hashes[first_synthetic_block],
                "miss",
            ),
        },
        "interpretation": (
            "Unsafe ordinary token hashes make request B hit request A's "
            "synthetic KV because both compacted prompts expose the same "
            "placeholder token IDs. Full AM-keyed hashes split the whole "
            "compressed prompt namespace by compacted source state, so AM "
            "runs do not fall back to ordinary prefix-cache identity."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--block-size", type=int, default=16)
    args = parser.parse_args()

    result = run_demo(block_size=args.block_size)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
