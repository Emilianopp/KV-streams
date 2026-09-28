#!/usr/bin/env python3
"""Direct HTTP test for protected-prefix compaction.

Sends a multi-turn chat conversation to a vLLM server with compaction
enabled, generating enough tokens to trigger compaction. Inspects the
response for compaction_events to verify turn-based eviction works.

Usage:
    python scripts/test_compaction_direct.py [--base-url http://localhost:8000/v1]
"""

import argparse
import json
import time
import requests


def build_long_conversation(num_turns: int = 8) -> list[dict]:
    """Build a multi-turn conversation long enough to trigger compaction.

    System prompt (~100 tokens) + many turns of obs/action pairs to fill
    up context and trigger the sliding window eviction.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant navigating a grid world. "
                "In each turn you receive an observation describing your "
                "surroundings and must choose an action. Valid actions are: "
                "turn left, turn right, go forward, pick up, drop, toggle. "
                "Always respond with exactly one action and brief reasoning."
            ),
        }
    ]

    # Build fake multi-turn conversation history to inflate context.
    # Each observation is padded to ~200 tokens so the total prompt
    # exceeds the compaction window (2048) and triggers eviction.
    base_observations = [
        "You see a red ball to your left. There is a grey wall in front of you. Behind you is an open door.",
        "You turned left. Now you see the red ball directly ahead. The grey wall is to your right.",
        "You moved forward. The red ball is right in front of you. You can pick it up.",
        "You picked up the red ball. You are holding a red ball. You see an open door ahead.",
        "You moved forward through the door. You are in a new room. There is a blue key on the floor to your right.",
        "You turned right. The blue key is directly ahead of you on the floor.",
        "You moved forward. The blue key is right in front of you.",
        "You picked up the blue key. You see a locked blue door to your left.",
        "You turned left. The locked blue door is ahead. You have a blue key and a red ball.",
        "You used the blue key to unlock the door. The door is now open. Ahead is a green goal square.",
        "You moved forward. You are standing on the green goal square! Task nearly complete.",
        "A new task begins. You see a yellow box to your right and a purple door ahead. The door is locked.",
    ]

    # Pad observations so each turn contributes ~200 tokens
    padding = (
        " The room is 6x6 tiles. The walls are grey stone. "
        "The floor is dark brown wood. You can see a window "
        "on the north wall showing blue sky. There are cobwebs "
        "in the corners. A torch on the east wall provides dim "
        "yellow light. You hear distant footsteps echoing through "
        "the corridor behind you. The air smells slightly musty. "
        "Your inventory shows the items you have collected so far."
    )
    observations = [obs + padding for obs in base_observations]

    actions = [
        "turn left",
        "go forward",
        "pick up",
        "go forward",
        "turn right",
        "go forward",
        "pick up",
        "turn left",
        "toggle",
        "go forward",
        "go forward",
        "turn right",
    ]

    for i in range(min(num_turns, len(observations))):
        messages.append({"role": "user", "content": f"Observation: {observations[i]}"})
        messages.append({
            "role": "assistant",
            "content": (
                f"I'll {actions[i]}. Based on the observation: "
                f"{observations[i][:80]}... This action is the best "
                f"choice because it moves us toward our goal. The room "
                f"layout suggests we should continue exploring. Let me "
                f"think about what objects I need and where to go next."
            ),
        })

    # Final turn: ask for one more action
    messages.append({
        "role": "user",
        "content": "Observation: You see a yellow box ahead and a locked purple door to your left. What do you do?",
    })

    return messages


def send_chat_request(
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> dict:
    """Send a chat completion request and return the full response."""
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--num-turns", type=int, default=10,
                        help="Number of conversation turns to include")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens to generate per response")
    parser.add_argument("--num-requests", type=int, default=3,
                        help="Number of independent requests to send")
    args = parser.parse_args()

    print(f"Sending {args.num_requests} requests with {args.num_turns} turns each")
    print(f"Server: {args.base_url}, Model: {args.model}")
    print(f"Max tokens per response: {args.max_tokens}")
    print()

    for req_idx in range(args.num_requests):
        messages = build_long_conversation(num_turns=args.num_turns)
        msg_token_est = sum(len(m["content"].split()) * 1.3 for m in messages)

        print(f"Request {req_idx + 1}/{args.num_requests}: "
              f"{len(messages)} messages, ~{int(msg_token_est)} estimated prompt tokens")

        t0 = time.time()
        try:
            response = send_chat_request(
                base_url=args.base_url,
                model=args.model,
                messages=messages,
                max_tokens=args.max_tokens,
            )
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        elapsed = time.time() - t0

        # Extract response info
        choice = response.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        usage = response.get("usage", {})

        print(f"  Time: {elapsed:.1f}s")
        print(f"  Usage: prompt={usage.get('prompt_tokens', '?')} "
              f"completion={usage.get('completion_tokens', '?')} "
              f"total={usage.get('total_tokens', '?')}")
        print(f"  Response preview: {content[:100]}...")

        # Check for compaction events in the response
        compaction_events = response.get("compaction_events")
        if compaction_events:
            print(f"  COMPACTION EVENTS: {len(compaction_events)}")
            for i, ev in enumerate(compaction_events):
                print(f"    [{i}] tokens_evicted={ev.get('tokens_evicted')} "
                      f"output_tokens={ev.get('num_output_tokens_at_compaction')} "
                      f"pos_offset={ev.get('position_offset_after')} "
                      f"prompt_tokens={ev.get('num_prompt_tokens', 'N/A')}")
        else:
            print("  No compaction events (prompt + completion may be under window size)")

        print()

    print("Done. Check vLLM server logs for [COMPACT] diagnostic lines.")


if __name__ == "__main__":
    main()
