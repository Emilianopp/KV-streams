from __future__ import annotations

import argparse
import asyncio

from .actions import CRAFTER_ACTIONS
from .env import load_environment
from .rewards import verified_crafter_reward


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny BALROG Crafter environment smoke test.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = load_environment(num_episodes=args.episodes, max_steps=args.max_steps, seed=args.seed)
    row = env.dataset[0]
    state = dict(row)
    state = await env.setup_state(state)
    messages = list(state["prompt"])

    for index, action in enumerate(CRAFTER_ACTIONS[: args.max_steps], start=1):
        messages.append({"role": "assistant", "content": f"<action>{action}</action>"})
        response = await env.env_response(messages, state)
        messages.extend(response)
        print(f"{index:02d}: {action} reward_total={state['native_reward_total']:.3f} done={state['done']}")
        if state["done"]:
            break

    await env.cleanup_crafter(state)
    print("verified_reward=", verified_crafter_reward(state=state))


if __name__ == "__main__":
    asyncio.run(main())
