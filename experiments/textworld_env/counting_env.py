"""Synthetic counting benchmark env for kv-eviction.

Purpose: compare full-context vs managed-context (CPU-offload) inference under a
CONTROLLED per-turn decode volume. The agent is asked to count upward each turn;
the *exact* number of decode tokens per turn is forced by the eval's sampling
params (min_tokens == max_tokens with ignore_eos), so the per-turn token count N
is a clean dial. Over T turns each rollout's KV grows by ~N+prompt per turn:

  - managed-context: old turns are evicted + offloaded to CPU server-side
    (compaction triggers on turn count) -> live GPU KV per rollout stays small.
  - full-context: all turns retained -> KV grows -> at high concurrency the
    collective KV exceeds the GPU pool -> vLLM preempts/recomputes -> throughput
    and effective concurrency collapse.

This isolates THROUGHPUT-under-KV-load (no recall is needed -- counting never
references old turns), which is exactly the regime where CPU offload should let
managed-context hold concurrency static while full-context caps out.

The env is deliberately trivial: it drives exactly `max_episode_steps` turns and
replies "start a new count". It needs no compaction-aware code -- the server
evicts on turn count and the kv_eviction client monkey-patches apply to any
verifiers MultiTurnEnv. The per-turn token volume is set by SAMPLING, not here.
"""

import os
import logging

import verifiers as vf
from datasets import Dataset
from verifiers.types import Messages, State

logger = logging.getLogger(__name__)

COUNTING_SYSTEM_PROMPT = (
    "You are a counting machine. When given a starting number, count upward from "
    "it, writing one number per line (for example '41\\n42\\n43\\n...'). Keep "
    "counting and do not stop until you are told to."
)


class CountingEnv(vf.MultiTurnEnv):
    """Drive T turns of forced-length counting generations (a throughput harness).

    The per-turn token volume is NOT set here -- it is forced by the eval's
    sampling params (min_tokens/max_tokens + ignore_eos). This env only supplies
    the counting prompts and runs exactly `max_episode_steps` turns.
    """

    def __init__(
        self,
        num_examples: int = 128,
        max_episode_steps: int = 8,
        start: int = 1,
        **kwargs,
    ):
        # Distinct per-example starts (default): identical streams across
        # games are a degenerate benchmark artifact — the prefix cache dedups
        # them onto SHARED physical blocks, and per-trace eviction rehash then
        # strips the single hash registration out from under sibling traces
        # (one block = one hash), forcing CPU reload ping-pong that no real
        # workload exhibits (found via soft-pin walkdiag 2026-06-10).
        # COUNTING_DISTINCT_START_STRIDE=0 restores legacy identical tasks.
        distinct_stride = int(
            os.environ.get("COUNTING_DISTINCT_START_STRIDE", "1009")
        )
        rows = [
            {
                "question": (
                    f"Begin counting upward from {start + i * distinct_stride}, "
                    "one number per line. Keep going until told to stop."
                ),
                "answer": i,
                "task": "count",
            }
            for i in range(max(1, num_examples))
        ]
        self._distinct_stride = distinct_stride
        dataset = Dataset.from_list(rows)

        # Parser/rubric are required by the verifiers API but irrelevant here
        # (this is a throughput harness, not a scored task). The reward is just a
        # sanity signal: fraction of the intended turns the rollout actually ran.
        parser = vf.XMLParser(fields=["x"], answer_field="x")
        rubric = vf.Rubric(parser=parser)

        async def turns_completed_reward(state: State, **kwargs) -> float:
            target = max(1, int(state.get("count_target_turns", 1)))
            return float(state.get("count_turns", 0)) / float(target)

        rubric.add_reward_func(turns_completed_reward)

        super().__init__(
            dataset=dataset,
            eval_dataset=dataset,
            max_turns=max_episode_steps,
            rubric=rubric,
            parser=parser,
            system_prompt=COUNTING_SYSTEM_PROMPT,
            message_type="chat",
        )
        self._start = int(start)

    async def setup_state(self, state: State) -> State:
        state["count_turns"] = 0
        state["count_next"] = int(self._start) + (
            int(state.get("answer") or 0) * self._distinct_stride
        )
        state["count_target_turns"] = int(self.max_turns)
        self._trace_metadata(state, 0)
        return state

    async def env_response(self, messages: Messages, state: State) -> Messages:
        # One env turn elapsed. Advance the start by a large stride so each turn
        # is fresh work (numbers never repeat); the volume itself is forced by
        # sampling, not by what the model emits.
        state["count_turns"] = int(state.get("count_turns", 0)) + 1
        nxt = int(state.get("count_next", self._start)) + 100000
        state["count_next"] = nxt
        self._trace_metadata(state, int(state["count_turns"]))
        recall_demand = ""
        if os.environ.get("COUNTING_RECALL_FIRST", "0") == "1":
            # Forces recall of EVICTED context: the very first start number
            # leaves the live window at the first compaction (turn ~6), so
            # answering requires restored KV (or dragged full history).
            recall_demand = (
                " First, repeat the very first number our count started "
                "from at the beginning of this conversation, then continue."
            )
        return [
            {
                "role": "user",
                "content": (
                    f"Stop. Now start a fresh count from {nxt}, one number per "
                    f"line, and keep going.{recall_demand}"
                ),
            }
        ]

    @vf.stop
    async def max_turns_reached(self, state: State) -> bool:
        if self.max_turns <= 0:
            return False
        return len(state.get("trajectory", [])) >= self.max_turns

    @vf.cleanup
    async def cleanup(self, state: State) -> None:
        return None

    @staticmethod
    def _trace_metadata(state: State, turn: int) -> None:
        try:
            from kv_eviction.env import set_phase4_rollout_metadata

            set_phase4_rollout_metadata(
                env="counting",
                example_id=state.get("answer"),
                game_id=state.get("answer"),
                task="count",
                current_turn=int(turn),
            )
        except Exception:
            pass


def load_environment(
    num_examples: int = 128,
    max_episode_steps: int = 8,
    start: int = 1,
    **kwargs,
) -> vf.Environment:
    """Entry point for `vf.load_environment("counting-env", ...)`."""
    return CountingEnv(
        num_examples=num_examples,
        max_episode_steps=max_episode_steps,
        start=start,
    )
