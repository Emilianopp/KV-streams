"""TextWorld environment for kv-eviction multi-turn RL training.

Plain `vf.MultiTurnEnv` subclass. Compaction (turn-based eviction) and
block-aligned message padding are handled transparently by
`src/kv_eviction/env.py`'s module-level monkey-patches — this file does NOT
import or reference compaction in any way.

Ported from mkv-rl/experiments/textworld_rl/textworld_env.py with one
surgical change: game file paths are resolved RELATIVE to `dataset_path`
so the dataset directory is relocatable across machines (the original
stored absolute `/pscratch/...` paths in metadata.json).

Supports all bundled TextWorld challenge families. Each rollout plays one
game episode; the final normalized game score becomes the reward.
"""

import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

import textworld
from datasets import Dataset, load_from_disk

import verifiers as vf
from verifiers.types import Messages, State

logger = logging.getLogger(__name__)

ALL_GAMES_SYSTEM_PROMPT = """You are playing a TextWorld interactive-fiction game. Infer the objective from the opening instructions and observations, then maximize the score until the game ends.

POSSIBLE OBJECTIVES:
- Find and take a coin or a requested treasure.
- Explore rooms, unlock or open passages, and manipulate household objects as instructed.
- Read a cookbook, gather and prepare ingredients, prepare the meal, and eat it.

GENERAL COMMANDS (use exact object names and valid game syntax):
- look; inventory; examine [object]
- go north/south/east/west
- open/close [object]; unlock/lock [object] with [key]
- take [object]; take [object] from [container]; drop [object]
- put [object] on [supporter]; insert [object] into [container]
- eat [object]

COOKING COMMANDS WHEN A RECIPE IS PRESENT:
- examine cookbook
- slice/dice/chop [ingredient] with knife
- cook [ingredient] with oven/stove/BBQ
- prepare meal; eat meal

IMPORTANT RULES:
- Follow the stated objective. If it is hidden or incomplete, explore and use score changes to infer progress.
- Use exact names from observations. Examine or inventory objects when uncertain.
- Track visited rooms, object locations, inventory, locked doors, and completed steps.
- Do not repeat an invalid action without changing your approach.
- For cooking, read the recipe first, match preparation and appliance exactly, and do not cook an ingredient twice.

Respond with your reasoning, then exactly one game command in this format:
<action>your command here</action>"""

SYSTEM_PROMPT = """\
You are playing a text-based cooking game. Your goal is to prepare a meal by following a recipe.

GAME OVERVIEW:
1. Find and read the cookbook to learn the recipe
2. Gather the required ingredients (from fridge, counter, etc.)
3. Prepare ingredients as directed (slice, dice, or chop with knife)
4. Cook ingredients as directed (using the correct appliance)
5. Run "prepare meal" to assemble the dish
6. Run "eat meal" to finish

COMMANDS (use exact syntax):
- look — describe current room
- inventory — list items you are carrying
- examine [object] — inspect something (e.g. "examine cookbook")
- go [direction] — move (north/south/east/west)
- open [container] — open fridge, oven, door, etc.
- take [object] from [container] — pick up item (e.g. "take carrot from fridge")
- take [object] — pick up item from the room
- drop [object] — drop item from inventory
- slice/dice/chop [object] with knife — prepare ingredient (take knife first!)
- cook [object] with oven — ROASTS the ingredient
- cook [object] with stove — FRIES the ingredient
- cook [object] with BBQ — GRILLS the ingredient
- prepare meal — assemble the dish (only works when all ingredients are correctly prepared)
- eat meal — eat the finished meal to win

IMPORTANT RULES:
- Read the recipe FIRST. It tells you exactly what ingredients and steps are needed.
- Use the EXACT object name shown in the game (e.g. "sliced carrot", not just "carrot").
- Match the cooking method to the right appliance: roast=oven, fry=stove, grill=BBQ.
- Cook each ingredient only ONCE. Cooking twice burns it and you lose!
- You must take the knife before you can slice/dice/chop.
- Items in the fridge may already be pre-sliced/diced/chopped. Check the recipe to see what's still needed.

Respond with your reasoning, then your action in this format:
<action>your command here</action>"""

COMPACT_SYSTEM_PROMPT = """\
You are playing a text-based cooking game. Your goal is to follow the recipe, prepare the meal, and eat it.

Reply with exactly one game command inside XML:
<action>your command here</action>

Useful commands include: look, inventory, examine cookbook, go north/south/east/west, open container, take object, take object from container, slice/dice/chop object with knife, cook object with oven/stove/BBQ, prepare meal, eat meal.

Read the cookbook first. Use exact object names from the observation."""

THINKING_SYSTEM_PROMPT = """\
You are playing a text-based cooking game. Your goal is to follow the recipe, prepare the meal, and eat it.

Think carefully before each move. Reason about the recipe, current room, inventory, object locations, and the next required preparation or cooking step. After reasoning, reply with exactly one game command inside XML:
<action>your command here</action>

Useful commands include: look, inventory, examine cookbook, go north/south/east/west, open container, take object, take object from container, slice/dice/chop object with knife, cook object with oven/stove/BBQ, prepare meal, eat meal.

Read the cookbook first. Use exact object names from the observation. The final visible answer must contain the action tag."""

# Lock to serialize textworld.start() calls within a worker process.
# TextWorld's tatsu parser uses a module-level singleton (_PARSER) whose
# internal _rule_stack is not thread-safe. Without this lock, concurrent
# run_in_executor threads corrupt the parser state.
_tw_start_lock = threading.Lock()


def _message_field(message: Any, field: str, default: Any = None) -> Any:
    if isinstance(message, dict):
        return message.get(field, default)
    return getattr(message, field, default)


def _reset_tatsu_parser():
    """Reset the tatsu parser singleton after fork.

    When the orchestrator forks env_worker subprocesses, the tatsu parser
    singleton is inherited from the parent. Its internal _rule_stack (a list)
    can be in a dirty state. Creating a fresh instance avoids corruption.
    """
    try:
        import textworld.logic as twl
        from textworld.logic.parser import GameLogicParser
        twl._PARSER = GameLogicParser()
    except (ImportError, AttributeError):
        pass


def _start_game(game_file: str):
    """Start a TextWorld game with thread-safe parser access.

    Serializes textworld.start() calls to avoid tatsu parser corruption
    from concurrent threads in the same worker process.
    """
    request_infos = textworld.EnvInfos(
        score=True, max_score=True, won=True, description=True,
        inventory=True, admissible_commands=False,
    )
    with _tw_start_lock:
        _reset_tatsu_parser()
        env = textworld.start(game_file, request_infos)
        game_state = env.reset()
    return env, game_state


def _set_kv_eviction_textworld_metadata(state: State) -> None:
    """Expose TextWorld rollout metadata to kv-eviction tracing when enabled."""
    try:
        from kv_eviction.env import set_phase4_rollout_metadata
    except Exception:
        return
    game_id = state.get("answer")
    set_phase4_rollout_metadata(
        env="textworld",
        example_id=state.get("example_id", game_id),
        game_id=game_id,
        task=state.get("task"),
        current_turn=int(state.get("tw_turn", 0)),
    )


def _messages_to_plain_dicts(messages: Messages) -> list[dict]:
    out: list[dict] = []
    for message in messages:
        if isinstance(message, dict):
            role = message.get("role")
            content = message.get("content")
        else:
            role = getattr(message, "role", None)
            content = getattr(message, "content", None)
        if isinstance(role, str) and isinstance(content, str):
            out.append({"role": role, "content": content})
    return out


def _build_kv_eviction_preobs_memory_message(messages: Messages) -> str | None:
    try:
        from kv_eviction.env import build_managed_context_pre_observation_message
    except Exception:
        return None
    return build_managed_context_pre_observation_message(
        _messages_to_plain_dicts(messages)
    )


def _resolve_game_files(dataset_path: Path, raw_game_files: list[str]) -> list[str]:
    """Resolve each metadata.json game file path against the dataset dir.

    Supports three input forms:
    1. Relative path ('games/game_00042.z8') — joined with dataset_path.
    2. Absolute path that still exists on this box — used verbatim (legacy).
    3. Absolute path from another box — basename is rebased under
       dataset_path/games/ so a copied dataset still works.

    Raises FileNotFoundError if a resolved path doesn't exist on disk.
    """
    resolved: list[str] = []
    for raw in raw_game_files:
        p = Path(raw)
        # Form 1: relative
        if not p.is_absolute():
            candidate = dataset_path / p
            if candidate.exists():
                resolved.append(str(candidate))
                continue
        # Form 2: absolute and still valid
        if p.is_absolute() and p.exists():
            resolved.append(str(p))
            continue
        # Form 3: absolute-from-other-box, rebase by basename
        candidate = dataset_path / "games" / p.name
        if candidate.exists():
            resolved.append(str(candidate))
            continue
        raise FileNotFoundError(
            f"Could not resolve game file for entry {raw!r}. "
            f"Tried relative, absolute, and rebased forms under {dataset_path}."
        )
    return resolved


class TextWorldEnv(vf.MultiTurnEnv):
    """TextWorld interactive fiction environment for RL training.

    Subclasses `vf.MultiTurnEnv`. The base class handles turn-based rollouts
    (alternating model generation and env_response), token masking, and
    integration with verifiers' async client. This class only implements
    the game-stepping logic plus subprocess cleanup.
    """

    def __init__(
        self,
        dataset_path: str,
        max_episode_steps: int = 50,
        num_train_examples: Optional[int] = None,
        num_eval_examples: Optional[int] = None,
        seed: int = 42,
        include_initial_obs_in_system: bool = False,
        system_prompt_variant: str = "default",
    ):
        self.max_rollout_seq_len: int | None = None
        self._include_initial_obs_in_system = include_initial_obs_in_system
        if system_prompt_variant == "all-games":
            system_prompt = ALL_GAMES_SYSTEM_PROMPT
        elif system_prompt_variant == "default":
            system_prompt = SYSTEM_PROMPT
        elif system_prompt_variant == "compact":
            system_prompt = COMPACT_SYSTEM_PROMPT
        elif system_prompt_variant == "thinking":
            system_prompt = THINKING_SYSTEM_PROMPT
        else:
            raise ValueError(
                "system_prompt_variant must be 'all-games', 'default', "
                "'compact', or 'thinking', got "
                f"{system_prompt_variant!r}"
            )
        dataset_path = Path(dataset_path)

        # Load pre-generated dataset and metadata
        with open(dataset_path / "metadata.json") as f:
            meta = json.load(f)

        self._game_files = _resolve_game_files(dataset_path, meta["game_files"])
        self._max_scores = meta["max_scores"]  # list of max scores per game

        hf_ds = load_from_disk(str(dataset_path / "dataset"))
        rows = [dict(row) for row in hf_ds]

        eval_path = dataset_path / "eval_dataset"
        eval_rows_all: list[dict] = []
        if eval_path.exists():
            eval_rows_all = [dict(row) for row in load_from_disk(str(eval_path))]

        # Split train/eval
        num_train = num_train_examples if num_train_examples is not None else meta.get("num_train", len(rows))
        num_eval = num_eval_examples if num_eval_examples is not None else meta.get("num_eval", 0)
        train_rows = rows[:num_train]
        if num_eval > 0 and eval_rows_all:
            if len(eval_rows_all) < num_eval:
                raise ValueError(
                    "Requested "
                    f"num_eval_examples={num_eval}, but {dataset_path / 'eval_dataset'} "
                    f"contains only {len(eval_rows_all)} rows. Regenerate the "
                    "TextWorld dataset with enough held-out games."
                )
            eval_rows = eval_rows_all[:num_eval]
        elif num_eval > 0:
            # Legacy datasets saved train+eval in a single HF dataset. Modern
            # datasets save held-out rows under eval_dataset/.
            if len(rows) < num_train + num_eval:
                raise ValueError(
                    "Requested "
                    f"num_train_examples={num_train} and num_eval_examples={num_eval}, "
                    f"but {dataset_path / 'dataset'} contains only {len(rows)} rows."
                )
            eval_rows = rows[num_train : num_train + num_eval]
        else:
            eval_rows = []

        dataset = Dataset.from_list(train_rows) if train_rows else None
        eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None

        if dataset is None and eval_dataset is None:
            raise ValueError(
                "Both train and eval splits are empty. "
                f"num_train={num_train}, num_eval={num_eval}, total_rows={len(rows)}."
            )
        # verifiers requires a non-None `dataset` even for eval-only use.
        if dataset is None:
            dataset = eval_dataset

        # Reward: normalized terminal game score
        parser = vf.XMLParser(fields=["action"], answer_field="action")
        rubric = vf.Rubric(parser=parser)

        async def game_score_reward(state: State, **kwargs) -> float:
            score = state.get("tw_score")
            if score is None:
                score = 0
            max_score = state.get("tw_max_score", 1)
            reward = float(score) / max(float(max_score), 1.0)
            if reward > 0:
                logger.warning(f"REWARD: {score}/{max_score} = {reward:.3f}")
            return reward

        rubric.add_reward_func(game_score_reward)

        super().__init__(
            dataset=dataset,
            eval_dataset=eval_dataset,
            max_turns=max_episode_steps,
            rubric=rubric,
            parser=parser,
            system_prompt=system_prompt,
            message_type="chat",
        )

    def set_max_seq_len(self, max_seq_len: int | None) -> None:
        """Apply one cumulative token limit to full-context and KV rollouts."""
        self.max_rollout_seq_len = max_seq_len
        super().set_max_seq_len(max_seq_len)

    async def setup_state(self, state: State) -> State:
        """Create per-rollout TextWorld game instance on state dict.

        State lives on the dict (not self) because run_group() uses
        asyncio.gather for concurrent rollouts — shared state on self
        would corrupt across rollouts.

        Uses textworld.start() (not raw jericho.FrotzEnv) for proper score
        tracking. Fork-safety:
          1. _reset_tatsu_parser() resets the module singleton after fork
          2. _tw_start_lock serializes parser access across threads
        """
        game_idx = int(state["answer"])
        game_file = self._game_files[game_idx]

        loop = asyncio.get_event_loop()

        # Blocking I/O → run_in_executor. The lock inside _start_game
        # serializes tatsu parser access across concurrent threads.
        env, game_state = await loop.run_in_executor(
            None, _start_game, game_file
        )

        state["tw_env"] = env
        state["tw_score"] = 0
        state["tw_max_score"] = self._max_scores[game_idx]
        state["tw_done"] = False
        state["tw_won"] = False
        state["tw_turn"] = 0
        state["tw_token_count"] = len(game_state.feedback) // 3

        # Merge the first user message (initial observation) into the system
        # prompt so it's part of the protected prefix for compaction/truncation.
        if self._include_initial_obs_in_system:
            prompt = state["prompt"]
            if (
                len(prompt) >= 2
                and _message_field(prompt[0], "role") == "system"
                and _message_field(prompt[1], "role") == "user"
            ):
                merged_content = (
                    str(_message_field(prompt[0], "content", ""))
                    + "\n\n---\nINITIAL OBSERVATION:\n"
                    + str(_message_field(prompt[1], "content", ""))
                )
                if isinstance(prompt[0], dict) or isinstance(prompt[1], dict):
                    state["prompt"] = [
                        {"role": "system", "content": merged_content},
                        {"role": "user", "content": "What do you do?"},
                    ]
                else:
                    from verifiers.types import SystemMessage, UserMessage
                    state["prompt"] = [
                        SystemMessage(role="system", content=merged_content),
                        UserMessage(role="user", content="What do you do?"),
                    ]

        _set_kv_eviction_textworld_metadata(state)
        return state

    async def env_response(self, messages: Messages, state: State) -> Messages:
        """Step TextWorld with parsed action, return observation.

        Called from get_prompt_messages() at start of each turn (except turn 0).
        Receives previous turn's full conversation, parses action from LLM output.
        """
        pending_preobs_response = state.pop("tw_pending_preobs_env_response", None)
        if pending_preobs_response is not None:
            state["tw_memory_manager_turns"] = (
                int(state.get("tw_memory_manager_turns", 0)) + 1
            )
            _set_kv_eviction_textworld_metadata(state)
            return pending_preobs_response

        # Parse action from LLM's XML output
        action_text = self.parser.parse_answer(messages)
        if action_text is None:
            # Fallback: DON'T set state["error"] — that zeros ALL completion_mask
            # in the entire rollout (interleave_rollout lines 56-58).
            action_text = "look"

        # Step TextWorld (blocking I/O → run_in_executor)
        # env.step() doesn't use the tatsu parser, so no lock needed.
        loop = asyncio.get_event_loop()
        env = state["tw_env"]
        try:
            result = await loop.run_in_executor(
                None, lambda: env.step(str(action_text))
            )
            game_state, score, done = result
            if score is None:
                score = state["tw_score"]
            obs = game_state.feedback
            won = bool(getattr(game_state, "won", False))
        except Exception as e:
            logger.warning(f"TextWorld step error: {e}")
            obs = "Something went wrong. Try a different action."
            score = state["tw_score"]
            done = False
            won = bool(state.get("tw_won", False))

        # score from textworld.step() is cumulative
        if score != state["tw_score"]:
            logger.warning(f"SCORE CHANGE: {state['tw_score']} -> {score} (action={action_text!r})")
        state["tw_score"] = score
        state["tw_done"] = done
        state["tw_won"] = won
        state["tw_turn"] = int(state.get("tw_turn", 0)) + 1

        # Track token budget (approximate: 4 chars ≈ 1 token)
        state["tw_token_count"] += len(obs) // 3 + 100  # +100 for action+thinking tokens
        _set_kv_eviction_textworld_metadata(state)

        # Signal termination via final_env_response
        if done:
            state["final_env_response"] = [
                {
                    "role": "user",
                    "content": f"Game Over! Final score: {score}/{state['tw_max_score']}",
                }
            ]
            return []

        memory_message = _build_kv_eviction_preobs_memory_message(messages)
        if memory_message:
            state["tw_pending_preobs_env_response"] = [
                {"role": "user", "content": obs}
            ]
            return [{"role": "user", "content": memory_message}]

        return [{"role": "user", "content": obs}]

    @vf.stop
    async def max_turns_reached(self, state: State) -> bool:
        """Count real TextWorld action generations, not memory-control turns."""
        if self.max_turns <= 0:
            return False
        memory_turns = int(state.get("tw_memory_manager_turns", 0))
        if state.get("tw_pending_preobs_env_response") is not None:
            memory_turns += 1
        real_turns = len(state.get("trajectory", [])) - memory_turns
        return real_turns >= self.max_turns

    @vf.stop
    async def max_sequence_length_reached(self, state: State) -> bool:
        """Stop once the cumulative model-token stream reaches its limit."""
        max_seq_len = self.max_rollout_seq_len
        trajectory = state.get("trajectory", [])
        if max_seq_len is None or max_seq_len <= 0 or not trajectory:
            return False

        last_step = trajectory[-1]
        extras = last_step.get("extras") or {}
        budget_capped = extras.get("logical_sequence_budget_capped") is True
        logical_seq_len = extras.get(
            "logical_sequence_limit_len",
            extras.get("logical_seq_len"),
        )
        if type(logical_seq_len) is not int:
            tokens = last_step.get("tokens")
            if tokens is None:
                return False
            logical_seq_len = len(tokens.get("prompt_ids", [])) + len(
                tokens.get("completion_ids", [])
            )

        if not budget_capped and logical_seq_len < max_seq_len:
            return False
        state["is_truncated"] = True
        state["max_sequence_length"] = max_seq_len
        return True

    @vf.stop
    async def game_done(self, state: State) -> bool:
        """Stop when the TextWorld game is complete."""
        return state.get("tw_done", False)

    @vf.cleanup
    async def cleanup_game(self, state: State) -> None:
        """Release retained Phase4 KV and close the Jericho subprocess.

        Jericho spawns a Z-machine subprocess with 3-4 FDs per rollout.
        Without explicit cleanup each rollout leaks a subprocess. env_worker
        disables GC during rollouts, so __del__ won't fire.
        The @vf.cleanup handler runs deterministically on every exit path.
        """
        try:
            from kv_eviction.env import release_phase4_trace

            await release_phase4_trace()
        except Exception:
            logger.exception("Failed to release the rollout's Phase4 trace")

        env = state.pop("tw_env", None)
        if env is not None:
            loop = asyncio.get_event_loop()
            try:
                await loop.run_in_executor(None, env.close)
            except Exception:
                pass  # Best-effort cleanup


def load_environment(
    dataset_path: str,
    max_episode_steps: int = 50,
    num_train_examples: Optional[int] = None,
    num_eval_examples: Optional[int] = None,
    seed: int = 42,
    include_initial_obs_in_system: bool = False,
    system_prompt_variant: str = "default",
    **kwargs,
) -> vf.Environment:
    """Entry point for `vf.load_environment("textworld-env", ...)`.

    Parameters
    ----------
    dataset_path : str
        Directory containing metadata.json, dataset/ (HF), and games/*.z8.
    max_episode_steps : int
        Hard cap on env_response calls per rollout (one per game turn).
    num_train_examples, num_eval_examples : Optional[int]
        Select train/eval split sizes. Modern datasets load held-out eval rows
        from eval_dataset/; legacy datasets fall back to slicing dataset/.
    seed : int
        Unused by the env itself (games are pre-generated with fixed
        seeds); kept for verifiers API compatibility.
    """
    return TextWorldEnv(
        dataset_path=dataset_path,
        max_episode_steps=max_episode_steps,
        num_train_examples=num_train_examples,
        num_eval_examples=num_eval_examples,
        seed=seed,
        include_initial_obs_in_system=include_initial_obs_in_system,
        system_prompt_variant=system_prompt_variant,
    )
