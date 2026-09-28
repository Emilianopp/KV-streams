import asyncio

from experiments.textworld_env.textworld_env import TextWorldEnv


def _env_with_limit(max_seq_len: int) -> TextWorldEnv:
    env = object.__new__(TextWorldEnv)
    env.max_rollout_seq_len = max_seq_len
    return env


def test_kv_rollout_stops_at_logical_sequence_limit():
    env = _env_with_limit(8)
    state = {
        "is_truncated": False,
        "trajectory": [
            {
                "extras": {"logical_seq_len": 8},
                "tokens": {
                    "prompt_ids": [1],
                    "completion_ids": [2],
                },
            }
        ],
    }

    assert asyncio.run(env.max_sequence_length_reached(state))
    assert state["is_truncated"] is True


def test_full_context_rollout_stops_at_sequence_limit():
    env = _env_with_limit(8)
    state = {
        "is_truncated": False,
        "trajectory": [
            {
                "extras": {},
                "tokens": {
                    "prompt_ids": [1, 2, 3, 4, 5],
                    "completion_ids": [6, 7, 8],
                },
            }
        ],
    }

    assert asyncio.run(env.max_sequence_length_reached(state))
    assert state["is_truncated"] is True


def test_rollout_continues_below_sequence_limit():
    env = _env_with_limit(8)
    state = {
        "is_truncated": False,
        "trajectory": [
            {
                "extras": {"logical_seq_len": 7},
                "tokens": {
                    "prompt_ids": [1],
                    "completion_ids": [2],
                },
            }
        ],
    }

    assert not asyncio.run(env.max_sequence_length_reached(state))
    assert state["is_truncated"] is False


def test_rollout_uses_selected_non_padding_sequence_limit():
    env = _env_with_limit(10)
    state = {
        "is_truncated": False,
        "trajectory": [
            {
                "extras": {
                    "logical_seq_len": 15,
                    "logical_sequence_limit_len": 9,
                },
                "tokens": {
                    "prompt_ids": [1],
                    "completion_ids": [2],
                },
            }
        ],
    }

    assert not asyncio.run(env.max_sequence_length_reached(state))
    assert state["is_truncated"] is False


def test_rollout_stops_when_generation_budget_was_capped():
    env = _env_with_limit(8)
    state = {
        "is_truncated": False,
        "trajectory": [
            {
                "extras": {
                    "logical_seq_len": 7,
                    "logical_sequence_budget_capped": True,
                },
                "tokens": {
                    "prompt_ids": [1],
                    "completion_ids": [2],
                },
            }
        ],
    }

    assert asyncio.run(env.max_sequence_length_reached(state))
    assert state["is_truncated"] is True
