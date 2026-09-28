from __future__ import annotations

import asyncio
import math
from types import SimpleNamespace

import kv_eviction.env as env
import pytest
import torch
import verifiers as vf
from kv_eviction.types import compute_turn_compaction_state_id
from openai.types.chat import ChatCompletion
from verifiers.clients.openai_chat_completions_client import (
    OpenAIChatCompletionsClient,
)
from verifiers.errors import EmptyModelResponseError, ModelError
from verifiers.utils.response_utils import parse_response_tokens

from prime_rl.orchestrator.trajectories import interleave_rollout
from prime_rl.trainer.batch import prepare_sample
from prime_rl.trainer.rl.loss import compute_loss, sft_loss_fn


def _native_sglang_response(
    *,
    prompt_ids=None,
    completion_ids=None,
    completion_logprobs=None,
) -> ChatCompletion:
    prompt_ids = prompt_ids or [10, 11, 12, 13, 14, 15]
    completion_ids = completion_ids if completion_ids is not None else [20, 21]
    completion_logprobs = (
        completion_logprobs
        if completion_logprobs is not None
        else [-0.125, -0.375]
    )
    event = {
        "num_output_tokens_at_compaction": 0,
        "tokens_evicted": 2,
        "position_offset_after": 2,
        "num_prompt_tokens": 4,
        "evict_start": 2,
        "new_user_fragment_len": 1,
        "kept_indices": [0, 1, 4, 5],
        "kept_token_ids": [10, 11, 14, 15],
        "last_turn_evicted": 0,
        "num_turns_evicted_after": 1,
    }
    logprob_content = [
        {
            "token": f"token-{token_id}",
            "bytes": [token_id],
            "logprob": logprob,
            "top_logprobs": [],
        }
        for token_id, logprob in zip(completion_ids, completion_logprobs)
    ]
    return ChatCompletion.model_validate(
        {
            "id": "chatcmpl-sglang-exact",
            "object": "chat.completion",
            "created": 1,
            "model": "test-model",
            "prompt_token_ids": prompt_ids,
            "compaction_replay_mode": "prefill_trim",
            "compaction_events": [event],
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop",
                    "token_ids": completion_ids,
                    "logprobs": {"content": logprob_content},
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion_ids),
                "total_tokens": len(prompt_ids) + len(completion_ids),
            },
        }
    )


def _convert_native_response(native: ChatCompletion):
    client = object.__new__(OpenAIChatCompletionsClient)
    return asyncio.run(client.from_native_response(native))


class _StaticCompletions:
    def __init__(self, response: ChatCompletion):
        self.response = response

    async def create(self, **kwargs):
        return self.response


def _get_response(native: ChatCompletion):
    client = OpenAIChatCompletionsClient(
        SimpleNamespace(chat=SimpleNamespace(completions=_StaticCompletions(native)))
    )
    return asyncio.run(
        client.get_response(
            prompt=[vf.UserMessage(content="question")],
            model="test-model",
            sampling_args={},
        )
    )


def test_sglang_response_reaches_mode_one_batch_without_retokenization():
    submitted = [10, 11, 12, 13, 14, 15]
    native = _native_sglang_response(prompt_ids=submitted)
    env._stash_prompt_token_ids(native, submitted)

    response = _convert_native_response(native)
    tokens = asyncio.run(parse_response_tokens(response))
    assert tokens is not None
    step = vf.TrajectoryStep(
        prompt=[{"role": "system", "content": "rules"}],
        completion=[{"role": "assistant", "content": "answer"}],
        response=response,
        tokens=tokens,
        reward=None,
        advantage=None,
        is_truncated=False,
        trajectory_id="trajectory-1",
        extras={},
    )
    env.attach_compaction_events_from_response(step, response)
    env.attach_prompt_token_ids_from_response(step, response)
    env.attach_submitted_prompt_token_ids_from_response(step, response)
    output = vf.RolloutOutput(
        example_id=7,
        trajectory=[step],
        sampling_args={"temperature": 0.8},
        error=None,
    )

    samples = interleave_rollout(output)
    assert samples is not None and len(samples) == 1
    sample = samples[0]
    assert sample.prompt_ids == [10, 11, 14, 15]
    assert sample.completion_ids == [20, 21]
    assert sample.completion_logprobs == [-0.125, -0.375]
    assert sample.completion_temperatures == [0.8, 0.8]
    assert sample.calls is not None and len(sample.calls) == 1
    assert sample.calls[0].submitted_prompt_ids == submitted
    assert sample.calls[0].completion_temperatures == [0.8, 0.8]

    micro_batch = prepare_sample(sample, seq_len=4)
    assert micro_batch.input_ids == [10, 11, 14, 15, 20, 21]
    assert micro_batch.position_ids == [0, 1, 4, 5, 6, 7]
    assert micro_batch.inference_logprobs[-2:] == [-0.125, -0.375]
    assert micro_batch.temperatures == [0.8] * 6
    assert micro_batch.loss_mask == [False, False, False, False, True, True]
    assert micro_batch.prompt_len == 4
    assert micro_batch.compaction_replay_mode == 1


def test_sglang_carried_state_without_event_survives_native_conversion():
    prompt_ids = [10, 11, 12, 13, 14, 15]
    carried_prefix_len = 4
    state = {
        "version": 1,
        "position_offset": 8,
        "protected_prefix_len": 2,
        "num_turns_evicted": 2,
        "carried_prefix_num_live_turns": 1,
        "carried_prefix_len": carried_prefix_len,
    }
    state["state_id"] = compute_turn_compaction_state_id(
        **state,
        carried_prefix_token_ids=prompt_ids[:carried_prefix_len],
    )
    native = _native_sglang_response(prompt_ids=prompt_ids)
    native.compaction_events = None
    native.turn_compaction_state = state
    env._stash_prompt_token_ids(native, prompt_ids)

    response = _convert_native_response(native)
    tokens = asyncio.run(parse_response_tokens(response))
    assert tokens is not None
    assert response.turn_compaction_state == state
    step = vf.TrajectoryStep(
        prompt=[{"role": "system", "content": "rules"}],
        completion=[{"role": "assistant", "content": "answer"}],
        response=response,
        tokens=tokens,
        reward=None,
        advantage=None,
        is_truncated=False,
        trajectory_id="trajectory-carried",
        extras={},
    )
    env.attach_compaction_events_from_response(step, response)
    env.attach_prompt_token_ids_from_response(step, response)
    env.attach_submitted_prompt_token_ids_from_response(step, response)

    assert step["extras"]["compaction_events"] == []
    assert step["extras"]["turn_compaction_state"] == state
    samples = interleave_rollout(
        vf.RolloutOutput(
            example_id=9,
            trajectory=[step],
            sampling_args={"temperature": 0.8},
            error=None,
        )
    )
    assert samples is not None and len(samples) == 1
    sample = samples[0]
    assert sample.prompt_ids == prompt_ids
    assert sample.calls is not None
    assert sample.calls[0].compaction_events == []
    assert sample.calls[0].turn_compaction_state is not None
    micro_batch = prepare_sample(sample, seq_len=4)
    assert micro_batch.position_ids == [0, 1, 10, 11, 12, 13, 14, 15]


def test_sglang_zero_token_mode_one_builds_finite_eventless_loss_sample():
    submitted = [10, 11, 12, 13, 14, 15]
    native = _native_sglang_response(
        prompt_ids=submitted,
        completion_ids=[],
        completion_logprobs=[],
    )
    native.choices[0].message.content = ""
    env._stash_prompt_token_ids(native, submitted)
    response = _get_response(native)
    tokens = asyncio.run(parse_response_tokens(response))
    assert tokens is not None
    step = vf.TrajectoryStep(
        prompt=[{"role": "system", "content": "rules"}],
        completion=[{"role": "assistant", "content": ""}],
        response=response,
        tokens=tokens,
        reward=None,
        advantage=None,
        is_truncated=False,
        trajectory_id="trajectory-zero",
        extras={},
    )
    env.attach_compaction_events_from_response(step, response)
    env.attach_prompt_token_ids_from_response(step, response)
    env.attach_submitted_prompt_token_ids_from_response(step, response)
    output = vf.RolloutOutput(
        example_id=8,
        trajectory=[step],
        sampling_args={"temperature": 0.0},
        error=None,
    )

    samples = interleave_rollout(output)

    assert samples is not None and len(samples) == 1
    sample = samples[0]
    sample.advantage = 0.0
    assert sample.completion_ids == []
    assert sample.completion_logprobs == []
    assert sample.completion_temperatures == []
    assert sample.calls is not None
    assert sample.calls[0].completion_ids == []
    micro_batch = prepare_sample(sample, seq_len=4)
    assert micro_batch.input_ids == [10, 11, 14, 15]
    assert micro_batch.loss_mask == [False] * 4
    assert micro_batch.temperatures == [1.0] * 4
    assert micro_batch.compaction_events is None
    assert micro_batch.calls is None

    trainer_logprobs = torch.zeros(4, requires_grad=True)
    loss_scale = max(sum(micro_batch.loss_mask), 1)
    loss, _ = compute_loss(
        trainer_logprobs=[trainer_logprobs],
        inference_logprobs=[torch.tensor(micro_batch.inference_logprobs)],
        teacher_logprobs=None,
        advantages=[torch.tensor(micro_batch.advantages)],
        loss_mask=[torch.tensor(micro_batch.loss_mask)],
        loss_fn=sft_loss_fn,
        loss_scale=loss_scale,
    )
    assert math.isfinite(loss.item())
    assert loss.item() == 0.0
    loss.backward()


def test_sglang_empty_ordinary_response_keeps_verifier_rejection():
    native = _native_sglang_response(
        completion_ids=[],
        completion_logprobs=[],
    )
    native.choices[0].message.content = ""
    native.compaction_replay_mode = None
    env._stash_prompt_token_ids(native, [10, 11, 12, 13, 14, 15])

    with pytest.raises(EmptyModelResponseError):
        _get_response(native)


@pytest.mark.parametrize(
    ("malformation", "error"),
    [
        ("nonempty_token_ids", "choice.token_ids == []"),
        ("missing_submitted_ids", "submitted prompt token IDs"),
        ("missing_native_ids", "native prompt_token_ids"),
        ("missing_completion_ids", "completion token_ids"),
        ("missing_logprobs", "standard completion logprobs"),
        ("missing_event", "exactly one raw compaction event"),
        ("non_admission_event", "admission events"),
        ("missing_finish_reason", "terminal finish_reason"),
        ("unsupported_mode", "unsupported compaction_replay_mode"),
    ],
)
def test_sglang_malformed_zero_token_mode_one_stays_rejected(
    malformation,
    error,
):
    if malformation == "nonempty_token_ids":
        native = _native_sglang_response(
            completion_ids=[20],
            completion_logprobs=[-0.125],
        )
    else:
        native = _native_sglang_response(
            completion_ids=[],
            completion_logprobs=[],
        )
    native.choices[0].message.content = ""

    if malformation != "missing_submitted_ids":
        env._stash_prompt_token_ids(native, [10, 11, 12, 13, 14, 15])
    if malformation == "missing_native_ids":
        native.prompt_token_ids = None
    elif malformation == "missing_completion_ids":
        native.choices[0].token_ids = None
    elif malformation == "missing_logprobs":
        native.choices[0].logprobs = None
    elif malformation == "missing_event":
        native.compaction_events = []
    elif malformation == "non_admission_event":
        native.compaction_events[0]["num_output_tokens_at_compaction"] = 1
    elif malformation == "missing_finish_reason":
        native.choices[0].finish_reason = None
    elif malformation == "unsupported_mode":
        native.compaction_replay_mode = "unknown"

    with pytest.raises(ModelError) as exc_info:
        _get_response(native)
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert error in str(exc_info.value.__cause__)


@pytest.mark.parametrize(
    ("submitted", "completion_ids", "logprobs", "error"),
    [
        ([10, 11, 99, 13, 14, 15], [20, 21], [-0.125, -0.375], "prompt"),
        ([10, 11, 12, 13, 14, 15], [20, 21], [-0.125], "length mismatch"),
        (
            [10, 11, 12, 13, 14, 15],
            [20, 21],
            [-0.125, float("nan")],
            "non-finite",
        ),
    ],
)
def test_sglang_native_transport_fails_closed(
    submitted,
    completion_ids,
    logprobs,
    error,
):
    native = _native_sglang_response(
        completion_ids=completion_ids,
        completion_logprobs=logprobs,
    )
    env._stash_prompt_token_ids(native, submitted)

    with pytest.raises(ValueError, match=error):
        _convert_native_response(native)


@pytest.mark.parametrize(
    ("malformation", "error"),
    [
        ("wrong_length", "kept_indices length"),
        ("unordered", "strictly increasing"),
        ("out_of_range", "outside the submitted prompt"),
        ("wrong_indices", "do not match the eviction range"),
        ("wrong_tokens", "survivor tokens"),
    ],
)
def test_sglang_native_transport_rejects_malformed_survivors(
    malformation,
    error,
):
    submitted = [10, 11, 12, 13, 14, 15]
    native = _native_sglang_response(prompt_ids=submitted)
    event = native.compaction_events[0]
    if malformation == "wrong_length":
        event["kept_indices"] = [0, 1, 4]
    elif malformation == "unordered":
        event["kept_indices"] = [0, 1, 5, 4]
    elif malformation == "out_of_range":
        event["kept_indices"] = [0, 1, 4, 6]
    elif malformation == "wrong_indices":
        event["kept_indices"] = [0, 2, 4, 5]
    elif malformation == "wrong_tokens":
        event["kept_token_ids"] = [10, 11, 14, 999]
    env._stash_prompt_token_ids(native, submitted)

    with pytest.raises(ValueError, match=error):
        _convert_native_response(native)


def test_sglang_native_transport_accepts_zero_length_survivor_prefix():
    submitted = [10, 11, 12, 13, 14, 15]
    native = _native_sglang_response(prompt_ids=submitted)
    native.compaction_events[0].update(
        evict_start=0,
        kept_indices=[2, 3, 4, 5],
        kept_token_ids=[12, 13, 14, 15],
    )
    env._stash_prompt_token_ids(native, submitted)

    response = _convert_native_response(native)

    assert response.compaction_events[0]["kept_indices"] == [2, 3, 4, 5]


@pytest.mark.parametrize(
    ("missing_field", "error"),
    [
        ("token_ids", "completion token_ids"),
        ("logprobs", "standard completion logprobs"),
    ],
)
def test_sglang_native_transport_rejects_missing_requested_metadata(
    missing_field,
    error,
):
    native = _native_sglang_response()
    setattr(native.choices[0], missing_field, None)
    env._stash_prompt_token_ids(native, [10, 11, 12, 13, 14, 15])

    with pytest.raises(ValueError, match=error):
        _convert_native_response(native)
