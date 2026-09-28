from types import SimpleNamespace
from unittest.mock import patch

import pytest
from verifiers.errors import OverlongPromptError

from kv_eviction import env


class _BudgetTokenizer:
    def apply_chat_template(
        self,
        messages,
        tools=None,
        add_generation_prompt=True,
        tokenize=False,
    ):
        assert add_generation_prompt
        assert not tokenize
        return f"{len(messages)}:{len(tools or [])}"

    def encode(self, rendered, add_special_tokens=False):
        assert not add_special_tokens
        message_count, tool_count = (
            int(value) for value in rendered.split(":")
        )
        return list(range(message_count + tool_count + 1))


def _messages():
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "initial task"},
        {"role": "assistant", "content": "first action"},
        {"role": "tool", "content": "first environment response"},
        {"role": "assistant", "content": "second action"},
        {"role": "user", "content": "current environment response"},
    ]


@pytest.fixture(autouse=True)
def reset_interceptor_configs():
    env._markovian_config = None
    env._summary_config = None
    env._padding_config = None
    env._LOGICAL_EVICTED_TOKENS.set(0)
    yield
    env._markovian_config = None
    env._summary_config = None
    env._padding_config = None


def test_markovian_reports_untruncated_rollout_length():
    import asyncio

    from openai.resources.chat.completions.completions import AsyncCompletions

    called_with = {}
    response = SimpleNamespace(
        prompt_token_ids=None,
        choices=[SimpleNamespace(token_ids=[101, 102, 103])],
    )

    async def fake_orig(self, *args, **kwargs):
        called_with.update(kwargs)
        return response

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        env.configure_markovian_thinker(
            enabled=True,
            tokenizer=_BudgetTokenizer(),
            max_turns=1,
            max_logical_seq_len=11,
        )
        result = asyncio.run(
            AsyncCompletions.create(
                self=None,
                model="model",
                messages=_messages(),
                tools=[{"type": "function"}],
                max_completion_tokens=20,
            )
        )

    assert called_with["max_completion_tokens"] == 3
    assert len(called_with["messages"]) == 2
    assert len(called_with["extra_body"]["prompt_token_ids"]) == 4
    assert result.logical_seq_len == 11
    assert result.logical_padding_seq_len == 0
    assert result.logical_non_padding_seq_len == 11
    assert result.logical_sequence_limit_len == 11
    assert result.context_seq_len == 7
    assert result.context_padding_seq_len == 0
    assert result.context_non_padding_seq_len == 7
    assert result.logical_sequence_budget_capped is True

    step = {"extras": {}}
    env.attach_logical_seq_len_from_response(step, result)
    env.attach_logical_sequence_budget_capped_from_response(step, result)
    assert step["extras"]["logical_seq_len"] == 11
    assert step["extras"]["context_seq_len"] == 7
    assert step["extras"]["logical_sequence_budget_capped"] is True


def test_markovian_stops_before_an_over_budget_request():
    import asyncio

    from openai.resources.chat.completions.completions import AsyncCompletions

    called = False

    async def fake_orig(self, *args, **kwargs):
        nonlocal called
        called = True
        return SimpleNamespace()

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        env.configure_markovian_thinker(
            enabled=True,
            tokenizer=_BudgetTokenizer(),
            max_turns=1,
            max_logical_seq_len=8,
        )
        with pytest.raises(
            OverlongPromptError,
            match="Markovian logical sequence reached its limit",
        ):
            asyncio.run(
                AsyncCompletions.create(
                    self=None,
                    model="model",
                    messages=_messages(),
                    tools=[{"type": "function"}],
                    max_completion_tokens=20,
                )
            )

    assert not called


def test_markovian_clamps_minimum_tokens_to_remaining_budget():
    kwargs = {
        "extra_body": {"min_tokens": 5},
        "max_completion_tokens": 20,
        "tools": [{"type": "function"}],
    }

    logical_prompt_len, was_capped = (
        env._apply_markovian_logical_sequence_budget(
            kwargs,
            tokenizer=_BudgetTokenizer(),
            logical_messages=_messages(),
            max_logical_seq_len=11,
        )
    )

    assert logical_prompt_len == 8
    assert was_capped is True
    assert kwargs["max_completion_tokens"] == 3
    assert kwargs["extra_body"]["min_tokens"] == 3


class _SummaryBudgetTokenizer(_BudgetTokenizer):
    """Chat-template tokenizer that also handles plain-text encode.

    ``_BudgetTokenizer.encode`` only parses the ``"n:m"`` rendered form. The
    summary accounting also encodes the raw instruction string, so accept both.
    """

    def encode(self, text, add_special_tokens=False):
        try:
            return super().encode(text, add_special_tokens=add_special_tokens)
        except ValueError:
            return list(range(len(str(text).split())))


def test_markovian_summary_tokens_count_against_the_logical_budget():
    """A fired summary is not an env turn, but its [I, S] exchange is real
    context and real generated tokens, so it must be charged to the logical
    sequence and shrink the budget left for the outer turn."""
    import asyncio

    from openai.resources.chat.completions.completions import AsyncCompletions

    instruction = "summarize now please"          # 3 words -> 3 tokens
    summary_completion_ids = [7, 8, 9, 10]        # 4 tokens

    outer_calls = []
    summary_response = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        choices=[
            SimpleNamespace(
                token_ids=list(summary_completion_ids),
                message=SimpleNamespace(content="THE SUMMARY"),
                logprobs=None,
            )
        ],
    )
    outer_response = SimpleNamespace(
        prompt_token_ids=None,
        choices=[SimpleNamespace(token_ids=[201, 202])],
    )

    async def fake_orig(self, *args, **kwargs):
        if env._IN_SUMMARY_CALL.get():
            return summary_response
        outer_calls.append(dict(kwargs))
        return outer_response

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        env.configure_markovian_thinker(
            enabled=True,
            tokenizer=_SummaryBudgetTokenizer(),
            max_turns=1,
            max_logical_seq_len=1000,
        )
        env.configure_markovian_summary(
            enabled=True,
            mode="markovian",
            compaction_max_turns=1,
            max_len_summary=64,
            instruction_text=instruction,
            resume_text="continue",
        )
        result = asyncio.run(
            AsyncCompletions.create(
                self=None,
                model="model",
                messages=_messages(),
                tools=[{"type": "function"}],
                max_completion_tokens=20,
            )
        )

    assert outer_calls, "outer request never fired"
    # Baseline logical prompt for _messages() + 1 tool under _BudgetTokenizer.
    base_prompt_len = len(
        env._tokenize_chat_prompt(
            _SummaryBudgetTokenizer(), _messages(), [{"type": "function"}]
        )
    )
    expected_summary_tokens = len(summary_completion_ids) + len(instruction.split())
    outer_completion_len = len(outer_response.choices[0].token_ids)

    # The summary's tokens are charged to the logical stream...
    assert result.logical_seq_len == (
        base_prompt_len + expected_summary_tokens + outer_completion_len
    )
    # ...and are NOT silently dropped the way they were before the fix.
    assert result.logical_seq_len > base_prompt_len + outer_completion_len


def _degenerate_choice(n_sampled: int, content: str):
    """A completion that sampled ``n_sampled`` tokens whose text retokenizes
    to ``len(content.split())`` tokens under ``_SummaryBudgetTokenizer``."""
    return SimpleNamespace(
        prompt_token_ids=None,
        choices=[
            SimpleNamespace(
                token_ids=list(range(1000, 1000 + n_sampled)),
                message=SimpleNamespace(content=content, tool_calls=None),
            )
        ],
    )


def test_sampled_token_overage_charges_future_turns():
    """A completion whose sampled stream is longer than its text's canonical
    retokenization (the newline-stuffing degeneracy) must keep charging the
    logical budget on later turns instead of collapsing to the retokenized
    size -- and a clean round-tripping completion must charge nothing."""
    import asyncio

    from openai.resources.chat.completions.completions import AsyncCompletions

    responses = [
        _degenerate_choice(40, "spam"),                # 40 sampled vs 1 retok
        _degenerate_choice(2, "two words"),            # clean: 2 vs 2
        _degenerate_choice(3, "three little words"),   # clean: 3 vs 3
    ]

    async def fake_orig(self, *args, **kwargs):
        return responses.pop(0)

    async def episode():
        results = []
        for _ in range(3):
            results.append(
                await AsyncCompletions.create(
                    self=None,
                    model="model",
                    messages=_messages(),
                    tools=[{"type": "function"}],
                    max_completion_tokens=100,
                )
            )
        return results

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        env.configure_markovian_thinker(
            enabled=True,
            tokenizer=_SummaryBudgetTokenizer(),
            max_turns=1,
            max_logical_seq_len=1000,
        )
        r1, r2, r3 = asyncio.run(episode())

    base = len(
        env._tokenize_chat_prompt(
            _SummaryBudgetTokenizer(), _messages(), [{"type": "function"}]
        )
    )
    # Turn 1 stamps its own completion at sampled size.
    assert r1.logical_seq_len == base + 40
    # Turn 2's prompt re-measure keeps turn 1's 39-token excess
    # (40 sampled vs 1 retokenized) instead of forgetting it.
    assert r2.logical_seq_len == base + 39 + 2
    # Turn 3 proves the clean turn 2 banked nothing new.
    assert r3.logical_seq_len == base + 39 + 3


def test_sampled_token_overage_enforces_the_budget():
    """The banked overage must make the budget preflight bind: an episode
    that generated past the cap in sampled tokens stops, even though its
    retokenized history alone would still fit."""
    import asyncio

    from openai.resources.chat.completions.completions import AsyncCompletions

    n_calls = 0

    async def fake_orig(self, *args, **kwargs):
        nonlocal n_calls
        n_calls += 1
        return _degenerate_choice(40, "spam")

    async def episode():
        await AsyncCompletions.create(
            self=None,
            model="model",
            messages=_messages(),
            tools=[{"type": "function"}],
            max_completion_tokens=100,
        )
        # Retokenized history alone (8 tokens) fits the cap of 30; the
        # 39-token banked excess must push the preflight past it.
        await AsyncCompletions.create(
            self=None,
            model="model",
            messages=_messages(),
            tools=[{"type": "function"}],
            max_completion_tokens=100,
        )

    with patch.object(AsyncCompletions, "create", fake_orig):
        env._install_message_padding_interceptor()
        env.configure_markovian_thinker(
            enabled=True,
            tokenizer=_SummaryBudgetTokenizer(),
            max_turns=1,
            max_logical_seq_len=30,
        )
        with pytest.raises(
            OverlongPromptError,
            match="Markovian logical sequence reached its limit",
        ):
            asyncio.run(episode())

    assert n_calls == 1, "the over-budget second request must not be sent"
