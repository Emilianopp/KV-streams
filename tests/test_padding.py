# SPDX-License-Identifier: Apache-2.0
"""Unit tests for kv_eviction.padding.

Validates the block-alignment invariant for AFTER-padding: the position
immediately after each `<|im_end|>`-plus-filler-run is a multiple of
`block_size`. `<|im_end|>` itself sits at its natural position; fillers
occupy the inter-turn gap.

No real tokenizer: a minimal stub that echoes per-char ids is enough to
exercise the padding loop deterministically. The end-to-end guarantee
(against a real Qwen3 tokenizer and vLLM server) is covered by Gates 3-4
of the plan, not here.
"""

from dataclasses import dataclass, field

from kv_eviction.padding import (
    render_padded_prompt,
    resolve_filler_token_id,
    resolve_im_end_token_id,
)

IM_END = 999
FILLER = 888
NL = 777


@dataclass
class _StubTokenizer:
    """Minimal stub: `apply_chat_template` returns a tagged string,
    `encode` parses the tagged string into a fixed id sequence.

    Format: "TOK:<id>,<id>,...|GEN:<id>,<id>,..." — the GEN suffix (if
    any) appears only when `add_generation_prompt=True`, modeling real
    chat templates whose with-gen output is a prefix-preserving
    extension of the without-gen output.
    """

    pad_token_id: int | None = FILLER
    gen_prefix_ids: list[int] = field(default_factory=list)
    _vocab: dict[str, int] = field(
        default_factory=lambda: {"<|im_end|>": IM_END, "<|endoftext|>": 0}
    )

    def apply_chat_template(
        self, messages, tools=None, add_generation_prompt=True, tokenize=False
    ):
        body = f"TOK:{messages[0]['content']}"
        if add_generation_prompt and self.gen_prefix_ids:
            body += "|GEN:" + ",".join(str(i) for i in self.gen_prefix_ids)
        return body

    def encode(self, s, add_special_tokens=False):
        if s == "\n":
            return []
        assert s.startswith("TOK:"), s
        payload = s[len("TOK:") :]
        msg_part, _, gen_part = payload.partition("|GEN:")
        tokens = [int(x) for x in msg_part.split(",") if x]
        if gen_part:
            tokens.extend(int(x) for x in gen_part.split(",") if x)
        return tokens

    def convert_tokens_to_ids(self, tok):
        return self._vocab.get(tok, -1)


def _render(
    raw_ids: list[int],
    block_size: int,
    gen_prefix_ids: list[int] | None = None,
):
    tok = _StubTokenizer(gen_prefix_ids=list(gen_prefix_ids or []))
    messages = [{"role": "user", "content": ",".join(str(i) for i in raw_ids)}]
    return render_padded_prompt(
        tokenizer=tok,
        messages=messages,
        tools=None,
        block_size=block_size,
        filler_token_id=FILLER,
        im_end_token_id=IM_END,
    )


def _im_end_positions(ids: list[int]) -> list[int]:
    return [i for i, t in enumerate(ids) if t == IM_END]


def test_single_im_end_next_turn_starts_on_block_boundary():
    # 5 non-im_end tokens then <|im_end|>. With block_size=8, after
    # appending <|im_end|> out has length 6; 2 filler tokens are appended
    # AFTER <|im_end|> so the next turn would start at index 8.
    raw = [1, 2, 3, 4, 5, IM_END]
    _, padded, pads = _render(raw, block_size=8)

    assert pads == [2]
    assert padded == [1, 2, 3, 4, 5, IM_END, FILLER, FILLER]
    # <|im_end|> sits at its natural position (index 5).
    assert _im_end_positions(padded) == [5]
    # Length after <|im_end|>-plus-fillers is a block boundary.
    assert len(padded) % 8 == 0


def test_multi_turn_every_boundary_block_aligned():
    # Three mini-turns of varying length, each closed with <|im_end|>.
    raw = (
        [1, 2, 3, IM_END]          # turn 1: 3 tokens body
        + [4, 5, 6, 7, 8, IM_END]  # turn 2: 5 tokens body
        + [9, IM_END]              # turn 3: 1 token body
    )
    _, padded, pads = _render(raw, block_size=4)

    # Trace with block_size=4 (AFTER padding):
    #  after turn 1 <|im_end|>: out len=4 -> remainder=0, pad=0.
    #  after turn 2 <|im_end|>: out len=10 -> remainder=2, pad=2. len=12.
    #  after turn 3 <|im_end|>: out len=14 -> remainder=2, pad=2. len=16.
    assert pads == [0, 2, 2]
    assert _im_end_positions(padded) == [3, 9, 13]
    # After each <|im_end|>'s filler run, the next slot is block-aligned.
    # The cumulative position immediately past each turn's fillers should
    # be a multiple of block_size.
    running = 0
    for tok in padded:
        running += 1
    assert running % 4 == 0
    # And the first-boundary snapshots:
    # end of turn 1 fillers = 4, end of turn 2 = 12, end of turn 3 = 16.
    assert [3 + 1 + 0, 9 + 1 + 2, 13 + 1 + 2] == [4, 12, 16]


def test_zero_pad_when_already_aligned():
    # Body length such that <|im_end|> lands exactly on a block boundary
    # means no padding needed.
    raw = [1, 2, 3, 4, 5, 6, 7, IM_END]
    _, padded, pads = _render(raw, block_size=8)
    assert pads == [0]
    assert padded == raw
    # <|im_end|> at index 7; len(out)=8 is already block-aligned.
    assert _im_end_positions(padded) == [7]
    assert len(padded) % 8 == 0


def test_gen_prefix_appears_at_tail_with_pad_inserted_before():
    # Message tokens end at the last <|im_end|>; gen_prefix_ids=[77,78,79]
    # models '<|im_start|>assistant\n'. After the fix, the final alignment
    # pad must sit BEFORE the gen prefix so the model's last
    # pre-generation tokens are natural (not <|endoftext|>).
    raw = [1, 2, IM_END]
    gen = [77, 78, 79]
    _, padded, pads = _render(raw, block_size=4, gen_prefix_ids=gen)

    # Per-<|im_end|> pad: after appending IM_END, len=3, pad=1 -> len=4.
    # Then extra pad so total + len(gen) is block-aligned:
    #   total_before_gen = 4, gen_len = 3 -> need 1 more filler -> len=5.
    # Then append gen -> len=8 (block-aligned).
    assert pads == [1, 1]
    assert padded == [1, 2, IM_END, FILLER, FILLER, 77, 78, 79]
    # CRITICAL: gen prefix is the tail — model's last tokens before
    # generation are the natural gen prompt, not filler.
    assert padded[-len(gen) :] == gen
    assert len(padded) % 4 == 0
    assert _im_end_positions(padded) == [2]


def test_qwen_newline_separator_is_padded_after_boundary():
    class NewlineTokenizer(_StubTokenizer):
        def encode(self, s, add_special_tokens=False):
            if s == "\n":
                return [NL]
            return super().encode(s, add_special_tokens=add_special_tokens)

    raw = [1, 2, IM_END, NL, 3, 4, IM_END, NL]
    tok = NewlineTokenizer()
    messages = [{"role": "user", "content": ",".join(str(i) for i in raw)}]
    _, padded, pads = render_padded_prompt(
        tokenizer=tok,
        messages=messages,
        tools=None,
        block_size=4,
        filler_token_id=FILLER,
        im_end_token_id=IM_END,
        add_generation_prompt=False,
    )

    assert pads == [0, 0]
    assert padded == raw
    assert padded[3] == NL
    assert padded[7] == NL


def test_qwen_newline_separator_gets_filler_after_newline_not_before():
    class NewlineTokenizer(_StubTokenizer):
        def encode(self, s, add_special_tokens=False):
            if s == "\n":
                return [NL]
            return super().encode(s, add_special_tokens=add_special_tokens)

    raw = [1, IM_END, NL, 2, IM_END, NL]
    tok = NewlineTokenizer()
    messages = [{"role": "user", "content": ",".join(str(i) for i in raw)}]
    _, padded, pads = render_padded_prompt(
        tokenizer=tok,
        messages=messages,
        tools=None,
        block_size=4,
        filler_token_id=FILLER,
        im_end_token_id=IM_END,
        add_generation_prompt=False,
    )

    assert pads == [1, 1]
    assert padded == [
        1,
        IM_END,
        NL,
        FILLER,
        2,
        IM_END,
        NL,
        FILLER,
    ]


def test_gen_prefix_no_extra_pad_when_total_already_aligned():
    # Message-region pads already leave total + gen_len on a block
    # boundary, so no extra filler is inserted before gen_prefix.
    raw = [1, 2, 3, 4, 5, IM_END]  # len=6 after IM_END, pad=2 -> len=8
    gen = [77, 78, 79, 80]  # len 4 -> total 12 (block-aligned)
    _, padded, pads = _render(raw, block_size=4, gen_prefix_ids=gen)

    assert pads == [2]
    assert padded == [1, 2, 3, 4, 5, IM_END, FILLER, FILLER, 77, 78, 79, 80]
    assert padded[-len(gen) :] == gen


def test_resolve_filler_override():
    tok = _StubTokenizer(pad_token_id=42)
    assert resolve_filler_token_id(tok, override=100) == 100
    # Override None falls through to pad_token_id.
    assert resolve_filler_token_id(tok, override=None) == 42


def test_resolve_filler_no_pad_token_falls_back_to_space():
    class NoPad:
        pad_token_id = None

        def encode(self, s, add_special_tokens=False):
            return [32]  # pretend ' ' tokenizes to id 32

        def convert_tokens_to_ids(self, tok):
            return -1

    assert resolve_filler_token_id(NoPad(), override=None) == 32


def test_resolve_filler_skips_turn_end_collision():
    tok = _StubTokenizer(pad_token_id=IM_END)
    assert (
        resolve_filler_token_id(
            tok,
            override=None,
            forbidden_token_ids=(IM_END,),
        )
        == 0
    )


def test_resolve_filler_rejects_override_collision():
    tok = _StubTokenizer(pad_token_id=FILLER)
    try:
        resolve_filler_token_id(
            tok,
            override=IM_END,
            forbidden_token_ids=(IM_END,),
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("expected collision with <|im_end|> to fail")


def test_resolve_im_end_asserts_present():
    tok = _StubTokenizer()
    assert resolve_im_end_token_id(tok) == IM_END

    class NoImEnd:
        def convert_tokens_to_ids(self, tok):
            return -1

    try:
        resolve_im_end_token_id(NoImEnd())
    except AssertionError:
        pass
    else:
        raise AssertionError("expected AssertionError when <|im_end|> is missing")
