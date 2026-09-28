# SPDX-License-Identifier: Apache-2.0
"""Unit tests for segmented_forward without a real model.

Uses a mock HF-like model that records every (input_ids, position_ids,
past_key_values) triple it's called with, so we can verify segment slicing
and the KV drop logic without needing GPU inference.

The REAL correctness check (logit match vs vllm inference) lives in Phase
3.4's live KL test on Qwen3-4B. These tests only validate the bookkeeping:
segment ranges, drop offsets, retained KV identities.
"""

from dataclasses import dataclass

import pytest
import torch
from prime_rl.trainer.models.layers.lora import (
    get_lora_num_tokens,
    set_lora_num_tokens,
)
from transformers import DynamicCache

import kv_eviction.segmented_forward as segmented_forward_mod
from kv_eviction.segmented_forward import compute_num_segments, segmented_forward


@dataclass
class MockConfig:
    use_cache: bool = False


@dataclass
class _MockOutput:
    logits: torch.Tensor
    past_key_values: DynamicCache | None


class _Backbone(torch.nn.Module):
    """Tiny backbone that fabricates KV entries from positional embeddings.

    Each forward pass:
    1. Extends the passed-in DynamicCache by `seq_len` new entries whose
       per-layer K/V tensors encode the positional index (so we can
       trace which original positions survived).
    2. Returns a _MockOutput with the updated cache and dummy hidden states.
    """

    def __init__(self, num_layers: int = 2, num_heads: int = 1, head_dim: int = 4):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.calls: list[dict] = []

    def forward(self, input_ids, position_ids, past_key_values=None, use_cache=True):
        assert input_ids.shape[0] == 1, "batch_size=1 only"
        seq_len = input_ids.shape[1]
        pre_past_kv_len = (
            past_key_values.layers[0].keys.shape[2]
            if (past_key_values is not None
                and hasattr(past_key_values, "layers")
                and past_key_values.layers)
            else 0
        )
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        # Fabricate new K/V: each position encoded as a constant tensor
        # equal to the position_id. Shape [1, heads, seq, dim].
        pos = position_ids[0]  # [seq]
        if use_cache:
            for layer_idx in range(self.num_layers):
                new_K = pos.view(1, 1, seq_len, 1).expand(
                    1, self.num_heads, seq_len, self.head_dim
                ).float()
                new_V = new_K.clone()
                past_key_values.update(new_K, new_V, layer_idx)
        # Dummy hidden states, ignored by segmented_forward
        hidden = torch.zeros(1, seq_len, 8)
        self.calls.append({
            "seq_len": seq_len,
            "position_ids": position_ids[0].tolist(),
            "had_past_kv": past_key_values is not None,
            "past_kv_len": pre_past_kv_len,
        })
        return _MockOutput(logits=hidden, past_key_values=past_key_values)


class MockModel(torch.nn.Module):
    """Mock HF CausalLM that returns per-position logits and records calls."""

    def __init__(self, vocab_size: int = 100, num_layers: int = 2):
        super().__init__()
        self.config = MockConfig()
        self.model = _Backbone(num_layers=num_layers)
        self.vocab_size = vocab_size
        self.logit_scale = torch.nn.Parameter(torch.ones(()))
        self.calls: list[dict] = []

    def forward(self, input_ids, position_ids=None, past_key_values=None, use_cache=True):
        # Record past_kv_len BEFORE the backbone extends the cache, otherwise
        # we'd measure the post-extension size and see `prev_len + seq_len`.
        pre_past_kv_len = (
            past_key_values.layers[0].keys.shape[2]
            if (past_key_values is not None
                and hasattr(past_key_values, "layers")
                and past_key_values.layers)
            else 0
        )
        # Delegate to backbone so the hook captures past_key_values from there.
        backbone_out = self.model(input_ids, position_ids, past_key_values, use_cache)
        # Per-position logits = position_id (broadcast to vocab). This lets
        # us verify the correct tokens ended up in the correct segments.
        seq_len = input_ids.shape[1]
        logits = (
            position_ids.float().unsqueeze(-1).expand(1, seq_len, self.vocab_size)
            * self.logit_scale
        )
        self.calls.append({
            "seq_len": seq_len,
            "position_ids": position_ids[0].tolist(),
            "use_cache": use_cache,
            "had_past_kv": past_key_values is not None,
            "past_kv_len": pre_past_kv_len,
        })
        return {"logits": logits, "past_key_values": backbone_out.past_key_values}


class LoraAwareMockModel(MockModel):
    def __init__(self, vocab_size: int = 100, num_layers: int = 2):
        super().__init__(vocab_size=vocab_size, num_layers=num_layers)
        self.lora_token_counts_seen: list[list[int]] = []

    def forward(self, input_ids, position_ids=None, past_key_values=None, use_cache=True):
        self.lora_token_counts_seen.append(get_lora_num_tokens().cpu().tolist())
        return super().forward(input_ids, position_ids, past_key_values, use_cache)


class RootOnlyBackbone(_Backbone):
    def __init__(self, num_layers: int = 2):
        super().__init__(num_layers=num_layers)
        self.allow_call = False

    def forward(self, input_ids, position_ids, past_key_values=None, use_cache=True):
        if not self.allow_call:
            raise AssertionError("backbone was called directly")
        return super().forward(input_ids, position_ids, past_key_values, use_cache)


class RootOnlyMockModel(MockModel):
    """Mock an FSDP2 layout where direct backbone calls are invalid."""

    def __init__(self, vocab_size: int = 100, num_layers: int = 2):
        super().__init__(vocab_size=vocab_size, num_layers=num_layers)
        self.model = RootOnlyBackbone(num_layers=num_layers)
        self.prefill_logits_to_keep_seen: list[int | None] = []

    def forward(
        self,
        input_ids,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        logits_to_keep=None,
    ):
        self.prefill_logits_to_keep_seen.append(logits_to_keep)
        self.model.allow_call = True
        try:
            return super().forward(input_ids, position_ids, past_key_values, use_cache)
        finally:
            self.model.allow_call = False


def _am_selected_indices(
    *,
    num_layers: int = 2,
    num_heads: int = 1,
    synthetic: int = 2,
) -> list[list[list[int]]]:
    return [
        [list(range(synthetic)) for _ in range(num_heads)]
        for _ in range(num_layers)
    ]


def test_attention_matching_missing_selected_indices_fails_loudly():
    with pytest.raises(RuntimeError, match="missing selected OMP indices"):
        segmented_forward_mod._attention_matching_selected_indices_for_layer(
            None,
            layer_idx=0,
            num_heads=1,
            synthetic=2,
            compact_len=4,
            device=torch.device("cpu"),
            seg_idx=0,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_boundary_basic():
    """prompt_len=10, boundaries=[20], completion has 30 tokens, seq_len=40.

    Expected segments (my convention):
    - Seg 0: [0, 10+20) = [0, 30): prompt + gen[0..19]
    - Tail:  [10+20-1, 40) = [29, 40): gen[19..29] (with overlap)
    """
    model = MockModel()
    input_ids = torch.arange(40).unsqueeze(0)  # [1, 40]
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 40)

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[20],
        prompt_len=10,
        prompt_aligned_len=10,
        stride=8,
        temperature=temperature,
    )
    assert out["logits"].shape == (1, 40, model.vocab_size)

    # Two forward calls: segment 0 and tail.
    assert len(model.calls) == 2, f"Expected 2 calls, got {len(model.calls)}"
    assert model.calls[0]["seq_len"] == 30  # prompt_len + boundary
    assert model.calls[0]["had_past_kv"] is False
    assert model.calls[1]["seq_len"] == 11  # 40 - (10 + 20 - 1) = 11
    assert model.calls[1]["had_past_kv"] is True
    # After segment 0 + eviction, past_kv should be shorter:
    # 30 original - 8 stride - 1 boundary = 21 retained
    assert model.calls[1]["past_kv_len"] == 21, (
        f"Expected 21 retained KV entries, got {model.calls[1]['past_kv_len']}"
    )


def test_segmented_forward_slices_lora_token_counts_per_segment():
    model = LoraAwareMockModel()
    input_ids = torch.arange(40).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 40)
    lora_num_tokens = torch.tensor([30, 10], dtype=torch.int32)

    set_lora_num_tokens(lora_num_tokens.clone(), reset_reference=True)
    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[20],
        prompt_len=10,
        prompt_aligned_len=10,
        stride=8,
        temperature=temperature,
        lora_num_tokens=lora_num_tokens,
    )

    assert model.lora_token_counts_seen == [[30, 0], [1, 10]]
    assert get_lora_num_tokens().cpu().tolist() == [30, 10]


def test_multiple_boundaries():
    """prompt_len=10, boundaries=[20, 40, 60], completion_len=70, seq_len=80.

    Expected segments:
    - Seg 0: [0, 30): prompt + gen[0..19]
    - Seg 1: [29, 50): gen[19..39] (with overlap)
    - Seg 2: [49, 70): gen[39..59] (with overlap)
    - Tail:  [69, 80): gen[59..69] (with overlap)
    """
    model = MockModel()
    input_ids = torch.arange(80).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 80)

    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[20, 40, 60],
        prompt_len=10,
        prompt_aligned_len=10,
        stride=8,
        temperature=temperature,
    )
    assert len(model.calls) == 4, f"Expected 4 calls, got {len(model.calls)}"
    assert [c["seq_len"] for c in model.calls] == [30, 21, 21, 11], (
        f"Segment lengths: {[c['seq_len'] for c in model.calls]}"
    )
    # past_kv lengths after each eviction:
    # After seg 0: 30 - 8 - 1 = 21. Seg 1 feeds 21 entries.
    # After seg 1: (21 + 21) - 8 - 1 = 33. Seg 2 feeds 33 entries.
    # After seg 2: (33 + 21) - 8 - 1 = 45. Tail feeds 45 entries.
    assert model.calls[1]["past_kv_len"] == 21
    assert model.calls[2]["past_kv_len"] == 33
    assert model.calls[3]["past_kv_len"] == 45


def test_boundary_exactly_at_completion_end():
    """Last compaction fires at the very last token. No tail segment."""
    model = MockModel()
    input_ids = torch.arange(30).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 30)

    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[20],  # boundary at completion_len
        prompt_len=10,
        prompt_aligned_len=10,
        stride=8,
        temperature=temperature,
    )
    # Only 1 segment: the first. No tail because last_covered == seq_len.
    assert len(model.calls) == 1, f"Expected 1 call, got {len(model.calls)}"
    assert model.calls[0]["seq_len"] == 30


def test_prompt_aligned_len_differs_from_prompt_len():
    """prompt_len=50, prompt_aligned_len=64, boundaries=[30], stride=16.

    The drop should start at position 64 (NOT 50): the 14 gen tokens that
    sit in the tail of the last prompt block (positions 50..63) must be
    retained through every eviction.
    """
    model = MockModel()
    input_ids = torch.arange(200).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 200)

    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[30],
        prompt_len=50,
        prompt_aligned_len=64,
        stride=16,
        temperature=temperature,
    )
    # Segment 0: [0, 80), past_kv_len=0
    # Tail: [79, 200), past_kv fed
    assert len(model.calls) == 2
    assert model.calls[0]["seq_len"] == 80  # prompt_len + boundary[0]
    assert model.calls[1]["seq_len"] == 121  # 200 - 79

    # After eviction: kv had 80 entries. Drop [64, 64+16) = [64, 80) = 16
    # entries. Also drop boundary token at position 79, BUT position 79
    # is already in the stride drop range, so trim doesn't remove anything
    # extra. Retained: [0, 64), which is 64 entries.
    #
    # Wait: in my code, keys[prompt_aligned + stride : -trim] is [80 : -1]
    # = [80 : 79] which is empty. So retained = keys[:64] = 64 entries.
    assert model.calls[1]["past_kv_len"] == 64, (
        f"Expected 64 retained KV entries (prompt_aligned only), got "
        f"{model.calls[1]['past_kv_len']}"
    )


def test_short_assistant_content_clamped_stride():
    """Stride larger than available asst content. actual_stride = asst_len."""
    model = MockModel()
    input_ids = torch.arange(40).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 40)

    # Segment 0 processes [0, 15). kv_seq_len=15. prompt_aligned=10.
    # asst_len = 5. stride=100 -> actual_stride = min(100, 5) = 5.
    # Retained: keys[:10] + keys[10+5:-1] = keys[:10] + keys[15:14] (empty)
    # = 10 entries.
    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[5],
        prompt_len=10,
        prompt_aligned_len=10,
        stride=100,
        temperature=temperature,
    )
    assert len(model.calls) == 2
    assert model.calls[0]["seq_len"] == 15
    assert model.calls[1]["past_kv_len"] == 10


def test_fsdp_padding_runs_dummy_passes():
    """max_forward_passes > actual causes dummy forwards to keep FSDP sync."""
    model = MockModel()
    input_ids = torch.arange(30).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 30)

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[20],
        prompt_len=10,
        prompt_aligned_len=10,
        stride=8,
        temperature=temperature,
        max_forward_passes=5,  # 1 actual + 4 dummy
    )
    # 1 real segment (completion_len==boundary, no tail) + 4 dummies = 5 calls.
    assert len(model.calls) == 5, f"Expected 5 calls, got {len(model.calls)}"
    # The 4 dummy passes are 1-token forward passes on input_ids[:, :1].
    dummy_calls = model.calls[1:]
    for c in dummy_calls:
        assert c["seq_len"] == 1
    # Output shape unchanged.
    assert out["logits"].shape == (1, 30, model.vocab_size)


def test_empty_boundaries_use_explicit_positions_without_splicing(monkeypatch):
    """The prefill_trim dispatch is one eventless segment with no KV splice."""
    import kv_eviction.segmented_forward as segmented_forward_module

    def unexpected_splice(*args, **kwargs):
        raise AssertionError("_splice_dynamic_cache must not run")

    monkeypatch.setattr(
        segmented_forward_module,
        "_splice_dynamic_cache",
        unexpected_splice,
    )
    model = MockModel()

    def unexpected_hook(*args, **kwargs):
        raise AssertionError("empty-boundary forward must not install a cache hook")

    monkeypatch.setattr(model.model, "register_forward_hook", unexpected_hook)
    input_ids = torch.arange(10).unsqueeze(0)
    position_ids = torch.tensor([[0, 1, 4, 5, 6, 7, 8, 9, 10, 11]])
    temperature = torch.ones(1, 10)

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[],
        prompt_len=8,
        prompt_aligned_len=8,
        stride=1,
        temperature=temperature,
    )

    assert len(model.calls) == 1
    assert model.calls[0]["position_ids"] == position_ids[0].tolist()
    assert model.calls[0]["use_cache"] is False
    assert not model.calls[0]["had_past_kv"]
    assert model.config.use_cache is False
    assert out["logits"][0, :, 0].tolist() == position_ids[0].tolist()


class _CheckpointedBlock(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = torch.nn.Linear(hidden_size, hidden_size)
        self.calls = 0
        self.cache_mutations = 0
        self.cache: DynamicCache | None = None

    def forward(self, hidden_states):
        self.calls += 1
        if self.cache is not None:
            seq_len = hidden_states.shape[1]
            kv = hidden_states[:, None, :, :].contiguous()
            self.cache.update(kv, kv, 0)
            self.cache_mutations += 1
            assert self.cache.get_seq_length() == seq_len, (
                "activation-checkpoint recomputation mutated the same cache"
            )
        return torch.sin(self.proj(hidden_states))


class _ActivationCheckpointingModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 16, hidden_size: int = 8):
        super().__init__()
        self.config = MockConfig()
        self.embed = torch.nn.Embedding(vocab_size, hidden_size)
        self.block = _CheckpointedBlock(hidden_size)
        self.lm_head = torch.nn.Linear(hidden_size, vocab_size)
        self.use_cache_values: list[bool] = []

    def forward(self, input_ids, position_ids=None, use_cache=True):
        del position_ids
        self.use_cache_values.append(use_cache)
        self.block.cache = DynamicCache() if use_cache else None
        hidden = self.embed(input_ids)
        hidden = torch.utils.checkpoint.checkpoint(
            self.block,
            hidden,
            use_reentrant=False,
        )
        return {"logits": self.lm_head(hidden)}


def test_empty_boundaries_backward_does_not_mutate_checkpointed_cache():
    model = _ActivationCheckpointingModel()
    input_ids = torch.arange(8).unsqueeze(0)
    position_ids = torch.tensor([[0, 1, 4, 5, 6, 7, 8, 9]])
    loss_ranges = []

    def loss_fn(logits, start, end):
        loss_ranges.append((start, end, tuple(logits.shape)))
        return logits.float().square().mean()

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[],
        prompt_len=6,
        prompt_aligned_len=6,
        stride=1,
        temperature=torch.ones(1, 8),
        loss_fn=loss_fn,
    )

    assert out["n_segments"] == 1
    assert loss_ranges == [(0, 8, (1, 8, 16))]
    assert model.use_cache_values == [False]
    assert model.block.calls >= 2
    assert model.block.cache_mutations == 0
    assert model.lm_head.weight.grad is not None


def test_empty_boundaries_external_checkpoint_is_cache_free():
    model = _ActivationCheckpointingModel()
    input_ids = torch.arange(8).unsqueeze(0)
    position_ids = torch.tensor([[0, 1, 4, 5, 6, 7, 8, 9]])

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[],
        prompt_len=6,
        prompt_aligned_len=6,
        stride=1,
        temperature=torch.ones(1, 8),
        activation_checkpointing=True,
    )
    out["logits"].float().square().mean().backward()

    assert model.use_cache_values
    assert all(use_cache is False for use_cache in model.use_cache_values)
    assert model.block.cache_mutations == 0
    assert model.lm_head.weight.grad is not None


def test_empty_boundaries_runs_single_segment():
    """No-event samples route through one full segment in compaction runs."""
    model = MockModel()
    input_ids = torch.arange(30).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 30)

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[],
        prompt_len=10,
        prompt_aligned_len=10,
        stride=8,
        temperature=temperature,
    )

    assert len(model.calls) == 1
    assert model.calls[0]["seq_len"] == 30
    assert out["logits"].shape == (1, 30, model.vocab_size)


def test_attention_matching_empty_boundaries_replay_prefill_then_decode():
    """No-event AM samples should still match vLLM prefill/decode shape."""
    model = MockModel()
    input_ids = torch.arange(30).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 30)

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[],
        prompt_len=10,
        prompt_aligned_len=16,
        stride=8,
        temperature=temperature,
        compaction_strategy="attention_matching",
        attention_matching_zerobeta=True,
    )

    assert compute_num_segments(
        30,
        10,
        [],
        compaction_strategy="attention_matching",
        compaction_events=[],
    ) == 2
    assert len(model.calls) == 2
    assert [c["seq_len"] for c in model.calls] == [10, 20]
    assert model.calls[0]["past_kv_len"] == 0
    assert model.calls[1]["past_kv_len"] == 10
    assert out["logits"].shape == (1, 30, model.vocab_size)
    assert out["logits"][0, 9, 0].item() == 9.0
    assert out["logits"][0, 10, 0].item() == 10.0


def test_attention_matching_decode_chunk_size_splits_decode_replay():
    """AM can replay post-prompt decode one token at a time for KL checks."""
    model = MockModel()
    input_ids = torch.arange(14).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 14)

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[],
        prompt_len=10,
        prompt_aligned_len=16,
        stride=8,
        temperature=temperature,
        compaction_strategy="attention_matching",
        attention_matching_zerobeta=True,
        attention_matching_decode_chunk_size=1,
    )

    assert compute_num_segments(
        14,
        10,
        [],
        compaction_strategy="attention_matching",
        compaction_events=[],
        attention_matching_decode_chunk_size=1,
    ) == 5
    assert [c["seq_len"] for c in model.calls] == [10, 1, 1, 1, 1]
    assert [c["past_kv_len"] for c in model.calls] == [0, 10, 11, 12, 13]
    assert out["logits"].shape == (1, 14, model.vocab_size)


def test_asst_len_equals_stride_fully_aligned():
    """Regression test for the canonical 'needs_compaction just fired on a
    fully-filled evict block' case: after segment 0 runs, the post-prompt
    KV length equals stride exactly. Verify:
    1. The retained KV is prompt_aligned_len entries (prompt block only).
    2. The eviction drops the boundary token (index kv_seq_len - 1).
    3. The tail segment re-feeds the boundary token under post-eviction
       context.
    (This case was flagged as a potential off-by-one by RSA review R2; the
    tensor is correct but the log-message formula was off by 1 in this
    exact edge case, now fixed.)
    """
    model = MockModel()
    # prompt_len == prompt_aligned_len == 64 (block-aligned prompt), stride=16.
    # boundary=16 means seg 0 processes input_ids[0:80], kv_seq_len=80,
    # asst_len=16=stride.
    input_ids = torch.arange(120).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 120)

    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[16],
        prompt_len=64,
        prompt_aligned_len=64,
        stride=16,
        temperature=temperature,
    )
    # 2 calls: seg 0 and tail.
    assert len(model.calls) == 2
    assert model.calls[0]["seq_len"] == 80
    # After eviction, retained KV = prompt_aligned_len (64) entries. The
    # boundary token (index 79) is dropped via the stride range [64, 80).
    # Trim -1 overlaps with the stride range but doesn't remove anything
    # extra. Accounting previously over-subtracted; now fixed.
    assert model.calls[1]["past_kv_len"] == 64, (
        f"Expected 64 retained KV entries (prompt_aligned only), got "
        f"{model.calls[1]['past_kv_len']}"
    )
    # Tail starts at index prompt_len + boundary - 1 = 79 (the boundary
    # token, re-fed under post-eviction context).
    # seq_len - 79 = 41 tokens in the tail segment.
    assert model.calls[1]["seq_len"] == 41


def test_asst_len_equals_stride_large_stride():
    """Same as above but with stride == 64 blocks to exercise R2's exact
    scenario (pa_len=64, stride=64). Confirms the fix works for any
    stride size."""
    model = MockModel()
    input_ids = torch.arange(200).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 200)

    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[64],
        prompt_len=64,
        prompt_aligned_len=64,
        stride=64,
        temperature=temperature,
    )
    assert len(model.calls) == 2
    assert model.calls[0]["seq_len"] == 128  # prompt_len + boundary
    # asst_len=64=stride. Retained = prompt_aligned_len = 64 entries.
    assert model.calls[1]["past_kv_len"] == 64


def test_attention_matching_pre_sample_refeeds_boundary_token():
    """AM replay re-feeds the boundary token under compacted KV.

    The current fork discards the provisional sampled token/logprob when AM
    fires. The trainer mirrors the real emitted trace: segment 0 builds the
    source KV for AM, AM physically omits the final KV entry, and segment 1
    re-feeds the boundary token so its logit is recomputed under post-AM KV.
    """
    model = MockModel()
    input_ids = torch.arange(30).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 30)
    prompt_len = 10
    boundary = 12
    source_len = prompt_len + boundary
    protected = 10
    synthetic = 2
    exact = 5
    target_len = protected + synthetic + exact
    tokens_evicted = source_len - target_len

    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[boundary],
        prompt_len=prompt_len,
        prompt_aligned_len=prompt_len,
        stride=synthetic,
        temperature=temperature,
        compaction_strategy="attention_matching",
        compaction_events=[{
            "num_output_tokens_at_compaction": boundary,
            "tokens_evicted": tokens_evicted,
            "position_offset_after": tokens_evicted,
            "num_prompt_tokens": prompt_len,
            "source_len": source_len,
            "target_len": target_len,
            "protected_prefix_len": protected,
            "synthetic_prefix_len": synthetic,
            "exact_kept_tokens": exact,
            "attention_matching_query_source": "random_queries",
            "attention_matching_max_queries_per_kv_head": 2,
            "attention_matching_query_seed": 123,
            "attention_matching_selected_indices": _am_selected_indices(
                synthetic=synthetic
            ),
            "attention_matching_zerobeta": True,
            "attention_matching_pre_sample": True,
        }],
        attention_matching_query_source="random_queries",
        attention_matching_max_queries_per_kv_head=2,
        attention_matching_zerobeta=True,
    )

    assert len(model.calls) == 2
    assert model.calls[0]["seq_len"] == source_len
    assert model.calls[1]["past_kv_len"] == target_len - 1
    assert model.calls[1]["position_ids"][0] == prompt_len + boundary - 1


def test_attention_matching_prompt_partial_cache_hit_warms_uncached_suffix():
    """Prompt-time AM may reuse only the protected+synthetic cached prefix.

    The missing exact tail is then re-fed as warmup under the compacted cache.
    This is the trainer-side mirror of a vLLM partial compressed-prefix hit.
    """
    model = MockModel()
    input_ids = torch.arange(16).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 16)
    prompt_len = 12
    boundary = 0
    source_len = prompt_len
    protected = 2
    synthetic = 2
    exact = 6
    target_len = protected + synthetic + exact
    cache_hit_tokens = protected + synthetic
    tokens_evicted = source_len - target_len

    def loss_fn(seg_logits, _start, _end):
        return seg_logits.mean() * 0.0

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[boundary],
        prompt_len=prompt_len,
        prompt_aligned_len=prompt_len,
        stride=synthetic,
        temperature=temperature,
        compaction_strategy="attention_matching",
        compaction_events=[{
            "num_output_tokens_at_compaction": boundary,
            "tokens_evicted": tokens_evicted,
            "position_offset_after": tokens_evicted,
            "num_prompt_tokens": prompt_len,
            "source_len": source_len,
            "target_len": target_len,
            "protected_prefix_len": protected,
            "synthetic_prefix_len": synthetic,
            "exact_kept_tokens": exact,
            "attention_matching_query_source": "random_queries",
            "attention_matching_max_queries_per_kv_head": 2,
            "attention_matching_query_seed": 123,
            "attention_matching_selected_indices": _am_selected_indices(
                synthetic=synthetic
            ),
            "attention_matching_zerobeta": True,
            "attention_matching_pre_sample": True,
            "attention_matching_replay_steps": [{
                "source_len": source_len,
                "target_len": target_len,
                "protected_prefix_len": protected,
                "synthetic_prefix_len": synthetic,
                "exact_kept_tokens": exact,
                "attention_matching_query_seed": 123,
                "attention_matching_selected_indices": _am_selected_indices(
                    synthetic=synthetic
                ),
            }],
            "attention_matching_cache_hit_tokens": cache_hit_tokens,
        }],
        attention_matching_query_source="random_queries",
        attention_matching_max_queries_per_kv_head=2,
        attention_matching_zerobeta=True,
        loss_fn=loss_fn,
    )

    assert out["n_segments"] == 2
    assert len(model.model.calls) == 2
    assert model.model.calls[0]["seq_len"] == prompt_len
    assert model.model.calls[1]["past_kv_len"] == cache_hit_tokens
    # target_len - cache_hit_tokens == 6 uncached prompt tokens are warmed up.
    assert model.model.calls[1]["position_ids"][0] == prompt_len - 6
    assert model.model.calls[1]["seq_len"] == 10


def test_detached_prompt_prefill_uses_root_forward_for_dtensor_backbone(monkeypatch):
    """FSDP2 sharded embeddings require root-module input handling.

    The production optimization still avoids autograd for prompt-only AM
    prefill, but it must not directly call `model.model(...)` when that would
    bypass FSDP2's Tensor -> DTensor input path.
    """
    monkeypatch.setattr(
        segmented_forward_mod,
        "_backbone_embedding_weight_is_dtensor",
        lambda _backbone: True,
    )
    model = RootOnlyMockModel()
    input_ids = torch.arange(16).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 16)
    prompt_len = 12
    boundary = 0
    source_len = prompt_len
    protected = 2
    synthetic = 2
    exact = 6
    target_len = protected + synthetic + exact
    cache_hit_tokens = protected + synthetic
    tokens_evicted = source_len - target_len

    def loss_fn(seg_logits, _start, _end):
        return seg_logits.mean() * 0.0

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[boundary],
        prompt_len=prompt_len,
        prompt_aligned_len=prompt_len,
        stride=synthetic,
        temperature=temperature,
        compaction_strategy="attention_matching",
        compaction_events=[{
            "num_output_tokens_at_compaction": boundary,
            "tokens_evicted": tokens_evicted,
            "position_offset_after": tokens_evicted,
            "num_prompt_tokens": prompt_len,
            "source_len": source_len,
            "target_len": target_len,
            "protected_prefix_len": protected,
            "synthetic_prefix_len": synthetic,
            "exact_kept_tokens": exact,
            "attention_matching_query_source": "random_queries",
            "attention_matching_max_queries_per_kv_head": 2,
            "attention_matching_query_seed": 123,
            "attention_matching_selected_indices": _am_selected_indices(
                synthetic=synthetic
            ),
            "attention_matching_zerobeta": True,
            "attention_matching_pre_sample": True,
            "attention_matching_replay_steps": [{
                "source_len": source_len,
                "target_len": target_len,
                "protected_prefix_len": protected,
                "synthetic_prefix_len": synthetic,
                "exact_kept_tokens": exact,
                "attention_matching_query_seed": 123,
                "attention_matching_selected_indices": _am_selected_indices(
                    synthetic=synthetic
                ),
            }],
            "attention_matching_cache_hit_tokens": cache_hit_tokens,
        }],
        attention_matching_query_source="random_queries",
        attention_matching_max_queries_per_kv_head=2,
        attention_matching_zerobeta=True,
        loss_fn=loss_fn,
    )

    assert out["n_segments"] == 2
    assert len(model.calls) == 2
    assert model.prefill_logits_to_keep_seen[0] == 1


def test_attention_matching_turn_cache_hit_warms_uncached_suffix_after_boundary():
    """Cross-turn AM cache hits can occur after earlier sampled tokens.

    The cached compressed state may cover only part of the post-AM physical
    prompt. The trainer must warm the missing exact tail immediately before the
    turn boundary. The boundary logit itself is owned by the post-AM segment
    because the provisional pre-AM sample/logprob is discarded.
    """
    model = MockModel()
    input_ids = torch.arange(20).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 20)
    prompt_len = 10
    boundary = 4
    source_len = prompt_len + boundary
    protected = 2
    synthetic = 2
    exact = 6
    target_len = protected + synthetic + exact
    cache_hit_tokens = 8
    tokens_evicted = source_len - target_len

    segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[boundary],
        prompt_len=prompt_len,
        prompt_aligned_len=prompt_len,
        stride=synthetic,
        temperature=temperature,
        compaction_strategy="attention_matching",
        compaction_events=[{
            "num_output_tokens_at_compaction": boundary,
            "tokens_evicted": tokens_evicted,
            "position_offset_after": tokens_evicted,
            "num_prompt_tokens": prompt_len,
            "source_len": source_len,
            "target_len": target_len,
            "protected_prefix_len": protected,
            "synthetic_prefix_len": synthetic,
            "exact_kept_tokens": exact,
            "attention_matching_query_source": "random_queries",
            "attention_matching_max_queries_per_kv_head": 2,
            "attention_matching_query_seed": 123,
            "attention_matching_selected_indices": _am_selected_indices(
                synthetic=synthetic
            ),
            "attention_matching_zerobeta": True,
            "attention_matching_pre_sample": True,
            "attention_matching_replay_steps": [{
                "source_len": source_len,
                "target_len": target_len,
                "protected_prefix_len": protected,
                "synthetic_prefix_len": synthetic,
                "exact_kept_tokens": exact,
                "attention_matching_query_seed": 123,
                "attention_matching_selected_indices": _am_selected_indices(
                    synthetic=synthetic
                ),
            }],
            "attention_matching_cache_hit_tokens": cache_hit_tokens,
        }],
        attention_matching_query_source="random_queries",
        attention_matching_max_queries_per_kv_head=2,
        attention_matching_zerobeta=True,
    )

    assert len(model.calls) == 2
    assert model.calls[0]["seq_len"] == source_len
    assert model.calls[1]["past_kv_len"] == cache_hit_tokens
    assert model.calls[1]["position_ids"][0] == prompt_len + boundary - 2


def test_attention_matching_runtime_pre_sample_owns_boundary_logit_post_am():
    """Runtime pre-sample AM keeps the emitted boundary logit post-AM.

    The worker may compute a provisional boundary logit before AM, but that
    provisional token/logprob is discarded. The next forward recomputes the
    boundary logit under compacted KV, so trainer replay must transfer
    ownership of that logit to the post-AM segment.
    """
    model = MockModel()
    input_ids = torch.arange(18).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 18)
    prompt_len = 10
    boundary = 3
    source_len = prompt_len + boundary
    protected = 2
    synthetic = 2
    exact = 6
    target_len = protected + synthetic + exact
    tokens_evicted = source_len - target_len
    owned_ranges: list[tuple[int, int]] = []

    def loss_fn(seg_logits, start, end):
        owned_ranges.append((start, end))
        return seg_logits.mean() * 0.0

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[boundary],
        prompt_len=prompt_len,
        prompt_aligned_len=prompt_len,
        stride=synthetic,
        temperature=temperature,
        compaction_strategy="attention_matching",
        compaction_events=[{
            "num_output_tokens_at_compaction": boundary,
            "tokens_evicted": tokens_evicted,
            "position_offset_after": tokens_evicted,
            "num_prompt_tokens": prompt_len,
            "source_len": source_len,
            "target_len": target_len,
            "protected_prefix_len": protected,
            "synthetic_prefix_len": synthetic,
            "exact_kept_tokens": exact,
            "attention_matching_query_source": "random_queries",
            "attention_matching_max_queries_per_kv_head": 2,
            "attention_matching_query_seed": 123,
            "attention_matching_selected_indices": _am_selected_indices(
                synthetic=synthetic
            ),
            "attention_matching_zerobeta": True,
            "attention_matching_pre_sample": True,
            "attention_matching_replay_steps": [{
                "source_len": source_len,
                "target_len": target_len,
                "protected_prefix_len": protected,
                "synthetic_prefix_len": synthetic,
                "exact_kept_tokens": exact,
                "attention_matching_query_seed": 123,
                "attention_matching_selected_indices": _am_selected_indices(
                    synthetic=synthetic
                ),
            }],
        }],
        attention_matching_query_source="random_queries",
        attention_matching_max_queries_per_kv_head=2,
        attention_matching_zerobeta=True,
        loss_fn=loss_fn,
    )

    assert out["n_segments"] == 2
    assert len(model.calls) == 2
    assert model.calls[0]["seq_len"] == source_len
    assert model.calls[1]["past_kv_len"] == target_len - 1
    assert model.calls[1]["position_ids"][0] == prompt_len + boundary - 1
    assert owned_ranges == [
        (0, prompt_len + boundary - 1),
        (prompt_len + boundary - 1, 18),
    ]


def test_temperature_scaling_applied():
    """Per-token temperature scales logits."""
    model = MockModel()
    input_ids = torch.arange(30).unsqueeze(0)
    position_ids = input_ids.clone()
    # temperature=2.0 everywhere -> logits halved
    temperature = torch.full((1, 30), 2.0)

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[20],
        prompt_len=10,
        prompt_aligned_len=10,
        stride=8,
        temperature=temperature,
    )
    # Our MockModel returns logits[i, :] = position_id[i]. After scaling by
    # 2.0, logits[i, :] = position_id[i] / 2.
    # Position 5 should yield 5.0 / 2.0 = 2.5.
    assert out["logits"][0, 5, 0].item() == 2.5
    assert out["logits"][0, 29, 0].item() == 14.5


def test_full_bptt_dummy_padding_builds_one_backward_graph():
    """Full-BPTT mode pads dummy forward graphs before the single backward."""
    model = MockModel()
    input_ids = torch.arange(40).unsqueeze(0)
    position_ids = input_ids.clone()
    temperature = torch.ones(1, 40)

    def loss_fn(seg_logits, _start, _end):
        return seg_logits.float().mean()

    out = segmented_forward(
        model=model,
        input_ids=input_ids,
        position_ids=position_ids,
        segment_boundaries=[10, 20],
        prompt_len=10,
        prompt_aligned_len=10,
        stride=8,
        temperature=temperature,
        max_forward_passes=5,
        loss_fn=loss_fn,
        bptt_segments=None,
    )

    assert out["n_segments"] == 3
    assert len(model.calls) == 5
    dummy_calls = model.calls[3:]
    assert [c["seq_len"] for c in dummy_calls] == [1, 1]
    assert model.logit_scale.grad is not None
    assert torch.isfinite(model.logit_scale.grad)
