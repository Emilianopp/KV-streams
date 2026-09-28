# SPDX-License-Identifier: Apache-2.0
"""Unit tests for per_call_segmented_forward (persistent-cache path).

After the single-forward-pre-eviction refactor (plans/single_forward_pre_eviction.md
Phase 5), per_call_segmented_forward runs ONE HF forward per call against a
DynamicCache carried across calls. Admission events are handled inline via
eviction-aware position_ids — no two-phase split, no cache splice.

These tests use a mock HF-style model that returns deterministic logits
(logits[i] = position_ids[i] broadcast to vocab) so we can verify:
  1. Owned ranges partition [0, full_seq_len).
  2. compute_num_per_call_forwards matches the actual split prefill/tail count.
  3. Eviction-aware position_ids match vLLM's RoPE frame after admission.
  4. The persistent cache carries K/V across calls.
  5. Dummy passes pad to max_forward_passes for FSDP2 sync.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from kv_eviction.segmented_forward import (
    _build_flex_mask_writer_timeline,
    _build_pre_trim_plan,
    batched_flex_mask_segmented_forward,
    compute_flex_mask_writer_len,
    compute_num_per_call_forwards,
    compute_per_call_bptt_window_forward_counts,
    packed_flex_mask_segmented_forward,
    per_call_segmented_forward,
)


@dataclass
class _CallSpec:
    """Minimal CallWire shape (we only access the fields per_call uses)."""

    submitted_prompt_ids: list[int]
    completion_ids: list[int]
    compaction_events: list = None  # default empty in CallWire schema

    def __post_init__(self):
        if self.compaction_events is None:
            self.compaction_events = []


@dataclass
class _Event:
    num_output_tokens_at_compaction: int = 0
    evict_start: int = 0
    tokens_evicted: int = 0
    new_user_fragment_len: int = 0
    position_offset_after: int = 0
    num_prompt_tokens: int = 0


class _RecordingModel(torch.nn.Module):
    """Mock HF model: returns logits[i] = position_ids[i] (broadcast to vocab).

    Records every forward call's seq_len, position_ids, and incoming
    cache length so we can assert on persistent-cache invariants.
    """

    def __init__(self, vocab_size: int = 32, num_layers: int = 2,
                 num_heads: int = 1, head_dim: int = 4):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.calls: list[dict] = []
        # A tiny trainable parameter so .backward() has somewhere to flow.
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(
        self,
        input_ids,
        position_ids,
        past_key_values=None,
        use_cache=False,
        **_kwargs,
    ):
        seq_len = input_ids.shape[1]
        base = position_ids.float().unsqueeze(-1).expand(-1, -1, self.vocab_size)
        logits = base + self._dummy

        # Capture pre-extension cache length per layer (for assertions).
        pre_cache_len = 0
        if past_key_values is not None and hasattr(past_key_values, "layers") \
                and past_key_values.layers:
            pre_cache_len = past_key_values.layers[0].keys.shape[2]
        elif past_key_values is not None and hasattr(past_key_values, "key_cache") \
                and past_key_values.key_cache:
            pre_cache_len = past_key_values.key_cache[0].shape[2]

        self.calls.append({
            "seq_len": seq_len,
            "position_ids": position_ids[0].tolist(),
            "input_ids": input_ids[0].tolist(),
            "pre_cache_len": pre_cache_len,
            "had_past_kv": past_key_values is not None,
            "attention_mask_is_none": _kwargs.get("attention_mask") is None,
        })

        # Append K/V to the cache so subsequent calls inherit them.
        if use_cache and past_key_values is not None:
            for layer in range(self.num_layers):
                k = position_ids.float().view(1, 1, seq_len, 1).expand(
                    1, self.num_heads, seq_len, self.head_dim
                ).contiguous()
                v = (position_ids.float() + 100).view(1, 1, seq_len, 1).expand(
                    1, self.num_heads, seq_len, self.head_dim
                ).contiguous()
                past_key_values.update(k, v, layer)

        return {"logits": logits, "past_key_values": past_key_values}


class _CrossCallGradModel(torch.nn.Module):
    """Mock model whose current logits depend on previously written K rows."""

    def __init__(self, vocab_size: int = 8, num_layers: int = 1,
                 num_heads: int = 1, head_dim: int = 1):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.cache_scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, position_ids, past_key_values=None, use_cache=False):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        past_signal = torch.zeros((), device=device)
        if past_key_values is not None and hasattr(past_key_values, "layers") \
                and past_key_values.layers:
            past_signal = past_key_values.layers[0].keys.sum()
        elif past_key_values is not None and hasattr(past_key_values, "key_cache") \
                and past_key_values.key_cache:
            past_signal = past_key_values.key_cache[0].sum()

        logits = past_signal.reshape(1, 1, 1).expand(
            batch, seq_len, self.vocab_size
        )

        if use_cache and past_key_values is not None:
            for layer in range(self.num_layers):
                k = (
                    self.cache_scale
                    * position_ids.float().view(batch, 1, seq_len, 1)
                ).expand(batch, self.num_heads, seq_len, self.head_dim).contiguous()
                v = torch.zeros_like(k)
                past_key_values.update(k, v, layer)

        return {"logits": logits, "past_key_values": past_key_values}


class FusedOutputLinear(torch.nn.Linear):
    """Test double for PrimeRL's selected-logprob fused head."""

    def forward(self, hidden_states, labels=None, temperature=None):
        assert labels is not None, "FusedOutputLinear requires labels"
        return {
            "logprobs": hidden_states[..., 0],
            "entropy": torch.zeros_like(hidden_states[..., 0]),
        }


class _PrimeFusedHeadModel(torch.nn.Module):
    """Patched-HF-style model whose lm_head normally refuses full logits."""

    def __init__(self, vocab_size: int = 8):
        super().__init__()
        self.vocab_size = vocab_size
        self.calls: list[dict] = []
        self._dummy = torch.nn.Parameter(torch.zeros(1))
        self.lm_head = FusedOutputLinear(1, vocab_size, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.arange(vocab_size).float().unsqueeze(1))

    def forward(
        self,
        input_ids,
        position_ids,
        labels=None,
        temperature=None,
        **kwargs,
    ):
        hidden = position_ids.float().unsqueeze(-1) + self._dummy
        self.calls.append({
            "seq_len": input_ids.shape[1],
            "attention_mask_is_none": kwargs.get("attention_mask") is None,
        })
        return self.lm_head(hidden, labels=labels, temperature=temperature)


def _merged_from_calls(calls: list[_CallSpec]) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the merged sequence for a list of EXTENSION calls.

    For no-admission rollouts, each call's submitted_prompt is a strict
    prefix of the merged sample. The merged sequence equals the LAST
    call's (submitted_prompt + completion).
    """
    last = calls[-1]
    merged = list(last.submitted_prompt_ids) + list(last.completion_ids)
    input_ids = torch.tensor(merged, dtype=torch.long).unsqueeze(0)
    position_ids = torch.arange(len(merged), dtype=torch.long).unsqueeze(0)
    return input_ids, position_ids, len(merged)


# ─── compute_num_per_call_forwards ────────────────────────────────────


def test_compute_num_per_call_forwards_no_admission_splits_prompt_tail():
    """Default regular path splits prompt prefill from completion tail."""
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3]),
        _CallSpec(submitted_prompt_ids=[1, 2, 3, 4], completion_ids=[5]),
        _CallSpec(submitted_prompt_ids=[1, 2, 3, 4, 5, 6], completion_ids=[7]),
    ]
    assert compute_num_per_call_forwards(calls) == 6


def test_compute_num_per_call_forwards_with_admission_splits_main():
    """Admission calls still split newly submitted prompt from completion."""
    admission_event = _Event(
        evict_start=1,
        tokens_evicted=1,
        new_user_fragment_len=1,
        position_offset_after=1,
        num_prompt_tokens=4,
    )
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3]),
        _CallSpec(
            submitted_prompt_ids=[1, 2, 3, 4],
            completion_ids=[5],
            compaction_events=[admission_event],
        ),
    ]
    assert compute_num_per_call_forwards(calls) == 4


def test_compute_num_per_call_forwards_mixed_counts_total():
    """Mixed admission/non-admission calls: split count is summed."""
    admission_event = _Event(
        evict_start=0,
        tokens_evicted=1,
        new_user_fragment_len=1,
        position_offset_after=1,
        num_prompt_tokens=2,
    )
    calls = [
        _CallSpec(submitted_prompt_ids=[1], completion_ids=[2]),
        _CallSpec(
            submitted_prompt_ids=[1, 2], completion_ids=[3],
            compaction_events=[admission_event],
        ),
        _CallSpec(submitted_prompt_ids=[1, 2, 3], completion_ids=[4]),
        _CallSpec(
            submitted_prompt_ids=[1, 2, 3, 4], completion_ids=[5],
            compaction_events=[admission_event],
        ),
    ]
    assert compute_num_per_call_forwards(calls) == 6


def test_compute_num_per_call_forwards_with_midgen_only():
    """Mid-gen samples are routed to legacy before per-call counting."""
    midgen_event = _Event(num_output_tokens_at_compaction=16)
    calls = [
        _CallSpec(
            submitted_prompt_ids=[1, 2], completion_ids=[3],
            compaction_events=[midgen_event],
        ),
    ]
    with pytest.raises(AssertionError, match="mid-generation event"):
        compute_num_per_call_forwards(calls)


def test_compute_per_call_bptt_window_forward_counts_groups_calls():
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3]),
        _CallSpec(submitted_prompt_ids=[1, 2, 3, 4], completion_ids=[5]),
        _CallSpec(submitted_prompt_ids=[1, 2, 3, 4, 5, 6], completion_ids=[7]),
    ]

    assert compute_per_call_bptt_window_forward_counts(calls, 2) == [4, 2]
    assert compute_per_call_bptt_window_forward_counts(calls, None) == [6]
    assert compute_per_call_bptt_window_forward_counts(calls, -1) == [6]


# ─── flex-mask writer timeline ────────────────────────────────────────


def test_flex_mask_timeline_no_admission_is_plain_causal_chain():
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3]),
        _CallSpec(submitted_prompt_ids=[1, 2, 3, 4], completion_ids=[5]),
    ]

    timeline = _build_flex_mask_writer_timeline(calls)

    assert timeline.input_ids == [1, 2, 3, 4, 5]
    assert timeline.position_ids == [0, 1, 2, 3, 4]
    assert timeline.death_indices == [5, 5, 5, 5, 5]
    assert timeline.loss_ranges == [(0, 3, 0, 3), (3, 5, 3, 5)]
    assert compute_flex_mask_writer_len(calls) == 5


def test_batched_flex_mask_segmented_forward_routes_rows_and_ranges():
    pytest.importorskip("torch.nn.attention.flex_attention")
    calls_a = [_CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3])]
    calls_b = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3]),
        _CallSpec(submitted_prompt_ids=[1, 2, 3, 4], completion_ids=[5]),
    ]
    merged_ids = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
    )
    pos_ids = torch.tensor(
        [
            [0, 1, 2, 0, 1],
            [0, 1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    model = _RecordingModel(vocab_size=8)
    seen: list[tuple[int, int, int, tuple[int, ...]]] = []

    def fake_loss(seg_logits, full_start, full_end, batch_idx):
        seen.append((batch_idx, full_start, full_end, tuple(seg_logits.shape)))
        return seg_logits.float().sum() * 0.0

    batched_flex_mask_segmented_forward(
        model,
        [calls_a, calls_b],
        merged_input_ids=merged_ids,
        merged_position_ids=pos_ids,
        loss_fn=fake_loss,
        max_forward_passes=1,
        device=torch.device("cpu"),
    )

    assert seen == [
        (0, 0, 3, (1, 3, 8)),
        (1, 0, 3, (1, 3, 8)),
        (1, 3, 5, (1, 2, 8)),
    ]
    assert len(model.calls) == 1
    assert model.calls[0]["seq_len"] == 5


def test_packed_flex_mask_segmented_forward_supports_distillation_hooks():
    pytest.importorskip("torch.nn.attention.flex_attention")
    calls_a = [_CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3])]
    calls_b = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3]),
        _CallSpec(submitted_prompt_ids=[1, 2, 3, 4], completion_ids=[5]),
    ]
    merged_ids = torch.tensor(
        [
            [1, 2, 3, 0, 0],
            [1, 2, 3, 4, 5],
        ],
        dtype=torch.long,
    )
    pos_ids = torch.tensor(
        [
            [0, 1, 2, 0, 1],
            [0, 1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    model = _RecordingModel(vocab_size=8)
    prepared: list[tuple[int, int, int, tuple[int, ...]]] = []
    teacher_seen: list[tuple[int, int, int, tuple[int, ...]]] = []
    distill_seen: list[tuple[int, int, int, tuple[int, ...]]] = []
    loss_seen: list[tuple[int, int, int, tuple[int, ...]]] = []

    def prepare_fn(seg_logits, full_start, full_end, batch_idx):
        prepared.append((batch_idx, full_start, full_end, tuple(seg_logits.shape)))

    def teacher_fn(seg_logits, full_start, full_end, batch_idx):
        teacher_seen.append((batch_idx, full_start, full_end, tuple(seg_logits.shape)))

    def distill_loss_fn(seg_logits, full_start, full_end, batch_idx):
        distill_seen.append((batch_idx, full_start, full_end, tuple(seg_logits.shape)))
        return seg_logits.float().sum() * 0.0

    def loss_fn(seg_logits, full_start, full_end, batch_idx):
        loss_seen.append((batch_idx, full_start, full_end, tuple(seg_logits.shape)))
        return seg_logits.float().sum() * 0.0

    packed_flex_mask_segmented_forward(
        model,
        [calls_a, calls_b],
        merged_input_ids=merged_ids,
        merged_position_ids=pos_ids,
        loss_fn=loss_fn,
        distillation_prepare_fn=prepare_fn,
        distillation_teacher_fn=teacher_fn,
        distillation_loss_fn=distill_loss_fn,
        max_forward_passes=1,
        device=torch.device("cpu"),
    )

    expected = [
        (0, 0, 3, (1, 3, 8)),
        (1, 0, 3, (1, 3, 8)),
        (1, 3, 5, (1, 2, 8)),
    ]
    assert prepared == expected
    assert teacher_seen == expected
    assert distill_seen == expected
    assert loss_seen == expected
    assert len(model.calls) == 3
    assert [call["seq_len"] for call in model.calls] == [8, 8, 8]
    assert all(not call["attention_mask_is_none"] for call in model.calls)


def test_packed_flex_distillation_bypasses_selected_logprob_fused_head():
    pytest.importorskip("torch.nn.attention.flex_attention")
    calls = [[_CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3])]]
    merged_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    pos_ids = torch.tensor([[0, 1, 2]], dtype=torch.long)
    model = _PrimeFusedHeadModel(vocab_size=8)
    seen: list[tuple[int, int, int, tuple[int, ...]]] = []

    def prepare_fn(seg_logits, full_start, full_end, batch_idx):
        seen.append((batch_idx, full_start, full_end, tuple(seg_logits.shape)))

    def teacher_fn(seg_logits, full_start, full_end, batch_idx):
        seen.append((batch_idx, full_start, full_end, tuple(seg_logits.shape)))

    def distill_loss_fn(seg_logits, full_start, full_end, batch_idx):
        seen.append((batch_idx, full_start, full_end, tuple(seg_logits.shape)))
        return seg_logits.float().sum() * 0.0

    def loss_fn(seg_logits, full_start, full_end, batch_idx):
        seen.append((batch_idx, full_start, full_end, tuple(seg_logits.shape)))
        return seg_logits.float().sum() * 0.0

    packed_flex_mask_segmented_forward(
        model,
        calls,
        merged_input_ids=merged_ids,
        merged_position_ids=pos_ids,
        loss_fn=loss_fn,
        distillation_prepare_fn=prepare_fn,
        distillation_teacher_fn=teacher_fn,
        distillation_loss_fn=distill_loss_fn,
        max_forward_passes=1,
        device=torch.device("cpu"),
    )

    assert seen == [(0, 0, 3, (1, 3, 8))] * 4
    assert len(model.calls) == 3
    assert all(not call["attention_mask_is_none"] for call in model.calls)


def test_packed_flex_mask_segmented_forward_supports_direct_distillation_fn():
    pytest.importorskip("torch.nn.attention.flex_attention")
    calls_a = [_CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3])]
    calls_b = [_CallSpec(submitted_prompt_ids=[4, 5], completion_ids=[6])]
    merged_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    pos_ids = torch.tensor([[0, 1, 2], [0, 1, 2]], dtype=torch.long)
    model = _RecordingModel(vocab_size=8)
    distill_seen: list[tuple[int, int, int, tuple[int, ...], tuple[int, ...]]] = []

    def distillation_fn(student_logits, teacher_logits, full_start, full_end, batch_idx):
        distill_seen.append(
            (
                batch_idx,
                full_start,
                full_end,
                tuple(student_logits.shape),
                tuple(teacher_logits.shape),
            )
        )
        return student_logits.float().sum() * 0.0

    def loss_fn(seg_logits, _full_start, _full_end, _batch_idx):
        return seg_logits.float().sum() * 0.0

    packed_flex_mask_segmented_forward(
        model,
        [calls_a, calls_b],
        merged_input_ids=merged_ids,
        merged_position_ids=pos_ids,
        loss_fn=loss_fn,
        distillation_fn=distillation_fn,
        max_forward_passes=1,
        device=torch.device("cpu"),
    )

    assert distill_seen == [
        (0, 0, 3, (1, 3, 8), (1, 3, 8)),
        (1, 0, 3, (1, 3, 8), (1, 3, 8)),
    ]
    assert len(model.calls) == 2
    assert [call["seq_len"] for call in model.calls] == [6, 6]
    assert all(not call["attention_mask_is_none"] for call in model.calls)


def test_flex_mask_timeline_b2b_admission_kills_prior_cache_rows():
    admission_event = _Event(
        evict_start=1,
        tokens_evicted=1,
        new_user_fragment_len=1,
        position_offset_after=1,
        num_prompt_tokens=4,
    )
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3]),
        _CallSpec(
            submitted_prompt_ids=[1, 2, 3, 4],
            completion_ids=[5],
            compaction_events=[admission_event],
        ),
    ]

    timeline = _build_flex_mask_writer_timeline(calls)

    assert timeline.input_ids == [1, 2, 3, 4, 5]
    assert timeline.position_ids == [0, 1, 2, 3, 4]
    assert timeline.death_indices == [5, 3, 5, 5, 5]
    assert timeline.loss_ranges == [(0, 3, 0, 3), (3, 5, 3, 5)]


def test_flex_mask_timeline_b2a_admission_keeps_prefill_context_until_splice():
    admission_event = _Event(
        evict_start=1,
        tokens_evicted=2,
        new_user_fragment_len=1,
        position_offset_after=2,
        num_prompt_tokens=4,
    )
    calls = [
        _CallSpec(
            submitted_prompt_ids=[1, 2, 3, 4],
            completion_ids=[5],
            compaction_events=[admission_event],
        )
    ]

    timeline = _build_flex_mask_writer_timeline(calls)

    assert timeline.input_ids == [1, 2, 3, 4, 5]
    assert timeline.position_ids == [0, 1, 2, 3, 4]
    assert timeline.death_indices == [5, 4, 4, 5, 5]
    assert timeline.loss_ranges == [(0, 5, 0, 5)]


def test_flex_mask_timeline_prefix_replay_then_b2b_admission_matches_plan():
    synthetic_prefix_event = _Event(
        evict_start=1,
        tokens_evicted=0,
        new_user_fragment_len=4,
        position_offset_after=0,
        num_prompt_tokens=6,
    )
    admission_event = _Event(
        evict_start=1,
        tokens_evicted=1,
        new_user_fragment_len=1,
        position_offset_after=1,
        num_prompt_tokens=6,
    )
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2, 3], completion_ids=[4]),
        _CallSpec(
            submitted_prompt_ids=[1, 2, 3, 4, 5, 6],
            completion_ids=[7],
            compaction_events=[synthetic_prefix_event, admission_event],
        ),
    ]

    plans, _ = _build_pre_trim_plan(calls)
    plan = plans[1]
    timeline = _build_flex_mask_writer_timeline(calls)

    assert plan["prefix_cache_keep_len"] == (
        plan["synthetic_num_prompt"] - plan["synthetic_nuf_len"]
    )
    assert plan["prefix_cache_keep_len"] == 2
    assert plan["prefix_replay_tokens"] == [3, 4]
    assert plan["splices"] == [(1, 1)]
    assert plan["writer_offset"] == 0
    assert plan["admission_offset_after"] == 1
    assert plan["protected_prefix_len"] == 1
    assert timeline.input_ids == [1, 2, 3, 4, 3, 4, 5, 6, 7]
    assert timeline.position_ids == [0, 1, 2, 3, 2, 3, 4, 5, 6]
    assert timeline.death_indices == [9, 7, 4, 4, 9, 9, 9, 9, 9]
    assert timeline.loss_ranges == [(0, 4, 0, 4), (6, 9, 4, 7)]


# ─── No-admission per-call dispatch ───────────────────────────────────


def test_per_call_owned_ranges_partition_merged_frame():
    """Owned ranges across all calls must exactly partition [0, full_seq_len)
    when summed up. No overlaps, no gaps."""
    calls = [
        _CallSpec(submitted_prompt_ids=[10, 11], completion_ids=[20, 21]),  # len 4
        _CallSpec(submitted_prompt_ids=[10, 11, 20, 21, 30, 31], completion_ids=[40]),  # len 7
        _CallSpec(
            submitted_prompt_ids=[10, 11, 20, 21, 30, 31, 40, 50, 51],
            completion_ids=[60, 61],
        ),  # len 11
    ]
    input_ids, pos_ids, full_len = _merged_from_calls(calls)
    model = _RecordingModel()

    owned_ranges: list[tuple[int, int]] = []

    def fake_loss(seg_logits, full_start, full_end):
        owned_ranges.append((full_start, full_end))
        return seg_logits.float().sum() * 0.0

    per_call_segmented_forward(
        model,
        calls,
        merged_input_ids=input_ids,
        merged_position_ids=pos_ids,
        loss_fn=fake_loss,
        max_forward_passes=3,
        device=torch.device("cpu"),
    )

    assert owned_ranges[0][0] == 0
    for i in range(len(owned_ranges) - 1):
        assert owned_ranges[i][1] == owned_ranges[i + 1][0], (
            f"gap or overlap at boundary {i}: "
            f"prev_end={owned_ranges[i][1]} next_start={owned_ranges[i + 1][0]}"
        )
    assert owned_ranges[-1][1] == full_len


def test_per_call_default_mirrors_all_decode_tails():
    """Default replay mirrors vLLM decode shape for every completion tail."""
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3, 4]),
        _CallSpec(submitted_prompt_ids=[1, 2, 3, 4, 5], completion_ids=[6, 7]),
    ]
    input_ids, pos_ids, _ = _merged_from_calls(calls)
    model = _RecordingModel()

    def fake_loss(seg_logits, start, end):
        return seg_logits.float().sum() * 0.0

    assert compute_num_per_call_forwards(calls) == 6
    per_call_segmented_forward(
        model, calls, input_ids, pos_ids,
        loss_fn=fake_loss, max_forward_passes=compute_num_per_call_forwards(calls),
        device=torch.device("cpu"),
    )

    assert len(model.calls) == 6
    # Each call keeps prompt/new-user prefill grouped, then mirrors vLLM decode.
    assert model.calls[0]["input_ids"] == [1, 2]
    assert model.calls[1]["input_ids"] == [3]
    assert model.calls[2]["input_ids"] == [4]
    assert model.calls[3]["input_ids"] == [5]
    assert model.calls[4]["input_ids"] == [6]
    assert model.calls[5]["input_ids"] == [7]
    assert model.calls[5]["pre_cache_len"] == 6


def test_per_call_splits_prompt_tail_with_persistent_cache(monkeypatch):
    """Each call forwards only new tokens, split into prompt and tail chunks."""
    monkeypatch.setenv("KVE_TRAINER_MIRROR_VLLM_DECODE", "0")
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3, 4]),  # contributes [0, 4)
        _CallSpec(submitted_prompt_ids=[1, 2, 3, 4, 5], completion_ids=[6, 7]),  # contributes [4, 7)
    ]
    input_ids, pos_ids, _ = _merged_from_calls(calls)
    model = _RecordingModel()

    def fake_loss(seg_logits, start, end):
        return seg_logits.float().sum() * 0.0

    per_call_segmented_forward(
        model, calls, input_ids, pos_ids,
        loss_fn=fake_loss, max_forward_passes=compute_num_per_call_forwards(calls),
        device=torch.device("cpu"),
    )

    assert len(model.calls) == 4
    # Call 0 prompt prefill: [1, 2], cache starts empty.
    assert model.calls[0]["seq_len"] == 2
    assert model.calls[0]["position_ids"] == [0, 1]
    assert model.calls[0]["pre_cache_len"] == 0
    # Call 0 completion tail: [3, 4].
    assert model.calls[1]["seq_len"] == 2
    assert model.calls[1]["position_ids"] == [2, 3]
    assert model.calls[1]["pre_cache_len"] == 2
    # Call 1 prompt prefill: [5], with call 0 K/V retained.
    assert model.calls[2]["seq_len"] == 1
    assert model.calls[2]["position_ids"] == [4]
    assert model.calls[2]["pre_cache_len"] == 4
    # Call 1 completion tail: [6, 7].
    assert model.calls[3]["seq_len"] == 2
    assert model.calls[3]["position_ids"] == [5, 6]
    assert model.calls[3]["pre_cache_len"] == 5


def test_per_call_pads_with_dummy_passes_for_fsdp_sync():
    """When max_forward_passes > local forward count, dummy passes fill the gap."""
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3]),
    ]
    input_ids, pos_ids, _ = _merged_from_calls(calls)
    model = _RecordingModel()

    def fake_loss(seg_logits, start, end):
        return seg_logits.float().sum() * 0.0

    per_call_segmented_forward(
        model, calls, input_ids, pos_ids,
        loss_fn=fake_loss, max_forward_passes=4,
        device=torch.device("cpu"),
    )

    assert len(model.calls) == 4
    # Real forwards: prompt [1, 2], completion [3]. Dummies: 1-token each.
    assert model.calls[0]["seq_len"] == 2
    assert model.calls[1]["seq_len"] == 1
    for dummy in model.calls[2:]:
        assert dummy["seq_len"] == 1


def test_per_call_bptt_segments_one_keeps_backward_per_call(monkeypatch):
    monkeypatch.setenv("KVE_TRAINER_MIRROR_VLLM_DECODE", "0")
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3]),
        _CallSpec(submitted_prompt_ids=[1, 2, 3, 4], completion_ids=[5]),
    ]
    input_ids, pos_ids, _ = _merged_from_calls(calls)
    model = _RecordingModel()
    backward_calls = []
    orig_backward = torch.Tensor.backward

    def counted_backward(self, *args, **kwargs):
        backward_calls.append(self)
        return orig_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", counted_backward)

    def fake_loss(seg_logits, start, end):
        return seg_logits.float().sum()

    per_call_segmented_forward(
        model, calls, input_ids, pos_ids,
        loss_fn=fake_loss, max_forward_passes=compute_num_per_call_forwards(calls),
        bptt_segments=1,
        device=torch.device("cpu"),
    )

    assert len(backward_calls) == 2


def test_per_call_bptt_segments_two_backwards_once_and_pads_window(monkeypatch):
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3]),
        _CallSpec(submitted_prompt_ids=[1, 2, 3, 4], completion_ids=[5]),
    ]
    input_ids, pos_ids, _ = _merged_from_calls(calls)
    model = _RecordingModel()
    backward_calls = []
    orig_backward = torch.Tensor.backward

    def counted_backward(self, *args, **kwargs):
        backward_calls.append(self)
        return orig_backward(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "backward", counted_backward)

    def fake_loss(seg_logits, start, end):
        return seg_logits.float().sum()

    per_call_segmented_forward(
        model, calls, input_ids, pos_ids,
        loss_fn=fake_loss,
        max_forward_passes=compute_num_per_call_forwards(calls),
        max_bptt_window_forward_passes=[5],
        bptt_segments=2,
        device=torch.device("cpu"),
    )

    assert len(model.calls) == 5
    assert len(backward_calls) == 1


def test_per_call_bptt_segments_two_backprops_through_prior_call_cache():
    calls = [
        _CallSpec(submitted_prompt_ids=[1, 2], completion_ids=[3]),
        _CallSpec(submitted_prompt_ids=[1, 2, 3, 4], completion_ids=[5]),
    ]
    input_ids, pos_ids, _ = _merged_from_calls(calls)

    def loss_from_second_call_first_logit(seg_logits, start, end):
        if start == 0:
            return seg_logits.float().sum() * 0.0
        # The first logit of call 1 is computed before call 1 writes its
        # own K rows. It can only depend on call 0's carried cache.
        return seg_logits[:, :1, :].float().sum()

    bptt_model = _CrossCallGradModel()
    per_call_segmented_forward(
        bptt_model, calls, input_ids, pos_ids,
        loss_fn=loss_from_second_call_first_logit,
        max_forward_passes=compute_num_per_call_forwards(calls),
        bptt_segments=2,
        device=torch.device("cpu"),
    )
    assert bptt_model.cache_scale.grad is not None
    assert bptt_model.cache_scale.grad.abs().item() > 0

    full_chain_model = _CrossCallGradModel()
    per_call_segmented_forward(
        full_chain_model, calls, input_ids, pos_ids,
        loss_fn=loss_from_second_call_first_logit,
        max_forward_passes=compute_num_per_call_forwards(calls),
        bptt_segments=-1,
        device=torch.device("cpu"),
    )
    assert full_chain_model.cache_scale.grad is not None
    assert full_chain_model.cache_scale.grad.abs().item() > 0

    detached_model = _CrossCallGradModel()
    per_call_segmented_forward(
        detached_model, calls, input_ids, pos_ids,
        loss_fn=loss_from_second_call_first_logit,
        max_forward_passes=compute_num_per_call_forwards(calls),
        bptt_segments=1,
        device=torch.device("cpu"),
    )
    detached_grad = detached_model.cache_scale.grad
    assert detached_grad is None or detached_grad.abs().item() == 0


# ─── Admission per-call dispatch ──────────────────────────────────────


def test_per_call_admission_warmup_splice_then_decode_tail(monkeypatch):
    """Call-0 admission prefills the full prompt, splices, then decodes tail."""
    monkeypatch.setenv("KVE_TRAINER_MIRROR_VLLM_DECODE", "0")
    admission_event = _Event(
        evict_start=2,
        tokens_evicted=2,
        new_user_fragment_len=2,
        position_offset_after=2,
        num_prompt_tokens=6,
    )
    # Pre-trim submitted = [10, 11, 99, 99, 30, 31] (6 tokens).
    # phase1_token_count = 6 - 2 = 4. evict_end = 4.
    # L1_kept = 4 - 2 = 2; P2 = nuf_len + comp_len = 2 + 2 = 4.
    # Merged contribution is pre-trim: full submitted prompt + completion.
    calls = [
        _CallSpec(
            submitted_prompt_ids=[10, 11, 99, 99, 30, 31],
            completion_ids=[40, 41],
            compaction_events=[admission_event],
        ),
    ]
    merged = [10, 11, 99, 99, 30, 31, 40, 41]
    input_ids = torch.tensor([merged], dtype=torch.long)
    pos_ids = torch.arange(len(merged), dtype=torch.long).unsqueeze(0)
    model = _RecordingModel()

    captured_ranges: list[tuple[int, int]] = []

    def fake_loss(seg_logits, start, end):
        captured_ranges.append((start, end))
        return seg_logits.float().sum() * 0.0

    per_call_segmented_forward(
        model, calls, input_ids, pos_ids,
        loss_fn=fake_loss, max_forward_passes=compute_num_per_call_forwards(calls),
        device=torch.device("cpu"),
    )

    assert len(model.calls) == 2
    # Warmup covers all submitted prompt tokens before admission splices cache.
    assert model.calls[0]["seq_len"] == 6
    assert model.calls[0]["position_ids"] == [0, 1, 2, 3, 4, 5]
    # Tail writes completion under the post-admission offset.
    assert model.calls[1]["seq_len"] == 2
    assert model.calls[1]["position_ids"] == [6, 7]
    assert model.calls[1]["pre_cache_len"] == 4
    assert captured_ranges == [(0, 8)]


def test_per_call_admission_with_kept_middle_inserts_eviction_gap(monkeypatch):
    """Regression: when phase1 has tokens AFTER the evicted range
    (kept_middle), the trainer's position_ids must skip the evicted
    range in the middle of the kept prompt. Previously the code
    treated all L1_kept tokens as contiguous [0, L1_kept), which
    matched vLLM only when evict_end == phase1_token_count.

    Concrete: dummy-chat-env step 2 has sub_len=116, nuf_len=9,
    evict_start=16, tokens_evicted=48 → phase1_token_count=107,
    evict_end=64, L1_kept=59 (= 16 protected + 43 kept middle).
    Correct logical positions: [0..15] ++ [64..106] ++ [107..123]."""
    monkeypatch.setenv("KVE_TRAINER_MIRROR_VLLM_DECODE", "0")
    admission_event = _Event(
        evict_start=16,
        tokens_evicted=48,
        new_user_fragment_len=9,
        position_offset_after=48,
        num_prompt_tokens=116,
    )
    sub_len = 116
    comp_len = 8
    calls = [
        _CallSpec(
            submitted_prompt_ids=list(range(sub_len)),
            completion_ids=list(range(1000, 1000 + comp_len)),
            compaction_events=[admission_event],
        ),
    ]
    # The merged training frame is pre-trim: full submitted prompt + completion.
    merged_len = sub_len + comp_len
    input_ids = torch.zeros((1, merged_len), dtype=torch.long)
    pos_ids = torch.arange(merged_len, dtype=torch.long).unsqueeze(0)
    model = _RecordingModel()

    def fake_loss(seg_logits, start, end):
        return seg_logits.float().sum() * 0.0

    per_call_segmented_forward(
        model, calls, input_ids, pos_ids,
        loss_fn=fake_loss, max_forward_passes=compute_num_per_call_forwards(calls),
        device=torch.device("cpu"),
    )

    assert len(model.calls) == 2
    assert model.calls[0]["position_ids"] == list(range(0, 116))
    assert model.calls[1]["position_ids"] == list(range(116, 124))
    assert model.calls[1]["pre_cache_len"] == 68


def test_per_call_plan_combines_multiple_admission_events():
    """Multiple admission iterations are preserved as composed splice events."""
    from kv_eviction.segmented_forward import _build_pre_trim_plan

    event_a = _Event(
        evict_start=4,
        tokens_evicted=16,
        new_user_fragment_len=8,
        position_offset_after=16,
        num_prompt_tokens=100,
    )
    event_b = _Event(
        evict_start=4,
        tokens_evicted=16,
        new_user_fragment_len=8,
        position_offset_after=32,
        num_prompt_tokens=84,
    )
    call = _CallSpec(
        submitted_prompt_ids=[0] * 100,
        completion_ids=[1] * 10,
        compaction_events=[event_a, event_b],
    )
    plans, _ = _build_pre_trim_plan([call])
    plan = plans[0]
    assert plan["splices"] == [(4, 16), (4, 16)]
    assert plan["admission_total_evicted"] == 32
    assert plan["nuf_len"] == 8


def test_per_call_negative_evict_start_rejected():
    """Negative evict_start is rejected while evict_start=0 is representable."""
    from kv_eviction.segmented_forward import _build_pre_trim_plan

    event = _Event(
        evict_start=-1,
        tokens_evicted=4,
        new_user_fragment_len=2,
        position_offset_after=4,
        num_prompt_tokens=10,
    )
    call = _CallSpec(
        submitted_prompt_ids=[0] * 10,
        completion_ids=[1] * 2,
        compaction_events=[event],
    )
    with pytest.raises(ValueError, match="bad event"):
        _build_pre_trim_plan([call])


def test_per_call_cache_splice_helper_drops_evicted_range():
    """_splice_dynamic_cache removes [evict_start, evict_end) from each
    layer's K/V. Kept for use by debug/diagnostic tooling and as the
    inverse-operation building block; the production per-call path
    handles admission via position_ids rather than splicing."""
    from kv_eviction.segmented_forward import _splice_dynamic_cache
    from transformers import DynamicCache

    cache = DynamicCache()
    for layer in range(2):
        k = torch.arange(10, dtype=torch.float32).view(1, 1, 10, 1)
        v = torch.arange(10, dtype=torch.float32).view(1, 1, 10, 1) + 100
        cache.update(k, v, layer)

    spliced = _splice_dynamic_cache(cache, evict_start=3, evict_end=7)
    layer0 = spliced.layers[0] if hasattr(spliced, "layers") else None
    if layer0 is not None:
        k = layer0.keys.squeeze().tolist()
        v = layer0.values.squeeze().tolist()
    else:
        k = spliced.key_cache[0].squeeze().tolist()
        v = spliced.value_cache[0].squeeze().tolist()
    assert k == [0, 1, 2, 7, 8, 9]
    assert v == [100, 101, 102, 107, 108, 109]


def test_per_call_admission_then_extension_chains_position_offset(monkeypatch):
    """A sample with admission at C0 followed by an extension call C1
    must continue C1's positions in the pre-eviction frame (offset by
    total_evicted from C0)."""
    monkeypatch.setenv("KVE_TRAINER_MIRROR_VLLM_DECODE", "0")
    admission_event = _Event(
        evict_start=2,
        tokens_evicted=2,
        new_user_fragment_len=2,
        position_offset_after=2,
        num_prompt_tokens=6,
    )
    # C0 (admission): pre-trim sub=[10,11,99,99,30,31], comp=[40,41].
    #   Training frame keeps all 8 pre-trim contribution tokens.
    # C1 (extension after C0's trim, no new admission):
    #   sub (post-C0-trim) = [10,11,30,31,40,41,50] (one new user token).
    #   comp = [60].
    #   call_input_len = 7 + 1 = 8 → covers merged [0, 8).
    #   New contribution: merged [6, 8) = [50, 60].
    calls = [
        _CallSpec(
            submitted_prompt_ids=[10, 11, 99, 99, 30, 31],
            completion_ids=[40, 41],
            compaction_events=[admission_event],
        ),
        _CallSpec(
            submitted_prompt_ids=[10, 11, 30, 31, 40, 41, 50],
            completion_ids=[60],
        ),
    ]
    # Merged frame is pre-trim cumulative.
    merged = [10, 11, 99, 99, 30, 31, 40, 41, 50, 60]
    input_ids = torch.tensor([merged], dtype=torch.long)
    pos_ids = torch.arange(len(merged), dtype=torch.long).unsqueeze(0)
    model = _RecordingModel()

    captured_ranges: list[tuple[int, int]] = []

    def fake_loss(seg_logits, start, end):
        captured_ranges.append((start, end))
        return seg_logits.float().sum() * 0.0

    per_call_segmented_forward(
        model, calls, input_ids, pos_ids,
        loss_fn=fake_loss, max_forward_passes=compute_num_per_call_forwards(calls),
        device=torch.device("cpu"),
    )

    assert len(model.calls) == 4
    # C0 prompt warmup then completion tail.
    assert model.calls[0]["position_ids"] == [0, 1, 2, 3, 4, 5]
    assert model.calls[0]["pre_cache_len"] == 0
    assert model.calls[1]["position_ids"] == [6, 7]
    assert model.calls[1]["pre_cache_len"] == 4
    # C1: continues at position 8 with offset preserved.
    assert model.calls[2]["seq_len"] == 1
    assert model.calls[2]["position_ids"] == [8]
    assert model.calls[2]["pre_cache_len"] == 6
    assert model.calls[3]["seq_len"] == 1
    assert model.calls[3]["position_ids"] == [9]
    assert model.calls[3]["pre_cache_len"] == 7
    assert captured_ranges == [(0, 8), (8, 10)]


def test_per_call_rejects_mixed_admission_midgen():
    """Per-call dispatch must not receive samples with mid-generation events."""
    from kv_eviction.segmented_forward import _build_pre_trim_plan

    admission_event = _Event(
        evict_start=4,
        tokens_evicted=16,
        new_user_fragment_len=8,
        position_offset_after=16,
        num_prompt_tokens=100,
    )
    midgen_event = _Event(
        num_output_tokens_at_compaction=32,
        evict_start=4,
        tokens_evicted=8,
    )
    call = _CallSpec(
        submitted_prompt_ids=[0] * 100,
        completion_ids=[1] * 50,
        compaction_events=[admission_event, midgen_event],
    )
    with pytest.raises(AssertionError, match="mid-generation event"):
        _build_pre_trim_plan([call])


def test_per_call_owned_logits_match_position_ids(monkeypatch):
    """For a no-admission rollout, owned logits for each call should
    equal their merged-frame position ids (mock model returns
    logits = position_ids broadcast)."""
    monkeypatch.setenv("KVE_TRAINER_MIRROR_VLLM_DECODE", "0")
    calls = [
        _CallSpec(submitted_prompt_ids=[5, 6], completion_ids=[7]),
        _CallSpec(submitted_prompt_ids=[5, 6, 7, 8], completion_ids=[9]),
    ]
    input_ids, pos_ids, _ = _merged_from_calls(calls)
    model = _RecordingModel()

    captured: list[torch.Tensor] = []

    def fake_loss(seg_logits, start, end):
        captured.append(seg_logits.detach().clone())
        return seg_logits.float().sum() * 0.0

    per_call_segmented_forward(
        model, calls, input_ids, pos_ids,
        loss_fn=fake_loss, max_forward_passes=2,
        device=torch.device("cpu"),
    )

    # Call 0: new tokens [0, 3) → 3 positions [0, 1, 2].
    # Call 1: new tokens [3, 5) → 2 positions [3, 4].
    assert captured[0].shape == (1, 3, 32)
    for i in range(3):
        assert torch.all(captured[0][0, i] == float(i)).item()
    assert captured[1].shape == (1, 2, 32)
    for i in range(2):
        assert torch.all(captured[1][0, i] == float(3 + i)).item()
