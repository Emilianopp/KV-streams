# SPDX-License-Identifier: Apache-2.0
"""Unit tests for managed-context interval visibility in the flex timeline.

Hidden recall makes per-token visibility non-monotone (alive -> dead ->
alive). These tests verify that _build_flex_mask_writer_timeline maps
kind-1/2 restore events (attach/release) plus the archiving eviction's
archived_span_bounds into extra (row, birth, death) visibility windows,
and that _build_visibility_mask_mod realizes them in the mask.

Chain layout shared by the tests (writer rows in parentheses):
  call0: sub=[1,2,3,4]            comp=[5]   -> rows 0-4
  call1: sub=[1..5, 6,7]          comp=[8]   -> rows 5-7
  call2: sub=[1..8, 9,10]         comp=[11]  -> rows 8-10; admission evicts
         current positions [2,4) = writer rows 2,3 (tokens 3,4), archived
         as span T0001 with bounds [2,4); death index 8.
  call3: sub=[kept 9 tokens, 12,13] comp=[14] -> rows 11-13; restore ops
         under test attach/release T0001.

See plans/flex_full_bptt_alignment.md phase M1.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from kv_eviction.segmented_forward import (
    _build_flex_mask_writer_timeline,
    _build_visibility_mask_mod,
)


def _evict_event(**overrides):
    base = dict(
        num_output_tokens_at_compaction=0,
        evict_start=2,
        tokens_evicted=2,
        new_user_fragment_len=2,
        position_offset_after=2,
        num_prompt_tokens=10,
        archived_span_ids=["T0001"],
        archived_span_bounds=[2, 4],
        event_kind=0,
        restored_span_ids=[],
        visibility_boundary_computed=-1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _restore_event(kind: int, boundary: int, span_ids=("T0001",)):
    return SimpleNamespace(
        num_output_tokens_at_compaction=0,
        evict_start=0,
        tokens_evicted=0,
        new_user_fragment_len=0,
        position_offset_after=2,
        num_prompt_tokens=11,
        archived_span_ids=[],
        archived_span_bounds=[],
        event_kind=kind,
        restored_span_ids=list(span_ids),
        visibility_boundary_computed=boundary,
    )


def _chain(call3_events):
    return [
        SimpleNamespace(
            submitted_prompt_ids=[1, 2, 3, 4],
            completion_ids=[5],
            trailing_pad_ids=[],
            compaction_events=[],
        ),
        SimpleNamespace(
            submitted_prompt_ids=[1, 2, 3, 4, 5, 6, 7],
            completion_ids=[8],
            trailing_pad_ids=[],
            compaction_events=[],
        ),
        SimpleNamespace(
            submitted_prompt_ids=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            completion_ids=[11],
            trailing_pad_ids=[],
            compaction_events=[_evict_event()],
        ),
        SimpleNamespace(
            # Post-eviction kept state [1,2,5,6,7,8,9,10,11] + new [12,13].
            submitted_prompt_ids=[1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13],
            completion_ids=[14],
            trailing_pad_ids=[],
            compaction_events=list(call3_events),
        ),
    ]


def test_eviction_only_timeline_has_no_extra_intervals():
    timeline = _build_flex_mask_writer_timeline(_chain([]))
    assert timeline.extra_intervals == []
    assert len(timeline.input_ids) == 14
    # Rows 2,3 (tokens 3,4) died when call2's admission spliced them.
    assert timeline.death_indices[2] == 8
    assert timeline.death_indices[3] == 8
    assert all(
        d == 14 for i, d in enumerate(timeline.death_indices) if i not in (2, 3)
    )


def test_upfront_attach_clamps_birth_to_call_start():
    # Upfront attach: engine boundary 0 = "everything this request
    # computes". Rows before call3's first appended row (11) had their
    # queries run by earlier requests, so birth clamps to 11; no release
    # -> the window closes at call end (14).
    timeline = _build_flex_mask_writer_timeline(
        _chain([_restore_event(kind=1, boundary=0)])
    )
    assert sorted(timeline.extra_intervals) == [(2, 11, 14), (3, 11, 14)]
    # Base death indices unchanged by the restore.
    assert timeline.death_indices[2] == 8
    assert timeline.death_indices[3] == 8


def test_deferred_attach_and_release_boundaries():
    # Deferred attach at boundary 10 = the last prompt token of call3's
    # request frame -> writer row 12. Release at boundary 12 = the full
    # request frame (11 prompt + 1 completion) -> end-of-known-stream,
    # writer index 14.
    timeline = _build_flex_mask_writer_timeline(
        _chain(
            [
                _restore_event(kind=1, boundary=10),
                _restore_event(kind=2, boundary=12),
            ]
        )
    )
    assert sorted(timeline.extra_intervals) == [(2, 12, 14), (3, 12, 14)]


def test_attach_for_unknown_span_fails_loudly():
    with pytest.raises(ValueError, match="no archived rows"):
        _build_flex_mask_writer_timeline(
            _chain([_restore_event(kind=1, boundary=0, span_ids=("T9999",))])
        )


def test_release_without_attach_fails_loudly():
    with pytest.raises(ValueError, match="without a preceding attach"):
        _build_flex_mask_writer_timeline(
            _chain([_restore_event(kind=2, boundary=12)])
        )


def _dense_mask(timeline):
    n = len(timeline.input_ids)
    death = torch.tensor(timeline.death_indices, dtype=torch.long)
    mask_mod = _build_visibility_mask_mod(
        death, timeline.extra_intervals, n, torch.device("cpu")
    )
    q = torch.arange(n).view(-1, 1).expand(n, n)
    kv = torch.arange(n).view(1, -1).expand(n, n)
    return mask_mod(0, 0, q, kv)


def test_mask_mod_reopens_restored_rows():
    timeline = _build_flex_mask_writer_timeline(
        _chain([_restore_event(kind=1, boundary=0)])
    )
    mask = _dense_mask(timeline)
    # Dead region: rows 2,3 invisible to queries in [8, 11).
    assert not mask[8, 2] and not mask[10, 3]
    # Restored window: visible again to call3's queries [11, 14).
    assert mask[11, 2] and mask[13, 3]
    # Never-evicted row behaves classically.
    assert mask[13, 4] and not mask[3, 4]
    # Causality holds everywhere.
    assert not mask[2, 3]


def test_mask_mod_without_intervals_matches_death_only():
    timeline = _build_flex_mask_writer_timeline(_chain([]))
    mask = _dense_mask(timeline)
    n = len(timeline.input_ids)
    death = torch.tensor(timeline.death_indices, dtype=torch.long)
    q = torch.arange(n).view(-1, 1).expand(n, n)
    kv = torch.arange(n).view(1, -1).expand(n, n)
    expected = (kv <= q) & (q < death[kv])
    assert torch.equal(mask, expected)


def test_dropped_fork_kills_dead_branch_rows():
    # call1 contributes [6,7,8]; call2's prompt DROPS [6,7] (an abandoned
    # exchange) and continues from the surviving branch — the engine
    # would prefix-hit [1..5] and prefill [8,9,10]. The trainer must kill
    # rows of 6,7 at call2's start and keep frames aligned.
    calls = [
        SimpleNamespace(
            submitted_prompt_ids=[1, 2, 3, 4],
            completion_ids=[5],
            trailing_pad_ids=[],
            compaction_events=[],
        ),
        SimpleNamespace(
            submitted_prompt_ids=[1, 2, 3, 4, 5, 6, 7],
            completion_ids=[8],
            trailing_pad_ids=[],
            compaction_events=[],
        ),
        SimpleNamespace(
            # expected stream is [1..8]; this prompt drops [6,7].
            submitted_prompt_ids=[1, 2, 3, 4, 5, 8, 9, 10],
            completion_ids=[11],
            trailing_pad_ids=[],
            compaction_events=[],
        ),
    ]
    timeline = _build_flex_mask_writer_timeline(calls)
    # writer rows: [1,2,3,4,5] = 0-4, [6,7,8] = 5-7, then call2 appends
    # [9,10,11] = 8-10 (token 8 at row 7 survives via the branch).
    assert timeline.input_ids == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    # rows of 6,7 (writer 5,6) die at call2's start (= 8).
    assert timeline.death_indices[5] == 8
    assert timeline.death_indices[6] == 8
    assert all(
        d == 11 for i, d in enumerate(timeline.death_indices) if i not in (5, 6)
    )


def test_eviction_only_chain_has_no_dead_branches():
    from kv_eviction.segmented_forward import _build_pre_trim_plan

    plans, _ = _build_pre_trim_plan(_chain([]))
    assert all(not p.get("dead_branches") for p in plans)
