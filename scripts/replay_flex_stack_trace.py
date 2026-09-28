#!/usr/bin/env python3
"""Replay real production MicroBatch bins through the flex stack functions.

This reads trainer-ready ``rank_N.bin`` files written by the production packer
and runs the exact flex compaction forward functions on the real calls/events
and token tensors. It is intended to isolate trainer-side stacking performance
from vLLM/TextWorld rollout generation.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import threading
import time
from pathlib import Path

import msgspec
import psutil
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from transformers import AutoConfig, AutoModelForCausalLM

from kv_eviction.segmented_forward import (
    _build_flex_mask_writer_timeline,
    _flex_kernel_options_from_env,
    _set_attn_implementation_if_needed,
    batched_flex_mask_segmented_forward,
    compute_flex_mask_writer_len,
    flex_mask_segmented_forward,
    packed_flex_mask_segmented_forward,
    selected_logprob_batched_flex_mask_segmented_forward,
    selected_logprob_flex_mask_segmented_forward,
    selected_logprob_packed_flex_mask_segmented_forward,
)
from prime_rl.configs.trainer import ActivationOffloadingConfig, DefaultLossConfig
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.rl.loss import (
    compute_loss,
    default_loss_fn,
    selective_log_softmax,
    shift_tensor_left,
)
from prime_rl.trainer.rl.microbatch_stacking import (
    is_flex_compaction_stackable_micro_batch,
    make_micro_batch_groups,
    pack_horizontal_flex_compaction_micro_batches,
    stack_flex_compaction_micro_batches,
)
from prime_rl.transport.types import MicroBatch
from prime_rl.utils.act_offloading import maybe_activation_offloading


def rss_gib() -> float:
    return psutil.Process().memory_info().rss / (1024**3)


class RssSampler:
    def __init__(self, *, enabled: bool, interval_s: float) -> None:
        self.enabled = enabled
        self.interval_s = max(0.01, float(interval_s))
        self.peak_gib = rss_gib()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "RssSampler":
        if not self.enabled:
            return self

        def _sample() -> None:
            while not self._stop.is_set():
                self.peak_gib = max(self.peak_gib, rss_gib())
                self._stop.wait(self.interval_s)
            self.peak_gib = max(self.peak_gib, rss_gib())

        self._thread = threading.Thread(target=_sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)


def load_micro_batches(path: Path) -> list[MicroBatch]:
    return msgspec.msgpack.Decoder(type=list[MicroBatch]).decode(path.read_bytes())


def micro_batch_to_tensor(mb: MicroBatch) -> dict:
    return {
        "input_ids": torch.tensor(mb.input_ids, dtype=torch.long).unsqueeze(0),
        "position_ids": torch.tensor(mb.position_ids, dtype=torch.long).unsqueeze(0),
        "advantages": torch.tensor(mb.advantages, dtype=torch.float32).unsqueeze(0),
        "inference_logprobs": torch.tensor(mb.inference_logprobs, dtype=torch.float32).unsqueeze(0),
        "teacher_logprobs": (
            torch.tensor(mb.teacher_logprobs, dtype=torch.float32).unsqueeze(0)
            if mb.teacher_logprobs is not None
            else None
        ),
        "loss_mask": torch.tensor(mb.loss_mask, dtype=torch.bool).unsqueeze(0),
        "temperatures": torch.tensor(mb.temperatures, dtype=torch.float32).unsqueeze(0),
        "lora_num_tokens": torch.tensor(mb.lora_num_tokens or [len(mb.input_ids)], dtype=torch.int32),
        "routed_experts": None,
        "pixel_values": None,
        "image_grid_thw": None,
        "compaction_events": mb.compaction_events,
        "prompt_len": mb.prompt_len,
        "calls": mb.calls,
    }


def move_tensors(mb: dict, device: torch.device) -> dict:
    out = dict(mb)
    for key, value in list(out.items()):
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
    return out


def apply_prime_block_checkpointing(model: torch.nn.Module, *, freq: int = 1) -> None:
    language_model = getattr(model, "model", model)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise ValueError("expected model.model.layers for block checkpointing")
    freq = max(1, int(freq))
    for layer_idx, (layer_name, layer) in enumerate(list(layers.named_children())):
        if layer_idx % freq != 0:
            continue
        layers.register_module(
            layer_name,
            checkpoint_wrapper(layer, preserve_rng_state=False),
        )


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(
        args.model,
        attn_implementation="flex_attention",
        trust_remote_code=args.trust_remote_code,
    )
    config.use_cache = False
    if args.debug_num_layers is not None:
        target = getattr(config, "text_config", config)
        target.num_hidden_layers = min(args.debug_num_layers, target.num_hidden_layers)
    dtype = getattr(torch, args.dtype)
    if args.random_init:
        with torch.device(device):
            model = AutoModelForCausalLM.from_config(
                config,
                trust_remote_code=args.trust_remote_code,
            )
        model.to(dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            config=config,
            trust_remote_code=args.trust_remote_code,
            dtype=dtype,
        ).to(device)
    if args.selected_logprob_head:
        inject_prime_lm_head(model, chunk_size=args.selected_logprob_chunk_size)
    model.train()
    if args.activation_checkpointing:
        apply_prime_block_checkpointing(model, freq=args.activation_checkpointing_freq)
    return model


def row_slice(
    tensor: torch.Tensor,
    *,
    horizontal: bool,
    sequence_offsets: list[int],
    batch_idx: int,
    start: int,
    end: int,
) -> torch.Tensor:
    if horizontal:
        offset = int(sequence_offsets[batch_idx])
        return tensor[:, offset + start : offset + end]
    return tensor[batch_idx : batch_idx + 1, start:end]


def make_segment_loss_fn(mb: dict, *, horizontal: bool, loss_scale: int):
    labels = shift_tensor_left(mb["input_ids"])
    advantages = mb["advantages"]
    inference_logprobs = mb["inference_logprobs"]
    teacher_logprobs = mb["teacher_logprobs"]
    loss_mask = mb["loss_mask"]
    full_seq_len = int(mb["input_ids"].shape[1])
    full_seq_lens = mb.get("sequence_lengths")
    if full_seq_lens is None:
        full_seq_lens = [full_seq_len] * int(mb["input_ids"].shape[0])
    sequence_offsets = mb.get("sequence_offsets") or [0] * len(full_seq_lens)
    loss_cfg = DefaultLossConfig(kl_tau=0.0)

    def _loss_fn(
        seg_logits: torch.Tensor,
        full_logit_start: int,
        full_logit_end: int,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        row_full_seq_len = int(full_seq_lens[batch_idx])
        effective_logit_end = min(int(full_logit_end), row_full_seq_len - 1)
        if effective_logit_end <= full_logit_start:
            return seg_logits.sum() * 0.0
        seg_logits_effective = seg_logits[:, : effective_logit_end - full_logit_start, :]
        seg_labels = row_slice(
            labels,
            horizontal=horizontal,
            sequence_offsets=sequence_offsets,
            batch_idx=batch_idx,
            start=full_logit_start,
            end=effective_logit_end,
        )
        seg_raw_logprobs = selective_log_softmax(seg_logits_effective, seg_labels)
        tgt_start = full_logit_start + 1
        tgt_end = effective_logit_end + 1
        seg_adv = row_slice(
            advantages,
            horizontal=horizontal,
            sequence_offsets=sequence_offsets,
            batch_idx=batch_idx,
            start=tgt_start,
            end=tgt_end,
        )
        seg_mask = row_slice(
            loss_mask,
            horizontal=horizontal,
            sequence_offsets=sequence_offsets,
            batch_idx=batch_idx,
            start=tgt_start,
            end=tgt_end,
        )
        seg_inf = row_slice(
            inference_logprobs,
            horizontal=horizontal,
            sequence_offsets=sequence_offsets,
            batch_idx=batch_idx,
            start=tgt_start,
            end=tgt_end,
        )
        seg_teach = (
            row_slice(
                teacher_logprobs,
                horizontal=horizontal,
                sequence_offsets=sequence_offsets,
                batch_idx=batch_idx,
                start=tgt_start,
                end=tgt_end,
            )
            if teacher_logprobs is not None
            else None
        )
        loss, _ = compute_loss(
            trainer_logprobs=(seg_raw_logprobs.squeeze(0),),
            inference_logprobs=(seg_inf.squeeze(0),),
            teacher_logprobs=(seg_teach.squeeze(0),) if seg_teach is not None else None,
            advantages=(seg_adv.squeeze(0),),
            loss_mask=(seg_mask.squeeze(0),),
            loss_fn=lambda inputs: default_loss_fn(inputs, loss_cfg),
            loss_scale=loss_scale,
        )
        # Match the production path's entropy side computation enough to keep
        # the softmax/logsumexp cost represented without retaining it in graph.
        with torch.no_grad():
            pd = torch.nn.functional.softmax(seg_logits_effective, dim=-1)
            _entropy = torch.logsumexp(seg_logits_effective, dim=-1) - torch.sum(
                pd * seg_logits_effective,
                dim=-1,
            )
            if seg_mask.any():
                _ = _entropy.squeeze(0)[seg_mask.squeeze(0).bool()].detach().to("cpu")
        return loss

    return _loss_fn


def group_micro_batches(
    args: argparse.Namespace,
    micro_batches: list[dict],
    *,
    seq_lens: list[int] | None = None,
    flex_compaction_stackable: list[bool] | None = None,
) -> list[list[dict]]:
    if args.mode == "single":
        return [[mb] for mb in micro_batches]
    if args.stack_token_budget is not None and seq_lens is None:
        seq_lens = []
        for mb in micro_batches:
            if mb.get("calls"):
                seq_lens.append(compute_flex_mask_writer_len(mb["calls"]))
            else:
                seq_lens.append(int(mb["input_ids"].shape[1]))
    return make_micro_batch_groups(
        micro_batches,
        stack_size=args.stack_size,
        stack_token_budget=args.stack_token_budget,
        flex_compaction_stack_mode=args.mode,
        seq_lens=seq_lens,
        flex_compaction_enabled=True,
        flex_compaction_stackable=flex_compaction_stackable,
        compaction_enabled=True,
        cp_enabled=False,
        lora_enabled=False,
        multi_run_enabled=False,
    )


def calls_batch_from_maybe_stacked(calls_obj) -> list[list]:
    if isinstance(calls_obj, list) and calls_obj and isinstance(calls_obj[0], list):
        return calls_obj
    return [calls_obj]


def _model_backbone(model: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(model, "model", None)
    if backbone is None:
        raise ValueError("selected logprob replay expects model.model backbone")
    return backbone


def _last_hidden_state(output) -> torch.Tensor:
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, dict) and "last_hidden_state" in output:
        return output["last_hidden_state"]
    return output[0]


def _build_writer_inputs_for_group(
    calls_batch: list[list],
    *,
    mode: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, object, list, list[int], list[int]]:
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as exc:  # pragma: no cover - depends on torch build
        raise RuntimeError("selected logprob replay requires torch flex_attention") from exc

    timelines = [_build_flex_mask_writer_timeline(calls) for calls in calls_batch]
    if any(not timeline.input_ids for timeline in timelines):
        raise ValueError("selected logprob replay got empty writer timeline")
    seq_lens = [len(timeline.input_ids) for timeline in timelines]

    if mode == "vertical":
        batch_size = len(timelines)
        max_seq_len = max(seq_lens)
        writer_ids = torch.full((batch_size, max_seq_len), 1, device=device, dtype=torch.long)
        writer_positions = torch.zeros((batch_size, max_seq_len), device=device, dtype=torch.long)
        death_idx = torch.zeros((batch_size, max_seq_len), device=device, dtype=torch.long)
        for row, timeline in enumerate(timelines):
            row_len = len(timeline.input_ids)
            writer_ids[row, :row_len] = torch.tensor(timeline.input_ids, device=device, dtype=torch.long)
            writer_positions[row, :row_len] = torch.tensor(timeline.position_ids, device=device, dtype=torch.long)
            death_idx[row, :row_len] = torch.tensor(timeline.death_indices, device=device, dtype=torch.long)
            if row_len < max_seq_len:
                writer_positions[row, row_len:] = torch.arange(
                    max_seq_len - row_len,
                    device=device,
                    dtype=torch.long,
                )
        seq_lens_t = torch.tensor(seq_lens, device=device, dtype=torch.long)

        def _live_cache_mask(batch_idx, _head_idx, q_idx, kv_idx):
            row_len = seq_lens_t[batch_idx]
            valid_q = q_idx < row_len
            valid_kv = kv_idx < row_len
            live = (
                valid_q
                & valid_kv
                & (kv_idx <= q_idx)
                & (q_idx < death_idx[batch_idx, kv_idx])
            )
            pad_self = (~valid_q) & (q_idx == kv_idx)
            return live | pad_self

        block_mask = create_block_mask(
            _live_cache_mask,
            B=batch_size,
            H=None,
            Q_LEN=max_seq_len,
            KV_LEN=max_seq_len,
            device=device,
        )
        writer_offsets = [0] * batch_size
        return writer_ids, writer_positions, block_mask, timelines, seq_lens, writer_offsets

    if mode == "horizontal":
        writer_offsets = []
        total_seq_len = 0
        for row_len in seq_lens:
            writer_offsets.append(total_seq_len)
            total_seq_len += row_len
        writer_ids = torch.empty((1, total_seq_len), device=device, dtype=torch.long)
        writer_positions = torch.empty((1, total_seq_len), device=device, dtype=torch.long)
        sample_id = torch.empty((total_seq_len,), device=device, dtype=torch.long)
        death_idx = torch.empty((total_seq_len,), device=device, dtype=torch.long)
        for row, (offset, timeline) in enumerate(zip(writer_offsets, timelines, strict=True)):
            row_len = len(timeline.input_ids)
            row_slice = slice(offset, offset + row_len)
            writer_ids[0, row_slice] = torch.tensor(timeline.input_ids, device=device, dtype=torch.long)
            writer_positions[0, row_slice] = torch.tensor(timeline.position_ids, device=device, dtype=torch.long)
            sample_id[row_slice] = row
            death_idx[row_slice] = offset + torch.tensor(timeline.death_indices, device=device, dtype=torch.long)

        def _live_cache_mask(_batch_idx, _head_idx, q_idx, kv_idx):
            same_sample = sample_id[q_idx] == sample_id[kv_idx]
            return same_sample & (kv_idx <= q_idx) & (q_idx < death_idx[kv_idx])

        block_mask = create_block_mask(
            _live_cache_mask,
            B=1,
            H=None,
            Q_LEN=total_seq_len,
            KV_LEN=total_seq_len,
            device=device,
        )
        return writer_ids, writer_positions, block_mask, timelines, seq_lens, writer_offsets

    if len(timelines) != 1:
        raise ValueError("single selected logprob replay expects one timeline")
    timeline = timelines[0]
    seq_len = len(timeline.input_ids)
    writer_ids = torch.tensor(timeline.input_ids, device=device, dtype=torch.long).unsqueeze(0)
    writer_positions = torch.tensor(timeline.position_ids, device=device, dtype=torch.long).unsqueeze(0)
    death_idx = torch.tensor(timeline.death_indices, device=device, dtype=torch.long)

    def _live_cache_mask(_batch_idx, _head_idx, q_idx, kv_idx):
        return (kv_idx <= q_idx) & (q_idx < death_idx[kv_idx])

    block_mask = create_block_mask(
        _live_cache_mask,
        B=1,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )
    return writer_ids, writer_positions, block_mask, timelines, seq_lens, [0]


def run_group_selected_logprob(
    model: torch.nn.Module,
    group: list[dict],
    *,
    mode: str,
    device: torch.device,
    activation_offloading_config: ActivationOffloadingConfig | None = None,
) -> tuple[float, int, int]:
    if mode == "single":
        assert len(group) == 1
        mb = group[0]
        horizontal = False
    elif mode == "vertical":
        mb = stack_flex_compaction_micro_batches(group)
        horizontal = False
    elif mode == "horizontal":
        mb = pack_horizontal_flex_compaction_micro_batches(group)
        horizontal = True
    else:
        raise ValueError(f"unknown mode {mode!r}")

    mb = move_tensors(mb, device)
    calls_batch = calls_batch_from_maybe_stacked(mb["calls"])
    loss_scale = max(1, int(mb["loss_mask"].sum().item()))
    labels = shift_tensor_left(mb["input_ids"])
    advantages = mb["advantages"]
    inference_logprobs = mb["inference_logprobs"]
    teacher_logprobs = mb["teacher_logprobs"]
    loss_mask = mb["loss_mask"]
    temperatures = mb["temperatures"]
    full_seq_lens = mb.get("sequence_lengths")
    if full_seq_lens is None:
        full_seq_lens = [int(mb["input_ids"].shape[1])] * int(mb["input_ids"].shape[0])
    sequence_offsets = mb.get("sequence_offsets") or [0] * len(full_seq_lens)
    loss_cfg = DefaultLossConfig(kl_tau=0.0)

    def _full_slice(tensor: torch.Tensor, row: int, start: int, end: int) -> torch.Tensor:
        if horizontal:
            offset = int(sequence_offsets[row])
            return tensor[:, offset + start : offset + end]
        return tensor[row : row + 1, start:end]

    def _inputs_fn(full_start: int, full_end: int, row: int = 0):
        effective_logit_end = min(int(full_end), int(full_seq_lens[row]) - 1)
        if effective_logit_end <= full_start:
            return (
                _full_slice(labels, row, full_start, full_start),
                _full_slice(temperatures, row, full_start, full_start),
                effective_logit_end,
            )
        return (
            _full_slice(labels, row, full_start, effective_logit_end),
            _full_slice(temperatures, row, full_start, effective_logit_end),
            effective_logit_end,
        )

    def _selected_loss_fn(
        seg_logprobs: torch.Tensor,
        seg_entropy: torch.Tensor,
        full_start: int,
        effective_logit_end: int,
        row: int = 0,
    ) -> torch.Tensor:
        if effective_logit_end <= full_start:
            return seg_logprobs.sum() * 0.0
        tgt_start = full_start + 1
        tgt_end = effective_logit_end + 1
        seg_adv = _full_slice(advantages, row, tgt_start, tgt_end)
        seg_mask = _full_slice(loss_mask, row, tgt_start, tgt_end)
        seg_inf = _full_slice(inference_logprobs, row, tgt_start, tgt_end)
        seg_teach = (
            _full_slice(teacher_logprobs, row, tgt_start, tgt_end)
            if teacher_logprobs is not None
            else None
        )
        loss_val, _ = compute_loss(
            trainer_logprobs=(seg_logprobs.squeeze(0),),
            inference_logprobs=(seg_inf.squeeze(0),),
            teacher_logprobs=(seg_teach.squeeze(0),) if seg_teach is not None else None,
            advantages=(seg_adv.squeeze(0),),
            loss_mask=(seg_mask.squeeze(0),),
            loss_fn=lambda inputs: default_loss_fn(inputs, loss_cfg),
            loss_scale=loss_scale,
        )
        with torch.no_grad():
            if seg_mask.any():
                _ = (
                    seg_entropy.squeeze(0)[seg_mask.squeeze(0).bool()]
                    .detach()
                    .to("cpu")
                )
        return loss_val

    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with maybe_activation_offloading(activation_offloading_config):
        if mode == "single":
            selected_logprob_flex_mask_segmented_forward(
                model=model,
                calls=mb["calls"],
                merged_input_ids=mb["input_ids"],
                merged_position_ids=mb["position_ids"],
                inputs_fn=_inputs_fn,
                loss_fn=_selected_loss_fn,
                device=device,
                backward=True,
            )
        elif mode == "vertical":
            selected_logprob_batched_flex_mask_segmented_forward(
                model=model,
                calls_batch=calls_batch,
                merged_input_ids=mb["input_ids"],
                merged_position_ids=mb["position_ids"],
                inputs_fn=_inputs_fn,
                loss_fn=_selected_loss_fn,
                device=device,
                backward=True,
            )
        else:
            selected_logprob_packed_flex_mask_segmented_forward(
                model=model,
                calls_batch=calls_batch,
                merged_input_ids=mb["input_ids"],
                merged_position_ids=mb["position_ids"],
                inputs_fn=_inputs_fn,
                loss_fn=_selected_loss_fn,
                device=device,
                backward=True,
            )

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device)
    writer_tokens = sum(compute_flex_mask_writer_len(calls) for calls in calls_batch)
    return elapsed, writer_tokens, peak


def run_group(
    model: torch.nn.Module,
    group: list[dict],
    *,
    mode: str,
    device: torch.device,
    activation_offloading_config: ActivationOffloadingConfig | None = None,
) -> tuple[float, int, int]:
    if mode == "single":
        assert len(group) == 1
        mb = group[0]
        horizontal = False
    elif mode == "vertical":
        mb = stack_flex_compaction_micro_batches(group)
        horizontal = False
    elif mode == "horizontal":
        mb = pack_horizontal_flex_compaction_micro_batches(group)
        horizontal = True
    else:
        raise ValueError(f"unknown mode {mode!r}")

    mb = move_tensors(mb, device)
    calls_batch = calls_batch_from_maybe_stacked(mb["calls"])
    loss_scale = max(1, int(mb["loss_mask"].sum().item()))
    loss_fn = make_segment_loss_fn(mb, horizontal=horizontal, loss_scale=loss_scale)
    input_ids = mb["input_ids"]
    position_ids = mb["position_ids"]

    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    with maybe_activation_offloading(activation_offloading_config):
        if mode == "single":
            flex_mask_segmented_forward(
                model=model,
                calls=mb["calls"],
                merged_input_ids=input_ids,
                merged_position_ids=position_ids,
                loss_fn=loss_fn,
                device=device,
                backward=True,
            )
        elif mode == "vertical":
            batched_flex_mask_segmented_forward(
                model=model,
                calls_batch=calls_batch,
                merged_input_ids=input_ids,
                merged_position_ids=position_ids,
                loss_fn=loss_fn,
                device=device,
                backward=True,
            )
        else:
            packed_flex_mask_segmented_forward(
                model=model,
                calls_batch=calls_batch,
                merged_input_ids=input_ids,
                merged_position_ids=position_ids,
                loss_fn=loss_fn,
                device=device,
                backward=True,
            )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated(device)
    writer_tokens = 0
    calls_obj = mb["calls"]
    if mode == "single":
        writer_tokens = compute_flex_mask_writer_len(calls_obj)
    else:
        writer_tokens = sum(compute_flex_mask_writer_len(calls) for calls in calls_batch)
    return elapsed, writer_tokens, peak


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank-bin", type=Path, required=True)
    parser.add_argument(
        "--global-rank-bin",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional rank_N.bin paths from the same step. When present, "
            "stack token-budget grouping uses the same DP-global max writer "
            "lengths as production."
        ),
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--mode", choices=["single", "vertical", "horizontal"], default="single")
    parser.add_argument("--stack-size", type=int, default=16)
    parser.add_argument("--stack-token-budget", type=int, default=None)
    parser.add_argument("--max-groups", type=int, default=4)
    parser.add_argument("--skip-groups", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--random-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--activation-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--activation-checkpointing-freq", type=int, default=1)
    parser.add_argument("--activation-offloading", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--activation-offload-pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--activation-offload-max-inflight", type=int, default=1)
    parser.add_argument("--selected-logprob-head", action="store_true")
    parser.add_argument("--selected-logprob-chunk-size", type=int, default=8192)
    parser.add_argument("--rss-trace", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rss-trace-interval", type=float, default=0.05)
    parser.add_argument("--debug-num-layers", type=int, default=None)
    args = parser.parse_args()

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(1234)
    torch.backends.cuda.matmul.allow_tf32 = True

    raw = load_micro_batches(args.rank_bin)
    micro_batches = [micro_batch_to_tensor(mb) for mb in raw]
    micro_batches = [
        mb
        for mb in micro_batches
        if is_flex_compaction_stackable_micro_batch(
            mb,
            compaction_enabled=True,
            flex_compaction_enabled=True,
            cp_enabled=False,
            lora_enabled=False,
            multi_run_enabled=False,
        )
    ]
    global_seq_lens = None
    global_stackable = None
    if args.global_rank_bin and args.mode != "single" and args.stack_token_budget is not None:
        per_rank_batches = []
        for rank_bin in args.global_rank_bin:
            rank_raw = load_micro_batches(rank_bin)
            rank_batches = []
            for raw_mb in rank_raw:
                tensor_mb = micro_batch_to_tensor(raw_mb)
                if is_flex_compaction_stackable_micro_batch(
                    tensor_mb,
                    compaction_enabled=True,
                    flex_compaction_enabled=True,
                    cp_enabled=False,
                    lora_enabled=False,
                    multi_run_enabled=False,
                ):
                    rank_batches.append(tensor_mb)
            per_rank_batches.append(rank_batches)
        counts = {len(rank_batches) for rank_batches in per_rank_batches}
        counts.add(len(micro_batches))
        if len(counts) != 1:
            raise ValueError(
                "global rank bins must have the same stackable micro-batch count, "
                f"got {sorted(counts)}"
            )
        global_stackable = []
        global_seq_lens = []
        for idx in range(len(micro_batches)):
            flags = [
                is_flex_compaction_stackable_micro_batch(
                    rank_batches[idx],
                    compaction_enabled=True,
                    flex_compaction_enabled=True,
                    cp_enabled=False,
                    lora_enabled=False,
                    multi_run_enabled=False,
                )
                for rank_batches in per_rank_batches
            ]
            global_stackable.append(all(flags))
            seq_lens = []
            for rank_batches in per_rank_batches:
                mb = rank_batches[idx]
                if global_stackable[-1] and mb.get("calls"):
                    seq_lens.append(int(compute_flex_mask_writer_len(mb["calls"])))
                else:
                    seq_lens.append(int(mb["input_ids"].shape[1]))
            global_seq_lens.append(max(seq_lens))

    groups = group_micro_batches(
        args,
        micro_batches,
        seq_lens=global_seq_lens,
        flex_compaction_stackable=global_stackable,
    )
    if args.max_groups <= 0:
        groups = groups[args.skip_groups :]
    else:
        groups = groups[args.skip_groups : args.skip_groups + args.max_groups]

    print(
        f"loaded={len(raw)} stackable={len(micro_batches)} groups={len(groups)} "
        f"mode={args.mode} stack_size={args.stack_size} budget={args.stack_token_budget} "
        f"global_rank_bins={len(args.global_rank_bin)} "
        f"activation_offloading={args.activation_offloading} "
        f"pin_memory={args.activation_offload_pin_memory} "
        f"max_inflight={args.activation_offload_max_inflight}",
        flush=True,
    )
    if not groups:
        raise SystemExit("no groups selected")
    group_lens = [[int(mb["input_ids"].shape[1]) for mb in group] for group in groups]
    writer_lens = [
        [compute_flex_mask_writer_len(mb["calls"]) for mb in group]
        for group in groups
    ]
    print(f"group_lens={group_lens}", flush=True)
    print(f"writer_lens={writer_lens}", flush=True)

    model = build_model(args, device)
    runner = run_group_selected_logprob if args.selected_logprob_head else run_group
    activation_offloading_config = (
        ActivationOffloadingConfig(
            pin_memory=args.activation_offload_pin_memory,
            max_inflight_activations=args.activation_offload_max_inflight,
        )
        if args.activation_offloading
        else None
    )

    for _ in range(args.warmup):
        runner(
            model,
            groups[0],
            mode=args.mode,
            device=device,
            activation_offloading_config=activation_offloading_config,
        )

    rows = []
    for group_idx, group in enumerate(groups):
        times = []
        writer_tokens = 0
        peak = 0
        rss_before = rss_gib()
        rss_peak = rss_before
        rss_after = rss_before
        for _ in range(args.iters):
            with RssSampler(
                enabled=args.rss_trace,
                interval_s=args.rss_trace_interval,
            ) as rss_sampler:
                elapsed, writer_tokens, peak = runner(
                    model,
                    group,
                    mode=args.mode,
                    device=device,
                    activation_offloading_config=activation_offloading_config,
                )
            rss_after = rss_gib()
            rss_peak = max(rss_peak, rss_sampler.peak_gib, rss_after)
            times.append(elapsed)
        mean_time = statistics.mean(times)
        rows.append((mean_time, writer_tokens, peak))
        print(
            f"group={group_idx} rows={len(group)} sec={mean_time:.4f} "
            f"writer_tokens={writer_tokens} sec_per_mtok={mean_time / (writer_tokens / 1e6):.2f} "
            f"peak_gib={peak / (1024 ** 3):.2f} "
            f"rss_before_gib={rss_before:.2f} rss_peak_gib={rss_peak:.2f} "
            f"rss_after_gib={rss_after:.2f}",
            flush=True,
        )

    total_time = sum(row[0] for row in rows)
    total_tokens = sum(row[1] for row in rows)
    max_peak = max(row[2] for row in rows)
    print(
        f"SUMMARY mode={args.mode} groups={len(rows)} total_sec={total_time:.4f} "
        f"writer_tokens={total_tokens} sec_per_mtok={total_time / (total_tokens / 1e6):.2f} "
        f"peak_gib={max_peak / (1024 ** 3):.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
