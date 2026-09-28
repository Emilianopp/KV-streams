#!/usr/bin/env python3
"""Local 1-2 GPU benchmark for FlexAttention compaction stacking.

This isolates the trainer-side flex compaction forward/backward from
TextWorld rollout generation. Run with torchrun, for example:

  CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=prime-rl/src:src \
    .venv/bin/torchrun --standalone --nproc-per-node=2 \
    scripts/bench_flex_stack_local.py --seq-len 16384 --stack-sizes 1,2,4
"""

from __future__ import annotations

import argparse
import os
import statistics
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from transformers import AutoConfig, AutoModelForCausalLM

from kv_eviction.segmented_forward import (
    batched_flex_mask_segmented_forward,
    compute_flex_mask_writer_len,
    flex_mask_segmented_forward,
    packed_flex_mask_segmented_forward,
)
from prime_rl.trainer.rl.loss import selective_log_softmax
from prime_rl.transport.types import CallWire, CompactionEventWire


@dataclass
class SyntheticSample:
    calls: list[CallWire]
    input_ids: torch.Tensor
    labels: torch.Tensor
    loss_mask: torch.Tensor
    writer_len: int


def _parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def _all_reduce_float(value: float, op: dist.ReduceOp) -> float:
    if not _dist_ready():
        return value
    device = torch.device("cuda", torch.cuda.current_device())
    tensor = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def _make_tokens(length: int, *, start: int, vocab_size: int) -> list[int]:
    # Avoid special-token edges and keep ids inside the model vocabulary.
    usable = max(1, vocab_size - 32)
    return [32 + ((start + i) % usable) for i in range(length)]


def make_synthetic_sample(
    *,
    seq_len: int,
    turns: int,
    protected_prefix: int,
    evict_tokens: int,
    vocab_size: int,
    rank: int,
    sample_idx: int,
    jitter_tokens: int,
    completion_tokens: int | None,
) -> SyntheticSample:
    """Build a multi-call sample with admission evictions after call 0."""
    if turns < 1:
        raise ValueError("turns must be >= 1")
    if protected_prefix < 0 or evict_tokens < 0:
        raise ValueError("protected_prefix and evict_tokens must be non-negative")

    # Split seq_len into one initial prompt plus per-call completion and
    # later user fragments. Jitter gives vertical stacking a realistic amount
    # of padding without changing the mode under test. For tiny smoke tests,
    # clip eviction geometry so the generated sample still stays near seq_len.
    local_seq_len = max(128, seq_len - max(0, jitter_tokens))
    effective_protected_prefix = min(protected_prefix, max(0, local_seq_len // 4))
    effective_evict_tokens = min(evict_tokens, max(0, local_seq_len // 8))
    min_initial_prompt_len = effective_protected_prefix + effective_evict_tokens + 8
    if completion_tokens is None:
        completion_len = max(8, local_seq_len // (2 * turns))
    else:
        max_completion_len = max(1, (local_seq_len - min_initial_prompt_len) // turns)
        completion_len = max(1, min(int(completion_tokens), max_completion_len))
    initial_prompt_len = max(min_initial_prompt_len, completion_len)
    remaining = max(0, local_seq_len - initial_prompt_len - turns * completion_len)
    fragment_len = max(1, remaining // max(1, turns - 1)) if turns > 1 else 0

    token_cursor = rank * 1_000_000 + sample_idx * 100_000
    post_tokens: list[int] = []
    merged_pretrim: list[int] = []
    calls: list[CallWire] = []
    total_offset_after = 0

    for turn in range(turns):
        if turn == 0:
            new_fragment = _make_tokens(
                initial_prompt_len,
                start=token_cursor,
                vocab_size=vocab_size,
            )
        else:
            new_fragment = _make_tokens(
                fragment_len,
                start=token_cursor,
                vocab_size=vocab_size,
            )
        token_cursor += len(new_fragment)
        completion = _make_tokens(
            completion_len,
            start=token_cursor,
            vocab_size=vocab_size,
        )
        token_cursor += len(completion)

        submitted_prompt = post_tokens + new_fragment if turn > 0 else new_fragment
        events: list[CompactionEventWire] = []
        can_evict = (
            turn > 0
            and effective_evict_tokens > 0
            and len(post_tokens) >= effective_protected_prefix + effective_evict_tokens
        )
        if can_evict:
            total_offset_after += effective_evict_tokens
            events.append(
                CompactionEventWire(
                    num_output_tokens_at_compaction=0,
                    tokens_evicted=effective_evict_tokens,
                    position_offset_after=total_offset_after,
                    num_prompt_tokens=len(submitted_prompt),
                    evict_start=effective_protected_prefix,
                    new_user_fragment_len=len(new_fragment),
                )
            )

        calls.append(
            CallWire(
                submitted_prompt_ids=submitted_prompt,
                completion_ids=completion,
                completion_logprobs=[0.0] * len(completion),
                completion_temperatures=[1.0] * len(completion),
                compaction_events=events,
            )
        )

        merged_pretrim.extend(new_fragment)
        merged_pretrim.extend(completion)
        if can_evict:
            del post_tokens[
                effective_protected_prefix:
                effective_protected_prefix + effective_evict_tokens
            ]
        post_tokens.extend(new_fragment)
        post_tokens.extend(completion)

    input_ids = torch.tensor(merged_pretrim, dtype=torch.long).unsqueeze(0)
    labels = torch.cat(
        [input_ids[:, 1:], torch.zeros((1, 1), dtype=torch.long)],
        dim=1,
    )
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    loss_mask[:, 0] = False
    writer_len = compute_flex_mask_writer_len(calls)
    return SyntheticSample(
        calls=calls,
        input_ids=input_ids,
        labels=labels,
        loss_mask=loss_mask,
        writer_len=writer_len,
    )


def _pad_rows(tensors: list[torch.Tensor], *, value: int | bool) -> torch.Tensor:
    max_len = max(int(t.shape[1]) for t in tensors)
    rows = []
    for tensor in tensors:
        pad_len = max_len - int(tensor.shape[1])
        if pad_len == 0:
            rows.append(tensor)
            continue
        pad = torch.full(
            (1, pad_len),
            value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        rows.append(torch.cat([tensor, pad], dim=1))
    return torch.cat(rows, dim=0)


def _pack_rows(tensors: list[torch.Tensor]) -> torch.Tensor:
    return torch.cat(tensors, dim=1)


def _make_loss_fn(
    samples: list[SyntheticSample],
    *,
    device: torch.device,
    normalize_tokens: int,
):
    labels = [sample.labels.to(device) for sample in samples]
    masks = [sample.loss_mask.to(device) for sample in samples]

    def loss_fn(
        seg_logits: torch.Tensor,
        full_logit_start: int,
        full_logit_end: int,
        batch_idx: int = 0,
    ) -> torch.Tensor:
        row_len = int(labels[batch_idx].shape[1])
        effective_end = min(int(full_logit_end), row_len - 1)
        if effective_end <= full_logit_start:
            return seg_logits.sum() * 0.0
        seg_logits_effective = seg_logits[
            :, : effective_end - full_logit_start, :
        ]
        seg_labels = labels[batch_idx][
            :, full_logit_start:effective_end
        ]
        seg_logprobs = selective_log_softmax(seg_logits_effective, seg_labels)
        tgt_start = full_logit_start + 1
        tgt_end = effective_end + 1
        seg_mask = masks[batch_idx][:, tgt_start:tgt_end]
        if not seg_mask.any():
            return seg_logits.sum() * 0.0
        return -seg_logprobs[seg_mask].sum() / normalize_tokens

    return loss_fn


def run_mode(
    *,
    mode: str,
    model: torch.nn.Module,
    samples: list[SyntheticSample],
    device: torch.device,
) -> tuple[float, int, int]:
    normalize_tokens = sum(int(sample.loss_mask.sum().item()) for sample in samples)
    loss_fn = _make_loss_fn(samples, device=device, normalize_tokens=normalize_tokens)
    writer_tokens = sum(sample.writer_len for sample in samples)

    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start = time.perf_counter()

    if mode == "single":
        for sample in samples:
            input_ids = sample.input_ids.to(device)
            position_ids = torch.arange(
                input_ids.shape[1],
                device=device,
                dtype=torch.long,
            ).unsqueeze(0)
            local_loss_fn = _make_loss_fn(
                [sample],
                device=device,
                normalize_tokens=normalize_tokens,
            )
            flex_mask_segmented_forward(
                model=model,
                calls=sample.calls,
                merged_input_ids=input_ids,
                merged_position_ids=position_ids,
                loss_fn=local_loss_fn,
                device=device,
                backward=True,
            )
    elif mode == "vertical":
        merged_input_ids = _pad_rows(
            [sample.input_ids.to(device) for sample in samples],
            value=1,
        )
        merged_position_ids = _pad_rows(
            [
                torch.arange(
                    sample.input_ids.shape[1],
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(0)
                for sample in samples
            ],
            value=0,
        )
        batched_flex_mask_segmented_forward(
            model=model,
            calls_batch=[sample.calls for sample in samples],
            merged_input_ids=merged_input_ids,
            merged_position_ids=merged_position_ids,
            loss_fn=loss_fn,
            device=device,
            backward=True,
        )
    elif mode == "horizontal":
        merged_input_ids = _pack_rows([sample.input_ids.to(device) for sample in samples])
        merged_position_ids = _pack_rows(
            [
                torch.arange(
                    sample.input_ids.shape[1],
                    device=device,
                    dtype=torch.long,
                ).unsqueeze(0)
                for sample in samples
            ]
        )
        packed_flex_mask_segmented_forward(
            model=model,
            calls_batch=[sample.calls for sample in samples],
            merged_input_ids=merged_input_ids,
            merged_position_ids=merged_position_ids,
            loss_fn=loss_fn,
            device=device,
            backward=True,
        )
    else:
        raise ValueError(f"unknown mode {mode!r}")

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak_mem = torch.cuda.max_memory_allocated(device)
    return elapsed, writer_tokens, peak_mem


def _apply_prime_block_checkpointing(model: torch.nn.Module, *, freq: int = 1) -> None:
    language_model = getattr(model, "model", model)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        raise ValueError(
            "production-style block activation checkpointing requires "
            "model.model.layers"
        )
    for layer_id, (layer_name, layer) in enumerate(layers.named_children()):
        if layer_id % freq == 0:
            layers.register_module(
                layer_name,
                checkpoint_wrapper(layer, preserve_rng_state=False),
            )


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    config = AutoConfig.from_pretrained(
        args.model,
        attn_implementation=args.attn,
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
    model.train()
    if args.activation_checkpointing == "prime":
        _apply_prime_block_checkpointing(model, freq=1)
    elif (
        args.activation_checkpointing == "hf"
        and hasattr(model, "gradient_checkpointing_enable")
    ):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--attn", default="flex_attention")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--random-init", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--activation-checkpointing",
        choices=["prime", "hf", "none"],
        default="prime",
        help=(
            "prime matches trainer.model.ac mode=full,freq=1 by wrapping "
            "decoder blocks with checkpoint_wrapper."
        ),
    )
    parser.add_argument("--debug-num-layers", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument(
        "--completion-tokens",
        type=int,
        default=None,
        help="Synthetic completion tokens per call. Omit for seq_len-derived sizing.",
    )
    parser.add_argument("--protected-prefix", type=int, default=1024)
    parser.add_argument("--evict-tokens", type=int, default=512)
    parser.add_argument("--stack-sizes", default="1,2,4")
    parser.add_argument("--modes", default="single,vertical,horizontal")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--jitter-step", type=int, default=0)
    args = parser.parse_args()

    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(1234 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    model = build_model(args, device)
    vocab_size = int(getattr(model.config, "vocab_size", 32000))
    stack_sizes = _parse_ints(args.stack_sizes)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]

    if rank == 0:
        print(
            "mode\tstack\tlocal_sec_mean\tglobal_sec_mean\tglobal_sec_max\t"
            "sec_per_mtok\twriter_tokens_per_rank\tpeak_mem_gib_max",
            flush=True,
        )

    for stack_size in stack_sizes:
        samples = [
            make_synthetic_sample(
                seq_len=args.seq_len,
                turns=args.turns,
                protected_prefix=args.protected_prefix,
                evict_tokens=args.evict_tokens,
                vocab_size=vocab_size,
                rank=rank,
                sample_idx=idx,
                jitter_tokens=args.jitter_step * idx,
                completion_tokens=args.completion_tokens,
            )
            for idx in range(stack_size)
        ]

        for mode in modes:
            if mode != "single" and stack_size == 1:
                continue
            for _ in range(args.warmup):
                run_mode(mode=mode, model=model, samples=samples, device=device)
            times: list[float] = []
            writer_tokens = 0
            peak_mem = 0
            for _ in range(args.iters):
                elapsed, writer_tokens, peak_mem = run_mode(
                    mode=mode,
                    model=model,
                    samples=samples,
                    device=device,
                )
                times.append(elapsed)

            local_mean = statistics.mean(times)
            global_sum = _all_reduce_float(local_mean, dist.ReduceOp.SUM)
            global_max = _all_reduce_float(local_mean, dist.ReduceOp.MAX)
            global_writer_tokens = _all_reduce_float(float(writer_tokens), dist.ReduceOp.SUM)
            global_peak = _all_reduce_float(float(peak_mem), dist.ReduceOp.MAX)
            global_mean = global_sum / world_size
            sec_per_mtok = global_max / max(global_writer_tokens / 1_000_000.0, 1e-9)
            if rank == 0:
                print(
                    f"{mode}\t{stack_size}\t{local_mean:.4f}\t{global_mean:.4f}\t"
                    f"{global_max:.4f}\t{sec_per_mtok:.2f}\t"
                    f"{writer_tokens}\t{global_peak / (1024 ** 3):.2f}",
                    flush=True,
                )

    if _dist_ready():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
