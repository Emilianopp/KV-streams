import sys
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vllm"))

from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.core.single_type_kv_cache_manager import MambaManager
from vllm.v1.outputs import AttentionMatchingCompactionResult
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _bind_mrope_helpers(runner: SimpleNamespace) -> None:
    runner._mrope_mm_token_ids = MethodType(
        GPUModelRunner._mrope_mm_token_ids, runner
    )
    runner._assert_attention_matching_text_only_mrope_request = MethodType(
        GPUModelRunner._assert_attention_matching_text_only_mrope_request,
        runner,
    )


def _bind_kv_slot_helpers(runner: SimpleNamespace) -> None:
    for name in (
        "_attention_matching_try_token_kv_cache_view",
        "_attention_matching_token_kv_cache_view",
        "_attention_matching_validate_block_kv_cache",
        "_attention_matching_index_select_kv_slots",
        "_attention_matching_index_copy_kv_plane_slots",
        "_attention_matching_index_copy_kv_slots",
        "_attention_matching_kv_cache_gid_for_layer",
        "_attention_matching_logical_block_size",
        "_build_attention_matching_slot_mapping",
    ):
        setattr(runner, name, MethodType(getattr(GPUModelRunner, name), runner))


def test_attention_matching_mrope_positions_include_compaction_offset():
    runner = SimpleNamespace(
        uses_mrope=True,
        _attention_matching_enabled=True,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(image_token_id=99),
            hf_text_config=None,
        ),
        input_batch=SimpleNamespace(
            req_ids=["req"],
            num_computed_tokens_cpu=[5],
        ),
        requests={
            "req": SimpleNamespace(
                prompt_token_ids=[1, 2, 3],
                prompt_embeds=None,
                mm_features=[],
                position_offset=100,
            )
        },
        arange_np=np.arange(16, dtype=np.int64),
        mrope_positions=SimpleNamespace(np=np.zeros((3, 16), dtype=np.int64)),
    )
    _bind_mrope_helpers(runner)

    GPUModelRunner._calc_text_only_mrope_positions_for_attention_matching(
        runner,
        SimpleNamespace(num_scheduled_tokens={"req": 3}),
    )

    expected = np.array([105, 106, 107], dtype=np.int64)
    assert np.array_equal(runner.mrope_positions.np[0, :3], expected)
    assert np.array_equal(runner.mrope_positions.np[1, :3], expected)
    assert np.array_equal(runner.mrope_positions.np[2, :3], expected)


def test_attention_matching_mrope_init_rejects_multimodal_sentinels():
    runner = SimpleNamespace(
        uses_mrope=True,
        _attention_matching_enabled=True,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(image_token_id=99),
            hf_text_config=None,
        ),
    )
    _bind_mrope_helpers(runner)
    req = SimpleNamespace(
        prompt_token_ids=[1, 99, 2],
        prompt_embeds=None,
        mm_features=[],
    )

    with pytest.raises(RuntimeError, match="multimodal sentinel"):
        GPUModelRunner._init_text_only_mrope_positions_for_attention_matching(
            runner, req
        )


def test_attention_matching_hybrid_validation_selects_full_attention_layers_only():
    full_spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=16,
        dtype=torch.float16,
    )
    mamba_spec = MambaSpec(
        block_size=16,
        shapes=((1,),),
        dtypes=(torch.float16,),
    )
    runner = SimpleNamespace(
        _attention_matching_enabled=True,
        _shuffle_control_enabled=False,
        _noise_control_enabled=False,
        speculative_config=None,
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            attention_matching_zerobeta=True,
        ),
        uses_mrope=True,
        uses_xdrope_dim=0,
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=mamba_spec),
                SimpleNamespace(kv_cache_spec=full_spec),
            ]
        ),
        attn_groups=[
            [
                SimpleNamespace(
                    kv_cache_spec=mamba_spec,
                    layer_names=["linear_layer"],
                )
            ],
            [
                SimpleNamespace(
                    kv_cache_spec=full_spec,
                    layer_names=["full_layer_0", "full_layer_1"],
                )
            ],
        ],
        _attention_matching_layer_names=[],
        _attention_matching_layer_to_kv_cache_gid={},
    )

    GPUModelRunner._validate_attention_matching_support(runner)

    assert runner._attention_matching_layer_names == [
        "full_layer_0",
        "full_layer_1",
    ]
    assert runner._attention_matching_layer_to_kv_cache_gid == {
        "full_layer_0": 1,
        "full_layer_1": 1,
    }


def test_attention_matching_slot_mapping_uses_layer_kv_cache_group():
    block_size = 4
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        cache_config=SimpleNamespace(block_size=block_size),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=block_size)),
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=block_size)),
            ]
        ),
        _attention_matching_positions=torch.arange(16, dtype=torch.long),
        _attention_matching_layer_to_kv_cache_gid={"full_layer": 1},
    )
    _bind_kv_slot_helpers(runner)
    req_state = SimpleNamespace(
        block_ids=(
            [0, 10],
            [0, 20],
        )
    )

    group0_slots = runner._build_attention_matching_slot_mapping(
        req_state,
        8,
        kv_cache_gid=0,
    )
    full_layer_slots = runner._build_attention_matching_slot_mapping(
        req_state,
        8,
        kv_cache_gid=runner._attention_matching_kv_cache_gid_for_layer("full_layer"),
    )

    assert torch.equal(group0_slots, torch.tensor([0, 1, 2, 3, 40, 41, 42, 43]))
    assert torch.equal(
        full_layer_slots,
        torch.tensor([0, 1, 2, 3, 80, 81, 82, 83]),
    )


def test_attention_matching_kv_slot_helpers_keep_flattened_fast_path():
    block_size = 4
    layer = SimpleNamespace(impl=SimpleNamespace(num_kv_heads=2, head_size=3))
    runner = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        _attention_matching_flat_kv_view_cache={},
    )
    _bind_kv_slot_helpers(runner)
    kv_cache = torch.arange(
        2 * 3 * block_size * 2 * 3,
        dtype=torch.float32,
    ).reshape(2, 3, block_size, 2, 3)
    slots = torch.tensor([0, 3, 4, 9], dtype=torch.long)

    selected = runner._attention_matching_index_select_kv_slots(
        kv_cache,
        layer,
        slots,
    )

    flattened = kv_cache.view(2, -1, 2, 3)
    assert runner._attention_matching_flat_kv_view_cache[id(kv_cache)] is True
    assert torch.equal(selected, flattened.index_select(1, slots))

    values = torch.full_like(selected, -7.0)
    runner._attention_matching_index_copy_kv_slots(
        kv_cache,
        layer,
        slots,
        values,
    )
    assert torch.equal(flattened.index_select(1, slots), values)


def test_attention_matching_kv_slot_helpers_support_noncontiguous_block_cache():
    block_size = 4
    layer = SimpleNamespace(impl=SimpleNamespace(num_kv_heads=2, head_size=3))
    runner = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=block_size),
        _attention_matching_flat_kv_view_cache={},
    )
    _bind_kv_slot_helpers(runner)
    kv_cache = (
        torch.arange(2 * 3 * block_size * 2 * 3, dtype=torch.float32)
        .reshape(2, 3, block_size, 2, 3)
        .transpose(1, 2)
        .contiguous()
        .transpose(1, 2)
    )
    assert not kv_cache.is_contiguous()
    with pytest.raises(RuntimeError):
        kv_cache.view(2, -1, 2, 3)

    slots = torch.tensor([0, 3, 4, 9], dtype=torch.long)
    block_indices = torch.div(slots, block_size, rounding_mode="floor")
    block_offsets = slots.remainder(block_size)

    selected = runner._attention_matching_index_select_kv_slots(
        kv_cache,
        layer,
        slots,
    )

    assert runner._attention_matching_flat_kv_view_cache[id(kv_cache)] is False
    assert torch.equal(selected, kv_cache[:, block_indices, block_offsets])

    values = torch.full_like(selected, -11.0)
    runner._attention_matching_index_copy_kv_slots(
        kv_cache,
        layer,
        slots,
        values,
    )
    assert torch.equal(kv_cache[:, block_indices, block_offsets], values)


def test_attention_matching_kv_slot_helpers_support_hybrid_kernel_block_cache():
    logical_block_size = 8
    physical_block_size = 4
    layer = SimpleNamespace(impl=SimpleNamespace(num_kv_heads=2, head_size=3))
    runner = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=logical_block_size),
        _attention_matching_flat_kv_view_cache={},
    )
    _bind_kv_slot_helpers(runner)
    kv_cache = (
        torch.arange(2 * 4 * physical_block_size * 2 * 3, dtype=torch.float32)
        .reshape(2, 4, physical_block_size, 2, 3)
        .transpose(1, 2)
        .contiguous()
        .transpose(1, 2)
    )
    assert not kv_cache.is_contiguous()
    with pytest.raises(RuntimeError):
        kv_cache.view(2, -1, 2, 3)

    # Hybrid managers use larger logical blocks, while full-attention kernels
    # still store KV in smaller physical blocks. AM slots are linear token slots,
    # so they must be decomposed by the physical cache block size.
    slots = torch.tensor([0, 3, 4, 7, 8, 11, 12, 15], dtype=torch.long)
    block_indices = torch.div(slots, physical_block_size, rounding_mode="floor")
    block_offsets = slots.remainder(physical_block_size)

    selected = runner._attention_matching_index_select_kv_slots(
        kv_cache,
        layer,
        slots,
    )

    assert runner._attention_matching_flat_kv_view_cache[id(kv_cache)] is False
    assert torch.equal(selected, kv_cache[:, block_indices, block_offsets])

    values = torch.full_like(selected, -13.0)
    runner._attention_matching_index_copy_kv_slots(
        kv_cache,
        layer,
        slots,
        values,
    )
    assert torch.equal(kv_cache[:, block_indices, block_offsets], values)


def test_attention_matching_mamba_align_manager_reindexes_exact_state_after_shrink():
    class FakeBlock:
        def __init__(self, block_id: int) -> None:
            self.block_id = block_id
            self.is_null = False

    class FakeBlockPool:
        def __init__(self) -> None:
            self.freed: list[FakeBlock] = []

        def free_blocks(self, blocks: list[FakeBlock]) -> None:
            self.freed.extend(blocks)

    blocks = [FakeBlock(i) for i in range(14)]
    block_pool = FakeBlockPool()
    manager = SimpleNamespace(
        mamba_cache_mode="align",
        req_to_blocks={"req": list(blocks)},
        block_size=4,
        num_speculative_blocks=0,
        block_pool=block_pool,
        last_state_block_idx={},
        _allocated_block_reqs=set(),
        num_cached_block={"req": 14},
    )
    manager.finalize_attention_matching_compaction = MethodType(
        MambaManager.finalize_attention_matching_compaction,
        manager,
    )

    freed_tokens = manager.finalize_attention_matching_compaction("req", 40)

    assert len(manager.req_to_blocks["req"]) == 10
    assert manager.req_to_blocks["req"][9] is blocks[13]
    assert manager.last_state_block_idx["req"] == 9
    assert manager.num_cached_block["req"] == 10
    assert "req" in manager._allocated_block_reqs
    assert [block.block_id for block in block_pool.freed] == [9, 10, 11, 12]
    assert freed_tokens == 16


def test_attention_matching_worker_clears_stale_mamba_state_idx_after_shrink():
    tokens = list(range(14))
    req_state = SimpleNamespace(
        req_id="req",
        num_tokens=len(tokens),
        get_token_id=lambda i: tokens[i],
        prompt_token_ids=tokens[:4],
        prompt_embeds=None,
        num_prompt_tokens=4,
        output_token_ids=tokens[4:],
        position_offset=0,
        num_computed_tokens=len(tokens),
        attention_matching_synthetic_prefix_len=0,
        attention_matching_prefix_cache_key=None,
        attention_matching_prefix_cache_key_start=0,
        attention_matching_block_ids_tensor=object(),
        attention_matching_plan_skip_logged=True,
    )
    runner = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        mamba_state_idx={"req": 13},
        uses_mrope=False,
    )
    result = AttentionMatchingCompactionResult(
        request_id="req",
        source_len=12,
        target_len=8,
        protected_prefix_len=2,
        synthetic_prefix_len=2,
        exact_kept_tokens=4,
        position_offset_delta=4,
        pre_sample=True,
        prefix_cache_key="am-key",
        prefix_cache_key_start=2,
        physical_token_ids=[0, 1, 0, 0, 8, 9, 10, 11],
    )

    physical = GPUModelRunner._apply_attention_matching_result_to_worker_state(
        runner,
        req_state,
        result,
    )

    assert physical == [0, 1, 0, 0, 8, 9, 10, 11]
    assert "req" not in runner.mamba_state_idx
    assert req_state.num_computed_tokens == 7
    assert req_state.position_offset == 4
    assert req_state.attention_matching_block_ids_tensor is None
