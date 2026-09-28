from __future__ import annotations

import torch

from kv_eviction import segmented_forward as sf


class _FakeBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        return {"past_key_values": kwargs["past_key_values"]}


class _FakeRoot(torch.nn.Module):
    def __init__(self, backbone: _FakeBackbone) -> None:
        super().__init__()
        self.model = backbone
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        assert False, "FusedOutputLinear requires labels for chunked logprob computation"


def test_no_grad_prefill_retries_backbone_when_chunked_lm_head_needs_labels(monkeypatch) -> None:
    backbone = _FakeBackbone()
    model = _FakeRoot(backbone)
    monkeypatch.setattr(sf, "_backbone_embedding_weight_is_dtensor", lambda _backbone: True)

    out = sf._forward_no_grad_prefill(
        model=model,
        backbone=backbone,
        input_ids=torch.tensor([[1, 2, 3]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        past_key_values=None,
    )

    assert out == {"past_key_values": None}
    assert model.calls == 1
    assert backbone.calls == 1
