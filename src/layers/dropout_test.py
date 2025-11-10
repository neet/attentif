import torch
import pytest

from .dropout import Dropout

def test_dropout_noop_when_not_training():
    x = torch.randn(8, 16)
    dropout_layer = Dropout(prob=0.5)
    dropout_layer.eval()
    y = dropout_layer(x)

    dropout_layer_zero = Dropout(prob=0.0)
    dropout_layer_zero.train()
    z = dropout_layer_zero(x)

    assert torch.allclose(y, x)
    assert torch.allclose(z, x)
    assert y.dtype == x.dtype and y.device == x.device
    assert z.dtype == x.dtype and z.device == x.device

def test_dropout_scale_preserves_mean_statistically():
    torch.manual_seed(0)
    prob = 0.3
    # 十分大きいテンソルで期待値 ~ 入力平均 を確認
    x = torch.ones(20000, dtype=torch.float32)
    dropout_layer = Dropout(prob=prob)
    dropout_layer.train()
    y = dropout_layer(x)
    # 期待値は 1.0（keep率*(1/(1-prob))*1 = 1）
    assert y.mean().item() == pytest.approx(1.0, rel=0.0, abs=0.03)
    # 落ちている要素が存在する（確率試験のスモーク）
    assert (y == 0).sum().item() > 0
    # スケールの上限（保持要素は 1/(1-prob)）
    assert y.max().item() == pytest.approx(1.0 / (1.0 - prob), rel=0.02)

def test_dropout_preserves_zeros_and_dtype_device():
    x = torch.zeros(1024, device="cpu", dtype=torch.float32)
    dropout_layer = Dropout(prob=0.4)
    dropout_layer.train()
    y = dropout_layer(x)
    assert torch.count_nonzero(y) == 0
    assert y.dtype == x.dtype and y.device == x.device
