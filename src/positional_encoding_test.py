import torch
from .positional_encoding import positional_encoding

def test_shape_and_dtype():
    s, h = 7, 16
    pe = positional_encoding(s, h)
    assert pe.shape == (s, h)
    assert isinstance(pe, torch.Tensor)
    assert pe.dtype == torch.float32 or pe.dtype == torch.float64

def test_first_position_is_zero():
    """pos=0 のとき、sin(0)=0, cos(0)=1 なのでそれが反映されている"""
    s, h = 8, 10
    pe = positional_encoding(s, h)
    assert torch.allclose(pe[0, 0::2], torch.zeros(h // 2))
    assert torch.allclose(pe[0, 1::2], torch.ones(h // 2))

def test_value_range_is_reasonable():
    """sin, cos なので全要素は [-1, 1] 範囲にある"""
    s, h = 32, 64
    pe = positional_encoding(s, h)
    assert torch.all(pe <= 1.0)
    assert torch.all(pe >= -1.0)

def test_periodicity_pattern():
    """同じ位置差では同じ周期的パターンを持つこと"""
    s, h = 20, 6
    pe = positional_encoding(s, h)
    diff = pe[1] - pe[0]
    assert not torch.allclose(diff, torch.zeros_like(diff))
    # sin/cos の波形がちゃんと変化しているか
    assert torch.any(torch.abs(diff) > 1e-3)

def test_grad_disabled():
    """positional encoding は定数テンソルなので requires_grad=False"""
    s, h = 4, 8
    pe = positional_encoding(s, h)
    assert pe.requires_grad is False
