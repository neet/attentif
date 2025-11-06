import torch
from .positional_encoding import positional_encoding

def test_shape_and_dtype():
    seq_len, hidden_size = 7, 16
    pe = positional_encoding(seq_len, hidden_size)
    assert pe.shape == (seq_len, hidden_size)
    assert isinstance(pe, torch.Tensor)
    assert pe.dtype == torch.float32 or pe.dtype == torch.float64

def test_first_position_is_zero():
    """pos=0 のとき、sin(0)=0, cos(0)=1 なのでそれが反映されている"""
    seq_len, hidden_size = 8, 10
    pe = positional_encoding(seq_len, hidden_size)
    assert torch.allclose(pe[0, 0::2], torch.zeros(hidden_size // 2))
    assert torch.allclose(pe[0, 1::2], torch.ones(hidden_size // 2))

def test_value_range_is_reasonable():
    """sin, cos なので全要素は [-1, 1] 範囲にある"""
    seq_len, hidden_size = 32, 64
    pe = positional_encoding(seq_len, hidden_size)
    assert torch.all(pe <= 1.0)
    assert torch.all(pe >= -1.0)

def test_periodicity_pattern():
    """同じ位置差では同じ周期的パターンを持つこと"""
    seq_len, hidden_size = 20, 6
    pe = positional_encoding(seq_len, hidden_size)
    diff = pe[1] - pe[0]
    assert not torch.allclose(diff, torch.zeros_like(diff))
    # sin/cos の波形がちゃんと変化しているか
    assert torch.any(torch.abs(diff) > 1e-3)

def test_grad_disabled():
    """positional encoding は定数テンソルなので requires_grad=False"""
    seq_len, hidden_size = 4, 8
    pe = positional_encoding(seq_len, hidden_size)
    assert pe.requires_grad is False
