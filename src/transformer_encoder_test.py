import torch
import torch.nn as nn

from .transformer_encoder import TransformerEncoder

def _make_x(batch_size=2, seq_len=5, hidden_size=16, device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    return torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

def test_forward_shape_and_registration():
    hidden_size, num_attention_heads, num_blocks = 16, 4, 3
    m = TransformerEncoder(hidden_size, num_attention_heads, num_blocks)

    x = _make_x(hidden_size=hidden_size)
    y = m(x)

    # 形は保存される
    assert y.shape == x.shape

    # サブモジュールが正しく登録されている（ModuleList/長さ/パラメータ数）
    assert isinstance(m.blocks, nn.ModuleList)
    assert len(m.blocks) == num_blocks
    assert sum(p.numel() for p in m.parameters()) > 0

def test_backward_produces_grads():
    hidden_size, num_attention_heads, num_blocks = 16, 4, 2
    m = TransformerEncoder(hidden_size, num_attention_heads, num_blocks)

    x = _make_x(hidden_size=hidden_size).requires_grad_()
    y = m(x)
    loss = y.sum()
    loss.backward()

    # どれかのパラメータに勾配が載る
    assert any(p.grad is not None for p in m.parameters())

def test_eval_mode_noerror():
    hidden_size, num_attention_heads, num_blocks = 16, 4, 1
    m = TransformerEncoder(hidden_size, num_attention_heads, num_blocks).eval()

    x = _make_x(hidden_size=hidden_size)
    with torch.no_grad():
        y = m(x)

    assert y.shape == x.shape
