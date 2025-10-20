import torch
import torch.nn as nn

from .transformer_encoder import TransformerEncoder

def _make_x(B=2, S=5, H=16, device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    return torch.randn(B, S, H, device=device, dtype=dtype)

def test_forward_shape_and_registration():
    H, h, n = 16, 4, 3
    m = TransformerEncoder(H, h, n)

    x = _make_x(H=H)
    y = m(x)

    # 形は保存される
    assert y.shape == x.shape

    # サブモジュールが正しく登録されている（ModuleList/長さ/パラメータ数）
    assert isinstance(m.blocks, nn.ModuleList)
    assert len(m.blocks) == n
    assert sum(p.numel() for p in m.parameters()) > 0

def test_backward_produces_grads():
    H, h, n = 16, 4, 2
    m = TransformerEncoder(H, h, n)

    x = _make_x(H=H).requires_grad_()
    y = m(x)
    loss = y.sum()
    loss.backward()

    # どれかのパラメータに勾配が載る
    assert any(p.grad is not None for p in m.parameters())

def test_eval_mode_noerror():
    H, h, n = 16, 4, 1
    m = TransformerEncoder(H, h, n).eval()

    x = _make_x(H=H)
    with torch.no_grad():
        y = m(x)

    assert y.shape == x.shape
