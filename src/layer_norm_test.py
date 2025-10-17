import torch
import pytest

from .layer_norm import LayerNorm

@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)

@pytest.mark.parametrize("shape", [(2, 3, 4), (1, 5, 8), (4, 1, 16)])
def test_output_shape_matches_input(shape):
    B, S, H = shape
    x = torch.randn(B, S, H)
    ln = LayerNorm(H)
    y = ln(x)
    assert y.shape == x.shape

def test_normalization_zero_mean_unit_var_when_affine_identity():
    B, S, H = 2, 3, 7
    x = torch.randn(B, S, H)
    ln = LayerNorm(H, eps=1e-5)

    # γ=1, β=0 にして純粋な正規化の性質を確認
    with torch.no_grad():
        ln.gamma.fill_(1.0)
        ln.beta.zero_()

    y = ln(x)
    mean = y.mean(dim=-1)  # (B,S)
    std  = y.std(dim=-1, unbiased=False)  # (B,S)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std,  torch.ones_like(std),  atol=1e-4)

def test_affine_broadcast_and_values():
    B, S, H = 2, 2, 6
    x = torch.randn(B, S, H)
    eps = 1e-5
    ln = LayerNorm(H, eps=eps)

    # わかりやすい γ/β
    with torch.no_grad():
        ln.gamma.copy_(torch.linspace(0.5, 1.5, H))
        ln.beta.copy_(torch.linspace(-0.3, 0.3, H))

    # 期待値を手計算
    mean = x.mean(dim=-1, keepdim=True)
    var  = x.var(dim=-1, unbiased=False, keepdim=True)
    normed = (x - mean) / torch.sqrt(var + eps)
    expected = normed * ln.gamma + ln.beta

    y = ln(x)
    assert torch.allclose(y, expected, atol=1e-6)

def test_matches_torch_nn_layernorm():
    B, S, H = 3, 4, 9
    x = torch.randn(B, S, H)
    eps = 1e-5

    ours = LayerNorm(H, eps=eps)
    ref  = torch.nn.LayerNorm(H, eps=eps, elementwise_affine=True)

    # 同じ γ/β を両者に設定
    with torch.no_grad():
        g = torch.randn(H)
        b = torch.randn(H)
        ours.gamma.copy_(g)
        ours.beta.copy_(b)
        ref.weight.copy_(g)
        ref.bias.copy_(b)

    y_ours = ours(x)
    y_ref  = ref(x)

    assert torch.allclose(y_ours, y_ref, atol=1e-6)

def test_backward_grads_exist_and_have_reasonable_shapes():
    B, S, H = 2, 3, 5
    x = torch.randn(B, S, H, requires_grad=True)
    ln = LayerNorm(H)

    y = ln(x)
    loss = (y ** 2).mean()
    loss.backward()

    assert x.grad is not None and x.grad.shape == x.shape
    assert ln.gamma.grad is not None and ln.gamma.grad.shape == (H,)
    assert ln.beta.grad is not None and ln.beta.grad.shape == (H,)
    # 勾配が全部ゼロではないこと（パスが繋がっているかのスモーク）
    assert ln.gamma.grad.abs().sum() > 0
    assert ln.beta.grad.abs().sum() > 0

def test_numerical_stability_constant_input():
    B, S, H = 4, 6, 8
    x = torch.ones(B, S, H) * 3.14
    ln = LayerNorm(H, eps=1e-5)

    y = ln(x)
    assert torch.isfinite(y).all()

    with torch.no_grad():
        ln.gamma.fill_(1.0)
        ln.beta.zero_()
    y2 = ln(x)

    assert torch.allclose(y2, torch.zeros_like(y2), atol=1e-4)

def test_dtype_and_device_cpu_float32():
    x = torch.randn(2, 3, 4, dtype=torch.float32)
    ln = LayerNorm(4)
    y = ln(x)
    assert y.dtype == torch.float32

