import torch

from .multi_head_attention import (
    MultiHeadAttention,
    make_causal_mask,
    make_padding_mask,
)

def make_model_and_inputs(*, B=2, S=5, H=16, h=4, d_k=4, d_v=4,
                          pad_token=0, device="cpu", dtype=torch.float32, seed=0):
    assert H == h * d_k, "H must equal h * d_k"
    torch.manual_seed(seed)

    model = MultiHeadAttention(h=h, d_k=d_k, d_v=d_v, pad_token=pad_token).to(device)

    # 入力（埋め込み）と input_ids（末尾をpad）
    x = torch.randn(B, S, H, device=device, dtype=dtype)
    input_ids = torch.randint(1, 50, (B, S), device=device)
    input_ids[:, -2:] = pad_token
    return model, x, input_ids

def test_forward_shape_and_dtype_device():
    m, x, input_ids = make_model_and_inputs(B=3, S=7, H=24, h=6, d_k=4, d_v=4,
                                            device="cpu", dtype=torch.float32)
    y = m(x, input_ids)
    assert y.shape == x.shape == (3, 7, 24)
    assert y.dtype == x.dtype
    assert y.device == x.device

def test_masks_match_scores_dtype_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    S = 5
    cm = make_causal_mask(S, dtype=dtype, device=device)
    input_ids = torch.tensor([[1, 2, 3, 0, 0]], device=device)
    pm = make_padding_mask(input_ids, pad_token=0, dtype=dtype, device=device)

    assert cm.dtype == dtype and pm.dtype == dtype
    assert cm.device == device and pm.device == device
    assert cm.shape == (S, S)
    assert pm.shape == (1, S)

def test_padding_positions_do_not_affect_nonpad_queries():
    m, x, input_ids = make_model_and_inputs(B=1, S=6, H=16, h=4, d_k=4, d_v=4,
                                            pad_token=0, device="cpu")
    y0 = m(x, input_ids).detach()

    x_perturbed = x.clone()
    x_perturbed[:, -2:, :] += 1e3  # padキー側に巨大ノイズ
    y1 = m(x_perturbed, input_ids).detach()

    torch.testing.assert_close(y0[:, :-2, :], y1[:, :-2, :], rtol=1e-4, atol=1e-4)

def test_causal_mask_blocks_future_information():
    m, x, input_ids = make_model_and_inputs(B=1, S=6, H=16, h=4, d_k=4, d_v=4,
                                            pad_token=0, device="cpu")
    t = 3
    y0 = m(x, input_ids).detach()

    x_future = x.clone()
    x_future[:, t+1:, :] += 1e3
    y1 = m(x_future, input_ids).detach()

    torch.testing.assert_close(y0[:, :t+1, :], y1[:, :t+1, :], rtol=1e-4, atol=1e-4)

def test_output_projection_zero_makes_zero_output():
    m, x, input_ids = make_model_and_inputs(B=2, S=5, H=16, h=4, d_k=4, d_v=4)

    with torch.no_grad():
        m.W_O.zero_()
        m.b_O.zero_()

    y = m(x, input_ids)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)

def test_dv_not_equal_dk_supported():
    m, x, input_ids = make_model_and_inputs(B=2, S=5, H=16, h=4, d_k=4, d_v=8)
    y = m(x, input_ids)
    assert y.shape == (2, 5, 16)

def test_backward_grads_exist():
    m, x, input_ids = make_model_and_inputs(B=2, S=5, H=16, h=4, d_k=4, d_v=4,
                                            dtype=torch.float32)
    # 勾配計算
    x.requires_grad_(True)
    y = m(x, input_ids)
    loss = y.pow(2).mean()
    loss.backward()

    # 主要パラメータに勾配があること
    for name, p in m.named_parameters():
        assert p.requires_grad, f"{name} requires_grad is False"
        assert p.grad is not None, f"{name} has no grad"

    assert x.grad is not None
