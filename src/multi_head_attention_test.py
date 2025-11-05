import torch

from .multi_head_attention import MultiHeadAttention
from .mask_padding import make_padding_mask
from .mask_causal import make_causal_mask


def make_model_and_inputs(
    *, batch_size=2, seq_len=5, hidden_size=16, num_attention_heads=4, d_k=4, d_v=4, device="cpu", dtype=torch.float32, seed=0
):
    """
    NOTE: pad_token_id は MultiHeadAttention 側には渡さない。
    テスト内で pad_token_id=0 前提の padding mask を合成する。
    """
    assert hidden_size == num_attention_heads * d_k, "H must equal h * d_k"
    torch.manual_seed(seed)

    m = MultiHeadAttention(num_attention_heads=num_attention_heads, d_k=d_k, d_v=d_v).to(device)
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    # ランダム input_ids は [1, 49] で生成（0 は pad 扱いにするため除外）
    input_ids = torch.randint(1, 50, (batch_size, seq_len), device=device)
    return m, x, input_ids


def _full_mask(
    input_ids: torch.Tensor, *, pad_token_id: int, dtype: torch.dtype, device: torch.device
):
    """
    (B,S) pad_mask を (B,1,S) にして key列へ、(S,S) causal はそのまま。
    ふつうに足し算するだけで (B,S,S) にブロードキャストされる。
    """
    pm = make_padding_mask(input_ids, pad_token_id=0)
    cm = make_causal_mask(input_ids)
    mask = pm.unsqueeze(1) + cm
    return mask


def test_forward_shape_and_dtype_device():
    m, x, input_ids = make_model_and_inputs(batch_size=3, seq_len=7, hidden_size=24, num_attention_heads=6, d_k=4, d_v=4,
                                            device="cpu", dtype=torch.float32)
    pad = 0
    mask = _full_mask(input_ids, pad_token_id=pad, dtype=x.dtype, device=x.device)
    y = m(x, mask)
    assert y.shape == (3, 7, 24)
    assert y.dtype == x.dtype and y.device == x.device

def test_none_mask_equals_zero_mask():
    m, x, _ = make_model_and_inputs(batch_size=2, seq_len=5, hidden_size=16, num_attention_heads=4, d_k=4, d_v=4,
                                    device="cpu", dtype=torch.float32)
    B, S, _H = x.shape
    zero = torch.zeros(B, S, S, dtype=x.dtype, device=x.device)
    y_none = m(x, None)
    y_zero = m(x, zero)
    torch.testing.assert_close(y_none, y_zero, rtol=1e-5, atol=1e-6)


def test_padding_positions_do_not_affect_nonpad_queries():
    m, x, input_ids = make_model_and_inputs(batch_size=1, seq_len=6, hidden_size=16, num_attention_heads=4, d_k=4, d_v=4,
                                            device="cpu")
    pad = 0
    mask = _full_mask(input_ids, pad_token_id=pad, dtype=x.dtype, device=x.device)
    y0 = m(x, mask).detach()

    # key 側の PAD だけ巨大ノイズ → 非PADな query の出力は不変
    x2 = x.clone()
    x2[:, -2:, :] += 1e3
    y1 = m(x2, mask).detach()

    torch.testing.assert_close(y0[:, :-2, :], y1[:, :-2, :], rtol=1e-4, atol=1e-4)


def test_causal_mask_blocks_future_information():
    m, x, input_ids = make_model_and_inputs(batch_size=1, seq_len=6, hidden_size=16, num_attention_heads=4, d_k=4, d_v=4,
                                            device="cpu")
    pad = 0
    mask = _full_mask(input_ids, pad_token_id=pad, dtype=x.dtype, device=x.device)
    t = 3
    y0 = m(x, mask).detach()

    # 未来 (t+1..S-1) に巨大ノイズ → 因果で見えないので ≤t の出力は不変
    x_future = x.clone()
    x_future[:, t+1:, :] += 1e3
    y1 = m(x_future, mask).detach()

    torch.testing.assert_close(y0[:, :t+1, :], y1[:, :t+1, :], rtol=1e-4, atol=1e-4)


def test_output_projection_zero_makes_zero_output():
    m, x, input_ids = make_model_and_inputs(batch_size=2, seq_len=5, hidden_size=16, num_attention_heads=4, d_k=4, d_v=4)
    pad = 0
    mask = _full_mask(input_ids, pad_token_id=pad, dtype=x.dtype, device=x.device)

    with torch.no_grad():
        m.W_O.zero_()
        m.b_O.zero_()

    y = m(x, mask)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_dv_not_equal_dk_supported():
    m, x, input_ids = make_model_and_inputs(batch_size=2, seq_len=5, hidden_size=16, num_attention_heads=4, d_k=4, d_v=8)
    pad = 0
    mask = _full_mask(input_ids, pad_token_id=pad, dtype=x.dtype, device=x.device)
    y = m(x, mask)
    assert y.shape == (2, 5, 16)

def test_backward_grads_exist():
    m, x, input_ids = make_model_and_inputs(batch_size=2, seq_len=5, hidden_size=16, num_attention_heads=4, d_k=4, d_v=4,
                                            dtype=torch.float32)
    pad = 0
    mask = _full_mask(input_ids, pad_token_id=pad, dtype=x.dtype, device=x.device)

    x.requires_grad_(True)
    y = m(x, mask)
    loss = y.pow(2).mean()
    loss.backward()

    for name, p in m.named_parameters():
        assert p.requires_grad, f"{name} requires_grad is False"
        assert p.grad is not None, f"{name} has no grad"

    assert x.grad is not None
