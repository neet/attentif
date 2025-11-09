import torch

from .multi_head_attention import MultiHeadAttention


def make_model_and_inputs(
    *, batch_size=2, seq_len=5, hidden_size=16, num_attention_heads=4, device="cpu", dtype=torch.float32, seed=0
):
    """
    NOTE: pad_token_id は MultiHeadAttention 側には渡さない。
    テスト内で pad_token_id=0 前提の padding mask を合成する。
    d_k と d_v は hidden_size // num_attention_heads として自動的に導出される。
    """
    torch.manual_seed(seed)

    m = MultiHeadAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads).to(device)
    m.eval()  # テスト時はdropoutを無効にする
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    # ランダム input_ids は [1, 49] で生成（0 は pad 扱いにするため除外）
    input_ids = torch.randint(1, 50, (batch_size, seq_len), device=device)
    return m, x, input_ids


def _make_padding_mask(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Create padding mask: -inf for padding positions, 0.0 for valid positions."""
    # (B, S)
    mask = (input_ids == pad_token_id).float()
    # Convert to -inf for padding, 0.0 for valid
    mask = mask.masked_fill(mask == 1.0, float('-inf'))
    return mask


def _make_causal_mask(seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create causal mask: -inf for future positions, 0.0 for current/past positions."""
    # (S, S) upper triangular
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=dtype), diagonal=1)
    mask = mask.masked_fill(mask == 1.0, float('-inf'))
    return mask


def _full_mask(
    input_ids: torch.Tensor, *, pad_token_id: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """
    Combine padding mask and causal mask.
    padding mask: (B, S) → (B, S, S) broadcasts over query dimension
    causal mask: (S, S) → broadcasts over batch dimension
    """
    batch_size, seq_len = input_ids.shape

    # Padding mask: (B, S) - applies to key positions
    pad_mask = _make_padding_mask(input_ids, pad_token_id)  # (B, S)
    pad_mask = pad_mask.unsqueeze(1).expand(batch_size, seq_len, seq_len)  # (B, S, S)

    # Causal mask: (S, S) - applies to all batches
    causal_mask = _make_causal_mask(seq_len, device, dtype)  # (S, S)

    # Combine: (B, S, S) + (S, S) → (B, S, S)
    full_mask = pad_mask + causal_mask

    return full_mask


def test_forward_shape_and_dtype_device():
    m, x, input_ids = make_model_and_inputs(batch_size=3, seq_len=7, hidden_size=24, num_attention_heads=6,
                                            device="cpu", dtype=torch.float32)
    pad = 0
    mask = _full_mask(input_ids, pad_token_id=pad, dtype=x.dtype, device=x.device)
    y = m(x, x, x, mask)
    assert y.shape == (3, 7, 24)
    assert y.dtype == x.dtype and y.device == x.device

def test_none_mask_equals_zero_mask():
    m, x, _ = make_model_and_inputs(batch_size=2, seq_len=5, hidden_size=16, num_attention_heads=4,
                                    device="cpu", dtype=torch.float32)
    B, S, _H = x.shape
    zero = torch.zeros(B, S, S, dtype=x.dtype, device=x.device)
    y_none = m(x, x, x, None)
    y_zero = m(x, x, x, zero)
    torch.testing.assert_close(y_none, y_zero, rtol=1e-5, atol=1e-6)


def test_padding_positions_do_not_affect_nonpad_queries():
    m, x, input_ids = make_model_and_inputs(batch_size=1, seq_len=6, hidden_size=16, num_attention_heads=4,
                                            device="cpu")
    pad = 0
    mask = _full_mask(input_ids, pad_token_id=pad, dtype=x.dtype, device=x.device)
    y0 = m(x, x, x, mask).detach()

    # key 側の PAD だけ巨大ノイズ → 非PADな query の出力は不変
    x2 = x.clone()
    x2[:, -2:, :] += 1e3
    y1 = m(x, x2, x2, mask).detach()

    torch.testing.assert_close(y0[:, :-2, :], y1[:, :-2, :], rtol=1e-4, atol=1e-4)


def test_causal_mask_blocks_future_information():
    m, x, input_ids = make_model_and_inputs(batch_size=1, seq_len=6, hidden_size=16, num_attention_heads=4,
                                            device="cpu")
    pad = 0
    mask = _full_mask(input_ids, pad_token_id=pad, dtype=x.dtype, device=x.device)
    t = 3
    y0 = m(x, x, x, mask).detach()

    # 未来 (t+1..S-1) に巨大ノイズ → 因果で見えないので ≤t の出力は不変
    x_future = x.clone()
    x_future[:, t+1:, :] += 1e3
    y1 = m(x_future, x_future, x_future, mask).detach()

    torch.testing.assert_close(y0[:, :t+1, :], y1[:, :t+1, :], rtol=1e-4, atol=1e-4)


def test_output_projection_zero_makes_zero_output():
    m, x, input_ids = make_model_and_inputs(batch_size=2, seq_len=5, hidden_size=16, num_attention_heads=4)
    pad = 0
    mask = _full_mask(input_ids, pad_token_id=pad, dtype=x.dtype, device=x.device)

    with torch.no_grad():
        m.o_proj.weight.zero_()
        m.o_proj.bias.zero_()

    y = m(x, x, x, mask)
    assert torch.allclose(y, torch.zeros_like(y), atol=1e-6)


def test_backward_grads_exist():
    m, x, input_ids = make_model_and_inputs(batch_size=2, seq_len=5, hidden_size=16, num_attention_heads=4,
                                            dtype=torch.float32)
    pad = 0
    mask = _full_mask(input_ids, pad_token_id=pad, dtype=x.dtype, device=x.device)

    x.requires_grad_(True)
    y = m(x, x, x, mask)
    loss = y.pow(2).mean()
    loss.backward()

    for name, p in m.named_parameters():
        assert p.requires_grad, f"{name} requires_grad is False"
        assert p.grad is not None, f"{name} has no grad"

    assert x.grad is not None
