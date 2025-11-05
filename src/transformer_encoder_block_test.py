import torch

from .transformer_encoder_block import TransformerEncoderBlock

def _make_block_and_inputs(batch_size=2, seq_len=5, hidden_size=16, num_attention_heads=4, device="cpu", dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    m = TransformerEncoderBlock(hidden_size, num_attention_heads).to(device)
    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)
    return m, x

def _zero_mask(B, S, dtype, device):
    return torch.zeros(B, S, S, dtype=dtype, device=device)

def _ban_last_k_columns(B, S, k, dtype, device):
    mask = torch.zeros(B, S, S, dtype=dtype, device=device)
    mask[..., -k:] = float("-inf")  # key側の末尾k列を全面禁止
    return mask

def test_forward_shape_dtype_device_cpu():
    m, x = _make_block_and_inputs(batch_size=3, seq_len=7, hidden_size=24, num_attention_heads=6, device="cpu", dtype=torch.float32)
    mask = _zero_mask(B=3, S=7, dtype=x.dtype, device=x.device)
    y = m(x, attention_mask=mask)
    assert y.shape == (3, 7, 24)
    assert y.dtype == x.dtype and y.device == x.device

def test_attention_mask_actually_masks_keys():
    m, x = _make_block_and_inputs(batch_size=1, seq_len=6, hidden_size=16, num_attention_heads=4, device="cpu", dtype=torch.float32)
    m.eval()  # dropout無効化で決定的にする

    zero = _zero_mask(1, 6, x.dtype, x.device)
    banned = _ban_last_k_columns(1, 6, k=2, dtype=x.dtype, device=x.device)

    # 末尾2トークン（key側）だけ巨大変更
    x2 = x.clone()
    x2[:, -2:, :] += 1e3

    # ゼロマスク：影響が出る（≠）
    y_zero_0 = m(x, attention_mask=zero).detach()
    y_zero_1 = m(x2, attention_mask=zero).detach()
    assert not torch.allclose(y_zero_0, y_zero_1, atol=1e-5, rtol=1e-5)

    # 禁止マスク：非変更クエリ行（:-2）は不変（＝）
    y_ban_0 = m(x, attention_mask=banned).detach()
    y_ban_1 = m(x2, attention_mask=banned).detach()
    torch.testing.assert_close(
        y_ban_0[:, :-2, :], y_ban_1[:, :-2, :], atol=1e-5, rtol=1e-5
    )

    # 参考：末尾2行（変更クエリ行）は残差のせいで違ってよい
    assert not torch.allclose(y_ban_0[:, -2:, :], y_ban_1[:, -2:, :], atol=1e-5, rtol=1e-5)

def test_eval_disables_dropout_deterministic():
    m, x = _make_block_and_inputs(batch_size=2, seq_len=5, hidden_size=16, num_attention_heads=4, device="cpu", dtype=torch.float32, seed=123)
    mask = _zero_mask(2, 5, x.dtype, x.device)

    m.eval()
    with torch.no_grad():
        y1 = m(x, attention_mask=mask)
        y2 = m(x, attention_mask=mask)
    # eval() では dropout 無効なので同一入力で完全一致
    torch.testing.assert_close(y1, y2, atol=0.0, rtol=0.0)

def test_backward_grads_exist():
    m, x = _make_block_and_inputs(batch_size=2, seq_len=5, hidden_size=16, num_attention_heads=4, device="cpu", dtype=torch.float32)
    mask = _zero_mask(2, 5, x.dtype, x.device)

    x.requires_grad_(True)
    y = m(x, attention_mask=mask)
    loss = y.pow(2).mean()
    loss.backward()

    # 勾配が流れていること
    assert x.grad is not None
    for name, p in m.named_parameters():
        assert p.requires_grad, f"{name} requires_grad is False"
        assert p.grad is not None, f"{name} has no grad"

def test_none_mask_equals_zero_mask():
    m, x = _make_block_and_inputs(batch_size=2, seq_len=5, hidden_size=16, num_attention_heads=4, device="cpu", dtype=torch.float32)
    zero = _zero_mask(2, 5, x.dtype, x.device)

    m.eval()  # ← これが大事（dropout停止）
    with torch.no_grad():
        y_none = m(x, attention_mask=None)
        y_zero = m(x, attention_mask=zero)

    torch.testing.assert_close(y_none, y_zero, atol=1e-6, rtol=1e-6)

