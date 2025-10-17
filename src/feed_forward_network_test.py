import torch
import pytest

from .feed_forward_network import FeedForwardNetwork

@pytest.mark.parametrize("B,S,H,d_ff", [(2, 5, 8, 32), (1, 1, 16, 64)])
def test_ffn_output_shape(B, S, H, d_ff):
    x = torch.randn(B, S, H)
    ffn = FeedForwardNetwork(H=H, d_ff=d_ff, p_dropout=0.0).eval()
    y = ffn(x)
    assert y.shape == (B, S, H)

def test_ffn_position_wise_independence():
    B, S, H, d_ff = 1, 6, 12, 48
    x = torch.randn(B, S, H)
    x[:, 1, :] = torch.randn(H)          # pos=1 と pos=4 を同一ベクトルに
    x[:, 4, :] = x[:, 1, :].clone()

    ffn = FeedForwardNetwork(H=H, d_ff=d_ff, p_dropout=0.0).eval()
    y = ffn(x)

    # 同一入力 → 同一出力（FFNは各位置を独立に処理）
    assert torch.allclose(y[:, 1, :], y[:, 4, :], atol=1e-6)

def test_ffn_dropout_training_changes_output():
    B, S, H, d_ff = 2, 5, 8, 32
    x = torch.randn(B, S, H)
    ffn = FeedForwardNetwork(H=H, d_ff=d_ff, p_dropout=0.5).train()

    torch.manual_seed(0)
    y1 = ffn(x)
    torch.manual_seed(1)
    y2 = ffn(x)

    # 異なる乱数 → 異なるdropout mask → 出力は（ほぼ確実に）異なる
    assert not torch.allclose(y1, y2)

def test_ffn_dropout_eval_is_disabled():
    B, S, H, d_ff = 2, 5, 8, 32
    x = torch.randn(B, S, H)
    ffn = FeedForwardNetwork(H=H, d_ff=d_ff, p_dropout=0.7).eval()

    # eval() ではdropoutが無効 → 乱数に依らず同一出力
    torch.manual_seed(0)
    y1 = ffn(x)
    torch.manual_seed(1)
    y2 = ffn(x)
    assert torch.allclose(y1, y2, atol=0.0)

def test_ffn_train_vs_eval_same_when_p0():
    B, S, H, d_ff = 2, 5, 8, 32
    x = torch.randn(B, S, H)
    ffn = FeedForwardNetwork(H=H, d_ff=d_ff, p_dropout=0.0)

    ffn.train()
    y_train = ffn(x)
    ffn.eval()
    y_eval = ffn(x)

    # p=0 なら train()/eval() でも出力は一致
    assert torch.allclose(y_train, y_eval, atol=0.0)

def test_ffn_backward_gradients_exist_and_finite():
    B, S, H, d_ff = 3, 7, 16, 64
    x = torch.randn(B, S, H)
    ffn = FeedForwardNetwork(H=H, d_ff=d_ff, p_dropout=0.0).train()

    y = ffn(x)
    loss = y.pow(2).mean()
    loss.backward()

    for name, p in ffn.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
        assert p.grad.abs().sum() > 0, f"zero grad in {name}"

def test_ffn_is_deterministic_in_eval_mode_with_fixed_weights():
    # 同じ入力・同じ重み・eval → 同一出力
    B, S, H, d_ff = 2, 4, 10, 40
    x = torch.randn(B, S, H)

    ffn = FeedForwardNetwork(H=H, d_ff=d_ff, p_dropout=0.5).eval()

    # 重みを固定してから2回実行
    with torch.no_grad():
        torch.manual_seed(42)
        ffn.W1.copy_(torch.randn_like(ffn.W1))
        ffn.W2.copy_(torch.randn_like(ffn.W2))
        ffn.b1.zero_()
        ffn.b2.zero_()

    y1 = ffn(x)
    y2 = ffn(x)
    assert torch.allclose(y1, y2, atol=0.0)

