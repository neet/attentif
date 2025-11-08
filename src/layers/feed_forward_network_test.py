import torch
import pytest

from .feed_forward_network import FeedForwardNetwork

@pytest.mark.parametrize("B,S,hidden_size,intermediate_size", [(2, 5, 8, 32), (1, 1, 16, 64)])
def test_ffn_output_shape(B, S, hidden_size, intermediate_size):
    x = torch.randn(B, S, hidden_size)
    ffn = FeedForwardNetwork(hidden_size=hidden_size, intermediate_size=intermediate_size).eval()
    y = ffn(x)
    assert y.shape == (B, S, hidden_size)

def test_ffn_position_wise_independence():
    B, S, hidden_size, intermediate_size = 1, 6, 12, 48
    x = torch.randn(B, S, hidden_size)
    x[:, 1, :] = torch.randn(hidden_size)          # pos=1 と pos=4 を同一ベクトルに
    x[:, 4, :] = x[:, 1, :].clone()

    ffn = FeedForwardNetwork(hidden_size=hidden_size, intermediate_size=intermediate_size).eval()
    y = ffn(x)

    # 同一入力 → 同一出力（FFNは各位置を独立に処理）
    assert torch.allclose(y[:, 1, :], y[:, 4, :], atol=1e-6)

def test_ffn_train_vs_eval_same_when_p0():
    B, S, hidden_size, intermediate_size = 2, 5, 8, 32
    x = torch.randn(B, S, hidden_size)
    ffn = FeedForwardNetwork(hidden_size=hidden_size, intermediate_size=intermediate_size)

    ffn.train()
    y_train = ffn(x)
    ffn.eval()
    y_eval = ffn(x)

    # p=0 なら train()/eval() でも出力は一致
    assert torch.allclose(y_train, y_eval, atol=0.0)

def test_ffn_backward_gradients_exist_and_finite():
    B, S, hidden_size, intermediate_size = 3, 7, 16, 64
    x = torch.randn(B, S, hidden_size)
    ffn = FeedForwardNetwork(hidden_size=hidden_size, intermediate_size=intermediate_size).train()

    y = ffn(x)
    loss = y.pow(2).mean()
    loss.backward()

    for name, p in ffn.named_parameters():
        assert p.grad is not None, f"no grad for {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
        assert p.grad.abs().sum() > 0, f"zero grad in {name}"

def test_ffn_is_deterministic_in_eval_mode_with_fixed_weights():
    # 同じ入力・同じ重み・eval → 同一出力
    B, S, hidden_size, intermediate_size = 2, 4, 10, 40
    x = torch.randn(B, S, hidden_size)

    ffn = FeedForwardNetwork(hidden_size=hidden_size, intermediate_size=intermediate_size).eval()

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

