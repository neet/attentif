import torch

from .softmax import softmax

def test_softmax_shape_and_sum():
    """出力のshapeが入力と一致し、各行の総和が1になる"""
    x = torch.randn(4, 10)
    y = softmax(x)
    assert y.shape == x.shape
    sums = y.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

def test_softmax_translation_invariance():
    """softmax(x + c) == softmax(x) を確認"""
    x = torch.randn(2, 5)
    c = torch.randn(2, 1)  # 各行に定数を足す
    y1 = softmax(x)
    y2 = softmax(x + c)
    assert torch.allclose(y1, y2, atol=1e-6)

def test_softmax_non_negative():
    """出力がすべて非負（確率として妥当）"""
    x = torch.randn(3, 4)
    y = softmax(x)
    assert torch.all(y >= 0)

def test_softmax_numeric_stability():
    """大きな値を与えてもNaNやInfにならない"""
    x = torch.tensor([[1000.0, 1001.0, 999.0]])
    y = softmax(x)
    assert not torch.isnan(y).any()
    assert not torch.isinf(y).any()
    # 数値的にほぼ [0.2447, 0.6652, 0.0900] くらいになるはず
    assert torch.allclose(y.sum(), torch.tensor(1.0), atol=1e-6)

def test_softmax_uniform_input():
    """全要素が同じなら一様分布になる"""
    x = torch.ones(1, 6)
    y = softmax(x)
    expected = torch.full_like(x, 1/6)
    assert torch.allclose(y, expected, atol=1e-6)
