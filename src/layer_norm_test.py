import numpy as np
from pytest import approx

from .layer_norm import layer_norm


def test_shape_is_preserved():
    B, S, H = 3, 5, 7
    x = np.random.randn(B, S, H).astype(np.float32)
    eps = 1e-5
    gamma = np.ones(H, dtype=np.float32)
    beta = np.zeros(H, dtype=np.float32)

    y = layer_norm(x, eps, gamma, beta)
    assert y.shape == x.shape


def test_per_token_zero_mean_unit_var_when_gamma1_beta0():
    # gamma=1, beta=0 のとき、正規化後そのものが出るはず
    B, S, H = 2, 4, 8
    x = np.random.randn(B, S, H).astype(np.float32)
    eps = 1e-5
    gamma = np.ones(H, dtype=np.float32)
    beta = np.zeros(H, dtype=np.float32)

    y = layer_norm(x, eps, gamma, beta)

    # 最後の軸(H)ごとに平均≈0, 分散≈1 になっていること
    mean = y.mean(axis=-1)
    var = y.var(axis=-1)
    assert np.allclose(mean, 0.0, atol=1e-5)
    assert np.allclose(var, 1.0, atol=1e-4)


def test_gamma_beta_affine_effect():
    # 正規化した後に gamma でスケール、beta でシフトされること
    x = np.array([[[1.0, 2.0, 3.0, 4.0]]], dtype=np.float32)
    eps = 1e-5
    gamma = np.array([2.0, 0.5, 1.0, 3.0], dtype=np.float32)
    beta  = np.array([0.1, -0.2, 0.0, 1.5], dtype=np.float32)

    y = layer_norm(x, eps, gamma, beta)

    # 期待値は「z = (x-μ)/σ」を各要素に作ってから、gamma*z + beta
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    z = (x - mu) / np.sqrt(var + eps)
    expected = z * gamma + beta

    assert y.shape == expected.shape
    assert y == approx(expected, abs=1e-6)


def test_broadcast_gamma_beta_hidden_dim():
    B, S, H = 2, 3, 5
    rng = np.random.default_rng(0)  # ついでに再現性
    x = rng.standard_normal((B, S, H)).astype(np.float32)
    eps = 1e-5
    gamma = np.linspace(0.5, 1.5, H).astype(np.float32)
    beta  = np.linspace(-0.3, 0.3, H).astype(np.float32)

    y = layer_norm(x, eps, gamma, beta)

    # 逆変換して z を取り出す
    z = (y - beta) / gamma  # shape (B,S,H)

    # 期待される分散: var(x) / (var(x) + eps)
    x_var = x.var(axis=-1)  # shape (B,S)
    expected_var = x_var / (x_var + eps)

    # 計算は float64 側で評価すると誤差が安定する
    var_z = z.astype(np.float64).var(axis=-1)

    assert np.allclose(var_z, expected_var, rtol=1e-3, atol=3e-4)


def test_constant_vector_returns_beta():
    # 全要素が定数だと、正規化後はゼロ→出力は beta になる
    B, S, H = 2, 2, 6
    x = np.full((B, S, H), 5.0, dtype=np.float32)
    eps = 1e-5
    gamma = np.random.randn(H).astype(np.float32)
    beta  = np.linspace(-1.0, 1.0, H).astype(np.float32)

    y = layer_norm(x, eps, gamma, beta)

    # 各トークンの出力は beta がそのまま入っているはず
    for b in range(B):
        for s in range(S):
            assert y[b, s, :] == approx(beta, abs=1e-6)


def test_input_is_not_modified():
    # 参照透過性: 入力配列が破壊されないこと
    B, S, H = 2, 3, 4
    x = np.random.randn(B, S, H).astype(np.float32)
    x_copy = x.copy()
    eps = 1e-5
    gamma = np.ones(H, dtype=np.float32)
    beta = np.zeros(H, dtype=np.float32)

    _ = layer_norm(x, eps, gamma, beta)
    assert np.allclose(x, x_copy)

