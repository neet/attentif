import numpy as np

from .softmax import softmax

def test_softmax_basic() -> None:
    xs = [1.0, 2.0, 3.0]
    ys = softmax(xs)

    # 1. 出力は確率（全部足すと1）
    assert abs(sum(ys) - 1.0) < 1e-6

    # 2. 大きい入力ほど確率も大きい
    assert ys[2] > ys[1] > ys[0]

def test_softmax_negative_values() -> None:
    xs = [-100.0, 0.0, 100.0]
    ys = softmax(xs)

    # 最大値の要素がほぼ1になる
    assert ys[2] > 0.999

def test_softmax_uniform() -> None:
    xs = [10.0, 10.0, 10.0]
    ys = softmax(xs)

    # 全部同じ確率になる
    assert all(abs(y - 1/3) < 1e-6 for y in ys)

def test_softmax_large_numbers() -> None:
    xs = [10000.0, 10001.0, 10002.0]
    ys = softmax(xs)

    # ちゃんと確率として機能する（NaNにならない）
    assert not any(map(np.isnan, ys))
    assert abs(sum(ys) - 1.0) < 1e-6

def test_softmax_numpy_input() -> None:
    xs = np.array([[1.0, 2.0, 3.0],
                   [3.0, 2.0, 1.0]])
    ys = softmax(xs)

    # 入力と同じ形状で返る
    assert ys.shape == xs.shape

    # 各行が確率として正規化されている
    assert np.allclose(np.sum(ys, axis=1), np.ones(xs.shape[0]))
