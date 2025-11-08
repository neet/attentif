import torch
import pytest

from .relu import relu

DTYPES = [torch.float32, torch.float64]

@pytest.mark.parametrize("dtype", DTYPES)
def test_relu_basic(dtype):
    x = torch.tensor([-2.0, -0.0, 0.0, 0.5, 3.14], dtype=dtype)
    y = relu(x)
    expected = torch.tensor([0.0, 0.0, 0.0, 0.5, 3.14], dtype=dtype)
    assert torch.allclose(y, expected)
    assert y.dtype == dtype and y.device == x.device

def test_relu_large_tensor_nonneg():
    x = torch.randn(4096) - 0.2
    y = relu(x)
    assert torch.all(y >= 0)
    # 正の部分はそのまま通す
    pos = x > 0
    assert torch.allclose(y[pos], x[pos])
