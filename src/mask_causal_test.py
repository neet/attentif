import torch
import math
import pytest

from .mask_causal import make_causal_mask

DTYPES = [torch.float32, torch.float64]

@pytest.mark.parametrize("dtype", DTYPES)
def test_causal_mask_triangle(dtype):
    device = torch.device("cpu")
    S = 4
    m = make_causal_mask(S, dtype=dtype, device=device)  # (S,S)

    assert m.shape == (S, S)
    assert m.dtype == dtype and m.device == device

    # 下三角(含対角)が0、上三角が -inf
    for i in range(S):
        for j in range(S):
            if i >= j:
                assert m[i, j].item() == pytest.approx(0.0)
            else:
                assert math.isinf(m[i, j].item()) and m[i, j].item() < 0

def test_causal_mask_sanity_for_seq_len_1():
    m = make_causal_mask(1, dtype=torch.float32, device=torch.device("cpu"))
    assert m.shape == (1, 1)
    assert m.item() == pytest.approx(0.0)
