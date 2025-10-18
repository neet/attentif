import torch
import pytest

from .mask_padding import make_padding_mask

DTYPES = [torch.float32, torch.float64]

@pytest.mark.parametrize("dtype", DTYPES)
def test_padding_mask_values_and_types(dtype):
    device = torch.device("cpu")
    PAD = 0
    input_ids = torch.tensor([[1, PAD, 2, 3, PAD]], dtype=torch.long, device=device)  # (1,S)
    m = make_padding_mask(input_ids, pad_token=PAD, dtype=dtype, device=device)       # (1,S)

    assert m.shape == input_ids.shape
    assert m.dtype == dtype and m.device == device

    # PADの場所が -inf、非PADが 0.0
    is_pad = (input_ids == PAD)
    assert torch.isneginf(m[is_pad]).all()
    assert torch.allclose(m[~is_pad], torch.zeros_like(m[~is_pad]))

def test_padding_mask_no_pad_all_zeros():
    device = torch.device("cpu")
    PAD = 99
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device=device)
    m = make_padding_mask(input_ids, pad_token=PAD, dtype=torch.float32, device=device)
    assert torch.allclose(m, torch.zeros_like(m))
    assert not torch.isneginf(m).any()
