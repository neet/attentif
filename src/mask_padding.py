import torch

# (B, S) -> (B, S)
def make_padding_mask(input_ids: torch.Tensor, pad_token_id: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = input_ids == pad_token_id

    zero = torch.tensor(0.0, dtype=dtype, device=device)
    neginf = torch.tensor(float("-inf"), dtype=dtype, device=device)

    return torch.where(mask, neginf, zero)

