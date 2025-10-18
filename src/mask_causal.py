import torch

# S -> (S, S)
def make_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    x = torch.arange(seq_len)
    y = torch.arange(seq_len)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

    zero = torch.tensor(0.0, dtype=dtype, device=device)
    neginf = torch.tensor(float("-inf"), dtype=dtype, device=device)

    return torch.where(grid_x >= grid_y, zero, neginf)

