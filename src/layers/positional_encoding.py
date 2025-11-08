import torch

# (S,) -> (S, H)
def positional_encoding(seq_len: int, hidden_size: int, device=None, dtype=torch.float32) -> torch.Tensor:
    # (S, 1)
    pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    # (1, H)
    i = torch.arange(hidden_size, device=device, dtype=dtype).unsqueeze(0)
    # (S, H)
    theta = pos / (10000.0 ** (2 * (i // 2) / hidden_size))

    positions = torch.zeros_like(theta)
    positions[:, 0::2] = torch.sin(theta[:, 0::2])
    positions[:, 1::2] = torch.cos(theta[:, 1::2])

    return positions
