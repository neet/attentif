import torch

# (S,) -> (S, H)
def positional_encoding(S: int, H: int, device=None, dtype=torch.float32) -> torch.Tensor:
    # (S, 1)
    pos = torch.arange(S, device=device, dtype=dtype).unsqueeze(1)
    # (1, H)
    i = torch.arange(H, device=device, dtype=dtype).unsqueeze(0)
    # (S, H)
    theta = pos / (10000.0 ** (2 * (i // 2) / H))

    positions = torch.zeros_like(theta)
    positions[:, 0::2] = torch.sin(theta[:, 0::2])
    positions[:, 1::2] = torch.cos(theta[:, 1::2])

    return positions
