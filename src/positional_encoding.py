import torch

# (S,) -> (S, H)
def positional_encoding(s: int, h: int = 512) -> torch.Tensor:
    # (S, 1)
    pos = torch.arange(s).unsqueeze(1)
    # (1, H)
    i = torch.arange(h).unsqueeze(0)
    # (S, H)
    theta = pos/10000**(2*i/h)

    positions = torch.zeros_like(theta)
    positions[:, 0::2] = torch.sin(theta[:, 0::2])
    positions[:, 1::2] = torch.cos(theta[:, 1::2])

    return positions
