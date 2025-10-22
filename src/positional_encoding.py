import torch

# (S,) -> (S, H)
def positional_encoding(S: int, H: int) -> torch.Tensor:
    # (S, 1)
    pos = torch.arange(S).unsqueeze(1)
    # (1, H)
    i = torch.arange(H).unsqueeze(0)
    # (S, H)
    theta = pos/10000**(2*i/H)

    positions = torch.zeros_like(theta)
    positions[:, 0::2] = torch.sin(theta[:, 0::2])
    positions[:, 1::2] = torch.cos(theta[:, 1::2])

    return positions
