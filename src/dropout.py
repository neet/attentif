import torch

def dropout(x: torch.Tensor, p: float = 0.1, training: bool = True) -> torch.Tensor:
    if not training or p == 0.0:
        return x
    mask = (torch.rand_like(x) > p).float()
    return x * mask / (1.0 - p)

