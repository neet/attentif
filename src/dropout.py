import torch

def dropout(x: torch.Tensor, prob: float = 0.1, training: bool = True) -> torch.Tensor:
    if not training or prob == 0.0:
        return x
    mask = (torch.rand_like(x) > prob).float()
    return x * mask / (1.0 - prob)

