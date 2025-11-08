import torch

def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max, _ = x.max(dim=dim, keepdim=True)
    e_x = torch.exp(x - x_max)
    avg = e_x.sum(dim=dim, keepdim=True)
    return e_x / avg

