import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, H: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(H))
        self.beta = nn.Parameter(torch.zeros(H))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)

        return self.gamma * (x - mean) * torch.rsqrt(var + self.eps) + self.beta

