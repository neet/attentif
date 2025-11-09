import torch
import torch.nn as nn

class Dropout(nn.Module):
    prob: float

    def __init__(self, prob: float = 0.1):
        super().__init__()
        self.prob = prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.prob == 0.0:
            return x

        mask = (torch.rand_like(x) > self.prob).float()

        return x * mask / (1.0 - self.prob)

