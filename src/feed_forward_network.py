import torch
import torch.nn as nn
from typing import Optional

from .relu import relu

class FeedForwardNetwork(nn.Module):
    def __init__(self, H: int, d_ff: Optional[int] = None, p_dropout: float = 0.1):
        super().__init__()

        self.H = H
        self.d_ff = d_ff or (4 * H)
        self.p_dropout = p_dropout

        self.W1 = nn.Parameter(torch.empty(H, self.d_ff))
        self.b1 = nn.Parameter(torch.zeros(self.d_ff))

        self.W2 = nn.Parameter(torch.empty(self.d_ff, H))
        self.b2 = nn.Parameter(torch.zeros(H))

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W1)
        torch.nn.init.xavier_uniform_(self.W2)

    # (B, S, H) -> (B, S, H)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, S, H) -> (B, S, d_ff)
        y = x @ self.W1 + self.b1
        y = relu(y)

        # (B, S, d_ff) -> (B, S, H)
        y = y @ self.W2 + self.b2

        return y

