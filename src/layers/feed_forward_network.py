import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or (4 * hidden_size)

        self.W1 = nn.Parameter(torch.empty(hidden_size, self.intermediate_size))
        self.b1 = nn.Parameter(torch.zeros(self.intermediate_size))

        self.W2 = nn.Parameter(torch.empty(self.intermediate_size, hidden_size))
        self.b2 = nn.Parameter(torch.zeros(hidden_size))

        self._reset_parameters()

    def _reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.W1)
        torch.nn.init.xavier_uniform_(self.W2)

    # (B, S, hidden_size) -> (B, S, hidden_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, S, hidden_size) -> (B, S, intermediate_size)
        y = x @ self.W1 + self.b1
        y = F.gelu(y)

        # (B, S, intermediate_size) -> (B, S, hidden_size)
        y = y @ self.W2 + self.b2

        return y

