import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: Optional[int] = None):
        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size or (4 * hidden_size)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)

    # (B, S, H) -> (B, S, H)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, S, H) -> (B, S, H)
        y = self.up_proj(x)
        y = F.gelu(y)

        # (B, S, H) -> (B, S, H)
        y = self.down_proj(y)

        return y

