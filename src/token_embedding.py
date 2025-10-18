import torch
import torch.nn as nn

from typing import Optional

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, H: int, pad_token: Optional[int] = None):
        super().__init__()

        self.H = H
        self.E = nn.Parameter(torch.empty(vocab_size, H))
        self.pad_token = pad_token

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.E, mean=0.0, std=self.H ** -0.5)

    # (B, S) -> (B, S, H)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y =  self.E[x] * (self.H ** 0.5)

        if self.pad_token is not None:
            y = y.masked_fill((x == self.pad_token).unsqueeze(-1), 0.0)

        return y

