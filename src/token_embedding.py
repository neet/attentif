import torch
import torch.nn as nn
import math

from typing import Optional


class TokenEmbedding(nn.Module):
    def __init__(self, E: torch.Tensor, pad_token_id: Optional[int] = None):
        super().__init__()
        self.E = E  # (V, H)
        self.H = E.shape[1]
        self.pad_token_id = pad_token_id

    # (B, S) -> (B, S, H)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # sqrt(H)スケーリングを削除（weight tyingとの相互作用を避けるため）
        y = self.E[x]

        if self.pad_token_id is not None:
            y = y.masked_fill((x == self.pad_token_id).unsqueeze(-1), 0.0)

        return y
