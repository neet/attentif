import torch
import torch.nn as nn

from typing import Optional

class TokenEmbedding(nn.Module):
    def __init__(self, embedding: torch.Tensor, pad_token_id: Optional[int] = None):
        super().__init__()
        self.embedding = embedding  # (V, H)
        self.hidden_size = embedding.shape[1]
        self.pad_token_id = pad_token_id

    # (B, S) -> (B, S, H)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.embedding[x]

        if self.pad_token_id is not None:
            y = y.masked_fill((x == self.pad_token_id).unsqueeze(-1), 0.0)

        return y
