import torch
import torch.nn as nn

from typing import Optional

from .transformer_encoder_block import TransformerEncoderBlock
from .layer_norm import LayerNorm

class TransformerEncoder(nn.Module):
    blocks: nn.ModuleList
    ln: LayerNorm

    def __init__(self, hidden_size: int, num_attention_heads: int, num_hidden_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(hidden_size, num_attention_heads)
            for _ in range(0, num_hidden_layers)
        ])
        self.ln = LayerNorm(hidden_size)

    def forward(self, batch: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for block in self.blocks:
            batch = block(batch, attention_mask)

        return self.ln(batch)

