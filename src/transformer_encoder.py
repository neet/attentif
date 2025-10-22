import torch
import torch.nn as nn

from typing import Optional

from .transformer_encoder_block import TransformerEncoderBlock
from .layer_norm import LayerNorm

class TransformerEncoder(nn.Module):
    blocks: nn.ModuleList
    ln: LayerNorm

    def __init__(self, H: int, h: int, n_transformer_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(H, h)
            for _ in range(0, n_transformer_blocks)
        ])
        self.ln = LayerNorm(H)

    def forward(self, batch: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        out = batch

        for (index, block) in enumerate(self.blocks):
            out = block(out, attention_mask)
            print(f"#{index}", out)

        return self.ln(out)

