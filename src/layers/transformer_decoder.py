import torch
import torch.nn as nn

from typing import Optional

from .transformer_decoder_block import TransformerDecoderBlock
from .layer_norm import LayerNorm

class TransformerDecoder(nn.Module):
    blocks: nn.ModuleList
    ln: LayerNorm

    def __init__(self, hidden_size: int, num_attention_heads: int, num_hidden_layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(hidden_size, num_attention_heads)
            for _ in range(0, num_hidden_layers)
        ])
        self.ln = LayerNorm(hidden_size)

    def forward(
        self,
        x_dec: torch.Tensor,
        x_enc: Optional[torch.Tensor] = None,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attetnion_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for block in self.blocks:
            x_dec = block(x_dec, x_enc, self_attention_mask, cross_attetnion_mask)

        return self.ln(x_dec)

