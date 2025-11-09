import torch
import torch.nn as nn
from typing import Optional

from .multi_head_attention import MultiHeadAttention
from .layer_norm import LayerNorm
from .feed_forward_network import FeedForwardNetwork
from .dropout import Dropout

class TransformerDecoderBlock(nn.Module):
    self_attn: MultiHeadAttention
    cross_attn: MultiHeadAttention
    ffn: FeedForwardNetwork
    ln1: LayerNorm
    ln3: LayerNorm
    ln2: LayerNorm
    dropout: Dropout

    def __init__(self, hidden_size: int, num_attention_heads: int, decoder_only: bool = False) -> None:
        super().__init__()
        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        self.ln1 = LayerNorm(hidden_size)
        self.self_attn = MultiHeadAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads)

        if not decoder_only:
            self.ln2 = LayerNorm(hidden_size)
            self.cross_attn = MultiHeadAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads)

        self.ln3 = LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size)
        self.dropout = Dropout()


    def forward(
        self,
        x_dec: torch.Tensor,
        x_enc: Optional[torch.Tensor] = None,
        self_attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # x = x + dropout(attntion(ln(x_dec), ln(x_dec), ln(x_dec)))
        x_dec_self_attn = self.ln1(x_dec)
        x_dec_self_attn = self.self_attn(x_dec_self_attn, x_dec_self_attn, x_dec_self_attn, attention_mask=self_attention_mask)
        x_dec_self_attn = x_dec + self.dropout(x_dec_self_attn)

        if x_enc is not None:
            # x = x + dropout(attntion(ln(x_dec), x_enc, x_enc))
            x_dec_cross_attn = self.ln2(x_dec_self_attn)
            x_dec_cross_attn = self.cross_attn(x_dec_cross_attn, x_enc, x_enc, attention_mask=cross_attention_mask)
            x_dec_cross_attn = x_dec_self_attn + self.dropout(x_dec_cross_attn)
        else:
            x_dec_cross_attn = x_dec_self_attn

        # z = y + dropout(ffn(ln(y), ln(y), ln(y)))
        x_dec_ffn = self.ln3(x_dec_cross_attn)
        x_dec_ffn = self.ffn(x_dec_ffn)
        x_dec_ffn = x_dec_cross_attn + self.dropout(x_dec_ffn)

        return x_dec_ffn

