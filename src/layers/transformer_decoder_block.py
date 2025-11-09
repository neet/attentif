import torch
import torch.nn as nn
from typing import Optional

from .multi_head_attention import MultiHeadAttention
from .layer_norm import LayerNorm
from .feed_forward_network import FeedForwardNetwork
from .dropout import dropout

class TransformerDecoderBlock(nn.Module):
    self_attn: MultiHeadAttention
    cross_attn: MultiHeadAttention
    ffn: FeedForwardNetwork

    ln1: LayerNorm
    ln2: LayerNorm
    ln3: LayerNorm

    def __init__(self, hidden_size: int, num_attention_heads: int) -> None:
        super().__init__()
        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        self.ln1 = LayerNorm(hidden_size)
        self.ln2 = LayerNorm(hidden_size)
        self.ln3 = LayerNorm(hidden_size)
        self.self_attn = MultiHeadAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads)
        self.cross_attn = MultiHeadAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads)
        self.ffn = FeedForwardNetwork(hidden_size)

    # TODO: Apply causal mask to attention_mask
    def forward(self, x_dec: torch.Tensor, x_enc: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x = x + dropout(attntion(ln(x_dec), ln(x_dec), ln(x_dec)))
        x_dec_self_attn = self.ln1(x_dec)
        x_dec_self_attn = self.self_attn(x_dec_self_attn, x_dec_self_attn, x_dec_self_attn, attention_mask=attention_mask)
        x_dec_self_attn = x_dec + dropout(x_dec_self_attn, training=self.training)

        # x = x + dropout(attntion(ln(x_dec), x_enc, x_enc))
        x_dec_cross_attn = self.ln2(x_dec_self_attn)
        x_dec_cross_attn = self.cross_attn(x_dec_cross_attn, x_enc, x_enc, attention_mask=attention_mask)
        x_dec_cross_attn = x_dec_self_attn + dropout(x_dec_cross_attn, training=self.training)

        # z = y + dropout(ffn(ln(y), ln(y), ln(y)))
        x_dec_ffn = self.ln3(x_dec_cross_attn)
        x_dec_ffn = self.ffn(x_dec_ffn)
        x_dec_ffn = x_dec_cross_attn + dropout(x_dec_ffn, training=self.training)

        return x_dec_ffn

