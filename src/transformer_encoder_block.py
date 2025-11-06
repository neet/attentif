import torch
import torch.nn as nn
from typing import Optional

from .multi_head_attention import MultiHeadAttention
from .layer_norm import LayerNorm
from .feed_forward_network import FeedForwardNetwork
from .dropout import dropout

class TransformerEncoderBlock(nn.Module):
    # H: 隠れ次元
    # h: ヘッドの数
    # vocab_size: 語彙サイズ
    def __init__(self, hidden_size: int, num_attention_heads: int) -> None:
        super().__init__()
        assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        self.ln1 = LayerNorm(hidden_size)
        self.ln2 = LayerNorm(hidden_size)
        self.mha = MultiHeadAttention(hidden_size=hidden_size, num_attention_heads=num_attention_heads)
        self.ffn = FeedForwardNetwork(hidden_size)

    # (B, S, H) -> (B, S, H)
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_ln = self.ln1(x)
        x_mha = self.mha(x_ln, attention_mask=attention_mask)
        y = x + dropout(x_mha, training=self.training)
        y_ln = self.ln2(y)
        y_ffn = self.ffn(y_ln)

        z = y + dropout(y_ffn, training=self.training)

        return z

