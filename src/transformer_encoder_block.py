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
    def __init__(self, H: int, h: int) -> None:
        super().__init__()
        assert H % h == 0, "H must be divisible by h"
        d = H // h
        self.ln1 = LayerNorm(H=H)
        self.ln2 = LayerNorm(H=H)
        self.mha = MultiHeadAttention(h=h, d_k=d, d_v=d)
        self.ffn = FeedForwardNetwork(H=H)

    # (B, S, H) -> (B, S, H)
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = x + dropout(self.mha(self.ln1(x), attention_mask=attention_mask), training=self.training)
        z = y + dropout(self.ffn(self.ln2(y)), training=self.training)
        return z

