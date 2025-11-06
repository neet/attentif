import torch
import torch.nn as nn
import math
from typing import Optional

from .softmax import softmax

# B: バッチサイズ
# S: トークンの最大長
# h: ヘッドの個数
# d_*: 1つのヘッドが持つ隠れ状態の次元
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.W_Q = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_K = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.W_V = nn.Parameter(torch.empty(hidden_size, hidden_size))

        self.b_Q = nn.Parameter(torch.zeros(hidden_size))
        self.b_K = nn.Parameter(torch.zeros(hidden_size))
        self.b_V = nn.Parameter(torch.zeros(hidden_size))

        self.W_O = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.b_O = nn.Parameter(torch.zeros(hidden_size))

        self._reset_parameters()

    def _reset_parameters(self):
        for param in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(param)

    # f: (B, S, h*d_k) -> (B, S, S) -> (B, S, h*d_v)
    def forward(self, batch: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, hidden_size = batch.shape
        assert hidden_size == self.hidden_size

        # (B, S, h*d)
        Q = batch @ self.W_Q + self.b_Q
        K = batch @ self.W_K + self.b_K
        V = batch @ self.W_V + self.b_V

        # (B, S, h, d)
        Q = Q.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        K = K.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        V = V.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)

        # (B, h, S, d)
        Q = Q.permute(0, 2, 1, 3).contiguous()
        K = K.permute(0, 2, 1, 3).contiguous()
        V = V.permute(0, 2, 1, 3).contiguous()

        # (B, h, S, S)
        scores = Q @ K.mT / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # (B, h, S, S) + (B, 1, S, S)
            scores = scores + attention_mask.unsqueeze(1).to(dtype=scores.dtype, device=scores.device)

        # (B, h, S, S)
        attention_weights = softmax(scores, dim=-1)

        # (B, h, S, d_v)
        output = attention_weights @ V

        # (B, S, h, d_v)
        output = output.permute(0, 2, 1, 3).contiguous()

        # (B, S, h*d_v)
        output = output.view(batch_size, seq_len, self.num_attention_heads * self.attention_head_size)

        # (B, S, h*d_v)
        output = output @ self.W_O + self.b_O

        return output

