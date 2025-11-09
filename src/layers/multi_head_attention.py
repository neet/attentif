import torch
import torch.nn as nn
import math
from typing import Optional

from .softmax import softmax
from .dropout import Dropout

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, attention_probs_dropout_prob: float = 0.1) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.attention_probs_dropout_prob = attention_probs_dropout_prob

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = Dropout(self.attention_probs_dropout_prob)

    def forward(self, x_q: torch.Tensor, x_k: torch.Tensor, x_v: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x_q.shape == x_k.shape == x_v.shape, "x_q, x_k, and x_v must have an identical shape"
        batch_size, seq_len, hidden_size = x_q.shape

        # (B, S, h*d)
        query = self.q_proj(x_q)
        key = self.k_proj(x_k)
        value = self.v_proj(x_v)

        # (B, S, h, d)
        query = query.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        key = key.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)
        value = value.view(batch_size, seq_len, self.num_attention_heads, self.attention_head_size)

        # (B, h, S, d)
        query = query.permute(0, 2, 1, 3).contiguous()
        key = key.permute(0, 2, 1, 3).contiguous()
        value = value.permute(0, 2, 1, 3).contiguous()

        # (B, h, S, S)
        scores = query @ key.mT / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # 1/0 マスクを 0/-inf マスクに変換
            attention_mask = torch.where(attention_mask == 1, 0, -torch.inf)
            attention_mask = attention_mask.unsqueeze(1).to(dtype=scores.dtype, device=scores.device)
            # (B, h, S, S) + (B, 1, S, S)
            scores = scores + attention_mask

        # (B, h, S, S)
        attention_weights = softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # (B, h, S, d_v)
        output = attention_weights @ value

        # (B, S, h, d_v)
        output = output.permute(0, 2, 1, 3).contiguous()

        # (B, S, h*d_v)
        output = output.view(batch_size, seq_len, self.num_attention_heads * self.attention_head_size)

        # (B, S, h*d_v)
        output = self.o_proj(output)

        return output

