import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from .transformer_encoder import TransformerEncoder
from .token_embedding import TokenEmbedding
from .positional_encoding import positional_encoding


class LMHead(nn.Module):
    def __init__(self, embedding: torch.Tensor) -> None:
        super().__init__()
        vocab_size = embedding.shape[0]
        self.embedding = embedding  # (V, H)
        self.bias = nn.Parameter(torch.zeros(vocab_size))

    # (B, S, H) -> (B, S, V)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Weight tyingを使用。スケーリングはTokenEmbeddingのみで行う
        return x @ self.embedding.mT + self.bias

@dataclass
class MaskedLMConfig():
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    vocab_size: int
    pad_token_id: int

class MaskedLM(nn.Module):
    def __init__(self, config: MaskedLMConfig) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id

        self.embedding = nn.Parameter(torch.empty(self.vocab_size, self.hidden_size))
        self.transformer_encoder = TransformerEncoder(self.hidden_size, self.num_attention_heads, self.num_hidden_layers)
        self.token_embedding = TokenEmbedding(self.embedding, self.pad_token_id)
        self.lm_head = LMHead(self.embedding)

        self._reset_parameters()

    def _reset_parameters(self):
        # 埋め込み行列は通常、normal分布で初期化
        # RoBERTaではstd=0.02が使われることが多い
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, batch: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # attention_maskは(B, S)の形状で渡される
        # MultiHeadAttentionは(B, S, S)を期待しているので変換が必要
        if attention_mask is not None and attention_mask.dim() == 2:
            # (B, S) -> (B, 1, S) -> (B, S, S) にブロードキャスト
            # padding maskの場合、各query行が同じmaskを持つ
            attention_mask = attention_mask.unsqueeze(1).expand(-1, batch.shape[1], -1)

        # (B, S, H) + (S, H) -> (B, S, H)
        embedding = positional_encoding(batch.shape[-1], self.hidden_size, device=batch.device, dtype=torch.float32)
        embedding = embedding + self.token_embedding(batch) * (self.hidden_size ** 0.5)

        output = self.transformer_encoder(embedding, attention_mask)

        return self.lm_head(output)
