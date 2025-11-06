import torch
import torch.nn as nn

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


class MaskedLM(nn.Module):
    def __init__(self, vocab_size: int, pad_token_id: int) -> None:
        super().__init__()

        self.hidden_size = 512
        self.num_attention_heads = 8
        self.num_hidden_layers = 12

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.embedding = nn.Parameter(torch.empty(self.vocab_size, self.hidden_size))
        self.transformer_encoder = TransformerEncoder(self.hidden_size, self.num_attention_heads, self.num_hidden_layers)
        self.token_embedding = TokenEmbedding(self.embedding, self.pad_token_id)
        self.lm_head = LMHead(self.embedding)

        self._reset_parameters()

    def _reset_parameters(self):
        # 埋め込み行列は通常、normal分布で初期化
        # RoBERTaではstd=0.02が使われることが多い
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, batch: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # attention_maskは(B, S)の形状で渡される
        # MultiHeadAttentionは(B, S, S)を期待しているので変換が必要
        if attention_mask.dim() == 2:
            # (B, S) -> (B, 1, S) -> (B, S, S) にブロードキャスト
            # padding maskの場合、各query行が同じmaskを持つ
            attention_mask = attention_mask.unsqueeze(1).expand(-1, batch.shape[1], -1)

        # (B, S, H) + (S, H) -> (B, S, H)
        pe = positional_encoding(batch.shape[-1], self.hidden_size, device=batch.device, dtype=torch.float32)
        embeddings = self.token_embedding(batch)

        # PEをembeddingと同じスケールに（std=0.02 → 約0.1の範囲）
        input = embeddings + pe * 0.1
        output = self.transformer_encoder(input, attention_mask)

        return self.lm_head(output)
