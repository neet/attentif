import torch
import torch.nn as nn

from .transformer_encoder import TransformerEncoder
from .token_embedding import TokenEmbedding
from .positional_encoding import positional_encoding


class LMHead(nn.Module):
    def __init__(self, E: torch.Tensor) -> None:
        super().__init__()
        self.E = E  # (V, H)
        self.H = E.shape[1]
        self.b = nn.Parameter(torch.zeros(E.shape[0]))

    # (B, S, H) -> (B, S, V)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Weight tyingを使用し、TokenEmbeddingではスケーリングなし
        return x @ self.E.mT + self.b


class MaskedLM(nn.Module):
    def __init__(
        self,
        V: int,
        pad_token_id: int,
    ) -> None:
        super().__init__()

        self.H = 512
        self.h = 8
        self.n_transformer_blocks = 12

        self.V = V
        self.pad_token_id = pad_token_id
        self.E = nn.Parameter(torch.empty(self.V, self.H))
        self.transformer_encoder = TransformerEncoder(
            self.H, self.h, self.n_transformer_blocks
        )
        self.token_embedding = TokenEmbedding(self.E, self.pad_token_id)
        self.lm_head = LMHead(self.E)

        self._reset_parameters()

    def _reset_parameters(self):
        # 埋め込み行列は通常、normal分布で初期化
        # RoBERTaではstd=0.02が使われることが多い
        nn.init.normal_(self.E, mean=0.0, std=0.02)

    def forward(
        self, batch: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # attention_maskは(B, S)の形状で渡される
        # MultiHeadAttentionは(B, S, S)を期待しているので変換が必要
        if attention_mask.dim() == 2:
            # (B, S) -> (B, 1, S) -> (B, S, S) にブロードキャスト
            # padding maskの場合、各query行が同じmaskを持つ
            attention_mask = attention_mask.unsqueeze(1).expand(-1, batch.shape[1], -1)

        # (B, S, H) + (S, H) -> (B, S, H)
        pe = positional_encoding(
            batch.shape[-1], self.H, device=batch.device, dtype=batch.dtype
        )
        embeddings = self.token_embedding(batch)
        input = embeddings + pe
        output = self.transformer_encoder(input, attention_mask)

        return self.lm_head(output)
