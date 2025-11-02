import torch
import torch.nn as nn

from .transformer_encoder import TransformerEncoder
from .token_embedding import TokenEmbedding
from .positional_encoding import positional_encoding

class LMHead(nn.Module):
    def __init__(self, E: torch.Tensor) -> None:
        super().__init__()
        self.E = E # (V, H)
        self.b = nn.Parameter(torch.zeros(E.shape[0]))

    # (B, S, H) -> (B, S)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        self.transformer_encoder = TransformerEncoder(self.H, self.h, self.n_transformer_blocks)
        self.token_embedding = TokenEmbedding(self.E, self.pad_token_id)
        self.lm_head = LMHead(self.E)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.E)

    def forward(self, batch: torch.Tensor, attention_mask: torch.Tensor) -> None:
        attention_mask = attention_mask.unsqueeze(1)

        # (B, S, H) + (S, H) -> (B, S, H)
        pe = positional_encoding(batch.shape[-1], self.H, device=batch.device, dtype=batch.dtype)
        embeddings = self.token_embedding(batch)
        input = embeddings + pe
        output = self.transformer_encoder(input, attention_mask)

        return self.lm_head(output)

