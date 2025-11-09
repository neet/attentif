import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional

from ..layers import TransformerDecoder, TokenEmbedding, positional_encoding
from .lm_head import LMHead

@dataclass
class GPT2Config():
    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    vocab_size: int
    pad_token_id: int

class GPT2Model(nn.Module):
    def __init__(self, config: GPT2Config) -> None:
        super().__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_hidden_layers = config.num_hidden_layers
        self.vocab_size = config.vocab_size
        self.pad_token_id = config.pad_token_id

        self.embedding = nn.Parameter(torch.empty(self.vocab_size, self.hidden_size))
        self.transformer_decoder = TransformerDecoder(self.hidden_size, self.num_attention_heads, self.num_hidden_layers, decoder_only=True)
        self.token_embedding = TokenEmbedding(self.embedding, self.pad_token_id)
        self.lm_head = LMHead(self.embedding)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.embedding, mean=0.0, std=0.02)

    def forward(self, batch: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if attention_mask is not None and attention_mask.dim() == 2:
            # (B, S) -> (B, 1, S) -> (B, S, S)
            attention_mask = attention_mask.unsqueeze(1).expand(-1, batch.shape[1], -1)

        # (B, S, H) + (S, H) -> (B, S, H)
        embedding = positional_encoding(batch.shape[-1], self.hidden_size, device=batch.device, dtype=torch.float32)
        embedding = embedding + self.token_embedding(batch) * (self.hidden_size ** 0.5)

        # たぶんCausal Maskを適用するならここ
        output = self.transformer_decoder(embedding, None, attention_mask)

        return self.lm_head(output)
