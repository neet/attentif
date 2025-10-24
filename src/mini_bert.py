import torch
import torch.nn as nn
from transformers import RobertaTokenizer
from datasets import load_dataset

from .transformer_encoder import TransformerEncoder
from .token_embedding import TokenEmbedding
from .positional_encoding import positional_encoding

from .mask_padding import make_padding_mask

class LMHead(nn.Module):
    def __init__(self, H: int, V: int) -> None:
        self.W = nn.Parameter(torch.empty(H, V))
        self.b = nn.Parameter(torch.zeros(V))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, batch) -> None:
        return batch @ self.W.mT + self.b

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
        self.transformer_encoder = TransformerEncoder(self.H, self.h, self.n_transformer_blocks)
        self.token_embedding = TokenEmbedding(self.V, self.H, self.pad_token_id)
        self.lm_head = LMHead(self.H, self.V)

    def forward(self, batch) -> None:
        padding_mask = make_padding_mask(batch, self.pad_token_id).unsqueeze(1)

        # (B, S, H) + (S, H) -> (B, S, H)
        input = self.token_embedding(batch) + positional_encoding(batch.shape[-1], self.H)
        output = self.transformer_encoder(input, padding_mask)

        return self.lm_head(output)

if __name__ == "__main__":
    dataset = load_dataset("allenai", "realnewslike")
    dataset = dataset.map(lambda x: x["text"])

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokens = tokenizer(dataset, return_tensors="pt", padding=True)

    model = MaskedLM(tokenizer.vocab_size, tokenizer.pad_token_id)
    output = model(tokens["input_ids"])

