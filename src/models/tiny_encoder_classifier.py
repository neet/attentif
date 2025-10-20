import torch.nn as nn

from ..transformer_encoder import TransformerEncoder
from ..token_embedding import TokenEmbedding
from ..positional_encoding import positional_encoding

class TinyEncoderClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.H = 512
        self.h = 512
        self.vocab_size = 30_000
        self.pad_token = 0
        self.n_transformer_blocks = 10

        self.transformer_block = TransformerEncoder(self.H, self.h, self.n_transformer_blocks)
        self.token_embedding = TokenEmbedding(self.vocab_size, self.H, self.pad_token)

    def forward(self, batch) -> None:
        # あとで考える
        attention_mask = None

        input = self.token_embedding(batch, attention_mask) + positional_encoding(batch.shape[-1], self.h)
        output = self.transformer_block(input, attention_mask)

        return output

