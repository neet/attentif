import torch.nn as nn
from transformers import RobertaTokenizer

from .transformer_encoder import TransformerEncoder
from .token_embedding import TokenEmbedding
from .positional_encoding import positional_encoding

from .mask_padding import make_padding_mask
from .mask_causal import make_causal_mask

class MiniBert(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
    ) -> None:
        super().__init__()

        self.H = 512
        self.h = 8
        self.n_transformer_blocks = 12

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.transformer_encoder = TransformerEncoder(self.H, self.h, self.n_transformer_blocks)
        self.token_embedding = TokenEmbedding(self.vocab_size, self.H, self.pad_token_id)

    def forward(self, batch) -> None:
        padding_mask = make_padding_mask(batch, tokenizer.pad_token_id).unsqueeze(1)

        # (B, S, H) + (S, H) -> (B, S, H)
        input = self.token_embedding(batch) + positional_encoding(batch.shape[-1], self.H)

        return self.transformer_encoder(input, padding_mask)

if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = MiniBert(tokenizer.vocab_size, tokenizer.pad_token_id)

    tokens = tokenizer(["hello world", "I like an apple"], return_tensors="pt", padding=True)
    output = model(tokens["input_ids"])
    print(output)
