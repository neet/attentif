import torch.nn as nn
from transformers import RobertaTokenizer

from .transformer_encoder import TransformerEncoder
from .token_embedding import TokenEmbedding
from .positional_encoding import positional_encoding

# padding_mask = make_padding_mask(batch, self.pad_token_id, dtype=batch.dtype, device=batch.device)
# causal_mask = make_causal_mask(batch.shape[-1], dtype=batch.dtype, device=batch.device)
# attention_mask = padding_mask + causal_mask

class MiniBert(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
    ) -> None:
        super().__init__()

        self.H = 512
        self.h = 16
        self.n_transformer_blocks = 10

        self.vocab_size = vocab_size
        self.pad_token_id = pad_token_id
        self.transformer_encoder = TransformerEncoder(self.H, self.h, self.n_transformer_blocks)
        self.token_embedding = TokenEmbedding(self.vocab_size, self.H, self.pad_token_id)

    def forward(self, batch) -> None:
        # (B, S, H) + (S, H)
        input = self.token_embedding(batch) + positional_encoding(batch.shape[-1], self.H)
        return self.transformer_encoder(input)

if __name__ == "__main__":
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    model = MiniBert(tokenizer.vocab_size, tokenizer.pad_token_id)

    tokens = tokenizer(["hello world"], return_tensors="pt")
    output = model(tokens["input_ids"])

    # print(output)
