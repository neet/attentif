import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

from .transformer_encoder import TransformerEncoder
from .token_embedding import TokenEmbedding
from .positional_encoding import positional_encoding

class LMHead(nn.Module):
    def __init__(self, H: int, V: int) -> None:
        super().__init__()
        self.W = nn.Parameter(torch.empty(H, V))
        self.b = nn.Parameter(torch.zeros(V))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch @ self.W + self.b

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

    def forward(self, batch: torch.Tensor, attention_mask: torch.Tensor) -> None:
        attention_mask = attention_mask.unsqueeze(1)

        # (B, S, H) + (S, H) -> (B, S, H)
        input = self.token_embedding(batch) + positional_encoding(batch.shape[-1], self.H)
        output = self.transformer_encoder(input, attention_mask)

        return self.lm_head(output)

def train() -> None:
    dataset = load_dataset("allenai/c4", "realnewslike", split="train")
    dataset = dataset.select(range(1000))
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    dataset = dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            max_length=512,
        ),
        remove_columns=dataset.column_names,
        batched=True,
    )
    model = MaskedLM(tokenizer.vocab_size, tokenizer.pad_token_id)

    collator = DataCollatorForLanguageModeling(tokenizer)
    batches = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collator)

    output: torch.Tensor

    for batch in tqdm(batches, desc="バッチ処理を実行中"):
        output = model(batch["input_ids"], batch["attention_mask"])

    print(output)

if __name__ == "__main__":
    train()
