import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

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
        input = self.token_embedding(batch) + positional_encoding(batch.shape[-1], self.H)
        output = self.transformer_encoder(input, attention_mask)

        return self.lm_head(output)

def train() -> None:
    dataset = load_dataset("allenai/c4", "realnewslike", split="train[:1000]")
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
    model.train()

    collator = DataCollatorForLanguageModeling(tokenizer)
    batches = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collator)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    running_loss = 0.0
    for (step, batch) in enumerate(batches):
        logits = model(batch["input_ids"], batch["attention_mask"])
        loss = criterion(logits.view(-1, logits.size(-1)), batch["labels"].view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        print(f"step {step}, loss {running_loss / (step+1):.4f}")

if __name__ == "__main__":
    train()
