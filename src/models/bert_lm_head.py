import torch
import torch.nn as nn

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
