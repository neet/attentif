import torch
import math

# S -> (S, S)
def make_causal_mask(seq_len: int) -> torch.Tensor:
    x = torch.arange(seq_len)
    y = torch.arange(seq_len)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    return torch.where(grid_x >= grid_y, torch.tensor(0.0), float("-inf"))

# (B, S) -> (B, S)
def make_padding_mask(input_ids: torch.Tensor, pad_token: int) -> torch.Tensor:
    mask = input_ids == pad_token
    return torch.where(mask, float("-inf"), torch.tensor(0.0))

class SingleHeadAttention:
    input_ids: torch.Tensor # (B, S)
    pad_token: int

    W_Q: torch.Tensor # (H, d_k)
    b_Q: torch.Tensor # (d_k, )

    W_K: torch.Tensor # (H, d_k)
    b_K: torch.Tensor # (d_k, )

    W_V: torch.Tensor # (H, d_v)
    b_V: torch.Tensor # (d_v, )

    d_k: int

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        # (B, S, d_k)
        Q = batch @ self.W_Q + self.b_Q
        # (B, S, d_k)
        K = batch @ self.W_K + self.b_K
        # (B, S, d_v)
        V = batch @ self.W_V + self.b_V

        # (B, S, S)
        scores = Q @ K.mT / math.sqrt(self.d_k)

        # (B, S, S)
        mask = (
            # (S, S) -> (B, S, S)
            make_causal_mask(scores.shape[-1])[None, :, :] +
            # (B, S) -> (B, S, S)
            make_padding_mask(self.input_ids, self.pad_token)[:, None, :]
        )

        # (B, S, S)
        attention_weights = torch.softmax(scores + mask, dim=-1)

        # (B, S, d_v)
        output = attention_weights @ V

        return output


