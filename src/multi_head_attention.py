import torch
import torch.nn as nn
import math

# S -> (S, S)
def make_causal_mask(seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    x = torch.arange(seq_len)
    y = torch.arange(seq_len)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

    zero = torch.tensor(0.0, dtype=dtype, device=device)
    neginf = torch.tensor(float("-inf"), dtype=dtype, device=device)

    return torch.where(grid_x >= grid_y, zero, neginf)

# (B, S) -> (B, S)
def make_padding_mask(input_ids: torch.Tensor, pad_token: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    mask = input_ids == pad_token

    zero = torch.tensor(0.0, dtype=dtype, device=device)
    neginf = torch.tensor(float("-inf"), dtype=dtype, device=device)

    return torch.where(mask, neginf, zero)

class MultiHeadAttention(nn.Module):
    pad_token: int

    # B: バッチサイズ
    # S: トークンの最大長
    # h: ヘッドの個数
    # d_k: 1つのヘッドが持つ隠れ状態の次元
    # H: モデル全体の隠れ状態の次元 (H = h*d_k)
    def __init__(self, h: int, d_k: int, d_v: int, pad_token: int) -> None:
        super().__init__()

        self.h = h
        self.d_k = d_k
        self.d_v = d_v
        self.pad_token = pad_token
        self.H = h * d_k

        self.W_Q = nn.Parameter(torch.empty(self.H, h * d_k))
        self.W_K = nn.Parameter(torch.empty(self.H, h * d_k))
        self.W_V = nn.Parameter(torch.empty(self.H, h * d_v))

        self.b_Q = nn.Parameter(torch.zeros(h * d_k))
        self.b_K = nn.Parameter(torch.zeros(h * d_k))
        self.b_V = nn.Parameter(torch.zeros(h * d_v))

        self.W_O = nn.Parameter(torch.empty(h * d_v, self.H))
        self.b_O = nn.Parameter(torch.empty(self.H))

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier初期化など好きな方式でOK
        for param in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_uniform_(param)

    # (B, S, H) -> (B, S, H)
    def forward(self, batch: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        outputs: list[torch.Tensor] = []

        # (S, S) -> (B, S, S)
        causal_mask = make_causal_mask(batch.shape[-2], dtype=batch.dtype, device=batch.device)[None, :, :]
        # (B, S) -> (B, S, S)
        padding_mask = make_padding_mask(input_ids, self.pad_token, dtype=batch.dtype, device=batch.device)[:, None, :]
        # (B, S, S)
        mask = causal_mask + padding_mask

        for i in range(0, self.h):
            # i番目のheadにかける重みをスライス
            W_Q = self.W_Q[:, i*self.d_k : (i+1)*self.d_k]
            W_K = self.W_K[:, i*self.d_k : (i+1)*self.d_k]
            W_V = self.W_V[:, i*self.d_v : (i+1)*self.d_v]

            b_Q = self.b_Q[i*self.d_k : (i+1)*self.d_k]
            b_K = self.b_K[i*self.d_k : (i+1)*self.d_k]
            b_V = self.b_V[i*self.d_v : (i+1)*self.d_v]

            # (B, S, d_k)
            Q = batch @ W_Q + b_Q
            # (B, S, d_k)
            K = batch @ W_K + b_K
            # (B, S, d_v)
            V = batch @ W_V + b_V

            # (B, S, S)
            scores = Q @ K.mT / math.sqrt(self.d_k)
            # (B, S, S)
            attention_weights = torch.softmax(scores + mask, dim=-1)
            # (B, S, d_v)
            output = attention_weights @ V

            outputs.append(output)

        # (B, S, H)
        output = torch.cat(outputs, dim=-1)

        # (B, S, H)
        return output @ self.W_O + self.b_O

