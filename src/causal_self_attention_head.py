import numpy as np

from .softmax import softmax

# S -> (S, S)
def make_causal_mask(seq_len: int) -> np.ndarray:
    i, j = np.ogrid[:seq_len, :seq_len]
    return np.where(i >= j, 0, -np.inf)

# (B, S) -> (B, S)
def make_padding_mask(input_ids: np.ndarray, pad_token: int) -> np.ndarray:
    mask = input_ids == pad_token
    return np.where(mask, -np.inf, 0)

class CausalSelfAttentionHead:
    input_ids: np.ndarray # (B, S)
    pad_token: int

    W_Q: np.ndarray # (H, d_k)
    b_Q: np.ndarray # (d_k, )

    W_K: np.ndarray # (H, d_k)
    b_K: np.ndarray # (d_k, )

    W_V: np.ndarray # (H, d_v)
    b_V: np.ndarray # (d_v, )

    d_k: int

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        # (B, S, d_k)
        Q = batch @ self.W_Q + self.b_Q
        # (B, S, d_k)
        K = batch @ self.W_K + self.b_K
        # (B, S, d_v)
        V = batch @ self.W_V + self.b_V

        # (B, S, S)
        scores = Q @ K.transpose(0, 2, 1) / np.sqrt(self.d_k)

        # (B, S, S)
        mask = (
            # (S, S) -> (B, S, S)
            make_causal_mask(scores.shape[-1])[None, :, :] +
            # (B, S) -> (B, S, S)
            make_padding_mask(self.input_ids, self.pad_token)[:, None, :]
        )

        # (B, S, S)
        attention_weights = softmax(scores + mask)

        # (B, S, d_v)
        output = attention_weights @ V

        return output


