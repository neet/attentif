import numpy as np

def mask_causal(seq_len: int) -> np.ndarray:
    x = np.empty((seq_len, seq_len), dtype=bool)

    for i in range(0, seq_len):
        x[i, :i+1] = True
        x[i, i+1:] = False

    return x

def mask_padding(max_seq_len: int, lengths: np.ndarray) -> np.ndarray:
    x = np.empty((lengths.shape[0], max_seq_len), dtype=bool)

    for (i, length) in enumerate(lengths):
        x[i, :length] = True
        x[i, length:] = False

    return x
