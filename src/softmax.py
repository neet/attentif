import numpy as np
import numpy.typing as npt

# https://manabitimes.jp/math/1371
def softmax(_xs: npt.ArrayLike) -> np.ndarray:
    xs = np.asarray(_xs, dtype=float)
    xs = xs - np.max(xs, axis=-1, keepdims=True)
    return np.exp(xs) / np.sum(np.exp(xs), axis=-1, keepdims=True)
