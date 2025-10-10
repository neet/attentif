import numpy as np

def layer_norm(x: np.ndarray, ε: float, γ: np.ndarray, β: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var  = np.var(x, axis=-1, keepdims=True)
    std  = np.sqrt(var + ε)

    x_cap = (x - mean) / std

    return γ * x_cap + β

