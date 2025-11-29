import torch

from ..layers import softmax

class CrossEntropy():
    eps: float

    def __init__(self) -> None:
        super().__init__()
        self.eps = 1e-12

    # (S, V) -> (S,) -> (S,)
    def __call__(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        probs = softmax(logits)
        probs = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        probs = probs.clamp(min=self.eps)
        return -1 * torch.log(probs)
