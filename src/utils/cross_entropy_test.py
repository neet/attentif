import torch
from ..layers import softmax
from .cross_entropy import CrossEntropy

def test_cross_entropy_basic():
    logits = torch.tensor([[1.0, 2.0, 3.0],
                           [1.0, 0.0, -1.0]])
    labels = torch.tensor([2, 0])

    loss = CrossEntropy()
    out = loss(logits, labels)

    probs = softmax(logits, dim=-1)
    expected = -torch.log(probs[torch.arange(2), labels])

    assert out.shape == (2,)
    assert torch.allclose(out, expected, atol=1e-6)
