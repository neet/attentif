import random
from pytest import approx

from .softmax import softmax

def test_softmax() -> None:
  xs = random.sample(range(-10, 10), 10)
  ys = softmax(xs)

  assert max(ys) >= 0
  assert sum(ys) == approx(1)

