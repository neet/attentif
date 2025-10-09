import numpy as np

# https://manabitimes.jp/math/1371
def softmax(xs):
  denominator = sum([
    np.exp(x)
    for x in xs
  ])

  return [
      np.exp(x) / denominator
      for x in xs
  ]

