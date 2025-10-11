import numpy as np

from .mask import mask_causal, mask_padding

def test_mask_causal() -> None:
    mask = mask_causal(3)

    np.testing.assert_equal(
        mask,
        [
            [True, False, False],
            [True, True, False],
            [True, True, True]
        ]
    )

def test_mask_padding() -> None:
    mask = mask_padding(5, np.array([1, 3, 4]))

    np.testing.assert_equal(
        mask,
        [
            [True, False, False, False, False],
            [True, True,  True,  False, False],
            [True, True,  True,  True,  False],
        ]
    )
