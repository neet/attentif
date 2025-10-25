import torch
import torch.nn as nn
import math

from .token_embedding import TokenEmbedding

def test_token_embedding_reference_and_shape_and_values():
    V, H = 5, 4
    E = nn.Parameter(torch.arange(V * H, dtype=torch.float32).view(V, H))
    pad_id = 0

    emb = TokenEmbedding(E, pad_token_id=pad_id)

    # 参照が共有されている（コピーされていない）こと
    assert emb.E is E

    # 入力: (B, S)
    x = torch.tensor([[0, 1, 4],
                      [3, 2, 0]], dtype=torch.long)  # 各行に pad_id を含む

    y = emb(x)  # (B, S, H)
    assert y.shape == (2, 3, H)

    # 期待値を手計算：E[x] * sqrt(H)、ただし pad は 0 ベクトルに
    expected = E.detach()[x] * math.sqrt(H)
    expected[x == pad_id] = 0.0

    assert torch.allclose(y, expected)

def test_token_embedding_grad_flow_with_padding_mask():
    V, H = 5, 4
    E = nn.Parameter(torch.arange(V * H, dtype=torch.float32).view(V, H), requires_grad=True)
    pad_id = 0
    emb = TokenEmbedding(E, pad_token_id=pad_id)

    # x には 0（pad）と 1,2,3,4 を各1回ずつ出現させる
    x = torch.tensor([[0, 1, 2],
                      [3, 4, 0]], dtype=torch.long)

    # 前向き + 単純なスカラー損失（総和）
    y = emb(x)                 # (B, S, H)
    loss = y.sum()             # スカラー
    loss.backward()

    # 勾配チェック：pad_id=0 に対応する行の勾配は 0、他は +sqrt(H) * 出現回数
    expected_grad = torch.zeros_like(E)
    sqrtH = math.sqrt(H)

    token_counts = {1: 1, 2: 1, 3: 1, 4: 1}  # 0 は pad なので数えない
    for t, c in token_counts.items():
        expected_grad[t] = c * sqrtH  # 各要素が同一に寄与（y=sum(E[x]*sqrtH)）

    assert torch.allclose(E.grad, expected_grad)

def test_token_embedding_dtype_and_device_passthrough():
    V, H = 3, 4
    E = nn.Parameter(torch.randn(V, H, dtype=torch.float32))
    emb = TokenEmbedding(E, pad_token_id=None)

    # half でも正しく動く（戻りは E に従う）
    x = torch.tensor([[1, 2]], dtype=torch.long)
    y = emb(x)
    assert y.dtype == E.dtype
    assert y.device == E.device
