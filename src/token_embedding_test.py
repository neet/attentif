import torch
import torch.nn as nn
import math

from .token_embedding import TokenEmbedding

def test_token_embedding_reference_and_shape_and_values():
    vocab_size, hidden_size = 5, 4
    embedding = nn.Parameter(torch.arange(vocab_size * hidden_size, dtype=torch.float32).view(vocab_size, hidden_size))
    pad_id = 0

    token_embedding = TokenEmbedding(embedding, pad_token_id=pad_id)

    # 参照が共有されている（コピーされていない）こと
    assert token_embedding.embedding is embedding

    # 入力: (batch_size, seq_len)
    x = torch.tensor([[0, 1, 4],
                      [3, 2, 0]], dtype=torch.long)  # 各行に pad_id を含む

    y = token_embedding(x)  # (batch_size, seq_len, hidden_size)
    assert y.shape == (2, 3, hidden_size)

    # 期待値を手計算：embedding[x]（sqrt(hidden_size)スケーリングなし）、ただし pad は 0 ベクトルに
    expected = embedding.detach()[x]
    expected[x == pad_id] = 0.0

    assert torch.allclose(y, expected)

def test_token_embedding_grad_flow_with_padding_mask():
    vocab_size, hidden_size = 5, 4
    embedding = nn.Parameter(torch.arange(vocab_size * hidden_size, dtype=torch.float32).view(vocab_size, hidden_size), requires_grad=True)
    pad_id = 0
    token_embedding = TokenEmbedding(embedding, pad_token_id=pad_id)

    # x には 0（pad）と 1,2,3,4 を各1回ずつ出現させる
    x = torch.tensor([[0, 1, 2],
                      [3, 4, 0]], dtype=torch.long)

    # 前向き + 単純なスカラー損失（総和）
    y = token_embedding(x)                 # (batch_size, seq_len, hidden_size)
    loss = y.sum()             # スカラー
    loss.backward()

    # 勾配チェック：pad_id=0 に対応する行の勾配は 0、他は +1 * 出現回数
    expected_grad = torch.zeros_like(embedding)

    token_counts = {1: 1, 2: 1, 3: 1, 4: 1}  # 0 は pad なので数えない
    for t, c in token_counts.items():
        expected_grad[t] = c  # 各要素が同一に寄与（y=sum(embedding[x])）

    assert torch.allclose(embedding.grad, expected_grad)

def test_token_embedding_dtype_and_device_passthrough():
    vocab_size, hidden_size = 3, 4
    embedding = nn.Parameter(torch.randn(vocab_size, hidden_size, dtype=torch.float32))
    emb = TokenEmbedding(embedding, pad_token_id=None)

    # half でも正しく動く（戻りは embedding に従う）
    x = torch.tensor([[1, 2]], dtype=torch.long)
    y = emb(x)
    assert y.dtype == embedding.dtype
    assert y.device == embedding.device
