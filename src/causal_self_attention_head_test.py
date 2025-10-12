import math
import torch
import pytest

from .causal_self_attention_head import make_causal_mask, make_padding_mask, CausalSelfAttentionHead

torch.manual_seed(0)

def device_param():
    return ["cuda" if torch.cuda.is_available() else "cpu"]

@pytest.mark.parametrize("device", device_param())
def test_make_causal_mask(device):
    S = 5
    m = make_causal_mask(S).to(device)
    # 形状
    assert m.shape == (S, S)
    # 対角より上: -inf
    upper = torch.triu(torch.ones(S, S, dtype=torch.bool, device=device), diagonal=1)
    assert torch.isinf(m[upper]).all()
    # 対角およびそれ以下: 0
    lower_eq = ~upper
    assert (m[lower_eq] == 0).all()

@pytest.mark.parametrize("device", device_param())
def test_make_padding_mask(device):
    pad = 0
    input_ids = torch.tensor([[1,2,0,0],[3,4,5,0]], device=device)
    m = make_padding_mask(input_ids, pad)
    # pad の位置が -inf, それ以外 0
    expect = torch.tensor([[0,0,-math.inf,-math.inf],
                           [0,0,0,-math.inf]], device=device, dtype=m.dtype)
    assert torch.equal(torch.isinf(m), torch.isinf(expect))
    assert torch.equal(torch.nan_to_num(m, nan=0.0), torch.nan_to_num(expect, nan=0.0))

def _build_head(B=2, S=4, H=8, d_k=4, d_v=6, device="cpu"):
    head = CausalSelfAttentionHead()
    head.d_k = d_k
    head.pad_token = 0
    # 入力 ID（各バッチで末尾を pad）
    ids = torch.tensor([[10,11,12,0],[21,22,0,0]], device=device)
    head.input_ids = ids

    # 重み（学習させる前提で requires_grad=True）
    head.W_Q = torch.randn(H, d_k, device=device, requires_grad=True)
    head.b_Q = torch.randn(d_k, device=device, requires_grad=True)
    head.W_K = torch.randn(H, d_k, device=device, requires_grad=True)
    head.b_K = torch.randn(d_k, device=device, requires_grad=True)
    head.W_V = torch.randn(H, d_v, device=device, requires_grad=True)
    head.b_V = torch.randn(d_v, device=device, requires_grad=True)

    batch = torch.randn(B, S, H, device=device, requires_grad=False)
    return head, batch

@pytest.mark.parametrize("device", device_param())
def test_forward_shapes_and_row_probs(device):
    head, batch = _build_head(device=device)
    y = head(batch)
    # 形状
    B, S, H = batch.shape
    assert y.shape == (B, S, head.b_V.shape[0])

    # attention の行ごとの和が 1 になる（softmax dim=-1 が効いている）か検査
    # 内部変数にアクセスしないため、期待値をテスト側で再構成
    Q = batch @ head.W_Q + head.b_Q
    K = batch @ head.W_K + head.b_K
    scores = Q @ K.mT / math.sqrt(head.d_k)
    mask = (
        make_causal_mask(scores.shape[-1]).to(batch)  # (S,S)
        .unsqueeze(0)                                 # (1,S,S)
        + make_padding_mask(head.input_ids, head.pad_token).to(batch)[:, None, :]  # (B,1,S)
    )
    attn = torch.softmax(scores + mask, dim=-1)
    row_sums = attn.sum(dim=-1)
    # クエリが実トークン行では ≈1（pad 行は 1 にならないこともあるので <=1 を許容）
    assert torch.all(row_sums <= 1.0000001)
    assert (row_sums[~torch.isinf(make_padding_mask(head.input_ids, head.pad_token)).all(dim=-1)]
            .mean().item()) > 0.95

@pytest.mark.parametrize("device", device_param())
def test_padding_keys_get_zero_attention(device):
    head, batch = _build_head(device=device)
    B, S, H = batch.shape
    pad_mask_keys = (head.input_ids == head.pad_token)  # (B,S)

    Q = batch @ head.W_Q + head.b_Q
    K = batch @ head.W_K + head.b_K
    scores = Q @ K.mT / math.sqrt(head.d_k)
    mask = (
        make_causal_mask(S).to(batch).unsqueeze(0)
        + make_padding_mask(head.input_ids, head.pad_token).to(batch)[:, None, :]
    )
    attn = torch.softmax(scores + mask, dim=-1)  # (B,S,S)

    # キー側が pad の列は 0 になる
    for b in range(B):
        cols = pad_mask_keys[b].nonzero(as_tuple=True)[0]
        if len(cols) > 0:
            assert torch.allclose(attn[b, :, cols].sum(), torch.tensor(0.0, device=device))

@pytest.mark.parametrize("device", device_param())
def test_backward_grads_flow(device):
    head, batch = _build_head(device=device)
    y = head(batch)            # (B,S,d_v)
    loss = y.pow(2).mean()     # 適当なスカラー損失
    loss.backward()

    # 勾配が計算されていること
    for p in [head.W_Q, head.b_Q, head.W_K, head.b_K, head.W_V, head.b_V]:
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
