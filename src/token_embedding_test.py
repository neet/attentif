import math
import torch
import pytest

from .token_embedding import TokenEmbedding

B, S, H = 4, 7, 16
V = 50
SQRT_H = math.sqrt(H)

@pytest.fixture
def ids_ok():
    torch.manual_seed(0)
    return torch.randint(0, V, (B, S), dtype=torch.long)

def make_module():
    m = TokenEmbedding(vocab_size=V, H=H)
    # テスト安定化のため明示初期化（実装側でやっていてもOK）
    with torch.no_grad():
        torch.manual_seed(123)
        m.E.copy_(torch.randn(V, H))
    return m

def test_shape_and_dtype(ids_ok):
    m = make_module()
    y = m(ids_ok)
    assert y.shape == (B, S, H)
    assert y.dtype.is_floating_point

def test_device_matches_input(ids_ok):
    m = make_module()
    if torch.cuda.is_available():
        m = m.cuda()
        ids_ok = ids_ok.cuda()
    y = m(ids_ok)
    assert y.device == ids_ok.device

def test_out_of_range_raises():
    m = make_module()
    x = torch.full((1, 1), V, dtype=torch.long)  # 0..V-1 が有効
    with pytest.raises(IndexError):
        _ = m(x)

def test_equals_one_hot_gather_with_sqrtH(ids_ok):
    """E[x] * √H と (one_hot @ E) * √H が一致する"""
    m = make_module()
    y_gather = m(ids_ok)  # 内部で * √H される前提

    oh = torch.nn.functional.one_hot(ids_ok, num_classes=V).to(dtype=m.E.dtype)
    y_onehot = oh @ m.E
    y_onehot = y_onehot * SQRT_H

    assert torch.allclose(y_gather, y_onehot, atol=0, rtol=0)

def test_grad_flows_only_to_used_rows_scaled():
    """
    loss = sum(y) のとき、使われた行 r の勾配は
    grad[r, :] = (出現回数 count[r]) * √H * 1 になる（各次元同値）
    """
    m = make_module()
    x = torch.tensor([[0, 1, 1, 4],
                      [4, 4, 1, 2]], dtype=torch.long, device=m.E.device)
    y = m(x)                  # (2,4,H), 内部で * √H
    loss = y.sum()
    loss.backward()

    counts = torch.bincount(x.flatten(), minlength=V).to(m.E.device)
    grad = m.E.grad           # (V,H)

    # 使っていない行は勾配ゼロ
    unused_rows = (counts == 0).nonzero(as_tuple=True)[0]
    if len(unused_rows) > 0:
        assert torch.allclose(grad[unused_rows], torch.zeros_like(grad[unused_rows]))

    # 使った行は count * √H が各次元に入る
    used_rows = (counts > 0).nonzero(as_tuple=True)[0]
    for r in used_rows.tolist():
        expected = torch.full((H,), float(counts[r]) * SQRT_H,
                              dtype=grad.dtype, device=grad.device)
        assert torch.allclose(grad[r], expected, atol=0, rtol=0)

def test_batch_and_seq_independence():
    """同じIDなら位置やバッチが違っても同じベクトル（√Hスケール後でも同一）"""
    m = make_module()
    x = torch.tensor([[3, 3, 3],
                      [3, 3, 3]], dtype=torch.long, device=m.E.device)
    y = m(x)
    assert torch.allclose(y[0,0], y[0,1])
    assert torch.allclose(y[0,1], y[1,2])

def test_norm_scales_with_sqrtH_expectation():
    """スケールの直観テスト：E[x] と m(x) のノルム比がおおむね √H"""
    m = make_module()
    x = torch.randint(0, V, (2, 5), dtype=torch.long, device=m.E.device)

    with torch.no_grad():
        raw = m.E[x]                 # スケール前
        scaled = m(x)                # スケール後（* √H）
        # 平均ノルム比がだいたい √H（乱数ゆえ誤差を許容）
        r = (scaled.norm(dim=-1) / (raw.norm(dim=-1) + 1e-12)).mean().item()
        assert abs(r - SQRT_H) / SQRT_H < 0.1  # ±10% 以内

PAD = 0

def make_module_with_pad():
    m = TokenEmbedding(vocab_size=V, H=H, pad_token=PAD)
    with torch.no_grad():
        torch.manual_seed(777)
        m.E.copy_(torch.randn(V, H))
        # PAD 行はわざと非ゼロにしておく（出力でゼロ化されることを検証）
        m.E[PAD].fill_(123.456)
    return m

def test_pad_positions_are_zero():
    m = make_module_with_pad()
    x = torch.tensor([[1, PAD, 2, PAD, 3, 4, 5]], dtype=torch.long, device=m.E.device)
    y = m(x)  # (1,S,H)

    # (1,S,1) → (1,S,H) に拡張してから使う
    pad_mask = (x == PAD).unsqueeze(-1).expand_as(y)  # (1,S,H)

    # PAD 位置が完全ゼロ
    assert torch.allclose(y[pad_mask], torch.zeros_like(y[pad_mask]))

    # 非PAD 位置はゼロではない（統計的に非ゼロ）
    nonpad_mask = (~(x == PAD)).unsqueeze(-1).expand_as(y)
    assert (y[nonpad_mask].abs().sum() > 0).item() is True

def test_equals_one_hot_with_pad_mask():
    """
    実装の挙動：
      y = (one_hot @ E) * √H
      y[PAD位置] = 0
    となっていることを検証
    """
    m = make_module_with_pad()
    x = torch.randint(0, V, (B, S), dtype=torch.long, device=m.E.device)
    # 一部をPADに置換
    x[:, -2:] = PAD

    y = m(x)

    oh = torch.nn.functional.one_hot(x, num_classes=V).to(dtype=m.E.dtype)  # (B,S,V)

    y_expected = (oh @ m.E) * SQRT_H
    y_expected = y_expected.masked_fill((x == PAD).unsqueeze(-1), 0.0)

    assert torch.allclose(y, y_expected, atol=0, rtol=0)

def test_pad_row_gets_no_grad():
    """
    loss = sum(y) のとき、PAD 行には勾配が流れない。
    （出力で PAD 位置をゼロ定数にしているため）
    """
    m = make_module_with_pad()
    # PAD を含む入力
    x = torch.tensor([[PAD, 1, 1, PAD],
                      [2,   2, PAD,  3]], dtype=torch.long, device=m.E.device)

    y = m(x)              # (2,4,H)
    loss = y.sum()
    loss.backward()

    # PAD 行の勾配はゼロ
    assert torch.allclose(m.E.grad[PAD], torch.zeros_like(m.E.grad[PAD]))

    # 非PADで使われた行には count * √H の勾配が各次元に入る（実装が y=E[x]*√H のため）
    counts = torch.bincount(x.flatten(), minlength=V).to(m.E.device)
    for r in (counts > 0).nonzero(as_tuple=True)[0].tolist():
        if r == PAD:
            continue
        expected = torch.full((H,), float(counts[r]) * SQRT_H,
                              dtype=m.E.grad.dtype, device=m.E.grad.device)
        assert torch.allclose(m.E.grad[r], expected, atol=0, rtol=0)

def test_pad_row_value_does_not_leak_to_output():
    """
    E[PAD] に巨視的に大きい値を入れても、出力は常にゼロであることを検証。
    """
    m = make_module_with_pad()
    with torch.no_grad():
        m.E[PAD].fill_(1e6)  # 極端に大きく

    x = torch.full((1, S), PAD, dtype=torch.long, device=m.E.device)
    y = m(x)  # すべて PAD

    assert torch.allclose(y, torch.zeros_like(y))

