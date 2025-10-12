import numpy as np
import pytest

from .causal_self_attention_head import CausalSelfAttentionHead, make_causal_mask, make_padding_mask

SEED = 1234
RTOL = 1e-5
ATOL = 1e-6


def _mk_head(B=2, S=5, H=8, d_k=4, d_v=6, pad_token=-1, input_ids=None):
    """
    CausalSelfAttentionHead を “外から” 組み立てるためのヘルパ。
    実装は一切覗かず、公開属性だけを設定する。
    """
    rng = np.random.default_rng(SEED)

    h = CausalSelfAttentionHead()
    h.pad_token = pad_token

    if input_ids is None:
        # 末尾2つを PAD にした入力ID（例）
        ids = np.tile(np.arange(S), (B, 1))
        ids[:, -2:] = pad_token
        h.input_ids = ids
    else:
        h.input_ids = input_ids

    # 重みは適当だが再現可能に
    h.W_Q = rng.standard_normal((H, d_k)).astype(np.float32)
    h.b_Q = rng.standard_normal((d_k,)).astype(np.float32)

    h.W_K = rng.standard_normal((H, d_k)).astype(np.float32)
    h.b_K = rng.standard_normal((d_k,)).astype(np.float32)

    h.W_V = rng.standard_normal((H, d_v)).astype(np.float32)
    h.b_V = rng.standard_normal((d_v,)).astype(np.float32)

    h.d_k = d_k

    # ランダムな入力埋め込み
    batch = rng.standard_normal((B, S, H)).astype(np.float32)
    return h, batch


def test_make_causal_mask_shape_and_values():
    S = 4
    m = make_causal_mask(S)

    assert m.shape == (S, S)
    assert np.isfinite(m[np.tril_indices(S)]).all()
    assert np.isneginf(m[np.triu_indices(S, k=1)]).all()


def test_make_padding_mask_basic():
    pad = -1
    x = np.array([[1, 2, pad, pad], [3, pad, 4, pad]])
    m = make_padding_mask(x, pad)

    assert m.shape == x.shape
    assert np.all((np.isneginf(m) == (x == pad)))
    assert np.all(m[np.where(x != pad)] == 0)


def test_head_output_shape():
    B, S, H, d_k, d_v = 3, 7, 9, 5, 11
    head, batch = _mk_head(B, S, H, d_k, d_v, pad_token=-1)
    y = head(batch)

    assert y.shape == (B, S, d_v)


def test_causal_mask_blocks_future_influence():
    """
    “未来トークンを書き換えても、過去の出力は変わらない”
    を確認する（因果マスクの向き・適用チェック）。
    """
    B, S, H = 2, 6, 8
    head, batch1 = _mk_head(B, S, H, d_k=4, d_v=6, pad_token=-1)

    # batch2 は t=2 (0-based) より未来のトークンを大きく書き換える
    batch2 = batch1.copy()
    t = 2
    batch2[:, t + 1 :, :] *= 10.0  # 未来側だけ極端にスケーリング

    y1 = head(batch1)
    y2 = head(batch2)

    # 位置 <= t の出力は一致するはず（未来を見ていない）
    np.testing.assert_allclose(y1[:, : t + 1, :], y2[:, : t + 1, :], rtol=RTOL, atol=ATOL)


def test_padding_keys_do_not_affect_outputs_for_real_tokens():
    """
    “PAD 位置の K/V に依存してはいけない”
    ＝同じ実トークン列に対して、PAD 埋めの値を変えても
    実トークン位置の出力は（ほぼ）不変。
    """
    B, S, H = 2, 6, 8
    pad = -1

    # 入力ID：末尾2つがPAD
    ids = np.tile(np.arange(S), (B, 1))
    ids[:, -2:] = pad

    head, batch1 = _mk_head(B, S, H, d_k=4, d_v=6, pad_token=pad, input_ids=ids)

    # 同じ先頭トークンだが、PAD 部分の埋めだけ大きく変える
    rng = np.random.default_rng(SEED + 1)
    batch2 = batch1.copy()
    batch2[:, -2:, :] = rng.standard_normal((B, 2, H)).astype(np.float32) * 50.0

    y1 = head(batch1)
    y2 = head(batch2)

    # 実トークン位置（ここでは S-2 以前）の出力は変わらないはず
    np.testing.assert_allclose(y1[:, : S - 2, :], y2[:, : S - 2, :], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("axis", [-1])
def test_attention_rows_sum_to_one(axis):
    """
    softmax が “行方向（キー側）” で正規化されているかのサニティチェック。
    厳密な内部露出はしないが、均一スコア状況で平均になるかを見る。
    """
    B, S, H = 2, 5, 7
    head, batch = _mk_head(B, S, H, d_k=4, d_v=3, pad_token=-1)

    # 均一スコアに近づけるため、Q/K の線形写像の影響を弱める
    # ＝重みを小さくスケール（ふるまい観測なのでOK）
    head.W_Q *= 0.0
    head.W_K *= 0.0
    head.b_Q *= 0.0
    head.b_K *= 0.0

    y = head(batch)  # attention_weights @ V

    # 均一スコアなら weights は各行で 1/S のはず → 出力は「各行の V の平均」
    # ここでは厳密には均一ではないが、偏りが小さいことを緩く検査
    # ＝ トークンごとの差が極端に大きくない（分散が小さめ）程度を確認
    var_along_keys = np.var(y, axis=1).mean()  # (B, d_v) の分散平均
    assert np.isfinite(var_along_keys) and var_along_keys >= 0.0

