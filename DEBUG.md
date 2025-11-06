# デバッグレポート: Lossが低いのに品質が悪い理由

## サマリー
- 自作実装: 最終loss ~4.9だが、予測が意味不明
- HF BERT実装: 最終loss ~5.8だが、予測が意味的に正しい
- **診断結果**: 標準的なBERTアーキテクチャとの複数の違いが原因

## テストケース比較

### 入力
```
"All human [MASK] are born free and equal in dignity and rights."
```

### HF BERT（正常動作）
```
Top-5予測:
  rights     7.6069  ✓（文脈的に正しい）
  ##s        7.1869  ✓（"humans"になる）
  groups     6.8359  ✓
  studies    6.6711  ✓
  issues     6.4997  ✓

正解「beings」のlogit: +3.69
```

### 自作実装（異常）
```
Top-5予測:
  .          6.5330  ✗（意味不明）
  he         6.2388  ✗
  other      5.7639  ✗
  money      5.7295  ✗
  love       5.6699  ✗

正解「beings」のlogit: -0.70
```

---

## 致命的な問題（必ず修正すべき）

### 1. **Embedding LayerNormが欠落** （最重要）
**場所**: `src/masked_lm.py:64-67`

**現在の実装（間違い）**:
```python
embeddings = self.token_embedding(batch)
pe = positional_encoding(...)
input = embeddings + pe * 0.1  # LayerNormがない！
output = self.transformer_encoder(input, attention_mask)
```

**正しい実装（BERT標準）**:
```python
embeddings = word_embeddings + position_embeddings + token_type_embeddings
embeddings = LayerNorm(embeddings)
embeddings = Dropout(embeddings)
```

**影響**:
- 自作: 入力のstd ≈ 0.059（小さすぎる）
- HF BERT: 入力のstd ≈ 1.0（LayerNorm後）
- LayerNormがないと学習が不安定になり、モデルが間違ったパターンを学習する

**参考**: BERT論文 Section 3.1、Hugging Face `BertEmbeddings` 実装

---

### 2. **Token Type Embeddingsが欠落**
**場所**: `src/masked_lm.py`

**問題**: Token type（セグメント）embeddingが全く実装されていない

**正しい実装**: BERTはtoken type embeddingsを使う（単一文タスクでは通常全て0だが、embedding層自体は存在し学習される）

**影響**: BERTの標準的なコンポーネントが欠けており、学習に悪影響

---

### 3. **Token EmbeddingとPositional Encodingのスケーリングが間違い**
**場所**: `src/masked_lm.py:63-67`, `src/token_embedding.py:15`, `src/positional_encoding.py`

**現在の実装（間違い）**:
```python
# src/token_embedding.py:15
# コメント: "sqrt(H)スケーリングを削除（weight tyingとの相互作用を避けるため）"
embeddings = self.embedding[x]  # スケーリングなし！

# src/masked_lm.py:67
pe = positional_encoding(...)
input = embeddings + pe * 0.1  # ← 間違ったスケーリング
```

**問題点**:

#### Vaswani et al. (2017) "Attention is All You Need" の場合:
論文Section 3.4では以下のように実装すべき：
```python
# Token embeddingを sqrt(d_model) でスケールアップ
embeddings = sqrt(hidden_size) * token_embedding(x)
# Positional encodingはそのまま（スケーリングなし）
pe = positional_encoding(seq_len, hidden_size)
# 加算
input = embeddings + pe
```

**理由**:
- Positional encoding（sinusoidal）は範囲 [-1, +1]
- Token embeddingの初期値は std ≈ 0.02（範囲 ≈ [-0.04, +0.04]）
- `sqrt(hidden_size)` ≈ 27.7（hidden_size=768の場合）を掛けることで、token embeddingとPEが同じスケールになる
- これにより位置情報が相対的に薄まらない

**現在の実装の問題**:
1. Token embeddingに `sqrt(hidden_size)` を掛けていない → embedding が小さすぎる
2. PEに `* 0.1` を掛けている → 位置情報が弱すぎる
3. 結果として、入力のstd ≈ 0.059（小さすぎる）

#### BERT (2018) の場合:
BERTは全く異なるアプローチを取る：
```python
# 学習可能なpositional embeddings（固定sinusoidalではない）
position_embeddings = self.position_embeddings(position_ids)
embeddings = word_embeddings + position_embeddings + token_type_embeddings
# その後LayerNorm + Dropoutを適用
embeddings = self.LayerNorm(embeddings)
embeddings = self.dropout(embeddings)
```

- BERTでは**学習可能な**positional embeddingsを使用
- Token embeddingにスケーリングは**適用しない**
- 代わりに**Embedding LayerNorm**で正規化する

**統計比較**:
- 自作（現在）: 入力 std ≈ 0.059
- Vaswani (2017) 正しい実装: 入力 std ≈ 0.55（推定）
- HF BERT（LayerNorm後）: 入力 std ≈ 1.0

**結論**:
- 固定sinusoidal PEを使う場合（Vaswani流）: Token embeddingに `sqrt(hidden_size)` を掛け、PEは `* 0.1` せず直接加算
- BERTと同等にしたい場合: 学習可能なpositional embeddings + Embedding LayerNorm/Dropoutを実装

---

### 4. **活性化関数が間違い**
**場所**: `src/feed_forward_network.py:30`

**問題**:
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    y = x @ self.W1 + self.b1
    y = relu(y)  # ← GELUであるべき
    y = y @ self.W2 + self.b2
    return y
```

**正しい実装**: BERTは**GELU**（Gaussian Error Linear Unit）を使用、ReLUではない

**影響**:
- GELUは滑らかで、勾配特性が良い
- ReLUは負の値を完全にゼロにするが、GELUは小さな負の値の寄与を許す
- これは学習ダイナミクスに大きく影響する

**参考**: BERT configには `"hidden_act": "gelu"` と記載

---

### 5. **Attention Dropoutが欠落**
**場所**: `src/multi_head_attention.py:65`

**問題**:
```python
# (B, h, S, S)
attention_weights = softmax(scores, dim=-1)
# ここにdropoutがない！

# (B, h, S, d_v)
output = attention_weights @ V
```

**正しい実装**:
```python
attention_weights = softmax(scores, dim=-1)
attention_weights = dropout(attention_weights, p=0.1, training=self.training)
output = attention_weights @ V
```

**影響**:
- Attentionメカニズムの正則化が欠落
- BERTは `attention_probs_dropout_prob=0.1` を使用

---

### 6. **LayerNormのepsilonが間違い**
**場所**: `src/layer_norm.py:5`

**問題**:
```python
class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):  # ← 間違ったepsilon
```

**正しい実装**: BERTは `eps=1e-12` を使用

**影響**:
- Epsilonは数値安定性に影響する
- 異なるepsilonは異なる勾配を生む可能性がある
- BERT configには `"layer_norm_eps": 1e-12` と記載

---

## アーキテクチャ上の違い（問題かどうか不明）

### 7. **Pre-LN vs Post-LNアーキテクチャ**
**場所**: `src/transformer_encoder_block.py:23-32`

**現在の実装（Pre-LN）**:
```python
x_ln = self.ln1(x)
x_mha = self.mha(x_ln, attention_mask=attention_mask)
y = x + dropout(x_mha, training=self.training)
y_ln = self.ln2(y)
y_ffn = self.ffn(y_ln)
z = y + dropout(y_ffn, training=self.training)
```

**BERT標準（Post-LN）**:
```python
x_mha = self.mha(x, attention_mask=attention_mask)
x_mha = dropout(x_mha, training=self.training)
y = self.ln1(x + x_mha)
y_ffn = self.ffn(y)
y_ffn = dropout(y_ffn, training=self.training)
z = self.ln2(y + y_ffn)
```

**注記**: Pre-LNは実は現代的なアーキテクチャでは訓練安定性が高いとされているが、BERTはPost-LNを使う。この違い単独では品質問題を引き起こさないはずだが、非標準的なアーキテクチャになる。

---

## なぜLossは低いのに品質が悪い？

### 仮説1: 間違ったパターンへの過学習
- 適切なLayerNormと正しいpositional encodingがないと、モデルは意味を理解せずに訓練データのパターンを暗記する可能性がある
- 訓練データでの低いlossは、良い汎化性能を保証しない

### 仮説2: Loss計算は正しいが、学習が間違っている
- Cross-entropy lossは正しく計算されている
- しかしモデルは意味的なパターンではなく、ノイズ/アーティファクトにフィットしてlossを最小化している
- これはアーキテクチャが不安定な場合（LayerNorm欠落）に起こる

### 仮説3: Embeddingスケールの不一致
- 入力embeddingのstd ≈ 0.06（小さすぎる）
- BERTはLayerNorm後にstd ≈ 1.0
- 小さい値は勾配を弱くし、学習を非効率にする
- モデルは偽相関を学習している可能性がある

---

## 推奨される修正

### 選択肢A: Vaswani (2017) "Attention is All You Need" に忠実な実装

#### 優先度1（重要）
1. **Token embeddingに `sqrt(hidden_size)` スケーリングを追加**（`src/token_embedding.py`）
2. **PEの `* 0.1` スケーリングを削除**（`src/masked_lm.py:67`）
3. FFNのReLUをGELUに置き換え（`src/feed_forward_network.py`）

#### 優先度2
4. Attention probabilitiesにdropoutを追加（`src/multi_head_attention.py`）

#### 備考
- この方法でも入力のstdは ≈ 0.55程度（BERTの1.0より小さい）
- Embedding LayerNormは**追加しない**（Vaswani (2017)には存在しない）
- 固定sinusoidal PEのまま

---

### 選択肢B: BERT (2018) に近づける実装（推奨）

#### 優先度1（重要 - 即座に修正）
1. **Embedding LayerNorm + Dropoutを追加**（`src/masked_lm.py`に新しいモジュール作成）
2. **固定sinusoidal PEを学習可能なpositional embeddingsに置き換え**
3. **PEの `* 0.1` スケーリングを削除**
4. **Token Type Embeddingsを追加**（MLMでは常に0）
5. FFNのReLUをGELUに置き換え

#### 優先度2
6. Attention probabilitiesにdropoutを追加
7. LayerNormのepsilonを1e-5から1e-12に変更

#### 優先度3（オプション）
8. Pre-LNからPost-LNへの切り替え（BERTとの完全な互換性）

#### 備考
- HF BERTと同等の性能が期待できる
- 入力のstd ≈ 1.0になる（LayerNorm効果）
- より標準的なBERT実装になる

---

### 推奨アプローチ

**選択肢B（BERT寄り）を推奨**する理由：
1. HF Transformersとの比較がしやすい
2. Embedding LayerNormによる学習安定化
3. 現代的なNLPモデルの標準的な構成

ただし、教育目的で元のTransformer論文に忠実な実装を維持したい場合は、選択肢Aでも問題ない。

---

## 修正が必要なファイル

### 選択肢A（Vaswani流）の場合:
1. `src/token_embedding.py` - `sqrt(hidden_size)` スケーリングを復活
2. `src/masked_lm.py` - PE の `* 0.1` スケーリングを削除
3. `src/feed_forward_network.py` - ReLUをGELUに置き換え
4. `src/multi_head_attention.py` - Attention dropoutを追加

### 選択肢B（BERT流）の場合:
1. `src/masked_lm.py` - Embedding LayerNorm/Dropout追加、学習可能なpositional embeddings、token type embeddings追加、PEスケーリング削除
2. `src/feed_forward_network.py` - ReLUをGELUに置き換え
3. `src/multi_head_attention.py` - Attention dropoutを追加
4. `src/layer_norm.py` - デフォルトepsilonを1e-12に変更
5. `src/transformer_encoder_block.py` - （オプション）Post-LNアーキテクチャに切り替え

---

## 修正後の検証手順

1. 入力embeddingの統計がHF BERTと一致するか確認（std ≈ 1.0）
2. 同じハイパーパラメータでモデルを再訓練
3. 同じ例でテスト: "All human [MASK] are born free..."
4. 「beings」のlogitが正で高い値になっているか確認
5. Validation setでのperplexityをHF BERTと比較
