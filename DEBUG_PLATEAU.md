# Loss Plateau デバッグ記録

## 問題
損失が7.5-7.9付近で停滞（plateau）し、それ以上改善しない。
- 初期: 10.89 → 80ステップで7.75まで降下
- その後: 7.5-7.9を行ったり来たり
- 期待: 1.0付近まで下がるはず

## 試したこと（効果なし）

### ハイパーパラメータ調整
1. ❌ Learning rate: 1e-4 → 5e-4
2. ❌ Weight decay: デフォルト0.01 → 0.0
3. ❌ Warmup ratio: 0.06 → 0.1
4. ❌ Gradient clipping: 1.0 → 削除
5. ❌ データサイズ: 1k → 10k samples

### 実装修正
1. ✅ FFN内の二重dropout削除（`feed_forward_network.py:33`）- 論文準拠に修正
2. ✅ Positional Encoding スケール調整（`mini_bert.py:67`で `* 0.1`）
   - PE: [-1, 1]、Embedding: [-0.06, 0.06] のスケール差を解消
   - Gradient norm: 2.91 → 3.32（正常化）
   - Logits: 2.5前後（正常範囲）

## 診断結果

### 正常な値
- Embeddings norm: 101.4（正常）
- Gradient norm: 3.32（正常）
- Logits range: -2.5 to 2.5（正常）
- Attention mask: 0.0 と -1e9（正常）

### データ確認
- MLM mask率: 約15%（20トークン中3つが学習対象）
- Labels: -100でignore、正しく機能
- Max length: 512

### 損失の意味
- 損失7.7 = exp(-7.7) ≈ 1/2200の確率を正解に割り当て
- ランダム（10.82）よりマシだが、ほぼ学習していない状態
- 頻度分布を少し覚えただけで停滞

## 結論

**ハイパーパラメータの問題ではなく、根本的な実装バグの可能性が高い**

複数の設定を試しても同じ7.5-7.9で停滞することから、確定的な何かが学習を阻害している。

## 次のステップ（明日実行）

### 優先度1: PyTorchの標準Transformerでベースライン確認
```python
# 同じデータ・設定でPyTorchの nn.TransformerEncoder を使う
# これで学習が進めば → 自作実装にバグ
# これでも停滞すれば → データや設定の問題
```

### 優先度2: Attention mask処理の詳細診断
- (B, S) → (B, S, S) 変換が正しいか
- Padding位置へのattentionが漏れていないか
- MHA内でのmask適用を確認

### 優先度3: 疑わしい箇所
1. LayerNorm実装（数値安定性）
2. Softmax + -inf の相互作用
3. Pre-LN vs Post-LN アーキテクチャ
4. MPSデバイスの数値精度問題

## 修正済みファイル

1. `src/feed_forward_network.py`: FFN内部のdropout削除
2. `src/mini_bert.py:67`: PE に `* 0.1` スケーリング追加
3. Notebook: gradient clipping削除、LR=5e-4、weight_decay=0.0、warmup=0.1

## 設定

```python
# 現在の設定
warmup_step_ratio = 0.1
num_epochs = 20
batch_size = 32
max_length = 512
lr = 5e-4
weight_decay = 0.0
# gradient clipping: 削除
device = "mps"
dataset = "allenai/c4" 10k samples
```
