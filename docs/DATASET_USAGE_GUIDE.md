# 📊 データセット使用ガイド - Google Colab

このガイドでは、既存の高品質DPOデータセット（3,566サンプル）をGoogle Colabで活用する方法を説明します。

## 🎯 概要

- **データセットファイル**: `dpo_dataset.jsonl`
- **サンプル数**: 3,566
- **内容**: 広告技術分野の専門知識データ
- **形式**: DPO（Direct Preference Optimization）用

## 🚀 データセット使用方法

### 方法1: GitHub経由で取得 ⭐ 推奨

```python
# 1. リポジトリをクローン
!git clone https://github.com/your-username/AD_Tech_SLM.git

# 2. データセットパスを確認
dataset_path = "AD_Tech_SLM/data/dpo_dataset.jsonl"
```

**メリット:**
- 最新版が取得できる
- プロジェクト全体の構造を維持
- バージョン管理が可能

### 方法2: Google Drive経由で使用

```python
# 1. Google Driveをマウント
from google.colab import drive
drive.mount('/content/drive')

# 2. ファイルをGoogle Driveにアップロード
# 3. パスを指定して読み込み
dataset_path = "/content/drive/MyDrive/dpo_dataset.jsonl"
```

**メリット:**
- 大きなファイルも安定して転送
- 複数のColabセッションで再利用可能
- 永続的な保存

### 方法3: 直接アップロード

```python
# ファイルを直接アップロード
from google.colab import files
uploaded = files.upload()
```

**メリット:**
- 簡単で迅速
- 追加設定不要
- 小さなファイルに最適

## 📋 データセット仕様

### ファイル形式
```json
{
  "prompt": "質問やタスクの説明",
  "chosen": "好ましい回答（高品質な回答）",
  "rejected": "好ましくない回答（低品質な回答）"
}
```

### データ分野
- プログラマティック広告
- RTB（Real-Time Bidding）
- DMP/CDP技術
- 広告配信技術
- デジタルマーケティング

### 統計情報
- **総サンプル数**: 3,566
- **平均プロンプト長**: 約50文字
- **平均chosen長**: 約200文字
- **平均rejected長**: 約100文字

## ⚙️ トレーニング設定推奨値

### 大規模データセット用（3000+サンプル）
```python
training_config = {
    'num_train_epochs': 3,
    'per_device_train_batch_size': 2,
    'gradient_accumulation_steps': 8,
    'learning_rate': 5e-7,
    'max_steps': 1000,
    'eval_steps': 100,
    'save_steps': 200
}
```

### GPU メモリ最適化
- **T4 GPU (15GB)**: batch_size=2, max_length=512
- **メモリ不足時**: batch_size=1, max_length=256
- **高速化**: fp16=True, gradient_checkpointing=True

## 🔍 データ品質確認

### 基本検証
```python
import json

# データ形式確認
with open('dpo_dataset.jsonl', 'r', encoding='utf-8') as f:
    sample = json.loads(f.readline())
    print("Keys:", sample.keys())
    print("Sample:", sample)
```

### 統計分析
```python
# 文字数分布確認
lengths = {'prompt': [], 'chosen': [], 'rejected': []}
with open('dpo_dataset.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        for key in lengths:
            lengths[key].append(len(data[key]))

for key, values in lengths.items():
    print(f"{key}: avg={sum(values)/len(values):.1f}, max={max(values)}")
```

## 🚨 よくある問題と解決策

### 1. GPU メモリ不足
```
RuntimeError: CUDA out of memory
```
**解決策:**
- `per_device_train_batch_size`を1に減少
- `max_length`を256に減少
- `gradient_accumulation_steps`を増加

### 2. データ形式エラー
```
KeyError: 'chosen'
```
**解決策:**
- データセットファイルのJSONL形式を確認
- 必要なキー（prompt, chosen, rejected）の存在確認

### 3. 依存関係エラー
```
ImportError: No module named 'trl'
```
**解決策:**
- Colabノートブックの依存関係修正セルを実行
- ランタイムを再起動してから再実行

## 💡 最適化のヒント

### 1. データセット分割
```python
# 訓練・検証データの適切な分割
train_test_split(test_size=0.1, stratify=None, random_state=42)
```

### 2. ハイパーパラメータ調整
- 学習率: 1e-7 ～ 5e-6の範囲で実験
- ベータ値（DPO）: 0.1 ～ 0.5
- 温度パラメータ: 0.7 ～ 1.0

### 3. 評価指標
- Loss値の推移監視
- 生成テキストの品質評価
- ドメイン固有用語の使用頻度

## 🎯 次のステップ

1. **基本実装**: Colabノートブックでのトレーニング実行
2. **性能評価**: 生成品質の詳細分析
3. **最適化**: ハイパーパラメータチューニング
4. **本番化**: モデルサービングの準備

## 📞 サポート

問題が発生した場合：
1. Colabノートブックのエラーメッセージを確認
2. 依存関係修正セルを実行
3. ランタイムを再起動
4. GPU設定を確認

---

**Note**: このデータセットは広告技術分野に特化した高品質なコンテンツです。商用利用の際は適切なライセンス確認を行ってください。
