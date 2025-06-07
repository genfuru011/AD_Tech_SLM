# DPO Training Environment Setup Guide

このガイドでは、MacBook Air M2 8GB環境でAD_Tech_SLMのDPO（Direct Preference Optimization）トレーニング環境をセットアップする方法を説明します。

## 📋 必要環境

- **ハードウェア**: MacBook Air M2 8GB
- **OS**: macOS (Apple Silicon対応)
- **Python**: 3.8以上
- **VSCode**: 推奨IDE
- **GPU**: M2チップ (Metal Performance Shaders対応)

## 🚀 クイックスタート

### 1. 環境セットアップ

```bash
# リポジトリをクローン
git clone https://github.com/genfuru011/AD_Tech_SLM.git
cd AD_Tech_SLM

# セットアップスクリプトを実行
./setup.sh
```

### 2. 仮想環境の有効化

```bash
source venv/bin/activate
```

### 3. データの確認

```bash
# サンプルデータセットの検証
python scripts/validate_data.py data/sample_dpo_dataset.jsonl
```

### 4. DPOトレーニングの実行

```bash
# 基本的なトレーニング
python scripts/train_dpo.py

# カスタム設定でトレーニング
python scripts/train_dpo.py --config configs/dpo_config.yaml
```

### 5. 推論テスト

```bash
# インタラクティブな推論
python scripts/inference.py
```

## 📁 プロジェクト構造

```
AD_Tech_SLM/
├── configs/
│   └── dpo_config.yaml          # DPOトレーニング設定
├── data/
│   └── sample_dpo_dataset.jsonl # サンプルデータセット
├── scripts/
│   ├── train_dpo.py            # DPOトレーニングスクリプト
│   ├── validate_data.py        # データ検証ユーティリティ
│   └── inference.py            # 推論スクリプト
├── notebooks/
│   └── dpo_training_experiment.ipynb # Jupyter実験ノート
├── requirements.txt            # Python依存関係
├── setup.sh                   # 環境セットアップスクリプト
└── README_DPO_TRAINING.md     # このファイル
```

## ⚙️ 設定詳細

### DPO設定 (`configs/dpo_config.yaml`)

M2 8GB環境に最適化された設定：

```yaml
# モデル設定
model_name: "google/gemma-2b-it"  # 軽量モデル
max_length: 512
max_prompt_length: 256

# トレーニングパラメータ
num_train_epochs: 3
per_device_train_batch_size: 1    # メモリ節約
gradient_accumulation_steps: 4    # 実効バッチサイズ = 4
learning_rate: 5e-6
beta: 0.1  # DPO beta パラメータ

# LoRA設定
use_lora: true
lora_r: 16
lora_alpha: 32
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]

# ハードウェア設定
device: "mps"  # M2 GPU
fp16: true     # メモリ効率化
```

### データセット形式

JSONLファイルで、各行は以下の形式：

```json
{
  "prompt": "【テーマ】アプリのプロモーション文を書いてください",
  "chosen": "魅力的で効果的な広告文",
  "rejected": "平凡で効果の低い広告文"
}
```

## 📊 データセットの準備

### 1. 独自データセットの作成

```bash
# サンプルデータセットの作成
python scripts/validate_data.py --create-sample data/my_dataset.jsonl 100

# データの検証
python scripts/validate_data.py data/my_dataset.jsonl
```

### 2. データセット要件

- **prompt**: 広告作成のお題や条件
- **chosen**: CTRが高い、または評価の高い広告文
- **rejected**: CTRが低い、または評価の低い広告文

各フィールドは日本語で、感情や魅力を表現するテキストが含まれることが推奨されます。

## 🎯 トレーニングのベストプラクティス

### M2 8GB環境での最適化

1. **バッチサイズ**: 1に設定し、gradient_accumulationで調整
2. **混合精度**: fp16を有効にしてメモリ使用量を削減
3. **LoRA**: フルファインチューニングではなくLoRAを使用
4. **グラデーションチェックポイント**: メモリ使用量をさらに削減

### 推奨トレーニング手順

```bash
# 1. データの検証
python scripts/validate_data.py data/your_dataset.jsonl

# 2. 小規模テスト（1エポック）
# configs/dpo_config.yaml で num_train_epochs: 1 に設定
python scripts/train_dpo.py

# 3. 結果確認
python scripts/inference.py

# 4. 本格トレーニング
# configs/dpo_config.yaml で num_train_epochs: 3-5 に設定
python scripts/train_dpo.py
```

## 📈 モニタリングと評価

### TensorBoard での監視

```bash
# TensorBoardの起動
tensorboard --logdir outputs/logs

# ブラウザで http://localhost:6006 にアクセス
```

### 評価指標

- **Training Loss**: トレーニング損失の減少
- **Evaluation Loss**: 検証損失の監視
- **Preference Accuracy**: DPOの preference accuracy

## 🔧 トラブルシューティング

### よくある問題と解決方法

#### 1. メモリ不足エラー

```bash
# バッチサイズを1に、gradient_accumulation_stepsを減らす
# configs/dpo_config.yaml で以下を調整:
per_device_train_batch_size: 1
gradient_accumulation_steps: 2
```

#### 2. MPS が利用できない

```bash
# CPUフォールバックの確認
python -c "import torch; print(torch.backends.mps.is_available())"

# CPUモードでのトレーニング
# configs/dpo_config.yaml で device: "cpu" に変更
```

#### 3. モデル読み込みエラー

```bash
# キャッシュのクリア
rm -rf ~/.cache/huggingface/transformers/

# 再実行
python scripts/train_dpo.py
```

## 📚 参考資料

- [DPO (Direct Preference Optimization) 論文](https://arxiv.org/abs/2305.18290)
- [TRL (Transformer Reinforcement Learning) ドキュメント](https://huggingface.co/docs/trl/index)
- [PEFT (Parameter-Efficient Fine-tuning) ガイド](https://huggingface.co/docs/peft/index)
- [Metal Performance Shaders (MPS) ガイド](https://developer.apple.com/metal/pytorch/)

## 🤝 コントリビューション

バグ報告や機能要求は、GitHubのIssueでお知らせください。
プルリクエストも歓迎します！

## 📄 ライセンス

このプロジェクトはApache License 2.0の下で公開されています。