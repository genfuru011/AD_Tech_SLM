# 高性能PC移行ガイド

## 概要
MacBook Air M2 8GB環境でのTinySwallow-1.5B-Instruct DPO訓練において、メモリ不足のため高性能PCへの移行が必要となりました。
このドキュメントでは、移行先環境での迅速なセットアップと継続のための情報をまとめています。

## 現在の進捗状況

### ✅ 完了した作業
1. **環境セットアップ**: conda環境 `dpo_training` の構築完了
2. **モデル検証**: SakanaAI/TinySwallow-1.5B-Instruct モデルの動作確認済み
3. **データセット準備**: 3,565サンプルのDPOデータセット準備完了
4. **設定ファイル**: TinySwallow用最適化設定完了
5. **訓練スクリプト**: DPO訓練スクリプト開発・デバッグ完了

### ⚠️ 発生した課題
- **メモリ不足**: MacBook Air M2 8GB では TinySwallow-1.5B (15.48億パラメータ) の訓練が困難
- **TRL APIエラー**: TRL 0.18.1でのDPOTrainerパラメータ互換性問題

### 📊 技術仕様
- **モデル**: SakanaAI/TinySwallow-1.5B-Instruct (1,548,072,448 パラメータ)
- **データセット**: 3,208訓練サンプル + 357検証サンプル
- **訓練手法**: DPO (Direct Preference Optimization) + LoRA
- **ハードウェア**: MPS (Metal Performance Shaders) 対応

## 移行先環境要件

### 推奨ハードウェア仕様
- **GPU**: NVIDIA RTX 4070以上 (VRAM 12GB以上)
- **メモリ**: 32GB以上
- **ストレージ**: SSD 100GB以上の空き容量

### 必要なソフトウェア
```bash
# Python環境
Python 3.10+
conda または miniconda

# 主要ライブラリ
torch>=2.0.0
transformers>=4.35.0
trl>=0.18.0
peft>=0.7.0
datasets>=2.14.0
accelerate>=0.24.0
```

## セットアップ手順

### 1. リポジトリクローン
```bash
git clone [リポジトリURL]
cd AD_Tech_SLM
```

### 2. conda環境構築
```bash
conda create -n dpo_training python=3.10
conda activate dpo_training
pip install -r requirements.txt
```

### 3. GPU環境確認
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"
```

### 4. 設定ファイル調整
移行先のハードウェアに応じて `configs/tiny_swallow_config.yaml` を調整：

```yaml
# GPU環境用設定
hardware:
  use_mps: false  # macOS以外ではfalse
  use_cuda: true  # NVIDIA GPU使用時
  memory_efficient: false  # 高性能環境では無効化可能

# バッチサイズ調整
training:
  per_device_train_batch_size: 4  # GPUメモリに応じて調整
  per_device_eval_batch_size: 4
  gradient_accumulation_steps: 2
```

### 5. 訓練実行
```bash
python scripts/training/tiny_swallow_dpo_training.py
```

## 既知の問題と解決策

### TRL DPOTrainer API問題
TRL 0.18.1では一部のパラメータが変更されています。現在のスクリプトには互換性対応コードが含まれています。

### データセット形式
DPOデータセットは以下の形式で準備済み：
```json
{"prompt": "質問", "chosen": "良い回答", "rejected": "悪い回答"}
```

### モデルファイル
SakanaAI/TinySwallow-1.5B-Instructは初回実行時に自動ダウンロードされます（約3.1GB）。

## ログファイル分析

### 最新の実行ログ
主要なログファイル：
- `outputs/logs/tinyswallow_dpo_20250612_150019.log` - 最新実行
- `outputs/logs/tinyswallow_dpo_20250612_145629.log` - DPOTrainer初期化エラー詳細

### エラー詳細
```
DPOTrainer.__init__() got an unexpected keyword argument 'tokenizer'
```
→ TRL 0.18.1のAPI変更による互換性問題

## 期待される結果

### 訓練完了時の成果物
- **学習済みモデル**: `outputs/tiny_swallow_dpo/final_model/`
- **チェックポイント**: 100ステップごとの中間保存
- **評価結果**: 広告技術分野での回答品質向上

### 予想される訓練時間
- **高性能GPU**: 2-4時間
- **中性能GPU**: 6-8時間
- **CPU専用**: 24時間以上（非推奨）

## 次のステップ

1. **環境移行**: 高性能PCでの環境構築
2. **訓練実行**: DPO訓練の完全実行
3. **モデル評価**: 広告技術分野での性能評価
4. **結果分析**: 訓練前後の比較分析

## 連絡先・参考情報

### 重要ファイル
- **設定**: `configs/tiny_swallow_config.yaml`
- **データ**: `data/dpo_dataset.jsonl`
- **スクリプト**: `scripts/training/tiny_swallow_dpo_training.py`
- **ログ**: `outputs/logs/`

### トラブルシューティング
問題が発生した場合は、以下のテストスクリプトで環境確認：
- `scripts/testing/test_tinyswallow.py`
- `scripts/testing/test_dpo_trainer_api.py`

---
**作成日時**: 2025年6月12日  
**対象環境**: MacBook Air M2 → 高性能PC  
**ステータス**: 移行準備完了
