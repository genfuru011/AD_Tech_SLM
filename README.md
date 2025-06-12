# TinySwallow-1.5B DPO Training for Ad Tech

## 概要
SakanaAI/TinySwallow-1.5B-Instructモデルの広告技術分野特化DPO（Direct Preference Optimization）訓練プロジェクト。

## 🚀 クイックスタート

### 環境セットアップ
```bash
conda create -n dpo_training python=3.10
conda activate dpo_training
pip install -r requirements.txt
```

### 訓練実行
```bash
python scripts/training/tiny_swallow_dpo_training.py
```

## 📁 プロジェクト構造
```
AD_Tech_SLM/
├── configs/
│   └── tiny_swallow_config.yaml    # TinySwallow用設定
├── data/
│   └── dpo_dataset.jsonl          # DPO訓練データ（3,565サンプル）
├── docs/
│   └── HIGH_PERFORMANCE_MIGRATION.md  # 高性能PC移行ガイド
├── notebooks/
│   └── colab_dpo_training.ipynb    # Google Colabノートブック
├── outputs/
│   └── logs/                       # 訓練ログ
├── scripts/
│   ├── testing/
│   │   └── test_tinyswallow.py     # モデル検証
│   └── training/
│       └── tiny_swallow_dpo_training.py  # メイン訓練スクリプト
└── requirements.txt                # 依存関係
```

## 🎯 技術仕様
- **モデル**: SakanaAI/TinySwallow-1.5B-Instruct (15.48億パラメータ)
- **データセット**: 広告技術分野DPOデータ（3,208訓練 + 357検証）
- **手法**: DPO + LoRA
- **対象**: MacBook Air M2 8GB → 高性能PC移行

## ⚠️ 重要事項
- MacBook Air M2 8GBではメモリ不足のため、高性能PC（VRAM 12GB以上推奨）での実行が必要
- 移行手順は `docs/HIGH_PERFORMANCE_MIGRATION.md` を参照

## 📊 期待される成果
- 広告技術分野での日本語応答品質向上
- プログラマティック広告、RTB、DMPなどの専門知識強化
