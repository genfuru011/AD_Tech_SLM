# 🚀 AD_Tech_SLM

広告特化型 SLM（Small Language Model）開発・運用プロジェクト

[![DPO Training](https://img.shields.io/badge/DPO-Training-blue)](./docs/README_DPO_TRAINING.md)
[![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](./notebooks/colab_dpo_training.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

## 📋 プロジェクト概要

本プロジェクトは、**"広告コピーの品質向上"** と **"CTR（クリック率）改善"** を目指す、広告業界向けの特化型言語モデルを開発・運用するものです。

DPO（Direct Preference Optimization）手法を用いて、人間の選好を直接学習し、より魅力的で効果的な広告コピーを生成できるモデルを構築します。

## 📁 プロジェクト構造

```
AD_Tech_SLM/
├── 📚 docs/                    # ドキュメント
│   ├── README_DPO_TRAINING.md # DPOトレーニングガイド
│   └── README_COLAB.md        # Google Colabガイド
├── 🎯 scripts/                # 実行スクリプト
│   ├── training/              # トレーニング関連
│   ├── testing/               # テスト関連
│   └── utils/                 # ユーティリティ
├── 📓 notebooks/              # Jupyterノートブック
├── ⚙️ configs/                # 設定ファイル
├── 📊 data/                   # データセット
├── 📈 outputs/                # 出力結果・チェックポイント
└── 🛠️ tools/                  # 開発ツール
```

## 🎯 ロードマップ

- ✅ **Phase 1**: DPO手法による基本トレーニング環境構築
- ✅ **Phase 2**: Google Colab対応・簡易実行環境
- 🔄 **Phase 3**: 本格的なDPOトレーニング実行中
- 🔜 **Phase 4**: モデル評価・本番デプロイ

## 🚀 クイックスタート

### 🌐 Google Colab（推奨・初心者向け）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](./notebooks/colab_dpo_training.ipynb)

1. **上記のColabリンクをクリック**
2. **ランタイム設定**: GPU（T4推奨）を選択
3. **データセット準備**: 以下の3つの方法から選択
   - 📁 GitHub経由でデータセット取得
   - ☁️ Google Drive経由でアップロード
   - 📤 直接ファイルアップロード
4. **トレーニング実行**: セルを順次実行
5. **モデルテスト**: 学習済みモデルの性能確認

📋 **詳細ガイド**: [Google Colab使用方法](./docs/README_COLAB.md) | [データセット活用ガイド](./docs/DATASET_USAGE_GUIDE.md)

### 🎯 ローカル環境でのDPOトレーニング

```bash
# 1. 環境セットアップ
./tools/setup.sh

# 2. DPOトレーニング開始
python scripts/training/conda_dpo_training.py

# 3. 進捗監視
python scripts/utils/check_progress.py
```

### ☁️ Google Colabで簡単実行

1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](./notebooks/colab_dpo_training.ipynb) をクリック
2. ランタイムタイプをGPUに変更
3. セルを順番に実行（約15-30分で完了）

### 📊 現在のトレーニング進捗

- **進捗**: 500/2,139 ステップ (23.4%)
- **推定残り時間**: 約1.6時間
- **使用手法**: DPO (Direct Preference Optimization)
- **モデル**: GPT-2 + LoRA

## 🔬 DPO手法について

### 人間のフィードバックを活用（RLHF, DPO手法）

DPOは人間の選好を直接学習する手法で、以下の特徴があります：

- **直接最適化**: 人間のフィードバックを直接モデルに反映
- **効率的学習**: RLHFより簡単で安定した学習
- **主観的品質**: トーン、スタイル、表現力の向上
- **広告特化**: CTR向上に直結する選好学習

![DPO概念図](https://github.com/user-attachments/assets/a0c9bf1c-a7e9-4ead-a18d-dc47a0b874f1)

### 🗂️ 必要データセット形式（JSONL）

```json
{
  "prompt": "【テーマ】雨の日でもワクワクするニュースアプリを紹介してください", 
  "chosen": "雨が降っても最新トレンドをスマホでサクッとチェック！天気と話題を同時にキャッチして、移動中も退屈知らず♪", 
  "rejected": "ニュースが見られるよ！便利！"
}
```

- **prompt**: コピーを書かせたいお題や前提条件
- **chosen**: CTRが高かった"良い広告テキスト"
- **rejected**: CTRが低かった、または不採用になった文

## 📚 ドキュメント

- [DPOトレーニング詳細ガイド](./docs/README_DPO_TRAINING.md)
- [Google Colab実行ガイド](./docs/README_COLAB.md)

## 🛠️ 開発・貢献

### 前提条件

- Python 3.10+
- PyTorch 2.0+
- CUDA対応GPU または Apple Silicon (MPS)

### セットアップ

```bash
git clone https://github.com/your-repo/AD_Tech_SLM.git
cd AD_Tech_SLM
./tools/setup.sh
```

### テスト実行

```bash
# 基本テスト
python scripts/testing/test_model_loading.py

# データセットテスト
python scripts/testing/test_data.py

# 推論テスト
python scripts/testing/quick_model_test.py
```

## 📄 ライセンス

このプロジェクトは [MIT License](./LICENSE) の下で公開されています。

## 🤝 コントリビューション

プルリクエストやイシューの投稿を歓迎します！

1. フォークして機能ブランチを作成
2. 変更をコミット
3. プルリクエストを作成

---

**Happy Coding! 🎉**
