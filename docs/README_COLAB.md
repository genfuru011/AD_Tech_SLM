# 🚀 DPO Training on Google Colab

Google ColabでDPO（Direct Preference Optimization）トレーニングを実行するための完全なガイドです。

## 📂 ファイル構成

```
colab_dpo_training.ipynb          # メインのColabノートブック
dpo_training_experiment.ipynb     # 詳細な実験用ノートブック
README_COLAB.md                   # このファイル
```

## 🚀 クイックスタート

### 1. Google Colabで開く

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. `colab_dpo_training.ipynb` をGoogle Colabにアップロード
2. ランタイムタイプをGPUに変更
3. セルを順番に実行

### 2. 推奨設定

- **ランタイム**: GPU (T4/V100推奨)
- **RAM**: 標準 (12.7GB)
- **ディスク**: 標準
- **実行時間**: 約15-30分

## 📊 特徴

- ✅ **自動環境構築**: 必要なパッケージを自動インストール
- ✅ **軽量モデル**: ColabのGPUメモリに最適化
- ✅ **効率的学習**: LoRAによる効率的な微調整
- ✅ **リアルタイム監視**: 学習進捗をリアルタイム表示
- ✅ **結果可視化**: 損失グラフとメトリクス表示
- ✅ **自動保存**: Google Driveへの自動保存

## 🔧 技術仕様

### モデル
- **ベースモデル**: `microsoft/DialoGPT-small` (117M parameters)
- **微調整手法**: LoRA (Low-Rank Adaptation)
- **最適化手法**: DPO (Direct Preference Optimization)

### データセット
- **サンプル数**: 1,000件（自動生成）
- **形式**: prompt, chosen, rejected
- **言語**: 日本語
- **分野**: AI、プログラミング、データサイエンス

### トレーニング設定
- **エポック数**: 1
- **バッチサイズ**: 1 (gradient_accumulation_steps=4)
- **学習率**: 5e-6
- **最大長**: 128トークン

## 📈 期待される結果

### 学習メトリクス
- **初期損失**: 約0.7
- **最終損失**: 約0.1-0.3
- **精度**: 80-95%
- **GPU使用量**: 2-3GB

### 実行時間
- **セットアップ**: 2-3分
- **トレーニング**: 10-20分
- **評価・保存**: 2-3分
- **合計**: 約15-30分

## 🛠️ トラブルシューティング

### よくある問題と解決法

#### 1. GPU メモリ不足
```python
# メモリクリア
torch.cuda.empty_cache()

# バッチサイズを減らす
per_device_train_batch_size=1
gradient_accumulation_steps=2
```

#### 2. パッケージエラー
```bash
# 最新版を再インストール
!pip install --upgrade transformers trl peft
```

#### 3. 依存関係エラー (fastai, fsspec)
```bash
# fastaiとtorchの競合エラー
ERROR: fastai 2.7.19 requires torch<2.7,>=1.10, but you have torch 2.7.1 which is incompatible.

# fsspec関連のエラー
ERROR: gcsfs 2025.3.2 requires fsspec==2025.3.2, but you have fsspec 2025.3.0 which is incompatible.

# 📝 解決手順:
# 1. Runtime → Restart runtime をクリック
# 2. ノートブックの「緊急クイックフィックス」セルで QUICK_FIX = True に設定
# 3. または手動で以下を実行:

!pip uninstall -y fastai torch gcsfs fsspec
!pip install 'torch>=2.0,<2.7' --index-url https://download.pytorch.org/whl/cu118
!pip install fsspec==2025.3.2 gcsfs==2025.3.2
!pip install 'fastai>=2.7.0'
```
!pip install fsspec==2025.3.2
!pip install gcsfs==2025.3.2
```

#### 4. 学習が進まない
```python
# 学習率を調整
learning_rate=1e-5  # より小さく

# 勾配クリッピング
max_grad_norm=1.0
```

## 📥 モデルの使用方法

### 1. Colabでの推論
```python
# トレーニング済みモデルの推論
response = generate_response("質問: AIとは何ですか？")
print(response)
```

### 2. ローカルでの使用
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ベースモデル読み込み
base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")

# LoRAアダプター適用
model = PeftModel.from_pretrained(base_model, "./path/to/saved/model")

# 推論実行
inputs = tokenizer.encode("質問:", return_tensors="pt")
outputs = model.generate(inputs, max_length=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## 📚 参考資料

### DPO関連
- [DPO論文](https://arxiv.org/abs/2305.18290)
- [TRL Documentation](https://huggingface.co/docs/trl/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)

### Colab関連
- [Google Colab使い方](https://colab.research.google.com/)
- [GPU使用方法](https://colab.research.google.com/notebooks/gpu.ipynb)

## 🤝 貢献とサポート

### 改良提案
- より効率的なデータセット生成
- 異なるモデルサイズでの実験
- 評価指標の追加

### 問題報告
GitHub Issues または以下で報告してください：
- モデルの性能問題
- メモリ使用量の最適化
- Colab環境での互換性問題

## 📝 ライセンス

MITライセンスの下で公開されています。

---

## 🎯 次のステップ

1. **基本実行**: `colab_dpo_training.ipynb` で基本的なDPOトレーニングを体験
2. **詳細実験**: `dpo_training_experiment.ipynb` でより詳細な実験を実行
3. **カスタマイズ**: 独自のデータセットやモデルでの実験
4. **本格運用**: より大きなモデルや本格的なデータセットでの学習

Happy Training! 🚀
