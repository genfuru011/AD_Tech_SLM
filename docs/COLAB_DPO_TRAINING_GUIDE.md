# 🦆 TinySwallow DPO Training - Improved Colab Notebook

## 概要

このノートブックは、SakanaAI/TinySwallow-1.5B-Instructモデルに対して、Direct Preference Optimization (DPO) を使用した効率的なファインチューニングを Google Colab 環境で実行するための改良版です。

## 🎯 主な改良点

### 1. TRL API 互換性対応
- **TRL 0.18.1 対応**: 最新APIの変更に完全対応
- **フォールバック機能**: 新旧APIの両方をサポート
- **エラーハンドリング**: 堅牢なエラー処理とリカバリ

### 2. モデル読み込みの強化
- **複数モデル対応**: メインモデルが利用できない場合のバックアップモデル
- **自動フォールバック**: モデル読み込み失敗時の自動切り替え
- **メモリ最適化**: 4bit量子化による効率的なメモリ使用

### 3. データセット処理の改善
- **柔軟な入力方式**: ファイルアップロード、サンプルデータ使用
- **データ検証**: 入力データの形式チェックとエラー処理
- **統計情報表示**: データセットの詳細分析

### 4. トレーニング設定の最適化
- **Colab環境最適化**: GPU制限を考慮したパラメータ設定
- **LoRA設定**: Parameter Efficient Fine-tuning による効率化
- **勾配チェックポイント**: メモリ使用量の削減

## 📚 参考資料

### 公式ドキュメント
- [TRL Documentation](https://huggingface.co/docs/trl/) - Transformers Reinforcement Learning
- [DPO Paper](https://arxiv.org/abs/2305.18290) - Direct Preference Optimization
- [Transformers Documentation](https://huggingface.co/docs/transformers/) - Hugging Face Transformers
- [PEFT Documentation](https://huggingface.co/docs/peft/) - Parameter Efficient Fine-tuning

### 技術リソース
- [BitsAndBytes Documentation](https://huggingface.co/docs/bitsandbytes/) - 量子化ライブラリ
- [Accelerate Documentation](https://huggingface.co/docs/accelerate/) - 分散学習ライブラリ
- [LoRA Paper](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation手法

## 🛠️ 使用方法

### 1. Google Colab での実行

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/genfuru011/AD_Tech_SLM/blob/main/notebooks/colab_dpo_training_improved.ipynb)

1. **GPU ランタイムの設定**
   - ランタイム → ランタイムのタイプを変更
   - ハードウェア アクセラレータ: GPU (T4, V100, A100推奨)

2. **依存関係のインストール**
   - ノートブックの指示に従って必要ライブラリをインストール

3. **データセットの準備**
   - 方法1: DPOデータセット（JSONL形式）をアップロード
   - 方法2: 提供されるサンプルデータを使用

4. **トレーニング実行**
   - 設定を確認してトレーニングを開始

### 2. データセット形式

```jsonl
{"prompt": "質問やタスクの説明", "chosen": "高品質な回答", "rejected": "低品質な回答"}
{"prompt": "プログラマティック広告とは？", "chosen": "詳細で正確な説明...", "rejected": "簡潔すぎる回答"}
```

## ⚙️ 設定パラメータ

### モデル設定
- **メインモデル**: `SakanaAI/TinySwallow-1.5B-Instruct`
- **バックアップモデル**: 代替モデルの自動選択
- **量子化**: 4bit NF4 量子化によるメモリ効率化

### トレーニング設定
- **学習率**: `5e-7` (安定した学習)
- **バッチサイズ**: `1` (メモリ制限対応)
- **勾配蓄積**: `8` (実質バッチサイズ 8)
- **最大ステップ**: `500` (Colab時間制限対応)

### LoRA設定
- **ランク**: `16` (バランスの取れた性能)
- **アルファ**: `32` (適切な学習強度)
- **対象モジュール**: `q_proj, v_proj, k_proj, o_proj`

## 🔧 トラブルシューティング

### よくある問題と解決策

#### 1. メモリ不足エラー
```
CUDA out of memory
```
**解決策:**
- バッチサイズを1に減少
- 勾配蓄積ステップを増加
- より小さなモデルを使用

#### 2. TRL API エラー
```
DPOTrainer.__init__() got an unexpected keyword argument 'tokenizer'
```
**解決策:**
- ノートブックが自動的にレガシーAPIにフォールバック
- 最新のTRLバージョンを確認

#### 3. モデル読み込みエラー
```
Model not found or access denied
```
**解決策:**
- バックアップモデルが自動的に選択される
- Hugging Face認証が必要な場合あり

### デバッグのヒント

1. **GPU使用量の確認**
   ```python
   import torch
   print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
   ```

2. **ライブラリバージョンの確認**
   ```python
   import transformers, trl
   print(f"Transformers: {transformers.__version__}")
   print(f"TRL: {trl.__version__}")
   ```

## 📊 期待される結果

### トレーニング指標
- **訓練ロス**: 段階的な減少
- **検証ロス**: 過学習の回避
- **学習率**: 適切なスケジューリング

### モデル性能
- **応答品質**: より詳細で正確な回答
- **専門知識**: 広告技術分野の知識向上
- **一貫性**: 安定した応答品質

## 🚀 次のステップ

### 1. 高度な評価
```python
# BLEU, ROUGE スコアの計算
# 人間評価の実施
# A/Bテストの設計
```

### 2. プロダクション展開
```python
# FastAPI による API化
# Docker コンテナ化
# スケーラブルな推論環境
```

### 3. 継続的改善
```python
# 追加データでの継続学習
# ハイパーパラメータ最適化
# より大規模なモデルへの適用
```

## 📞 サポート

### 問題報告
- [GitHub Issues](https://github.com/genfuru011/AD_Tech_SLM/issues)

### 貢献方法
- Pull Request の提出
- ドキュメントの改善
- バグ報告と修正

## 📄 ライセンス

このプロジェクトは元のリポジトリのライセンスに従います。

---

**🦆 TinySwallow DPO Training - Improved Version**  
*Robust, Efficient, and Production-Ready DPO Training for Google Colab*