# 🚨 Google Colab 依存関係エラー対応ガイド

Google ColabでDPOトレーニング実行時に発生する依存関係エラーの完全解決ガイドです。

## 🔍 よくあるエラーメッセージ

### 1. FastAI + PyTorch 競合
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. 
This behaviour is the source of the following dependency conflicts.
fastai 2.7.19 requires torch<2.7,>=1.10, but you have torch 2.7.1 which is incompatible.
```

### 2. fsspec + gcsfs バージョン競合
```
ERROR: gcsfs 2025.3.2 requires fsspec==2025.3.2, but you have fsspec 2025.3.0 which is incompatible.
```

## 🛠️ 解決方法

### 方法1: ノートブック内クイックフィックス（推奨）

1. **Runtime → Restart runtime** をクリック
2. ノートブックの「緊急クイックフィックス」セルで以下を変更：
   ```python
   QUICK_FIX = True  # ← Falseから変更
   ```
3. そのセルを実行
4. 再度メインのセットアップセルを実行

### 方法2: 手動コマンド実行

#### ステップ1: 問題パッケージの削除
```python
!pip uninstall -y fastai torch torchvision torchaudio gcsfs fsspec --quiet
```

#### ステップ2: 互換性のあるバージョンでインストール
```python
# PyTorchを互換性のあるバージョンで
!pip install 'torch>=2.0,<2.7' 'torchvision' 'torchaudio' --index-url https://download.pytorch.org/whl/cu118

# fsspecとgcsfsを統一バージョンで
!pip install fsspec==2025.3.2 gcsfs==2025.3.2

# fastaiを最後に
!pip install 'fastai>=2.7.0,<2.8'
```

#### ステップ3: DPO関連パッケージ
```python
!pip install transformers datasets accelerate tokenizers trl peft evaluate
```

### 方法3: 代替インストール（完全リセット）

ノートブックの「代替インストール手法」セルで：
```python
ALTERNATIVE_INSTALL = True  # ← Falseから変更
```

**注意：この方法を使用する前に必ずランタイムを再起動してください**

## 🔍 インストール確認

インストール後、以下で確認：

```python
import torch
import transformers
import trl
import peft
import fastai

print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"FastAI: {fastai.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
```

## 🎯 エラー別対応表

| エラー | 原因 | 解決法 |
|--------|------|--------|
| `torch 2.7.1 incompatible` | PyTorchが新しすぎる | torch<2.7でインストール |
| `fsspec version mismatch` | fsspecとgcsfsのバージョン不一致 | 両方を==2025.3.2で統一 |
| `ImportError: No module` | パッケージ未インストール | pip install再実行 |
| `CUDA not available` | GPUランタイム未選択 | Runtime→Change runtime type→GPU |

## 🚀 成功の確認

以下が表示されれば成功：
```
🚀 GPU Available: NVIDIA A100-SXM4-40GB
💾 GPU Memory: 42.5 GB
🔥 PyTorch CUDA Version: 11.8
✅ All critical packages imported successfully!
```

## 📝 予防策

1. **ランタイム再起動**: エラー後は必ず `Runtime → Restart runtime`
2. **段階的インストール**: 一度に全パッケージをインストールしない
3. **バージョン固定**: 重要なパッケージは具体的なバージョンを指定
4. **互換性確認**: 新しいパッケージ追加前に既存との互換性をチェック

## 🆘 それでも解決しない場合

1. **完全リセット**: 新しいColabノートブックを作成
2. **Pro版使用**: Colab Pro/Pro+でより安定した環境を利用
3. **ローカル環境**: 依存関係管理が容易なローカル環境での実行を検討

---

**💡 ヒント**: 依存関係エラーは主にColabの既存パッケージとの競合が原因です。ランタイム再起動とクリーンインストールで大部分は解決できます。
