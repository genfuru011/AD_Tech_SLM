#!/bin/bash

# =============================================================================
# AD_Tech_SLM プロジェクト環境セットアップ (conda版)
# =============================================================================

echo "🚀 AD_Tech_SLM 環境セットアップを開始します (conda使用)"

# condaの確認
if ! command -v conda &> /dev/null; then
    echo "❌ conda が見つかりません。Miniconda または Anaconda をインストールしてください。"
    exit 1
fi

echo "✅ conda が利用可能です"

# dpo_training環境の存在確認
if conda env list | grep -q "dpo_training"; then
    echo "✅ dpo_training環境が既に存在します"
    echo "🔄 環境をアクティベートします..."
    eval "$(conda shell.bash hook)"
    conda activate dpo_training
else
    echo "📦 dpo_training環境を作成します..."
    conda create -n dpo_training python=3.11 -y
    eval "$(conda shell.bash hook)"
    conda activate dpo_training
    
    echo "📦 必要なパッケージをインストールします..."
    pip install -r requirements.txt
fi

echo ""
echo "✅ セットアップ完了！"
echo "📝 使用方法:"
echo "   conda activate dpo_training"
echo "   python scripts/training/conda_dpo_training.py"
