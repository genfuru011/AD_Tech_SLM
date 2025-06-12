#!/bin/bash

# =============================================================================
# AD_Tech_SLM クイックスタート (conda版)
# =============================================================================

echo "🚀 AD_Tech_SLM クイックスタート (conda環境)"

# conda環境のアクティベート
echo "📦 conda環境をアクティベート中..."
eval "$(conda shell.bash hook)"

if conda env list | grep -q "dpo_training"; then
    conda activate dpo_training
    echo "✅ dpo_training環境をアクティベートしました"
else
    echo "❌ dpo_training環境が見つかりません"
    echo "🔧 tools/setup.sh を実行して環境を作成してください"
    exit 1
fi

echo "🔍 現在の環境:"
echo "   Python: $(which python)"
echo "   Conda環境: $CONDA_DEFAULT_ENV"

echo ""
echo "🎯 利用可能なスクリプト:"
echo "   📊 DPOトレーニング: python scripts/training/conda_dpo_training.py"
echo "   📈 進捗確認: python scripts/utils/check_progress.py"
echo "   🧪 モデルテスト: python scripts/testing/test_intermediate_model.py"

echo ""
echo "💡 次のステップ:"
echo "   1. python scripts/training/conda_dpo_training.py でトレーニング開始"
echo "   2. 別のターミナルで python scripts/utils/monitor_training.py で監視"
