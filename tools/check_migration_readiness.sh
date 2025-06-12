#!/bin/bash

# =============================================================================
# .venv から conda への完全移行スクリプト
# =============================================================================

echo "🔄 .venv から conda への移行を開始します..."

# 現在のディレクトリを確認
PROJECT_DIR="/Users/hiroto/Documents/AD_Tech_SLM"
cd "$PROJECT_DIR"

echo "📍 作業ディレクトリ: $(pwd)"

# Step 1: conda環境の状況確認
echo ""
echo "📋 Step 1: conda環境の確認"
echo "========================="
conda info --envs

# Step 2: dpo_training環境がアクティブかチェック
echo ""
echo "📋 Step 2: 現在の環境確認"
echo "========================="
echo "現在の Python パス: $(which python)"

# Step 3: .venv の使用状況を確認
echo ""
echo "📋 Step 3: .venv の現在の使用状況"
echo "================================"
if [ -d ".venv" ]; then
    echo "✅ .venv ディレクトリが存在します"
    echo "📊 .venv サイズ: $(du -sh .venv | cut -f1)"
else
    echo "❌ .venv ディレクトリが見つかりません"
fi

# Step 4: プロジェクト内のPythonスクリプトで.venvを参照している箇所をチェック
echo ""
echo "📋 Step 4: .venv 参照箇所の確認"
echo "=============================="
echo "🔍 .venv を参照するファイルを検索中..."

# Python scripts
find . -name "*.py" -exec grep -l "\.venv\|venv" {} \; 2>/dev/null | head -10

# Shell scripts  
find . -name "*.sh" -exec grep -l "\.venv\|venv" {} \; 2>/dev/null | head -10

# Config files
find . -name "*.yml" -o -name "*.yaml" -o -name "*.json" -exec grep -l "\.venv\|venv" {} \; 2>/dev/null | head -10

echo ""
echo "📋 Step 5: 移行準備完了確認"
echo "=========================="
echo "✅ conda環境 'dpo_training' の状況:"
conda list -n dpo_training | wc -l | awk '{print "   インストール済みパッケージ数: " $1}'

echo ""
echo "🎯 移行準備が完了しました！"
echo "次のステップ:"
echo "   1. tools/migrate_to_conda.sh を実行して実際の移行を開始"
echo "   2. すべてのスクリプトをconda環境で動作確認"
echo "   3. 問題がなければ .venv ディレクトリを削除"
