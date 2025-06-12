#!/bin/bash

# =============================================================================
# conda環境への完全移行実行スクリプト
# =============================================================================

set -e  # エラーが発生したら停止

echo "🚀 conda環境への完全移行を実行します"
echo "=================================="

PROJECT_DIR="/Users/hiroto/Documents/AD_Tech_SLM"
cd "$PROJECT_DIR"

# conda環境をアクティベート
echo "📦 conda環境 'dpo_training' をアクティベート中..."
eval "$(conda shell.bash hook)"
conda activate dpo_training

echo "✅ アクティブな環境: $CONDA_DEFAULT_ENV"
echo "✅ Python パス: $(which python)"

# Step 1: 環境変数やスクリプト内の.venv参照を更新
echo ""
echo "📋 Step 1: スクリプト内の.venv参照を更新"
echo "======================================="

# VS Code設定の更新 (.vscode/settings.json)
if [ ! -d ".vscode" ]; then
    mkdir -p .vscode
fi

cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "/opt/miniconda3/envs/dpo_training/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.condaPath": "/opt/miniconda3/bin/conda",
    "python.terminal.activateEnvInCurrentTerminal": true,
    "jupyter.kernels.filter": [
        {
            "path": "/opt/miniconda3/envs/dpo_training/bin/python",
            "type": "pythonEnvironment"
        }
    ]
}
EOF

echo "✅ VS Code設定を更新しました (.vscode/settings.json)"

# Step 2: シェルスクリプトの更新
echo ""
echo "📋 Step 2: シェルスクリプトの更新"
echo "==============================="

# setup.sh の更新
cat > tools/setup.sh << 'EOF'
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
EOF

chmod +x tools/setup.sh
echo "✅ tools/setup.sh を更新しました"

# quick_start.sh の更新
cat > tools/quick_start.sh << 'EOF'
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
EOF

chmod +x tools/quick_start.sh
echo "✅ tools/quick_start.sh を更新しました"

# Step 3: Python スクリプトのshebang行確認と更新
echo ""
echo "📋 Step 3: Python スクリプトの確認"
echo "==============================="

# 主要なPythonスクリプトのshebang行を確認・更新
update_python_shebang() {
    local file=$1
    if [ -f "$file" ] && [ -s "$file" ]; then
        # ファイルが存在し、空でない場合のみ処理
        if head -1 "$file" | grep -q "#!/.*python"; then
            echo "🔧 $file のshebang行を更新中..."
            sed -i '' '1s|#!/.*python.*|#!/opt/miniconda3/envs/dpo_training/bin/python|' "$file"
        fi
    fi
}

# 主要ファイルの更新
echo "🔍 Python スクリプトのshebang行を確認中..."
find scripts/ -name "*.py" -type f | while read file; do
    update_python_shebang "$file"
done

echo "✅ Python スクリプトの更新完了"

# Step 4: 環境テスト
echo ""
echo "📋 Step 4: 移行後の環境テスト"
echo "=========================="

echo "🧪 Python環境テスト:"
python -c "
import sys
print(f'Python パス: {sys.executable}')
print(f'Python バージョン: {sys.version}')

try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    if torch.backends.mps.is_available():
        print('✅ MPS (Metal Performance Shaders) 利用可能')
    else:
        print('⚠️  MPS 利用不可 (CPUモードで動作)')
except ImportError as e:
    print(f'❌ PyTorch インポートエラー: {e}')

try:
    import transformers
    print(f'Transformers: {transformers.__version__}')
except ImportError as e:
    print(f'❌ Transformers インポートエラー: {e}')

try:
    import trl
    print(f'TRL: {trl.__version__}')
except ImportError as e:
    print(f'❌ TRL インポートエラー: {e}')
"

echo ""
echo "🎉 conda環境への移行が完了しました！"
echo ""
echo "📋 移行後の確認事項:"
echo "   ✅ VS Code Python インタープリター設定済み"
echo "   ✅ 全スクリプトがconda環境を使用"
echo "   ✅ 環境テスト完了"
echo ""
echo "🚮 .venvディレクトリの削除:"
echo "   問題がないことを確認後、以下のコマンドで削除してください:"
echo "   rm -rf .venv"
echo ""
echo "🔄 環境の使用方法:"
echo "   conda activate dpo_training"
echo "   python scripts/training/conda_dpo_training.py"
EOF

chmod +x tools/migrate_to_conda.sh
