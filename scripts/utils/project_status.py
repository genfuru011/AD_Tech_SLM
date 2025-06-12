#!/opt/miniconda3/envs/dpo_training/bin/python
"""
プロジェクト状況確認スクリプト
ディレクトリ構造、ファイル配置、トレーニング進捗を総合的に確認
"""

import os
import json
import subprocess
from pathlib import Path

def check_project_structure():
    """プロジェクト構造の確認"""
    print("🏗️ プロジェクト構造確認")
    print("=" * 60)
    
    expected_dirs = [
        "docs", "scripts", "notebooks", "configs", 
        "data", "outputs", "tools"
    ]
    
    expected_subdirs = {
        "scripts": ["training", "testing", "utils"],
        "outputs": ["logs"]
    }
    
    for dir_name in expected_dirs:
        if os.path.exists(dir_name):
            print(f"✅ {dir_name}/")
            
            # サブディレクトリチェック
            if dir_name in expected_subdirs:
                for subdir in expected_subdirs[dir_name]:
                    subdir_path = os.path.join(dir_name, subdir)
                    if os.path.exists(subdir_path):
                        print(f"   ✅ {subdir}/")
                    else:
                        print(f"   ❌ {subdir}/ (missing)")
        else:
            print(f"❌ {dir_name}/ (missing)")

def check_key_files():
    """重要ファイルの存在確認"""
    print("\n📄 重要ファイル確認")
    print("=" * 60)
    
    key_files = {
        "README.md": "メインドキュメント",
        "requirements.txt": "依存パッケージ",
        "configs/dpo_config.yaml": "DPO設定",
        "data/dpo_dataset.jsonl": "トレーニングデータ",
        "scripts/training/conda_dpo_training.py": "メインDPOトレーニング",
        "scripts/utils/check_progress.py": "進捗確認",
        "notebooks/colab_dpo_training.ipynb": "Colabノートブック",
        "docs/README_DPO_TRAINING.md": "DPOガイド",
        "docs/README_COLAB.md": "Colabガイド"
    }
    
    for file_path, description in key_files.items():
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / 1024  # KB
            print(f"✅ {file_path} ({size:.1f} KB) - {description}")
        else:
            print(f"❌ {file_path} - {description}")

def check_training_progress():
    """DPOトレーニング進捗確認"""
    print("\n📊 DPOトレーニング進捗")
    print("=" * 60)
    
    # チェックポイント確認
    outputs_dir = "./outputs"
    checkpoints = []
    
    if os.path.exists(outputs_dir):
        for item in os.listdir(outputs_dir):
            if item.startswith("checkpoint-") and os.path.isdir(os.path.join(outputs_dir, item)):
                try:
                    step = int(item.split("-")[1])
                    checkpoints.append(step)
                except ValueError:
                    continue
    
    if checkpoints:
        checkpoints.sort()
        latest_step = checkpoints[-1]
        print(f"🎯 作成済みチェックポイント: {checkpoints}")
        print(f"📈 最新ステップ: {latest_step}")
        
        # 最新チェックポイントの詳細
        latest_checkpoint_path = f"./outputs/checkpoint-{latest_step}/trainer_state.json"
        
        if os.path.exists(latest_checkpoint_path):
            with open(latest_checkpoint_path, 'r') as f:
                state = json.load(f)
            
            current_step = state['global_step']
            max_steps = state['max_steps']
            epoch = state['epoch']
            progress = (current_step / max_steps) * 100
            
            print(f"📊 進捗: {current_step}/{max_steps} ({progress:.1f}%)")
            print(f"🔄 エポック: {epoch:.3f}")
            
            # 推定残り時間
            remaining_steps = max_steps - current_step
            estimated_hours = remaining_steps * 0.002  # 概算
            print(f"⏱️ 推定残り時間: 約{estimated_hours:.1f}時間")
    else:
        print("❌ チェックポイントが見つかりません")
    
    # プロセス確認
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        dpo_processes = [line for line in result.stdout.split('\n') 
                        if 'conda_dpo_training.py' in line]
        
        if dpo_processes:
            print(f"✅ DPOトレーニング実行中 ({len(dpo_processes)} プロセス)")
        else:
            print("⚠️ DPOトレーニングプロセスが見つかりません")
    except:
        print("⚠️ プロセス状況を確認できませんでした")

def check_git_status():
    """Git状況確認"""
    print("\n📝 Git状況")
    print("=" * 60)
    
    try:
        # Git状態確認
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            changes = result.stdout.strip().split('\n') if result.stdout.strip() else []
            if changes:
                print(f"📝 変更されたファイル: {len(changes)}件")
                for change in changes[:5]:  # 最初の5件のみ表示
                    print(f"   {change}")
                if len(changes) > 5:
                    print(f"   ... 他{len(changes) - 5}件")
            else:
                print("✅ 変更なし（クリーン状態）")
        else:
            print("❌ Gitリポジトリではありません")
    except:
        print("⚠️ Git状況を確認できませんでした")

def show_next_steps():
    """次のステップを表示"""
    print("\n🚀 次のステップ")
    print("=" * 60)
    print("1. 📊 トレーニング進捗監視:")
    print("   python scripts/utils/check_progress.py")
    print()
    print("2. 🧪 中間モデルテスト:")
    print("   python scripts/testing/test_intermediate_model.py")
    print()
    print("3. ☁️ Google Colabで実行:")
    print("   notebooks/colab_dpo_training.ipynb")
    print()
    print("4. 📚 詳細ドキュメント:")
    print("   docs/README_DPO_TRAINING.md")
    print("   docs/README_COLAB.md")

def main():
    print("🔍 AD_Tech_SLM プロジェクト状況確認")
    print("=" * 80)
    
    # 各種確認実行
    check_project_structure()
    check_key_files()
    check_training_progress()
    check_git_status()
    show_next_steps()
    
    print("\n" + "=" * 80)
    print("✅ プロジェクト状況確認完了！")

if __name__ == "__main__":
    main()
