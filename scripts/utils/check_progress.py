#!/usr/bin/env python3
"""
DPO進捗の簡単確認スクリプト
"""

import os
import json
from datetime import datetime

def check_progress():
    """現在の進捗を確認"""
    print("🔍 DPOトレーニング進捗確認")
    print("=" * 50)
    
    # チェックポイント一覧
    outputs_dir = "./outputs"
    checkpoints = []
    
    for item in os.listdir(outputs_dir):
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(outputs_dir, item)):
            try:
                step = int(item.split("-")[1])
                checkpoints.append(step)
            except ValueError:
                continue
    
    checkpoints.sort()
    
    if checkpoints:
        latest_step = checkpoints[-1]
        print(f"📊 作成済みチェックポイント: {checkpoints}")
        print(f"🎯 最新ステップ: {latest_step}")
        
        # 最新チェックポイントの詳細
        checkpoint_path = f"./outputs/checkpoint-{latest_step}/trainer_state.json"
        
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as f:
                state = json.load(f)
            
            current_step = state['global_step']
            max_steps = state['max_steps']
            epoch = state['epoch']
            progress = (current_step / max_steps) * 100
            
            print(f"📈 進捗: {current_step}/{max_steps} ({progress:.1f}%)")
            print(f"🔄 エポック: {epoch:.3f}")
            
            # 最新メトリクス
            log_history = state['log_history']
            if log_history:
                latest_log = log_history[-1]
                print(f"\n📊 最新メトリクス (Step {latest_log.get('step', 'N/A')}):")
                
                loss = latest_log.get('loss', 'N/A')
                acc = latest_log.get('rewards/accuracies', 'N/A')
                margin = latest_log.get('rewards/margins', 'N/A')
                
                print(f"   Loss: {loss:.4f}" if isinstance(loss, (int, float)) else f"   Loss: {loss}")
                print(f"   精度: {acc:.1%}" if isinstance(acc, (int, float)) else f"   精度: {acc}")
                print(f"   Margin: {margin:.3f}" if isinstance(margin, (int, float)) else f"   Margin: {margin}")
                
                # 改善を計算
                if len(log_history) > 1:
                    first_log = log_history[0]
                    initial_loss = first_log.get('loss', 0)
                    current_loss = latest_log.get('loss', 0)
                    improvement = ((initial_loss - current_loss) / initial_loss) * 100
                    print(f"   Loss改善: {improvement:.1f}%")
        
        # 推定残り時間（簡易版）
        remaining_steps = 2139 - latest_step
        steps_per_100 = 100  # 100ステップごとにチェックポイント
        remaining_checkpoints = remaining_steps // steps_per_100
        estimated_hours = remaining_checkpoints * 0.1  # 概算
        
        print(f"\n⏱️  推定残り時間: 約{estimated_hours:.1f}時間")
        print(f"🏁 残りステップ: {remaining_steps}")
        
    else:
        print("❌ チェックポイントが見つかりません")
    
    # プロセス状況
    import subprocess
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
        dpo_processes = [line for line in result.stdout.split('\n') if 'conda_dpo_training.py' in line]
        
        if dpo_processes:
            print(f"\n✅ DPOトレーニング実行中 ({len(dpo_processes)} プロセス)")
        else:
            print(f"\n⚠️  DPOトレーニングプロセスが見つかりません")
    except:
        pass
    
    print(f"\n🕐 確認時刻: {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    check_progress()
