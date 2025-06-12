#!/usr/bin/env python3
"""
DPOトレーニングの詳細進捗監視スクリプト
リアルタイムでメトリクスとETA（予想完了時間）を表示
"""

import os
import json
import time
import psutil
from datetime import datetime, timedelta

def get_latest_checkpoint():
    """最新のチェックポイントを取得"""
    outputs_dir = "./outputs"
    checkpoints = []
    
    for item in os.listdir(outputs_dir):
        if item.startswith("checkpoint-") and os.path.isdir(os.path.join(outputs_dir, item)):
            try:
                step = int(item.split("-")[1])
                checkpoints.append((step, item))
            except ValueError:
                continue
    
    if checkpoints:
        checkpoints.sort(key=lambda x: x[0])
        return checkpoints[-1]
    return None, None

def parse_trainer_state(checkpoint_dir):
    """trainer_state.jsonから詳細情報を取得"""
    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.json")
    
    if not os.path.exists(trainer_state_path):
        return None
    
    with open(trainer_state_path, 'r') as f:
        state = json.load(f)
    
    return state

def estimate_completion_time(current_step, max_steps, start_time):
    """完了予想時間を計算"""
    if current_step == 0:
        return "計算中..."
    
    elapsed = time.time() - start_time
    steps_per_second = current_step / elapsed
    remaining_steps = max_steps - current_step
    remaining_seconds = remaining_steps / steps_per_second
    
    eta = datetime.now() + timedelta(seconds=remaining_seconds)
    return eta.strftime("%H:%M:%S")

def format_duration(seconds):
    """秒を時:分:秒形式に変換"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def monitor_training():
    print("🔍 DPO詳細トレーニング監視を開始...")
    print("=" * 80)
    
    # トレーニング開始時刻を推定（プロセス開始時刻から）
    training_start_time = None
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'create_time']):
        try:
            if 'conda_dpo_training.py' in ' '.join(proc.info['cmdline'] or []):
                training_start_time = proc.info['create_time']
                break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    
    if not training_start_time:
        training_start_time = time.time()
        print("⚠️  トレーニング開始時刻を推定中...")
    
    try:
        while True:
            latest_step, latest_checkpoint = get_latest_checkpoint()
            
            if latest_checkpoint:
                checkpoint_path = f"./outputs/{latest_checkpoint}"
                state = parse_trainer_state(checkpoint_path)
                
                if state:
                    current_step = state['global_step']
                    max_steps = state['max_steps']
                    epoch = state['epoch']
                    
                    # 最新のログエントリを取得
                    log_history = state['log_history']
                    if log_history:
                        latest_log = log_history[-1]
                        
                        # 進捗計算
                        progress_pct = (current_step / max_steps) * 100
                        
                        # 時間計算
                        elapsed_time = time.time() - training_start_time
                        eta = estimate_completion_time(current_step, max_steps, training_start_time)
                        
                        # 表示
                        current_time = datetime.now().strftime("%H:%M:%S")
                        print(f"\n⏰ {current_time} - DPOトレーニング進捗状況")
                        print("-" * 60)
                        print(f"📊 進捗: {current_step:,}/{max_steps:,} ステップ ({progress_pct:.1f}%)")
                        print(f"🔄 エポック: {epoch:.3f}")
                        print(f"⏱️  経過時間: {format_duration(elapsed_time)}")
                        print(f"🏁 予想完了: {eta}")
                        
                        print(f"\n📈 最新メトリクス (Step {latest_log.get('step', 'N/A')}):")
                        loss = latest_log.get('loss', 'N/A')
                        acc = latest_log.get('rewards/accuracies', 'N/A')
                        margin = latest_log.get('rewards/margins', 'N/A')
                        chosen = latest_log.get('rewards/chosen', 'N/A')
                        rejected = latest_log.get('rewards/rejected', 'N/A')
                        
                        print(f"   🎯 Loss: {loss:.4f}" if isinstance(loss, (int, float)) else f"   🎯 Loss: {loss}")
                        print(f"   ⭐ 精度: {acc:.1%}" if isinstance(acc, (int, float)) else f"   ⭐ 精度: {acc}")
                        print(f"   📏 Margin: {margin:.3f}" if isinstance(margin, (int, float)) else f"   📏 Margin: {margin}")
                        print(f"   🎁 Chosen: {chosen:.3f}" if isinstance(chosen, (int, float)) else f"   🎁 Chosen: {chosen}")
                        print(f"   ❌ Rejected: {rejected:.3f}" if isinstance(rejected, (int, float)) else f"   ❌ Rejected: {rejected}")
                        
                        # 評価結果があるかチェック
                        eval_entries = [log for log in log_history if 'eval_loss' in log]
                        if eval_entries:
                            latest_eval = eval_entries[-1]
                            print(f"\n🧪 最新評価結果 (Step {latest_eval.get('step', 'N/A')}):")
                            print(f"   📉 Eval Loss: {latest_eval.get('eval_loss', 'N/A'):.4f}")
                            print(f"   ✅ Eval 精度: {latest_eval.get('eval_rewards/accuracies', 'N/A'):.1%}")
                            print(f"   📊 Eval Margin: {latest_eval.get('eval_rewards/margins', 'N/A'):.3f}")
                        
                        # メモリ使用量
                        try:
                            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'memory_info', 'cpu_percent']):
                                if 'conda_dpo_training.py' in ' '.join(proc.info['cmdline'] or []):
                                    memory_mb = proc.info['memory_info'].rss / 1024 / 1024
                                    cpu_pct = proc.info['cpu_percent']
                                    print(f"\n💻 システム情報:")
                                    print(f"   🧠 メモリ: {memory_mb:.1f} MB")
                                    print(f"   🖥️  CPU: {cpu_pct:.1f}%")
                                    break
                        except:
                            pass
                        
                        print("=" * 60)
                        
                    else:
                        print("⚠️  ログ履歴が見つかりません")
                else:
                    print(f"❌ trainer_state.jsonの読み込みに失敗: {checkpoint_path}")
            else:
                print("🔍 チェックポイントを検索中...")
            
            # 30秒間隔で更新
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n🛑 監視を停止しました")
        print("✅ トレーニングは継続中です")

if __name__ == "__main__":
    monitor_training()
