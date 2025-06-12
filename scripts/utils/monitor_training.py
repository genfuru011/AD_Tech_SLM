#!/opt/miniconda3/envs/dpo_training/bin/python
"""
DPO Training Monitor
Monitor the progress of DPO training and display key metrics
"""

import os
import time
import json
from pathlib import Path

def monitor_training():
    """Monitor DPO training progress"""
    output_dir = Path("./outputs")
    logs_dir = output_dir / "logs"
    
    print("🔍 DPO Training Monitor")
    print("=" * 50)
    
    # Check if training is running
    os.system("ps aux | grep conda_dpo_training | grep -v grep")
    
    print("\n📊 Output Directory Contents:")
    if output_dir.exists():
        for item in output_dir.iterdir():
            if item.is_file():
                size = item.stat().st_size / (1024*1024)  # MB
                print(f"  📄 {item.name}: {size:.2f} MB")
            elif item.is_dir():
                file_count = len(list(item.iterdir()))
                print(f"  📁 {item.name}/: {file_count} files")
    else:
        print("  ❌ Output directory not found")
    
    print("\n🎯 Looking for recent checkpoints...")
    if output_dir.exists():
        checkpoints = [d for d in output_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint")]
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"  ✅ Latest checkpoint: {latest_checkpoint.name}")
            
            # Try to read trainer state
            trainer_state_file = latest_checkpoint / "trainer_state.json"
            if trainer_state_file.exists():
                try:
                    with open(trainer_state_file, 'r') as f:
                        state = json.load(f)
                    current_step = state.get('global_step', 0)
                    best_metric = state.get('best_metric', 'N/A')
                    print(f"  📈 Current step: {current_step}")
                    print(f"  🏆 Best metric: {best_metric}")
                    
                    if 'log_history' in state:
                        recent_logs = state['log_history'][-3:]  # Last 3 entries
                        print("\n📝 Recent training logs:")
                        for log in recent_logs:
                            if 'loss' in log:
                                step = log.get('step', '?')
                                loss = log.get('loss', '?')
                                print(f"    Step {step}: Loss = {loss}")
                except Exception as e:
                    print(f"  ⚠️ Could not read trainer state: {e}")
        else:
            print("  ❌ No checkpoints found yet")
    
    print("\n💾 Memory Usage:")
    os.system("ps aux | grep conda_dpo_training | grep -v grep | awk '{print \"  🧠 Memory: \" $6/1024 \" MB, CPU: \" $3 \"%\"}'")

if __name__ == "__main__":
    monitor_training()
