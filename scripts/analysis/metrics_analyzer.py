#!/usr/bin/env python3
"""
DPOトレーニングメトリクス詳細確認
TensorBoardデータから進行状況を詳しく見る
"""

import os
import json
import glob
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_training_metrics():
    """トレーニングメトリクスの詳細分析"""
    
    project_root = Path("/Users/usr0302442/Documents/AD_Tech_SLM")
    checkpoint_dir = project_root / "outputs" / "tiny_swallow_dpo"
    
    logger.info("📊 DPOトレーニングメトリクス分析")
    logger.info("=" * 60)
    
    # 利用可能なチェックポイントを確認
    checkpoints = list(checkpoint_dir.glob("checkpoint-*"))
    checkpoints.sort(key=lambda x: int(x.name.split('-')[1]))
    
    if not checkpoints:
        logger.warning("チェックポイントが見つかりません")
        return
    
    logger.info(f"📁 利用可能なチェックポイント: {len(checkpoints)}個")
    
    for checkpoint in checkpoints:
        step_num = int(checkpoint.name.split('-')[1])
        logger.info(f"\n🔍 {checkpoint.name} の詳細:")
        
        # trainer_state.jsonを読み込み
        trainer_state_file = checkpoint / "trainer_state.json"
        if trainer_state_file.exists():
            with open(trainer_state_file, 'r') as f:
                state = json.load(f)
            
            logger.info(f"   ステップ: {state['global_step']}")
            logger.info(f"   エポック: {state['epoch']:.4f}")
            
            # ログ履歴の分析
            if state.get('log_history'):
                recent_logs = state['log_history'][-5:]  # 最新5件
                
                logger.info("   最新メトリクス:")
                for log_entry in recent_logs:
                    step = log_entry.get('step', 'N/A')
                    loss = log_entry.get('loss', 'N/A')
                    lr = log_entry.get('learning_rate', 'N/A')
                    accuracy = log_entry.get('rewards/accuracies', 'N/A')
                    
                    if loss != 'N/A' and loss != 0.0:
                        logger.info(f"     ステップ {step}: loss={loss:.6f}, lr={lr:.2e}, acc={accuracy}")
                    elif 'eval_loss' in log_entry:
                        eval_loss = log_entry.get('eval_loss', 'N/A')
                        eval_acc = log_entry.get('eval_rewards/accuracies', 'N/A')
                        logger.info(f"     評価 {step}: eval_loss={eval_loss}, eval_acc={eval_acc}")
        
        # チェックポイントのファイルサイズ
        total_size = sum(f.stat().st_size for f in checkpoint.rglob('*') if f.is_file())
        logger.info(f"   サイズ: {total_size / 1024 / 1024:.1f} MB")
        
        # 作成時刻
        created = datetime.fromtimestamp(checkpoint.stat().st_ctime)
        logger.info(f"   作成時刻: {created.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 進行速度の推定
    if len(checkpoints) >= 1:
        latest_checkpoint = checkpoints[-1]
        step_num = int(latest_checkpoint.name.split('-')[1])
        created_time = latest_checkpoint.stat().st_ctime
        
        # 開始時刻の推定（15:36頃）
        start_time_estimate = datetime(2025, 6, 12, 15, 36).timestamp()
        elapsed_minutes = (created_time - start_time_estimate) / 60
        
        steps_per_minute = step_num / elapsed_minutes if elapsed_minutes > 0 else 0
        
        logger.info(f"\n⏱️  進行速度推定:")
        logger.info(f"   {step_num}ステップを{elapsed_minutes:.1f}分で完了")
        logger.info(f"   速度: {steps_per_minute:.2f} ステップ/分")
        
        if steps_per_minute > 0:
            remaining_steps = 800 - step_num
            remaining_minutes = remaining_steps / steps_per_minute
            logger.info(f"   残り{remaining_steps}ステップの推定時間: {remaining_minutes:.1f}分 ({remaining_minutes/60:.1f}時間)")
    
    logger.info("\n" + "=" * 60)

if __name__ == "__main__":
    analyze_training_metrics()
