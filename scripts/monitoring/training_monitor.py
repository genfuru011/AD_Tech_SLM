#!/usr/bin/env python3
"""
TinySwallow DPOトレーニング監視スクリプト
リアルタイムでトレーニングの進行状況を確認
"""

import os
import time
import json
import psutil
import subprocess
from datetime import datetime
from pathlib import Path


class TrainingMonitor:
    """DPOトレーニング監視クラス"""
    
    def __init__(self, project_root: str = "/Users/usr0302442/Documents/AD_Tech_SLM"):
        self.project_root = Path(project_root)
        self.output_dir = self.project_root / "outputs" / "tiny_swallow_dpo"
        self.logs_dir = self.project_root / "outputs" / "logs"
        
    def find_training_process(self) -> dict:
        """DPOトレーニングプロセスを検索"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_info', 'create_time']):
            try:
                cmdline_list = proc.info['cmdline']
                if cmdline_list is None:
                    continue
                cmdline = ' '.join(cmdline_list) if isinstance(cmdline_list, list) else str(cmdline_list)
                if 'tiny_swallow_dpo_training.py' in cmdline:
                    return {
                        'pid': proc.info['pid'],
                        'cpu_percent': proc.info['cpu_percent'],
                        'memory_mb': proc.info['memory_info'].rss / 1024 / 1024,
                        'runtime': time.time() - proc.info['create_time'],
                        'cmdline': cmdline
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return None
    
    def get_latest_log(self) -> str:
        """最新のログファイルを取得"""
        if not self.logs_dir.exists():
            return "ログディレクトリが見つかりません"
        
        log_files = list(self.logs_dir.glob("tinyswallow_dpo_*.log"))
        if not log_files:
            return "ログファイルが見つかりません"
        
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_log, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"ログ読み込みエラー: {e}"
    
    def check_checkpoints(self) -> list:
        """チェックポイントディレクトリを確認"""
        checkpoints = []
        if self.output_dir.exists():
            for item in self.output_dir.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    try:
                        step_num = int(item.name.split('-')[1])
                        stat = item.stat()
                        checkpoints.append({
                            'name': item.name,
                            'step': step_num,
                            'created': datetime.fromtimestamp(stat.st_ctime),
                            'size_mb': sum(f.stat().st_size for f in item.rglob('*') if f.is_file()) / 1024 / 1024
                        })
                    except (ValueError, OSError):
                        continue
        
        return sorted(checkpoints, key=lambda x: x['step'])
    
    def get_tensorboard_info(self) -> dict:
        """TensorBoardログ情報を取得"""
        runs_dir = self.output_dir / "runs"
        if not runs_dir.exists():
            return {"status": "TensorBoardログディレクトリが見つかりません"}
        
        tb_dirs = list(runs_dir.iterdir())
        if not tb_dirs:
            return {"status": "TensorBoardログが見つかりません"}
        
        latest_tb_dir = max(tb_dirs, key=lambda x: x.stat().st_mtime)
        tb_files = list(latest_tb_dir.glob("events.out.tfevents.*"))
        
        return {
            "status": "利用可能",
            "directory": str(latest_tb_dir),
            "files": len(tb_files),
            "latest_update": datetime.fromtimestamp(latest_tb_dir.stat().st_mtime)
        }
    
    def display_status(self):
        """現在の状況を表示"""
        print("=" * 80)
        print(f"🦆 TinySwallow DPOトレーニング監視 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        # プロセス情報
        process_info = self.find_training_process()
        if process_info:
            runtime_hours = process_info['runtime'] / 3600
            print(f"✅ トレーニングプロセス: 実行中")
            print(f"   PID: {process_info['pid']}")
            print(f"   CPU使用率: {process_info['cpu_percent']:.1f}%")
            print(f"   メモリ使用量: {process_info['memory_mb']:.0f} MB")
            print(f"   実行時間: {runtime_hours:.2f} 時間 ({process_info['runtime']/60:.1f} 分)")
        else:
            print("❌ トレーニングプロセス: 見つかりません")
        
        print()
        
        # チェックポイント情報
        checkpoints = self.check_checkpoints()
        print(f"📁 チェックポイント: {len(checkpoints)}個")
        if checkpoints:
            latest_cp = checkpoints[-1]
            print(f"   最新: {latest_cp['name']} (ステップ {latest_cp['step']})")
            print(f"   作成日時: {latest_cp['created'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"   サイズ: {latest_cp['size_mb']:.1f} MB")
            
            # 進行状況推定（max_steps: 800）
            progress = (latest_cp['step'] / 800) * 100
            print(f"   推定進行率: {progress:.1f}% ({latest_cp['step']}/800 ステップ)")
        else:
            print("   まだチェックポイントは作成されていません")
        
        print()
        
        # TensorBoard情報
        tb_info = self.get_tensorboard_info()
        print(f"📊 TensorBoard: {tb_info['status']}")
        if "directory" in tb_info:
            print(f"   ログディレクトリ: {tb_info['directory']}")
            print(f"   最終更新: {tb_info['latest_update'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        print()
        
        # 最新ログの末尾
        print("📝 最新ログ（末尾10行）:")
        print("-" * 60)
        latest_log = self.get_latest_log()
        log_lines = latest_log.strip().split('\n')
        for line in log_lines[-10:]:
            print(f"   {line}")
        
        print("=" * 80)
    
    def monitor_loop(self, interval: int = 30):
        """監視ループ"""
        print("🔄 トレーニング監視を開始します (Ctrl+Cで終了)")
        print(f"更新間隔: {interval}秒")
        print()
        
        try:
            while True:
                self.display_status()
                
                # トレーニングプロセスが終了しているかチェック
                if not self.find_training_process():
                    print("🏁 トレーニングプロセスが終了しました")
                    break
                
                print(f"⏱️  {interval}秒後に更新...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n👋 監視を終了します")


def main():
    """メイン関数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TinySwallow DPOトレーニング監視")
    parser.add_argument("--interval", "-i", type=int, default=30,
                        help="更新間隔（秒）デフォルト: 30")
    parser.add_argument("--once", action="store_true",
                        help="一度だけ状況を表示して終了")
    
    args = parser.parse_args()
    
    monitor = TrainingMonitor()
    
    if args.once:
        monitor.display_status()
    else:
        monitor.monitor_loop(args.interval)


if __name__ == "__main__":
    main()
