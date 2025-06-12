#!/usr/bin/env python3
"""
TinySwallow DPOトレーニング - リアルタイム監視版
進行状況をリアルタイムで出力するバージョン
"""

import os
import sys
import json
import logging
import torch
import yaml
from datetime import datetime
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    set_seed
)
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType
import warnings
warnings.filterwarnings("ignore")

def setup_logging():
    """ログ設定"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'outputs/logs/dpo_realtime_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """設定ファイル読み込み"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def create_custom_callback():
    """カスタムコールバッククラス"""
    from transformers import TrainerCallback
    
    class RealtimeProgressCallback(TrainerCallback):
        def __init__(self):
            self.last_log_time = datetime.now()
            
        def on_step_end(self, args, state, control, **kwargs):
            current_time = datetime.now()
            if (current_time - self.last_log_time).seconds >= 30:  # 30秒ごと
                print(f"\n🔄 ステップ {state.global_step}/{args.max_steps} "
                      f"({state.global_step/args.max_steps*100:.1f}%) "
                      f"- エポック {state.epoch:.3f}")
                self.last_log_time = current_time
                
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                print(f"📊 ステップ {state.global_step}: "
                      f"loss={logs.get('loss', 'N/A'):.4f}, "
                      f"lr={logs.get('learning_rate', 'N/A'):.2e}")
                
        def on_save(self, args, state, control, **kwargs):
            print(f"💾 チェックポイント保存: ステップ {state.global_step}")
    
    return RealtimeProgressCallback()

def main():
    """メイン関数"""
    logger = setup_logging()
    logger.info("🦆 TinySwallow DPO リアルタイム監視トレーニング開始")
    
    # 設定読み込み
    config = load_config("configs/tiny_swallow_config.yaml")
    
    # デバイス設定
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"🔧 使用デバイス: {device}")
    
    # モデル・トークナイザー読み込み
    model_name = config['model']['name']
    logger.info(f"📥 モデル読み込み: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device == "mps" else torch.float16,
        trust_remote_code=True
    )
    
    # LoRA設定
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    
    # データセット読み込み
    logger.info("📊 データセット読み込み")
    with open(config['dataset']['train_file'], 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # データセット分割
    train_size = int(len(data) * 0.9)
    train_data = data[:train_size]
    eval_data = data[train_size:]
    
    train_dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data)
    
    logger.info(f"📊 訓練データ: {len(train_dataset)} サンプル")
    logger.info(f"📊 評価データ: {len(eval_dataset)} サンプル")
    
    # トレーニング設定
    training_config = config['training']
    
    # DPOConfig使用
    try:
        training_args = DPOConfig(
            output_dir=training_config['output_dir'],
            num_train_epochs=training_config['num_train_epochs'],
            max_steps=training_config['max_steps'],
            per_device_train_batch_size=training_config['per_device_train_batch_size'],
            per_device_eval_batch_size=training_config['per_device_eval_batch_size'],
            gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
            learning_rate=training_config['learning_rate'],
            warmup_steps=training_config['warmup_steps'],
            eval_steps=training_config['eval_steps'],
            save_steps=training_config['save_steps'],
            logging_steps=training_config['logging_steps'],
            eval_strategy=training_config['eval_strategy'],
            save_strategy=training_config['save_strategy'],
            load_best_model_at_end=training_config['load_best_model_at_end'],
            metric_for_best_model=training_config['metric_for_best_model'],
            fp16=training_config['fp16'],
            dataloader_pin_memory=training_config['dataloader_pin_memory'],
            remove_unused_columns=training_config['remove_unused_columns'],
            report_to=training_config['report_to'],
            beta=config['dpo']['beta'],
            max_length=config['dpo']['max_length'],
            max_prompt_length=config['dpo']['max_prompt_length']
        )
        logger.info("✅ DPOConfig使用")
    except Exception as e:
        logger.warning(f"DPOConfig失敗、TrainingArguments使用: {e}")
        training_args = TrainingArguments(**training_config)
    
    # DPOトレーナー初期化
    try:
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=[create_custom_callback()]
        )
        logger.info("✅ DPOTrainer初期化完了（TRL 0.18.1 API）")
    except Exception as e:
        logger.warning(f"新API失敗、レガシーAPI使用: {e}")
        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            callbacks=[create_custom_callback()]
        )
        logger.info("✅ DPOTrainer初期化完了（レガシーAPI）")
    
    # トレーニング開始
    logger.info("🚀 DPOトレーニング開始（リアルタイム監視）")
    print("=" * 60)
    print("🚀 DPOトレーニング開始 - リアルタイム進行表示")
    print(f"📊 総ステップ数: {training_config['max_steps']}")
    print(f"💾 保存間隔: {training_config['save_steps']} ステップ")
    print(f"📈 評価間隔: {training_config['eval_steps']} ステップ")
    print("=" * 60)
    
    try:
        trainer.train()
        logger.info("✅ トレーニング完了")
        print("\n🎉 DPOトレーニング完了！")
        
        # 最終モデル保存
        final_model_path = f"{training_config['output_dir']}/final_model"
        trainer.save_model(final_model_path)
        logger.info(f"💾 最終モデル保存: {final_model_path}")
        
    except KeyboardInterrupt:
        logger.info("⏹️ ユーザーによる中断")
        print("\n⏹️ トレーニングが中断されました")
    except Exception as e:
        logger.error(f"❌ トレーニングエラー: {e}")
        print(f"\n❌ エラー発生: {e}")

if __name__ == "__main__":
    main()
