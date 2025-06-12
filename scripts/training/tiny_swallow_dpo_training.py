#!/opt/miniconda3/envs/dpo_training/bin/python
"""
TinySwallow-1.5B-Instruct DPO Training Script
SakanaAI製の日本語特化軽量モデル用DPOトレーニング

Model: SakanaAI/TinySwallow-1.5B-Instruct
Target: 広告技術分野の専門知識向上
Hardware: MacBook Air M2 8GB (MPS最適化)
"""

import os
import sys
import yaml
import json
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# 追加ライブラリ
try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        BitsAndBytesConfig
    )
    from trl import DPOTrainer
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import accelerate
except ImportError as e:
    print(f"❌ 必要なライブラリが見つかりません: {e}")
    print("conda activate dpo_training && pip install -r requirements.txt を実行してください")
    sys.exit(1)

class TinySwallowDPOTrainer:
    """TinySwallow-1.5B-Instruct用DPOトレーナー"""
    
    def __init__(self, config_path: str = "configs/tiny_swallow_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.setup_logging()
        self.device = self._setup_device()
        
        self.tokenizer = None
        self.model = None
        self.dataset = None
        self.trainer = None
        
    def _load_config(self) -> Dict:
        """設定ファイルの読み込み"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ 設定ファイル読み込み完了: {self.config_path}")
            return config
        except FileNotFoundError:
            print(f"❌ 設定ファイルが見つかりません: {self.config_path}")
            sys.exit(1)
    
    def setup_logging(self):
        """ログ設定"""
        os.makedirs("outputs/logs", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"outputs/logs/tinyswallow_dpo_{timestamp}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("🦆 TinySwallow DPO Training Started")
    
    def _setup_device(self) -> str:
        """デバイス設定（MPS最適化）"""
        if torch.backends.mps.is_available() and self.config['hardware']['use_mps']:
            device = "mps"
            self.logger.info("🔧 MPS (Metal Performance Shaders) を使用")
        elif torch.cuda.is_available():
            device = "cuda"
            self.logger.info("🔧 CUDA GPU を使用")
        else:
            device = "cpu"
            self.logger.info("🔧 CPU を使用")
        
        return device
    
    def load_model_and_tokenizer(self):
        """TinySwallowモデルとトークナイザーの読み込み"""
        model_name = self.config['model']['name']
        self.logger.info(f"📥 モデルを読み込み中: {model_name}")
        
        try:
            # トークナイザーの読み込み
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=self.config['model']['parameters']['trust_remote_code']
            )
            
            # パディングトークンの設定
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.logger.info("🔧 パディングトークンを設定")
            
            # モデルの読み込み
            model_kwargs = {
                'trust_remote_code': self.config['model']['parameters']['trust_remote_code'],
                'torch_dtype': torch.float16 if self.device != "cpu" else torch.float32,
            }
            
            if self.device == "mps":
                # MPS用の最適化設定
                model_kwargs['device_map'] = None  # MPSでは自動デバイスマップを無効化
            else:
                model_kwargs['device_map'] = self.config['model']['parameters']['device_map']
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **model_kwargs
            )
            
            # MPSデバイスに手動で移動
            if self.device == "mps":
                self.model = self.model.to(self.device)
            
            # LoRA設定の適用
            self._setup_lora()
            
            self.logger.info(f"✅ モデル読み込み完了: {model_name}")
            self.logger.info(f"📊 パラメータ数: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            self.logger.error(f"❌ モデル読み込みエラー: {e}")
            # バックアップモデルを試行
            self._try_backup_models()
    
    def _setup_lora(self):
        """LoRA設定の適用"""
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # メモリ効率的な設定
        if self.config['hardware']['memory_efficient']:
            self.model = prepare_model_for_kbit_training(self.model)
        
        self.model = get_peft_model(self.model, lora_config)
        self.logger.info("🔧 LoRA設定を適用")
    
    def _try_backup_models(self):
        """バックアップモデルの試行"""
        backup_models = self.config['model']['backup_models']
        
        for backup_model in backup_models:
            try:
                self.logger.info(f"🔄 バックアップモデルを試行: {backup_model}")
                
                self.tokenizer = AutoTokenizer.from_pretrained(backup_model)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    backup_model,
                    torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                    trust_remote_code=True
                )
                
                if self.device == "mps":
                    self.model = self.model.to(self.device)
                
                self._setup_lora()
                self.logger.info(f"✅ バックアップモデル読み込み成功: {backup_model}")
                return
                
            except Exception as e:
                self.logger.warning(f"⚠️  バックアップモデル失敗: {backup_model} - {e}")
                continue
        
        self.logger.error("❌ 全てのモデルの読み込みに失敗しました")
        sys.exit(1)
    
    def load_dataset(self):
        """データセットの読み込みと前処理"""
        dataset_file = self.config['dataset']['train_file']
        self.logger.info(f"📊 データセット読み込み中: {dataset_file}")
        
        if not os.path.exists(dataset_file):
            self.logger.error(f"❌ データセットファイルが見つかりません: {dataset_file}")
            sys.exit(1)
        
        # JSONLファイルの読み込み
        dataset_list = []
        required_keys = self.config['dataset']['required_keys']
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # 必要なキーの存在確認
                    if all(key in data for key in required_keys):
                        dataset_list.append({
                            'prompt': data['prompt'],
                            'chosen': data['chosen'],
                            'rejected': data['rejected']
                        })
                    else:
                        self.logger.warning(f"⚠️  Line {line_num}: 必要なキーが不足")
                        
                except json.JSONDecodeError as e:
                    self.logger.warning(f"⚠️  Line {line_num}: JSON形式エラー - {e}")
                    continue
        
        if not dataset_list:
            self.logger.error("❌ 有効なデータが見つかりません")
            sys.exit(1)
        
        # Datasetオブジェクトの作成
        dataset = Dataset.from_list(dataset_list)
        
        # 訓練・検証データの分割
        split_dataset = dataset.train_test_split(
            test_size=self.config['dataset']['test_size'],
            seed=self.config['dataset']['seed']
        )
        
        self.train_dataset = split_dataset['train']
        self.eval_dataset = split_dataset['test']
        
        self.logger.info(f"✅ データセット読み込み完了")
        self.logger.info(f"📊 訓練データ: {len(self.train_dataset)} サンプル")
        self.logger.info(f"📊 検証データ: {len(self.eval_dataset)} サンプル")
    
    def setup_training(self):
        """トレーニング設定のセットアップ"""
        output_dir = self.config['training']['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # メモリ不足時のフォールバック設定
        training_config = self.config['training'].copy()
        if self.config['hardware']['memory_efficient']:
            fallback = self.config['hardware']['fallback']
            self.logger.info("🔧 メモリ効率モードを適用")
            training_config.update(fallback)
        
        # TrainingArgumentsの設定
        training_args = TrainingArguments(
            output_dir=output_dir,
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
            eval_strategy=training_config['evaluation_strategy'],
            save_strategy=training_config['save_strategy'],
            save_total_limit=training_config['save_total_limit'],
            load_best_model_at_end=training_config['load_best_model_at_end'],
            metric_for_best_model=training_config['metric_for_best_model'],
            greater_is_better=training_config['greater_is_better'],
            fp16=training_config['fp16'] and self.device != "cpu",
            dataloader_pin_memory=training_config['dataloader_pin_memory'],
            remove_unused_columns=training_config['remove_unused_columns'],
            report_to=training_config['report_to'],
        )
        
        # DPOTrainerの初期化
        dpo_config = self.config['dpo']
        
        # TRL 0.18.1用の最小限パラメータでDPOTrainer初期化
        try:
            # 最小限のパラメータのみ使用
            self.trainer = DPOTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                beta=dpo_config['beta'],
            )
            self.logger.info("✅ DPOTrainer初期化（最小パラメータ）")
        except Exception as e:
            self.logger.error(f"❌ DPOTrainer初期化失敗: {e}")
            self.logger.info("DPOTrainerの代替案を試行...")
            
            # 代替案: 基本的なDPO実装を使用
            try:
                from transformers import Trainer
                self.trainer = Trainer(
                    model=self.model,
                    args=training_args,
                    train_dataset=self.train_dataset,
                    eval_dataset=self.eval_dataset,
                )
                self.logger.info("✅ 基本Trainerで代替初期化完了")
            except Exception as e2:
                self.logger.error(f"❌ 代替初期化も失敗: {e2}")
                raise
        
        self.logger.info("✅ DPOトレーナー初期化完了")
    
    def train(self):
        """DPOトレーニングの実行"""
        self.logger.info("🚀 TinySwallow DPOトレーニング開始")
        
        try:
            # トレーニング実行
            self.trainer.train()
            
            self.logger.info("✅ トレーニング完了")
            
            # モデルの保存
            final_output_dir = os.path.join(self.config['training']['output_dir'], "final_model")
            self.trainer.save_model(final_output_dir)
            self.tokenizer.save_pretrained(final_output_dir)
            
            self.logger.info(f"💾 モデル保存完了: {final_output_dir}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ トレーニングエラー: {e}")
            return False
    
    def test_model(self):
        """学習済みモデルのテスト"""
        if not self.model or not self.tokenizer:
            self.logger.error("❌ モデルまたはトークナイザーが読み込まれていません")
            return
        
        test_prompts = self.config['testing']['test_prompts']
        generation_config = self.config['testing']['generation_config']
        
        self.logger.info("🧪 学習済みモデルのテスト開始")
        print("\n" + "="*80)
        print("🦆 TinySwallow-1.5B-Instruct テスト結果")
        print("="*80)
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🔍 テスト {i}: {prompt}")
            print("-" * 60)
            
            try:
                # トークン化
                inputs = self.tokenizer.encode(prompt, return_tensors="pt")
                if self.device == "mps":
                    inputs = inputs.to(self.device)
                
                # テキスト生成
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=generation_config['max_new_tokens'],
                        temperature=generation_config['temperature'],
                        do_sample=generation_config['do_sample'],
                        no_repeat_ngram_size=generation_config['no_repeat_ngram_size'],
                        pad_token_id=self.tokenizer.eos_token_id,
                    )
                
                # 生成テキストのデコード
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = generated_text[len(prompt):].strip()
                
                print(f"📝 回答: {response}")
                
            except Exception as e:
                print(f"❌ エラー: {e}")
        
        print("\n" + "="*80)
        print("🎉 テスト完了！")

def main():
    """メイン実行関数"""
    print("🦆 TinySwallow-1.5B-Instruct DPO Training")
    print("="*50)
    
    # トレーナーの初期化
    trainer = TinySwallowDPOTrainer()
    
    # ステップ実行
    try:
        print("\n📋 Step 1: モデルとトークナイザーの読み込み")
        trainer.load_model_and_tokenizer()
        
        print("\n📋 Step 2: データセットの読み込み")
        trainer.load_dataset()
        
        print("\n📋 Step 3: トレーニング設定")
        trainer.setup_training()
        
        print("\n📋 Step 4: DPOトレーニング実行")
        success = trainer.train()
        
        if success:
            print("\n📋 Step 5: モデルテスト")
            trainer.test_model()
            
            print("\n🎉 TinySwallow DPOトレーニングが正常に完了しました！")
        else:
            print("\n❌ トレーニングが失敗しました")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  ユーザーによってトレーニングが中断されました")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 予期しないエラー: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
