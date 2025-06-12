#!/usr/bin/env python3
"""
簡易版 TinySwallow DPO Training
API問題を回避したシンプルな実装
"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments
)
from datasets import Dataset
from trl import DPOTrainer
import warnings
warnings.filterwarnings('ignore')

# 環境設定
os.environ["WANDB_DISABLED"] = "true"

def main():
    print("🦆 簡易版 TinySwallow DPO Training")
    print("=" * 50)
    
    # デバイス設定
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"🔧 使用デバイス: {device}")
    
    # モデルとトークナイザーの読み込み
    print("📥 モデル読み込み中...")
    model_name = "SakanaAI/TinySwallow-1.5B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # MPS用
        trust_remote_code=True
    )
    
    if device == "mps":
        model = model.to(device)
    
    print(f"✅ モデル読み込み完了: {model_name}")
    
    # データセット読み込み
    print("📊 データセット読み込み中...")
    dataset_list = []
    
    with open('data/dpo_dataset.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            dataset_list.append({
                'prompt': data['prompt'],
                'chosen': data['chosen'],
                'rejected': data['rejected']
            })
    
    # データセット作成
    dataset = Dataset.from_list(dataset_list)
    train_test = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test['train']
    eval_dataset = train_test['test']
    
    print(f"📊 訓練データ: {len(train_dataset)} サンプル")
    print(f"📊 検証データ: {len(eval_dataset)} サンプル")
    
    # トレーニング設定
    print("⚙️ トレーニング設定...")
    training_args = TrainingArguments(
        output_dir="./outputs/simple_dpo",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        warmup_steps=20,
        max_steps=100,  # 短縮テスト
        eval_steps=25,
        save_steps=50,
        logging_steps=5,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=False,  # MPS環境では無効
    )
    
    # DPOトレーナー初期化
    print("🎯 DPOトレーナー初期化...")
    try:
        dpo_trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            beta=0.1,
            max_length=512,
            max_prompt_length=256,
        )
        print("✅ DPOTrainer初期化成功")
        
        # トレーニング実行
        print("🚀 DPOトレーニング開始...")
        train_result = dpo_trainer.train()
        
        # モデル保存
        print("💾 モデル保存中...")
        dpo_trainer.save_model("./outputs/simple_dpo/final_model")
        tokenizer.save_pretrained("./outputs/simple_dpo/final_model")
        
        print("✅ トレーニング完了！")
        print(f"最終loss: {train_result.training_loss:.4f}")
        
        # テスト
        print("\n🧪 モデルテスト")
        test_prompts = [
            "プログラマティック広告の利点について教えてください。",
            "DMPとCDPの違いを教えてください。"
        ]
        
        for prompt in test_prompts:
            print(f"\n🔍 プロンプト: {prompt}")
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            if device == "mps":
                inputs = inputs.to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response[len(prompt):].strip()
            print(f"📝 回答: {response}")
        
    except Exception as e:
        print(f"❌ エラー: {e}")
        print("基本的なファインチューニングにフォールバック...")
        
        # 基本のTrainer使用
        from transformers import Trainer, DataCollatorForLanguageModeling
        
        def preprocess_function(examples):
            texts = [f"{prompt} {chosen}" for prompt, chosen in zip(examples['prompt'], examples['chosen'])]
            return tokenizer(texts, truncation=True, padding=True, max_length=512)
        
        tokenized_train = train_dataset.map(preprocess_function, batched=True)
        tokenized_eval = eval_dataset.map(preprocess_function, batched=True)
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            data_collator=data_collator,
        )
        
        print("🚀 基本ファインチューニング開始...")
        trainer.train()
        trainer.save_model("./outputs/simple_dpo/fallback_model")
        print("✅ ファインチューニング完了")

if __name__ == "__main__":
    main()
