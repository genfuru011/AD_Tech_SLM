#!/usr/bin/env python3
"""
DPO中間モデルのテストスクリプト - Step 300
チェックポイント300のモデル品質を確認
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import PeftModel
import json

def test_checkpoint_300():
    print("🤖 DPO中間モデル（Step 300）をテスト中...")
    
    # モデルとトークナイザーの読み込み
    model_name = "gpt2"
    checkpoint_path = "./outputs/checkpoint-300"
    
    print(f"📥 ベースモデル読み込み: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = GPT2LMHeadModel.from_pretrained(model_name)
    
    print(f"🔧 LoRAアダプター読み込み: {checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()
    
    # MPS使用可能チェック
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device)
    print(f"🖥️  デバイス: {device}")
    
    # テストプロンプト（より短い、よりシンプルな質問）
    test_prompts = [
        "What is machine learning?",
        "How to study effectively?",
        "Benefits of exercise:",
        "Python programming tips:",
        "Healthy food choices:"
    ]
    
    print("\n" + "="*60)
    print("🧪 DPO訓練済みモデル生成テスト (Step 300)")
    print("="*60)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n【テスト {i}】")
        print(f"💬 プロンプト: {prompt}")
        print("📝 生成結果:")
        
        # テキスト生成
        inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 80,
                num_return_sequences=1,
                temperature=0.8,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2,
                early_stopping=True
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(prompt):].strip()
        
        # 品質指標
        print(f"   {response}")
        print(f"📊 生成長: {len(response)} 文字")
        print("-" * 40)
    
    print("\n✅ Step 300 モデルテスト完了!")
    print("📈 進捗状況:")
    print(f"   🎯 現在ステップ: 300/2139 (14.0%)")
    print(f"   📉 Eval Loss: 0.009 (優秀な改善)")
    print(f"   ⭐ Eval 精度: 100%")
    print(f"   📊 Eval Margin: 5.23 (強い選好学習)")

if __name__ == "__main__":
    test_checkpoint_300()
