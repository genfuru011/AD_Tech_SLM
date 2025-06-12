#!/usr/bin/env python3
"""
ベースモデルとDPO訓練済みモデルの比較評価
"""

import os
import json
import logging
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def compare_models():
    """ベースモデルとDPO訓練済みモデルの比較"""
    
    base_model_name = "SakanaAI/TinySwallow-1.5B-Instruct"
    trained_model_path = "./outputs/tiny_swallow_dpo/final_model"
    device = "cpu"
    
    # テストプロンプト
    test_prompts = [
        "RTBとは何ですか？",
        "プログラマティック広告の利点を教えてください。"
    ]
    
    logger.info("🔍 ベースモデルとDPO訓練済みモデルの比較開始")
    
    # トークナイザー読み込み
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    results = {"base_model": [], "dpo_model": []}
    
    # 1. ベースモデルのテスト
    logger.info("📥 ベースモデル読み込み...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    ).to(device)
    base_model.eval()
    
    logger.info("🧪 ベースモデル評価...")
    for prompt in test_prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            
            with torch.no_grad():
                outputs = base_model.generate(
                    inputs.input_ids,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            results["base_model"].append({"prompt": prompt, "response": response.strip()})
            print(f"\n🔵 ベース - {prompt}")
            print(f"   応答: {response.strip()}")
            
        except Exception as e:
            logger.error(f"ベースモデルエラー: {e}")
            results["base_model"].append({"prompt": prompt, "response": f"エラー: {e}"})
    
    # メモリクリア
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 2. DPO訓練済みモデルのテスト
    logger.info("📥 DPO訓練済みモデル読み込み...")
    dpo_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float32,
        trust_remote_code=True
    )
    
    if os.path.exists(trained_model_path):
        dpo_model = PeftModel.from_pretrained(dpo_model, trained_model_path)
        dpo_model = dpo_model.merge_and_unload()
    
    dpo_model.to(device)
    dpo_model.eval()
    
    logger.info("🧪 DPO訓練済みモデル評価...")
    for prompt in test_prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            
            with torch.no_grad():
                outputs = dpo_model.generate(
                    inputs.input_ids,
                    max_new_tokens=30,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            results["dpo_model"].append({"prompt": prompt, "response": response.strip()})
            print(f"\n🟢 DPO - {prompt}")
            print(f"   応答: {response.strip()}")
            
        except Exception as e:
            logger.error(f"DPOモデルエラー: {e}")
            results["dpo_model"].append({"prompt": prompt, "response": f"エラー: {e}"})
    
    # 結果保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"outputs/evaluation/model_comparison_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"💾 比較結果保存: {output_file}")
    logger.info("🎉 比較評価完了！")

if __name__ == "__main__":
    compare_models()
