#!/usr/bin/env python3
"""
TinySwallow DPO訓練済みモデルの軽量評価スクリプト
CPU環境での安全な評価
"""

import os
import json
import logging
import torch
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def evaluate_model():
    """軽量モデル評価"""
    
    base_model_name = "SakanaAI/TinySwallow-1.5B-Instruct"
    trained_model_path = "./outputs/tiny_swallow_dpo/final_model"
    
    # CPU強制使用
    device = "cpu"
    logger.info("🔧 CPU環境で評価実行")
    
    try:
        # トークナイザー読み込み
        logger.info("📥 トークナイザー読み込み...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # ベースモデル読み込み（CPU、float32）
        logger.info("📥 ベースモデル読み込み...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            device_map=None
        )
        
        # DPOアダプター読み込み
        if os.path.exists(trained_model_path):
            logger.info("📥 DPO訓練済みアダプター読み込み...")
            model = PeftModel.from_pretrained(model, trained_model_path)
            model = model.merge_and_unload()
            logger.info("✅ DPO訓練済みモデル使用")
        else:
            logger.warning("⚠️ 訓練済みモデルが見つからないため、ベースモデルを使用")
        
        model.to(device)
        model.eval()
        
        # 簡単なテストプロンプト
        test_prompts = [
            "RTBとは何ですか？",
            "プログラマティック広告の利点を教えてください。",
            "CPMの意味を説明してください。"
        ]
        
        logger.info("🧪 評価開始...")
        results = []
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"📝 テスト {i+1}/3: {prompt}")
            
            try:
                # トークン化
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True
                )
                
                # 生成（グリーディ生成で安定性向上）
                with torch.no_grad():
                    outputs = model.generate(
                        inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        max_new_tokens=50,  # 短い応答
                        do_sample=False,  # グリーディ生成
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
                
                # デコード
                response = tokenizer.decode(
                    outputs[0][inputs.input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                
                result = {
                    "prompt": prompt,
                    "response": response.strip(),
                    "success": True
                }
                results.append(result)
                
                print(f"\n📋 プロンプト: {prompt}")
                print(f"📝 応答: {response.strip()}")
                
            except Exception as e:
                logger.error(f"❌ エラー: {e}")
                results.append({
                    "prompt": prompt,
                    "response": "",
                    "success": False,
                    "error": str(e)
                })
        
        # 結果保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"outputs/evaluation/final_evaluation_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 評価結果保存: {output_file}")
        logger.info("🎉 評価完了！")
        
        # 統計
        success_count = sum(1 for r in results if r['success'])
        print(f"\n📊 評価統計: {success_count}/{len(results)} 成功")
        
    except Exception as e:
        logger.error(f"❌ 評価中にエラー: {e}")

if __name__ == "__main__":
    evaluate_model()
