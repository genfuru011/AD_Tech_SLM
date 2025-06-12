#!/usr/bin/env python3
"""
DPOトレーニング中の中間テスト
checkpoint-100でのモデル性能確認
"""

import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# ログ設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def quick_test():
    """簡単なテスト実行"""
    
    base_model_name = "SakanaAI/TinySwallow-1.5B-Instruct"
    checkpoint_path = "./outputs/tiny_swallow_dpo/checkpoint-100"
    
    if not os.path.exists(checkpoint_path):
        logger.error(f"チェックポイントが見つかりません: {checkpoint_path}")
        return
    
    logger.info("🔍 checkpoint-100でのクイックテスト開始")
    
    # デバイス設定
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"🔧 使用デバイス: {device}")
    
    try:
        # トークナイザー読み込み
        logger.info("📥 トークナイザー読み込み...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # ベースモデル読み込み
        logger.info("📥 ベースモデル読み込み...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float32 if device == "mps" else torch.float16,
            trust_remote_code=True
        )
        
        # DPOアダプター読み込み
        logger.info("📥 DPOアダプター読み込み...")
        model = PeftModel.from_pretrained(model, checkpoint_path)
        model = model.merge_and_unload()
        
        model.to(device)
        model.eval()
        
        # テストプロンプト
        test_prompts = [
            "RTB（リアルタイム入札）の仕組みを簡潔に説明してください。",
            "プログラマティック広告の主な利点は何ですか？",
            "CPMとCPCの違いを教えてください。"
        ]
        
        logger.info("🧪 テスト実行...")
        
        for i, prompt in enumerate(test_prompts):
            logger.info(f"\n--- テスト {i+1} ---")
            logger.info(f"プロンプト: {prompt}")
            
            # トークン化
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=256,
                truncation=True
            ).to(device)
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # デコード
            response = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            logger.info(f"応答: {response.strip()}")
        
        logger.info("\n✅ クイックテスト完了")
        
    except Exception as e:
        logger.error(f"❌ テスト中にエラー: {e}")

if __name__ == "__main__":
    quick_test()
