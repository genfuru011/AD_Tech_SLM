#!/opt/miniconda3/envs/dpo_training/bin/python
"""
TinySwallow-1.5B-Instruct モデル動作確認テスト
モデル読み込みと基本的な生成テスト
"""

import torch
import sys
import os

def test_tinyswallow():
    """TinySwallowモデルのテスト"""
    print("🦆 TinySwallow-1.5B-Instruct 動作確認テスト")
    print("="*50)
    
    # PyTorch環境確認
    print(f"🔧 PyTorch バージョン: {torch.__version__}")
    if torch.backends.mps.is_available():
        device = "mps"
        print("✅ MPS (Metal Performance Shaders) 利用可能")
    elif torch.cuda.is_available():
        device = "cuda"
        print("✅ CUDA GPU 利用可能")
    else:
        device = "cpu"
        print("⚠️  CPU モードで実行")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✅ Transformers ライブラリ読み込み成功")
    except ImportError as e:
        print(f"❌ Transformers ライブラリエラー: {e}")
        return False
    
    # モデル読み込み
    model_name = "SakanaAI/TinySwallow-1.5B-Instruct"
    print(f"\n📥 モデル読み込み中: {model_name}")
    
    try:
        # トークナイザー読み込み
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ トークナイザー読み込み成功")
        
        # パディングトークン設定
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("🔧 パディングトークン設定完了")
        
        # モデル読み込み
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            trust_remote_code=True
        )
        print("✅ モデル読み込み成功")
        
        # デバイスに移動
        if device == "mps":
            model = model.to(device)
            print("✅ MPS デバイスに移動完了")
        
        # パラメータ数確認
        param_count = sum(p.numel() for p in model.parameters())
        print(f"📊 パラメータ数: {param_count:,}")
        
    except Exception as e:
        print(f"❌ モデル読み込みエラー: {e}")
        return False
    
    # テスト生成
    test_prompts = [
        "プログラマティック広告とは何ですか？",
        "RTBの仕組みを教えてください。",
        "日本のデジタルマーケティングについて"
    ]
    
    print(f"\n🧪 テスト生成開始 (デバイス: {device})")
    print("-" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n🔍 テスト {i}: {prompt}")
        
        try:
            # トークン化
            inputs = tokenizer.encode(prompt, return_tensors="pt")
            if device == "mps":
                inputs = inputs.to(device)
            
            # 生成
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # デコード
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = generated_text[len(prompt):].strip()
            
            print(f"📝 生成結果: {response}")
            
        except Exception as e:
            print(f"❌ 生成エラー: {e}")
            return False
    
    print("\n" + "="*50)
    print("🎉 TinySwallow-1.5B-Instruct 動作確認完了！")
    print("✅ DPO トレーニングの準備ができています。")
    
    return True

if __name__ == "__main__":
    success = test_tinyswallow()
    if not success:
        sys.exit(1)
