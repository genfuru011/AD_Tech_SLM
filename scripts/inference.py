#!/opt/miniconda3/envs/dpo_training/bin/python
"""
Inference script for trained DPO model
"""

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_device():
    """Setup device for M2 Mac (MPS support)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("🚀 Using Metal Performance Shaders (MPS) for M2 GPU")
    else:
        device = torch.device("cpu")
        logger.info("⚠️ MPS not available, using CPU")
    return device

def load_model_and_tokenizer(model_path: str, base_model_name: str):
    """Load the trained model and tokenizer."""
    device = setup_device()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto" if torch.backends.mps.is_available() else None,
    )
    
    # Load LoRA weights if available
    try:
        model = PeftModel.from_pretrained(base_model, model_path)
        logger.info(f"✅ Loaded LoRA model from {model_path}")
    except:
        model = base_model
        logger.info(f"⚠️ Could not load LoRA weights, using base model")
    
    model.eval()
    return model, tokenizer

def generate_ad_text(
    model, 
    tokenizer, 
    prompt: str, 
    max_length: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
):
    """Generate advertising text from prompt."""
    device = next(model.parameters()).device
    
    # Tokenize input
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the original prompt from the output
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()
    
    return generated_text

def main():
    """Main inference function."""
    # Configuration
    config_path = "configs/dpo_config.yaml"
    model_path = "./outputs"  # Path to trained model
    
    try:
        config = load_config(config_path)
        base_model_name = config["model_name"]
    except:
        base_model_name = "google/gemma-2b-it"  # Default fallback
        logger.warning("Could not load config, using default model")
    
    # Load model and tokenizer
    try:
        model, tokenizer = load_model_and_tokenizer(model_path, base_model_name)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.info("Loading base model instead...")
        device = setup_device()
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto" if torch.backends.mps.is_available() else None,
        )
        model.eval()
    
    # Interactive inference
    print("\n" + "="*60)
    print("🚀 AD_Tech_SLM - 広告コピー生成AI")
    print("="*60)
    print("広告コピーのテーマを入力してください（'quit'で終了）")
    print("例: 【テーマ】雨の日でもワクワクするニュースアプリを紹介してください")
    print("="*60)
    
    while True:
        try:
            prompt = input("\n📝 プロンプト: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("👋 お疲れさまでした！")
                break
            
            if not prompt:
                continue
            
            print("🤖 生成中...")
            generated_text = generate_ad_text(model, tokenizer, prompt)
            
            print(f"\n✨ 生成されたコピー:")
            print(f"「{generated_text}」")
            print("-" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 お疲れさまでした！")
            break
        except Exception as e:
            print(f"❌ エラーが発生しました: {e}")
            continue

if __name__ == "__main__":
    main()