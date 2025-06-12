#!/opt/miniconda3/bin/python
"""
Test reliable models that work well with current setup
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model(model_name, description):
    """Test a specific model."""
    print(f"\n🔍 Testing: {model_name}")
    print(f"   Description: {description}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print("   ✅ Tokenizer loaded")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print("   ✅ Model loaded")
        print(f"   📊 Parameters: {model.num_parameters():,}")
        
        # Test Japanese text
        test_text = "こんにちは"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        if torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}
            model = model.to("mps")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 5,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   ✅ Test generation: {generated}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("🚀 Testing reliable models for DPO training\n")
    
    # List of models to test (from most to least preferred)
    models = [
        ("elyza/ELYZA-japanese-Llama-2-7b-fast-instruct", "Japanese-tuned Llama 2 (smaller version)"),
        ("rinna/japanese-gpt-neox-3.6b-instruction-sft", "Japanese GPT-NeoX instruction-tuned"),
        ("cyberagent/open-calm-3b", "CyberAgent's Japanese model"),
        ("stabilityai/japanese-stablelm-instruct-alpha-7b", "Stability AI Japanese model"),
        ("microsoft/DialoGPT-medium", "Microsoft DialoGPT (multilingual)"),
        ("gpt2", "GPT-2 base (for testing)"),
    ]
    
    successful_models = []
    
    for model_name, description in models:
        if test_model(model_name, description):
            successful_models.append(model_name)
            print(f"   🎉 {model_name} works!")
        else:
            print(f"   💥 {model_name} failed")
    
    print(f"\n📋 Summary:")
    print(f"   Working models: {len(successful_models)}")
    for model in successful_models:
        print(f"   ✅ {model}")
    
    if successful_models:
        print(f"\n🎯 Recommended model: {successful_models[0]}")
        print(f"   Update your config to use this model!")
    else:
        print(f"\n❌ No models worked. Check your environment setup.")

if __name__ == "__main__":
    main()
