#!/opt/miniconda3/envs/dpo_training/bin/python
"""
Test script to verify model loading and MPS support
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml

def test_mps():
    """Test MPS availability."""
    print("🔍 Testing MPS availability...")
    if torch.backends.mps.is_available():
        print("✅ MPS is available!")
        print(f"   Device: {torch.device('mps')}")
        return True
    else:
        print("❌ MPS is not available, using CPU")
        return False

def test_model_loading(model_name):
    """Test model loading."""
    print(f"\n🔍 Testing model loading: {model_name}")
    
    try:
        # Load tokenizer
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("   ✅ Tokenizer loaded successfully")
        
        # Load model
        print("   Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print("   ✅ Model loaded successfully")
        
        # Test inference
        print("   Testing inference...")
        test_text = "今日は良い天気です。"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        # Move to MPS if available
        if torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}
            model = model.to("mps")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + 10,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"   ✅ Generated text: {generated_text}")
        
        # Memory usage
        if torch.backends.mps.is_available():
            print(f"   📊 Model parameters: {model.num_parameters():,}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Model Loading Test\n")
    
    # Test MPS
    mps_available = test_mps()
    
    # Load config
    with open("configs/dpo_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    model_name = config["model_name"]
    
    # Test model loading
    success = test_model_loading(model_name)
    
    if success:
        print(f"\n🎉 All tests passed! {model_name} is ready for DPO training.")
    else:
        print(f"\n💥 Tests failed. Please check the model configuration.")

if __name__ == "__main__":
    main()
