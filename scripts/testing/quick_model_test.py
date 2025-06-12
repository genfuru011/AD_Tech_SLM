#!/opt/miniconda3/bin/python
"""
Quick model test
"""

import sys
import torch

print("Testing basic imports...")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✅ Transformers imported successfully")
except ImportError as e:
    print(f"❌ Failed to import transformers: {e}")
    sys.exit(1)

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Test model loading
model_name = "microsoft/Phi-3-mini-4k-instruct"
print(f"\nTesting model: {model_name}")

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("✅ Tokenizer loaded")
    
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto"
    )
    print("✅ Model loaded")
    print(f"Model parameters: {model.num_parameters():,}")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    
    # Try alternative model
    print("\nTrying alternative model: google/gemma-2-2b-it")
    try:
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-2-2b-it",
            torch_dtype=torch.float16,
            device_map="auto"
        )
        print("✅ Alternative model loaded successfully")
        print(f"Model parameters: {model.num_parameters():,}")
        
        # Update config recommendation
        print("\n📝 Recommendation: Use google/gemma-2-2b-it model")
        
    except Exception as e2:
        print(f"❌ Alternative model also failed: {e2}")
        
        # Try Qwen2.5
        print("\nTrying Qwen2.5-1.5B-Instruct...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
            model = AutoModelForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-1.5B-Instruct",
                torch_dtype=torch.float16,
                device_map="auto"
            )
            print("✅ Qwen2.5 model loaded successfully")
            print(f"Model parameters: {model.num_parameters():,}")
            print("\n📝 Recommendation: Use Qwen/Qwen2.5-1.5B-Instruct model")
            
        except Exception as e3:
            print(f"❌ Qwen2.5 also failed: {e3}")
            print("\n💡 Consider using a more basic model like 'distilgpt2' for testing")

print("\nTest completed!")
