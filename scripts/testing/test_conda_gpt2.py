#!/opt/miniconda3/envs/dpo_training/bin/python
"""
Test GPT-2 model in conda environment
"""

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")

try:
    print("Testing _lzma module...")
    import _lzma
    print("✅ _lzma module works!")
    
    print("\nTesting transformers import...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("✅ Transformers imported successfully")
    
    # Test GPT-2
    model_name = "gpt2"
    print(f"\n🔍 Testing model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("✅ Tokenizer loaded")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    print("✅ Model loaded")
    print(f"📊 Parameters: {model.num_parameters():,}")
    
    # Test generation
    test_text = "Hello, this is a test"
    inputs = tokenizer(test_text, return_tensors="pt")
    
    if torch.backends.mps.is_available():
        inputs = {k: v.to("mps") for k, v in inputs.items()}
        model = model.to("mps")
        print("✅ Model moved to MPS")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=inputs["input_ids"].shape[1] + 10,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"✅ Generated text: {generated}")
    
    print("\n🎉 GPT-2 model works perfectly in conda environment!")
    print("   Ready for real DPO training!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
