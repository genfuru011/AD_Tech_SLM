#!/opt/miniconda3/envs/dpo_training/bin/python
"""
M2 Mac System Check and Optimization Script
"""

import sys
import platform
import subprocess
import torch

def check_system():
    """Check system information and compatibility."""
    print("🖥️  System Information")
    print("=" * 40)
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Architecture: {platform.machine()}")
    print(f"Python Version: {sys.version}")
    
    # Check if running on Apple Silicon
    if platform.machine() == "arm64":
        print("✅ Running on Apple Silicon (M1/M2)")
    else:
        print("⚠️  Not running on Apple Silicon")
    
    print()

def check_pytorch_mps():
    """Check PyTorch MPS availability."""
    print("🔥 PyTorch MPS Check")
    print("=" * 40)
    print(f"PyTorch Version: {torch.__version__}")
    
    if torch.backends.mps.is_available():
        print("✅ MPS (Metal Performance Shaders) is available")
        print("✅ Can use M2 GPU for training")
        
        # Test MPS functionality
        try:
            device = torch.device("mps")
            x = torch.randn(10, 10).to(device)
            y = torch.randn(10, 10).to(device)
            z = torch.mm(x, y)
            print("✅ MPS tensor operations working")
        except Exception as e:
            print(f"❌ MPS test failed: {e}")
    else:
        print("❌ MPS is not available")
        print("💡 Consider updating PyTorch: pip install --upgrade torch")
    
    print()

def check_memory():
    """Check system memory."""
    print("💾 Memory Information")
    print("=" * 40)
    
    try:
        # Get system memory info (macOS specific)
        result = subprocess.run(
            ["sysctl", "hw.memsize"], 
            capture_output=True, 
            text=True
        )
        if result.returncode == 0:
            mem_bytes = int(result.stdout.split()[-1])
            mem_gb = mem_bytes / (1024**3)
            print(f"Total Memory: {mem_gb:.1f} GB")
            
            if mem_gb >= 16:
                print("✅ Plenty of memory for training")
            elif mem_gb >= 8:
                print("⚠️  Limited memory - use small batch sizes")
                print("💡 Recommended settings:")
                print("   - per_device_train_batch_size: 1")
                print("   - gradient_accumulation_steps: 4-8")
                print("   - fp16: true")
            else:
                print("❌ Insufficient memory for training")
        else:
            print("❓ Could not determine memory size")
    except Exception as e:
        print(f"❓ Could not check memory: {e}")
    
    print()

def optimization_recommendations():
    """Provide optimization recommendations."""
    print("⚡ Optimization Recommendations for M2 8GB")
    print("=" * 50)
    
    recommendations = [
        "✅ Use LoRA (Low-Rank Adaptation) instead of full fine-tuning",
        "✅ Enable mixed precision training (fp16: true)",
        "✅ Use small batch sizes (per_device_train_batch_size: 1)",
        "✅ Use gradient accumulation (gradient_accumulation_steps: 4-8)",
        "✅ Enable gradient checkpointing to save memory",
        "✅ Use smaller models (gemma-2b-it instead of larger models)",
        "✅ Monitor memory usage during training",
        "⚠️  Close unnecessary applications during training",
        "⚠️  Ensure good ventilation for thermal management",
    ]
    
    for rec in recommendations:
        print(f"  {rec}")
    
    print()

def check_dependencies():
    """Check if required dependencies are installed."""
    print("📦 Dependency Check")
    print("=" * 40)
    
    required_packages = [
        "torch",
        "transformers", 
        "datasets",
        "accelerate",
        "peft",
        "trl"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n💡 Install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
    else:
        print("\n✅ All required packages are installed")
    
    print()

def main():
    """Main system check function."""
    print("🚀 AD_Tech_SLM - M2 Mac System Check")
    print("=" * 50)
    print()
    
    check_system()
    check_pytorch_mps()
    check_memory()
    check_dependencies()
    optimization_recommendations()
    
    print("🎯 System check complete!")
    print("Ready for DPO training on M2 Mac! 🚀")

if __name__ == "__main__":
    main()