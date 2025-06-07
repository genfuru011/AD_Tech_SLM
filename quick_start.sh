#!/bin/bash

# Quick Start Script for DPO Training Environment
# AD_Tech_SLM Project

set -e

echo "🚀 AD_Tech_SLM DPO Training - Quick Start"
echo "========================================"

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -f "requirements.txt" ]; then
    echo "❌ Please run this script from the AD_Tech_SLM project root directory"
    exit 1
fi

echo "1️⃣ Setting up Python environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

echo "2️⃣ Activating virtual environment..."
source venv/bin/activate
echo "✅ Virtual environment activated"

echo "3️⃣ Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"

echo "4️⃣ Creating project directories..."
mkdir -p data/raw data/processed models outputs configs scripts notebooks
echo "✅ Project directories created"

echo "5️⃣ Validating sample dataset..."
python test_data.py data/sample_dpo_dataset.jsonl
echo "✅ Sample dataset validated"

echo ""
echo "🎉 Setup Complete!"
echo "==================="
echo ""
echo "Next steps:"
echo "1. Activate the environment: source venv/bin/activate"
echo "2. Check the sample dataset: python scripts/validate_data.py data/sample_dpo_dataset.jsonl"
echo "3. Start training: python scripts/train_dpo.py"
echo "4. Test inference: python scripts/inference.py"
echo ""
echo "📚 Documentation: README_DPO_TRAINING.md"
echo "💡 Jupyter Notebook: notebooks/dpo_training_experiment.ipynb"
echo ""
echo "Happy training! 🎯"