#!/bin/bash

# DPO Training Environment Setup Script for MacBook Air M2
# This script sets up the Python environment for DPO training

set -e

echo "🚀 Setting up DPO Training Environment for AD_Tech_SLM"
echo "Optimized for MacBook Air M2 8GB"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📍 Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python packages..."
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p models
mkdir -p outputs
mkdir -p configs
mkdir -p scripts
mkdir -p notebooks

echo "✅ Environment setup completed!"
echo ""
echo "To activate the environment, run:"
echo "source venv/bin/activate"
echo ""
echo "To start training, check the scripts/ directory for training examples."