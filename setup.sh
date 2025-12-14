#!/bin/bash

# Vietnamese RAG System - Quick Setup Script
# This script helps you set up the project quickly

set -e  # Exit on error

echo "=========================================="
echo "Vietnamese RAG System - Quick Setup"
echo "=========================================="
echo ""

# Check Python version
echo "[1/5] Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python $python_version detected"

# Create virtual environment
echo ""
echo "[2/5] Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "[3/5] Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install dependencies
echo ""
echo "[4/5] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✓ Dependencies installed"

# Check model file
echo ""
echo "[5/5] Checking model file..."
if [ -f "qwen3_06b.gguf" ]; then
    echo "✓ Model file found"
else
    echo "⚠ WARNING: Model file 'qwen3_06b.gguf' not found!"
    echo ""
    echo "Please download the model file and place it in the project root."
    echo "You can download from:"
    echo "  - Hugging Face: https://huggingface.co/Qwen"
    echo "  - Or convert from original model using llama.cpp"
fi

echo ""
echo "=========================================="
echo "Setup Complete! 🎉"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Run example: python example_usage.py"
echo "  3. Or query directly: python rag_ultimate_v2.py \"Your question?\""
echo ""
echo "Configuration (optional):"
echo "  export DOC_PATH=\"path/to/your/document.txt\""
echo "  export K_TOP=5"
echo "  export TEMPERATURE=0.1"
echo ""
echo "Happy coding! 🚀"
