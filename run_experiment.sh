#!/bin/bash

echo "=========================================="
echo "Self-Pruning Neural Network Experiment"
echo "=========================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo ""

# Run training
python train.py

echo ""
echo "=========================================="
echo "Running Evaluation"
echo "=========================================="
echo ""

# Run evaluation
python evaluate.py

echo ""
echo "=========================================="
echo "Experiment Complete!"
echo "=========================================="
echo ""
echo "Results saved to results/"
echo "Check the following files:"
echo "  - results/summary.json"
echo "  - results/model_comparison.png"
echo "  - results/lambda_*/gate_distribution.png"
echo "  - results/lambda_*/training_curves.png"
echo ""
