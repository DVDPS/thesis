#!/bin/bash
# Run training with H100 optimizations

# Set environment variables for distributed training
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Setting up environment..."

# Simplified process cleanup
echo "Cleaning up any existing processes..."
pkill -f train_h100_optimized || true
sleep 2

# Skip the port killing which might be causing issues
echo "Starting training..."

# Try single GPU training directly since distributed is having issues
echo "Running with single GPU..."
python3 -m src.thesis.train_h100_optimized --batch-size 4096 --learning-rate 3e-4 --episodes 20000 --eval-interval 100 --log-interval 10 --grad-accumulation-steps 4 --output-dir "h100_ppo_results" --single-gpu 