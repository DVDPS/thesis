#!/bin/bash
# Run training with H100 optimizations

# Set environment variables for distributed training
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Kill any existing processes that might be using the ports
pkill -f train_h100_optimized

# Wait a moment for processes to terminate
sleep 2

# Try to run with distributed training first
echo "Attempting distributed training across all GPUs..."
python3 -m src.thesis.train_h100_optimized --batch-size 4096 --learning-rate 3e-4 --episodes 20000 --eval-interval 100 --log-interval 10 --grad-accumulation-steps 4 --output-dir "h100_ppo_results"

# If the above fails, try with single GPU
if [ $? -ne 0 ]; then
    echo "Distributed training failed. Falling back to single GPU..."
    python3 -m src.thesis.train_h100_optimized --batch-size 4096 --learning-rate 3e-4 --episodes 20000 --eval-interval 100 --log-interval 10 --grad-accumulation-steps 4 --output-dir "h100_ppo_results" --single-gpu
fi 