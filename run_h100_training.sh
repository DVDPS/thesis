#!/bin/bash
# Run training with H100 optimizations

# Set environment variables for distributed training
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run with larger batch size and gradient accumulation
python3 -m src.thesis.train_h100_optimized --batch-size 2048 --learning-rate 3e-4 --episodes 20000 --eval-interval 100 --log-interval 10 --grad-accumulation-steps 4 --output-dir "h100_ppo_results" 