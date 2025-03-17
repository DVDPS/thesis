#!/bin/bash
# Script to train the PPO agent on an H100 GPU
# Usage: bash train_ppo_h100.sh [output_dir]

# Set default output directory
OUTPUT_DIR=${1:-"ppo_h100_results"}

# Create the output directory
mkdir -p $OUTPUT_DIR

# Set environment variables for better GPU utilization
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Use all GPUs
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Dynamic memory allocation

# Print GPU information
echo "===== GPU Information ====="
nvidia-smi
echo "=========================="

# Set PyTorch environment variables for H100
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Handle large tensor allocations
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # For debugging

# Run training with optimal H100 settings
python -m src.thesis.train_ppo \
    --mixed-precision \
    --use-data-parallel \
    --total-timesteps 5000000 \
    --timesteps-per-update 4096 \
    --batch-size 512 \
    --hidden-dim 512 \
    --learning-rate 3e-4 \
    --gamma 0.995 \
    --vf-coef 0.5 \
    --ent-coef 0.01 \
    --max-grad-norm 0.5 \
    --update-epochs 4 \
    --eval-interval 25 \
    --eval-episodes 10 \
    --output-dir $OUTPUT_DIR

echo "Training complete. Results saved to $OUTPUT_DIR"