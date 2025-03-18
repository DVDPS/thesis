#!/bin/bash
# Run DQN training with H100 optimizations for 2048 game

set -e  # Exit on error
trap 'echo "Script interrupted or error occurred. Cleaning up..."; cleanup' ERR INT TERM

# Function to clean up processes
cleanup() {
    echo "Performing cleanup..."
    pkill -f train_custom_dqn || true
    pkill -f "python3 -m src.thesis" || true
    # Kill any python processes using GPUs
    nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9 || true
    sleep 2
    echo "Cleanup completed"
}

# Ensure clean environment
echo "Setting up environment for DQN training on H100 GPU..."
cleanup

# Set PYTHONPATH correctly to include the root directory (one level above thesis folder)
cd ~/
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd thesis
echo "PYTHONPATH set to: ${PYTHONPATH}"

# Install required packages if not already installed
python -m pip install matplotlib tqdm || true
echo "Checked required packages"

# Optimized CUDA settings for H100
export CUDA_VISIBLE_DEVICES=0  # Use only the first GPU
export TF_FORCE_GPU_ALLOW_GROWTH=true
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

echo "Running DQN training with optimized settings..."

# Run the DQN training script with parameters that match the script's accepted arguments
python -u src/thesis/train_custom_dqn.py \
    --episodes 20000 \
    --max-steps 2000 \
    --buffer-size 100000 \
    --batch-size 4096 \
    --gamma 0.99 \
    --epsilon-start 1.0 \
    --epsilon-end 0.01 \
    --epsilon-decay 0.9999 \
    --learning-rate 3e-4 \
    --log-interval 10 \
    --eval-interval 100 \
    --eval-episodes 5 \
    --output-dir "h100_dqn_results" \
    --seed 42
    
# Final cleanup
cleanup
echo "DQN training script completed" 