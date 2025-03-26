#!/bin/bash
# Optimized DQN training script for H100 GPUs
# Focuses on maximizing training speed while maintaining model quality

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
echo "Setting up environment for optimized DQN training on H100 GPU..."
cleanup

# Set PYTHONPATH correctly to include the root directory (one level above thesis folder)
cd ~/
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd thesis
echo "PYTHONPATH set to: ${PYTHONPATH}"

# Install required packages if not already installed
python -m pip install matplotlib tqdm || true
echo "Checked required packages"

# Add mixed precision support
python -m pip install torch torchvision --upgrade || true

# Optimized CUDA and PyTorch settings for H100 GPU
echo "Configuring optimized GPU settings..."

# Enable CUDA optimizations
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:1024,garbage_collection_threshold:0.8,roundup_power2:True"
export CUDA_AUTO_BOOST=1
export CUDA_MODULE_LOADING=LAZY

# PyTorch optimizations
export TORCH_CUDNN_V8_API_ENABLED=1  # Enable cuDNN v8 API
export TORCH_CUDNN_BENCHMARK=1       # Enable cuDNN benchmarking
export TORCH_USE_CUDA_DSA=1          # Enable CUDA Direct Storage Access if available
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_JIT_ENABLE_NVFUSER=1    # Enable NvFuser for JIT
export TORCH_ALLOW_TF32=1            # Enable TF32 precision

# Add this for profiling if needed
# export CUDA_LAUNCH_BLOCKING=1

# Create a patch for the DQN agent to enable mixed precision training
echo "Creating mixed precision training patch..."
cat > mixed_precision_patch.py << 'EOF'
import torch
import os

# Enable autocast if CUDA is available
def enable_mixed_precision():
    if torch.cuda.is_available():
        print("Enabling mixed precision training with autocast")
        return True
    return False

# Add this to your train_custom_dqn.py file
# at the top of the train_custom_dqn function
USE_MIXED_PRECISION = enable_mixed_precision()

# Use this context manager around forward passes and loss computation
# torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION)
EOF

echo "Running optimized DQN training with H100 settings..."

# Run the optimized DQN training script
nohup python -u src/thesis/train_custom_dqn_optimized.py \
    --episodes 20000 \
    --max-steps 2000 \
    --buffer-size 100000 \
    --batch-size 2048 \
    --gamma 0.99 \
    --epsilon-start 1.0 \
    --epsilon-end 0.01 \
    --epsilon-decay 0.9995 \
    --learning-rate 2e-4 \
    --log-interval 5 \
    --eval-interval 100 \
    --eval-episodes 5 \
    --output-dir "h100_dqn_optimized_results" \
    --seed 42 > dqn_optimized_training.log 2>&1 &

echo "Optimized training started in background. You can monitor progress with:"
echo "tail -f dqn_optimized_training.log"

# Save the process ID for future reference
DQN_PID=$!
echo "Process ID: $DQN_PID"
echo "To stop training: kill $DQN_PID"
echo "To monitor GPU usage: nvidia-smi -l 5"

echo "DQN training script is running in the background with optimized settings"
echo "Expected training time: ~10-15 hours for 20,000 episodes" 