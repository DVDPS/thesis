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

# Set PyTorch optimizations for H100
export TORCH_CUDNN_V8_API_ENABLED=1  # Enable cuDNN v8 API
export TORCH_CUDNN_BENCHMARK=1      # Enable cuDNN benchmarking
export CUDA_MODULE_LOADING=LAZY     # Lazy loading of CUDA modules

# Add additional memory optimizations
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512,garbage_collection_threshold:0.6"

# Set environment variable to use tensor cores (mixed precision)
export NVIDIA_TF32_OVERRIDE=1

echo "Running DQN training with optimized settings..."

# Run the DQN training script with parameters that match the script's accepted arguments
# Add nohup to prevent terminal close from stopping the training
nohup python -u src/thesis/train_custom_dqn.py \
    --episodes 20000 \
    --max-steps 2000 \
    --buffer-size 200000 \
    --batch-size 8192 \
    --gamma 0.99 \
    --epsilon-start 1.0 \
    --epsilon-end 0.01 \
    --epsilon-decay 0.9995 \
    --learning-rate 1e-4 \
    --log-interval 5 \
    --eval-interval 50 \
    --eval-episodes 10 \
    --output-dir "h100_dqn_results" \
    --seed 42 > dqn_training.log 2>&1 &

echo "Training started in background. You can monitor progress with:"
echo "tail -f dqn_training.log"

# Save the process ID for future reference
DQN_PID=$!
echo "Process ID: $DQN_PID"
echo "To stop training: kill $DQN_PID"

# Don't cleanup at the end since we're running in the background
echo "DQN training script is running in the background" 