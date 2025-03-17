#!/bin/bash
# Run training with H100 optimizations for 2048 game PPO agent
# Enhanced script with better error handling and fallback mechanisms

set -e  # Exit on error
trap 'echo "Script interrupted or error occurred. Cleaning up..."; cleanup' ERR INT TERM

# Function to clean up processes
cleanup() {
    echo "Performing cleanup..."
    pkill -f train_h100_optimized || true
    pkill -f "python3 -m src.thesis" || true
    pkill -f "python.*torch.distributed" || true
    pkill -f "torch.distributed" || true
    # Kill any python processes using GPUs
    nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9 || true
    sleep 5
    echo "Cleanup completed"
}

# Function to find an available port
find_free_port() {
    local port
    for i in {1..10}; do
        port=$((RANDOM % 5000 + 60000))
        echo "Trying port: $port"
        # Check if port is in use
        if ! netstat -tuln | grep -q ":$port "; then
            echo "Found free port: $port"
            echo $port
            return 0
        fi
        echo "Port $port is in use, trying another one"
    done
    echo "Could not find a free port after 10 attempts"
    return 1
}

# Ensure clean environment
echo "Setting up environment for training on H100 GPUs..."
cleanup

# Set up Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "PYTHONPATH set to: $PYTHONPATH"

# Set environment variables for better NCCL performance
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_BLOCKING_WAIT=1  # Updated name for this env var
export TORCH_DISTRIBUTED_DEBUG=INFO

# Get number of available GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Detected $NUM_GPUS GPUs"

# First, try distributed training if we have multiple GPUs
if [ $NUM_GPUS -gt 1 ]; then
    echo "Attempting distributed training with $NUM_GPUS GPUs..."
    
    # Get a free port
    PORT=$(find_free_port)
    export MASTER_PORT=$PORT
    export MASTER_ADDR="localhost"
    
    echo "Using MASTER_PORT=$MASTER_PORT and MASTER_ADDR=$MASTER_ADDR"
    
    # Start distributed training
    echo "Starting distributed training with $NUM_GPUS GPUs..."
    python3 -m src.thesis.train_h100_optimized \
        --batch-size 4096 \
        --learning-rate 3e-4 \
        --episodes 20000 \
        --eval-interval 100 \
        --log-interval 10 \
        --grad-accumulation-steps 4 \
        --output-dir "h100_ppo_results" \
        --timeout 300  # 5 minutes timeout
    
    DIST_STATUS=$?
    if [ $DIST_STATUS -eq 0 ]; then
        echo "Distributed training completed successfully!"
        exit 0
    else
        echo "Distributed training failed with status $DIST_STATUS"
        echo "Falling back to single GPU training..."
        cleanup
    fi
fi

# If distributed training failed or we only have 1 GPU, use single GPU mode
echo "Starting single GPU training..."
# Get a new free port
PORT=$(find_free_port)
export MASTER_PORT=$PORT
export MASTER_ADDR="localhost"

python3 -m src.thesis.train_h100_optimized \
    --batch-size 4096 \
    --learning-rate 3e-4 \
    --episodes 20000 \
    --eval-interval 100 \
    --log-interval 10 \
    --grad-accumulation-steps 4 \
    --output-dir "h100_ppo_results" \
    --single-gpu

# Final cleanup
cleanup
echo "Training script completed" 