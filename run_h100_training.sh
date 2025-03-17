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
    # Check if netstat is available
    if command -v netstat &> /dev/null; then
        local port
        for i in {1..10}; do
            port=$((RANDOM % 5000 + 60000))
            # Check if port is in use (quietly)
            if ! netstat -tuln | grep -q ":$port "; then
                # Only return the port number, no other text
                echo "$port"
                return 0
            fi
        done
    else
        # Fallback: just use a random port without checking
        local port=$((RANDOM % 5000 + 60000))
        echo "$port"
        return 0
    fi
    
    # If we get here, we couldn't find a free port
    echo "65432"  # Return a default port as last resort
    return 1
}

# Ensure clean environment
echo "Setting up environment for training on H100 GPUs..."
cleanup

# Clean up environment variables that might interfere with distributed training
unset MASTER_PORT
unset MASTER_ADDR
unset WORLD_SIZE
unset RANK
unset LOCAL_RANK

# Set PYTHONPATH to include the current directory
export PYTHONPATH=${PYTHONPATH}:$(pwd)
echo "PYTHONPATH set to: ${PYTHONPATH}"

# Select a random port in a higher range (60000-65000) to avoid conflicts
PORT=$(( 60000 + RANDOM % 5000 ))
echo "Using port: ${PORT}"

# Set environment variables for distributed training
export MASTER_PORT=${PORT}
export MASTER_ADDR="localhost"
export NCCL_IB_DISABLE=1 # Disable Infiniband for better stability on VMs
echo "Using MASTER_PORT=${MASTER_PORT} and MASTER_ADDR=${MASTER_ADDR}"

# Count available GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected ${NUM_GPUS} GPUs"

if [ $NUM_GPUS -gt 1 ]; then
    echo "Attempting distributed training with ${NUM_GPUS} GPUs..."
    
    # Run the training script with distributed settings
    # Increase timeout and add parameters for stability
    python -u src/thesis/train_h100_optimized.py \
        --batch-size 4096 \
        --learning-rate 3e-4 \
        --episodes 20000 \
        --clip-ratio 0.2 \
        --gamma 0.99 \
        --gae-lambda 0.95 \
        --grad-accumulation-steps 4 \
        --timeout 300 \
        --eval-interval 100 \
        --eval-episodes 5
else
    echo "Only 1 GPU detected, running in single-GPU mode..."
    
    # Run the training script in single-GPU mode
    python -u src/thesis/train_h100_optimized.py \
        --batch-size 2048 \
        --learning-rate 3e-4 \
        --episodes 20000 \
        --single-gpu \
        --clip-ratio 0.2 \
        --gamma 0.99 \
        --gae-lambda 0.95 \
        --grad-accumulation-steps 4 \
        --eval-interval 100 \
        --eval-episodes 5
fi

# Final cleanup
cleanup
echo "Training script completed" 