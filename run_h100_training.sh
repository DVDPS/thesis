#!/bin/bash
# Run training with H100 optimizations

# Set environment variables for distributed training
export PYTHONPATH=$PYTHONPATH:$(pwd)

echo "Setting up environment for distributed training on 4 GPUs..."

# Kill any existing processes more thoroughly
echo "Cleaning up any existing processes..."
pkill -f train_h100_optimized || true
pkill -f "python3 -m src.thesis" || true
pkill -f "python.*torch.distributed" || true
sleep 3

# Set environment variables for better NCCL performance
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_BLOCKING_WAIT=1
export TORCH_DISTRIBUTED_DEBUG=INFO

# Choose a random high port in the 50000-60000 range to avoid conflicts
RANDOM_PORT=$((RANDOM % 10000 + 50000))
echo "Using random port: $RANDOM_PORT"
export MASTER_PORT=$RANDOM_PORT

# Start distributed training with explicit world size and rank settings
echo "Starting distributed training with 4 GPUs..."
python3 -m src.thesis.train_h100_optimized \
    --batch-size 4096 \
    --learning-rate 3e-4 \
    --episodes 20000 \
    --eval-interval 100 \
    --log-interval 10 \
    --grad-accumulation-steps 4 \
    --output-dir "h100_ppo_results" \
    --timeout 120

# If the above fails, try with single GPU
if [ $? -ne 0 ]; then
    echo "Distributed training failed. Cleaning up any remaining processes..."
    pkill -f train_h100_optimized || true
    pkill -f "python3 -m src.thesis" || true
    pkill -f "python.*torch.distributed" || true
    sleep 3
    
    echo "Starting single GPU training as fallback..."
    python3 -m src.thesis.train_h100_optimized \
        --batch-size 4096 \
        --learning-rate 3e-4 \
        --episodes 20000 \
        --eval-interval 100 \
        --log-interval 10 \
        --grad-accumulation-steps 4 \
        --output-dir "h100_ppo_results" \
        --single-gpu
fi 