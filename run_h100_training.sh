#!/bin/bash
# Run training with H100 optimizations

# Set environment variables for distributed training
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Kill any existing processes that might be using the ports
echo "Killing any existing training processes..."
pkill -f train_h100_optimized
pkill -f "python3 -m src.thesis"
pkill -f "torch.distributed"

# Kill any processes using the common ports
for port in $(seq 29500 29510) $(seq 35000 35100) $(seq 40000 40100) $(seq 45000 55100); do
  fuser -k $port/tcp 2>/dev/null
done

# Wait longer for processes to terminate
echo "Waiting for processes to terminate..."
sleep 5

# Try to run with distributed training first
echo "Attempting distributed training across all GPUs..."
python3 -m src.thesis.train_h100_optimized --batch-size 4096 --learning-rate 3e-4 --episodes 20000 --eval-interval 100 --log-interval 10 --grad-accumulation-steps 4 --output-dir "h100_ppo_results" --timeout 60

# Check if distributed training failed
if [ $? -ne 0 ]; then
    echo "Distributed training failed. Cleaning up any remaining processes..."
    pkill -f train_h100_optimized
    pkill -f "python3 -m src.thesis"
    pkill -f "torch.distributed"
    sleep 3
    
    echo "Falling back to single GPU training..."
    python3 -m src.thesis.train_h100_optimized --batch-size 4096 --learning-rate 3e-4 --episodes 20000 --eval-interval 100 --log-interval 10 --grad-accumulation-steps 4 --output-dir "h100_ppo_results" --single-gpu
fi 