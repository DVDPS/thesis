#!/bin/bash
# Run DQN training with CPU-only mode to avoid CUDA library issues

set -e  # Exit on error
trap 'echo "Script interrupted or error occurred. Cleaning up..."; cleanup' ERR INT TERM

# Function to clean up processes
cleanup() {
    echo "Performing cleanup..."
    pkill -f train_custom_dqn.py || true
    sleep 2
    echo "Cleanup completed"
}

# Ensure clean environment
echo "Setting up environment for DQN training..."
cleanup

# Set PYTHONPATH correctly to include the root directory (one level above thesis folder)
export PYTHONPATH=$PYTHONPATH:/home/ubuntu
echo "PYTHONPATH set to: ${PYTHONPATH}"

# Force CPU-only mode to avoid CUDA library issues
export CUDA_VISIBLE_DEVICES=""
# Ensure PyTorch uses CPU
export TORCH_DEVICE="cpu"

# Install required packages if not already installed
pip install matplotlib tqdm || true
echo "Checked required packages"

echo "Running DQN training in CPU-only mode..."

# Start DQN training with log redirection
python -u src/thesis/train_custom_dqn.py \
    --episodes 20000 \
    --max-steps 2000 \
    --buffer-size 50000 \
    --batch-size 64 \
    --gamma 0.99 \
    --epsilon-start 1.0 \
    --epsilon-end 0.01 \
    --epsilon-decay 0.9995 \
    --learning-rate 1e-4 \
    --log-interval 10 \
    --eval-interval 100 \
    --eval-episodes 5 \
    --output-dir "dqn_cpu_results" \
    --seed 42 > dqn_training.log 2>&1 &

PID=$!
echo "DQN training started in background with PID: $PID"
echo "Check training progress with: tail -f dqn_training.log" 