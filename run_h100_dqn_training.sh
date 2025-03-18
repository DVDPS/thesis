#!/bin/bash
# Run DQN training with basic settings for 2048 game

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
echo "Setting up environment for DQN training..."
cleanup

# Set PYTHONPATH correctly to include the root directory (one level above thesis folder)
cd ~/
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd thesis
echo "PYTHONPATH set to: ${PYTHONPATH}"

# Install required packages if not already installed
python -m pip install matplotlib tqdm || true
echo "Checked required packages"

echo "Running DQN training..."

# Run the DQN training script with reduced batch size
python -u src/thesis/train_custom_dqn.py \
    --episodes 20000 \
    --max-steps 2000 \
    --buffer-size 100000 \
    --batch-size 4096 \
    --gamma 0.99 \
    --epsilon-start 1.0 \
    --epsilon-end 0.01 \
    --epsilon-decay 0.9995 \
    --learning-rate 1e-4 \
    --log-interval 10 \
    --eval-interval 100 \
    --eval-episodes 5 \
    --output-dir "h100_dqn_results" \
    --seed 42

echo "DQN training completed" 