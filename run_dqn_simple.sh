#!/bin/bash
# Simple DQN training script for 2048 game
# Uses basic settings that should work on any CUDA-enabled GPU

set -e  # Exit on error
trap 'echo "Script interrupted or error occurred. Cleaning up..."; cleanup' ERR INT TERM

# Function to clean up processes
cleanup() {
    echo "Performing cleanup..."
    pkill -f train_custom_dqn.py || true
    pkill -f python.*train_custom_dqn.py || true
    sleep 2
    echo "Cleanup completed"
}

# Ensure clean environment
echo "Setting up environment for DQN training..."
cleanup

# Set PYTHONPATH correctly
cd ~/
export PYTHONPATH=$PYTHONPATH:$(pwd)
cd thesis
echo "PYTHONPATH set to: ${PYTHONPATH}"

# Install required packages if needed
python -m pip install matplotlib tqdm || true
echo "Checked required packages"

echo "Running DQN training with standard settings..."

# Run the original DQN training script with appropriate batch size for H100
python -u src/thesis/train_custom_dqn.py \
    --episodes 20000 \
    --max-steps 2000 \
    --buffer-size 100000 \
    --batch-size 2048 \
    --gamma 0.99 \
    --epsilon-start 1.0 \
    --epsilon-end 0.01 \
    --epsilon-decay 0.9995 \
    --learning-rate 2e-4 \
    --log-interval 10 \
    --eval-interval 100 \
    --eval-episodes 5 \
    --output-dir "dqn_results" \
    --seed 42

echo "DQN training script completed" 