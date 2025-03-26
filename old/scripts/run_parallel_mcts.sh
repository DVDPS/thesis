#!/bin/bash
# Script to run parallel MCTS evaluation optimized for H100
# Usage: bash run_parallel_mcts.sh [checkpoint_path] [num_games]

# Validate inputs
if [ -z "$1" ]; then
    echo "Error: Please provide a checkpoint path"
    echo "Usage: bash run_parallel_mcts.sh [checkpoint_path] [num_games]"
    exit 1
fi

# Set parameters
CHECKPOINT_PATH=$1
NUM_GAMES=${2:-20}
OUTPUT_DIR="parallel_mcts_results"
NUM_WORKERS=8
BATCH_SIZE=32
SIM_COUNT=800

# Create output directory
mkdir -p $OUTPUT_DIR

# Print configuration
echo "===== Parallel MCTS Configuration ====="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Number of games: $NUM_GAMES"
echo "Output directory: $OUTPUT_DIR"
echo "Number of workers: $NUM_WORKERS"
echo "Batch size: $BATCH_SIZE"
echo "Simulation count: $SIM_COUNT"
echo "======================================="

# Run the evaluation
python -m src.thesis.utils.evaluation.enhanced_mcts_evaluation \
    --checkpoint $CHECKPOINT_PATH \
    --games $NUM_GAMES \
    --simulations $SIM_COUNT \
    --num-workers $NUM_WORKERS \
    --batch-size $BATCH_SIZE \
    --output-dir $OUTPUT_DIR \
    --use-parallel

echo "Evaluation complete. Results saved to $OUTPUT_DIR"