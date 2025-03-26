#!/bin/bash
# Script to evaluate a trained model with different MCTS simulation counts
# Usage: bash evaluate_agent.sh [checkpoint_path] [output_dir]

# Validate inputs
if [ -z "$1" ]; then
    echo "Error: Please provide a checkpoint path"
    echo "Usage: bash evaluate_agent.sh [checkpoint_path] [output_dir]"
    exit 1
fi

# Set paths
CHECKPOINT_PATH=$1
OUTPUT_DIR=${2:-"evaluation_results"}

# Create output directory
mkdir -p $OUTPUT_DIR

# Define simulation counts to test
SIM_COUNTS=(0 50 100 200 400 800)

# Print info
echo "===== Evaluation Configuration ====="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "====================================="

# Run evaluations for each simulation count
for sims in "${SIM_COUNTS[@]}"; do
    echo "Running evaluation with $sims simulations..."
    
    # Create simulation-specific directory
    SIM_DIR="$OUTPUT_DIR/sims_$sims"
    mkdir -p $SIM_DIR
    
    # Run evaluation
    if [ $sims -eq 0 ]; then
        # Without MCTS
        python -m src.thesis.main \
            --checkpoint $CHECKPOINT_PATH \
            --games 25 \
            --output-dir $SIM_DIR
    else
        # With MCTS
        python -m src.thesis.main \
            --checkpoint $CHECKPOINT_PATH \
            --mcts-simulations $sims \
            --games 25 \
            --output-dir $SIM_DIR
    fi
    
    echo "Completed evaluation with $sims simulations"
    echo "----------------------------------------"
done

# Run analysis on all results
python -m src.thesis.utils.evaluation.analyze_results \
    --results-dir $OUTPUT_DIR \
    --output-dir "$OUTPUT_DIR/analysis"

echo "Evaluation complete. Results saved to $OUTPUT_DIR"