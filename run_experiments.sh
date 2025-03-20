#!/bin/bash
# Script to run multiple Transformer-based PPO experiments with different configurations

# Make script exit on any error
set -e

# Base directory for all experiments
BASE_DIR="transformer_ppo_experiments"
mkdir -p "$BASE_DIR"

# Log file
LOG_FILE="$BASE_DIR/experiments.log"
echo "Starting experiments at $(date)" > "$LOG_FILE"

# Function to run a single experiment
run_experiment() {
    exp_name=$1
    embed_dim=$2
    num_heads=$3
    num_layers=$4
    learning_rate=$5
    ent_coef=$6
    timesteps=$7
    additional_args=$8

    # Create experiment directory
    exp_dir="$BASE_DIR/$exp_name"
    mkdir -p "$exp_dir"
    
    echo "==============================================" | tee -a "$LOG_FILE"
    echo "Starting experiment: $exp_name" | tee -a "$LOG_FILE"
    echo "Embedding dimension: $embed_dim" | tee -a "$LOG_FILE"
    echo "Number of heads: $num_heads" | tee -a "$LOG_FILE"
    echo "Number of layers: $num_layers" | tee -a "$LOG_FILE"
    echo "Learning rate: $learning_rate" | tee -a "$LOG_FILE"
    echo "Entropy coefficient: $ent_coef" | tee -a "$LOG_FILE"
    echo "Total timesteps: $timesteps" | tee -a "$LOG_FILE"
    echo "Additional args: $additional_args" | tee -a "$LOG_FILE"
    echo "==============================================" | tee -a "$LOG_FILE"
    
    # Build command
    cmd="python -m src.thesis.train_transformer_ppo \
        --total-timesteps $timesteps \
        --embed-dim $embed_dim \
        --num-heads $num_heads \
        --num-layers $num_layers \
        --learning-rate $learning_rate \
        --ent-coef $ent_coef \
        --output-dir $exp_dir \
        --mixed-precision $additional_args"
    
    # Run the experiment
    echo "Running command: $cmd" | tee -a "$LOG_FILE"
    eval $cmd
    
    # Log completion
    echo "Experiment $exp_name completed at $(date)" | tee -a "$LOG_FILE"
    
    # Get the best tile from the training log
    best_tile=$(grep "Best Max Tile" "$exp_dir/training.log" | tail -n 1 | sed -E 's/.*Best Max Tile: ([0-9]+).*/\1/')
    
    # Log best result
    echo "Best tile achieved: $best_tile" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# Check if CUDA is available and set appropriate flag
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    CUDA_FLAG="--use-data-parallel"
    echo "CUDA available, using data parallel if multiple GPUs are present" | tee -a "$LOG_FILE"
else
    CUDA_FLAG=""
    echo "CUDA not available, running on CPU" | tee -a "$LOG_FILE"
fi

# Run a small experiment to verify setup
run_experiment "test_run" 128 2 2 0.0003 0.01 10000 "$CUDA_FLAG"

# Verify the test run completed successfully
if [ $? -eq 0 ]; then
    echo "Test run completed successfully, proceeding with main experiments" | tee -a "$LOG_FILE"
else
    echo "Test run failed, please check the logs" | tee -a "$LOG_FILE"
    exit 1
fi

# Main experiments - Embedding dimension variations
run_experiment "embed_dim_256" 256 4 4 0.0003 0.01 500000 "$CUDA_FLAG"
run_experiment "embed_dim_512" 512 4 4 0.0003 0.01 500000 "$CUDA_FLAG"

# Number of heads variations
run_experiment "heads_2" 256 2 4 0.0003 0.01 500000 "$CUDA_FLAG"
run_experiment "heads_8" 256 8 4 0.0003 0.01 500000 "$CUDA_FLAG"

# Number of layers variations
run_experiment "layers_2" 256 4 2 0.0003 0.01 500000 "$CUDA_FLAG"
run_experiment "layers_6" 256 4 6 0.0003 0.01 500000 "$CUDA_FLAG"

# Learning rate variations
run_experiment "lr_low" 256 4 4 0.0001 0.01 500000 "$CUDA_FLAG"
run_experiment "lr_high" 256 4 4 0.0006 0.01 500000 "$CUDA_FLAG"

# Entropy coefficient variations
run_experiment "ent_low" 256 4 4 0.0003 0.005 500000 "$CUDA_FLAG"
run_experiment "ent_high" 256 4 4 0.0003 0.03 500000 "$CUDA_FLAG"

# Best configuration based on previous results (with more timesteps)
run_experiment "best_config" 512 4 4 0.0003 0.01 1000000 "$CUDA_FLAG"

# Print summary
echo "All experiments completed!" | tee -a "$LOG_FILE"
echo "Summary of best tiles achieved:" | tee -a "$LOG_FILE"

for exp_dir in "$BASE_DIR"/*; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        best_tile=$(grep "Best Max Tile" "$exp_dir/training.log" | tail -n 1 | sed -E 's/.*Best Max Tile: ([0-9]+).*/\1/')
        if [ -z "$best_tile" ]; then
            best_tile="N/A"
        fi
        echo "$exp_name: $best_tile" | tee -a "$LOG_FILE"
    fi
done

# Generate a simple visualization of results
echo "Generating results visualization..."
python -c "
import os
import matplotlib.pyplot as plt
import numpy as np
import re

base_dir = '$BASE_DIR'
exp_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
exp_dirs.sort()

exp_names = []
best_tiles = []

for exp in exp_dirs:
    log_path = os.path.join(base_dir, exp, 'training.log')
    if not os.path.exists(log_path):
        continue
        
    with open(log_path, 'r') as f:
        log_content = f.read()
    
    # Extract the best tile
    best_tile_matches = re.findall(r'Best Max Tile: ([0-9]+)', log_content)
    if best_tile_matches:
        best_tile = int(best_tile_matches[-1])
    else:
        best_tile = 0
    
    exp_names.append(exp)
    best_tiles.append(best_tile)

# Create bar chart
plt.figure(figsize=(12, 6))
bars = plt.bar(exp_names, best_tiles)
plt.xlabel('Experiment')
plt.ylabel('Best Max Tile')
plt.title('Comparison of Best Tiles Across Experiments')
plt.xticks(rotation=45, ha='right')

# Add the values on top of the bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 20,
            f'{height}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'experiment_results.png'))
print(f'Results visualization saved to {os.path.join(base_dir, \"experiment_results.png\")}')
"

echo "Done! Check $BASE_DIR/experiment_results.png for a visual comparison of results." 