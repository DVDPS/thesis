#!/bin/bash
# Script to train the Transformer-based PPO agent for 2048 game

# Make script exit on any error
set -e

# Parse command line arguments
OUTPUT_DIR="transformer_ppo_results"
TIMESTEPS=1000000
EMBED_DIM=256
NUM_HEADS=4
NUM_LAYERS=4
LEARNING_RATE=0.0003
SEED=42
MIXED_PRECISION=true
DATA_PARALLEL=false
CHECKPOINT=""

# Process command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --output-dir)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        --timesteps)
            TIMESTEPS="$2"
            shift
            shift
            ;;
        --embed-dim)
            EMBED_DIM="$2"
            shift
            shift
            ;;
        --num-heads)
            NUM_HEADS="$2"
            shift
            shift
            ;;
        --num-layers)
            NUM_LAYERS="$2"
            shift
            shift
            ;;
        --lr|--learning-rate)
            LEARNING_RATE="$2"
            shift
            shift
            ;;
        --seed)
            SEED="$2"
            shift
            shift
            ;;
        --no-mixed-precision)
            MIXED_PRECISION=false
            shift
            ;;
        --use-data-parallel)
            DATA_PARALLEL=true
            shift
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --output-dir DIR       Output directory for results (default: transformer_ppo_results)"
            echo "  --timesteps N          Total timesteps to train (default: 1000000)"
            echo "  --embed-dim N          Embedding dimension (default: 256)"
            echo "  --num-heads N          Number of attention heads (default: 4)"
            echo "  --num-layers N         Number of transformer layers (default: 4)"
            echo "  --lr, --learning-rate R  Learning rate (default: 0.0003)"
            echo "  --seed N               Random seed (default: 42)"
            echo "  --no-mixed-precision   Disable mixed precision training"
            echo "  --use-data-parallel    Enable data parallelism for multi-GPU training"
            echo "  --checkpoint PATH      Path to checkpoint to resume training"
            echo "  --help, -h             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $key"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create the output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Print training configuration
echo "=== Transformer PPO Training Configuration ==="
echo "Output directory: $OUTPUT_DIR"
echo "Total timesteps: $TIMESTEPS"
echo "Embedding dimension: $EMBED_DIM"
echo "Number of attention heads: $NUM_HEADS"
echo "Number of transformer layers: $NUM_LAYERS"
echo "Learning rate: $LEARNING_RATE"
echo "Random seed: $SEED"
echo "Mixed precision: $MIXED_PRECISION"
echo "Data parallel: $DATA_PARALLEL"
if [ -n "$CHECKPOINT" ]; then
    echo "Resuming from checkpoint: $CHECKPOINT"
fi
echo "=========================================="

# Build the command - use python3.10 explicitly
CMD="python3.10 -m src.thesis.train_transformer_ppo --batch-size 4096 \
    --total-timesteps $TIMESTEPS \
    --embed-dim $EMBED_DIM \
    --num-heads $NUM_HEADS \
    --num-layers $NUM_LAYERS \
    --learning-rate $LEARNING_RATE \
    --seed $SEED \
    --output-dir $OUTPUT_DIR"

# Add optional flags
if [ "$MIXED_PRECISION" = true ]; then
    CMD="$CMD --mixed-precision"
fi

if [ "$DATA_PARALLEL" = true ]; then
    CMD="$CMD --use-data-parallel"
fi

if [ -n "$CHECKPOINT" ]; then
    CMD="$CMD --checkpoint $CHECKPOINT"
fi

# Print the command
echo "Running: $CMD"
echo "Training log will be saved to: $OUTPUT_DIR/training.log"

# Execute the command
eval $CMD

# Print completion message
echo "Training completed. Results saved to $OUTPUT_DIR"
echo "To view training curves, run: tensorboard --logdir $OUTPUT_DIR/tensorboard"

# Recommend next steps
echo "=== Recommended Next Steps ==="
echo "1. Evaluate the model performance using the best model:"
echo "   python3.10 -m src.thesis.evaluate \\"
echo "       --model transformer_ppo \\"
echo "       --model-path $OUTPUT_DIR/best_model.pt \\"
echo "       --num-games 100"
echo ""
echo "2. Compare with other agents (if available):"
echo "   python3.10 -m src.thesis.compare_agents \\"
echo "       --agents transformer_ppo standard_ppo dqn \\"
echo "       --model-paths $OUTPUT_DIR/best_model.pt ppo_results/best_model.pt dqn_results/best_model.pt \\"
echo "       --num-games 50" 