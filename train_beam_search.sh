#!/bin/bash

# Set output directory
OUTPUT_DIR="beam_search_results"

# Set evaluation parameters
NUM_GAMES=100
MAX_STEPS=2000
BEAM_WIDTH=20
SEARCH_DEPTH=30
SEED=42

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the evaluation
python -m src.thesis.train_beam_search \
    --output-dir $OUTPUT_DIR \
    --num-games $NUM_GAMES \
    --max-steps $MAX_STEPS \
    --beam-width $BEAM_WIDTH \
    --search-depth $SEARCH_DEPTH \
    --seed $SEED

# Print results
echo "Evaluation completed. Results saved in $OUTPUT_DIR" 