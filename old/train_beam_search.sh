#!/bin/bash
# Set output directory
OUTPUT_DIR="beam_search_results"
# Set evaluation parameters
NUM_GAMES=10000
MAX_STEPS=5000
BEAM_WIDTH=20
SEARCH_DEPTH=30
BATCH_SIZE=32
SEED=42
# Create output directory
mkdir -p $OUTPUT_DIR
# Set Python path to include user's local site-packages
export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/python3.10/site-packages
# Run the evaluation
python3 -m src.thesis.train_beam_search \
    --output-dir $OUTPUT_DIR \
    --num-games $NUM_GAMES \
    --max-steps $MAX_STEPS \
    --beam-width $BEAM_WIDTH \
    --search-depth $SEARCH_DEPTH \
    --batch-size $BATCH_SIZE \
    --seed $SEED
# Print results
echo "Evaluation completed. Results saved in $OUTPUT_DIR"