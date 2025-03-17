#!/bin/bash
# Shell script to train the Enhanced Dueling DQN with Prioritized Experience Replay (Fresh Start)

# Activate virtual environment if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Default values
EPISODES=15000
BATCH_SIZE=8192
LEARNING_RATE=0.00001
EPSILON_START=1.0
EPSILON_END=0.05
EPSILON_DECAY=0.9995
GAMMA=0.99
BUFFER_SIZE=500000
HIDDEN_DIM=1024
TARGET_UPDATE=500
UPDATE_FREQ=4
SAVE_FREQ=500
CHECKPOINT_DIR="models/enhanced_dqn_v2"
MODEL_NAME="enhanced_dqn_per"

# Parse command line arguments
if [ ! -z "$1" ]; then
    EPISODES=$1
fi
if [ ! -z "$2" ]; then
    BATCH_SIZE=$2
fi
if [ ! -z "$3" ]; then
    LEARNING_RATE=$3
fi
if [ ! -z "$4" ]; then
    RESUME_FROM=$4
fi

# Create checkpoint directory if it doesn't exist
mkdir -p "$CHECKPOINT_DIR"

# Display training settings
echo "Starting Fresh Enhanced Dueling DQN training with PER and Mixed Precision..."
echo "Episodes: $EPISODES"
echo "Batch Size: $BATCH_SIZE"
echo "Learning Rate: $LEARNING_RATE"
echo "Epsilon: $EPSILON_START to $EPSILON_END (decay: $EPSILON_DECAY)"
echo "Buffer Size: $BUFFER_SIZE"
echo "Hidden Dim: $HIDDEN_DIM"
echo "Target Update: Every $TARGET_UPDATE steps"
echo "Saving to: $CHECKPOINT_DIR/$MODEL_NAME"

# Run the training script
python -m src.thesis.training.train_dqn \
    --episodes $EPISODES \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --epsilon_start $EPSILON_START \
    --epsilon_end $EPSILON_END \
    --epsilon_decay $EPSILON_DECAY \
    --gamma $GAMMA \
    --buffer_size $BUFFER_SIZE \
    --hidden_dim $HIDDEN_DIM \
    --target_update_freq $TARGET_UPDATE \
    --update_freq $UPDATE_FREQ \
    --save_freq $SAVE_FREQ \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --model_name "$MODEL_NAME" \
    --use_per \
    --dueling

# Deactivate virtual environment if it was activated
if [ -f "venv/bin/deactivate" ] || [ -f ".venv/bin/deactivate" ]; then
    deactivate
fi

echo "Training complete!" 