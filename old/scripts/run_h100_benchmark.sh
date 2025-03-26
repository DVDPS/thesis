#!/bin/bash
# Script to run a complete benchmark suite on H100
# This script performs training, evaluation, and benchmarking with different configurations

# Create main output directory
MAIN_DIR="h100_benchmark_$(date +%Y%m%d_%H%M%S)"
mkdir -p $MAIN_DIR

# Log file
LOG_FILE="$MAIN_DIR/benchmark.log"
echo "Starting H100 benchmark suite at $(date)" | tee $LOG_FILE

# Print system information
echo "===== System Information =====" | tee -a $LOG_FILE
echo "Hostname: $(hostname)" | tee -a $LOG_FILE
echo "CPU: $(lscpu | grep 'Model name' | awk -F: '{print $2}' | sed 's/^[ \t]*//')" | tee -a $LOG_FILE
echo "Memory: $(free -h | grep Mem | awk '{print $2}')" | tee -a $LOG_FILE
nvidia-smi | tee -a $LOG_FILE
echo "============================" | tee -a $LOG_FILE

# Configuration
TRAINING_TIMESTEPS=1000000  # Reduced for benchmarking
NUM_EVAL_GAMES=10
PPO_CHECKPOINT="$MAIN_DIR/ppo_model/best_model.pt"

# STEP 1: Train PPO model
echo "STEP 1: Training PPO model..." | tee -a $LOG_FILE
mkdir -p "$MAIN_DIR/ppo_model"

# Run training with different batch sizes to benchmark
for BATCH_SIZE in 256 512 1024; do
    echo "Training with batch size $BATCH_SIZE..." | tee -a $LOG_FILE
    
    TRAIN_START_TIME=$(date +%s)
    python -m src.thesis.train_ppo \
        --mixed-precision \
        --total-timesteps $TRAINING_TIMESTEPS \
        --batch-size $BATCH_SIZE \
        --hidden-dim 512 \
        --output-dir "$MAIN_DIR/ppo_model_batch_$BATCH_SIZE" \
        --eval-episodes 5 2>&1 | tee -a "$MAIN_DIR/training_batch_$BATCH_SIZE.log"
    TRAIN_END_TIME=$(date +%s)
    
    TRAIN_DURATION=$((TRAIN_END_TIME - TRAIN_START_TIME))
    echo "Training with batch size $BATCH_SIZE completed in $TRAIN_DURATION seconds" | tee -a $LOG_FILE
    
    # Copy the best model to use for later steps
    if [ $BATCH_SIZE -eq 512 ]; then
        cp "$MAIN_DIR/ppo_model_batch_$BATCH_SIZE/best_model.pt" "$PPO_CHECKPOINT"
    fi
done

# STEP 2: Evaluate models
echo "STEP 2: Evaluating trained model..." | tee -a $LOG_FILE

# Run evaluation with different MCTS configurations
for PARALLEL in "true" "false"; do
    if [ "$PARALLEL" = "true" ]; then
        PARALLEL_FLAG="--use-parallel"
        PARALLEL_NAME="parallel"
    else
        PARALLEL_FLAG=""
        PARALLEL_NAME="sequential"
    fi
    
    for SIMS in 100 400 800; do
        echo "Evaluating with $SIMS simulations ($PARALLEL_NAME MCTS)..." | tee -a $LOG_FILE
        
        EVAL_START_TIME=$(date +%s)
        python -m src.thesis.utils.evaluation.enhanced_mcts_evaluation \
            --checkpoint "$PPO_CHECKPOINT" \
            --games $NUM_EVAL_GAMES \
            --simulations $SIMS \
            --agent-type ppo \
            $PARALLEL_FLAG \
            --output-dir "$MAIN_DIR/eval_${PARALLEL_NAME}_${SIMS}" 2>&1 | tee -a "$MAIN_DIR/eval_${PARALLEL_NAME}_${SIMS}.log"
        EVAL_END_TIME=$(date +%s)
        
        EVAL_DURATION=$((EVAL_END_TIME - EVAL_START_TIME))
        echo "Evaluation with $SIMS simulations ($PARALLEL_NAME MCTS) completed in $EVAL_DURATION seconds" | tee -a $LOG_FILE
    done
done

# STEP 3: Run parallel MCTS scalability test
echo "STEP 3: Testing scalability of parallel MCTS..." | tee -a $LOG_FILE

for WORKERS in 2 4 8 16; do
    for BATCH in 8 16 32; do
        echo "Testing parallel MCTS with $WORKERS workers and batch size $BATCH..." | tee -a $LOG_FILE
        
        TEST_START_TIME=$(date +%s)
        python -m src.thesis.utils.evaluation.enhanced_mcts_evaluation \
            --checkpoint "$PPO_CHECKPOINT" \
            --games 5 \
            --simulations 400 \
            --agent-type ppo \
            --use-parallel \
            --num-workers $WORKERS \
            --batch-size $BATCH \
            --output-dir "$MAIN_DIR/parallel_test_w${WORKERS}_b${BATCH}" 2>&1 | tee -a "$MAIN_DIR/parallel_test_w${WORKERS}_b${BATCH}.log"
        TEST_END_TIME=$(date +%s)
        
        TEST_DURATION=$((TEST_END_TIME - TEST_START_TIME))
        echo "Parallel MCTS test with $WORKERS workers and batch size $BATCH completed in $TEST_DURATION seconds" | tee -a $LOG_FILE
    done
done

# STEP 4: Generate comprehensive report
echo "STEP 4: Generating benchmark report..." | tee -a $LOG_FILE

# Create report
REPORT_FILE="$MAIN_DIR/benchmark_report.txt"

echo "H100 BENCHMARK REPORT" > $REPORT_FILE
echo "Generated on $(date)" >> $REPORT_FILE
echo "===============================" >> $REPORT_FILE
echo "" >> $REPORT_FILE

echo "1. TRAINING PERFORMANCE" >> $REPORT_FILE
echo "----------------------" >> $REPORT_FILE
for BATCH_SIZE in 256 512 1024; do
    TRAIN_LOG="$MAIN_DIR/training_batch_$BATCH_SIZE.log"
    
    # Extract final performance metrics from training log
    FINAL_AVG_TILE=$(grep "Avg Max Tile" $TRAIN_LOG | tail -1 | awk '{print $NF}')
    FINAL_BEST_TILE=$(grep "Best Max Tile" $TRAIN_LOG | tail -1 | awk '{print $NF}')
    
    echo "Batch size $BATCH_SIZE:" >> $REPORT_FILE
    echo "  - Final average max tile: $FINAL_AVG_TILE" >> $REPORT_FILE
    echo "  - Final best max tile: $FINAL_BEST_TILE" >> $REPORT_FILE
    
    # Extract training duration
    DURATION=$(grep "completed in" $LOG_FILE | grep "batch size $BATCH_SIZE" | awk '{print $(NF-1)}')
    echo "  - Training duration: $DURATION seconds" >> $REPORT_FILE
    echo "" >> $REPORT_FILE
done

echo "2. MCTS PERFORMANCE" >> $REPORT_FILE
echo "------------------" >> $REPORT_FILE
for PARALLEL in "parallel" "sequential"; do
    echo "$PARALLEL MCTS:" >> $REPORT_FILE
    
    for SIMS in 100 400 800; do
        EVAL_LOG="$MAIN_DIR/eval_${PARALLEL}_${SIMS}.log"
        
        # Extract performance metrics
        AVG_TILE=$(grep "Average Max Tile" $EVAL_LOG | head -1 | grep -o "Enhanced MCTS = [0-9.]*" | awk '{print $NF}')
        BEST_TILE=$(grep "Best Max Tile" $EVAL_LOG | head -1 | grep -o "Enhanced MCTS = [0-9]*" | awk '{print $NF}')
        
        echo "  Simulations $SIMS:" >> $REPORT_FILE
        echo "    - Average max tile: $AVG_TILE" >> $REPORT_FILE
        echo "    - Best max tile: $BEST_TILE" >> $REPORT_FILE
        
        # Extract evaluation duration
        DURATION=$(grep "completed in" $LOG_FILE | grep "$SIMS simulations ($PARALLEL MCTS)" | awk '{print $(NF-1)}')
        echo "    - Evaluation duration: $DURATION seconds" >> $REPORT_FILE
        echo "" >> $REPORT_FILE
    done
done

echo "3. PARALLEL MCTS SCALABILITY" >> $REPORT_FILE
echo "--------------------------" >> $REPORT_FILE
for WORKERS in 2 4 8 16; do
    for BATCH in 8 16 32; do
        TEST_LOG="$MAIN_DIR/parallel_test_w${WORKERS}_b${BATCH}.log"
        
        # Extract performance metrics
        AVG_TILE=$(grep "Average Max Tile" $TEST_LOG | head -1 | grep -o "Enhanced MCTS = [0-9.]*" | awk '{print $NF}')
        
        echo "Workers $WORKERS, Batch size $BATCH:" >> $REPORT_FILE
        echo "  - Average max tile: $AVG_TILE" >> $REPORT_FILE
        
        # Extract duration
        DURATION=$(grep "completed in" $LOG_FILE | grep "Parallel MCTS test with $WORKERS workers and batch size $BATCH" | awk '{print $(NF-1)}')
        echo "  - Evaluation duration: $DURATION seconds" >> $REPORT_FILE
        echo "" >> $REPORT_FILE
    done
done

echo "Benchmark complete. Report saved to $REPORT_FILE" | tee -a $LOG_FILE