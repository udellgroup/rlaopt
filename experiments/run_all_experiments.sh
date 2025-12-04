#!/bin/bash

# Shell script to run all experiments sequentially
# This ensures memory is cleared between each experiment run

# Configuration
DATASETS=("acsincome" "e2006" "realsim" "yearpredictionmsd")
DEVICES=("cuda:1" "cpu")
NUM_ITERS=500
SEED=42
OUTPUT_DIR="./experiment_results"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Log file
LOG_FILE="$OUTPUT_DIR/experiment_log.txt"
echo "Starting experiments at $(date)" | tee "$LOG_FILE"

# Run experiments
for device in "${DEVICES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "" | tee -a "$LOG_FILE"
        echo "======================================" | tee -a "$LOG_FILE"
        echo "Running: $dataset on $device" | tee -a "$LOG_FILE"
        echo "======================================" | tee -a "$LOG_FILE"

        # Run the experiment
        if python run_experiments.py \
            --dataset "$dataset" \
            --device "$device" \
            --num-iters "$NUM_ITERS" \
            --seed "$SEED" \
            --output-dir "$OUTPUT_DIR" 2>&1 | tee -a "$LOG_FILE"; then
            echo "SUCCESS: $dataset on $device" | tee -a "$LOG_FILE"
        else
            echo "FAILED: $dataset on $device" | tee -a "$LOG_FILE"
        fi

        # Wait a bit to ensure cleanup
        sleep 2
    done
done

echo "" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"
echo "All experiments completed at $(date)" | tee -a "$LOG_FILE"
echo "======================================" | tee -a "$LOG_FILE"
