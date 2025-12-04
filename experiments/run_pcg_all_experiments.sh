#!/bin/bash

# Run PCG experiments across multiple datasets and devices
# Usage: bash run_pcg_all_experiments.sh

set -e  # Exit on error

# Configuration
DATASETS=("acsincome" "e2006" "realsim" "yearpredictionmsd")
DEVICES=("cuda:1" "cpu")
NUM_ITERS=5000
REG=1e-2
RANK=100
SEED=42
OUTPUT_DIR="./experiment_results"

# Create output directory
mkdir -p $OUTPUT_DIR

echo "Starting PCG experiments..."
echo "Datasets: ${DATASETS[@]}"
echo "Devices: ${DEVICES[@]}"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Run experiments
for device in "${DEVICES[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        echo "=========================================="
        echo "Running: $dataset on $device"
        echo "=========================================="

        python run_pcg_experiments.py \
            --dataset $dataset \
            --device $device \
            --num-iters $NUM_ITERS \
            --reg $REG \
            --rank $RANK \
            --seed $SEED \
            --output-dir $OUTPUT_DIR

        echo ""
    done
done

echo "All experiments completed!"
echo "Results saved in: $OUTPUT_DIR"
