#!/bin/bash

# Shell script to run ADMM timing experiments on CPU and GPU, then visualize results

set -e  # Exit on error

DATASET="e2006"
NUM_ITERS=20
SEED=42
OUTPUT_DIR="./experiment_results"

echo "======================================================================"
echo "Running ADMM Timing Comparison: $DATASET"
echo "======================================================================"
echo ""

# Run on CPU
echo "Running ADMM on CPU for $NUM_ITERS iterations..."
python experiments/admm_iteration_timing.py \
    --dataset $DATASET \
    --device cpu \
    --num-iters $NUM_ITERS \
    --seed $SEED \
    --output-dir $OUTPUT_DIR

echo ""
echo "----------------------------------------------------------------------"
echo ""

# Run on GPU
echo "Running ADMM on GPU for $NUM_ITERS iterations..."
python experiments/admm_iteration_timing.py \
    --dataset $DATASET \
    --device cuda \
    --num-iters $NUM_ITERS \
    --seed $SEED \
    --output-dir $OUTPUT_DIR

echo ""
echo "----------------------------------------------------------------------"
echo ""

# Generate comparison plot
echo "Generating comparison plot..."
python experiments/plot_admm_comparison.py \
    --dataset $DATASET \
    --cpu-device cpu \
    --gpu-device cuda \
    --results-dir $OUTPUT_DIR \
    --seed $SEED \
    --output $OUTPUT_DIR/admm_comparison_${DATASET}_seed${SEED}.pdf

echo ""
echo "======================================================================"
echo "Comparison complete! Check results in $OUTPUT_DIR"
echo "======================================================================"
