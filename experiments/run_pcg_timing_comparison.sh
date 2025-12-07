#!/bin/bash

# Shell script to run PCG timing experiments on CPU and GPU, then visualize results

set -e  # Exit on error

DATASET="acsincome"
NUM_ITERS=100
SEED=42
OUTPUT_DIR="./experiment_results"

echo "======================================================================"
echo "Running PCG Timing Comparison: $DATASET"
echo "======================================================================"
echo ""

# Run on CPU
echo "Running PCG on CPU for $NUM_ITERS iterations..."
python experiments/pcg_iteration_timing.py \
    --dataset $DATASET \
    --device cpu \
    --num-iters $NUM_ITERS \
    --seed $SEED \
    --output-dir $OUTPUT_DIR

echo ""
echo "----------------------------------------------------------------------"
echo ""

# Run on GPU
echo "Running PCG on GPU for $NUM_ITERS iterations..."
python experiments/pcg_iteration_timing.py \
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
python experiments/plot_pcg_comparison.py \
    --dataset $DATASET \
    --cpu-device cpu \
    --gpu-device cuda \
    --results-dir $OUTPUT_DIR \
    --seed $SEED \
    --output $OUTPUT_DIR/pcg_comparison_${DATASET}_seed${SEED}.pdf

echo ""
echo "======================================================================"
echo "Comparison complete! Check results in $OUTPUT_DIR"
echo "======================================================================"
