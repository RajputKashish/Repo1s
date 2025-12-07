#!/bin/bash
# PPCM-X: Extended Privacy-Preserving CNN Pipeline
# Run script for training and encrypted inference

set -e

echo "=========================================="
echo "PPCM-X: Privacy-Preserving CNN Extended"
echo "=========================================="

# Configuration
EPOCHS=20
BATCH_SIZE=64
POLY_DEGREE="adaptive"
DATASET="mnist"
OUTPUT_DIR="experiments"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --poly-degree)
            POLY_DEGREE="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --test-only)
            TEST_ONLY=true
            shift
            ;;
        --help)
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --epochs N        Number of training epochs (default: 20)"
            echo "  --batch-size N    Batch size (default: 64)"
            echo "  --poly-degree D   Polynomial degree: 2,3,4,adaptive (default: adaptive)"
            echo "  --dataset D       Dataset: mnist, cifar10 (default: mnist)"
            echo "  --test-only       Run tests only"
            echo "  --help            Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create output directories
mkdir -p checkpoints
mkdir -p $OUTPUT_DIR/metrics_plots
mkdir -p data

# Run tests if requested
if [ "$TEST_ONLY" = true ]; then
    echo "[1/1] Running tests..."
    python -m pytest tests/ -v --cov=src --cov-report=html
    echo "Tests completed. Coverage report in htmlcov/"
    exit 0
fi

echo "[1/5] Installing dependencies..."
pip install -q -r requirements.txt

echo "[2/5] Training plaintext baseline model..."
python src/train.py \
    --mode plain \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR

echo "[3/5] Training HE-compatible model with adaptive activations..."
python src/train.py \
    --mode he_compatible \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --poly_degree $POLY_DEGREE \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR

echo "[4/5] Running encrypted inference benchmark..."
python src/infer_encrypted.py \
    --model checkpoints/he_compatible_best.pt \
    --dataset $DATASET \
    --output_dir $OUTPUT_DIR

echo "[5/5] Generating visualizations..."
python -c "
import json
import matplotlib.pyplot as plt
import os

results_path = '$OUTPUT_DIR/results.json'
if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        results = json.load(f)
    print('Results loaded successfully')
    print(json.dumps(results, indent=2))
else:
    print('No results file found. Run training first.')
"

echo "=========================================="
echo "Pipeline completed successfully!"
echo "Results saved to: $OUTPUT_DIR/"
echo "=========================================="
