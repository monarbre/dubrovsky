#!/bin/bash
# TRAIN LAMBDA 15K - Baseline Dubrovsky, same architecture, more iterations
#
# Same 9.51M params, 15K iters instead of 10K.
# Control group for comparing with boosted version.
#
# Usage:
#   chmod +x train_lambda_15k.sh
#   ./train_lambda_15k.sh

set -e

CHECKPOINT_DIR="subtitles_15k"

echo "DUBROVSKY 15K BASELINE"
echo "=========================================="
echo "Architecture: dim=384, layers=6, heads=6 (9.51M params)"
echo "Iterations: 15000"
echo "Output: $CHECKPOINT_DIR"
echo ""

if command -v nvidia-smi &> /dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
    echo ""
fi

if [ ! -f "dubrovsky.txt" ]; then
    echo "Dataset not found: dubrovsky.txt"
    exit 1
fi

mkdir -p "$CHECKPOINT_DIR"

echo "Starting baseline training..."
echo "=========================================="

python train.py \
    --lambda_mode \
    --max_iters 15000 \
    --batch_size 128 \
    --learning_rate 3e-4 \
    --out_dir "$CHECKPOINT_DIR"

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Training complete!"

    if [ -f "$CHECKPOINT_DIR/dubrovsky_final.pt" ]; then
        echo "Exporting weights..."
        python export_weights.py \
            "$CHECKPOINT_DIR/dubrovsky_final.pt" \
            "$CHECKPOINT_DIR/dubrovsky_15k.bin" \
            --verify

        echo ""
        echo "Final files:"
        ls -lh "$CHECKPOINT_DIR/"
        echo ""
        echo "Done! Test with:"
        echo "   python generate.py --weights $CHECKPOINT_DIR/dubrovsky_15k.bin --prompt 'Q: What is life?'"
    fi
else
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
    exit 1
fi
