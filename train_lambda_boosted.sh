#!/bin/bash
# TRAIN LAMBDA BOOSTED - Dubrovsky v2: more heads, deeper
#
# Architecture changes from v1:
#   n_heads:  6 -> 8  (more attention diversity, head_dim 64->48)
#   n_layers: 6 -> 7  (one more layer of depth)
#   dim, hidden_dim, kv_heads: unchanged
#
# Result: ~11.1M params (was 9.51M, +17%)
#
# Usage:
#   chmod +x train_lambda_boosted.sh
#   ./train_lambda_boosted.sh

set -e

CHECKPOINT_DIR="subtitles_boosted"

echo "DUBROVSKY BOOSTED v2"
echo "=========================================="
echo "Architecture: dim=384, layers=7, heads=8 (~11.1M params)"
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

echo "Starting boosted training..."
echo "=========================================="

python train.py \
    --lambda_mode \
    --max_iters 15000 \
    --batch_size 128 \
    --learning_rate 3e-4 \
    --n_layers 7 \
    --n_heads 8 \
    --out_dir "$CHECKPOINT_DIR"

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Training complete!"

    if [ -f "$CHECKPOINT_DIR/dubrovsky_final.pt" ]; then
        echo "Exporting weights..."
        python export_weights.py \
            "$CHECKPOINT_DIR/dubrovsky_final.pt" \
            "$CHECKPOINT_DIR/dubrovsky_boosted.bin" \
            --verify

        echo ""
        echo "Final files:"
        ls -lh "$CHECKPOINT_DIR/"
        echo ""
        echo "Done! Test with:"
        echo "   python generate.py --weights $CHECKPOINT_DIR/dubrovsky_boosted.bin --prompt 'Q: What is life?'"
    fi
else
    echo "Training failed with exit code $TRAIN_EXIT_CODE"
    exit 1
fi
