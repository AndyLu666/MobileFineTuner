#!/bin/bash
set -e

# =============================================================================
# Gemma 3 1B LoRA Finetuning Script
# Configuration aligned with gemma-3-270m run, adjusted for 1B model size
# =============================================================================

echo "=== Starting Gemma 3 1B LoRA Finetuning ==="
echo "Model: gemma-3-1b"
echo "Start time: $(date)"

# Output directory
OUT="/Users/yiyilu/Desktop/FT_gemma_gpt2/runs/gemma_1b_lora_s128_b4_acc1_50p"
mkdir -p "$OUT"
echo "Output: $OUT"

# Binary with BLAS optimization
BINARY="/Users/yiyilu/Desktop/FT_gemma_gpt2/operators/build_fast/train_lora_gemma"
if [ ! -f "$BINARY" ]; then
    echo "Error: Could not find binary at $BINARY"
    exit 1
fi

echo "Using binary: $BINARY"

# Run training
# Configuration aligned with 270M run:
# - seq_len: 128
# - batch: 4 (reduced from 8 for 1B model)
# - grad_accum: 2 (increased to maintain effective batch size of 8)
# - epochs: 1
# - data_fraction: 0.5
# - lr: 2e-4
# - warmup_ratio: 0.05
# - max_grad_norm: 1.0
# - targets: attn (q, k, v, o)

nohup $BINARY \
  --model_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/gemma-3-1b" \
  --data_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/data/wikitext2/wikitext-2-raw" \
  --output_dir "$OUT" \
  --targets attn \
  --seq_len 128 \
  --batch 8 \
  --grad_accum 1 \
  --epochs 1 \
  --data_fraction 0.5 \
  --lr 2e-4 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  > "$OUT/train.log" 2>&1 &

PID=$!
echo "Training started in background with PID: $PID"
echo "Log file: $OUT/train.log"
echo ""
echo "Monitor progress with:"
echo "  tail -f $OUT/train.log"

