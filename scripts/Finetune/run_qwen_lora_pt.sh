#!/bin/bash
set -e

# Paths
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-$ROOT/data/wikitext2/wikitext-2-raw}"
MODEL_DIR="${MODEL_DIR:-$ROOT/qwen2.5-0.5b}"
OUT_BASE="${OUT_BASE:-$ROOT/runs}"
OUT_DIR="${OUT_QWEN:-$OUT_BASE/qwen_lora_pt_s128_b4_acc1_e1_lr2e-4}"

mkdir -p "$OUT_DIR"

echo "Using Python: $PYTHON"
echo "Model dir:   $MODEL_DIR"
echo "Data dir:    $DATA_DIR"
echo "Output dir:  $OUT_DIR"

$PYTHON "$ROOT/pytorch_alignment/qwen_lora_finetune.py" \
  --model_dir "$MODEL_DIR" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_DIR" \
  --epochs 1 \
  --steps 0 \
  --seq_len 128 \
  --batch 4 \
  --grad_accum 1 \
  --learning_rate 2e-4 \
  --warmup_steps 0 \
  --lr_scheduler cosine \
  --max_grad_norm 1.0 \
  --data_fraction 0.5 \
  --weight_decay 0.0 \
  --target_mode qv \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --logging_steps 1 \
  --eval_steps 200 \
  --eval_batches 50 \
  --seed 42 \
  > "$OUT_DIR/train.log" 2>&1

echo "[DONE] Qwen PyTorch LoRA training finished."
echo "  Log: $OUT_DIR/train.log"

