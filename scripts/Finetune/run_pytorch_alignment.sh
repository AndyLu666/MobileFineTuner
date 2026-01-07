#!/bin/bash
set -e

# Root and executables
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-$ROOT/data/wikitext2/wikitext-2-raw}"

# Model locations (override with MODEL_* env vars if different)
MODEL_GPT2="${MODEL_GPT2:-$ROOT/gpt2_lora_finetune/pretrained/gpt2}"
MODEL_GEMMA="${MODEL_GEMMA:-$ROOT/gemma-3-270m}"
MODEL_QWEN="${MODEL_QWEN:-$ROOT/qwen2.5-0.5b}"

# Output base (override with OUT_BASE)
OUT_BASE="${OUT_BASE:-$ROOT/runs}"
mkdir -p "$OUT_BASE"

echo "Using Python: $PYTHON"
echo "Data dir:    $DATA_DIR"
echo "Outputs:     $OUT_BASE"

# ============ PyTorch GPT-2 ============
OUT_GPT2="$OUT_BASE/gpt2_lora_pt_s128_b8_acc1_e1_lr2e-4"
mkdir -p "$OUT_GPT2"
echo "[PyTorch] Starting GPT-2 LoRA..."
$PYTHON "$ROOT/pytorch_alignment/gpt2_lora_finetune.py" \
  --data_dir "$DATA_DIR" \
  --pretrained_dir "$MODEL_GPT2" \
  --lora_out "$OUT_GPT2/adapter" \
  --epochs 1 \
  --batch_size 8 \
  --grad_accum_steps 1 \
  --seq_len 128 \
  --rank 8 \
  --alpha 16 \
  --lr 2e-4 \
  --warmup_steps 500 \
  --clip_grad_norm 1.0 \
  --log_interval 1 \
  --eval_interval 500 \
  --eval_batches 200 \
  --eval_batch_size 2 \
  --save_every 1000 \
  --seed 42 \
  --data_fraction 0.5 \
  > "$OUT_GPT2/train.log" 2>&1 &
PID_GPT2=$!

# ============ PyTorch Gemma ============
OUT_GEMMA="$OUT_BASE/gemma_lora_pt_s128_b8_acc1_50p"
mkdir -p "$OUT_GEMMA"
echo "[PyTorch] Starting Gemma LoRA..."
$PYTHON "$ROOT/pytorch_alignment/gemma_lora_finetune.py" \
  --model_dir "$MODEL_GEMMA" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_GEMMA" \
  --target_mode attn \
  --seq_len 128 \
  --batch 8 \
  --grad_accum 1 \
  --epochs 1 \
  --data_fraction 0.5 \
  --learning_rate 2e-4 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  --lora_r 8 \
  --lora_alpha 16.0 \
  --lora_dropout 0.0 \
  > "$OUT_GEMMA/train.log" 2>&1 &
PID_GEMMA=$!

# ============ PyTorch Qwen ============
OUT_QWEN="$OUT_BASE/qwen_lora_pt_s128_b4_acc1_e1_lr2e-4"
mkdir -p "$OUT_QWEN"
echo "[PyTorch] Starting Qwen LoRA..."
$PYTHON "$ROOT/pytorch_alignment/qwen_lora_finetune.py" \
  --model_dir "$MODEL_QWEN" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_QWEN" \
  --epochs 1 \
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
  > "$OUT_QWEN/train.log" 2>&1 &
PID_QWEN=$!

echo ""
echo "========== Tasks started =========="
echo "PyTorch GPT-2: PID $PID_GPT2 -> $OUT_GPT2/train.log"
echo "PyTorch Gemma: PID $PID_GEMMA -> $OUT_GEMMA/train.log"
echo "PyTorch Qwen:  PID $PID_QWEN -> $OUT_QWEN/train.log"
echo "==================================="

wait $PID_GPT2 $PID_GEMMA $PID_QWEN
echo "All PyTorch tasks finished."

