#!/bin/bash
set -e

echo "========== Starting All Training Tasks =========="

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
DATA_DIR="${DATA_DIR:-$ROOT/data/wikitext2/wikitext-2-raw}"
PRETOK_PATH="${PRETOK_PATH:-$ROOT/data/wikitext2/pretokenized_gemma/wt2_gemma_tokens.bin}"
PRETOK_META="${PRETOK_META:-$ROOT/data/wikitext2/pretokenized_gemma/meta.json}"
BUILD_DIR="${BUILD_DIR:-$ROOT/operators/build_opt}"
MODEL_GPT2="${MODEL_GPT2:-$ROOT/gpt2_lora_finetune/pretrained/gpt2}"
MODEL_GEMMA="${MODEL_GEMMA:-$ROOT/gemma-3-270m}"
MODEL_QWEN="${MODEL_QWEN:-$ROOT/qwen2.5-0.5b}"
OUT_BASE="${OUT_BASE:-$ROOT/runs}"

mkdir -p "$OUT_BASE"

echo "Root:      $ROOT"
echo "Data dir:  $DATA_DIR"
echo "Build dir: $BUILD_DIR"
echo "Outputs:   $OUT_BASE"

# ============ C++ GPT-2 ============
echo "[C++] Starting GPT-2 LoRA..."
cd "$BUILD_DIR"
OUT_GPT2_CPP="$OUT_BASE/gpt2_lora_s128_b8_acc1_e1_lr2e-4"
mkdir -p "$OUT_GPT2_CPP"
./gpt2_lora_finetune \
  --data_dir "$DATA_DIR" \
  --pretrained_dir "$MODEL_GPT2" \
  --lora_out "$OUT_GPT2_CPP/gpt2_lora.safetensors" \
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
  > "$OUT_GPT2_CPP/train.log" 2>&1 &
PID_GPT2_CPP=$!

# ============ C++ Gemma ============
echo "[C++] Starting Gemma LoRA..."
OUT_GEMMA_CPP="$OUT_BASE/gemma_lora_s128_b8_acc1_50p"
mkdir -p "$OUT_GEMMA_CPP"
./train_lora_gemma \
  --model_dir "$MODEL_GEMMA" \
  --pretokenized_path "$PRETOK_PATH" \
  --pretokenized_meta "$PRETOK_META" \
  --output_dir "$OUT_GEMMA_CPP" \
  --targets attn \
  --seq_len 128 \
  --batch 8 \
  --grad_accum 1 \
  --epochs 1 \
  --data_fraction 0.5 \
  --lr 2e-4 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  > "$OUT_GEMMA_CPP/train.log" 2>&1 &
PID_GEMMA_CPP=$!

cd "$ROOT"

# ============ PyTorch GPT-2 ============
echo "[PyTorch] Starting GPT-2 LoRA..."
OUT_GPT2_PT="$OUT_BASE/gpt2_lora_pt_s128_b8_acc1_e1_lr2e-4"
mkdir -p "$OUT_GPT2_PT"
$PYTHON "$ROOT/pytorch_alignment/gpt2_lora_finetune.py" \
  --data_dir "$DATA_DIR" \
  --pretrained_dir "$MODEL_GPT2" \
  --lora_out "$OUT_GPT2_PT/adapter" \
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
  > "$OUT_GPT2_PT/train.log" 2>&1 &
PID_GPT2_PT=$!

# ============ PyTorch Gemma ============
echo "[PyTorch] Starting Gemma LoRA..."
OUT_GEMMA_PT="$OUT_BASE/gemma_lora_pt_s128_b8_acc1_50p"
mkdir -p "$OUT_GEMMA_PT"
$PYTHON "$ROOT/pytorch_alignment/gemma_lora_finetune.py" \
  --model_dir "$MODEL_GEMMA" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_GEMMA_PT" \
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
  --lora_alpha 32.0 \
  --lora_dropout 0.1 \
  > "$OUT_GEMMA_PT/train.log" 2>&1 &
PID_GEMMA_PT=$!

# ============ PyTorch Qwen ============
echo "[PyTorch] Starting Qwen LoRA..."
OUT_QWEN_PT="$OUT_BASE/qwen_lora_pt_s128_b4_acc1_e1_lr2e-4"
mkdir -p "$OUT_QWEN_PT"
$PYTHON "$ROOT/pytorch_alignment/qwen_lora_finetune.py" \
  --model_dir "$MODEL_QWEN" \
  --data_dir "$DATA_DIR" \
  --output_dir "$OUT_QWEN_PT" \
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
  > "$OUT_QWEN_PT/train.log" 2>&1 &
PID_QWEN_PT=$!

echo ""
echo "========== All Tasks Started =========="
echo "C++ GPT-2:     PID $PID_GPT2_CPP  -> $OUT_GPT2_CPP/train.log"
echo "C++ Gemma:     PID $PID_GEMMA_CPP -> $OUT_GEMMA_CPP/train.log"
echo "PyTorch GPT-2: PID $PID_GPT2_PT   -> $OUT_GPT2_PT/train.log"
echo "PyTorch Gemma: PID $PID_GEMMA_PT  -> $OUT_GEMMA_PT/train.log"
echo "PyTorch Qwen:  PID $PID_QWEN_PT   -> $OUT_QWEN_PT/train.log"
echo ""
echo "Waiting for all tasks to complete..."
wait $PID_GPT2_CPP $PID_GEMMA_CPP $PID_GPT2_PT $PID_GEMMA_PT $PID_QWEN_PT
echo "========== All Tasks Finished =========="

