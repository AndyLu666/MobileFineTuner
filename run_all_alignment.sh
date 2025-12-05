#!/bin/bash
set -e

echo "========== Starting All Training Tasks =========="

# ============ C++ GPT-2 ============
echo "[C++] Starting GPT-2 Finetune..."
cd /Users/yiyilu/Desktop/FT_gemma_gpt2/operators/build_opt
OUT_GPT2_CPP="/Users/yiyilu/Desktop/FT_gemma_gpt2/runs/gpt2_lora_s128_b8_acc1_e1_lr2e-4"
mkdir -p "$OUT_GPT2_CPP"

./gpt2_lora_finetune \
  --data_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/data/wikitext2/wikitext-2-raw" \
  --pretrained_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/gpt2_lora_finetune/pretrained/gpt2" \
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
echo "[C++] GPT-2 started with PID $PID_GPT2_CPP"

# ============ C++ Gemma ============
echo "[C++] Starting Gemma Finetune..."
OUT_GEMMA_CPP="/Users/yiyilu/Desktop/FT_gemma_gpt2/runs/gemma_lora_s128_b8_acc1_50p"
mkdir -p "$OUT_GEMMA_CPP"

./train_lora_gemma \
  --model_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/gemma-3-270m" \
  --pretokenized_path "/Users/yiyilu/Desktop/FT_gemma_gpt2/data/wikitext2/pretokenized_gemma/wt2_gemma_tokens.bin" \
  --pretokenized_meta "/Users/yiyilu/Desktop/FT_gemma_gpt2/data/wikitext2/pretokenized_gemma/meta.json" \
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
echo "[C++] Gemma started with PID $PID_GEMMA_CPP"

# ============ PyTorch GPT-2 ============
echo "[PyTorch] Starting GPT-2 Finetune..."
cd /Users/yiyilu/Desktop/FT_gemma_gpt2
PYTHON="/opt/anaconda3/envs/compsci371/bin/python"
OUT_GPT2_PT="/Users/yiyilu/Desktop/FT_gemma_gpt2/runs/gpt2_lora_pt_s128_b8_acc1_e1_lr2e-4"
mkdir -p "$OUT_GPT2_PT"

$PYTHON pytorch_alignment/gpt2_lora_finetune.py \
  --data_dir "data/wikitext2/wikitext-2-raw" \
  --pretrained_dir "gpt2_lora_finetune/pretrained/gpt2" \
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
echo "[PyTorch] GPT-2 started with PID $PID_GPT2_PT"

# ============ PyTorch Gemma ============
echo "[PyTorch] Starting Gemma Finetune..."
OUT_GEMMA_PT="/Users/yiyilu/Desktop/FT_gemma_gpt2/runs/gemma_lora_pt_s128_b8_acc1_50p"
mkdir -p "$OUT_GEMMA_PT"

$PYTHON pytorch_alignment/gemma_lora_finetune.py \
  --model_dir "gemma-3-270m" \
  --data_dir "data/wikitext2/wikitext-2-raw" \
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
echo "[PyTorch] Gemma started with PID $PID_GEMMA_PT"

echo ""
echo "========== All 4 Tasks Started =========="
echo "C++ GPT-2:     PID $PID_GPT2_CPP  -> $OUT_GPT2_CPP/train.log"
echo "C++ Gemma:     PID $PID_GEMMA_CPP -> $OUT_GEMMA_CPP/train.log"
echo "PyTorch GPT-2: PID $PID_GPT2_PT   -> $OUT_GPT2_PT/train.log"
echo "PyTorch Gemma: PID $PID_GEMMA_PT  -> $OUT_GEMMA_PT/train.log"
echo ""
echo "Waiting for all tasks to complete..."
wait $PID_GPT2_CPP $PID_GEMMA_CPP $PID_GPT2_PT $PID_GEMMA_PT
echo "========== All Tasks Finished =========="

