#!/bin/bash
set -e

# Use the compsci371 environment's python
PYTHON="/opt/anaconda3/envs/compsci371/bin/python"

echo "Using Python: $PYTHON"

# GPT-2 PyTorch Alignment
echo "Starting PyTorch GPT-2 LoRA Finetune..."
OUT_GPT2="/Users/yiyilu/Desktop/FT_gemma_gpt2/runs/gpt2_lora_pt_s128_b8_acc1_e1_lr2e-4"
mkdir -p "$OUT_GPT2"

$PYTHON /Users/yiyilu/Desktop/FT_gemma_gpt2/pytorch_alignment/gpt2_lora_finetune.py \
  --data_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/data/wikitext2/wikitext-2-raw" \
  --pretrained_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/gpt2_lora_finetune/pretrained/gpt2" \
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
echo "PyTorch GPT-2 started with PID $PID_GPT2"

# Gemma PyTorch Alignment
echo "Starting PyTorch Gemma LoRA Finetune..."
OUT_GEMMA="/Users/yiyilu/Desktop/FT_gemma_gpt2/runs/gemma_lora_pt_s128_b8_acc1_50p"
mkdir -p "$OUT_GEMMA"

$PYTHON /Users/yiyilu/Desktop/FT_gemma_gpt2/pytorch_alignment/gemma_lora_finetune.py \
  --model_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/gemma-3-270m" \
  --data_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/data/wikitext2/wikitext-2-raw" \
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
echo "PyTorch Gemma started with PID $PID_GEMMA"

echo "Both PyTorch tasks started in parallel."
wait $PID_GPT2 $PID_GEMMA
echo "All PyTorch tasks finished."

