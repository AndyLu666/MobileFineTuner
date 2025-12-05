#!/bin/bash
set -e

# GPT-2
echo "Starting GPT-2 Finetune..."
cd /Users/yiyilu/Desktop/FT_gemma_gpt2/operators/build_opt
OUT_GPT2="/Users/yiyilu/Desktop/FT_gemma_gpt2/runs/gpt2_lora_s128_b8_acc1_e1_lr2e-4"
mkdir -p "$OUT_GPT2"

./gpt2_lora_finetune \
  --data_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/data/wikitext2/wikitext-2-raw" \
  --pretrained_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/gpt2_lora_finetune/pretrained/gpt2" \
  --lora_out "$OUT_GPT2/gpt2_lora.safetensors" \
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
echo "GPT-2 Finetune started with PID $PID_GPT2"

# Gemma
echo "Starting Gemma Finetune..."
OUT_GEMMA="/Users/yiyilu/Desktop/FT_gemma_gpt2/runs/gemma_lora_s128_b8_acc1_50p"
mkdir -p "$OUT_GEMMA"

./train_lora_gemma \
  --model_dir "/Users/yiyilu/Desktop/FT_gemma_gpt2/gemma-3-270m" \
  --pretokenized_path "/Users/yiyilu/Desktop/FT_gemma_gpt2/data/wikitext2/pretokenized_gemma/wt2_gemma_tokens.bin" \
  --pretokenized_meta "/Users/yiyilu/Desktop/FT_gemma_gpt2/data/wikitext2/pretokenized_gemma/meta.json" \
  --output_dir "$OUT_GEMMA" \
  --targets attn \
  --seq_len 128 \
  --batch 8 \
  --grad_accum 1 \
  --epochs 1 \
  --data_fraction 0.5 \
  --lr 2e-4 \
  --warmup_ratio 0.05 \
  --max_grad_norm 1.0 \
  > "$OUT_GEMMA/train.log" 2>&1 &
PID_GEMMA=$!
echo "Gemma Finetune started with PID $PID_GEMMA"

echo "Both tasks started in parallel."
wait $PID_GPT2 $PID_GEMMA
echo "All tasks finished."

