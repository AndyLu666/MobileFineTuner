#!/bin/bash
set -e

# 设置输出目录
OUT="runs/gpt2_medium_lora_s128_b8_acc1_e1_lr2e-4"
mkdir -p "$OUT"

echo "=== Starting GPT-2 Medium LoRA Finetuning ==="
echo "Model: gpt2-medium"
echo "Output: $OUT"

# 切换到构建目录 (假设 binary 在 operators/build_rt 或类似位置，或者在根目录)
# 根据之前的命令，binary 似乎在 operators/build_rt/gpt2_lora_finetune 或 ./gpt2_lora_finetune
# 之前的命令是 cd "/Users/tony/Documents/FT（gemma完成版）/operators/build_rt" ...
# 你的环境是在根目录，我直接用绝对路径或相对路径调用 binary

# 使用 build_fast 目录下的高性能二进制文件 (BLAS enabled + fixes)
BINARY="/Users/yiyilu/Desktop/FT_gemma_gpt2/operators/build_fast/gpt2_lora_finetune"
if [ ! -f "$BINARY" ]; then
    echo "Error: Could not find binary at $BINARY"
    # Fallback check
    if [ -f "./operators/build_fast/gpt2_lora_finetune" ]; then
         BINARY="./operators/build_fast/gpt2_lora_finetune"
    else
         echo "Please ensure you have built the target in operators/build_fast"
         exit 1
    fi
fi

echo "Using binary: $BINARY"

nohup $BINARY \
  --data_dir "data/wikitext2/wikitext-2-raw" \
  --pretrained_dir "gpt2_lora_finetune/pretrained/gpt2-medium" \
  --lora_out "$OUT/gpt2_medium_lora.safetensors" \
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
  > "$OUT/train.log" 2>&1 &

PID=$!
echo "Training started in background with PID: $PID"
echo "Log file: $OUT/train.log"

